import operator
import numpy as np
from numba import types, njit, i8, u8, i4, u1, i8, literally, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from numbert.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from numbert.experimental.context import kb_context
from numbert.experimental.structref import define_structref, define_structref_template
from numbert.experimental.kb import KnowledgeBaseType, KnowledgeBase, facts_for_t_id, fact_at_f_id
from numbert.experimental.fact import define_fact, BaseFactType, cast_fact
from numbert.experimental.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, decode_idrec, lower_getattr, _struct_from_pointer,  lower_setattr, lower_getattr, _pointer_from_struct
from numbert.experimental.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from numbert.experimental.vector import VectorType
from numbert.experimental.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
 get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models
from numbert.experimental.var import *

from operator import itemgetter
from copy import copy

pterm_fields_dict = {
    "str_val" : unicode_type,
    "pred_node" : BasePredicateNodeType,
    "negated" : u1,
    "is_alpha" : u1,
}

pterm_fields =  [(k,v) for k,v, in pterm_fields_dict.items()]

@structref.register
class PTermTypeTemplate(types.StructRef):
    pass

class PTerm(structref.StructRefProxy):
    def __new__(cls, *args):
        return pterm_ctor(*args)
        # return structref.StructRefProxy.__new__(cls, *args)
    def __str__(self):
        return pterm_get_str_val(self)

    @property
    def pred_node(self):
        return pterm_get_pred_node(self)



@njit(cach=True)
def pterm_get_str_val(self):
    return self.str_val

@njit(cach=True)
def pterm_get_pred_node(self):
    return self.pred_node

define_boxing(PTermTypeTemplate,PTerm)
PTermType = PTermTypeTemplate(pterm_fields)


@njit(cache=True)
def alpha_pterm_ctor(pn, left_var, op_str, right_var):
    st = new(PTermType)
    st.pred_node = pn
    st.str_val = str(left_var) + " " + op_str + " " + "?" #base_str + "?"#TODO str->float needs to work
    st.negated = False
    st.is_alpha = True
    return st

@njit(cache=True)
def beta_pterm_ctor(pn, left_var, op_str, right_var):
    st = new(PTermType)
    st.pred_node = pn
    st.str_val = str(left_var) + " " + op_str + " " + str(right_var)
    st.negated = False
    st.is_alpha = False
    return st


pnode_dat_cache = {}
def get_alpha_pnode_ctor(left_var, op_str, right_var):
    t = ("alpha",str(left_var), op_str, right_var)
    if(t not in pnode_dat_cache):
        left_fact_type = left_var.field_dict['fact_type'].instance_type
        left_type = left_var.field_dict['head_type'].instance_type
        left_fact_type_name = left_fact_type._fact_name

        ctor, _ = define_alpha_predicate_node(left_type, op_str, right_var)
        pnode_dat_cache[t] = (ctor, left_fact_type_name)
    return pnode_dat_cache[t]


def get_beta_pnode_ctor(left_var, op_str, right_var):
    t = ("beta",str(left_var), op_str,str(right_var))
    if(t not in pnode_dat_cache):
        left_fact_type = left_var.field_dict['fact_type'].instance_type
        left_type = left_var.field_dict['head_type'].instance_type
        left_fact_type_name = left_fact_type._fact_name

        right_fact_type = right_var.field_dict['fact_type'].instance_type
        right_type = right_var.field_dict['head_type'].instance_type
        right_fact_type_name = right_fact_type._fact_name

        ctor, _ = define_beta_predicate_node(left_type, op_str, right_type)
        pnode_dat_cache[t] = (ctor, left_fact_type_name, right_fact_type_name)
    return pnode_dat_cache[t]

@njit(cache=True)
def cpy_derefs(var):
    offsets = np.empty((len(var.deref_offsets),),dtype=np.int64)
    for i,x in enumerate(var.deref_offsets): offsets[i] = x
    return offsets

@njit(cache=True)
def cast_pn_to_base(pn):
    return _cast_structref(BasePredicateNodeType,pn) 

def gen_pterm_ctor_alpha(left_var, op_str, right_var):
    ctor, left_fact_type_name = \
            get_alpha_pnode_ctor(left_var, op_str, right_var)
    def impl(left_var, op_str, right_var):
        l_offsets = cpy_derefs(left_var)
        apn = ctor(str(left_fact_type_name), l_offsets, right_var)
        pn = cast_pn_to_base(apn) 
        return alpha_pterm_ctor(pn, left_var, op_str, right_var)
    return impl

def gen_pterm_ctor_beta(left_var, op_str, right_var):
    ctor, left_fact_type_name, right_fact_type_name = \
            get_beta_pnode_ctor(left_var, op_str, right_var)
    
    def impl(left_var, op_str, right_var):
        l_offsets = cpy_derefs(left_var)
        r_offsets = cpy_derefs(right_var)
        apn = ctor(str(left_fact_type_name), l_offsets, str(left_fact_type_name), r_offsets)
        pn = cast_pn_to_base(apn) 
        return beta_pterm_ctor(pn, left_var, op_str, right_var)            
    return impl

@generated_jit(cache=True)
@overload(PTerm)
def pterm_ctor(left_var, op_str, right_var):
    if(not isinstance(op_str, types.Literal)): return 
    if(not isinstance(left_var, VarTypeTemplate)): return

    op_str = op_str.literal_value
    if(not isinstance(right_var, VarTypeTemplate)):
        return gen_pterm_ctor_alpha(left_var, op_str, right_var)
    else:
        return gen_pterm_ctor_beta(left_var, op_str, right_var)


@overload(str)
def str_pterm(self):
    if(not isinstance(self, PTermTypeTemplate)): return
    def impl(self):
        return self.str_val
    return impl

@njit(cache=True)
def pterm_copy(self):
    st = new(PTermType)
    st.str_val = self.str_val
    st.pred_node = self.pred_node
    st.negated = self.negated
    st.is_alpha = self.is_alpha
    return st
    
@njit(cache=True)
def pterm_not(self):
    npt = pterm_copy(self)
    npt.negated = not npt.negated
    return npt


@njit(cache=True)
def comparator_jitted(left_var, op_str, right_var, negated):
    pt = PTerm(left_var, op_str, right_var)
    dnf = new_dnf(1)
    # ind = 0 if (pt.is_alpha) else 1
    dnf[0].append(pt)
    _vars = List([left_var.alias])
    # if(not is_alpha): _vars.append(right_var.alias)
    pt.negated = negated
    # print(type(right_var))
    c = Conditions(_vars, dnf)
    return c


@njit(cache=True)
def pt_to_cond(pt, left_alias, right_alias, negated):
    dnf = new_dnf(1)
    # ind = 0 if (pt.is_alpha) else 1
    dnf[0].append(pt)
    _vars = List.empty_list(unicode_type)
    _vars.append(left_alias)
    if(right_alias is not None):
        _vars.append(right_alias)
    pt.negated = negated
    c = Conditions(_vars, dnf)
    return c


def comparator_helper(op_str, left_var, right_var,negated=False):
    if(isinstance(left_var,VarTypeTemplate)):
        check_legal_cmp(left_var, op_str, right_var)
        op_str = types.unliteral(op_str)
        if(not isinstance(right_var,VarTypeTemplate)):
            # print("POOP")
            right_var_type = types.unliteral(right_var)
            # ctor = gen_pterm_ctor_alpha(left_var, op_str, right_var_type)
            # print("POOP")

            def impl(left_var, right_var):
                pt = PTerm(left_var, op_str, right_var)
                return pt_to_cond(pt, left_var.alias, None, negated)
        else:
            # ctor = gen_pterm_ctor_beta(left_var, op_str, right_var)
            def impl(left_var, right_var):
                pt = PTerm(left_var, op_str, right_var)
                return pt_to_cond(pt, left_var.alias, right_var.alias, negated)

            # return var_cmp_beta


        return impl


@generated_jit(cache=True)
@overload(operator.lt)
def var_lt(left_var, right_var):
    return comparator_helper("<", left_var, right_var)

@generated_jit(cache=True)
@overload(operator.le)
def var_le(left_var, right_var):
    return comparator_helper("<=", left_var, right_var)

@generated_jit(cache=True)
@overload(operator.gt)
def var_gt(left_var, right_var):
    return comparator_helper(">", left_var, right_var)

@generated_jit(cache=True)
@overload(operator.ge)
def var_ge(left_var, right_var):
    return comparator_helper(">=", left_var, right_var)

@generated_jit(cache=True)
@overload(operator.eq)
def var_eq(left_var, right_var):
    return comparator_helper("==", left_var, right_var)

@generated_jit(cache=True)
@overload(operator.ne)
def var_ne(left_var, right_var):
    return comparator_helper("==", left_var, right_var, True)



pterm_list_type = ListType(PTermType)
# pterm_list_x2_type = Tuple((pterm_list_type, pterm_list_type))
pterm_list_list_type = ListType(pterm_list_type)
dnf_type = pterm_list_list_type

conditions_fields_dict = {
    ### Fields that are filled on in instantiation ### 

    # The variables used by the condition
    'vars': ListType(BaseVarType),

    # The Disjunctive Normal Form of the condition but organized
    #  so that every conjunct has a seperate lists for alpha and 
    #  beta terms.
    'dnf': pterm_list_list_type,

    # A mapping from Var pointers to their associated index
    'var_map': DictType(i8,i8),

    # Wether or not the conditions object has been initialized
    'initialized' : u1,

    ### Fields that are filled in after initialization ### 

    # The alpha parts of '.dnf' organized by which Var in 'vars' they use 
    'alpha_dnfs': ListType(dnf_type),

    # The beta parts of '.dnf' organized by which left Var in 'vars' they use 
    'beta_dnfs': ListType(dnf_type),
}

conditions_fields =  [(k,v) for k,v, in conditions_fields_dict.items()]



@structref.register
class ConditionsTypeTemplate(types.StructRef):
    pass


# Manually register the type to avoid automatic getattr overloading 
# default_manager.register(VarTypeTemplate, models.StructRefModel)

class Conditions(structref.StructRefProxy):
    def __new__(cls, _vars, dnf=None):
        # return structref.StructRefProxy.__new__(cls, *args)
        return conditions_ctor(_vars, dnf)
    def __str__(self):
        return conditions_str(self)
    def __and__(self, other):
        return conditions_and(self, other)
    def __or__(self, other):
        return conditions_or(self, other)
    def __not__(self):
        return conditions_not(self)
    def __invert__(self):
        return conditions_not(self)

define_boxing(ConditionsTypeTemplate,Conditions)

@njit(cache=True)
def new_dnf(n):
    dnf = List.empty_list(pterm_list_type)
    for i in range(n):
        dnf.append( List.empty_list(PTermType) )
    return dnf

ConditionsType = ConditionsTypeTemplate(conditions_fields)

@generated_jit(cache=True)
@overload(Conditions,strict=False)
def conditions_ctor(_vars, dnf=None):
    # print("CONDITIONS CONSTRUCTOR", _vars, dnf)
    if(isinstance(_vars,VarTypeTemplate)):
        # _vars is single Var
        def impl(_vars,dnf=None):
            st = new(ConditionsType)
            st.vars = List.empty_list(BaseVarType)
            st.vars.append(_cast_structref(BaseVarType,_vars)) #TODO should actually make a thing
            st.var_map = build_var_map(st.vars)
            st.dnf = dnf if(dnf) else new_dnf(1)
            st.initialized = False
            return st
    elif(isinstance(_vars,DictType)):
         # _vars is a valid var_map dictionary
        def impl(_vars,dnf=None):
            st = new(ConditionsType)
            st.vars = build_var_list(_vars)
            st.var_map = _vars.copy() # is shallow copy
            st.dnf = dnf if(dnf) else new_dnf(len(_vars))
            st.initialized = False
            return st
    elif(isinstance(_vars,ListType)):
        # _vars is a list of Vars
        def impl(_vars,dnf=None):
            st = new(ConditionsType)
            st.vars = List([_cast_structref(BaseVarType,x) for x in _vars])
            st.var_map = build_var_map(st.vars)
            st.dnf = dnf if(dnf) else new_dnf(len(_vars))
            st.initialized = False
            return st

    return impl

@overload(str)
def str_pterm(self):
    if(not isinstance(self, ConditionsTypeTemplate)): return
    def impl(self):
        s = ""
        # for j, v in enumerate(self.vars):
        #     s += str(v)
        #     if(j < len(self.vars)-1): s += ", "
        # s += '\n'
        for j, conjunct in enumerate(self.dnf):
            # alphas, betas = conjunct
            for i, term in enumerate(conjunct):
                s += "~" if term.negated else ""
                s += "(" + str(term) + ")" 
                if(i < len(conjunct)-1): s += " & "

            # for i, beta_term in enumerate(betas):
            #     s += "~" if beta_term.negated else ""
            #     s += "(" + str(beta_term) + ")" 
            #     if(i < len(betas)-1): s += " & "

            if(j < len(self.dnf)-1): s += " |\\\n"
        return s
    return impl

@njit(cache=True)
def conditions_str(self):
    return str(self)


# NOT(ab+c) = NOT(ab)+c = (a'+b')c' = a'c'+b'c'
# AND((ab+c), (de+f)) = abde+abf+cde+cf
# OR((ab+c), (de+f)) = ab+c+de+f

@njit(cache=True)
def build_var_map(left_vars,right_vars=None):
    ''' Builds a dictionary that maps pointers to Var objects
          to indicies.
    '''
    var_map = Dict.empty(i8,i8)
    for v in left_vars:
        ptr = _pointer_from_struct(v)
        if(ptr not in var_map):
            var_map[ptr] = len(var_map)
    if(right_vars is not None):
        for v in right_vars:
            ptr = _pointer_from_struct(v)
            if(ptr not in var_map):
                var_map[ptr] = len(var_map)
                
    return var_map

@njit(cache=True)
def build_var_list(var_map):
    '''Makes a Var list from a var_map'''
    var_list = List.empty_list(BaseVarType)
    for ptr in var_map:
        var_list.append(_struct_from_pointer(BaseVarType,ptr))
    return var_list





@njit(cache=True)
def conditions_and(left,right):
    '''AND is distributive
    AND((ab+c), (de+f)) = abde+abf+cde+cf'''
    return Conditions(build_var_map(left.vars,right.vars),
                      dnf_and(left.dnf, right.dnf))

@njit(cache=True)
def dnf_and(l_dnf, r_dnf):
    dnf = new_dnf(len(l_dnf)*len(r_dnf))
    for i, l_conjuct in enumerate(l_dnf):
        for j, r_conjuct in enumerate(r_dnf):
            k = i*len(r_dnf) + j
            for x in l_conjuct: dnf[k].append(x)
            for x in r_conjuct: dnf[k].append(x)
            # for x in l_conjuct[0]: dnf[k][0].append(x)
            # for x in r_conjuct[0]: dnf[k][0].append(x)
            # for x in l_conjuct[1]: dnf[k][1].append(x)
            # for x in r_conjuct[1]: dnf[k][1].append(x)
    return dnf


@njit(cache=True)
def conditions_or(left,right):
    '''OR is additive like
    OR((ab+c), (de+f)) = ab+c+de+f'''
    return Conditions(build_var_map(left.vars,right.vars),
                      dnf_or(left.dnf, right.dnf))

@njit(cache=True)
def dnf_or(l_dnf, r_dnf):
    dnf = new_dnf(len(l_dnf)+len(r_dnf))
    for i, conjuct in enumerate(l_dnf):
        for x in conjuct: dnf[i].append(x)
        # for x in conjuct[1]: dnf[i][1].append(x)

    for i, conjuct in enumerate(r_dnf):
        k = len(l_dnf)+i
        for x in conjuct: dnf[k].append(x)
        # for x in conjuct[0]: dnf[k][0].append(x)
        # for x in conjuct[1]: dnf[k][1].append(x)

    return dnf

@njit(cache=True)
def dnf_not(c_dnf):
    dnfs = List.empty_list(pterm_list_list_type)
    for i, conjunct in enumerate(c_dnf):
        dnf = new_dnf(len(conjunct))
        for j, term in enumerate(conjunct):
            dnf[j].append(pterm_not(term))
        # for j, term in enumerate(conjunct[1]):
        #     k = len(conjunct[0]) + j
        #     dnf[k][1].append(pterm_not(term))
        dnfs.append(dnf)

    # print("PHAZZZZZZ")
    out_dnf = dnfs[0]
    for i in range(1,len(dnfs)):
        out_dnf = dnf_and(out_dnf,dnfs[i])
    return out_dnf


@njit(cache=True)
def conditions_not(c):
    '''NOT inverts the qualifiers and terms like
    NOT(ab+c) = NOT(ab)+c = (a'+b')c' = a'c'+b'c'''
    dnf = dnf_not(c.dnf)
    return Conditions(c.var_map, dnf)




@generated_jit(cache=True)
@overload(operator.and_)
def cond_and(l, r):
    if(not isinstance(l,ConditionsTypeTemplate)): return
    if(not isinstance(r,ConditionsTypeTemplate)): return
    return lambda l,r : conditions_and(l, r)

@generated_jit(cache=True)
@overload(operator.or_)
def cond_or(l, r):
    if(not isinstance(l,ConditionsTypeTemplate)): return
    if(not isinstance(r,ConditionsTypeTemplate)): return
    return lambda l,r : conditions_or(l, r)

@generated_jit(cache=True)
@overload(operator.not_)
@overload(operator.invert)
def cond_not(c):
    if(not isinstance(c,ConditionsTypeTemplate)): return
    return lambda c : conditions_not(c)



#### Initialization ####

@njit(cache=True)
def initialize_condition(cond):
    cond.alpha_dnfs = List.empty(dnf_type)
    cond.beta_dnfs = List.empty(dnf_type)

    # Prefill with empty dnfs in case there are unconditioned variables.
    for i,v in enumerate(cond.vars):
        cond.alpha_dnfs.append(new_dnf(0))
        cond.beta_dnfs.append(new_dnf(0))

    





#### PLANNING PLANNING ####

# Conditions should keep an untyped BaseVar 
# BaseVar needs an alias (which can be inferred w/ inspect)


# 4/11 How to organize pair matches
# Previously it was a List over i of Dicts (key concept ind j value
#  is a list of pairs of elements that are mutually consistent

# At Instantiation: 
# -organize the alpha terms by the variables that they test
# -organize the beta terms by the left variable that they test 
# 
