import operator
import numpy as np
from numba import types, njit, i8, u8, i4, u1, i8, literally, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple, UniTuple, optional
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template
from cre.memory import MemoryType, Memory, facts_for_t_id, fact_at_f_id
from cre.fact import define_fact, BaseFactType, cast_fact
from cre.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_ptr,  lower_setattr, lower_getattr, _raw_ptr_from_struct, _ptr_from_struct_incref, _decref_ptr
from cre.utils import assign_to_alias_in_parent_frame, meminfo_type
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.op import GenericOpType, op_str, op_repr, Op
# from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
 # get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode, \
 # LiteralLinkDataType, generate_link_data
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models
from cre.var import *

from operator import itemgetter
from copy import copy

#### Literal Link Data ####

literal_link_data_field_dict = {
    "left_t_id" : u8,
    "right_t_id" : u8,
    "left_facts" : VectorType, #Vector<*Fact>
    "right_facts" : VectorType, #Vector<*Fact>
    
    "change_head": i8,
    "grow_head": i8,
    "change_queue": VectorType,
    "grow_queue": VectorType,
    "mem_grow_queue" : VectorType,
    "mem_change_queue" : VectorType,



    "truth_values" : u1[:,:],
    "left_consistency" : u1[:],
    "right_consistency" : u1[:],
}

literal_link_data_fields = [(k,v) for k,v, in literal_link_data_field_dict.items()]
LiteralLinkData, LiteralLinkDataType = define_structref("LiteralLinkData", 
                literal_link_data_fields, define_constructor=False)


@njit(cache=True)
def generate_link_data(pn, mem):
    '''Takes a prototype predicate node and a knowledge base and returns
        a link_data instance for that predicate node.
    '''
    link_data = new(LiteralLinkDataType)
    link_data.left_t_id = mem.context_data.fact_to_t_id[pn.left_fact_type_name]
    link_data.left_facts = facts_for_t_id(mem.mem_data,i8(link_data.left_t_id)) 
    if(not pn.is_alpha):
        link_data.right_t_id = mem.context_data.fact_to_t_id[pn.right_fact_type_name]
        link_data.right_facts = facts_for_t_id(mem.mem_data,i8(link_data.right_t_id)) 
        link_data.left_consistency = np.empty((0,),dtype=np.uint8)
        link_data.right_consistency = np.empty((0,),dtype=np.uint8)
    else:
        link_data.right_t_id = -1


    link_data.change_head = 0
    link_data.grow_head = 0
    link_data.change_queue = new_vector(8)
    link_data.grow_queue = new_vector(8)

    link_data.mem_grow_queue = mem.mem_data.grow_queue
    link_data.mem_change_queue = mem.mem_data.change_queue
    link_data.truth_values = np.empty((0,0),dtype=np.uint8)
        
    
    return link_data

#### Literal ####


literal_fields_dict = {
    # "str_val" : unicode_type,
    "op" : GenericOpType,
    "var_base_ptrs" : i8[:],#UniTuple(i8,2),
    "negated" : u1,
    "is_alpha" : u1,
    "cre_mem_ptr" : i8,
    "link_data" : LiteralLinkDataType   
}

literal_fields =  [(k,v) for k,v, in literal_fields_dict.items()]

@structref.register
class LiteralTypeTemplate(types.StructRef):
    pass

class Literal(structref.StructRefProxy):
    def __new__(cls, *args):
        return literal_ctor(*args)
        # return structref.StructRefProxy.__new__(cls, *args)
    def __str__(self):
        return literal_str(self)

    def __repr__(self):
        return literal_str(self)

    @property
    def op(self):
        return literal_get_pred_node(self)

@njit(cache=True)
def literal_str(self):
    return op_str(self.op)

@njit(cache=True)
def literal_get_op(self):
    return self.op

define_boxing(LiteralTypeTemplate, Literal)
LiteralType = LiteralTypeTemplate(literal_fields)

@njit(LiteralType(GenericOpType),cache=True)
# @overload(Literal)
def literal_ctor(op):
    st = new(LiteralType)
    st.op = op
    st.var_base_ptrs = np.empty(len(op.base_var_map),dtype=np.int64)
    for i, ptr in enumerate(op.base_var_map):
        st.var_base_ptrs[i] = ptr
    st.negated = 0
    st.is_alpha = u1(len(st.var_base_ptrs) == 1)
    st.cre_mem_ptr = 0
    return st


#TODO: STR
@overload(str)
def _literal_str(self):
    if(not isinstance(self, LiteralTypeTemplate)): return
    def impl(self):
        return literal_str(self)
    return impl

@njit(cache=True)
def literal_copy(self):
    st = new(LiteralType)
    # st.str_val = self.str_val
    st.op = self.op
    st.var_base_ptrs = self.var_base_ptrs
    st.negated = self.negated
    st.is_alpha = self.is_alpha
    st.cre_mem_ptr = self.cre_mem_ptr
    if(self.cre_mem_ptr):
        st.link_data = self.link_data
    return st
    
@njit(cache=True)
def literal_not(self):
    n = literal_copy(self)
    n.negated = not n.negated
    return n

@njit(cache=True)
def literal_to_cond(lit):
    dnf = new_dnf(1)
    # ind = 0 if (lit.is_alpha) else 1
    dnf[0].append(lit)
    _vars = List.empty_list(GenericVarType)

    for ptr in lit.var_base_ptrs:
        _vars.append(_struct_from_ptr(GenericVarType, ptr))
    # if(right_var is not None):
    #     _vars.append(right_var)
    # pt.negated = negated
    c = _conditions_ctor_var_list(_vars, dnf)
    # c = Conditions(_vars, dnf)
    return c

@njit(cache=True)
def op_to_cond(op):
    return literal_to_cond(literal_ctor(op))


#TODO compartator helper?



## dnf_type ##
literal_list_type = ListType(LiteralType)
conjunct_type = literal_list_type
# ab_conjunct_type = #Tuple((literal_list_type, literal_list_type))
dnf_type = ListType(conjunct_type)

## distr_dnf_type ##e
# literal_list_list_type = ListType(literal_list_type)
distr_conj_type = ListType(ListType(LiteralType))
distr_dnf_type = ListType(ListType(ListType(LiteralType)))
# distr_var_spec_conj_type = ListType(LiteralType)

# distr_ab_conjunct_type = Tuple((literal_list_list_type, literal_list_list_type, i8[:,:]))
# distr_dnf_type = ListType(distr_ab_conjunct_type)


conditions_fields_dict = {
    ### Fields that are filled on in instantiation ### 

    # The variables used by the condition
    'vars': ListType(GenericVarType),

    # 'not_vars' : ListType(GenericVarType),

    # The Disjunctive Normal Form of the condition but organized
    #  so that every conjunct has a seperate lists for alpha and 
    #  beta terms.
    'dnf': dnf_type,

    # A mapping from Var pointers to their associated index
    'base_var_map': DictType(i8,i8),

    # Wether or not the conditions object has been initialized
    # 'is_initialized' : u1,

    # A pointer to the Memory the Conditions object is linked to.
    #   If the Memory is not linked defaults to 0.
    'mem_ptr' : i8,

    ### Fields that are filled in after initialization ### 
    "has_distr_dnf" : types.boolean,
    "distr_dnf" : distr_dnf_type,


    # Keep around a pointer and a meminfo for the matcher_inst
    "matcher_inst_ptr" : i8, # Keep this so we can check for zero
    "matcher_inst_meminfo" : meminfo_type, # Keep this so it is decreffed

    # # The alpha parts of '.dnf' organized by which Var in 'vars' they use 
    # 'alpha_dnfs': ListType(dnf_type),

    # # The beta parts of '.dnf' organized by which left Var in 'vars' they use 
    # 'beta_dnfs': ListType(dnf_type),
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
        if(isinstance(other,Op)): other = op_to_cond(other)
        return conditions_and(self, other)
    def __or__(self, other):
        if(isinstance(other,Op)): other = op_to_cond(other)
        return conditions_or(self, other)
    def __not__(self):
        return conditions_not(self)
    def __invert__(self):
        return conditions_not(self)

    def get_ptr_matches(self,mem=None):
        from cre.matching import get_ptr_matches
        return get_ptr_matches(self,mem)

    def get_matches(self, mem=None):
        from cre.rete import MatchIterator
        # return get_matches(self, self.var_base_types, mem=mem)


        return MatchIterator(mem, self)#get_match_iter(mem, self)

    # def __del__(self):
    #     print("CONDITIONS DTOR",self)
    #     conds_dtor(self)
        # if()

    @property
    def var_base_types(self):
        if(not hasattr(self,"_var_base_types")):
            context = cre_context()
            delimited_type_names = conds_get_delimited_type_names(self,";",True).split(";")
            self._var_base_types = tuple([context.type_registry[x] for x in delimited_type_names])
        return self._var_base_types



    def link(self,mem):
        get_linked_conditions_instance(self,mem,copy=False)

    @property
    def signature(self):
        if(not hasattr(self,"_signature")):
            context = cre_context()
            # print(self)
            sig_str = _get_sig_str(self)
            fact_types = sig_str[1:-1].split(",")
            print(fact_types)
            self._signature = types.void(*[context.type_registry[x] for x in fact_types])            

        return self._signature

    @property
    def vars(self):
        return conds_get_vars(self)

    @property
    def dnf(self):
        return conds_get_dnf(self)

    @property
    def distr_dnf(self):
        return conds_get_distr_dnf(self)

    def as_dnf_list(self):
        return as_dnf_list(self)

    def as_distr_dnf_list(self):
        return as_distr_dnf_list(self)

    def __str__(self):
        return conditions_str(self)

    def __repr__(self):
        return conditions_repr(self)

define_boxing(ConditionsTypeTemplate,Conditions)

ConditionsType = ConditionsTypeTemplate(conditions_fields)

@njit(unicode_type(ConditionsType,unicode_type,types.boolean), cache=True)
def conds_get_delimited_type_names(self,delim,ignore_ext_nots):
    s,l = "", len(self.vars)
    for i, v in enumerate(self.vars):
        if(ignore_ext_nots and v.is_not): continue
        s += v.base_type_name
        s += delim
    return s[:-len(delim)]

# @njit(void(ConditionsType),cache=True)
# def conds_dtor(self):
#     if(self.matcher_inst_ptr): 
#         _decref_ptr(self.matcher_inst_ptr)


### Helper Functions for expressing conditions as python lists of cre.Op instances ###

@njit(cache=True)
def _nxt_distr(distr_dnf, disj_i, var_i, lit_i):
    while(True):
        if(len(distr_dnf) <= disj_i):
            raise StopIteration()
        if(len(distr_dnf[disj_i]) <= var_i):
            disj_i +=1; var_i=0; continue;
        if(len(distr_dnf[disj_i][var_i]) <= lit_i):
            var_i +=1; lit_i=0; continue;
        break

    return distr_dnf[disj_i][var_i][lit_i].op, disj_i, var_i, lit_i


def as_distr_dnf_list(distr_dnf):
    lst = []
    disj_i, var_i, lit_i = 0,0,0
    while(True):
        try:
            lit, disj_i, var_i, lit_i = \
                _nxt_distr(distr_dnf, disj_i, var_i, lit_i)
        except StopIteration:
            return lst

        tmp = lst
        if(disj_i >= len(tmp)): tmp.append([])
        tmp = tmp[disj_i]
        if(var_i >= len(tmp)): tmp.append([])
        tmp = tmp[var_i]
        tmp.append(lit)
        lit_i += 1
    return lst


@njit(cache=True)
def nxt_dnf(distr_dnf, disj_i, lit_i):
    while(True):
        if(len(distr_dnf) <= disj_i):
            raise StopIteration()
        if(len(distr_dnf[disj_i]) <= lit_i):
            disj_i +=1; lit_i=0; continue
        break

    return distr_dnf[disj_i][lit_i].op, disj_i, lit_i


def as_dnf_list(dnf):
    lst = []
    disj_i, lit_i = 0,0,0
    while(True):
        try:
            lit, disj_i, var_i, lit_i = \
                _nxt_distr(distr_dnf, disj_i, var_i, lit_i)
        except StopIteration:
            return lst

        tmp = lst
        if(disj_i >= len(tmp)): tmp.append([])
        tmp = tmp[disj_i]
        if(var_i >= len(tmp)): tmp.append([])
        tmp = tmp[var_i]
        tmp.append(lit)
        lit_i += 1
    return lst



@njit(cache=True)
def conds_get_vars(self):
    return self.vars

@njit(cache=True)
def conds_get_dnf(self):
    return self.dnf

@njit(cache=True)
def conds_has_distr_dnf(self):
    return self.has_distr_dnf

@njit(cache=False)
def conds_get_distr_dnf(self):
    if(not self.has_distr_dnf):
        self.distr_dnf = build_distributed_dnf(self)
    return self.distr_dnf


@overload_method(ConditionsTypeTemplate,'get_matches')
def impl_get_matches(self,mem=None):
    from cre.matching import get_matches
    def impl(self,mem=None):
        return get_matches(get_matches)
    return impl


@njit(cache=True)
def new_dnf(n):
    dnf = List.empty_list(conjunct_type)
    for i in range(n):
        dnf.append(List.empty_list(LiteralType))
    return dnf

@njit(cache=True)
def _conditions_ctor_single_var(_vars,dnf=None):
    st = new(ConditionsType)
    st.vars = List.empty_list(GenericVarType)
    st.vars.append(_struct_from_ptr(GenericVarType,_vars.base_ptr)) 
    st.base_var_map = build_base_var_map(st.vars)
    # print("A",st.base_var_map)
    st.dnf = dnf if(dnf) else new_dnf(1)
    st.has_distr_dnf = False
    # st.is_initialized = False
    st.matcher_inst_ptr = 0
    return st

@njit(cache=True)
def _conditions_ctor_base_var_map(_vars,dnf=None):
    st = new(ConditionsType)
    st.vars = build_var_list(_vars)
    st.base_var_map = _vars.copy() # is shallow copy
    st.dnf = dnf if(dnf) else new_dnf(len(_vars))
    st.has_distr_dnf = False
    # st.is_initialized = False
    st.matcher_inst_ptr = 0
    return st

@njit(cache=True)
def _conditions_ctor_var_list(_vars,dnf=None):
    st = new(ConditionsType)
    st.vars = List.empty_list(GenericVarType)
    for x in _vars:
        st.vars.append(_struct_from_ptr(GenericVarType,x.base_ptr))
    # st.vars = List([ for x in _vars])
    st.base_var_map = build_base_var_map(st.vars)
    # print("C",st.base_var_map)
    st.dnf = dnf if(dnf) else new_dnf(len(_vars))
    st.has_distr_dnf = False
    # st.is_initialized = False
    st.matcher_inst_ptr = 0
    return st

@generated_jit(cache=True)
@overload(Conditions,strict=False)
def conditions_ctor(_vars, dnf=None):
    print("CONDITIONS CONSTRUCTOR", _vars, dnf)
    if(isinstance(_vars,VarTypeTemplate)):
        # _vars is single Var
        def impl(_vars,dnf=None):
            return _conditions_ctor_single_var(_vars,dnf)
    elif(isinstance(_vars,DictType)):
         # _vars is a valid base_var_map dictionary
        def impl(_vars,dnf=None):
            return _conditions_ctor_base_var_map(_vars,dnf)
    elif(isinstance(_vars,ListType)):
        def impl(_vars,dnf=None):
            return _conditions_ctor_var_list(_vars,dnf)
            

    return impl

@njit(cache=True)
def _get_sig_str(conds):
    s = "("
    # print("HERE")
    for i, var in enumerate(conds.vars):
        if(var.is_not): continue
        if(len(s) > 1): s += ","
        # print(var.fact_type_name)
        s += var_get_fact_type_name(var)#.fact_type_name
        # if(i < len(conds.vars)-1): s += ","
    return s + ")"

@njit(cache=True)
def conditions_repr(self,alias=None):
    s = ""
    for j, v in enumerate(self.vars):
        s += v.alias
        if(j < len(self.vars)-1): s += ", "
    s += " = "
    for j, v in enumerate(self.vars):
        prefix = "NOT" if(v.is_not) else "Var"
        s_v = prefix + "(" + v.base_type_name + ")"
        # : s_v = "NOT(" + s_v +")"
        s += s_v
        if(j < len(self.vars)-1): s += ", "
    s += "\n"
    if(alias is not None):
        s += alias +" = "

    s += conditions_str(self,add_non_conds=True)
    return s


@njit(cache=True)
def conditions_str(self,add_non_conds=False):
    s = ""
    # for j, v in enumerate(self.vars):
    #     s += str(v)
    #     if(j < len(self.vars)-1): s += ", "
    # s += '\n'
    used_var_ptrs = Dict.empty(i8,u1)
    for j, conjunct in enumerate(self.dnf):
        for i, lit in enumerate(conjunct):
            s += "~" if lit.negated else ""
            s += str(lit)
            if(i < len(conjunct)-1): s += " & "
            if(add_non_conds): 
                for var_ptr in lit.var_base_ptrs:
                    used_var_ptrs[var_ptr] = u1(1)

        # for i, beta_lit in enumerate(beta_conjunct):
        #     s += "~" if beta_lit.negated else ""
        #     s += str(beta_lit)
        #     if(i < len(beta_conjunct)-1): s += " & "
        #     if(add_non_conds): 
        #         used_var_ptrs[beta_lit.var_base_ptrs[0]] = u1(1)
        #         used_var_ptrs[beta_lit.var_base_ptrs[1]] = u1(1)

        if(j < len(self.dnf)-1): s += " |\\\n"

    if(add_non_conds):
        was_prev =  True if(len(used_var_ptrs) > 0) else False

        for j, v in enumerate(self.vars):
            if(v.base_ptr not in used_var_ptrs):
                if(was_prev): s += " & "
                s += v.alias
                was_prev = True

    return s

@overload(str)
def overload_conds_str(self):
    if(not isinstance(self, ConditionsTypeTemplate)): return
    def impl(self):
        return conds_str(self)
        
    return impl



# NOT(ab+c) = NOT(ab)+c = (a'+b')c' = a'c'+b'c'
# AND((ab+c), (de+f)) = abde+abf+cde+cf
# OR((ab+c), (de+f)) = ab+c+de+f

@njit(cache=True)
def build_base_var_map(left_vars,right_vars=None):
    ''' Builds a dictionary that maps pointers to Var objects
          to indicies.
    '''
    base_var_map = Dict.empty(i8,i8)
    for v in left_vars:
        ptr = v.base_ptr
        if(ptr not in base_var_map):
            base_var_map[ptr] = len(base_var_map)
    if(right_vars is not None):
        for v in right_vars:
            ptr = v.base_ptr
            if(ptr not in base_var_map):
                base_var_map[ptr] = len(base_var_map)
                
    return base_var_map

@njit(cache=True)
def build_var_list(base_var_map):
    '''Makes a Var list from a base_var_map'''
    var_list = List.empty_list(GenericVarType)
    for ptr in base_var_map:
        var_list.append(_struct_from_ptr(GenericVarType,ptr))
    return var_list





@njit(cache=True)
def _conditions_and(left, right):
    '''AND is distributive
    AND((ab+c), (de+f)) = abde+abf+cde+cf'''
    return _conditions_ctor_base_var_map(
                build_base_var_map(left.vars,right.vars),
                dnf_and(left.dnf, right.dnf)
            )

# @njit(cache=True)
# def conditions_and_var(left,right):
#     right_c = _conditions_ctor_single_var(right)
#     return conditions_and(left,right_c)

# @njit(cache=True)
# def var_and_conditions(left,right):
#     left_c = _conditions_ctor_single_var(left)
#     return conditions_and(left_c,right)

# @njit(cache=True)
# def var_and(left,right):
#     left_c = _conditions_ctor_single_var(left)
#     right_c = _conditions_ctor_single_var(right)
#     return conditions_and(left_c,right_c)


# @njit(cache=True)
@generated_jit(cache=True)    
def conditions_and(self, other):
    if(isinstance(other, VarTypeTemplate)):
        if(isinstance(self,VarTypeTemplate)):
            def impl(self,other):
                self_c = _conditions_ctor_single_var(self)
                other_c = _conditions_ctor_single_var(other)
                return _conditions_and(self_c,other_c)
        else:
            def impl(self,other):
                other_c = _conditions_ctor_single_var(other)
                return _conditions_and(self,other_c)
    else:
        if(isinstance(self,VarTypeTemplate)):
            def impl(self,other):
                self_c = _conditions_ctor_single_var(self)
                return _conditions_and(self_c,other)
        else:
            def impl(self,other):
                return _conditions_and(self,other)

    return impl


@njit(cache=True)
def dnf_and(l_dnf, r_dnf):
    dnf = new_dnf(len(l_dnf)*len(r_dnf))
    for i, l_conjunct in enumerate(l_dnf):
        for j, r_conjunct in enumerate(r_dnf):
            k = i*len(r_dnf) + j
            for x in l_conjunct: dnf[k].append(x)
            for x in r_conjunct: dnf[k].append(x)
            # for x in l_conjunct[0]: dnf[k][0].append(x)
            # for x in r_conjunct[0]: dnf[k][0].append(x)
            # for x in l_conjunct[1]: dnf[k][1].append(x)
            # for x in r_conjunct[1]: dnf[k][1].append(x)
    return dnf


@njit(cache=True)
def _conditions_or(left,right):
    '''OR is additive like
    OR((ab+c), (de+f)) = ab+c+de+f'''
    return Conditions(build_base_var_map(left.vars,right.vars),
                      dnf_or(left.dnf, right.dnf))

@generated_jit(cache=True)    
def conditions_or(self,other):
    if(isinstance(other, VarTypeTemplate)):
        if(isinstance(self,VarTypeTemplate)):
            def impl(self,other):
                self_c = _conditions_ctor_single_var(self)
                other_c = _conditions_ctor_single_var(other)
                return _conditions_or(self_c,other_c)
        else:
            def impl(self,other):
                other_c = _conditions_ctor_single_var(other)
                return _conditions_or(self,other_c)
    else:
        if(isinstance(self,VarTypeTemplate)):
            def impl(self,other):
                self_c = _conditions_ctor_single_var(self)
                return _conditions_or(self_c,other)
        else:
            def impl(self,other):
                return _conditions_or(self,other)

    return impl


@njit(cache=True)
def dnf_or(l_dnf, r_dnf):
    dnf = new_dnf(len(l_dnf)+len(r_dnf))
    for i, conjuct in enumerate(l_dnf):
        for x in conjuct: dnf[i].append(x)
        # for x in conjuct[0]: dnf[i][0].append(x)
        # for x in conjuct[1]: dnf[i][1].append(x)

    for i, conjuct in enumerate(r_dnf):
        k = len(l_dnf)+i
        for x in conjuct: dnf[k].append(x)
        # for x in conjuct[0]: dnf[k][0].append(x)
        # for x in conjuct[1]: dnf[k][1].append(x)

    return dnf

@njit(cache=True)
def dnf_not(c_dnf):
    dnfs = List.empty_list(dnf_type)
    # return new_dnf(len(c_dnf))
    for i, conjunct in enumerate(c_dnf):
        dnf = new_dnf(len(conjunct))
        for j, lit in enumerate(conjunct):
            # dnf[j].append(lit)
            dnf[j].append(literal_not(lit))
        # for j, term in enumerate(conjunct[0]):
    #     #     dnf[j][0].append(literal_not(term))
    #     # for j, term in enumerate(conjunct[1]):
    #     #     k = len(conjunct[0]) + j
    #     #     dnf[k][1].append(literal_not(term))
        dnfs.append(dnf)

    out_dnf = dnfs[0]
    for i in range(1,len(dnfs)):
        out_dnf = dnf_and(out_dnf,dnfs[i])
    return out_dnf


@njit(GenericVarType(GenericVarType,),cache=True)
def _build_var_conjugate(v):
    if(v.conj_ptr == 0):
        conj = new(GenericVarType)
        var_memcopy(v,conj)
        conj.is_not = u1(0) if v.is_not else u1(1)
        conj.conj_ptr = _ptr_from_struct_incref(v)
        v.conj_ptr = _ptr_from_struct_incref(conj)
    else:
        conj = _struct_from_ptr(GenericVarType, v.conj_ptr)
    return conj



@generated_jit(cache=True)
def _var_NOT(c):
    '''Implementation of NOT for Vars moved outside of NOT 
        definition to avoid recursion issue'''
    if(isinstance(c,VarTypeTemplate)):
        st_typ = c
        def impl(c):
            if(c.conj_ptr == 0):
                base = _struct_from_ptr(GenericVarType, c.base_ptr)

                conj_base = _build_var_conjugate(base)
                g_conj = _build_var_conjugate(_cast_structref(GenericVarType,c))

                g_conj.base_ptr = _raw_ptr_from_struct(conj_base)

            st = _struct_from_ptr(st_typ, c.conj_ptr)
            return st
        return impl

@njit(cache=True)
def _conditions_NOT(c):
    new_vars = List.empty_list(GenericVarType)
    ptr_map = Dict.empty(i8,i8)
    for var in c.vars:
        new_var = _var_NOT(var)
        ptr_map[_raw_ptr_from_struct(var)]  = _raw_ptr_from_struct(new_var)
        new_vars.append(new_var)

    dnf = dnf_copy(c.dnf,shallow=False)

    for i, (alpha_conjuncts, beta_conjuncts) in enumerate(dnf):
        for alpha_literal in alpha_conjuncts: 
            alpha_literal.var_base_ptrs = (ptr_map[alpha_literal.var_base_ptrs[0]],0)
        for beta_literal in beta_conjuncts:
            t = (ptr_map[beta_literal.var_base_ptrs[0]], ptr_map[beta_literal.var_base_ptrs[1]])
            beta_literal.var_base_ptrs = t

    return Conditions(new_vars, dnf)


def NOT(c, alias=None):
    ''' Applies existential NOT in python context)''' 
    if(isinstance(c, (ConditionsTypeTemplate, Conditions))):
        # When NOT() is applied to a Conditions object
        #  apply NOT() to all vars and make sure the pointers
        #  for the new vars are tracked in the dnf
        return _conditions_NOT(c)            
    elif(isinstance(c, (VarTypeTemplate, Var))):    
        # Mark a var as NOT(v) meaning we test that nothing
        #  binds to it. Return a new instance to maintain
        #  value semantics.
        out = _var_NOT(c)
        assign_to_alias_in_parent_frame(out,out.alias)
        return out
        
    elif(hasattr(c,'fact_type') or hasattr(c,'fact_ctor')):
        # 
        c = getattr(c, 'fact_type',c)
        var = Var(c,alias,skip_assign_alias=True)
        out = _var_NOT(var)
        out._fact_type = var._fact_type
        out._head_type = var._head_type
        assign_to_alias_in_parent_frame(out,alias)
        return out


@overload(NOT)
def overload_NOT(c, alias=None):
    ''' Applies existential NOT in numba context -- works same as above''' 
    if(isinstance(c,ConditionsTypeTemplate)):
        def impl(c,alias=None):
            return _conditions_NOT(c)
        return impl
    elif(isinstance(c,VarTypeTemplate)):
        #TODO: Possible to assign to alias in parent in jit context?
        def impl(c,alias=None):
            return _var_NOT(c)
        return impl
    elif(hasattr(c,'fact_type') or hasattr(c,'fact_ctor')):
        #TODO: Possible to assign to alias in parent in jit context?
        raise NotImplemented()
        c = getattr(c, 'fact_type',c)
        def impl(c, alias=None):
            return None




@generated_jit(cache=True)
def conditions_not(c):
    '''Defines ~x for Var and Conditions''' 
    if(isinstance(c,ConditionsTypeTemplate)):
        # ~ operation inverts the qualifiers and terms like
        #  ~(ab+c) = ~(ab)c' = (a'+b')c' = a'c'+b'c'
        def impl(c):
            dnf = dnf_not(c.dnf)
            return Conditions(c.base_var_map, dnf)
    elif(isinstance(c,VarTypeTemplate)):
        # If we applied to a var then serves as NOT()
        def impl(c):
            return _var_NOT(c)
    return impl



@njit(cache=True)
def conditions_not(c):
    '''NOT inverts the qualifiers and terms like
    NOT(ab+c) = NOT(ab)+c = (a'+b')c' = a'c'+b'c'''
    dnf = dnf_not(c.dnf)
    return Conditions(c.base_var_map, dnf)




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





#### Linking ####

@njit(cache=True)
def link_literal_instance(literal, mem):
    link_data = generate_link_data(literal.pred_node, mem)
    literal.link_data = link_data
    literal.mem_ptr = _raw_ptr_from_struct(mem)
    return literal

@njit(cache=True)
def dnf_copy(dnf,shallow=True):
    ndnf = new_dnf(len(dnf))
    for i, (alpha_lit, beta_lit) in enumerate(dnf):
        for alpha_literal in alpha_lit:
            new_alpha = alpha_literal if(shallow) else literal_copy(alpha_literal)
            ndnf[i][0].append(new_alpha)
        for beta_literal in beta_lit:
            new_beta = beta_literal if(shallow) else literal_copy(beta_literal)
            ndnf[i][1].append(new_beta)
    return ndnf

@njit(cache=True)
def get_linked_conditions_instance(conds, mem, copy=False):
    dnf = dnf_copy(conds.dnf,shallow=False) if copy else conds.dnf
    for alpha_conjunct, beta_conjunct in dnf:
        for term in alpha_conjunct: link_literal_instance(term, mem)
        for term in beta_conjunct: link_literal_instance(term, mem)
    if(copy):
        new_conds = Conditions(conds.base_var_map, dnf)
        # if(conds.is_initialized): initialize_conditions(new_conds)
        conds = new_conds

    #Note... maybe it's simpler to just make mem an optional(memType)
    old_ptr = conds.mem_ptr
    conds.mem_ptr = _ptr_from_struct_incref(mem)
    if(old_ptr != 0): _decref_ptr(old_ptr)
    return conds


#### Initialization ####


@njit(cache=True)
def build_distributed_dnf(c,index_map=None):
    # print("c.vars", c.vars)
    distr_dnf = List.empty_list(distr_conj_type)

    if(index_map is None):
        index_map = Dict.empty(i8, i8)
        for i, v in enumerate(c.vars):
            index_map[v.base_ptr] = i

    for conjunct in c.dnf:
        var_spec_conj_list = List.empty_list(literal_list_type)
        distr_dnf.append(var_spec_conj_list)
        for i, v in enumerate(c.vars):
            var_spec_conj_list.append(List.empty_list(LiteralType))

    for i, conjunct in enumerate(c.dnf):
        distr_conjuct = distr_dnf[i]
        for j, lit in enumerate(conjunct):
            max_ind = -1
            for base_ptr in lit.op.base_var_map:
                ind = index_map[base_ptr]
                if(ind > max_ind): max_ind = ind

            insertion_conj = distr_conjuct[max_ind]
            was_inserted = False
            
            lit_n_vars = len(lit.op.base_var_map)

            for k in range(len(insertion_conj)-1,-1,-1):
                if(lit_n_vars >= len(insertion_conj[k].op.base_var_map)):
                    was_inserted = True
                    insertion_conj.insert(k+1, lit)
                    break
                else:
                    continue
                    
            if(not was_inserted):
                insertion_conj.insert(0,lit)
    c.distr_dnf = distr_dnf
    c.has_distr_dnf = True

    # print(distr_dnf)
    return distr_dnf


# @njit(cache=True)
# def initialize_conditions(conds):
#     distr_dnf = List.empty_list(distr_ab_conjunct_type)
#     n_vars = len(conds.vars)
#     for conjunct in conds.dnf:
#         alpha_conjuncts = List.empty_list(literal_list_type)
#         beta_conjuncts = List.empty_list(literal_list_type)
        
#         for _ in range(n_vars): alpha_conjuncts.append(List.empty_list(LiteralType))
        
#         for term in ac:
#             i = conds.base_var_map[term.var_base_ptrs[0]]
#             alpha_conjuncts[i].append(term)

        

#         beta_inds = -np.ones((n_vars,n_vars),dtype=np.int64)
#         for term in bc:
#             i = conds.base_var_map[term.var_base_ptrs[0]]
#             j = conds.base_var_map[term.var_base_ptrs[1]]
#             if(beta_inds[i,j] == -1):
#                 k = len(beta_conjuncts)
#                 beta_inds[i,j] = k
#                 beta_conjuncts.append(List.empty_list(LiteralType))
                
#             beta_conjuncts[beta_inds[i,j]].append(term)
#             # beta_conjuncts.append(term)

#             # beta_conjuncts[j].append(term)

#         distr_dnf.append((alpha_conjuncts, beta_conjuncts, beta_inds))
#     conds.distr_dnf = distr_dnf




# @njit(cache=True)
# def initialize_conditions(conds):
#     print("INITIALIZING")
#     conds.alpha_dnfs = List.empty_list(dnf_type)
#     conds.beta_dnfs = List.empty_list(dnf_type)

#     print(conds.alpha_dnfs)

#     # Prefill with empty dnfs in case there are uncondsitioned variables.
#     for i,v in enumerate(conds.vars):
#         conds.alpha_dnfs.append(new_dnf(0))
#         conds.beta_dnfs.append(new_dnf(0))

#     # print(len(conds.vars))
#     for conjunct in conds.dnf:
#         a_is_in_this_conj = np.zeros(len(conds.vars),dtype=np.uint8)
#         b_is_in_this_conj = np.zeros(len(conds.vars),dtype=np.uint8)

#         for term in conjunct:
#             ptr = term.var_base_ptrs[0]
#             # ptr = _raw_ptr_from_struct(l_var)
#             # print(">>>", ptr, conds.base_var_map)
#             ind = conds.base_var_map[ptr]
#             # print(ind, len(a_is_in_this_conj))

#             if(term.is_alpha):
#                 if(not a_is_in_this_conj[ind]):
#                     alpha_conjunct = List.empty_list(PTermType)
#                     conds.alpha_dnfs[ind].append(alpha_conjunct)
#                     a_is_in_this_conj[ind] = 1
#                 else:
#                     print(len(conds.alpha_dnfs[ind]))
#                     alpha_conjunct = conds.alpha_dnfs[ind][-1]
#                 alpha_conjunct.append(term)
#             else:
#                 if(not b_is_in_this_conj[ind]):
#                     beta_conjunct = List.empty_list(PTermType)
#                     conds.beta_dnfs[ind].append(beta_conjunct)
#                     b_is_in_this_conj[ind] = 1
#                 else:
#                     print(len(conds.alpha_dnfs[ind]))
#                     beta_conjunct = conds.beta_dnfs[ind][-1]
#                 beta_conjunct.append(term)

            # print(a_is_in_this_conj)
    # print("BEF")
    # print(conds.alpha_dnfs)
    # print("AFT")
















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
