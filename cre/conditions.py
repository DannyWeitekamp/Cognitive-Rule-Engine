import operator
import numpy as np
from numba import types, njit, f8, i8, u8, i4, u1, i8, i2, u2, literally, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple, UniTuple, optional, boolean
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import lower_cast, overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic, SentryLiteralArgs
from numba.core.typing.templates import AttributeTemplate
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template
from cre.fact import define_fact, BaseFact, cast_fact, FactProxy
from cre.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_ptr,  lower_setattr, lower_getattr, _raw_ptr_from_struct, _ptr_from_struct_incref, _decref_ptr
from cre.utils import assign_to_alias_in_parent_frame, meminfo_type
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.op import GenericOpType, op_str, op_repr, Op, op_copy
from cre.cre_object import CREObjType, CREObjTypeClass
from cre.core import T_ID_CONDITIONS, T_ID_LITERAL, register_global_default
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
    # "grow_queue": VectorType,
    # "ms_grow_queue" : VectorType,
    "ms_change_queue" : VectorType,



    "truth_values" : u1[:,:],
    "left_consistency" : u1[:],
    "right_consistency" : u1[:],
}

literal_link_data_fields = [(k,v) for k,v, in literal_link_data_field_dict.items()]
LiteralLinkData, LiteralLinkDataType = define_structref("LiteralLinkData", 
                literal_link_data_fields, define_constructor=False)


@njit(cache=True)
def generate_link_data(pn, ms):
    '''Takes a prototype predicate node and a knowledge base and returns
        a link_data instance for that predicate node.
    '''
    link_data = new(LiteralLinkDataType)
    link_data.left_t_id = ms.context_data.fact_to_t_id[pn.left_fact_type_name]
    link_data.left_facts = facts_for_t_id(ms,i8(link_data.left_t_id)) 
    if(not pn.is_alpha):
        link_data.right_t_id = ms.context_data.fact_to_t_id[pn.right_fact_type_name]
        link_data.right_facts = facts_for_t_id(ms,i8(link_data.right_t_id)) 
        link_data.left_consistency = np.empty((0,),dtype=np.uint8)
        link_data.right_consistency = np.empty((0,),dtype=np.uint8)
    else:
        link_data.right_t_id = -1


    link_data.change_head = 0
    link_data.grow_head = 0
    link_data.change_queue = new_vector(8)
    # link_data.grow_queue = new_vector(8)

    # link_data.ms_grow_queue = ms.grow_queue
    link_data.ms_change_queue = ms.change_queue
    link_data.truth_values = np.empty((0,0),dtype=np.uint8)
        
    
    return link_data

#### Literal ####


literal_fields_dict = {
    # "str_val" : unicode_type,
    **cre_obj_field_dict,
    "op" : GenericOpType,
    "var_base_ptrs" : i8[:],#UniTuple(i8,2),
    "negated" : u1,
    "is_alpha" : u1,
    "cre_ms_ptr" : i8,
    "link_data" : LiteralLinkDataType   
}

literal_fields =  [(k,v) for k,v, in literal_fields_dict.items()]

@structref.register
class LiteralTypeClass(CREObjTypeClass):
    def __str__(self):
        return f"cre.LiteralType"


# @lower_cast(LiteralTypeClass, CREObjType)
# def upcast(context, builder, fromty, toty, val):
#     return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)

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

define_boxing(LiteralTypeClass, Literal)
LiteralType = LiteralTypeClass(literal_fields)
register_global_default("Literal", LiteralType)

@njit(LiteralType(GenericOpType),cache=True)
# @overload(Literal)
def literal_ctor(op):
    st = new(LiteralType)
    st.idrec = encode_idrec(T_ID_LITERAL, 0, 0)
    st.op = op
    st.var_base_ptrs = np.empty(len(op.base_var_map),dtype=np.int64)
    for i, ptr in enumerate(op.base_var_map):
        st.var_base_ptrs[i] = ptr
    st.negated = 0
    st.is_alpha = u1(len(st.var_base_ptrs) == 1)
    st.cre_ms_ptr = 0
    return st


#TODO: STR
@overload(str)
def _literal_str(self):
    if(not isinstance(self, LiteralTypeClass)): return
    def impl(self):
        return literal_str(self)
    return impl

@njit(cache=True)
def literal_copy(self):
    st = new(LiteralType)
    # st.str_val = self.str_val
    st.idrec = self.idrec
    st.op = self.op
    st.var_base_ptrs = self.var_base_ptrs
    st.negated = self.negated
    st.is_alpha = self.is_alpha
    st.cre_ms_ptr = self.cre_ms_ptr
    if(self.cre_ms_ptr):
        st.link_data = self.link_data
    return st
    
@njit(cache=True)
def literal_not(self):
    n = literal_copy(self)
    n.negated = not n.negated
    return n


literal_unique_tuple_type = Tuple((i8, i8, unicode_type))
@njit(literal_unique_tuple_type(LiteralType), cache=True)
def literal_get_unique_tuple(self):
    '''Outputs a tuple that uniquely identifies an instance
         of a literal independant of the base Vars of its underlying op.
    '''
    deref_str = ""
    for var in self.op.head_vars:
        deref_strs = List.empty_list(unicode_type)    
        if(len(var.deref_infos) > 0):
            s = ""
            for i, d in enumerate(var.deref_infos):
                delim = "," if i != len(var.deref_infos)-1 else ""
                s += f'({str(i8(d.type))},{str(i8(d.t_id))},{str(i8(d.a_id))},{str(i8(d.offset))}){delim}'
                deref_strs.append(s)
        
        deref_str += f"[{','.join(deref_strs)}]"

    return (i8(self.negated), self.op.match_head_ptrs_addr, deref_str)


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
    **cre_obj_field_dict,

    # The variables used by the condition
    'vars': ListType(GenericVarType),

    # 'not_vars' : ListType(GenericVarType),

    # The Disjunctive Normal Form of the condition.
    'dnf': dnf_type,

    # A mapping from Var pointers to their associated index
    'base_var_map': DictType(i8,i8),

    # Wether or not the conditions object has been initialized
    # 'is_initialized' : u1,

    # A pointer to the MemSet the Conditions object is linked to.
    #   If the MemSet is not linked defaults to 0.
    'ms_ptr' : i8,

    ### Fields that are filled in after initialization ### 
    "has_distr_dnf" : types.boolean,
    "distr_dnf" : distr_dnf_type,


    # Keep around a pointer and a meminfo for the matcher_inst
    "matcher_inst_ptr" : ptr_t, # Keep this so we can check for zero
    # "matcher_inst_meminfo" : meminfo_type, # Keep this so it is decreffed

    # # The alpha parts of '.dnf' organized by which Var in 'vars' they use 
    # 'alpha_dnfs': ListType(dnf_type),

    # # The beta parts of '.dnf' organized by which left Var in 'vars' they use 
    # 'beta_dnfs': ListType(dnf_type),
}

conditions_fields =  [(k,v) for k,v, in conditions_fields_dict.items()]

@structref.register
class ConditionsTypeClass(CREObjTypeClass):
    def __str__(self):
        return f"cre.ConditionsType"


ConditionsType = ConditionsTypeClass(conditions_fields)
register_global_default("Conditions", ConditionsType)

# lower_cast(ConditionsTypeClass, CREObjType)(impl_cre_obj_upcast)

# Manually register the type to avoid automatic getattr overloading 
# default_manager.register(VarTypeClass, models.StructRefModel)
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

    def get_ptr_matches(self,ms=None):
        from cre.matching import get_ptr_matches
        return get_ptr_matches(self,ms)

    def get_matches(self, ms=None):
        from cre.rete import MatchIterator
        return MatchIterator(ms, self)

    def check_match(self, match, ms=None):
        from cre.rete import check_match
        idrecs = np.array([x.idrec for x in match],dtype=np.uint64)
        return check_match(self, idrecs, ms)

    def score_match(self, match, ms=None):
        from cre.rete import score_match
        idrecs = np.array([x.idrec for x in match],dtype=np.uint64)
        return score_match(self, idrecs, ms)

    def antiunify(self, other, return_score=False, normalize='left',
         fix_same_var=False, fix_same_alias=False):
        return conds_antiunify(self, other, return_score, normalize,
                    fix_same_var, fix_same_alias) 
    
    @property
    def var_base_types(self):
        if(not hasattr(self,"_var_base_types")):
            context = cre_context()
            var_t_ids = conds_get_var_t_ids(self)
            self._var_base_types = tuple([context.get_type(t_id=t_id) for t_id in var_t_ids])
            # delimited_type_names = conds_get_delimited_type_names(self,";",True).split(";")
            # self._var_base_types = tuple([context.type_registry[x] for x in delimited_type_names])
            
        return self._var_base_types



    def link(self,ms):
        get_linked_conditions_instance(self,ms,copy=False)

    @property
    def signature(self):
        if(not hasattr(self,"_signature")):
            context = cre_context()
            # print(self)
            sig_str = _get_sig_str(self)
            fact_types = sig_str[1:-1].split(",")
            print(fact_types)
            self._signature = types.void(*[context.name_to_type[x] for x in fact_types])            

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

    @property
    def rete_graph(self):
        from cre.rete import conds_get_rete_graph
        return conds_get_rete_graph(self)

    def as_dnf_list(self):
        return as_dnf_list(self)

    def as_distr_dnf_list(self):
        return as_distr_dnf_list(self)

    def __str__(self):
        return conditions_str(self)

    def __repr__(self):
        return conditions_repr(self)


    def from_facts(facts,_vars=None, *args, **kwargs):
        return conditions_from_facts(facts, _vars, *args, **kwargs)


define_boxing(ConditionsTypeClass,Conditions)


@njit(u2[::1](ConditionsType))
def conds_get_var_t_ids(self):
    t_ids = np.empty((len(self.vars),),dtype=np.uint16)
    for i, v in enumerate(self.vars):
        t_ids[i] = v.base_t_id
    return t_ids


# @njit(unicode_type(ConditionsType,unicode_type,types.boolean), cache=True)
# def conds_get_delimited_type_names(self,delim,ignore_ext_nots):
#     s,l = "", len(self.vars)
#     for i, v in enumerate(self.vars):
#         if(ignore_ext_nots and v.is_not): continue
#         s += v.base_type_name
#         s += delim
#     return s[:-len(delim)]

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



# TODO: Reimplement
# @overload_method(ConditionsTypeClass,'get_matches')
# def impl_get_matches(self,ms=None):
#     from cre.matching import get_matches
#     def impl(self,ms=None):
#         return get_matches(ms)
#     return impl


@njit(cache=True)
def new_dnf(n):
    dnf = List.empty_list(conjunct_type)
    for i in range(n):
        dnf.append(List.empty_list(LiteralType))
    return dnf

@njit(cache=True)
def _conditions_ctor_dnf(dnf):
    st = new(ConditionsType)
    st.idrec = encode_idrec(T_ID_CONDITIONS, 0, 0)
    st.vars = List.empty_list(GenericVarType)    
    st.base_var_map = Dict.empty(i8,i8)
    for conj in dnf: 
        for lit in conj:
            for b_ptr in lit.var_base_ptrs:
                if(b_ptr not in st.base_var_map):
                   st.base_var_map[b_ptr] = len(st.base_var_map)
                   st.vars.append(_struct_from_ptr(GenericVarType, b_ptr))
    st.dnf = dnf
    st.has_distr_dnf = False
    st.matcher_inst_ptr = 0
    return st




@njit(cache=True)
def _conditions_ctor_single_var(_vars,dnf=None):
    st = new(ConditionsType)
    st.idrec = encode_idrec(T_ID_CONDITIONS, 0, 0)
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
    st.idrec = encode_idrec(T_ID_CONDITIONS, 0, 0)
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
    st.idrec = encode_idrec(T_ID_CONDITIONS, 0, 0)
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
    if(isinstance(_vars,VarTypeClass)):
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


from cre.var import get_base_type_name
@njit(cache=True)
def conditions_repr(self,alias=None):
    s = ""
    for j, v in enumerate(self.vars):
        s += v.alias
        if(j < len(self.vars)-1): s += ", "
    s += " = "
    for j, v in enumerate(self.vars):
        prefix = "NOT" if(v.is_not) else "Var"
        s_v = prefix + "(" + get_base_type_name(v) + ")"
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
    if(not isinstance(self, ConditionsTypeClass)): return
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
        ptr = i8(v.base_ptr)
        if(ptr not in base_var_map):
            base_var_map[ptr] = len(base_var_map)
    if(right_vars is not None):
        for v in right_vars:
            ptr = i8(v.base_ptr)
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



@njit(ConditionsType(LiteralType,),cache=True)
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

@njit(ConditionsType(GenericOpType,), cache=True)
def op_to_cond(op):
    return literal_to_cond(literal_ctor(op))




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
    if(isinstance(other, VarTypeClass)):
        if(isinstance(self,VarTypeClass)):
            def impl(self,other):
                self_c = _conditions_ctor_single_var(self)
                other_c = _conditions_ctor_single_var(other)
                return _conditions_and(self_c,other_c)
        else:
            def impl(self,other):
                other_c = _conditions_ctor_single_var(other)
                return _conditions_and(self,other_c)
    else:
        if(isinstance(self,VarTypeClass)):
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
    if(isinstance(other, VarTypeClass)):
        if(isinstance(self,VarTypeClass)):
            def impl(self,other):
                self_c = _conditions_ctor_single_var(self)
                other_c = _conditions_ctor_single_var(other)
                return _conditions_or(self_c,other_c)
        else:
            def impl(self,other):
                other_c = _conditions_ctor_single_var(other)
                return _conditions_or(self,other_c)
    else:
        if(isinstance(self,VarTypeClass)):
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
    if(isinstance(c,VarTypeClass)):
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
    if(isinstance(c, (ConditionsTypeClass, Conditions))):
        # When NOT() is applied to a Conditions object
        #  apply NOT() to all vars and make sure the pointers
        #  for the new vars are tracked in the dnf
        return _conditions_NOT(c)            
    elif(isinstance(c, (VarTypeClass, Var))):    
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
    if(isinstance(c,ConditionsTypeClass)):
        def impl(c,alias=None):
            return _conditions_NOT(c)
        return impl
    elif(isinstance(c,VarTypeClass)):
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
    if(isinstance(c,ConditionsTypeClass)):
        # ~ operation inverts the qualifiers and terms like
        #  ~(ab+c) = ~(ab)c' = (a'+b')c' = a'c'+b'c'
        def impl(c):
            dnf = dnf_not(c.dnf)
            return Conditions(c.base_var_map, dnf)
    elif(isinstance(c,VarTypeClass)):
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
    if(not isinstance(l,ConditionsTypeClass)): return
    if(not isinstance(r,ConditionsTypeClass)): return
    return lambda l,r : conditions_and(l, r)

@generated_jit(cache=True)
@overload(operator.or_)
def cond_or(l, r):
    if(not isinstance(l,ConditionsTypeClass)): return
    if(not isinstance(r,ConditionsTypeClass)): return
    return lambda l,r : conditions_or(l, r)

@generated_jit(cache=True)
@overload(operator.not_)
@overload(operator.invert)
def cond_not(c):
    if(not isinstance(c,ConditionsTypeClass)): return
    return lambda c : conditions_not(c)





#### Linking ####

@njit(cache=True)
def link_literal_instance(literal, ms):
    link_data = generate_link_data(literal.pred_node, ms)
    literal.link_data = link_data
    literal.ms_ptr = _raw_ptr_from_struct(ms)
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
def get_linked_conditions_instance(conds, ms, copy=False):
    dnf = dnf_copy(conds.dnf,shallow=False) if copy else conds.dnf
    for alpha_conjunct, beta_conjunct in dnf:
        for term in alpha_conjunct: link_literal_instance(term, ms)
        for term in beta_conjunct: link_literal_instance(term, ms)
    if(copy):
        new_conds = Conditions(conds.base_var_map, dnf)
        # if(conds.is_initialized): initialize_conditions(new_conds)
        conds = new_conds

    #Note... maybe it's simpler to just make mem an optional(memType)
    old_ptr = conds.ms_ptr
    conds.ms_ptr = _raw_ptr_from_struct(ms)#_ptr_from_struct_incref(mem)
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
            index_map[i8(v.base_ptr)] = i

    # Fill with empty lists
    for conjunct in c.dnf:
        var_spec_conj_list = List.empty_list(literal_list_type)
        distr_dnf.append(var_spec_conj_list)
        for i, v in enumerate(c.vars):
            var_spec_conj_list.append(List.empty_list(LiteralType))
    
    # Fill lists
    for i, conjunct in enumerate(c.dnf):
        distr_conjuct = distr_dnf[i]
        for j, lit in enumerate(conjunct):
            max_ind = -1
            for base_ptr in lit.op.base_var_map:
                ind = index_map[i8(base_ptr)]
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



# -----------------------------------------------------------------------
# : Conditions.antiunify()

lit_list = ListType(LiteralType)
lit_unq_tup_type = Tuple((i8,i8,unicode_type))

@njit(cache=True)
def conds_to_lit_sets(self):
    ''' Convert a Conditions object to a list of dictionaries (one for each
        conjunct) that map the unique tuple (negated, op_addr) for each Literal 
        to a list of literals that have the same unique tuple. This reorganization
        ensures that cases like antiunify( (a != b), (x != y) & (y != z) ) try
        remappings where like-terms get chances to be remapped.'''
    lit_sets = List()
    for conjunct in self.dnf:
        d = Dict.empty(lit_unq_tup_type, lit_list)#Dict.empty(i8, var_set_list_type)
        for lit in conjunct:
            unq_tup = literal_get_unique_tuple(lit)
            # print(lit, ":", unq_tup)

            if(unq_tup not in d):
                l = d[unq_tup] = List.empty_list(LiteralType)
            else:
                l = d[unq_tup]

            base_var_ptrs = np.empty(len(lit.op.base_vars), dtype=np.int64)
            for i, base_var in enumerate(lit.op.base_vars):
                base_var_ptrs[i] = base_var.base_ptr

            l.append(lit)

        # print(d)
        lit_sets.append(d)
    # print(lit_sets)
    return lit_sets

@njit(cache=True)
def count_literals(self):
    c = 0
    for conjunct in self.dnf:
        for lit in conjunct:
            c += 1
    return c

@njit(cache=True)
def intersect_keys(a, b):
    l = List()
    for k in a:
        # print(k, k in b)
        if(k in b): l.append(k)
    return l

u2_arr = u2[::1]
@njit(cache=True)
def lit_set_to_ind_sets(lit_set, base_ptrs_to_inds):
    ''' Takes a list of literals and a mapping from base_var_ptrs to indicies
        and outputs a list of arrays of indicies'''
    l = List.empty_list(u2_arr)
    for lit in lit_set:
        base_set = lit.var_base_ptrs
        base_inds = np.empty(len(base_set), dtype=np.uint16)
        for i, v in enumerate(base_set):
            base_inds[i] = base_ptrs_to_inds[v]
        l.append(base_inds)
    return l

@njit(cache=True)
def remap_is_valid(remap, poss_remaps):
    for i, j in enumerate(remap):
        if(j > poss_remaps.shape[1]): return False
        poss_remaps[i, j]

@njit(cache=True, locals={"remaps":i2[:,::1]})
def get_possible_remap_inds(a_ind_set, b_ind_set, n_a, n_b, a_fixed_inds):
    ''' Given arrays of var indicies corresponding to a common type of literal (e.g. (x<y) & (a<b))
        between Conditions A and B. Generate all possible remappings of variables in 
        A to variables in B. Each remapping is represented by an array of indices 
        of variables in B. For example if A has vars {a,b,c} and B has vars {x,y,z}
        then the remapping [a,b,c] -> [y,z,x] is represented as [1,2,0]. Unresolved
        vars are represented as -1. For example [a,b,-] -> [y,x,-] is represented as [1,0,-1]

        a_ind_set/b_ind_set : arrays of single or paired indicies associated with each literal
        n_a/n_b: the number of variables in A and B respectively
        a_fixed_inds : for each var in A var indicies in B that they are certain to map to.
    '''

    # Produce a boolean matrix identifying feasible remapping pairs
    poss_remap_matrix = np.zeros((n_a,n_b), dtype=np.uint8)
    for a_base_inds in a_ind_set:
        for b_base_inds in b_ind_set:
            if(len(a_base_inds) == len(b_base_inds)):
                for ind_a, ind_b in zip(a_base_inds, b_base_inds):
                    # print(a_fixed_inds, ind_a)
                    # if(a_fixed_inds[ind_a] == -1 or a_fixed_inds[ind_a] == ind_b):
                    poss_remap_matrix[ind_a, ind_b] = True

    # The marginal sums, i.e. number of remappings of each varibles in A and B, respectively
    num_remaps_per_a = poss_remap_matrix.sum(axis=1)
    # num_remaps_per_b = poss_remap_matrix.sum(axis=0)

    n_possibilities = 1
    for i in range(len(num_remaps_per_a)):
        if(a_fixed_inds[i] != -1): 
            num_remaps_per_a[i] = 1
        n_remaps = num_remaps_per_a[i]
        if(n_remaps != 0): n_possibilities *= n_remaps

    
    
    # Find the first variable that is remappable 
    first_i = 0
    for i,n in enumerate(num_remaps_per_a): 
        if(n > 0): first_i = i; break;

    # Use the poss_remap_matrix to fill in a 2d-array of remap options (-1 padded) 
    remap_options = -np.ones((n_a,n_b), dtype=np.int16)
    for i in range(n_a):
        if(a_fixed_inds[i] != -1):
            remap_options[i,0] = a_fixed_inds[i]
        else:
            k = 0
            for j in range(n_b):
                if(poss_remap_matrix[i,j]):
                    remap_options[i,k] = j
                    k += 1

    # Build the set of possible remaps by taking the cartsian product of remap_options
    remaps = np.empty((n_possibilities, n_a), dtype=np.int16)
    
    remap_inds = remap_options[:,0].copy()

    j_iters = np.zeros(n_a, dtype=np.int16)
    iter_i = first_i; iter_j = j_iters[first_i] = -1; done = False;
    c = 0
    while(not done):  
        # Increment to the next possibility for the first variable 
        iter_j = j_iters[iter_i] = j_iters[iter_i] + 1      

        # If this overflows find the next variable that can be incremented without overflowing
        while(iter_j >= num_remaps_per_a[iter_i]):
            # When the iter overflows the set of possibilities reset it to the beginning
            remap_inds[iter_i] = remap_options[iter_i,0]
            j_iters[iter_i] = 0

            # Move on to the next set and iterate it, if that would 
            #  go beyond the last var in A then we're done.
            iter_i += 1
            if(iter_i >= n_a):
                done = True; break; 
            iter_j = j_iters[iter_i] = j_iters[iter_i] + 1      
        if(done): break

        remap_inds[iter_i] = remap_options[iter_i,iter_j]
        iter_i = first_i

        remaps[c, :] = remap_inds
        c += 1
    return remaps


@njit(cache=True)
def get_matched_mask(a_ind_set, b_ind_set, remap):
    a_inds_remapped = np.zeros(len(a_ind_set[0]),dtype=np.int16)
    matched_As = np.zeros(len(a_ind_set),dtype=np.uint8)
    matched_Bs = np.zeros(len(b_ind_set),dtype=np.uint8)
    for i, a_base_inds in enumerate(a_ind_set):
        # Apply this mapping for A -> B 
        for ix, ind in enumerate(a_base_inds):
           a_inds_remapped[ix] = remap[a_base_inds[ix]]

        # Greedily assign literals in remapped A to literals in B
        for j, b_base_inds in enumerate(b_ind_set):
            if(not matched_Bs[j] and np.array_equal(a_inds_remapped, b_base_inds)):
                matched_As[i] = 1
                matched_Bs[j] = 1
                break
    return matched_As

@njit(cache=True)
def try_merge_remaps(ref_remap, remap):
    merged_remap = ref_remap.copy()
    status = 0 # 0 = same, 1 = new, 2 = not same
    for k, (ref_ind, ind) in enumerate(zip(ref_remap, remap)):
        if(ind == -1):
            pass
        elif(ref_ind == -1 and ind != -1):
            status = 1
            merged_remap[k] = ind
        elif(ref_ind != ind):
            status = 2
            break
    return status, merged_remap


i2_arr = i2[::1]
f8_i2_arr_tup = Tuple((f8,i2[::1]))

@njit(cache=True)
def score_remaps(lit_set_a, lit_set_b, bpti_a, bpti_b, a_fixed_inds, 
     op_key_intersection=None, max_remaps=15):
    # print(lit_set_a, lit_set_b)
    if(op_key_intersection is None):
        op_key_intersection = intersect_keys(lit_set_a, lit_set_b)

    # print("Q")

    scored_remaps = List.empty_list(f8_i2_arr_tup)
    # For every unique type of literal (e.g. (x<y) & (a<b))
    for k in op_key_intersection:
        # print("<<", k)
        # print(lit_set_a[k])
        # print("R", k)

        a_ind_set = lit_set_to_ind_sets(lit_set_a[k], bpti_a)
        b_ind_set = lit_set_to_ind_sets(lit_set_b[k], bpti_b)

        # print(a_ind_set)
        remaps = get_possible_remap_inds(a_ind_set, b_ind_set, len(bpti_a), len(bpti_b), a_fixed_inds)
        scores = np.empty(len(remaps),dtype=np.float64)
        for c, remap in enumerate(remaps):
            # print("remap: ", remap)
            matched_As = get_matched_mask(a_ind_set,b_ind_set,remap)
            scores[c] = np.sum(matched_As)

        # print(len(remaps), remaps)

        # print("T")
        # Order by score, drop remaps that have a score of zero 
        order = np.argsort(scores)
        first_nonzero = 0
        for ind in order:
            if(scores[ind] > 0): break
            first_nonzero +=1
        # print("first_nonzero", first_nonzero)
        order = order[first_nonzero:][::-1][:min(max_remaps, len(order))]
        # order = order[first_nonzero:][::-1]


        # if(len(order) > 100):
        #     raise ValueError()
        #     print(len(order))
        # print(len(order))
        # print(order)

        # print("U")

        og_L = len(scored_remaps)
        ref_was_merged = np.zeros(og_L, dtype=np.uint8)
        n_merged = 0
        for i, ind in enumerate(order):
            
            score, remap = scores[ind], remaps[ind]
            was_merged = False
            # print("i", i, remap)
            for j in range(len(scored_remaps)):
                
                if(not ref_was_merged[j]):
                    ref_score, ref_remap = scored_remaps[j]
                    # print("j", j, ref_remap)
                    merged_remap = ref_remap.copy()

                    # status : 0 = same, 1 = new, 2 = not same
                    status, merged_remap = try_merge_remaps(ref_remap, remap)
                    
                    if(status != 2):
                        if(status == 1):
                            # If they aren't quite same append the merge
                            scored_remaps.append((ref_score+score, merged_remap))
                        else:
                            # When they are the same replace the old one
                            scored_remaps[j] = (ref_score+score, merged_remap)
                            # Mask out j to ensure we don't merge two remaps from this 
                            #  literal set into a ref_remap from a previous one. We'll 
                            #  only mask the first possibility which has the largest score. 
                            ref_was_merged[j] = 1
                        was_merged = True

                        
                    
                    # print("<<", status, ref_remap, remap, merged_remap)
            if(not was_merged):
                scored_remaps.append((score,remap))

        # print("V")
        # Sort the scored_remap list
        scores = np.empty(len(scored_remaps),dtype=np.float64)
        for i,(s,_) in enumerate(scored_remaps): scores[i] = s
        order = np.argsort(scores)[::-1]
        scored_remaps = List([scored_remaps[ind] for ind in order])

        # print("X")
    # print(scored_remaps)

    return scored_remaps

        # best_remap = scored_remaps[0]
        # scored_remaps =  sorted(scored_remaps)

from numba.cpython.hashing import _PyHASH_XXPRIME_5
from cre.hashing import accum_item_hash


FrozenArr, FrozenArrTypei8 = define_structref("FrozenArr", [("arr" , i8[:]),])
FrozenArrTypei2 = type(FrozenArrTypei8)([("arr" , i2[::1]),])

@overload(hash)
@overload_method(type(FrozenArrTypei8), '__hash__')
def _impl_hash_FrozenArr(x):
    if("FrozenArr" in x._typename):
        def impl(x):
            acc = _PyHASH_XXPRIME_5
            for x_i in x.arr:
                #TODO: in the future if we implement this for float arrs
                #      then we need to cast the raw bits of x_i to ints
                acc = accum_item_hash(acc, x_i)

            return acc
        return impl

@overload(operator.eq)
def _impl_eq_FrozenArr(a, b):
    if("FrozenArr" in a._typename and "FrozenArr" in b._typename, ):
        def impl(a, b):
            return np.array_equal(a.arr, b.arr)
        return impl


f8_2darr_type = f8[:,::1]
i8_arr = i8[::1]

@njit(cache=True)
def _buid_score_aligment_matrices(ls_as, ls_bs, bpti_a, bpti_b, a_fixed_inds):
    '''For each unique remap make a matrix that holds the remap score between 
       conjunct_i and conjunct_j for all possible alignments of the conjuncts 
       in A and conjuncts in B. Return a list of remap matrix pairs.'''
    score_aligment_matrices = Dict.empty(FrozenArrTypei2, f8_2darr_type)
    for i, ls_a in enumerate(ls_as): 
        for j, ls_b in enumerate(ls_bs): 
            scored_remaps = score_remaps(ls_a, ls_b, bpti_a, bpti_b, a_fixed_inds)
            for score, remap in scored_remaps:
                f_remap = FrozenArr(remap)
                if(f_remap not in score_aligment_matrices):
                    rank = np.sum(remap == -1)
                    score_matrix =  np.zeros((len(ls_as), len(ls_bs)),dtype=np.float64)
                    score_aligment_matrices[f_remap] = score_matrix

                score_aligment_matrices[f_remap][i,j] = score
    return List(score_aligment_matrices.items())

@njit(cache=True)
def _max_remap_score_for_alignments(score_matrix):
    num_conj = score_matrix.shape[0]
    orders = List.empty_list(i8_arr)
    col_assign_per_row = -np.ones(num_conj,dtype=np.int64)
    n_assigned = 0
    row_maxes = np.zeros(num_conj,dtype=np.float64)
    for i in range(num_conj):
        orders_i = np.argsort(-score_matrix[i])
        orders.append(orders_i)
        row_maxes[i] = score_matrix[i][orders_i[0]]

    order_of_rows = np.argsort(-row_maxes)
    score = 0
    while(len(order_of_rows) > 0):
        row_ind = order_of_rows[0]

        c = 0 
        col_ind = orders[row_ind][c]
        while(col_ind in col_assign_per_row):
            col_ind = orders[row_ind][c]; c += 1
            if(c >= len(orders[row_ind])): break

        col_assign_per_row[row_ind] = col_ind
        score += score_matrix[row_ind,col_ind]
        order_of_rows = order_of_rows[1:]
    return score, col_assign_per_row

@njit(cache=True)
def _conj_from_litset_and_remap(ls_a,ls_b, remap, keys, bpti_a, bpti_b):
    conj = List.empty_list(LiteralType)
    for unq_key in keys:
        lit_set_a = ls_a[unq_key]
        lit_set_b = ls_b[unq_key]
        
        ind_set_a = lit_set_to_ind_sets(lit_set_a, bpti_a)
        ind_set_b = lit_set_to_ind_sets(lit_set_b, bpti_b)
        matched_mask = get_matched_mask(ind_set_a,ind_set_b,remap)

        # print(lit_set_a, lit_set_b, matched_mask)
        for keep_it, lit in zip(matched_mask, lit_set_a):
            if(keep_it):
                new_lit = literal_ctor(lit.op)
                new_lit.negated = lit.negated
                conj.append(new_lit)
    return conj

@njit(cache=True,locals={"i" : u2})
def make_base_ptrs_to_inds(self):
    ''' From a collection of vars a mapping between their base ptrs to unique indicies'''
    base_ptrs_to_inds = Dict.empty(i8,u2)
    i = u2(0)
    for v in self.vars:
        base_ptrs_to_inds[v.base_ptr] = i; i += 1;
    return base_ptrs_to_inds


@njit(cache=True)
def get_fixed_inds(c_a, c_b, map_same_var, map_same_alias):
    a_fixed_inds = -np.ones(len(c_a.vars), dtype=np.int16)

    if(map_same_var):
        for k, v_b in enumerate(c_b.vars):
            if(v_b.base_ptr in c_a.base_var_map):
                a_fixed_inds[c_a.base_var_map[v_b.base_ptr]] = k
    elif(map_same_alias):
        for i, v_a in enumerate(c_a.vars):
            for j, v_b in enumerate(c_b.vars):
                if(v_a.alias == v_b.alias):
                    a_fixed_inds[i] = j
    
    return a_fixed_inds



@njit(Tuple((ConditionsType, f8))(ConditionsType, ConditionsType, boolean, boolean), cache=True)
# @njit(cache=True)
def _conds_antiunify(c_a, c_b, fix_same_var, fix_same_alias):
    ls_as = conds_to_lit_sets(c_a)
    ls_bs = conds_to_lit_sets(c_b)

    bpti_a = make_base_ptrs_to_inds(c_a)
    bpti_b = make_base_ptrs_to_inds(c_b)

    remap_size = len(bpti_a)
    num_conj = len(ls_as)
    best_score = -np.inf
    best_remap = np.arange(remap_size, dtype=np.int16)


    a_fixed_inds = get_fixed_inds(c_a, c_b, fix_same_var, fix_same_alias)
    # a_fixed_inds = np.arange(len(c_a.base_var_map),dtype=np.int16)
    # a_fixed_inds[-2] = -1
    # a_fixed_inds = -np.ones(len(c_a.base_var_map), dtype=np.int16)
    
    # Case 1: c_a and c_b are both single conjunctions like (lit1 & lit2 & lit3)
    if(len(ls_as) == 1 and len(ls_bs) == 1):
        op_key_intersection = intersect_keys(ls_as[0], ls_bs[0])
        # print(op_key_intersection)
        scored_remaps = score_remaps(ls_as[0], ls_bs[0], bpti_a, bpti_b, a_fixed_inds,
                                    op_key_intersection=op_key_intersection)
        best_score, best_remap = scored_remaps[0]


        conj = _conj_from_litset_and_remap(ls_as[0], ls_bs[0], 
                 best_remap, op_key_intersection, bpti_a, bpti_b)

        dnf = List([conj])
        conds = _conditions_ctor_dnf(dnf)

    # Case 2: c_a or c_b have disjunction like ((lit1 & lit2 & lit3) | (lit4 & lit5))
    else:
        score_aligment_matrices = _buid_score_aligment_matrices(ls_as, ls_bs, bpti_a, bpti_b, a_fixed_inds)
        
        # Find the upperbounds on each remap
        score_upperbounds = np.zeros(len(score_aligment_matrices),dtype=np.float64)
        for i, (f_remap, score_matrix) in enumerate(score_aligment_matrices):
            for row in score_matrix:
                score_upperbounds[i] += np.max(row)

        # Look for the best remap trying remaps with large upperbounds first
        descending_upperbound_order = np.argsort(-score_upperbounds)
        best_score = 0
        best_alignment = np.empty(0,dtype=np.int64)
        for i in descending_upperbound_order:
            # Stop looking after couldn't possibly beat the best so far.
            upper_bound_score = score_upperbounds[i]
            if(upper_bound_score < best_score):
                break

            # Find the score for the best conjuct alignment for this remap 
            f_remap, score_matrix = score_aligment_matrices[i]
            score, alignment = _max_remap_score_for_alignments(score_matrix)
            if(score > best_score):
                best_remap = f_remap.arr
                best_score = score
                best_alignment = alignment

        dnf = List()
        for i, ls_a in enumerate(ls_as):
            ls_b = ls_bs[alignment[i]]

            op_key_intersection = intersect_keys(ls_a, ls_b)
            conj = _conj_from_litset_and_remap(ls_a, ls_b,
                     best_remap, op_key_intersection, bpti_a, bpti_b)
            dnf.append(conj)

        
        conds = _conditions_ctor_dnf(dnf)
    

    # Make sure base Vars are ordered like c_a
    unordered_base_vars = conds.base_var_map
    base_var_map = Dict.empty(i8,i8)
    c = 0 
    for base_ptr in bpti_a:
        if(base_ptr in unordered_base_vars):
            base_var_map[base_ptr] = c
            c += 1
    conds = _conditions_ctor_base_var_map(base_var_map, dnf)

    return conds, best_score



@generated_jit(cache=True)
@overload_method(ConditionsTypeClass, 'antiunify')
def conds_antiunify(c_a, c_b, return_score=False,
            normalize='left', fix_same_var=False, fix_same_alias=False):
    SentryLiteralArgs(['return_score','normalize']).for_function(
                    conds_antiunify).bind(c_a, c_b, return_score, normalize)
    normalize = normalize.literal_value
    if(return_score.literal_value):
        def impl(c_a, c_b, return_score=False,
                normalize='left', fix_same_var=False, fix_same_alias=False):
            c, score = _conds_antiunify(c_a, c_b, fix_same_var, fix_same_alias)
            if(normalize == 'left'):
                n_literals_a = count_literals(c_a)
                return c, score / n_literals_a
            elif(normalize == 'right'):
                n_literals_b = count_literals(c_b)
                return c, score / n_literals_b
            elif(normalize == 'max'):
                n_literals_a = count_literals(c_a)
                n_literals_b = count_literals(c_b)
                return c, score / max(n_literals_a, n_literals_b)
            else:
                return c, score
    else:
        def impl(c_a, c_b, return_score=False,
                normalize='left', fix_same_var=False, fix_same_alias=False):
            c, score = _conds_antiunify(c_a, c_b, fix_same_var, fix_same_alias)
            return c
    return impl

        

# -----------------------------------------------------------------------
# : Conditions.from_facts

def _add_adjacent(src_fact, _, fact, src_attr_var, conds,
         fact_ptr_map, attr_flags, add_back_relation=False, var_prefix=None):
    src_ptr = src_fact.get_ptr()
    fact_ptr = fact.get_ptr()

    
    # Make sure that the adjacent fact is in 'fact_ptr_map'
    if(fact_ptr not in fact_ptr_map):
        fact_var = Var(fact._fact_type, f"{var_prefix}{len(conds)}")
        fact_ptr_map[fact_ptr] = fact_var

    fact_var = fact_ptr_map[fact_ptr]
    src_fact_var = fact_ptr_map[src_ptr]
    _conds = conds.get(fact_ptr, [])
    
    if(add_back_relation):
        for attr, config in fact._fact_type.filter_spec(*attr_flags).items():
            attr_val = getattr(fact, attr)
            if(not isinstance(attr_val, FactProxy) or
                attr_val.get_ptr() != src_ptr):
                continue
            attr_var = getattr(fact_var, attr)
            _conds.append(attr_var==src_fact_var)

    # Add a condition from val to 
    _conds.append(src_attr_var==fact_var)            
    conds[fact_ptr] = _conds


from itertools import chain

def conditions_from_facts(facts, _vars=None, add_neighbors=True,
     add_neighbor_holes=False, neighbor_back_relation=False, neighbor_req_n_adj=1, 
     alpha_flags=('visible',), parent_flags=('parent',), beta_flags=('relational',)):
    
    
    if(_vars is None):
        _vars = [Var(x._fact_type, f'A{i}') for i, x in enumerate(facts)]

    fact_ptr_map = {fact.get_ptr() : var for fact, var in zip(facts, _vars)}
    inp_fact_ptrs = set(fact_ptr_map.keys())
    
    # Make flag sets based on 
    beta_not_parent_flags = [*beta_flags,*[f"~{f}" for f in parent_flags]]
    alpha_candidate_flags = alpha_flags
    if(add_neighbor_holes):
        alpha_candidate_flags = [*alpha_flags, *parent_flags, *beta_flags]
    
    # Make Alphas (i.e. nvar = 1)
    cond_set = []
    nbr_conds = {}
    parent_conds = {}
    beta_conds = {}
    for fact, var in zip(facts,_vars):
        for attr, config in fact._fact_type.filter_spec(*alpha_candidate_flags).items():
            val = getattr(fact, attr)
            if(isinstance(val, FactProxy)): continue
            attr_var = getattr(var, attr)
            cond_set.append(attr_var==val)

        # TODO: Make parents 
        
        nxt_parents = [fact]
        while(len(nxt_parents) > 0):
            for nxt_parent in nxt_parents:
                spec = nxt_parent._fact_type.filter_spec(*parent_flags)
                for attr, config in spec.items():
                    pass
            nxt_parents = []

        # Make Betas (i.e. n_var=2) + Neighbors
        
        for attr, config in fact._fact_type.filter_spec(*beta_not_parent_flags).items():
            attr_val =  getattr(fact, attr)
            if(not isinstance(attr_val, FactProxy)): continue
            attr_var = getattr(var, attr)

            # Add a beta between the input facts.
            if(attr_val.get_ptr() in inp_fact_ptrs):
                _add_adjacent(fact, var, attr_val, attr_var, beta_conds,
                        fact_ptr_map, beta_not_parent_flags, var_prefix=None)

            # Add a beta between an input fact and a non-input neighbor.
            elif(add_neighbors):
                _add_adjacent(fact, var, attr_val, attr_var, nbr_conds,
                 fact_ptr_map, beta_not_parent_flags, add_back_relation=neighbor_back_relation,
                var_prefix="Nbr")

    if(neighbor_req_n_adj > 1):
        for ptr, lst in list(nbr_conds.items()): 
            if len(lst) < neighbor_req_n_adj:
                del nbr_conds[ptr]
                del fact_ptr_map[ptr]
        # TODO: Rename aliases of remaining vars to 0,1,2...
        # for i, ptr in enumerate(nbr_conds): 
        #     v = fact_ptr_map[ptr]
        #     v.alias = f"Nbr{i}"
        #     print(v.alias)

    cond_set = [*cond_set, 
                *chain(*parent_conds.values()),
                *chain(*beta_conds.values()),    
                *chain(*nbr_conds.values())
                ]

    _vars = list({v.get_ptr():v for v in fact_ptr_map.values()}.values())
    # print(_vars)   
    conds = _vars[0]
    for i in range(1, len(_vars)):
        conds = conds & _vars[i]

    for c in cond_set:
        conds = conds & c

    # print(conds)
    return conds




        # print(orders)

        #     order = np.argsort(-score_matrix[i])


        # argsort(-)




    # print("----")
    # MERGE CODE Probably not needed
    # for i, (remap_i, a_sm_i) in enumerate(score_aligment_matrices.items()):
    #     sm_i = score_matricies[i]
    #     for j, (remap_j, a_sm_j) in enumerate(score_aligment_matrices.items()):
    #         if(i != j):
    #             sm_j = score_matricies[j]
    #             can_merge = True
    #             for k in range(remap_size):
    #                 m_i, m_j = remap_i.arr[k], remap_j.arr[k]
    #                 if(m_i != m_j and m_i != -1 and m_j != -1):
    #                     can_merge = False
    #                     break
    #             print(remap_i.arr, remap_j.arr, can_merge)

    #             if(can_merge):
    #                 a_sm_i[:] = a_sm_i + sm_j
    #                 a_sm_j[:] = a_sm_j + sm_i


    

    # order = np.argsort(-remap_ranks)
    # print(order)

    # # for ind in order:


    # for f_remap, score_matrix in score_aligment_matrices.items():
    #     print(f_remap.arr)
    #     print(score_matrix)


                # print(score, remap)

    '''Make an num_cunj_A by num_cunj_B sized array for each 
        frozen remap array by mapping frozen arr -> 2d arr.
        initialize on zero and fill in as it goes. 
        Next try and merge like mappings.
        For each of those merged 2d arrays find the best alignment.
        Argsort each row. Argsort the maxes of the sorted arrays.
        Assign the alignment based on the maxes. If the aligment
        element is filled, then the filled assignment must have a higher
        score. 



    '''

    # print("scored_remaps")
    # print(scored_remaps)
    # print("scored_remaps", scored_remaps)


















        
        # print("scores", scores)
        # print("remaps", remaps)


        # print("argsort scores",)










        
            # while(remap_options[iter_i,iter_j] == -1):
            #     pass

# 0 0 1
# 1 0 1
# 0 1 1
# 1 1 1
# 0 2 1
#

            






        
        # initial_remap = np.zeros(len(bpti_a), dtype=np.int16)

        # first_i = -1
        # for i in range(len(num_remaps_per_a)):
        #     if(first_i == -1 and num_remaps_per_a[i] > 0): first_i = i
        #     for j in range(len(bpti_a)):
        #         if(poss_remap_matrix[i,j]):
        #             initial_remap[i] = j; break;

        # for i in range(len(bpti_a)):
        #     if(num_remaps_per_b[i] <= 0): initial_remap[i] = -1

        # remap = initial_remap.copy()
        # iter_i = first_i
        # iter_j = remap[iter_i]

        # print("first:", first_i)
        # print("initial remap:", remap)



        
        # # present_inds = 
        # # next valid remap
        # done = False
        # while(not done):
        #     iter_j = remap[iter_i]
        #     if(poss_remap_matrix[iter_i, iter_j]):
        #         print(remap, True)            

        #     iter_j = remap[iter_i] = remap[iter_i] + 1
                
        #     # When iter_j exceeds the last var in B then we have exhausted the mapping
        #     #  possibilities for the first var in A. Find the next var in A to try the
        #     #  next possibility for.
        #     while(iter_j >= len(bpti_b)):
        #         remap[iter_i] = initial_remap[iter_i]

        #         iter_i += 1
        #         # Skip variables from A that aren't present in this literal set
        #         while(num_remaps_per_a[iter_i] <= 0): iter_i += 1

        #         # iter_i exceeds the last var in A then we're done
        #         if(iter_i >= len(bpti_a)): 
        #             done = True; break;


        #         iter_j = remap[iter_i] = remap[iter_i] + 1            
        #         while(poss_remap_matrix[iter_i, iter_j] == 0):
        #             iter_j = remap[iter_i] = remap[iter_i] + 1
                
        #     # 
        #     iter_i = first_i
            

        # # print("----")
                

        

        # for i in range(len(bpti_a)):
        #     for j in range(len(bpti_b)):
        #         if(poss_remap_matrix[i,j]):


            
                # print(a_base_inds, b_base_inds)

        # print(poss_remap_matrix)







        # possible_remaps = np.array()

        # print(a_base_sets,b_base_sets)






    # for k in op_intersection:
    #     a_base_sets = op_set_a[k]
    #     b_base_sets = op_set_b[k]

    #     for b_base_set in b_base_sets:
    #         if(remapping):
    #             new_b_base_set = np.empty(len(b_base_set),dtype=np.int64)
    #             for i, ptr in b_base_set:
    #                 new_b_base_set[i] = remapping[ptr]
    #             b_base_set = new_b_base_set
    #         for a_base_set in a_base_sets:
    #             if(np.array_equal(a_base_set, b_base_set))



    
#     return d



# def get_op_set_re():




# @njit(cache=True)
# def best_intersection(self, other):
#     return None
    

#     for conjunct_a in self.dnf:
#         for conjunct_b in other.dnf:
#             op_set_a = Dict.empty(i8,u1)
#             op_set_b = Dict.empty(i8,u1)


# Once the op intersection has been found then we can speed up the combinatorial
# checking of var agreement using np.arrays
# let x,y be (n*2) arrays M be a replacement matrix from Vy -> Vx that represents
# all of the ways of substituting a var in y for one in x
# i.e.
'''
x,y,z ->
[a,b,c,Ø] x [a,b,c,Ø] x [a,b,c,Ø]
[
 [a,b,c]
 [c,a,b]
 [b,c,a]
 []
]

Very few of these are actually worth checking, for instance (1)
(a < b) & (b < c) 
(x < y) & (y < z)

x -> [a,b,Ø], y -> [b,Ø], z -> [c,b,Ø]

there is no point in replacing x -> c for instance since that replacement isn't
consistent with any of the present literals

so the posibilities are 
x,y,z -> [a,b,c], [a,b,b], [b,b,c], [b,b,b], 
        ... or some others where we have to drop literals because of Ø

It probably doesn't make sense to reject repeat variables outright
Formally the mapping doesn't necessarily need to be injective

for instance (2)
(a < b) & (b < c) 
(x < y) & (q < z)

The best antiunification is 

x,y,q,z -> [a,b,b,c]

On this note it probably makes sense to choose the Conditions object
with the fewest variables to be the one that defines the set of variables that is kept,
or if this inconsistency is annoying then we could do this and then map back to the left one


This can probably actually be reformulated as a SAT problem, although
 it's not clear that this would be more efficient or easier.
 The plotkin idea seems like this, just combine each candidate pair to make a new literal like
 Xaz | Xay, etc.

One question is can we resolve variables greedily
for instance in (1) we might be tempted to resolve y -> b which
would mean z -> c satisfies the first literal and x -> a satisfies the second

One way of thinking about this is that the variable y spans 2 literals, and has one
option, otherwise we choose Ø which drops both literals. Thus if we can establish
that resolving the maximium spanning variable first can't screw us over then we can
make a greedy algorithm gaurenteed to find the best lgg. 


Let's try another example 

(x < y) & (y < z) & (y < r)
(a < b) & (b < c) 


x -> [a,c], y -> [b], z-> [c,b], r -> [c,b]

Here there are two best mappings x,y,z,r -> [a,b,c,Ø], [a,b,Ø,c]

I wonder if we can reformulate this as a sort of graph coloring problem in which case,
it is probably np-hard. For instance the set with more variables constitutes the nodes
and the set with fewer is the colors and the intersecting set of literals are the edges...
Not sure this totally works, since we're allowed to get rid of edges.

So what's a good way to try all the mappings?

Could probably just encode them all going off of the instance above

ref [[a,b],[b,c]]

ops [[a,b],[b,c]] : x,y,z,r -> [a,b,c,Ø]
ops [[a,b],[b,b]] : x,y,z,r -> [a,b,b,Ø]
ops [[b,b],[b,c]] : x,y,z,r -> [b,b,c,Ø]
ops [[b,b],[b,b]] : x,y,z,r -> [b,b,b,Ø]
ops [[a,b],[b,c]] : x,y,z,r -> [a,b,Ø,c]
ops [[a,b],[b,b]] : x,y,z,r -> [a,b,Ø,b]
ops [[b,b],[b,c]] : x,y,z,r -> [b,b,Ø,c]
ops [[b,b],[b,b]] : x,y,z,r -> [b,b,Ø,b]

so clearly the first one is best but if there were other literals added in
then we might find for instance that c -> b was actually a good choice,
in which case we would have to drop a literal like (b < c), but we might keep others,
in the process.

For instance: 

(x < y) & (y < z) & (z != x) & (y != 0) 
(a < b) & (b < c) & (b != a) & (c != 0)

x -> [a,c], y -> [b,c], z -> [c,b]

pre [[x,y],[y,z],[z,x],[y,-]]
ref [[a,b],[b,c],[b,a],[c,-]]

ops [[a,c],[c,c],[c,a],[c,-]] : x,y,z -> [a,c,c] 1
ops [[a,c],[c,b],[b,a],[c,-]] : x,y,z -> [a,c,b] 2
ops [[b,c],[c,c],[b,c],[c,-]] : x,y,z -> [b,c,c] 1
ops [[b,c],[c,b],[b,b],[c,-]] : x,y,z -> [b,c,b] 1

ops [[a,b],[b,c],[c,a],[b,-]] : x,y,z -> [a,b,c] 2
ops [[a,b],[b,b],[b,a],[b,-]] : x,y,z -> [a,b,b] 2
ops [[b,b],[b,c],[b,c],[c,-]] : x,y,z -> [b,b,c] 2
ops [[b,b],[b,b],[b,b],[b,-]] : x,y,z -> [b,b,b] 0

Here there are several best options 


In code we can represent this as 

ptrs -> inds dicts i.e.
{x,y,z} -> {0,1,2}
{a,b,c} -> {3,4,5}


arr = of_size(prod(n_possibilities), min(n_varl, n_varr)) 
for left_vars ..
    for remap_poss ...
        arr[?:?:?] = remap_poss

remapped_pre = empty
scores_arr = of_size(prod(n_possibilities))
for k,remap in enumerate(arr):
    for i,v in range(len(pre)):
        for j,v in range(d):
            remapped_pre[i,j] = remap[pre[i,j]]

    scores_arr[k] = sum(all(remapped_pre == ref,ind=-1)*weights)


    















'''
        

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
