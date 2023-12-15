import operator
import numpy as np
from numba import types, njit, f8, f4, i8, u8, i4, u1, i8, i2, u2, literally, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple, UniTuple, optional, boolean
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import lower_cast, overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic, SentryLiteralArgs
from numba.core.typing.templates import AttributeTemplate
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template, StructRefType
from cre.fact import define_fact, BaseFact, cast_fact, FactProxy
from cre.utils import cast, _incref_structref, decode_idrec, lower_getattr, lower_setattr, lower_getattr,  _ptr_from_struct_incref, _decref_ptr
from cre.utils import assign_to_alias_in_parent_frame, meminfo_type, decref_ptrs
from cre.vector import VectorType

# from cre.op import CREFuncType, op_str, Op
from cre.func import CREFuncType, CREFunc, cre_func_unique_string

from cre.obj import CREObjType, CREObjTypeClass
from cre.core import T_ID_VAR, T_ID_FUNC, T_ID_CONDITIONS, T_ID_LITERAL, register_global_default
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models
from cre.var import *
from cre.why_not import new_why_not, why_not_type, WN_NOT_MAP_LITERAL, WN_NOT_MAP_VAR

from operator import itemgetter
from copy import copy
import inspect
import sys

#### Literal ####

literal_fields_dict = {
    # "str_val" : unicode_type,
    **cre_obj_field_dict,
    "op" : CREFuncType,
    "base_var_ptrs" : i8[:],#UniTuple(i8,2),
    "var_inds" : i8[:],
    # "cre_ms_ptr" : i8,
    # A weight used for scoring matches and structure remapping  
    "weight" : f4,

    # Whether or not the literal is negated
    "negated" : u1,
    "is_alpha" : u1,

    # The index of the literal in its parent condition
    
    # "link_data" : LiteralLinkDataType,
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
    def weight(self):
        return literal_get_weight(self)

    def set_weight(self, weight):
        literal_set_weight(self, weight)

    @property
    def op(self):
        return literal_get_pred_node(self)

py_operator_chars = ("+", "-", "*", "/", "%",
                     "=", "~", "<", ">", "!",
                     "&", "|", "@", "^")
@njit(cache=True)
def has_operator_char(s):
    for py_operator in py_operator_chars:
        if(py_operator in s):
            return True
    return False

@njit(cache=True)
def literal_str(self):
    s = str(self.op)
    if(self.negated):
        if(has_operator_char(s)):
            s = f"({s})"
        return f"~{s}"
    return s    

@njit(cache=True)
def literal_get_op(self):
    return self.op

@njit(cache=True)
def literal_get_weight(self):
    return self.weight

@njit(cache=True)
def literal_set_weight(self,weight):
    self.weight = weight

@njit(cache=True)
def literal_from_ptr(ptr):
    return cast(ptr, LiteralType)

define_boxing(LiteralTypeClass, Literal)
LiteralType = LiteralTypeClass(literal_fields)
register_global_default("Literal", LiteralType)

@njit(LiteralType(CREFuncType),cache=True)
def literal_ctor(op):
    st = new(LiteralType)
    st.idrec = encode_idrec(T_ID_LITERAL, 0, 0)
    st.op = op
    st.base_var_ptrs = op.base_var_ptrs
    st.var_inds = np.arange(op.n_args, dtype=np.int64)
    st.weight = 1.0
    st.negated = 0
    st.is_alpha = u1(len(st.base_var_ptrs) == 1)
    return st


#TODO: STR
@overload_method(LiteralTypeClass, "__str__")
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
    st.base_var_ptrs = self.base_var_ptrs
    st.var_inds = self.var_inds.copy()
    st.weight = self.weight
    st.negated = self.negated
    st.is_alpha = self.is_alpha
    # st.cre_ms_ptr = self.cre_ms_ptr
    # if(self.cre_ms_ptr):
    #     st.link_data = self.link_data
    return st
    
@njit(cache=True)
def literal_not(self):
    n = literal_copy(self)
    n.negated = not n.negated
    return n


literal_unique_tuple_type = Tuple((i8, unicode_type))
# @njit(literal_unique_tuple_type(LiteralType), cache=True)
@njit( cache=True)
def literal_get_unique_tuple(self):
    '''Outputs a tuple that uniquely identifies an instance
         of a literal independant of the base Vars of its underlying op.
    '''
    return (i8(self.negated), cre_func_unique_string(self.op))


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

    # The variables used by the conditions
    'vars': ListType(VarType),

    # 'not_vars' : ListType(VarType),

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


    # Keep around an anonymous reference to the matcher_inst
    #  can be casted to specialize to implementation
    "matcher_inst" : types.optional(StructRefType), # Keep this so we can check for zero
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
        if(isinstance(_vars, (CREFunc, Var, Literal, Conditions))):
            return to_cond(_vars)
        else:
            return _conditions_ctor_var_list(_vars, dnf)

    def __str__(self):
        return conditions_repr(self)
    def __and__(self, other):
        # if(isinstance(other,CREFunc)): other = to_cond(other)
        conds = conditions_and(self, other) 
        auto_alias_unaliased_vars(conds)
        return conds
    def __or__(self, other):
        # if(isinstance(other,CREFunc)): other = to_cond(other)
        conds = conditions_or(self, other)
        auto_alias_unaliased_vars(conds)
        return conds
    def __not__(self):
        return conditions_not(self)
    def __invert__(self):
        return conditions_not(self)

    def get_ptr_matches(self,ms=None):
        from cre.matching import get_ptr_matches
        return get_ptr_matches(self,ms)

    def get_matches(self, ms=None):
        from cre.matching import MatchIterator
        return MatchIterator(ms, self)

    def _match_to_ptrs(self, match):
        ptrs = np.zeros(len(match), dtype=np.int64)
        for i, m in enumerate(match):
            if(m is not None):
                ptrs[i] = m.get_ptr()
        return ptrs    

    def check_match(self, match, ms=None):
        from cre.matching import check_match
        ptrs = self._match_to_ptrs(match)
        return check_match(self, ptrs)

    def set_weight(self, weight):
        conds_set_weight(self, weight)

    def score_match(self, match, ms=None):
        from cre.matching import score_match
        ptrs = self._match_to_ptrs(match)
        return score_match(self, ptrs)

    def why_not_match(self, match, ms=None):
        from cre.matching import score_match, WN_VAR_TYPE, why_not_match
        from cre.var import var_from_ptr
        ptrs = self._match_to_ptrs(match)
        why_not_arr = why_not_match(self, ptrs)
        why_nots = []
        for wn in why_not_arr:
            if(wn['kind'] == WN_VAR_TYPE):
                why_nots.append((var_from_ptr(wn['ptr']), wn))
            else:
                why_nots.append((literal_from_ptr(wn['ptr']), wn))
        return why_nots

    def antiunify(self, other, return_score=False, normalize='left',
         fix_same_var=False, fix_same_alias=False):
        norm_enum = resolve_normalize_enum(normalize)
        if(not isinstance(other, Conditions)):
            other = Conditions(other)
        new_conds, score = conds_antiunify(self, other, norm_enum,
                                    fix_same_var, fix_same_alias) 
        if(return_score):
            return new_conds, score
        else:
            return new_conds

    def structure_map(self, other, return_score=False, normalize='left',
         fix_same_var=False, fix_same_alias=False):
        print(_conds_structure_map(self, other, fix_same_var, fix_same_alias))
    
    @property
    def var_base_types(self):
        if(not hasattr(self,"_var_base_types")):
            context = cre_context()
            var_t_ids = conds_get_var_t_ids(self)
            self._var_base_types = tuple([context.get_type(t_id=t_id) for t_id in var_t_ids])
            # delimited_type_names = conds_get_delimited_type_names(self,";",True).split(";")
            # self._var_base_types = tuple([context.type_registry[x] for x in delimited_type_names])
            
        return self._var_base_types



    # def link(self,ms):
    #     get_linked_conditions_instance(self,ms,copy=False)

    @property
    def signature(self):
        if(not hasattr(self,"_signature")):
            context = cre_context()
            # print(self)
            sig_str = _get_sig_str(self)
            fact_types = sig_str[1:-1].split(",")
            # print(fact_types)
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

    # @property
    # def rete_graph(self):
    #     from cre.matching import conds_get_rete_graph
    #     return conds_get_rete_graph(self)

    def as_dnf_list(self):
        return as_dnf_list(self)

    def as_distr_dnf_list(self):
        return as_distr_dnf_list(self)

    def as_literal(self):
        return conditions_as_literal(self)

    def __str__(self):
        return conditions_repr(self)

    def __repr__(self):
        return conditions_repr_short(self)

    @classmethod
    def from_facts(cls, facts,_vars=None, *args, **kwargs):
        return conditions_from_facts(facts, _vars, *args, **kwargs)

    # def replace(self, d, c, lit):
    #     lit = finesse_to_literal(lit)
    #     return conditions_replace(self, d, c, lit, True)
        
    def replace(self, *args):
        # Replace a literal or var with another
        # Supports argument patterns:
        #  

        if(isinstance(args[0], list)):
            items = args[0]
        else:
            items = [args]

        conds = conditions_copy(self)
        items = [process_remove_item(conds, x) for x in items]
        items = sorted(items, key=edit_list_sortkey, reverse=True)
        # print(items)
        
        for item_args in items:
            if(len(item_args) == 2):
                x = item_args[0]
                lit = finesse_to_literal(item_args[1])
                if(isinstance(x, (Literal, Conditions, CREFunc))):
                    x = finesse_to_literal(x)
                    indicies = conditions_all_indicies_of(conds, x)
                    replacements = List([(d, c, lit) for d, c in indicies])
                    conds = conditions_replace_multi(conds, replacements)
                elif(isinstance(x, Var)):
                    raise NotImplemented()
                    var_ind = conditions_get_var_ind(conds, x)
                    conds = conditions_replace_var(conds, var_ind, item_args[1])
                elif(isinstance(x, (int,np.integer))):
                    raise NotImplemented()
                    conds = conditions_replace_var(conds, x, item_args[1])
            elif(len(item_args) == 3):
                d, c, lit = item_args
                lit = finesse_to_literal(lit)
                conds = conditions_replace(conds, d, c, lit)
            else:
                raise ValueError(f"Too many arguments for remove(): {args}")
        return conds

        # new_conds = conditions_copy(self)
        # for c, d, lit in replacements:
        #     lit = finesse_to_literal(lit)
        #     conditions_replace(new_conds, d, c, lit, False)
        # return new_conds

    def remove(self, *args):
        if(isinstance(args[0], list)):
            items = args[0]
        else:
            items = [args]

        conds = conditions_copy(self)
        items = [process_remove_item(conds, x) for x in items]
        items = sorted(items, key=edit_list_sortkey, reverse=True)
        # print(items)
        
        for item_args in items:
            # if(not isinstance(item_args, (tuple, list))):
            #     item_args = (item_args,)
            # print(item_args)
            if(len(item_args) == 1):
                x = item_args[0]
                # print(x, type(x))
                if(isinstance(x, (Literal, Conditions, CREFunc))):
                    x = finesse_to_literal(x)
                    indicies = conditions_all_indicies_of(conds, x)
                    conds = conditions_remove_multi(conds, indicies)
                elif(isinstance(x, Var)):
                    var_ind = conditions_get_var_ind(conds, x)
                    conds = conditions_remove_var(conds, var_ind)
                elif(isinstance(x, (int, np.integer))):
                    conds = conditions_remove_var(conds, x)
                else:
                    raise ValueError(f"Unrecognized removal pattern {item_args}")
            elif(len(item_args) == 2):
                d, c = item_args
                conds = conditions_remove(conds, d, c)
            else:
                raise ValueError(f"Too many arguments for remove(): {args}")
        return conds


    # def remove_multi(self, removals):
    #     removals = sorted(removals, reverse=True)
    #     new_conds = conditions_copy(self)
    #     for d, c in removals:
    #         conditions_remove(new_conds, d, c, False)
    #     return new_conds

    def indicies_of(self, lit):
        lit = finesse_to_literal(lit)
        return conditions_indicies_of(self, lit)

    def all_indicies_of(self, lit):
        lit = finesse_to_literal(lit)
        return conditions_all_indicies_of(self, lit)


define_boxing(ConditionsTypeClass,Conditions)

def finesse_to_literal(lit):
    if(isinstance(lit, Literal)):
        return lit
    elif(isinstance(lit, Conditions)):
        return lit.as_literal()
    elif(isinstance(lit, CREFunc)):    
        return literal_ctor(lit)

def edit_list_sortkey(x):
    lst = []
    for y in x:
        if(isinstance(y, (Literal, Conditions, CREFunc))):
            lst.append(-float('inf'))
        else:
            lst.append(y)
    return (len(lst), *lst)

def process_remove_item(conds, x):
    if(not isinstance(x, (list,tuple))):
        x = (x,)

    if(isinstance(x[0], Var)):
        x = (conditions_get_var_ind(conds, x[0]), *x[1:])

    return x

@njit(cache=True)
def conditions_is_literal(conds):
    return len(conds.dnf) == 1 and len(conds.dnf[0]) == 1

@njit(cache=True)
def conditions_as_literal(conds):
    if(conditions_is_literal(conds)):
        return conds.dnf[0][0]
    else:
        raise ValueError("Conditions contain more than 1 literal, conversion to Literal is ambiguous.")


@njit(u2[::1](ConditionsType),cache=True)
def conds_get_var_t_ids(self):
    t_ids = np.empty((len(self.vars),),dtype=np.uint16)
    for i, v in enumerate(self.vars):
        t_ids[i] = v.base_t_id
    return t_ids

@njit(types.void(ConditionsType, f4),cache=True)
def conds_set_weight(self, weight):
    for conj in self.dnf: 
        for lit in conj:
            lit.weight = weight



# -----------------------
# : Auto
@njit(cache=True)
def conds_get_unaliased_var_ptrs(conds):
    lst = List()
    for v in conds.vars:
        if(not v.alias):
            lst.append(v)

    ptrs = np.empty(len(lst),dtype=np.int64)
    for i,v in enumerate(lst):
        ptrs[i] = v.base_ptr

    return ptrs

# @njit(cache=True)
# def assign_alias_by_ptr(ptr,alias):
@njit(types.boolean(i8[::1], i8),cache=True)
def arr_index_of(arr, val):
    for i, x in enumerate(arr):
        if(x == val): return i
    return -1


def auto_alias_unaliased_vars(conds):
    unaliased_var_ptrs = conds_get_unaliased_var_ptrs(conds)
    if(len(unaliased_var_ptrs) > 0):
        frame = inspect.stack()[2][0]
        for k,v in frame.f_locals.items():
            if(isinstance(v, Var)):
                index = arr_index_of(unaliased_var_ptrs, v.base_ptr)
                if(index != -1):
                    var_assign_alias(v, k)
        



### Helper Functions for expressing conditions as python lists of cre.CREFunc instances ###

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
    st.vars = List.empty_list(VarType)    
    st.base_var_map = Dict.empty(i8,i8)
    for conj in dnf: 
        for lit in conj:
            for b_ptr in lit.base_var_ptrs:
                if(b_ptr not in st.base_var_map):
                   st.base_var_map[b_ptr] = len(st.base_var_map)
                   st.vars.append(cast(b_ptr, VarType))
    st.dnf = dnf
    st.has_distr_dnf = False
    # st.matcher_inst_ptr = 0
    return st




@njit(cache=True)
def _conditions_ctor_single_var(_vars,dnf=None):
    st = new(ConditionsType)
    st.idrec = encode_idrec(T_ID_CONDITIONS, 0, 0)
    st.vars = List.empty_list(VarType)
    st.vars.append(cast(_vars.base_ptr, VarType)) 
    st.base_var_map = build_base_var_map(st.vars)
    # print("A",st.base_var_map)
    st.dnf = dnf if(dnf) else new_dnf(1)
    st.has_distr_dnf = False
    # st.is_initialized = False
    # st.matcher_inst_ptr = 0
    return st

@njit(cache=True)
def _conditions_ctor_base_var_map(_vars,dnf=None):
    st = new(ConditionsType)
    st.idrec = encode_idrec(T_ID_CONDITIONS, 0, 0)
    st.vars = build_var_list(_vars)
    st.base_var_map = _vars.copy() # is shallow copy
    st.dnf = dnf if(dnf) else new_dnf(len(_vars))
    st.has_distr_dnf = False
    return st

@njit(cache=True)
def _conditions_ctor_var_list(_vars,dnf=None):
    st = new(ConditionsType)
    st.idrec = encode_idrec(T_ID_CONDITIONS, 0, 0)
    st.vars = List.empty_list(VarType)
    for x in _vars:
        st.vars.append(cast(x.base_ptr, VarType))
    # st.vars = List([ for x in _vars])
    st.base_var_map = build_base_var_map(st.vars)
    # print("C",st.base_var_map)
    st.dnf = dnf if(dnf) else new_dnf(len(_vars))
    st.has_distr_dnf = False
    # st.is_initialized = False
    # st.matcher_inst_ptr = 0
    return st



# @generated_jit(cache=True)
@overload(Conditions,strict=False)
def conditions_ctor(_vars, dnf=None):
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




GTE_PY39 = sys.version_info >= (3,9)




@njit(cache=True)
def conditions_repr(conds, indent="    "):
    distr_dnf = conds_get_distr_dnf(conds)
    is_disjunct = len(distr_dnf) > 1
    context_data = get_cre_context_data()
    s = "OR(\n" if is_disjunct  else ""
    ind1 = indent if is_disjunct else ""
    ind2 = indent*2 if is_disjunct else indent
    for k, distr_conjunct in enumerate(distr_dnf):
        s += f"{ind1}AND("
        for j, var_conjuct in enumerate(distr_conjunct):
            if(j > 0): s += ind2

            # Repr the Var 
            v = conds.vars[j]
            if(k == 0):
                base_name = context_data.t_id_to_type_names[u2(v.base_t_id)]
                if(GTE_PY39):
                    s += f"{v.alias}:=Var({base_name})"
                else:
                    s += f"Var({base_name}, '{v.alias}'"
            else:
                s += f"{v.alias}"

            # Repr lits 
            for i, lit in enumerate(var_conjuct):
                s += f", {str(lit)}"

            # End of lits for this var 
            if(j < len(distr_conjunct)-1):
                s += ",\n"
            elif(len(distr_conjunct) > 1):
                s += f"\n{ind1})"
            else:
                s += ")"
        if(k < len(distr_dnf)-1): s += ",\n"
    if is_disjunct:
        s += "\n)"
    return s


@njit(cache=True)
def conditions_repr_short(self, indent=" "):
    used_var_ptrs = Dict.empty(i8,u1)
    is_disjunct = len(self.dnf) > 1

    s = "OR(\n" if is_disjunct  else ""
    ind = indent if is_disjunct else ""

    for j, conjunct in enumerate(self.dnf):
        s += ind+"AND("
        for i, lit in enumerate(conjunct):
            s += str(lit)
            if(i < len(conjunct)-1): s += ", "
        s += ")"
        if(j < len(self.dnf)-1): s += ",\n"

    if is_disjunct:
        s += "\n)"        
    return s

@njit(cache=True)
def conditions_repr_ampersand(conds, indent=" "):
    distr_dnf = conds_get_distr_dnf(conds)
    is_disjunct = len(distr_dnf) > 1

    s = "OR(\n" if is_disjunct  else ""
    ind1 = indent if is_disjunct else ""
    ind2 = indent*2 if is_disjunct else indent
    for k, distr_conjunct in enumerate(distr_dnf):
        for j, var_conjuct in enumerate(distr_conjunct):
            v = conds.vars[j]
            s += ind1+"(" if j == 0 else ind2
            if(k == 0):
                if(GTE_PY39):
                    s += f"({v.alias}:=Var({get_base_type_name(v)}))"
                else:
                    s += f"(Var({get_base_type_name(v)}, '{v.alias}')"
            else:
                s += f" {v.alias}"
            for i, lit in enumerate(var_conjuct):
                s += f" & {str(lit)}"
            if(j < len(distr_conjunct)-1): s += " &\n"
        s += ")"
        if(k < len(distr_dnf)-1): s += ",\n"
    if is_disjunct:
        s += "\n)"
    return s

@njit(cache=True)
def conditions_repr_ampersand_short(self, indent=" "):
    used_var_ptrs = Dict.empty(i8,u1)
    is_disjunct = len(self.dnf) > 1

    s = "OR(\n" if is_disjunct  else ""
    ind = indent if is_disjunct else ""
    # ind2 = indent*2 if is_disjunct else indent

    for j, conjunct in enumerate(self.dnf):
        s += ind+"("
        for i, lit in enumerate(conjunct):
            # s += "~" if lit.negated else ""
            s += str(lit)
            if(i < len(conjunct)-1): s += " & "
            # if(add_non_conds): 
            #     for var_ptr in lit.base_var_ptrs:
            #         used_var_ptrs[var_ptr] = u1(1)
        s += ")"
        if(j < len(self.dnf)-1): s += ",\n"

    if is_disjunct:
        s += "\n)"        

    # if(add_non_conds):
    #     was_prev =  True if(len(used_var_ptrs) > 0) else False
    #     for j, v in enumerate(self.vars):
    #         if(v.base_ptr not in used_var_ptrs):
    #             if(was_prev): s += " & "
    #             s += v.alias
    #             was_prev = True
    return s











@overload_method(ConditionsTypeClass, "__str__")
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
    var_list = List.empty_list(VarType)
    for ptr in base_var_map:
        var_list.append(cast(ptr, VarType))
    return var_list


@njit(ConditionsType(CREObjType), cache=True)
def to_cond(self):
    t_id, _, _ = decode_idrec(self.idrec)
    if(t_id == T_ID_CONDITIONS):
        return cast(self, ConditionsType)

    elif(t_id == T_ID_VAR):
        v = cast(self,VarType)
        return _conditions_ctor_single_var(v)
    
    elif(t_id == T_ID_FUNC):
        fn = cast(self, CREFuncType)
        self = cast(literal_ctor(fn),CREObjType)
        t_id = T_ID_LITERAL
        # Continue on to construct from literal
    
    if(t_id == T_ID_LITERAL):
        lit = cast(self, LiteralType)
        dnf = new_dnf(1)
        dnf[0].append(lit)
        _vars = List.empty_list(VarType)

        for ptr in lit.base_var_ptrs:
            _vars.append(cast(ptr, VarType))
        return _conditions_ctor_var_list(_vars, dnf)

    print(t_id, T_ID_VAR, T_ID_FUNC, T_ID_LITERAL)
    raise Exception("Object type not recognized for conversion to Conditions object.")


@njit(cache=True)
def assign_lit_var_inds(dnf, base_var_map):
    for conjunct in dnf:
        for lit in conjunct:
            for i, ptr in enumerate(lit.base_var_ptrs):
                lit.var_inds[i] = base_var_map[ptr]

@njit(cache=True)
def conditions_get_var_ind(conds, var):
    for i, v in enumerate(conds.vars):
        if(v == var):
            return i
    return -1

@njit(cache=True)
def conditions_copy(conds):
    dnf = new_dnf(0)
    for conj in conds.dnf:
        new_conj = List.empty_list(LiteralType)
        for lit in conj:
            new_conj.append(lit)
        dnf.append(new_conj)

    new_bvm = Dict.empty(i8,i8)
    for a,b in conds.base_var_map.items():
        new_bvm[a] = b
    return _conditions_ctor_base_var_map(new_bvm, dnf)

        
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


@njit(ConditionsType(CREObjType, CREObjType), cache=True)
def conditions_and(a, b):
    c_a = to_cond(a)
    c_b = to_cond(b)
    
    bvm = build_base_var_map(c_a.vars, c_b.vars)
    dnf = dnf_and(c_a.dnf, c_b.dnf)
    assign_lit_var_inds(dnf, bvm)

    return _conditions_ctor_base_var_map(bvm, dnf)


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

@njit(ConditionsType(CREObjType, CREObjType), cache=True)
def conditions_or(a, b):
    c_a = to_cond(a)
    c_b = to_cond(b)

    bvm = build_base_var_map(c_a.vars, c_b.vars)
    dnf = dnf_or(c_a.dnf, c_b.dnf)
    assign_lit_var_inds(dnf, bvm)
    return _conditions_ctor_base_var_map(bvm, dnf)


def OR(*args):
    if(len(args) == 1):
        return Conditions(args[0])
    # assert len(args) >= 2, "OR requires at least two arguments"
    conds = args[0] | args[1]
    for i in range(2,len(args)):
        conds |= args[i]
    auto_alias_unaliased_vars(conds)
    return conds

def AND(*args):
    # Note: Can probably njit a lot of this to make it faster.

    if(len(args) == 1):
        return Conditions(args[0])
    # assert len(args) >= 2, "AND requires at least two arguments"

    conds = args[0] & args[1]
    for i in range(2,len(args)):
        conds &= args[i]
    auto_alias_unaliased_vars(conds)
    return conds




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


@njit(cache=True)
def conditions_not(c):
    '''NOT inverts the qualifiers and terms like
    NOT(ab+c) = NOT(ab)+c = (a'+b')c' = a'c'+b'c'''
    dnf = dnf_not(c.dnf)
    assign_lit_var_inds(dnf, c.base_var_map)
    return Conditions(c.base_var_map, dnf)




# @generated_jit(cache=True)
@overload(operator.and_)
def cond_overload_and(l, r):
    if(not isinstance(l,ConditionsTypeClass)): return
    if(not isinstance(r,ConditionsTypeClass)): return
    return lambda l,r : conditions_and(l, r)

# @generated_jit(cache=True)
@overload(operator.or_)
def cond_overload_or(l, r):
    if(not isinstance(l,ConditionsTypeClass)): return
    if(not isinstance(r,ConditionsTypeClass)): return
    return lambda l,r : conditions_or(l, r)

# @generated_jit(cache=True)
@overload(operator.not_)
@overload(operator.invert)
def cond_overload_not(c):
    if(not isinstance(c,ConditionsTypeClass)): return
    return lambda c : conditions_not(c)




# @generated_jit(cache=True)

# NOTE : Pretty sure these are not up to date
if(False):
    @njit(VarType(VarType,),cache=True)
    def _build_var_conjugate(v):
        if(v.conj_ptr == 0):
            conj = var_copy(v)
            # conj = new(VarType)
            # var_memcopy(v,conj)
            conj.is_not = u1(0) if v.is_not else u1(1)
            conj.conj_ptr = _ptr_from_struct_incref(v)
            v.conj_ptr = _ptr_from_struct_incref(conj)
        else:
            conj = cast(v.conj_ptr, VarType)
        return conj


    @njit(VarType(VarType,),cache=True)
    def var_NOT(v):
        '''Implementation of NOT for Vars moved outside of NOT 
            definition to avoid recursion issue'''
        if(v.conj_ptr == 0):
            base = cast(v.base_ptr, VarType)

            conj_base = _build_var_conjugate(base)
            g_conj = _build_var_conjugate(cast(v, VarType))

            g_conj.base_ptr = cast(conj_base, i8)

        st = cast(v.conj_ptr, VarType)
        return st

    @njit(cache=True)
    def _conditions_NOT(c):
        new_vars = List.empty_list(VarType)
        ptr_map = Dict.empty(i8,i8)
        for var in c.vars:
            new_var = var_NOT(var)
            ptr_map[cast(var, i8)]  = cast(new_var, i8)
            new_vars.append(new_var)

        dnf = dnf_copy(c.dnf,shallow=False)

        for i, (alpha_conjuncts, beta_conjuncts) in enumerate(dnf):
            for alpha_literal in alpha_conjuncts: 
                alpha_literal.base_var_ptrs = (ptr_map[alpha_literal.base_var_ptrs[0]],0)
            for beta_literal in beta_conjuncts:
                t = (ptr_map[beta_literal.base_var_ptrs[0]], ptr_map[beta_literal.base_var_ptrs[1]])
                beta_literal.base_var_ptrs = t

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




# @generated_jit(cache=True)
# def conditions_not(c):
#     '''Defines ~x for Var and Conditions''' 
#     if(isinstance(c,ConditionsTypeClass)):
#         # ~ operation inverts the qualifiers and terms like
#         #  ~(ab+c) = ~(ab)c' = (a'+b')c' = a'c'+b'c'
#         def impl(c):
#             dnf = dnf_not(c.dnf)
#             return Conditions(c.base_var_map, dnf)
#     elif(isinstance(c,VarTypeClass)):
#         # If we applied to a var then serves as NOT()
#         def impl(c):
#             return _var_NOT(c)
#     return impl








#### Linking ####

if(False):
    @njit(cache=True)
    def link_literal_instance(literal, ms):
        link_data = generate_link_data(literal.pred_node, ms)
        literal.link_data = link_data
        literal.ms_ptr = cast(ms, i8)
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
        conds.ms_ptr = cast(ms, i8)#_ptr_from_struct_incref(mem)
        if(old_ptr != 0): _decref_ptr(old_ptr)
        return conds


#### Initialization ####


@njit(cache=True)
def build_distributed_dnf(c, index_map=None):
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
            for base_ptr in lit.op.base_var_ptrs:
                ind = index_map[i8(base_ptr)]
                if(ind > max_ind): max_ind = ind

            insertion_conj = distr_conjuct[max_ind]
            was_inserted = False
            
            lit_n_vars = lit.op.n_args

            for k in range(len(insertion_conj)-1,-1,-1):
                if(lit_n_vars >= insertion_conj[k].op.n_args):
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
lit_pos_pair = Tuple((LiteralType, Tuple((i8, i8))))
lit_pos_pair_list = ListType(lit_pos_pair)
lit_unq_tup_type = Tuple((i8,unicode_type))

@njit(cache=True)
def conds_to_lit_sets(self):
    ''' Convert a Conditions object to a list of dictionaries (one for each
        conjunct) that map the unique tuple (negated, unique_str) for each Literal 
        to a list of literals that have the same unique tuple. This reorganization
        ensures that cases like antiunify( (a != b), (x != y) & (y != z) ) try
        remappings where like-terms get chances to be remapped.'''
    lit_sets = List()
    for d_ind, conjunct in enumerate(self.dnf):
        d = Dict.empty(lit_unq_tup_type, lit_pos_pair_list)#Dict.empty(i8, var_set_list_type)
        for c_ind, lit in enumerate(conjunct):
            unq_tup = literal_get_unique_tuple(lit)
            # print(lit, ":", unq_tup)

            if(unq_tup not in d):
                l = d[unq_tup] = List.empty_list(lit_pos_pair)
            else:
                l = d[unq_tup]

            # base_var_ptrs = np.empty(len(lit.op.base_vars), dtype=np.int64)
            # for i, base_var in enumerate(lit.op.base_vars):
            #     base_var_ptrs[i] = base_var.base_ptr

            l.append((lit, (d_ind, c_ind)))

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

@njit(cache=True)
def union_keys(a, b):
    l = List()
    for k in a:
        l.append(k)
    for k in b:
        if(k not in a): l.append(k)
    return l    

u2_arr = u2[::1]
@njit(cache=True)
def lit_set_to_ind_sets(lit_set, base_ptrs_to_inds):
    ''' Takes a list of literals and a mapping from base_var_ptrs to indicies
        and outputs a list of arrays of indicies'''
    l = List.empty_list(u2_arr)
    for lit in lit_set:
        base_set = lit.base_var_ptrs
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

@njit(cache=True)
def align_greedy(score_matrix):#, prt=False):
    N, M = score_matrix.shape
    alignment = -np.ones(N, dtype=np.int16)
    covered = -np.ones(M, dtype=np.int16)
    # scores = np.zeros(N, dtype=np.int64)
    # diag_penalty = 1.0/(N+M)
    # biased_scores = np.empty((N,M),dtype=np.float32)
    # for i in range(N):
    #     for j in range(M):
    #         biased_scores[i,j] = -float(score_matrix[i,j] 
    #                                 - abs(i-j)*diag_penalty
    #                               )
    
    c = 0
    # for flat_ind in np.argsort(biased_scores.flatten()):
    for flat_ind in np.argsort(score_matrix.flatten())[::-1]:
        i, j = flat_ind // M, flat_ind % M
        score = score_matrix[i,j]

        # Early stop
        if(score == 0.0 or c == N):
            break
                    
        if(alignment[i] == -1 and j not in covered):
            alignment[i] = i2(j)
            # scores[i] = score
            covered[c] = i2(j)
            c += 1

    # if(prt):
    #     print(score_matrix)
    #     print(np.sum(scores), alignment)
    return alignment

@njit(cache=True)
def score_remap(remap, score_matrix, beta_matrix):
    score = u2(0)
    for i, j in enumerate(remap):
        s = score_matrix[i,j]
        b = beta_matrix[i,j]
        score += s - (b >> 1) + u2(s != 0)
    return score

# @njit(cache=True, locals={"remaps":i2[:,::1]})
# def get_possible_remap_inds(a_ind_set, b_ind_set, n_a, n_b, a_fixed_inds):
#     ''' Given arrays of var indicies corresponding to a common type of literal (e.g. (x<y) & (a<b))
#         between Conditions A and B. Generate all possible remappings of variables in 
#         A to variables in B. Each remapping is represented by an array of indices 
#         of variables in B. For example if A has vars {a,b,c} and B has vars {x,y,z}
#         then the remapping [a,b,c] -> [y,z,x] is represented as [1,2,0]. Unresolved
#         vars are represented as -1. For example [a,b,-] -> [y,x,-] is represented as [1,0,-1]

#         a_ind_set/b_ind_set : arrays of single or paired indicies associated with each literal
#         n_a/n_b: the number of variables in A and B respectively
#         a_fixed_inds : for each var in A var indicies in B that they are certain to map to.
#     '''
#     # Produce a boolean matrix identifying feasible remapping pairs
#     poss_remap_matrix = np.zeros((n_a,n_b), dtype=np.uint8)
#     for a_base_inds in a_ind_set:
#         for b_base_inds in b_ind_set:
#             # A and B are both alphas or both betas
#             if(len(a_base_inds) == len(b_base_inds)):
#                 for ind_a, ind_b in zip(a_base_inds, b_base_inds):
#                     poss_remap_matrix[ind_a, ind_b] = True

#     # The marginal sums, i.e. number of remappings of each varibles in A and B, respectively
#     num_remaps_per_a = poss_remap_matrix.sum(axis=1)
#     # num_remaps_per_b = poss_remap_matrix.sum(axis=0)

#     n_possibilities = 1
#     for i in range(len(num_remaps_per_a)):
#         if(a_fixed_inds[i] != -1): 
#             num_remaps_per_a[i] = 1
#         n_remaps = num_remaps_per_a[i]
#         if(n_remaps != 0): n_possibilities *= n_remaps

#     print("n_possibilities", n_possibilities)
    
#     # Find the first variable that is remappable 
#     first_i = 0
#     for i,n in enumerate(num_remaps_per_a): 
#         if(n > 0): first_i = i; break;

#     # Use the poss_remap_matrix to fill in a 2d-array of remap options (-1 padded) 
#     remap_options = -np.ones((n_a,n_b), dtype=np.int16)
#     for i in range(n_a):
#         if(a_fixed_inds[i] != -1):
#             remap_options[i,0] = a_fixed_inds[i]
#         else:
#             k = 0
#             for j in range(n_b):
#                 if(poss_remap_matrix[i,j]):
#                     remap_options[i,k] = j
#                     k += 1

#     print("remap_options", remap_options)
#     # Build the set of possible remaps by taking the cartsian product of remap_options
#     remaps = np.empty((n_possibilities, n_a), dtype=np.int16)
    
#     remap_inds = remap_options[:,0].copy()

#     j_iters = np.zeros(n_a, dtype=np.int16)
#     iter_i = first_i; iter_j = j_iters[first_i] = -1; done = False;
#     c = 0
#     while(not done):  
#         # Increment to the next possibility for the first variable 
#         iter_j = j_iters[iter_i] = j_iters[iter_i] + 1      

#         # If this overflows find the next variable that can be incremented without overflowing
#         while(iter_j >= num_remaps_per_a[iter_i]):
#             # When the iter overflows the set of possibilities reset it to the beginning
#             remap_inds[iter_i] = remap_options[iter_i,0]
#             j_iters[iter_i] = 0

#             # Move on to the next set and iterate it, if that would 
#             #  go beyond the last var in A then we're done.
#             iter_i += 1
#             if(iter_i >= n_a):
#                 done = True; break; 
#             iter_j = j_iters[iter_i] = j_iters[iter_i] + 1      
#         if(done): break

#         remap_inds[iter_i] = remap_options[iter_i,iter_j]
#         iter_i = first_i

#         remaps[c, :] = remap_inds
#         c += 1
#     return remaps




# @njit(cache=True)
# def try_merge_remaps(ref_remap, remap):
#     merged_remap = ref_remap.copy()
#     status = 0 # 0 = same, 1 = new, 2 = not same
#     for k, (ref_ind, ind) in enumerate(zip(ref_remap, remap)):
#         if(ind == -1):
#             pass
#         elif(ref_ind == -1 and ind != -1):
#             status = 1
#             merged_remap[k] = ind
#         elif(ref_ind != ind):
#             status = 2
#             break
#     return status, merged_remap


# i2_arr = i2[::1]
# f8_i2_arr_tup = Tuple((f8,i2[::1]))

# @njit(cache=True)
# def score_remaps(lit_set_a, lit_set_b, bpti_a, bpti_b, a_fixed_inds, 
#      op_key_intersection=None, max_remaps=15):
#     if(op_key_intersection is None):
#         op_key_intersection = intersect_keys(lit_set_a, lit_set_b)

#     scored_remap_sets = List()



#     # Case 1: There are no common literals between the conditions
#     if(len(op_key_intersection) == 0):
#         a_ind_set = List([np.arange(len(bpti_a), dtype=np.uint16)])
#         b_ind_set = List([np.arange(len(bpti_b), dtype=np.uint16)])
#         remaps = get_possible_remap_inds(a_ind_set, b_ind_set, len(bpti_a), len(bpti_b), a_fixed_inds)
        
#         scores = np.empty(len(remaps),dtype=np.float64)
#         for c, remap in enumerate(remaps):
#             matched_As = get_matched_mask(a_ind_set, b_ind_set, remap)
#             scores[c] = np.sum(matched_As)

#         scored_remap_sets.append((remaps,scores))

#     # Case 2: There are some common literals between the conditions
#     else:
#         remap_score_matrix = np.zeros((len(bpti_a), len(bpti_b)), dtype=np.int32)

#         for k in op_key_intersection:
#             a_ind_set = lit_set_to_ind_sets(lit_set_a[k], bpti_a)
#             b_ind_set = lit_set_to_ind_sets(lit_set_b[k], bpti_b)


#             for a_base_inds in a_ind_set:
#                 for b_base_inds in b_ind_set:
#                     # A and B are both alphas or both betas
#                     if(len(a_base_inds) == len(b_base_inds)):
#                         for ind_a, ind_b in zip(a_base_inds, b_base_inds):
#                             remap_score_matrix[ind_a, ind_b] += 1
            
#         scores, remap = align_greedy(remap_score_matrix)
#         print("GREEDY", )

#         return List([(np.sum(scores), remap)])
#             # continue
#             # print("ENTER get_possible_remap_inds")
#             # remaps = get_possible_remap_inds(a_ind_set, b_ind_set, len(bpti_a), len(bpti_b), a_fixed_inds)
#             # print("EXIT get_possible_remap_inds")
#             # scores = np.empty(len(remaps),dtype=np.float64)
#             # for c, remap in enumerate(remaps):
#             #     matched_As = get_matched_mask(a_ind_set,b_ind_set,remap)
#             #     scores[c] = np.sum(matched_As)

#             # scored_remap_sets.append((remaps,scores))

        



#     scored_remaps = List.empty_list(f8_i2_arr_tup)
#     # For every unique type of literal (e.g. (x<y) & (a<b))
    
#     for remaps, scores in scored_remap_sets:    

#         # Order by score, drop remaps that have a score of zero 
#         order = np.argsort(scores)
#         first_nonzero = 0
#         for ind in order:
#             if(scores[ind] > 0): break
#             first_nonzero +=1

#         order = order[first_nonzero:][::-1][:min(max_remaps, len(order))]
        
#         og_L = len(scored_remaps)
#         ref_was_merged = np.zeros(og_L, dtype=np.uint8)
#         n_merged = 0
#         for i, ind in enumerate(order):
            
#             score, remap = scores[ind], remaps[ind]
#             was_merged = False
#             # print("i", i, remap)
#             for j in range(len(scored_remaps)):
                
#                 if(not ref_was_merged[j]):
#                     ref_score, ref_remap = scored_remaps[j]
#                     # print("j", j, ref_remap)
#                     merged_remap = ref_remap.copy()

#                     # status : 0 = same, 1 = new, 2 = not same
#                     status, merged_remap = try_merge_remaps(ref_remap, remap)
                    
#                     if(status != 2):
#                         if(status == 1):
#                             # If they aren't quite same append the merge
#                             scored_remaps.append((ref_score+score, merged_remap))
#                         else:
#                             # When they are the same replace the old one
#                             scored_remaps[j] = (ref_score+score, merged_remap)
#                             # Mask out j to ensure we don't merge two remaps from this 
#                             #  literal set into a ref_remap from a previous one. We'll 
#                             #  only mask the first possibility which has the largest score. 
#                             ref_was_merged[j] = 1
#                         was_merged = True

#             if(not was_merged):
#                 scored_remaps.append((score,remap))

#         # Sort the scored_remap list
#         scores = np.empty(len(scored_remaps),dtype=np.float64)
#         for i,(s,_) in enumerate(scored_remaps): scores[i] = s
#         order = np.argsort(scores)[::-1]
#         scored_remaps = List([scored_remaps[ind] for ind in order])

#     print(scored_remaps)
#     return scored_remaps

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

# @njit(cache=True)
# def _build_score_aligment_matrices(ls_as, ls_bs, bpti_a, bpti_b, a_fixed_inds):
#     '''For each unique remap make a matrix that holds the remap score between 
#        conjunct_i and conjunct_j for all possible alignments of the conjuncts 
#        in A and conjuncts in B. Return a list of remap matrix pairs.'''
#     score_aligment_matrices = Dict.empty(FrozenArrTypei2, f8_2darr_type)
#     for i, ls_a in enumerate(ls_as): 
#         for j, ls_b in enumerate(ls_bs): 
#             scored_remaps = score_remaps(ls_a, ls_b, bpti_a, bpti_b, a_fixed_inds)
#             for score, remap in scored_remaps:
#                 f_remap = FrozenArr(remap)
#                 if(f_remap not in score_aligment_matrices):
#                     rank = np.sum(remap == -1)
#                     score_matrix =  np.zeros((len(ls_as), len(ls_bs)),dtype=np.float64)
#                     score_aligment_matrices[f_remap] = score_matrix

#                 score_aligment_matrices[f_remap][i,j] = score
#     return List(score_aligment_matrices.items())

# @njit(cache=True)
# def _max_remap_score_for_alignments(score_matrix):
#     num_conj = score_matrix.shape[0]
#     orders = List.empty_list(i8_arr)
#     col_assign_per_row = -np.ones(num_conj,dtype=np.int64)
#     n_assigned = 0
#     row_maxes = np.zeros(num_conj,dtype=np.float64)
#     for i in range(num_conj):
#         orders_i = np.argsort(-score_matrix[i])
#         orders.append(orders_i)
#         row_maxes[i] = score_matrix[i][orders_i[0]]

#     order_of_rows = np.argsort(-row_maxes)
#     score = 0
#     while(len(order_of_rows) > 0):
#         row_ind = order_of_rows[0]

#         c = 0 
#         col_ind = orders[row_ind][c]
#         while(col_ind in col_assign_per_row):
#             col_ind = orders[row_ind][c]; c += 1
#             if(c >= len(orders[row_ind])): break

#         col_assign_per_row[row_ind] = col_ind
#         score += score_matrix[row_ind,col_ind]
#         order_of_rows = order_of_rows[1:]
#     return score, col_assign_per_row

# @njit(cache=True)
# def _conj_from_litset_and_remap(ls_a, ls_b, remap, keys, bpti_a, bpti_b):
#     conj = List.empty_list(LiteralType)

#     total = 0
#     kept = 0
#     for unq_key in keys:
#         lit_set_a = ls_a[unq_key]
#         lit_set_b = ls_b[unq_key]
        
#         # ind_set_a = lit_set_to_ind_sets(lit_set_a, bpti_a)
#         # ind_set_b = lit_set_to_ind_sets(lit_set_b, bpti_b)
#         matched_mask = get_matched_mask(lit_set_a, lit_set_b, remap)

#         # print(lit_set_a, lit_set_b, matched_mask)
#         for keep_it, lit in zip(matched_mask, lit_set_a):
#             total += 1
#             if(keep_it):
#                 new_lit = literal_copy(lit)
#                 # new_lit = literal_ctor(lit.op)
#                 # new_lit.negated = lit.negated
#                 conj.append(new_lit)
#                 kept += 1
#     return conj, kept

# @njit(cache=True,locals={"i" : u2})
# def make_base_ptrs_to_inds(self):
#     ''' From a collection of vars a mapping between their base ptrs to unique indicies'''
#     base_ptrs_to_inds = Dict.empty(i8,u2)
#     i = u2(0)
#     for v in self.vars:
#         base_ptrs_to_inds[v.base_ptr] = i; i += 1;
#     return base_ptrs_to_inds


@njit(cache=True)
def get_fixed_inds(c_a, c_b, map_same_var, map_same_alias):
    a_fixed_inds = np.full(len(c_a.vars), -2, dtype=np.int16)

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



# @njit(Tuple((f8, i8[::1], i8[::1]))(ConditionsType, ConditionsType, boolean, boolean), cache=True)
# def structure_map_conds(c_a, c_b, fix_same_var, fix_same_alias):

#     ls_as = conds_to_lit_sets(c_a)
#     ls_bs = conds_to_lit_sets(c_b)

#     bpti_a = make_base_ptrs_to_inds(c_a)
#     bpti_b = make_base_ptrs_to_inds(c_b)

#     remap_size = len(bpti_a)
#     num_conj = len(ls_as)
#     best_score = -np.inf
#     best_remap = np.arange(remap_size, dtype=np.int16)


#     a_fixed_inds = get_fixed_inds(c_a, c_b, fix_same_var, fix_same_alias)
    
#     # Case 1: c_a and c_b are both single conjunctions like (lit1 & lit2 & lit3)
#     if(len(ls_as) == 1 and len(ls_bs) == 1):
#         op_key_intersection = intersect_keys(ls_as[0], ls_bs[0])
#         scored_remaps = score_remaps(ls_as[0], ls_bs[0], bpti_a, bpti_b, a_fixed_inds,
#                                     op_key_intersection=op_key_intersection)
#         best_score, best_remap = scored_remaps[0]

#         return best_score, best_remap, np.zeros(1,dtype=np.int64)

#         # conj = _conj_from_litset_and_remap(ls_as[0], ls_bs[0], 
#         #          best_remap, op_key_intersection, bpti_a, bpti_b)

#         # dnf = List([conj])
#         # conds = _conditions_ctor_dnf(dnf)

#     # Case 2: c_a or c_b have disjunction like ((lit1 & lit2 & lit3) | (lit4 & lit5))
#     else:
#         score_aligment_matrices = _build_score_aligment_matrices(ls_as, ls_bs, bpti_a, bpti_b, a_fixed_inds)
        
#         # Find the upperbounds on each remap
#         score_upperbounds = np.zeros(len(score_aligment_matrices),dtype=np.float64)
#         for i, (f_remap, score_matrix) in enumerate(score_aligment_matrices):
#             for row in score_matrix:
#                 score_upperbounds[i] += np.max(row)

#         # Look for the best remap trying remaps with large upperbounds first
#         descending_upperbound_order = np.argsort(-score_upperbounds)
#         best_score = 0
#         best_alignment = np.empty(0,dtype=np.int64)
#         for i in descending_upperbound_order:
#             # Stop looking after couldn't possibly beat the best so far.
#             upper_bound_score = score_upperbounds[i]
#             if(upper_bound_score < best_score):
#                 break

#             # Find the score for the best conjuct alignment for this remap 
#             f_remap, score_matrix = score_aligment_matrices[i]
#             score, alignment = _max_remap_score_for_alignments(score_matrix)
#             if(score > best_score):
#                 best_remap = f_remap.arr
#                 best_score = score
#                 best_alignment = alignment

#         return best_score, best_remap, best_alignment

#         # dnf = List()
#         # for i, ls_a in enumerate(ls_as):
#         #     ls_b = ls_bs[alignment[i]]

#         #     op_key_intersection = intersect_keys(ls_a, ls_b)
#         #     conj = _conj_from_litset_and_remap(ls_a, ls_b,
#         #              best_remap, op_key_intersection, bpti_a, bpti_b)
#         #     dnf.append(conj)

        
#         # conds = _conditions_ctor_dnf(dnf)
    

#     # Make sure base Vars are ordered like c_a
#     unordered_base_vars = conds.base_var_map
#     base_var_map = Dict.empty(i8,i8)
#     c = 0 
#     for base_ptr in bpti_a:
#         if(base_ptr in unordered_base_vars):
#             base_var_map[base_ptr] = c
#             c += 1
#     conds = _conditions_ctor_base_var_map(base_var_map, dnf)

#     return conds, best_score

# -------------------------------
# : Structure Mapping

@njit
def _calc_remap_score_matrices(lit_groups, shape, a_fixed_inds):
    Da, Db, N, M = shape
    remap_score_matrices = np.zeros((Da, Db, N, M), dtype=np.uint16)
    beta_score_matrices = np.zeros((Da, Db, N, M), dtype=np.uint16)

    b_fixed_inds = np.full(M, -2, dtype=np.int16)
    for i,j in enumerate(a_fixed_inds):
        if(j != -2):
            b_fixed_inds[j] = i

    # print(b_fixed_inds)
    # For each conjunct pair
    for i, groups_i in enumerate(lit_groups):
        for j, groups_ij in enumerate(groups_i):
            score_matrix = remap_score_matrices[i, j]
            beta_matrix = beta_score_matrices[i, j]

            for _, var_inds_a, _, var_inds_b in groups_ij: 
            # Intersect the literals which are common between the conjuncts
            # op_key_intersection = intersect_keys(ls_a, ls_b)
            # for k in op_key_intersection:
                # lits_a, lits_b = ls_a[k], ls_b[k]

                # For each literal in A
                for v_inds_a in var_inds_a:
                    ind_a0 = v_inds_a[0]
                    fix_b0 = a_fixed_inds[ind_a0]

                    # Alpha Case
                    if(len(v_inds_a) == 1):
                        # if(a_fixed_inds[ind_a0] != -1):

                        for v_inds_b in var_inds_b:
                            ind_b0 = v_inds_b[0]
                            if((fix_b0 == -2 or fix_b0 == ind_b0)):
                                # print("alpha", lit_a, lit_b)
                                score_matrix[ind_a0, ind_b0] += 1
                        # print(f"({ind_a0}, {ind_b0})")

                    # Beta Case
                    else:
                        ind_a1 = v_inds_a[1]
                        # print("INDS", ind_a0, ind_a1, a_fixed_inds[ind_a0], a_fixed_inds[ind_a1])
                        fix_b1 = a_fixed_inds[ind_a1]
                        for v_inds_b in var_inds_b:
                            ind_b0 = v_inds_b[0]
                            ind_b1 = v_inds_b[1]
                            fix_a0 = b_fixed_inds[ind_b0]
                            fix_a1 = b_fixed_inds[ind_b1]
                            if((fix_b0 == -2 or fix_b0 == ind_b0) and
                               (fix_a0 == -2 or fix_a0 == ind_a0) and
                               (fix_b1 == -2 or fix_b1 == ind_b1) and
                               (fix_a1 == -2 or fix_a1 == ind_a1) ):
                                # print("beta", lit_a, lit_b)
                                # print("did apply", ind_a0, ind_b0, ";", ind_a1, ind_b1)
                                score_matrix[ind_a0, ind_b0] += 1
                                score_matrix[ind_a1, ind_b1] += 1
                                beta_matrix[ind_a0, ind_b0] += 1
                                beta_matrix[ind_a1, ind_b1] += 1

                        # print(f"({ind_a0} -> {ind_b0}) => ({ind_a1} -> {ind_b1})")
                        # print(f"({ind_a1} -> {ind_b1}) => ({ind_a0} -> {ind_b0})")
    return remap_score_matrices, beta_score_matrices



# @njit
# def _get_supported_remaps(ls_as, ls_bs, N, M, a_fixed_inds):
#     for i, ls_a in enumerate(ls_as):
#         for j, ls_b in enumerate(ls_bs):
#             # supported_remaps = np.full((N, M, N+M, 2),-1,dtype=np.int16)
#             # print("SIZE:", N*M*(N+M)*2)
#             matrix = np.zeros((N, M),dtype=np.int16)
#             # remap_score_matrix = remap_score_matrices[i, j]
            
#                     # for v_ind, ind_b in enumerate(lit.var_inds):
#                     #     b_vars_w_ind[ind_b, v_ind] += 1

#             # Intersect the literals which are common between the conjuncts
#             op_key_intersection = intersect_keys(ls_a, ls_b)
#             for k in op_key_intersection:
#                 lits_a, lits_b = ls_a[k], ls_b[k]

#                 # For each literal in A
#                 for lit_a, _ in lits_a:
#                     ind_a0 = lit_a.var_inds[0]
#                     fix_b0 = a_fixed_inds[ind_a0]

#                     # Alpha Case
#                     if(len(lit_a.var_inds) == 1):
#                         # if(a_fixed_inds[ind_a0] != -1):

#                         for lit_b, _ in lits_b:
#                             ind_b0 = lit_b.var_inds[0]
#                             if((fix_b0 == -1 or fix_b0 == ind_b0)):
#                                 matrix[ind_a0, ind_b0] += 1
#                         print(f"({ind_a0}, {ind_b0})")

#                     # Beta Case
#                     else:
#                         ind_a1 = lit_a.var_inds[1]
#                         fix_b1 = a_fixed_inds[ind_a1]
#                         for lit_b, _ in lits_b:
#                             ind_b0 = lit_b.var_inds[0]
#                             ind_b1 = lit_b.var_inds[1]
#                             if((fix_b0 == -1 or fix_b0 == ind_b0) and
#                                (fix_b1 == -1 or fix_b1 == ind_b1)):
#                                 matrix[ind_a0, ind_b0] += 1
#                                 matrix[ind_a1, ind_b1] += 1
#                         print(f"({ind_a0} -> {ind_b0}) => ({ind_a1} -> {ind_b1})")
#                         print(f"({ind_a1} -> {ind_b1}) => ({ind_a0} -> {ind_b0})")

#             print(matrix)

#             # for ind_a in range(N):
#             #     if(a_fixed_inds[ind_a] != -1):
#             #         ind_b = a_fixed_inds[ind_a]
#             #         s = np.sum(np.minimum(a_vars_w_ind[ind_a], b_vars_w_ind[ind_b]))
#             #         remap_score_matrix[ind_a, ind_b] += s
#             #     else:    
#             #         for ind_b in range(M):
#             #             s = np.sum(np.minimum(a_vars_w_ind[ind_a], b_vars_w_ind[ind_b]))
#             #             remap_score_matrix[ind_a, ind_b] += s



@njit(cache=True)
def _get_best_alignment(remap_score_matrices, beta_score_matrices):
    l_a, l_b, N, M = remap_score_matrices.shape

    if(N == 1 and M == 1):
        return (np.zeros(1, dtype=np.int16),
                remap_score_matrices[0,0],
                beta_score_matrices[0,0])

    # Fill in a matrix which captures the best remap 
    #  for each conjunct pair. This is an upper bound
    #  on the score contribution of the best remap of the
    #  the best alignment.
    upb_alignment_matrix = np.zeros((l_a, l_b), dtype=np.uint16)
    remaps = np.empty((l_a, l_b, N), dtype=np.int16)
    for i in range(l_a):
        for j in range(l_b):
            remap = align_greedy(remap_score_matrices[i,j])
            upb_alignment_matrix[i, j] = score_remap(remap,
                remap_score_matrices[i,j], beta_score_matrices[i,j])

    # Assume that the best alignment is just the one that
    #  maximizes these upper bounds
    alignment = align_greedy(upb_alignment_matrix)

    cum_score_matrix = np.zeros((N, M), dtype=np.uint16)
    cum_beta_matrix = np.zeros((N, M), dtype=np.uint16)
    for i,j in enumerate(alignment):
        cum_score_matrix += remap_score_matrices[i,j]
        cum_beta_matrix += beta_score_matrices[i,j]

    
    return alignment, cum_score_matrix, cum_beta_matrix


@njit(cache=True)
def get_matched_masks(group, remap):
# def get_matched_mask(a_ind_set, b_ind_set, remap):
    _, var_inds_a, _, var_inds_b = group

    a_inds_remapped = np.empty(len(var_inds_a[0]),dtype=np.int16)
    matched_As = np.zeros(len(var_inds_a),dtype=np.uint8)
    matched_Bs = np.zeros(len(var_inds_b),dtype=np.uint8)
    
    for i, v_inds_a in enumerate(var_inds_a):
        # Apply this mapping for A -> B 
        for ix, var_ind in enumerate(v_inds_a):
           a_inds_remapped[ix] = remap[var_ind]

        # Greedily assign literals in remapped A to literals in B
        for j, v_inds_b in enumerate(var_inds_b):
            if(not matched_Bs[j] and np.array_equal(a_inds_remapped, v_inds_b)):
                matched_As[i] = 1
                matched_Bs[j] = 1
                break
    return matched_As, matched_Bs

# from numba import i8
@njit(cache=True)
def _score_and_mask_conj(conj_a, conj_b, groups, remap):
    
    keep_mask_a = np.zeros(len(conj_a), dtype=np.uint8)
    keep_mask_b = np.zeros(len(conj_b), dtype=np.uint8)

    # op_key_intersection = intersect_keys(ls_a, ls_b)
    score = u2(0)
    for group in groups: 
        c_inds_a, _, c_inds_b, _ = group
        # lit_set_a = ls_a[unq_key]
        # lit_set_b = ls_b[unq_key]
        # mm_a, mm_b = get_matched_masks(lit_set_a, lit_set_b, remap)
        mm_a, mm_b = get_matched_masks(group, remap)

        for keep_it, c_ind in zip(mm_a, c_inds_a):
            # print("KEEP" if keep_it else "TOSS", lit)
            keep_mask_a[c_ind] = keep_it
            score += keep_it

        for keep_it, c_ind in zip(mm_b, c_inds_b):
            # print("KEEP" if keep_it else "TOSS", lit)
            keep_mask_b[c_ind] = keep_it

    return score, keep_mask_a, keep_mask_b

@njit
def _score_and_mask(c_a, c_b, lit_groups, alignment, remap):
    max_conj_len_a = max([len(conj) for conj in c_a.dnf])
    max_conj_len_b = max([len(conj) for conj in c_b.dnf])
    keep_mask_a = np.zeros((len(c_a.dnf), max_conj_len_a), dtype=np.uint8)
    keep_mask_b = np.zeros((len(c_b.dnf), max_conj_len_b), dtype=np.uint8)
    
    # +1 for every Var which is kept 
    score = np.sum(remap != -1, dtype=np.uint16)

    # +1 for every literal across all disjunctions kept 
    for i, j in enumerate(alignment):
        groups_ij = lit_groups[i][j]
        
        # ls_a, ls_b = ls_as[i], ls_bs[j]
        conj_a, conj_b = c_a.dnf[i], c_b.dnf[j],
        _score, _keep_mask_a, _keep_mask_b = _score_and_mask_conj(
            conj_a, conj_b, groups_ij, remap
        )
        # print("grp", i, j, _keep_mask_a, f"L={len(groups_ij)}")
        keep_mask_a[i,:len(_keep_mask_a)] = _keep_mask_a
        keep_mask_b[j,:len(_keep_mask_b)] = _keep_mask_b
        
        score += _score
    return score, keep_mask_a, keep_mask_b

@njit(cache=True)
def get_unambiguous_inds(cum_score_matrix, a_fixed_inds):
    unamb_inds = a_fixed_inds.copy()
    unconstr_mask = np.zeros(len(a_fixed_inds),dtype=np.uint8)
    new_unamb = 0
    N, M = cum_score_matrix.shape
    for a_ind in range(N):
        # Don't touch if already assigned  
        if(a_fixed_inds[a_ind] != -2):
            continue

        # Find any assignments with non-zero score
        cnt = 0
        non_zero_b_ind = -1
        for b_ind in range(M):
            if(cum_score_matrix[a_ind,b_ind] != 0):
                cnt += 1
                non_zero_b_ind = b_ind

        # If there is exactly one assignment with a non-zero
        #  score then apply that assignment.
        if(cnt == 1):
            new_unamb += 1
            unamb_inds[a_ind] = non_zero_b_ind
        # Or if they all have a score of zero then mark
        #  them as unconstrainted.
        elif(cnt == 0):
            unconstr_mask[a_ind] = 1
            
    # For variables which are made unconstrained by the
    #  remap so far, greedily assign each i -> j which 
    #  is maximally diagonal, otherwise drop (i.e. i -> -1).
    unassigned_j_mask = np.ones(M,dtype=np.uint8)
    unassigned_j_mask[unamb_inds[unamb_inds >= 0]] = 0
    # print(": ", unamb_inds)
    # print(cum_score_matrix)
    # print("unconstrained i:", np.nonzero(unconstr_mask)[0])
    # print("unassigned j:", np.nonzero(unassigned_j_mask)[0])
    for i in np.nonzero(unconstr_mask)[0]:
        min_j  = -1
        min_dist = 100000
        for j, unassigned in enumerate(unassigned_j_mask):
            if(unassigned):
                dist = abs(i-j)
                if(dist < min_dist):
                    min_j = j
                    min_dist = dist

        if(min_j != -1):
            unassigned_j_mask[min_j] = 0
            unamb_inds[i] = min_j
        else:
            unamb_inds[i] = -1
    # print(":>", unamb_inds)
    # print()

    return unamb_inds, new_unamb


@njit(cache=True)
def get_best_ind_iter(cum_score_matrix, a_fixed_inds):
    best_iter = None
    best_unamb = (-1, 0.0)
    for i in range(len(cum_score_matrix)):
        # Skip if already assigned
        if(a_fixed_inds[i] != -2):
            continue

        row = cum_score_matrix[i].astype(np.int32)
        inds = np.argsort(-row)

        # Don't make iterators for rows of all zeros
        if(row[inds[0]] == 0):
            continue

        inds = inds[:np.argmin(row[inds])]
        scores = row[inds]
        # NOTE: Maybe harmonic mean is better?
        # amb = (len(scores)-1)/np.mean(1/(1+scores[0] - scores[1:]))
        unamb = (scores[0],np.mean(scores[0] - scores[1:]))
        if(unamb > best_unamb):
            best_iter = (i, inds)
            best_unamb = unamb

        # print(row)
    #     print(i, inds, scores, unamb)
    # print("BEST", best_unamb, best_iter)
    return best_iter


lit_groups_type = Tuple((i8[::1], i8[:,::1],i8[::1], i8[:,::1]))
lit_groups_list_type =  ListType(lit_groups_type)

@njit
def make_lit_groups(c_a, c_b):
    ls_as = conds_to_lit_sets(c_a)
    ls_bs = conds_to_lit_sets(c_b)

    lit_groups = List()
    for i, ls_a in enumerate(ls_as):
        groups_i = List.empty_list(lit_groups_list_type)
        lit_groups.append(groups_i)

        for j, ls_b in enumerate(ls_bs):
            
            groups_ij = List.empty_list(lit_groups_type)
            groups_i.append(groups_ij)
            # Intersect the literals which are common between the conjuncts
            op_key_intersection = intersect_keys(ls_a, ls_b)
            for k in op_key_intersection:
                lits_a, lits_b = ls_a[k], ls_b[k]

                c_inds_a = np.empty(len(lits_a),dtype=np.int64)
                var_inds_a = np.empty((len(lits_a),len(lits_a[0][0].var_inds)),dtype=np.int64)
                c_inds_b = np.empty(len(lits_b),dtype=np.int64)
                var_inds_b = np.empty((len(lits_b),len(lits_b[0][0].var_inds)),dtype=np.int64)

                for ind, (lit_a, (_,c_ind)) in enumerate(lits_a):
                    c_inds_a[ind] = c_ind
                    var_inds_a[ind] = lit_a.var_inds

                for ind, (lit_b, (_,c_ind)) in enumerate(lits_b):
                    c_inds_b[ind] = c_ind
                    var_inds_b[ind] = lit_b.var_inds


                groups_ij.append((c_inds_a, var_inds_a, c_inds_b, var_inds_b))
            print("IJ", i, j, len(groups_ij))
    return lit_groups







align_remap_type = Tuple((i8, i2[::1], i2[::1]))
stack_item_type = Tuple((i8,i8,i8[::1],i2[::1]))


@njit(cache=True)
def _conds_structure_map(c_a, c_b, fix_same_var, fix_same_alias):
    Da, Db = len(c_a.dnf), len(c_b.dnf)
    N, M = len(c_a.vars), len(c_b.vars)

    # Fixed indicies -2 for unassigned -1 for no valid assignment
    a_fixed_inds = get_fixed_inds(c_a, c_b, fix_same_var, fix_same_alias)
    lit_groups = make_lit_groups(c_a, c_b)
    
    fixed_inds = a_fixed_inds.copy()
    remaps = List.empty_list(align_remap_type)
    iter_stack = List.empty_list(stack_item_type)
    it = None
    c = 0
    score, best_score, score_bound = u2(0), u2(0), u2(0)
    best_result = None


    # Outer loop handles generation of iterators over ambiguous
    #  variable assignments. 
    while(True):
        # Inner loop recalcs score matricies, from current fixed_inds.
        #  Loops multiple times if new scores imply some previously
        #  unfixed variable now has an unambiguous mapping.
        while(True):
            # Recalculate the score matrix w/ fixed_inds
            remap_score_matrices, beta_score_matrices = (
                _calc_remap_score_matrices(
                    lit_groups, (Da, Db, N, M), fixed_inds
            ))

            # Find the alignment and cumulative score matrix. Required
            #  for when either condition is disjoint (i.e. has an OR()).
            alignment, cum_score_matrix, cum_beta_matrix = (
                _get_best_alignment(remap_score_matrices, beta_score_matrices)
            )
            # print(cum_score_matrix)
            # print(alignment, fixed_inds)

            # Look for unambiguous remaps in the new matrix
            fixed_inds, unamb_cnt = get_unambiguous_inds(cum_score_matrix, fixed_inds)
            
            if(unamb_cnt == 0):
                break

        # print("new_unambiguious", fixed_inds, unamb_cnt)
        
        score_bound = score_remap(
            np.argmax(cum_score_matrix, axis=1),
            cum_score_matrix, cum_beta_matrix
        )

        # print(f"BEST={best_score}", f"BOUND={score_bound}")
        backtrack = False

        # Case: Abandon if the upper bound on the current assignment's 
        #   score is less than the current best score. 
        if(score_bound < best_score):
            backtrack = True
        

        # Case: All vars fixed so store remap. Then backtrack. 
        elif(np.all(fixed_inds != -2)):

            # Future NOTE: If remap is recalculated w/ 
            #  align_greedy then unmatched symbols are dropped, 
            #  but they are kept if just use fixed_inds.
            # remap = align_greedy(cum_score_matrix)
            remap = fixed_inds.copy()
            remaps.append((score_bound, alignment, remap))
            score, keep_mask_a, keep_mask_b = _score_and_mask(
                c_a, c_b, lit_groups, alignment, remap)

            # print(cum_score_matrix)
            # print("IS SAME!", np.array_equal(fixed_inds, remap), fixed_inds, remap)

            if(score > best_score):
                best_result = (alignment, remap, keep_mask_a, keep_mask_b)
                best_score = score

            backtrack = True

        if(backtrack):
            if(len(iter_stack) == 0):
                # Case: All iterators exhausted (i.e. Finished)
                break

            while(len(iter_stack) > 0):
                i, c, js, old_fixed_inds = iter_stack.pop()
                # print("POP", i, c, js, old_fixed_inds)
                fixed_inds = old_fixed_inds.copy()
                fixed_inds[i] = js[c]
                c += 1
                if(c < len(js)):
                    iter_stack.append((i, c, js, old_fixed_inds))
                    # print("PUSH", i, c, js, old_fixed_inds)
                    break

            # Case: fixed_inds has been set by popping from stack
        else:
            # Case: some assignments ambiguous so make next iter.
            #  'fixed_inds' is set to the first choice of i -> j.
            #  Iterator for rest are pushed to stack. 
            (i,js) = get_best_ind_iter(cum_score_matrix, fixed_inds)
            iter_stack.append((i, 1, js, fixed_inds.copy()))
            fixed_inds[i] = js[0]
            # print("PUSH", i, 1, js, fixed_inds.copy())

    if(best_result is not None):
        alignment, remap, keep_mask_a, keep_mask_b = best_result
        return best_score, alignment, remap, keep_mask_a, keep_mask_b


# -------------------------------
# : Why Not for structure mapping

@njit(cache=True)
def _insert_var_why_nots(conds, remap, why_nots, wn_len):
    for i, j in enumerate(remap):
        if(j == -1):
            why_nots[wn_len] = new_why_not(
                 cast(conds.vars[i], i8), i,
                 kind=WN_NOT_MAP_VAR
            )
            wn_len += 1
    return wn_len

@njit(cache=True)
def _insert_lit_why_nots(conj, keep_mask, d_ind, why_nots, wn_len):
    for c_ind in range(len(conj)):
        keep_it = keep_mask[c_ind]
        if(not keep_it):
            lit = conj[c_ind]
            why_nots[wn_len] = new_why_not(
                cast(lit, i8),
                lit.var_inds[0],
                var_ind1=-1 if len(lit.var_inds) < 2 else lit.var_inds[1],
                d_ind=d_ind, c_ind=c_ind,
            )
            wn_len += 1
    return wn_len

@njit(cache=True)
def _structure_map_make_why_nots(c_a, c_b, alignment, remap, keep_mask_a, keep_mask_b):
    wn_len_a, wn_len_b = 0, 0
    n_lits_a = sum([len(conj) for conj in c_a.dnf]) 
    n_lits_b = sum([len(conj) for conj in c_b.dnf]) 

    why_nots_a = np.empty(len(c_a.vars)+n_lits_a, dtype=why_not_type)
    why_nots_b = np.empty(len(c_b.vars)+n_lits_b, dtype=why_not_type)

    wn_len_a = _insert_var_why_nots(c_a, remap, why_nots_a, wn_len_a)
    inv_remap = -np.ones(len(c_b.vars),dtype=np.int16)
    for i, j in enumerate(remap):
        if(j != -1):
            inv_remap[j] = i
    wn_len_b = _insert_var_why_nots(c_b, inv_remap, why_nots_b, wn_len_b)

    for i, j in enumerate(alignment):
        conj_a, conj_b = c_a.dnf[i], c_b.dnf[j],
        wn_len_a = _insert_lit_why_nots(conj_a,
            keep_mask_a[i], i, why_nots_a, wn_len_a)
        wn_len_b = _insert_lit_why_nots(conj_b,
            keep_mask_b[j], j, why_nots_b, wn_len_b)

    return why_nots_a[:wn_len_a], why_nots_b[:wn_len_b]



# @njit
# def conds_structure_map(c_a, c_b, fix_same_var, fix_same_alias):
#     ls_as = conds_to_lit_sets(c_a)
#     ls_bs = conds_to_lit_sets(c_b)

#     # bpti_a = make_base_ptrs_to_inds(c_a)
#     # bpti_b = make_base_ptrs_to_inds(c_b)

#     a_fixed_inds = get_fixed_inds(c_a, c_b, fix_same_var, fix_same_alias)

#     alignment, remap, score = _structure_map(c_a, c_b, ls_as, ls_bs, a_fixed_inds)


# -------------------------------
# : Antiunify
@njit(cache=True)
def _conds_antiunify(c_a, c_b, fix_same_var, fix_same_alias):
    score, alignment, remap, keep_mask_a, keep_mask_b = \
        _conds_structure_map(c_a, c_b, fix_same_var, fix_same_alias)

    print("REMAP", remap)
    print("MA", keep_mask_a)
    print("MB", keep_mask_b)
    # why_nots_a, why_nots_b = _structure_map_make_why_nots(
    #     c_a, c_b, alignment, remap, keep_mask_a, keep_mask_b)

    # copy c_a, but omit any of the bad vars and literals
    dnf = List()
    for d_ind, conj in enumerate(c_a.dnf):
        new_conj = List.empty_list(LiteralType)
        for c_ind, lit in enumerate(conj):
            var_is_bad = False
            for v_ind in lit.var_inds:
                if(remap[v_ind] == -1):
                    var_is_bad = True
                    break
            
            if(var_is_bad or 
                not keep_mask_a[d_ind, c_ind]):
                continue
            new_lit = literal_copy(lit)
            new_conj.append(new_lit)
        dnf.append(new_conj)

    _vars = List([v for i, v in enumerate(c_a.vars) if remap[i] != -1])
    print(_vars)
    return _conditions_ctor_var_list(_vars, dnf), score
    # new_lit = literal_ctor(lit.op)
    # new_lit.negated = lit.negated


    
    

# @njit(Tuple((ConditionsType, i8))(ConditionsType, ConditionsType, boolean, boolean), boundscheck=True, cache=True)
# @njit(cache=True)
# def _conds_antiunify(c_a, c_b, fix_same_var, fix_same_alias):
#     ls_as = conds_to_lit_sets(c_a)
#     ls_bs = conds_to_lit_sets(c_b)

#     bpti_a = make_base_ptrs_to_inds(c_a)
#     bpti_b = make_base_ptrs_to_inds(c_b)

#     remap_size = len(bpti_a)
#     num_conj = len(ls_as)
#     best_score = 0
#     best_remap = np.arange(remap_size, dtype=np.int16)


#     a_fixed_inds = get_fixed_inds(c_a, c_b, fix_same_var, fix_same_alias)

#     remap_score_matrices = _calc_remap_score_matrices(
#         ls_as, ls_bs, bpti_a, bpti_b, a_fixed_inds
#     )

#     # Case 1: c_a and c_b are both single conjunctions like (lit1 & lit2 & lit3)
#     if(len(ls_as) == 1 and len(ls_bs) == 1):
#         op_key_intersection = intersect_keys(ls_as[0], ls_bs[0])

#         scores, best_remap = align_greedy(remap_score_matrices[0,0])
#         best_score = np.sum(scores)
#         conj, kept = _conj_from_litset_and_remap(ls_as[0], ls_bs[0], 
#                  best_remap, op_key_intersection, bpti_a, bpti_b)

#         dnf = List([conj])
#         conds = _conditions_ctor_dnf(dnf)

#     # Case 2: c_a or c_b have disjunction like ((lit1 & lit2 & lit3) | (lit4 & lit5))
#     else:
#         alignment, remap, best_score = \
#             _get_best_alignment_and_remap(remap_score_matrices)

#         kept = 0 
#         dnf = List()
#         # print("Align", alignment)
#         # print("Remap", remap)
#         for i, ls_a in enumerate(ls_as):
#             ls_b = ls_bs[alignment[i]]

#             op_key_intersection = intersect_keys(ls_a, ls_b)
#             conj, kpt = _conj_from_litset_and_remap(ls_a, ls_b,
#                      remap, op_key_intersection, bpti_a, bpti_b)
#             dnf.append(conj)
#             kept += kpt

#         conds = _conditions_ctor_dnf(dnf)
    

#     # Make sure base Vars are ordered like c_a
#     unordered_base_vars = conds.base_var_map
#     base_var_map = Dict.empty(i8,i8)
#     c = 0 
#     for base_ptr in bpti_a:
#         if(base_ptr in unordered_base_vars):
#             base_var_map[base_ptr] = c
#             c += 1
#     conds = _conditions_ctor_base_var_map(base_var_map, dnf)

#     # print("best_score", best_score)
#     # out_score = (f8(kept) / f8(total)) if total != 0 else 0.0
#     print("KEPT TOT", kept)
#     return conds, kept


AU_NORMALIZE_NONE = u1(0)
AU_NORMALIZE_LEFT = u1(1)
AU_NORMALIZE_RIGHT = u1(2)
AU_NORMALIZE_MAX = u1(3)

def resolve_normalize_enum(option):
    if(option == "" or option == "none"):
        norm_enum = AU_NORMALIZE_NONE
    elif(option == "left"):
        norm_enum = AU_NORMALIZE_LEFT
    elif(option == "right"):
        norm_enum = AU_NORMALIZE_RIGHT
    elif(option == "max"):
        norm_enum = AU_NORMALIZE_MAX
    else:
        raise ValueError(f"Unknown Normalization option {option}")
    return norm_enum



@njit(cache=True)
def conds_antiunify(c_a, c_b, normalize=1,
         fix_same_var=False, fix_same_alias=False):
    c, kept = _conds_antiunify(c_a, c_b, fix_same_var, fix_same_alias)
    if(normalize == AU_NORMALIZE_LEFT):
        n_total_a = len(c_a.vars) + count_literals(c_a)
        print("N TOTAL", n_total_a)
        return c, kept / n_total_a
    elif(normalize == AU_NORMALIZE_RIGHT):
        n_total_b = len(c_b.vars) + count_literals(c_b)
        return c, kept / n_total_b
    elif(normalize == AU_NORMALIZE_MAX):
        n_total_a = len(c_a.vars) + count_literals(c_a)
        n_total_b = len(c_b.vars) + count_literals(c_b)
        return c, kept / max(max(n_total_a, n_total_b),1)
    else:
        return c, f8(kept)

@overload_method(ConditionsTypeClass, 'antiunify')
def overload_conds_antiunify(c_a, c_b, return_score=False,
            normalize='left', fix_same_var=False, fix_same_alias=False):
    SentryLiteralArgs(['return_score','normalize']).for_function(
                    conds_antiunify).bind(c_a, c_b, return_score, normalize)

    if(not return_score.literal_value):
        def impl(c_a, c_b, return_score=False,
                normalize='left', fix_same_var=False, fix_same_alias=False):
            c, score = conds_antiunify(c_a, c_b, AU_NORMALIZE_NONE, fix_same_var, fix_same_alias)
            return c
    else:
        norm_enum = resolve_normalize_enum(normalize.literal_value)
        def impl(c_a, c_b, return_score=False,
                normalize='left', fix_same_var=False, fix_same_alias=False):
            c, score = conds_antiunify(c_a, c_b, norm_enum, fix_same_var, fix_same_alias)
            return c, score
    return impl

        

# -----------------------------------------------------------------------
# : Conditions.from_facts

def _add_adjacent(src_fact, _, fact, src_attr_var, conds,
         fact_ptr_map, attr_flags, weight=1.0, add_back_relation=False, var_prefix=None):
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
        for attr, config in fact._fact_type.filter_spec(attr_flags).items():
            attr_val = getattr(fact, attr)
            if(not isinstance(attr_val, FactProxy) or
                attr_val.get_ptr() != src_ptr):
                continue
            attr_var = getattr(fact_var, attr)

            lit = attr_var==src_fact_var
            lit.set_weight(weight)
            _conds.append(lit)

    # Add a condition from val to 
    lit = src_attr_var==fact_var
    lit.set_weight(weight)
    _conds.append(lit)
    conds[fact_ptr] = _conds


from itertools import chain

def flags_and(l_flags, r_flags):
    dnf = [[] for i in range(len(l_flags)*len(r_flags))]
    for i, l_conjunct in enumerate(l_flags):
        for j, r_conjunct in enumerate(r_flags):
            k = i*len(r_flags) + j
            for x in l_conjunct: dnf[k].append(x)
            for x in r_conjunct: dnf[k].append(x)
    return dnf

def flags_not(flags):
    dnfs = []
    for i, conjunct in enumerate(flags):
        dnf = [[] for i in range(len(conjunct))]
        for j, flag in enumerate(conjunct):
            if(flag[0] == "~"):
                flag = flag[1:]
            else:
                flag = f"~{flag}"
            dnf[j].append(flag)
        dnfs.append(dnf)

    out_dnf = dnfs[0]
    for i in range(1,len(dnfs)):
        out_dnf = flags_and(out_dnf,dnfs[i])
    return out_dnf



def conditions_from_facts(facts, _vars=None, add_neighbors=True,
     add_neighbor_holes=False, neighbor_back_relation=False, neighbor_req_n_adj=1, 
     alpha_weight=1.0, beta_weight = 1.0, neighbor_alpha_weight = 1.0, neighbor_beta_weight = 1.0, 
     alpha_flags=[('visible',)], parent_flags=[('parent',)], beta_flags=[('relational',)]):
    
    if(_vars is None):
        _vars = [Var(x._fact_type, f'A{i}') for i, x in enumerate(facts)]

    fact_ptr_map = {fact.get_ptr() : var for fact, var in zip(facts, _vars)}
    inp_fact_ptrs = set(fact_ptr_map.keys())
    
    # Make flag sets based on 
    beta_not_parent_flags =  flags_and(beta_flags, flags_not(parent_flags))
    alpha_candidate_flags = alpha_flags
    # print("::", alpha_candidate_flags)
    if(add_neighbor_holes):
        alpha_candidate_flags = flags_and(alpha_flags, flags_and(parent_flags, beta_flags))
        # alpha_candidate_flags = [*alpha_flags, *parent_flags, *beta_flags]
    
    
    cond_set = []
    nbr_conds = {}
    parent_conds = {}
    beta_conds = {}
    for fact, var in zip(facts,_vars):
        # Make Alphas (i.e. nvar = 1)
        for attr, config in fact._fact_type.filter_spec(alpha_candidate_flags).items():
            # print("attr", attr)
            val = getattr(fact, attr)
            if(isinstance(val, FactProxy)): continue
            attr_var = getattr(var, attr)

            lit = Conditions(attr_var==val)
            lit.set_weight(alpha_weight)
            cond_set.append(lit)

        # TODO: Make parents 
        nxt_parents = [fact]
        while(len(nxt_parents) > 0):
            for nxt_parent in nxt_parents:
                spec = nxt_parent._fact_type.filter_spec(parent_flags)
                for attr, config in spec.items():
                    pass
            nxt_parents = []

        # Make Betas (i.e. n_var=2) + Neighbors
        for attr, config in fact._fact_type.filter_spec(beta_not_parent_flags).items():
            attr_val =  getattr(fact, attr)
            if(not isinstance(attr_val, FactProxy)): continue
            attr_var = getattr(var, attr)

            # Add a beta between the input facts.
            if(attr_val.get_ptr() in inp_fact_ptrs):
                _add_adjacent(fact, var, attr_val, attr_var, beta_conds,
                        fact_ptr_map, beta_not_parent_flags, weight=beta_weight,
                        var_prefix=None)

            # Add a beta between an input fact and a non-input neighbor.
            elif(add_neighbors):
                _add_adjacent(fact, var, attr_val, attr_var, nbr_conds,
                 fact_ptr_map, beta_not_parent_flags, weight=neighbor_beta_weight,
                 add_back_relation=neighbor_back_relation, var_prefix="Nbr")

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

    # print("----------------------")
    # print(conds)
    # print("----------------------")
    return conds

# ----------------------------------
# : Replace

@njit(cache=True)
def conditions_replace(conds, d, c, lit):
    conds.has_distr_dnf = False
    conds.dnf[d][c] = lit
    return conds

@njit(cache=True)
def conditions_replace_multi(conds, replacements):
    conds.has_distr_dnf = False
    for (d, c, lit) in replacements:
        conds.dnf[d][c] = lit
    return conds

@njit(cache=True)
def conditions_remove(conds, d, c):
    conds.has_distr_dnf = False    
    conj = conds.dnf[d]
    del conj[c]
    return conds

@njit(cache=True)
def conditions_remove_multi(conds, indicies):
    # Sort to avoid index invalidation 
    indicies = sorted(indicies, reverse=True)
    conds.has_distr_dnf = False
    for (d, c) in indicies:
        conj = conds.dnf[d]
        del conj[c]
    return conds

@njit(cache=True)
def conditions_remove_var(conds, var_ind):
    var = conds.vars[var_ind]

    # Collect indicies of literals that use this var
    remove_indicies = List.empty_list(i8_pair)
    for d, conj in enumerate(conds.dnf):
        for c, lit in enumerate(conj):
            lit_okay = True
            for ptr in lit.base_var_ptrs:
                other_var = cast(ptr, VarType)
                if(var == other_var):
                    lit_okay = False
                    break

            if(not lit_okay):
                remove_indicies.append((d,c))


    # Make a new instance with these removed
    conds = conditions_remove_multi(conds, remove_indicies)
    v_ptr = cast(var, i8)
    
    # Remove from conds.vars and base_var_map
    del conds.vars[var_ind]
    conds.base_var_map = build_base_var_map(conds.vars)
    distr_dnf = conds_get_distr_dnf(conds)

    
    return conds



@njit(cache=True)
def conditions_indicies_of(conds, lit):
    for d, conj in enumerate(conds.dnf):
        for c, _lit in enumerate(conj):
            if(lit == _lit):
                return d, c
    return (-1,-1)


i8_pair = Tuple((i8,i8))
@njit(cache=True)
def conditions_all_indicies_of(conds, lit):
    indices = List.empty_list(i8_pair)
    for d, conj in enumerate(conds.dnf):
        for c, _lit in enumerate(conj):
            if(lit == _lit):
                indices.append((d,c))
    return indices




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
#             i = conds.base_var_map[term.base_var_ptrs[0]]
#             alpha_conjuncts[i].append(term)

        

#         beta_inds = -np.ones((n_vars,n_vars),dtype=np.int64)
#         for term in bc:
#             i = conds.base_var_map[term.base_var_ptrs[0]]
#             j = conds.base_var_map[term.base_var_ptrs[1]]
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
#             ptr = term.base_var_ptrs[0]
#             # ptr = cast(l_var, i8)
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
