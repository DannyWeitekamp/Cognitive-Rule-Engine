import operator
import numpy as np
import numba
from numba import types, njit, i8, u8, i4, u1,u2,u4,  i8, literally, generated_jit, objmode
from numba.typed import List
from numba.core.types import ListType, unicode_type, void, Tuple
from numba.experimental import structref
from numba.experimental.structref import new, define_attributes, _Utils
from numba.extending import SentryLiteralArgs, lower_cast, overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from cre.context import cre_context
from cre.structref import define_structref, define_boxing, define_structref_template, CastFriendlyStructref
from cre.fact import define_fact, BaseFact, cast_fact, DeferredFactRefType, Fact, _standardize_type
from cre.utils import PrintElapse, ptr_t, _struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_ptr,  lower_setattr, lower_getattr, _raw_ptr_from_struct, _decref_ptr, _incref_ptr, _incref_structref, _ptr_from_struct_incref
from cre.utils import assign_to_alias_in_parent_frame, encode_idrec, _obj_cast_codegen
from cre.vector import VectorType
from cre.cre_object import cre_obj_field_dict,CREObjType, CREObjTypeClass, CREObjProxy, set_chr_mbrs
# from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
 # get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models


from operator import itemgetter
from copy import copy
from os import getenv
from cre.utils import deref_info_type, DEREF_TYPE_ATTR, DEREF_TYPE_LIST, listtype_sizeof_item, _obj_cast_codegen
from cre.core import T_ID_RULE, register_global_default
from cre.conditions import ConditionsType, Conditions
from cre.memset import MemSetType
import cloudpickle
import inspect

rhs_ptrs_type = types.FunctionType(types.void(MemSetType, i8[::1]))

rule_fields_dict = {
    **cre_obj_field_dict,
    'lhs' : ConditionsType,
    'decl_rhs' : types.optional(ConditionsType),
    'imper_rhs' : types.optional(rhs_ptrs_type),
}


class RuleTypeClass(CREObjTypeClass):
    t_id = T_ID_RULE

    def __str__(self):
        return "cre.RuleType"

default_manager.register(RuleTypeClass, models.StructRefModel)
define_attributes(RuleTypeClass)

RuleType = RuleTypeClass([(k,v) for k,v in rule_fields_dict.items()])
register_global_default("Rule", RuleType)


class RuleMeta(type):
    def __call__(cls, *args, **kwargs):
        lhs = args[0]
        if(len(args)==1):
            def wrapper(rhs_func):
                rule = new_rule(lhs, rhs_func)
                return rule
            return wrapper
        else:
            return new_rule(*args)


class Rule(CREObjProxy,metaclass=RuleMeta):
    t_id = T_ID_RULE
    def __new__(cls, lhs, rhs):
        return new_rule(lhs, rhs)
    def __str__(self):
        return f"Rule(?)"
    def __repr__(self):
        return f"Rule(?)"
        # print("Rule.__new__", lhs, rhs)


define_boxing(RuleTypeClass,Rule)


@njit(RuleType(ConditionsType),cache=True)
def rule_ctor(lhs):
    st = new(RuleType)
    st.lhs = lhs
    return st

def gen_rhs_source(match_types, rhs_pyfunc):
    sig = types.void(MemSetType, *match_types)

    source = \
f'''from numba import njit, i8 
from numba import types
from cre.utils import _struct_tuple_from_pointer_arr
from cre.memset import MemSetType
import cloudpickle
# from numba.experimental.function_type import _get_wrapper_address

match_types = cloudpickle.loads({cloudpickle.dumps(match_types)})
rhs_pyfunc = cloudpickle.loads({cloudpickle.dumps(rhs_pyfunc)})

print(match_types)
signature = types.void(MemSetType, *match_types)
print(signature)
rhs = njit(signature, cache=True)(rhs_pyfunc)

@njit(types.void(MemSetType, i8[::1]), cache=True)
def rhs_ptrs(wm, ptrs):
    args = _struct_tuple_from_pointer_arr(match_types, ptrs)
    rhs(wm, *args)

# rhs_ptrs_addr = _get_wrapper_address(rhs_ptrs, types.void(MemSetTypem, i8[::1]))

'''
    return source


def new_rule(lhs, rhs):
    with PrintElapse("rule_ctor"):
        st = rule_ctor(lhs)

    if(isinstance(rhs, Conditions)):
        print("!", rhs)

    elif(hasattr(rhs,'__call__')):
        with PrintElapse("make source"):
            match_types = tuple([x.base_type for x in lhs.vars])
            func_src = inspect.getsource(rhs)
            long_hash = unique_hash([match_types, func_src])
            if(not source_in_cache('Rule_RHS', long_hash)):
                source = gen_rhs_source(match_types, rhs)
                source_to_cache('Rule_RHS', long_hash, source)

        with PrintElapse("run"):
            rhs_ptrs = import_from_cached('Rule_RHS', long_hash, ['rhs_ptrs'])['rhs_ptrs']
            print(list(rhs_ptrs.overloads.keys()))
        with PrintElapse("assign"):
            rule_assign_imper_rhs(st, rhs_ptrs)
            st._rhs_pyfunc = rhs
        return st


@njit(cache=True)
def rule_assign_decl_rhs(st, decl_rhs):
    st.decl_rhs = decl_rhs

@njit(types.void(RuleType, rhs_ptrs_type),cache=True)
def rule_assign_imper_rhs(st, imper_rhs):
    st.imper_rhs = imper_rhs

