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
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
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
# import inspect

rule_fields_dict = {
    **cre_obj_field_dict,
    'lhs' : ConditionsType,
    'decl_rhs' : types.optional(ConditionsType),
    'imper_rhs' : types.optional(types.FunctionType(types.void(i8[::1]))),
}



class RuleTypeClass(CREObjTypeClass):
    t_id = T_ID_RULE

    def __str__(self):
        return "cre.RuleType"

# default_manager.register(RuleTypeClass, models.StructRefModel)

RuleType = VarTypeClass([(k,v) for k,v in rule_fields_dict.items()])
register_global_default("Rule", RuleType)


class RuleMeta(type):
    def __call__(*args):
        lhs = args[0]
        # if(isinstance(args[0], Conditions)):
        def wrapper(rhs_func):
            print("1:", lhs)
            print("2:", rhs_func)
            return st
        return wrapper
        # else:



        # if(hasattr(args[0],'__call__')):


class Rule(CREObjProxy,metaclass=RuleMeta):
    t_id = T_ID_RULE
    def __new__(cls, lhs, rhs):
        st = rule_ctor(lhs)

        if(isinstance(rhs, Conditions)):
            print("!", rhs)

        elif(hasattr(rhs,'__call__')):
            print("?", rhs)


        return st

    
            




@njit(cache=True)
def rule_ctor(lhs):
    st = new(RuleType)
    st.lhs = lhs
    return st

@njit(cache=True)
def rule_assing_decl_rhs(st, decl_rhs):
    st.decl_rhs = decl_rhs

@njit(cache=True)
def rule_assing_imper_rhs(st, imper_rhs):
    st.imper_rhs = imper_rhs






if __name__ == "__main__":
    from cre import define_fact, Var

    BOOP = define_fact("BOOP", {"A" : str, "B" : float})
    l1 = 



    @Rule() 
