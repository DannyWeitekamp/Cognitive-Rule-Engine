import numpy as np
from numba import njit, generated_jit, types, literal_unroll, u8, i8, f8, u1, u2
from numba.types import unicode_type,  intp, Tuple,  Tuple, DictType, ListType
from numba.typed import Dict, List
from numba.experimental.structref import new
from cre.core import short_name
from cre.obj import CREObjType
from cre.fact import define_fact, UntypedFact, call_untyped_fact, BaseFact
from cre.fact_intrinsics import fact_lower_getattr, resolve_fact_getattr_type
from cre.tuple_fact import TupleFact, TF
from cre.context import cre_context, CREContext
# from cre.default_ops import Add, Subtract, Divide
from cre.var import Var, VarType, var_append_deref
# from cre.op import GenericOpType
from cre.utils import _dict_from_ptr, ptr_t, _func_from_address, _obj_cast_codegen, _func_from_address, _incref_structref, _ptr_from_struct_incref
from cre.structref import define_structref
from cre.transform.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor
from cre.structref import CastFriendlyStructref, define_boxing
from numba.experimental import structref
from numba.extending import overload_method, overload, lower_cast, SentryLiteralArgs
from numba.experimental.function_type import _get_wrapper_address
import cloudpickle
from cre.gval import get_gval_type, new_gval, gval as gval_type


enumerizer_fields = {
    # Maps a t_id to a Dictionary that maps a value to a u8.
    "val_maps" : DictType(u2, ptr_t)
}

@structref.register
class EnumerizerTypeClass(CastFriendlyStructref):
    pass

EnumerizerType = EnumerizerTypeClass([(k,v) for k,v in enumerizer_fields.items()])

@njit(cache=True)    
def enumerizer_ctor():
    st = new(EnumerizerType)
    st.val_maps = Dict.empty(u2,ptr_t)
    return st
    
class Enumerizer(structref.StructRefProxy):
    def __new__(cls):
        self = enumerizer_ctor()
        return self

    def dict_for_type(self,val_type):
        return dict_for_type(self,val_type)

    def enumerize(self,val,d=None):
        return enumerize(self,val,d=d)

define_boxing(EnumerizerTypeClass, Enumerizer)

@generated_jit(cache=True,nopython=True)
@overload_method(EnumerizerTypeClass, 'dict_for_type')
def dict_for_type(self, val_type):
    val_type = val_type.instance_type
    context = CREContext.get_default_context()
    t_id = u2(context.get_t_id(_type=val_type))
    d_type = DictType(val_type,u8)
    def impl(self, val_type):
        if(t_id not in self.val_maps):
            d = Dict.empty(val_type,u8)
            self.val_maps[t_id] = _ptr_from_struct_incref(d)
        d = _dict_from_ptr(d_type, self.val_maps[t_id])
        return d
    return impl

@generated_jit(cache=True,nopython=True)
@overload_method(EnumerizerTypeClass, 'enumerize')
def enumerize(self, val, d=None):
    val_type = val
    def impl(self, val, d=None):
        if(d is None):
            d = dict_for_type(self, val_type)
        if(val not in d):
            d[val] = len(d)+1
        return d[val]
    return impl


# e = Enumerizer()
# print(e.enumerize(6.))
# print(e.enumerize(8.))
# print(e.enumerize(1.))
# print(e.enumerize(1.))
# print(e.enumerize(True))
# print(e.enumerize("A"))

