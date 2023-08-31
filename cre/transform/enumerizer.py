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
from cre.var import Var, VarType
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
from numba.core.typing.typeof import typeof


ptr_t_pair = Tuple((ptr_t,ptr_t))
enumerizer_fields = {
    # For each t_id a pair of dictionaries: 1) maps values to u8 enums 2) maps u8 enums to values.
    "val_maps" : DictType(u2, ptr_t_pair),
}

@structref.register
class EnumerizerTypeClass(CastFriendlyStructref):
    pass

EnumerizerType = EnumerizerTypeClass([(k,v) for k,v in enumerizer_fields.items()])

@njit(cache=True)    
def enumerizer_ctor():
    st = new(EnumerizerType)
    st.val_maps = Dict.empty(u2, ptr_t_pair)
    return st
    
class Enumerizer(structref.StructRefProxy):
    def __new__(cls):
        self = enumerizer_ctor()
        return self

    def dict_for_type(self,val_type):
        return dicts_for_type(self,val_type)

    def to_enum(self, val, d=None, inv_d=None):
        return to_enum(self, val, d, inv_d)

    def from_enum(self, enum, val_type, d=None, inv_d=None):
        return from_enum(self, enum, val_type, d, inv_d)


# # Classes that mimic behavior of a dictionary 

# class EnumerizerMap:
#     def __init__(self, enumerizer):
#         self.enumerizer = enumerizer

#     def __getitem__(self, val):
#         self.enumerizer.to_enum(val)

# class EnumerizerInvMapper(EnumerizerMap):
#     def __init__(self, enumerizer):
#         self.enumerizer = enumerizer

#     def __call__(self, enum, val_type):
        

#     def __getitem__(self, enum, val_type):
#         self.enumerizer.from_enum(enum, val_type)



define_boxing(EnumerizerTypeClass, Enumerizer)

@generated_jit(cache=True,nopython=True)
@overload_method(EnumerizerTypeClass, 'dicts_for_type')
def dicts_for_type(self, val_type):
    val_type = val_type.instance_type
    context = CREContext.get_default_context()
    t_id = u2(context.get_t_id(_type=val_type))
    d_type = DictType(val_type,u8)
    inv_d_type = DictType(u8, val_type)
    def impl(self, val_type):
        if(t_id not in self.val_maps):
            d = Dict.empty(val_type,u8)
            inv_d = Dict.empty(u8, val_type)
            d_ptr, inv_d_ptr = _ptr_from_struct_incref(d), _ptr_from_struct_incref(inv_d)
            self.val_maps[t_id] = (d_ptr, inv_d_ptr)

        d_ptr, inv_d_ptr = self.val_maps[t_id]
        d = _dict_from_ptr(d_type, d_ptr)
        inv_d = _dict_from_ptr(inv_d_type, inv_d_ptr)
        return d, inv_d
    return impl

# @generated_jit(cache=True,nopython=True)
# @overload_method(EnumerizerTypeClass, 'inv_dict_for_type')
# def inv_dict_for_type(self, val_type):
#     val_type = val_type.instance_type
#     context = CREContext.get_default_context()
#     t_id = u2(context.get_t_id(_type=val_type))
#     d_type = DictType(u8, val_type)
#     def impl(self, val_type):
#         if(t_id not in self.inv_val_maps):
#             d = Dict.empty(u8, val_type)
#             self.inv_val_maps[t_id] = _ptr_from_struct_incref(d)
#         d = _dict_from_ptr(d_type, self.inv_val_maps[t_id])
#         return d
#     return impl

@generated_jit(cache=True,nopython=True)
@overload_method(EnumerizerTypeClass, 'to_enum')
def to_enum(self, val, d=None, inv_d=None):
    val_type = val
    def impl(self, val, d=None, inv_d=None):
        if(d is None or inv_d is None):
            d,inv_d = dicts_for_type(self, val_type)
        if(val not in d):
            enum = u8(len(d)+1)
            d[val] = enum
            inv_d[enum] = val
        return d[val]
    return impl

@generated_jit(cache=True,nopython=True)
@overload_method(EnumerizerTypeClass, 'from_enum')
def from_enum(self, enum, val_type, d=None, inv_d=None):
    # val_type = val_type.instance_type
    def impl(self, enum, val_type, d=None, inv_d=None):
        if(d is None or inv_d is None):
            d, inv_d = dicts_for_type(self, val_type)

        if(enum not in inv_d):
            raise ValueError("No value found for enum.")

        val = inv_d[enum]
        return val
    return impl

# @



# e = Enumerizer()
# print(e.enumerize(6.))
# print(e.enumerize(8.))
# print(e.enumerize(1.))
# print(e.enumerize(1.))
# print(e.enumerize(True))
# print(e.enumerize("A"))

