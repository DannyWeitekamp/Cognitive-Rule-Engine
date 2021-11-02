from numba import njit, types, u8, i8, f8, generated_jit, cfunc
from numba.typed import List
from numba.types import unicode_type
from numba.types import CompileResultWAP
from cre.fact import base_fact_field_dict, BaseFactType, FactProxy, Fact
from cre.fact_intrinsics import fact_lower_setattr, _register_fact_structref
from cre.cre_object import CREObjType, CREObjProxy, CREObjTypeTemplate
from cre.utils import _struct_from_ptr, _obj_cast_codegen, _cast_structref, _ptr_from_struct_incref
from numba.core.imputils import (lower_cast)
from numba.experimental.structref import new, define_boxing, define_attributes
from numba.experimental.function_type import _get_wrapper_address
from numba.core.extending import overload


#CHANGE THIS
PRIMITIVE_T_ID = 0XFF


primitive_field_dict = {
    **base_fact_field_dict,
    "value" : types.Any
}

@_register_fact_structref
class PrimitiveTypeTemplate(Fact):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        typ = self._fields[-1][1]
        self._fact_name = f'cre.Primitive[{str(typ)}]'

    def __str__(self):
        return self._fact_name

def as_tup_list(dct):
    return [(k,v) for k,v in dct.items()]

# print(as_tup_list({**primitive_field_dict,'value' : types.boolean}))
BooleanPrimitiveType = PrimitiveTypeTemplate(as_tup_list({**primitive_field_dict,'value' : types.boolean}))
IntegerPrimitiveType = PrimitiveTypeTemplate(as_tup_list({**primitive_field_dict,'value' : i8}))
FloatPrimitiveType = PrimitiveTypeTemplate(as_tup_list({**primitive_field_dict,'value' : f8}))
StringPrimitiveType = PrimitiveTypeTemplate(as_tup_list({**primitive_field_dict,'value' : types.unicode_type}))

define_attributes(BooleanPrimitiveType)
define_attributes(IntegerPrimitiveType)
define_attributes(FloatPrimitiveType)
define_attributes(StringPrimitiveType)

@lower_cast(PrimitiveTypeTemplate, BaseFactType)
@lower_cast(PrimitiveTypeTemplate, CREObjType)
def downcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)


@njit(types.boolean(CREObjType, CREObjType),cache=True)
def int_primitive_eq(a, b):
    return _cast_structref(IntegerPrimitiveType,a).value == _cast_structref(IntegerPrimitiveType,b).value

# int_primitive_eq = list(int_primitive_eq.overloads.values())[0].entry_point
# print(int_primitive_eq)

@njit(u8(CREObjType,),cache=True)
def int_primitive_hash(a):
    return hash(_cast_structref(IntegerPrimitiveType,a).value)

# int_primitive_hash = list(int_primitive_hash.overloads.values())[0].entry_point

@njit(unicode_type(CREObjType,),cache=True)
def int_primitive_str(a):
    return str(_cast_structref(IntegerPrimitiveType,a).value)

# int_primitive_str = list(int_primitive_str.overloads.values())[0].entry_point

from cre.cre_object import new_cre_obj_method_table, eq_fn_typ, hash_fn_typ, str_fn_typ

mt = new_cre_obj_method_table(
    _get_wrapper_address(int_primitive_eq, eq_fn_typ.signature),
    _get_wrapper_address(int_primitive_hash, hash_fn_typ.signature),
    _get_wrapper_address(int_primitive_str, str_fn_typ.signature),
    _get_wrapper_address(int_primitive_str, str_fn_typ.signature),
)

@njit(cache=True)
def get_mt_ptr(mt):
    return _ptr_from_struct_incref(mt)

mt_ptr = get_mt_ptr(mt)
# @njit(unicode_type(CREObjType,),cache=True)
# def int_primitive_repr(a):
#     return repr(_cast_structref(IntegerPrimitiveType,a).value)


@generated_jit(cache=True)
def primitive_ctor(x, mt_ptr):
    if(x is types.boolean):
        def impl(x, mt_ptr):
            st = new(BooleanPrimitiveType); fact_lower_setattr(st,'value',x);
            return st

    elif(x is i8):
        def impl(x, mt_ptr):
            st = new(IntegerPrimitiveType);
            # mt_ptr = _ptr_from_struct_incref(mt)
                
            fact_lower_setattr(st,'value',x);
            fact_lower_setattr(st,'method_table_ptr',mt_ptr);
            # fact_lower_setattr(st,'__eq__', mt_ptr);
            # fact_lower_setattr(st,'__hash__', mt_ptr);
            # fact_lower_setattr(st,'__str__', mt_ptr);
            # fact_lower_setattr(st,'__repr__', mt_ptr);
            return st

    elif(x is f8):
        def impl(x, mt_ptr):
            st = new(FloatPrimitiveType); fact_lower_setattr(st,'value',x);
            return st

    elif(x is types.unicode_type):
        def impl(x, mt_ptr):
            st = new(StringPrimitiveType); fact_lower_setattr(st,'value',x);
            return st
    # else:
        # raise ValueError(f"Primitive Type not recognized {}.})
    return impl


class Primitive(CREObjProxy):
    __numba_ctor = primitive_ctor
    # _fact_type = PredType
    # _fact_name = 'Primitive'

    def __new__(cls, value):
        self = primitive_ctor(value, mt_ptr)
        return self

    def __str__(self):
        return f'{self.value}'

    def __repr__(self):
        return f'cre.Primitive(value={self.value})'

    @property
    def value(self):
        return primitive_get_value(self)

@njit(cache=True)
def primitive_get_value(self):
    return self.value


define_boxing(PrimitiveTypeTemplate, Primitive)
overload(Primitive)(primitive_ctor)








