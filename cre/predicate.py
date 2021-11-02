from numba import njit, i8, u8, types
from numba.typed import List
from numba.types import ListType
from cre.fact import base_fact_field_dict, BaseFactType, FactProxy, Fact
from cre.fact_intrinsics import fact_lower_setattr, _register_fact_structref
from cre.cre_object import CREObjType, CREObjProxy
from cre.utils import _struct_from_ptr, _obj_cast_codegen
from cre.primitive import Primitive
from numba.core.imputils import (lower_cast)
from numba.experimental.structref import new, define_boxing, define_attributes, StructRefProxy
from numba.core.extending import overload


#CHANGE THIS
PRED_T_ID = 0XFF


predicate_field_dict = {
    **base_fact_field_dict,
    "header": CREObjType,
    "members": ListType(CREObjType),
}

predicate_fields = [(k,v) for k,v in predicate_field_dict.items()]

@_register_fact_structref
class PredTypeTemplate(Fact):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._fact_name = "cre.PredType"

    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

    def __str__(self):
        return "cre.PredType"

PredType = PredTypeTemplate(predicate_fields)
# print("<<", PredType, repr(PredType), type(PredType))
define_attributes(PredType)

@lower_cast(PredTypeTemplate, BaseFactType)
@lower_cast(PredTypeTemplate, CREObjType)
def downcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)

@njit(ListType(CREObjType)(),cache=True)
def init_members():
    return List.empty_list(CREObjType)

@njit(types.void(ListType(CREObjType),i8),cache=True)
def add_member_from_ptr(members, ptr):
    members.append(_struct_from_ptr(CREObjType,ptr))

# @njit(CREObjType(i8),cache=True)
# def cre_object_from_ptr(ptr):
#     return _struct_from_ptr(CREObjType,ptr)

# Not sure why giving signatures produces wrong type
# @njit(PredType(CREObjType, ListType(CREObjType)),cache=False)
@njit(cache=True)
def pred_ctor(header, members):
    st = new(PredType)
    fact_lower_setattr(st,'idrec',u8(-1))
    fact_lower_setattr(st,'header',header)
    fact_lower_setattr(st,'members',members)
    return st

class Pred(CREObjProxy):
    __numba_ctor = pred_ctor
    # _fact_type = PredType
    # _fact_name = 'Pred'

    def __new__(cls, header, *args):
        members = init_members()
        for arg in args: 
            if(isinstance(arg, (FactProxy, CREObjProxy))):
                add_member_from_ptr(members, arg.get_ptr())
            else:
                prim = Primitive(arg)
                add_member_from_ptr(members, prim.get_ptr())

        if(isinstance(header, (FactProxy, CREObjProxy))):
            header = arg
        else:
            header = Primitive(arg)

        self = pred_ctor(header, members)
        return self

    def __str__(self):
        return f'{self.header}({self.members})'

    def __repr__(self):
        return str(self)

    @property
    def header(self):
        return pred_get_members(self)

    @property
    def members(self):
        return pred_get_members(self)



@njit(cache=True)
def pred_get_header(self):
    return self.header

@njit(cache=True)
def pred_get_members(self):
    return self.members

define_boxing(PredTypeTemplate, Pred)
overload(Pred)(pred_ctor)




