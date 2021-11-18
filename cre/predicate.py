from numba import njit, u1,u2, i4, i8, u8, types, literal_unroll, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type
from cre.fact import base_fact_field_dict, BaseFactType, FactProxy, Fact
from cre.fact_intrinsics import fact_lower_setattr, _register_fact_structref
from cre.cre_object import CREObjType, CREObjProxy, CREObjTypeTemplate, member_info_type
from cre.utils import _struct_get_attr_offset, _sizeof_type, _struct_get_data_ptr, _load_ptr, _struct_get_attr_offset, _struct_from_ptr, _cast_structref, _obj_cast_codegen, encode_idrec, decode_idrec, _incref_structref, _get_member_offset
from cre.primitive import Primitive
from numba.core.imputils import (lower_cast)
from numba.experimental.structref import new, define_boxing, define_attributes, StructRefProxy
from numba.core.extending import overload, intrinsic


#CHANGE THIS
PRED_T_ID = 0XFF



predicate_field_dict = {
    **base_fact_field_dict,
    # "header": CREObjType,
    # "num_members": u1,
    # # The data offset of the "members" attribute (unpredictable because of layout alignment)
    # # "member_offsets" : types.UniTuple(u2,1),
    # "member_info" : types.UniTuple(member_info_type,1),
    "members": types.Any,
    "chr_mbrs_infos" : types.UniTuple(member_info_type,1),
}

predicate_fields = [(k,v) for k,v in predicate_field_dict.items()]

@_register_fact_structref
class PredTypeTemplate(Fact):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # self._fact_name = "cre.PredType2"

    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

    def __str__(self):
        return "cre.PredType"

GenericPredType = PredTypeTemplate(predicate_fields)
# print("<<", PredType, repr(PredType), type(PredType))
define_attributes(GenericPredType)

@lower_cast(PredTypeTemplate, BaseFactType)
@lower_cast(PredTypeTemplate, CREObjType)
def downcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)

# @njit(cache=True)
# def init_members():
#     return List.empty_list(CREObjType)

# @njit(types.void(ListType(CREObjType),i8),cache=True)
# def add_member_from_ptr(members, ptr):
#     members.append(_struct_from_ptr(CREObjType,ptr))

# @njit(CREObjType(i8),cache=True)
# def cre_object_from_ptr(ptr):
#     return _struct_from_ptr(CREObjType,ptr)

# Not sure why giving signatures produces wrong type
# @njit(PredType(CREObjType, ListType(CREObjType)),cache=False)

from cre.core import T_ID_PREDICATE
default_idrec  = encode_idrec(T_ID_PREDICATE,0,0xFF)
# print("<<", decode_idrec(default_idrec))

def _down_cast_helper(x):
    if(isinstance(x, CREObjTypeTemplate)):
        return CREObjType
    else:
        return x

from cre.core import T_ID_UNRESOLVED, T_ID_BOOL_PRIMITIVE, T_ID_INTEGER_PRIMITIVE, T_ID_FLOAT_PRIMITIVE, T_ID_STRING_PRIMITIVE, T_ID_PREDICATE 

from cre.cre_object import _resolve_t_id_helper
# def _resolve_t_id_helper(x):
#     if(isinstance(x, types.Boolean)):
#         return T_ID_BOOL_PRIMITIVE
#     elif(isinstance(x, types.Integer)):
#         return T_ID_INTEGER_PRIMITIVE
#     elif(isinstance(x, types.Float)):
#         return T_ID_FLOAT_PRIMITIVE
#     elif(x is types.unicode_type):
#         return T_ID_STRING_PRIMITIVE
#     return T_ID_UNRESOLVED
    

from numba.experimental.structref import _Utils, imputils
from numba.core import cgutils, utils as numba_utils

@intrinsic
def _pred_get_chr_mbrs_infos(typingctx, pred_type):
    '''get the base address of the struct pointed to by structref 'inst' '''
    
    # ind = ind_type.literal_value
    # print(ind)
    members_type = [v for k,v in pred_type._fields if k == 'members'][0]
    t_ids = [_resolve_t_id_helper(x) for x in members_type.types]

    count = members_type.count
    member_infos_out_type = types.UniTuple(member_info_type, count)
    # print(members_type.__dict__)


    def codegen(context, builder, sig, args):
        [pred,] = args

        utils = _Utils(context, builder, pred_type)

        baseptr = utils.get_data_pointer(pred)
        baseptr_val = builder.ptrtoint(baseptr, cgutils.intp_t)
        dataval = utils.get_data_struct(pred)
        index_of_members = dataval._datamodel.get_field_position("members")

        member_infos = []
        for i in range(count):
            member_ptr = builder.gep(baseptr, [cgutils.int32_t(0), cgutils.int32_t(index_of_members), cgutils.int32_t(i)], inbounds=True)
            member_ptr = builder.ptrtoint(member_ptr, cgutils.intp_t)
            offset = builder.trunc(builder.sub(member_ptr, baseptr_val), cgutils.ir.IntType(16))
            t_id = context.get_constant(u2, t_ids[i])
            member_infos.append(context.make_tuple(builder, member_info_type, (t_id, offset) ))

        ret = context.make_tuple(builder,member_infos_out_type, member_infos)
        return ret

    sig = member_infos_out_type(pred_type)
    return sig, codegen

# @njit
# def get_member_offsets(typingctx, pred, member_info):
#     _pred_get_member_offsets(pred)
    # for i,x in enumerate(literal_unroll(members)):
    #     I = literally(i)



@generated_jit(cache=True)
def pred_ctor(*members):
    member_types = types.Tuple(tuple([_down_cast_helper(x) for x in members[0].types]))
    member_info_tup_type = types.UniTuple(member_info_type,len(member_types))
    member_t_ids = tuple([_resolve_t_id_helper(x) for x in member_types])

    pred_d = {**predicate_field_dict,"members" : member_types,"chr_mbrs_infos":member_info_tup_type}
    pred_type = PredTypeTemplate([(k,v) for k,v in pred_d.items()])

    

    def impl(*members):
        st = new(pred_type)
        fact_lower_setattr(st, 'idrec', default_idrec)
        fact_lower_setattr(st, 'num_identity_members', len(members))
        fact_lower_setattr(st, 'chr_mbrs_infos', _pred_get_chr_mbrs_infos(st))
        fact_lower_setattr(st, 'chr_mbrs_infos_offset', _get_member_offset(st,'chr_mbrs_infos'))
        # print(_pred_get_chr_mbrs_infos(st))
        # chr_mbrs_infos = id_members_info_ctor(_pred_get_chr_mbrs_infos(st))
        # fact_lower_setattr(st, 'chr_mbrs_infos', chr_mbrs_infos)
        # _pred_get_chr_mbrs_infos(st)
        fact_lower_setattr(st, 'members', members)
        # print("**",  _struct_get_attr_offset(st,"members"), _struct_get_attr_offset(st,"member_info") + st.length)
        return st
    return impl

class Pred(CREObjProxy):
    __numba_ctor = pred_ctor

    def __new__(cls, *args):
        self = pred_ctor(*args)
        return self

    def __str__(self):
        return f'{self.header}({self.members})'

    def __repr__(self):
        return str(self)

    # @property
    # def header(self):
    #     return pred_get_header(self)

    # @property
    # def members(self):
    #     return pred_get_members(self)


# @njit(cache=True)
# def pred_get_header(self):
#     return self.header

# @njit(cache=True)
# def pred_get_members(self):
#     return self.members

define_boxing(PredTypeTemplate, Pred)

@generated_jit(cache=True)
def assert_cre_obj(x):
    if(isinstance(x, types.Literal)): return
    if(isinstance(x, CREObjTypeTemplate)):
        def impl(x):
            return _cast_structref(CREObjType, x)
        return impl
    else:
        def impl(x):
            prim = Primitive(x)
            return _cast_structref(CREObjType, prim)
        return impl


@overload(Pred, prefer_literal=False,)
def _pred_ctor(*args):
    if(any([isinstance(x,types.Literal) for x in args])):
        return

    def impl(*args):
        return pred_ctor(*args)
    return impl

# Pred("HI", 1, 1)

# @njit
# def test_pred_item_offset():
#     print(_pred_get_member_offsets(Pred("HI", 1, 2, "HI", 3, 4)))

# test_pred_item_offset()

# raise ValueError()




# @njit(i8(GenericPredType), cache=True)
# def pred_get_length(x):
#     return x.length

# @njit(i8(GenericPredType), cache=True)
# def pred_get_member_info_ptr(x):
#     data_ptr = _struct_get_data_ptr(x.chr_mbrs_infos)
#     member_info_offset = _struct_get_attr_offset(x.chr_mbrs_infos,"data")
#     return _struct_get_data_ptr(x.chr_mbrs_infos) + member_info_offset

# @njit(i8(GenericPredType), cache=True)
# def pred_get_members_ptr(x):
#     l = i8(pred_get_length(x))
#     data_ptr = _struct_get_data_ptr(x)
#     # member_info_offset = _struct_get_attr_offset(x,"member_info")
#     return data_ptr + x.members_offset + l

# @njit(u8(GenericPredType, i8), cache=True)
# def pred_get_member_info(x, i):
#     member_info_ptr = pred_get_member_info_ptr(x)
#     return _load_ptr(member_info_type, member_info_ptr + i*3)

# @njit(u8(GenericPredType, i8), cache=True)
# def pred_get_item(x, typ, i):
#     member_info_ptr = pred_get_member_info_ptr(x)
#     return _load_ptr(u1, data_ptr + member_info_ptr + i)

# @njit(types.UniTuple(i8,2)(i8, i8), cache=True)
# def pred_get_next_ptrs(t_id_ptr, item_ptr):
#     t_id = _load_ptr(u1,t_id_ptr)
#     diff = _sizeof_type(i8) if t_id != T_ID_STRING_PRIMITIVE else _sizeof_type(unicode_type)
#     # print(t_id, diff)
#     return t_id_ptr+1, item_ptr+diff

@njit(cache=True)
def pred_iter_t_id_item_ptrs(_x):
    x = _cast_structref(GenericPredType,_x)
    data_ptr = _struct_get_data_ptr(x)
    # member_info_ptr = _struct_get_data_ptr(x.chr_mbrs_infos) + _struct_get_attr_offset(x.chr_mbrs_infos,"data")
    member_info_ptr = data_ptr + x.chr_mbrs_infos_offset
    
    for i in range(x.num_identity_members):
        t_id, member_offset = _load_ptr(member_info_type, member_info_ptr)
        yield t_id, data_ptr + member_offset
        member_info_ptr += _sizeof_type(member_info_type)
    # print()


# pred_iter_t_id_item_ptrs()


# print(">>", isinstance(2,CREObjType))



# @intrinsic 
# def 
