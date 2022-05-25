import numpy as np
from numba import i8, u8, u2, u1, types, njit, generated_jit, literal_unroll
from numba.types import FunctionType, unicode_type, Tuple
from numba.extending import  overload, lower_getattr, overload_method
from cre.core import register_global_default, T_ID_UNDEFINED, T_ID_BOOL, T_ID_INT, T_ID_FLOAT, T_ID_STR, T_ID_TUPLE_FACT 
from cre.utils import (_memcpy_structref, _obj_cast_codegen, ptr_t,
    _raw_ptr_from_struct, _raw_ptr_from_struct_incref, _incref_ptr,
    CastFriendlyMixin, decode_idrec, _func_from_address,
    _cast_structref, _get_member_offset, _struct_get_data_ptr,
    _sizeof_type, _load_ptr, _struct_from_ptr, encode_idrec)
from cre.structref import define_structref
from numba.core.datamodel import default_manager, models
from numba.core import cgutils
from numba.experimental.structref import define_attributes, StructRefProxy, new, _Utils, define_boxing
from numba.experimental import structref
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
    intrinsic,
    lower_cast,
    # infer_getattr,
    # lower_getattr_generic,
    # lower_setattr_generic,
    box,
    unbox,
    NativeValue,
    intrinsic,
    overload,
)

import operator


member_info_type = types.Tuple((u2,u2,u2))

# identity_members_info_type = {
#     "data" : types.UniTuple(member_info_type,1),
# }

# class IdentityMemberInfos(StructRefProxy):
#     @property
#     def data(self):
#         return identity_members_info_get_data(self)
#     # def __init__(self, fields):
#     #     super().__init__(fields)

# @structref.register
# class IdentityMemberInfosType(types.StructRef):
#     pass
#     # def preprocess_fields(self, fields):
#     #     if(isinstance(fields, int)):
#     #         return [('data',types.UniTuple(member_info_type, n))]

# def id_member_info_type(n):
#     # if(n <= 0): n =1
#     return IdentityMemberInfosType([('data',types.UniTuple(member_info_type, n))])


# BaseIdentityMemberInfosType = id_member_info_type(1)   

# define_attributes(IdentityMemberInfosType)
# define_boxing(IdentityMemberInfosType, IdentityMemberInfos)

# @njit(cache=True)
# def identity_members_info_get_data(self):
#     return self.data


# @overload(IdentityMemberInfos)
# @generated_jit(cache=True)
# def id_members_info_ctor(data):

#     # if(not isinstance(data, types.UniTuple)): return
#     n = data.count    

#     typ = id_member_info_type(n)   
#     def impl(data):
#         st = new(typ)
#         st.data = data
#         return _cast_structref(BaseIdentityMemberInfosType,st) 

#     return impl



# print(id_members_info_ctor(((0,0),(1,2))).data)




# chr_mbrs_info
from numba.cpython.hashing import _Py_hash_t
cre_obj_field_dict = {
    "idrec" : u8,
    # The number of members in the CREObj 
    "hash_val" : _Py_hash_t,
    "num_chr_mbrs": u1,
    "chr_mbrs_infos_offset" : u2,
    # The data offset of the "members" attribute (unpredictable because of layout alignment)
    # "chr_mbrs_infos" : BaseIdentityMemberInfosType,#types.UniTuple(member_info_type,1),
}

cre_obj_fields = [(k,v) for k,v in cre_obj_field_dict.items()]


class CREObjModel(models.StructRefModel):
    pass

def impl_cre_obj_upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)

class CREObjTypeTemplate(CastFriendlyMixin, types.StructRef):
    def __init__(self, fields):
        types.StructRef.__init__(self,fields)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Allow numba type inferencer to upcast this type to CREObjType
        lower_cast(cls, CREObjType)(impl_cre_obj_upcast)
        

    def __str__(self):
        return f"cre.CREObjType"

    def hrepr():
        '''A human-readable repr for cre objects'''
        return f"{str(self)}(fields=({', '.join([f'{k}={str(v)}' for k,v in self._fields])}))" 


    # def __repr__(self):
    #     '''Make repr more readable than'''
    #     return f"{str(self)}(fields=({', '.join([f'{k}={str(v)}' for k,v in self._fields])}))" 


# @lower_cast(types.MemInfoPointer, types.Integer)
# def cast_meminfo_to_ptr(context, builder, fromty, toty, val):
#     return builder.ptrtoint(val, cgutils.intp_t)


CREObjType = CREObjTypeTemplate(cre_obj_fields) 
register_global_default("CREObj", CREObjType)
default_manager.register(CREObjTypeTemplate, CREObjModel)
define_attributes(CREObjType)


meminfo_type = types.MemInfoPointer(types.voidptr)
@njit(u2(meminfo_type),cache=True)
def get_t_id(ptr):
    cre_obj =  _struct_from_ptr(CREObjType,ptr)
    t_id, _, _ = decode_idrec(cre_obj.idrec)
    return t_id


from cre.utils import PrintElapse
class CREObjProxy(StructRefProxy):
    @classmethod
    def _numba_box_(cls, ty, mi):
        instance = ty.__new__(cls)
        instance._type = ty
        instance._meminfo = mi
        instance.recover_type_safe()
        
        return instance

    # __numba_ctor = cre_obj_ctor
    # def __new__(cls, idrec):
    #     return cre_obj_ctor(idrec)
    def __str__(self):
        return f"<cre.CREObj at {self.get_ptr()}>"

    def get_ptr(self):
        return get_var_ptr(self)

    def get_ptr(self):
        return get_cre_obj_ptr(self)

    def get_ptr_incref(self):
        return get_cre_obj_ptr_incref(self)

    def asa(self, typ):
        return asa(self,typ)

    def recover_type(self,context=None):
        ''' Try to recover the true type that the object was instantiated as'''            
        from cre.context import cre_context
        context = cre_context(context)
        t_id = get_t_id(self._meminfo)

        if(getattr(self._type,'t_id',0) == t_id):
            return self

        if(t_id != T_ID_TUPLE_FACT):
            # The type associated with the object's t_id in 'context'

            t_id_type = context.get_type(t_id=t_id)
            if(self.__class__ is not t_id_type and hasattr(t_id_type,"_proxy_class")):
                self._type = t_id_type
                self.__class__ = t_id_type._proxy_class
        else:
            from cre.tuple_fact import define_tuple_fact
            mbr_t_ids = cre_obj_get_member_t_ids(self)
            tf_mbr_types = [context.get_type(t_id=t_id) for t_id in mbr_t_ids]
            tf_type,tf_proxy_type = define_tuple_fact(tf_mbr_types,context,return_proxy=True)
            self.__class__ = tf_proxy_type

        return self

    def recover_type_safe(self,context=None):
        try:
            return self.recover_type(context)
        except (ValueError, AttributeError) as e:
            print("RECOVER FAILED", e)
            return self


define_boxing(CREObjTypeTemplate, CREObjProxy)




# @njit(u2(i8),cache=False)





# def define_boxing(struct_type, obj_class):
#     '''
#     Variation on define_boxing for CREObjects returns the t_id at boxing.
#     '''
#     obj_ctor = obj_class._numba_box_

#     @box(struct_type)
#     def box_struct_ref(typ, val, c):
#         """
#         Convert a raw pointer to a Python int.
#         """
#         utils = _Utils(c.context, c.builder, typ)
#         struct_ref = utils.get_struct_ref(val)
#         meminfo = struct_ref.meminfo

#         # ptr = c.builder.ptrtoint(meminfo, cgutils.intp_t)

#         # print(ptr)

#         def get_t_id(cre_obj):
#             # cre_obj =  _struct_from_ptr(CREObjType,ptr)
#             t_id, _, _ = decode_idrec(cre_obj.idrec)
#             return t_id
#         t_id = c.context.compile_internal(c.builder, get_t_id, u2(CREObjType,), (val,))

#         mip_type = types.MemInfoPointer(types.voidptr)
#         boxed_meminfo = c.box(mip_type, meminfo)
#         boxed_t_id = c.box(i8, t_id)
#         # t_id = boxed_meminfo

#         ctor_pyfunc = c.pyapi.unserialize(c.pyapi.serialize_object(obj_ctor))
#         ty_pyobj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))

#         res = c.pyapi.call_function_objargs(
#             ctor_pyfunc, [ty_pyobj, boxed_meminfo, boxed_t_id],
#         )
#         c.pyapi.decref(ctor_pyfunc)
#         c.pyapi.decref(ty_pyobj)
#         c.pyapi.decref(boxed_meminfo)
#         return res

#     @unbox(struct_type)
#     def unbox_struct_ref(typ, obj, c):
#         mi_obj = c.pyapi.object_getattr_string(obj, "_meminfo")

#         mip_type = types.MemInfoPointer(types.voidptr)

#         mi = c.unbox(mip_type, mi_obj).value

#         utils = _Utils(c.context, c.builder, typ)
#         struct_ref = utils.new_struct_ref(mi)
#         out = struct_ref._getvalue()

#         c.pyapi.decref(mi_obj)
#         return NativeValue(out)






# overload(CREObjProxy)(cre_obj_ctor)


# def _fact_eq(a,b):
#     if(isinstance(a,Fact) and isinstance(b,Fact)):
#         def impl(a,b):
#             return _raw_ptr_from_struct(a) ==_raw_ptr_from_struct(b)
#         return impl

# # fact_eq = generated_jit(cache=True)(_fact_eq)
# overload(operator.eq)(_fact_eq)



# @njit(types.boolean(CREObjType,CREObjType), cache=True)
# def cre_obj_eq(self,other):
#     return _raw_ptr_from_struct(self)==_raw_ptr_from_struct(other)

@njit(i8(CREObjType,), cache=True)
def get_cre_obj_ptr(self):
    return _raw_ptr_from_struct(self)

@njit(i8(CREObjType,), cache=True)
def get_cre_obj_ptr_incref(self):
    return _raw_ptr_from_struct_incref(self)

@njit(u8(CREObjType,), cache=True)
def get_cre_obj_idrec(self):
    return self.idrec


    

# eq_fn_typ = FunctionType(types.boolean(CREObjType,CREObjType))
# hash_fn_typ = FunctionType(u8(CREObjType))
# str_fn_typ = FunctionType(unicode_type(CREObjType))

# cre_obj_method_table_field_dict = {
#     "eq_" : FunctionType(types.boolean(CREObjType,CREObjType)),
#     "hash_" : FunctionType(u8(CREObjType)),
#     "str_" : FunctionType(unicode_type(CREObjType)),
#     "repr_" : FunctionType(unicode_type(CREObjType)),
# }

# cre_obj_method_table_fields = [(k,v) for k,v in cre_obj_method_table_field_dict.items()]


# CREObjMethodTable, CREObjMethodTableType = define_structref("CREObjMethodTable", cre_obj_method_table_field_dict, define_constructor=False)

# @njit(cache=True)
# def new_cre_obj_method_table(eq,hsh,s,r):
#     st = new(CREObjMethodTableType)
#     st.eq_ = _func_from_address(eq_fn_typ, eq)
#     st.hash_ = _func_from_address(hash_fn_typ, hsh)
#     st.str_ = _func_from_address(str_fn_typ, s)
#     st.repr_ = _func_from_address(str_fn_typ, r)
#     return st










# def _resolve_t_id_helper(x):
#     if(isinstance(x, types.Boolean)):
#         return T_ID_BOOL
#     elif(isinstance(x, types.Integer)):
#         return T_ID_INT
#     elif(isinstance(x, types.Float)):
#         return T_ID_FLOAT
#     elif(x is types.unicode_type):
#         return T_ID_STR
#     return T_ID_UNDEFINED

PRIMITIVE_MBR_ID = 0
OBJECT_MBR_ID = 1
LIST_MBR_ID = 2
DICT_MBR_ID = 3

def resolve_member_id(x):
    if(isinstance(x, types.StructRef)):
        return OBJECT_MBR_ID
    elif(isinstance(x, types.ListType)):
        return LIST_MBR_ID
    elif(isinstance(x, types.DictType)):
        return DICT_MBR_ID
    else:
        return PRIMITIVE_MBR_ID

@intrinsic
def _get_chr_mbrs_infos_from_attrs(typingctx, st_type, attrs_lit):
    '''get the base address of the struct pointed to by structref 'inst' '''
    # assert isinstance(attrs_lit, types.Literal)
    from cre.context import CREContext
    context = CREContext.get_default_context()

    # st_type = st_type_ref.instance_type
    # print(attrs_lit)
    if(len(attrs_lit.types) >= 0 and
     not isinstance(attrs_lit.types[0],types.Literal)): return
    
    # print(attrs_lit.types)
    attrs = [x.literal_value for x in attrs_lit.types]
    # print(attrs)
    # print(ind)
    mbr_types = [v for k,v in st_type._fields if k in attrs]
    t_ids = [context.get_t_id(_type=x) for x in mbr_types]
    m_ids = [resolve_member_id(x) for x in mbr_types]

    count = len(mbr_types)
    member_infos_out_type = types.UniTuple(member_info_type, count)
    # print(members_type.__dict__)


    def codegen(context, builder, sig, args):
        [st,_] = args
        utils = _Utils(context, builder, st_type)

        baseptr = utils.get_data_pointer(st)
        baseptr_val = builder.ptrtoint(baseptr, cgutils.intp_t)
        dataval = utils.get_data_struct(st)

        member_infos = []
        # i = 0
        for i, (attr, typ) in enumerate(zip(attrs, mbr_types)):
            # print("<<", attr, base_fact_field_dict)
            # if(attr not in base_fact_field_dict):

            index_of_member = dataval._datamodel.get_field_position(attr)
            member_ptr = builder.gep(baseptr, [cgutils.int32_t(0), cgutils.int32_t(index_of_member)], inbounds=False)
            member_ptr = builder.ptrtoint(member_ptr, cgutils.intp_t)
            offset = builder.trunc(builder.sub(member_ptr, baseptr_val), cgutils.ir.IntType(16))
            t_id = context.get_constant(u2, t_ids[i])
            m_id = context.get_constant(u2, m_ids[i])
            member_infos.append(context.make_tuple(builder, member_info_type, (t_id, m_id, offset)))
            # i += 1

        ret = context.make_tuple(builder,member_infos_out_type, member_infos)
        return ret

    sig = member_infos_out_type(st_type, attrs_lit)
    return sig, codegen



@njit
def set_chr_mbrs(st, chr_mbr_attrs):
    st.chr_mbrs_infos_offset = _get_member_offset(st,'chr_mbrs_infos')
    st.num_chr_mbrs =  len(chr_mbr_attrs)
    if(len(chr_mbr_attrs) > 0):
        chr_mbrs_infos = _get_chr_mbrs_infos_from_attrs(st, chr_mbr_attrs)
        st.chr_mbrs_infos = chr_mbrs_infos
    else:
        st.chr_mbrs_infos = ()
    st.hash_val = 0 
        
@njit(Tuple((u2,u2,i8))(CREObjType, i8),cache=True)
def cre_obj_get_item_t_id_ptr(x, index):
    data_ptr = _struct_get_data_ptr(x)
    member_info_ptr = data_ptr + x.chr_mbrs_infos_offset + index*_sizeof_type(member_info_type)
    t_id, m_id, member_offset = _load_ptr(member_info_type, member_info_ptr)
    return t_id, m_id, data_ptr + member_offset

@njit(cache=True)
def cre_obj_get_item(obj, item_type, index):
    _, _, item_ptr = cre_obj_get_item_t_id_ptr(obj,index)
    out = _load_ptr(item_type, item_ptr)
    return out


@njit(cache=True)
def cre_obj_iter_t_id_item_ptrs(x):
    # x = _cast_structref(TupleFact,_x)
    data_ptr = _struct_get_data_ptr(x)
    member_info_ptr = data_ptr + x.chr_mbrs_infos_offset

    for i in range(x.num_chr_mbrs):
        t_id, m_id, member_offset = _load_ptr(member_info_type, member_info_ptr)
        yield t_id, m_id, data_ptr + member_offset
        member_info_ptr += _sizeof_type(member_info_type)

@njit(cache=True)
def cre_obj_get_member_t_ids(x):
    t_ids = np.empty((x.num_chr_mbrs,),dtype=np.uint16)
    for i, (t_id,_, _) in enumerate(cre_obj_iter_t_id_item_ptrs(x)):
        t_ids[i] = t_id
    return t_ids




@generated_jit(cache=True,nopython=True)
@overload_method(CREObjType, "asa")
def asa(self, typ):
    def impl(self, typ):
        return _cast_structref(typ, self)
    return impl


# from cre.utils import _raw_ptr_from_struct, _struct_from_ptr, _store, _cast_structref, decode_idrec, encode_idrec, _incref_ptr, _load_ptr
@generated_jit(cache=True)
def copy_cre_obj(fact):
    fact_type = fact
    def impl(fact):
        new_fact = _memcpy_structref(fact)
        a,b = _cast_structref(CREObjType, fact), _cast_structref(CREObjType, new_fact)

        t_id, _, _ = decode_idrec(a.idrec)
        b.idrec = encode_idrec(t_id,0,u1(-1))
        for info_a, info_b in zip(cre_obj_iter_t_id_item_ptrs(a),cre_obj_iter_t_id_item_ptrs(b)):
            t_id_a, m_id_a, data_ptr_a = info_a
            t_id_b, m_id_b, data_ptr_b = info_b

            if(m_id_b != 0):
                obj_ptr = _load_ptr(i8, data_ptr_a)
                _incref_ptr(obj_ptr)

        return new_fact
    return impl
