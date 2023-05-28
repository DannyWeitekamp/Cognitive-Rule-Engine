import numpy as np
from numba import i8, i4, u8, u1, u2, u4, types, njit, generated_jit, literal_unroll
from numba.types import FunctionType, unicode_type, Tuple
from numba.extending import  overload, lower_getattr, overload_method
from cre.core import register_global_default, T_ID_UNDEFINED, T_ID_BOOL, T_ID_INT, T_ID_FLOAT, T_ID_STR, T_ID_TUPLE_FACT 
from cre.utils import (cast, _memcpy_structref, _obj_cast_codegen, ptr_t,
    _raw_ptr_from_struct_incref, _incref_ptr,
    CastFriendlyMixin, decode_idrec, _func_from_address, _incref_structref,
    _get_member_offset, _struct_get_data_ptr, _store, _store_safe,
    _sizeof_type, _load_ptr, encode_idrec, _decref_ptr, _incref_ptr,
    _decref_structref, check_issue_6993, incref_meminfo)
from cre.structref import define_structref
from numba.core.datamodel import default_manager, models
from numba.core import cgutils
from numba.experimental.structref import define_attributes, StructRefProxy, new, _Utils, define_boxing
from numba.experimental import structref
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import (
    models,
    intrinsic,
    lower_cast,
    lower_builtin,
    register_model,

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
    "num_chr_mbrs": u4,
    "chr_mbrs_infos_offset" : u4,
    # The data offset of the "members" attribute (unpredictable because of layout alignment)
    # "chr_mbrs_infos" : BaseIdentityMemberInfosType,#types.UniTuple(member_info_type,1),
}

cre_obj_fields = [(k,v) for k,v in cre_obj_field_dict.items()]


class CREObjModel(models.StructRefModel):
    pass

def impl_cre_obj_upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)

class CREObjTypeClass(CastFriendlyMixin, types.StructRef):
    def __init__(self, fields,*args,**kwargs):
        if(isinstance(fields,dict)): fields = [(k,v) for k,v in fields.items()]
        types.StructRef.__init__(self,fields)
        # self.name = f'CREObjType'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Allow numba type inferencer to upcast this type to CREObjType
        lower_cast(cls, CREObjType)(impl_cre_obj_upcast)
        

    def __str__(self):
        return f"CREObjType"

    def __repr__(self):
        return f"CREObjType"



    def hrepr():
        '''A human-readable repr for cre objects'''
        return f"{str(self)}(fields=({', '.join([f'{k}={str(v)}' for k,v in self._fields])}))" 


    # def __repr__(self):
    #     '''Make repr more readable than'''
    #     return f"{str(self)}(fields=({', '.join([f'{k}={str(v)}' for k,v in self._fields])}))" 


# @lower_cast(types.MemInfoPointer, types.Integer)
# def cast_meminfo_to_ptr(context, builder, fromty, toty, val):
#     return builder.ptrtoint(val, cgutils.intp_t)


CREObjType = CREObjTypeClass(cre_obj_fields) 
register_global_default("CREObj", CREObjType)
default_manager.register(CREObjTypeClass, CREObjModel)
define_attributes(CREObjType) #Not sure why but define on type instead of type_class


meminfo_type = types.MemInfoPointer(types.voidptr)
@njit(u2(meminfo_type),cache=True)
def get_t_id(ptr):
    cre_obj = cast(ptr, CREObjType)
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

    # def get_ptr(self):
    #     return get_var_ptr(self)

    def get_ptr(self):
        return get_cre_obj_ptr(self)

    def get_ptr_incref(self):
        return get_cre_obj_ptr_incref(self)

    def asa(self, typ):
        if(typ._proxy_class is self._type): return self
        incref_meminfo(self._meminfo)
        proxy_typ = typ._proxy_class
        instance = super(StructRefProxy, proxy_typ).__new__(proxy_typ)
        instance._type = typ
        instance._meminfo = self._meminfo
        return instance
        # return asa(self,typ)

    def recover_type(self,context=None):
        ''' Try to recover the true type that the object was instantiated as'''            
        from cre.context import cre_context
        context = cre_context(context)# 2us

        t_id = get_t_id(self._meminfo)# .5us

        # print("<<", t_id, getattr(self._type,'t_id',0))
        # print(type(self),self._type,)
        if(getattr(self._type,'t_id',0) == t_id):# .5us
            return self

        # print("RECOVER",getattr(self._type,'t_id',0), t_id)
        if(t_id != T_ID_TUPLE_FACT):
            # The type associated with the object's t_id in 'context'
            t_id_type = context.get_type(t_id=t_id) # 3.9us
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

    # def __hash__(self):
    #     from cre.dynamic_exec import cre_obj_hash        
    #     return cre_obj_hash(self)

    def __eq__(self, other):
        from cre.dynamic_exec import cre_obj_eq
        if(not isinstance(other, CREObjProxy)): return False
        return cre_obj_eq(self, other)


CREObjType._proxy_class = CREObjProxy
define_boxing(CREObjTypeClass, CREObjProxy)


@njit(i8(CREObjType,), cache=True)
def get_cre_obj_ptr(self):
    return cast(self, i8)

@njit(i8(CREObjType,), cache=True)
def get_cre_obj_ptr_incref(self):
    return _raw_ptr_from_struct_incref(self)

@njit(u8(CREObjType,), cache=True)
def get_cre_obj_idrec(self):
    return self.idrec

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
def _get_chr_mbrs_infos_from_attrs(typingctx, structref, attrs_lit):
    '''get the base address of the struct pointed to by structref 'inst' '''
    # assert isinstance(attrs_lit, types.Literal)
    from cre.context import CREContext
    context = CREContext.get_default_context()

    # st_type = st_type_ref.instance_type
    # print("attrs_lit", attrs_lit)
    if(len(attrs_lit.types) >= 0 and
     not isinstance(attrs_lit.types[0],types.Literal)): return
    
    # print(attrs_lit.types)
    attrs = [x.literal_value for x in attrs_lit.types]
    # print(attrs)
    # print(ind)
    mbr_types = [v for k,v in structref._fields if k in attrs]
    t_ids = [context.get_t_id(_type=x) for x in mbr_types]
    m_ids = [resolve_member_id(x) for x in mbr_types]

    count = len(mbr_types)
    member_infos_out_type = types.UniTuple(member_info_type, count)
    # print(members_type.__dict__)


    def codegen(context, builder, sig, args):
        [st,_] = args
        utils = _Utils(context, builder, structref)

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

    sig = member_infos_out_type(structref, attrs_lit)
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

@generated_jit(cache=True,nopython=True)
@overload_method(CREObjTypeClass,'get_item')
def cre_obj_get_item(obj, item_type, index):
    # print("obj", obj)
    # if(obj is not CREObjType): return
    def impl(obj, item_type, index):
        _, _, item_ptr = cre_obj_get_item_t_id_ptr(obj, index)
        out = _load_ptr(item_type, item_ptr)
        return out
    return impl

@generated_jit(cache=True,nopython=True)
def cre_obj_set_item(obj, index, val):
    item_type = val
    if(isinstance(item_type,types.Literal)):
        item_type = item_type.literal_type

    from cre.context import cre_context
    context = cre_context()

    m_id = resolve_member_id(item_type)
    t_id = context.get_t_id(_type=item_type)

    from numba.core.datamodel import default_manager, StructModel
    # if isinstance(default_manager[item_type], StructModel):
    # self._datamodel = self._context.data_model_manager[self._fe_type]
    def impl(obj, index, val):
        _, m_id, item_ptr = cre_obj_get_item_t_id_ptr(obj, index)
        # old_ptr = _load_ptr(i8, item_ptr)

        _store_safe(item_type, item_ptr, val)
        # _incref_structref(val)

        cre_obj_set_member_t_id_m_id(obj, index, (u2(t_id), u2(m_id)))

        # if(old_ptr != 0):
        #     _decref_ptr(old_ptr)
    # else:
    #     def impl(obj, index, val):
    #         _, m_id, item_ptr = cre_obj_get_item_t_id_ptr(obj, index)
    #         _store(item_type, item_ptr, val)        
    #         cre_obj_set_member_t_id_m_id(obj, index, (u2(t_id), u2(m_id)))
    return impl


# np_t_id_item_ptrs_type = np.dtype([('t_id', np.uint16),  ('m_id', np.uint16), ('ptr', np.int64)])
# t_id_item_ptrs = numba.from_dtype(np_t_id_item_ptrs_type)


# -------------------------------------------------------------------
# : _iter_mbr_infos()


mbr_info_tup_type = types.Tuple((u2, u2, i8))

class MbrInfoIterator(types.SimpleIteratorType):
    def __init__(self):
        name = f"MbrInfoIterator"
        super().__init__(name, mbr_info_tup_type)

MbrInfoIteratorType = MbrInfoIterator()

@register_model(MbrInfoIterator)
class MbrInfoIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data_ptr', i8),
            ('member_info_ptr', types.EphemeralPointer(i8)),
            ('nitems', i4),
            ('index', types.EphemeralPointer(i4))
        ]
        super().__init__(dmm, fe_type, members)

from numba.core.imputils import lower_builtin, iternext_impl, RefType, impl_ret_borrowed

@intrinsic
def _iter_mbr_infos(typingctx, cre_obj):
    def codegen(context, builder, sig, args):
        (cre_obj,) = args
        # Make proxies for cre_obj and _iter
        cre_obj = cgutils.create_struct_proxy(CREObjType)(context, builder, value=args[0])
        _iter = cgutils.create_struct_proxy(MbrInfoIteratorType)(context, builder)

        # Set data_ptr
        raw_data_ptr = context.nrt.meminfo_data(builder, cre_obj.meminfo)
        _iter.data_ptr = builder.ptrtoint(raw_data_ptr, cgutils.intp_t)

        # Make struct proxy for underlying data
        valtype = CREObjType.get_data_type()
        model = context.data_model_manager[valtype]
        alloc_type = model.get_value_type()
        data_ptr = builder.bitcast(raw_data_ptr, alloc_type.as_pointer())
        data_struct = cgutils.create_struct_proxy(valtype)(
            context, builder, ref=data_ptr)

        # Extract and assign nitems
        nitems = builder.zext(getattr(data_struct, "num_chr_mbrs"), cgutils.int32_t)
        _iter.nitems = nitems

        # Extract chr_mbrs_infos_offset, assign member_info_ptr
        chr_mbrs_infos_offset = builder.zext(getattr(data_struct, "chr_mbrs_infos_offset"), cgutils.intp_t)
        data_ptr = builder.ptrtoint(data_ptr, cgutils.intp_t)
        member_info_ptr = builder.add(data_ptr, chr_mbrs_infos_offset)
        _iter.member_info_ptr = cgutils.alloca_once_value(builder, member_info_ptr)
        
        # Set index to 0.
        index = context.get_constant(types.int32, 0)
        _iter.index = cgutils.alloca_once_value(builder, index)

        return _iter._getvalue()
    sig = MbrInfoIteratorType(CREObjType)
    return sig, codegen


@lower_builtin('iternext', MbrInfoIterator)
@iternext_impl(RefType.BORROWED)
def iternext_listiter(context, builder, sig, args, result):
    # Define Constants
    llty = context.get_data_type(member_info_type)
    member_info_ptr_llty = llty.as_pointer()
    member_info_size = cgutils.sizeof(builder, member_info_ptr_llty)
    u2_ptr = context.get_data_type(u2).as_pointer()

    # Make iter proxy
    _iter = cgutils.create_struct_proxy(sig.args[0])(context, builder, value=args[0])

    # Load index and check inbounds.
    index = builder.load(_iter.index)
    is_valid = builder.icmp_signed('<', index, _iter.nitems)
    result.set_valid(is_valid)

    # If index is inbounds yeild and iterate 
    with builder.if_then(is_valid):
        member_info_ptr = builder.load(_iter.member_info_ptr)

        # Extract t_id
        t_id_ptr = builder.inttoptr(member_info_ptr, u2_ptr)
        t_id = builder.load(t_id_ptr)

        # Extract m_id
        m_id_ptr = builder.add(member_info_ptr, context.get_constant(i8, 2))
        m_id_ptr = builder.inttoptr(m_id_ptr, u2_ptr)
        m_id = builder.load(m_id_ptr)

        # Extract offset
        offset_ptr = builder.add(member_info_ptr, context.get_constant(i8, 4))
        offset_ptr = builder.inttoptr(offset_ptr, u2_ptr)
        offset = builder.load(offset_ptr)

        # DO: yield t_id, m_id, data_ptr + offset
        mbr_pointer = builder.add(_iter.data_ptr, builder.zext(offset, cgutils.intp_t))
        out = context.make_tuple(builder,mbr_info_tup_type,[t_id, m_id, mbr_pointer])
        result.yield_(out)

        # DO: _iter.index += 1
        index = builder.add(index, context.get_constant(i4, 1))
        builder.store(index, _iter.index)

        # DO:_iter.member_info_ptr += member_info_size1
        member_info_ptr = builder.load(_iter.member_info_ptr)
        member_info_ptr = builder.add(member_info_ptr, member_info_size)
        builder.store(member_info_ptr, _iter.member_info_ptr)
    

# has_not_fixed_6993 = check_issue_6993()

# @generated_jit(cache=False)
# def cre_obj_iter_t_id_item_ptrs(x):
#     def impl(x):
#         # x = _cast_structref(TupleFact,_x)
#         data_ptr = _struct_get_data_ptr(x)
#         member_info_ptr = data_ptr + x.chr_mbrs_infos_offset

#         # NOTE: Using as generator causes memory leak
#         for i in range(x.num_chr_mbrs):
#             t_id, m_id, member_offset = _load_ptr(member_info_type, member_info_ptr)
#             yield t_id, m_id, data_ptr + member_offset
#             member_info_ptr += _sizeof_type(member_info_type)
#             # _decref_structref(x)

#         if(has_not_fixed_6993):
#             _decref_structref(x)
            
#     return impl

t_id_m_id_type = types.Tuple((u2,u2))

@njit(cache=True)
def cre_obj_set_member_t_id_m_id(x, i, t_id_m_id):
    data_ptr = _struct_get_data_ptr(x)
    member_info_ptr = data_ptr + x.chr_mbrs_infos_offset + i * _sizeof_type(member_info_type)
    _store(t_id_m_id_type, member_info_ptr, t_id_m_id)



@njit(u2[::1](CREObjType),cache=True)
def cre_obj_get_member_t_ids(x):
    t_ids = np.empty((x.num_chr_mbrs,),dtype=np.uint16)
    for i, (t_id,_, _) in enumerate(_iter_mbr_infos(x)):
        t_ids[i] = t_id
    return t_ids


@generated_jit(cache=True,nopython=True)
@overload_method(CREObjType, "asa")
def asa(self, typ):
    def impl(self, typ):
        return cast(self, typ)
    return impl


@generated_jit(cache=True,nopython=True)
def copy_cre_obj(fact):
    fact_type = fact
    def impl(fact):
        new_fact = _memcpy_structref(fact)

        a,b = cast(fact, CREObjType), cast(new_fact, CREObjType)

        t_id, _, _ = decode_idrec(a.idrec)
        b.idrec = encode_idrec(t_id,0,u1(-1))
        b.hash_val = 0 
        for info_a in _iter_mbr_infos(a):
            t_id_a, m_id_a, data_ptr_a = info_a
            # t_id_b, m_id_b, data_ptr_b = info_b

            if(m_id_a != PRIMITIVE_MBR_ID):
                obj_ptr = _load_ptr(i8, data_ptr_a)
                _incref_ptr(obj_ptr)
            elif(t_id_a == T_ID_STR):
                s = _load_ptr(unicode_type, data_ptr_a)
                _incref_structref(s)
                #_store(unicode_type, data_ptr_b, s)

        # Weird extra refcounts... as much as 4 extra if call cre_obj_iter_t_id_item_ptrs
        #  on the new_fact
        _decref_structref(new_fact)

        return new_fact
    return impl


@njit(types.void(CREObjType), cache=True)
def cre_obj_clear_refs(fact):
    for info_a in _iter_mbr_infos(fact):
        t_id_a, m_id_a, data_ptr_a = info_a
        obj_ptr = _load_ptr(i8, data_ptr_a)
        if(m_id_a == OBJECT_MBR_ID):
            obj_ptr = _load_ptr(i8, data_ptr_a)
            _decref_ptr(obj_ptr)
            _store(i8, data_ptr_a, 0)
        # elif(t_id_a == T_ID_STR):
        #     s = _load_ptr(unicode_type, data_ptr_a)
        #     _incref_structref(s)
            
            



