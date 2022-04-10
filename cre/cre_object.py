from numba import i8, u8, u2, u1, types, njit, generated_jit, literal_unroll
from numba.types import FunctionType, unicode_type
from numba.extending import  overload, lower_getattr
from cre.utils import _obj_cast_codegen, ptr_t, _raw_ptr_from_struct, _raw_ptr_from_struct_incref, CastFriendlyMixin, decode_idrec, _func_from_address, _cast_structref, _get_member_offset
from cre.structref import define_structref
from numba.core.datamodel import default_manager, models
from numba.core import cgutils
from numba.experimental.structref import define_attributes, StructRefProxy, new, define_boxing, _Utils
from numba.experimental import structref
from numba.extending import intrinsic, lower_cast

import operator

member_info_type = types.Tuple((u2,u2))

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
    "num_chr_mbrs": u1,
    "chr_mbrs_infos_offset" : u2,
    "hash_val" : i8,
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
        super().__init__(fields)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Allow numba type inferencer to upcast this type to CREObjType
        lower_cast(cls, CREObjType)(impl_cre_obj_upcast)
        

    def __str__(self):
        return f"cre.CREObjType"

CREObjType = CREObjTypeTemplate(cre_obj_fields) 

# @njit(cache=True)
# def cre_obj_ctor(idrec):
#     st = new(CREObjType)
#     st.idrec = idrec
#     return st


class CREObjProxy(StructRefProxy):
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


default_manager.register(CREObjTypeTemplate, CREObjModel)

# overload(CREObjProxy)(cre_obj_ctor)
define_attributes(CREObjType)
define_boxing(CREObjTypeTemplate, CREObjProxy)


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








from cre.core import T_ID_UNRESOLVED, T_ID_BOOL, T_ID_INT, T_ID_FLOAT, T_ID_STR, T_ID_TUPLE_FACT 

def _resolve_t_id_helper(x):
    if(isinstance(x, types.Boolean)):
        return T_ID_BOOL
    elif(isinstance(x, types.Integer)):
        return T_ID_INT
    elif(isinstance(x, types.Float)):
        return T_ID_FLOAT
    elif(x is types.unicode_type):
        return T_ID_STR
    return T_ID_UNRESOLVED



@intrinsic
def _get_chr_mbrs_infos_from_attrs(typingctx, st_type, attrs_lit):
    '''get the base address of the struct pointed to by structref 'inst' '''
    # assert isinstance(attrs_lit, types.Literal)


    # st_type = st_type_ref.instance_type
    print(attrs_lit)
    if(len(attrs_lit.types) >= 0 and
     not isinstance(attrs_lit.types[0],types.Literal)): return
    
    print(attrs_lit.types)
    attrs = [x.literal_value for x in attrs_lit.types]
    print(attrs)
    # print(ind)
    mbr_types = [v for k,v in st_type._fields if k in attrs]
    t_ids = [_resolve_t_id_helper(x) for x in mbr_types]

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
            member_infos.append(context.make_tuple(builder, member_info_type, (t_id, offset)))
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

        
