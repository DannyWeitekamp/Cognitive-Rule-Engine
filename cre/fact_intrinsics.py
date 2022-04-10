from numba import types, njit, guvectorize, vectorize, prange, generated_jit
from numba.experimental import jitclass, structref
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple, Array
from numba.core.extending import (
    infer_getattr,
    lower_getattr_generic,
    lower_setattr_generic,
    overload_method,
    intrinsic,
    overload,
    box,
    unbox,
    NativeValue
)
from numba.core.typing.templates import AttributeTemplate
from numba.core.datamodel import default_manager, models
from numba.core import cgutils, utils as numba_utils
from numba.experimental.structref import _Utils, imputils
from numba.typed.typedobjectutils import _nonoptional

from cre.utils import _raw_ptr_from_struct, _struct_from_ptr, _cast_structref, _ptr_from_struct_codegen, _list_from_ptr, _cast_list
from cre.cre_object import _resolve_t_id_helper, member_info_type



def define_boxing(struct_type, obj_class):
    """
    
    Define the boxing & unboxing logic for `struct_type` to `obj_class`.

    Defines both boxing and unboxing.

    - boxing turns an instance of `struct_type` into a PyObject of `obj_class`
    - unboxing turns an instance of `obj_class` into an instance of
      `struct_type` in jit-code.


    Use this directly instead of `define_proxy()` when the user does not
    want any constructor to be defined.
    """
    if struct_type is types.StructRef:
        raise ValueError(f"cannot register {types.StructRef}")

    obj_ctor = obj_class._numba_box_

    @box(struct_type)
    def box_struct_ref(typ, val, c):
        # print("BOX", typ, obj_ctor)
        """
        Convert a raw pointer to a Python int.
        """
        utils = _Utils(c.context, c.builder, typ)
        struct_ref = utils.get_struct_ref(val)
        meminfo = struct_ref.meminfo

        mip_type = types.MemInfoPointer(types.voidptr)
        boxed_meminfo = c.box(mip_type, meminfo)

        ctor_pyfunc = c.pyapi.unserialize(c.pyapi.serialize_object(obj_ctor))
        # ty_pyobj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))

        res = c.pyapi.call_function_objargs(
            ctor_pyfunc, [boxed_meminfo],
        )
        c.pyapi.decref(ctor_pyfunc)
        # c.pyapi.decref(ty_pyobj)
        c.pyapi.decref(boxed_meminfo)
        return res

    @unbox(struct_type)
    def unbox_struct_ref(typ, obj, c):
        mi_obj = c.pyapi.object_getattr_string(obj, "_meminfo")

        mip_type = types.MemInfoPointer(types.voidptr)

        mi = c.unbox(mip_type, mi_obj).value

        utils = _Utils(c.context, c.builder, typ)
        struct_ref = utils.new_struct_ref(mi)
        out = struct_ref._getvalue()

        c.pyapi.decref(mi_obj)
        return NativeValue(out)


def resolve_fact_getattr_type(typ, attr):
    from cre.fact import DeferredFactRefType, Fact
    if (hasattr(typ,'spec') and attr in typ.spec):
        attrty = typ.spec[attr]['type']

        if(isinstance(attrty,DeferredFactRefType)):
            attrty = attrty.get()

        if(isinstance(attrty,Fact)):
            attrty = types.optional(attrty)
        if(isinstance(attrty, ListType)):
            if(isinstance(attrty.dtype,DeferredFactRefType)):
                attrty = ListType(attrty.dtype.get())
            attrty = types.optional(attrty)
             
        return attrty
    if attr in typ.field_dict:
        attrty = typ.field_dict[attr]
        # print("<<",attr, attrty)
        return attrty



# @intrinsic
@njit(cache=True)
def safe_get_fact_ptr(fact):
    if(fact is None):
        return 0
    else:
        return _raw_ptr_from_struct(_nonoptional(fact))

@intrinsic
def get_fact_attr_ptr(typingctx, inst_type, attr_type):
    from cre.fact import Fact
    attr = attr_type.literal_value
    def codegen(context, builder, sig, args):
        val, _ = args
        typ, _ = sig.args

        print(typ,attr)
        field_type = typ.field_dict[attr]

        if(not isinstance(field_type, (Fact))):
            raise ValueError()

        utils = _Utils(context, builder, typ)
        dataval = utils.get_data_struct(val)
        ret = getattr(dataval, attr)

        ret = _ptr_from_struct_codegen(context, builder, ret, field_type, False)

        return ret
    return u8(inst_type,attr_type), codegen

def fact_setattr_codegen(context, builder, sig, args, attr, mutability_protected=False):
    from cre.fact import Fact, BaseFact
    if(len(args) == 2):
        [inst_type, val_type] = sig.args
        [instance, val] = args
    else:
        [inst_type, _, val_type] = sig.args
        [instance, _, val] = args
    utils = _Utils(context, builder, inst_type)
    dataval = utils.get_data_struct(instance)

    if(mutability_protected):
        idrec = getattr(dataval, "idrec")
        # If (idec & 0xFF) != 0, throw an error 
        idrec_set = builder.icmp_unsigned('==', builder.and_(idrec, idrec.type(0xFF)), idrec.type(0))
        with builder.if_then(idrec_set):
            msg =("Facts objects are immutable once declared. Use mem.modify instead.",)
            context.call_conv.return_user_exc(builder, AttributeError, msg)

    
    field_type = inst_type.field_dict[attr]

    # print("&&", attr, field_type)
    if(isinstance(field_type, (ListType,))):
        dtype = field_type.dtype
        # if(isinstance(field_type, Fact)):
        if(isinstance(val_type, types.Optional)):
            # If list member assigned to none just instantiate an empty list
            def cast_obj(x):
                if(x is None):
                    return List.empty_list(dtype)
                return _cast_list(field_type, _nonoptional(x))
        else:
            def cast_obj(x):
                if(x is None):
                    return List.empty_list(dtype)
                return _cast_list(field_type, x)
        casted = context.compile_internal(builder, cast_obj, field_type(val_type,), (val,))

    elif(isinstance(field_type, (Fact,))):
        # If fact member assigned to none just assign to NULL pointer
        if(isinstance(val_type, types.Optional)):
            def cast_obj(x):
                if(x is None):
                    return _struct_from_ptr(field_type,0)
                return _cast_structref(field_type, _nonoptional(x))
        else:
            def cast_obj(x):
                if(x is None):
                    return _struct_from_ptr(field_type,0)
                return _cast_structref(field_type, x)

        casted = context.compile_internal(builder, cast_obj, field_type(val_type,), (val,))
    else:
        casted = context.cast(builder, val, val_type, field_type)

    pyapi = context.get_python_api(builder)
    
    # read old
    old_value = getattr(dataval, attr)
    # incref new value
    context.nrt.incref(builder, field_type, casted)
    # decref old value (must be last in case new value is old value)
    context.nrt.decref(builder, field_type, old_value)
    # write new
    setattr(dataval, attr, casted)

    if(mutability_protected):
        # Make sure that hash_val is 0 to force it to be recalculated
        setattr(dataval, "hash_val", cgutils.intp_t(0))
    return dataval
        # ret = _obj_cast_codegen(context, builder, ret, field_type, ret_type, False)

    # return imputils.impl_ret_borrowed(context, builder, ret_type, ret)


@intrinsic
def fact_lower_setattr(typingctx, inst_type, attr_type, val_type):
    if (isinstance(attr_type, types.Literal) and 
        isinstance(inst_type, types.StructRef)):
        # print("BB", isinstance(inst_type, types.StructRef), inst_type, attr_type)
        
        attr = attr_type.literal_value
        def codegen(context, builder, sig, args):
            fact_setattr_codegen(context, builder, sig, args, attr)
  
        sig = types.void(inst_type, attr_type, val_type)
        # print(sig)
        return sig, codegen


@intrinsic
def fact_mutability_protected_setattr(typingctx, inst_type, attr_type, val_type):
    if (isinstance(attr_type, types.Literal) and 
        isinstance(inst_type, types.StructRef)):
        # print("BB", isinstance(inst_type, types.StructRef), inst_type, attr_type)
        
        attr = attr_type.literal_value
        def codegen(context, builder, sig, args):
            fact_setattr_codegen(context, builder, sig, args, attr, 
                mutability_protected=True)
  
        sig = types.void(inst_type, attr_type, val_type)
        # print(sig)
        return sig, codegen


def define_attributes(struct_typeclass):
    from cre.fact import Fact, base_fact_fields
    """
    Copied from numba.experimental.structref 0.51.2, but added protected mutability
    """
    # print("REGISTER FACT")
    @infer_getattr
    class StructAttribute(AttributeTemplate):
        key = struct_typeclass

        def generic_resolve(self, _typ, attr):
            typ = resolve_fact_getattr_type(_typ, attr)
            # print(attr, _typ,"!>>", typ)
            return typ

    @lower_getattr_generic(struct_typeclass)
    def struct_getattr_impl(context, builder, typ, val, attr):
        field_type = typ.field_dict[attr]
        ret_type = resolve_fact_getattr_type(typ,attr)

        # Extract unoptional part of type
        if(isinstance(ret_type, types.Optional)):
            ret_type = ret_type.type

        utils = _Utils(context, builder, typ)
        dataval = utils.get_data_struct(val)
        ret = getattr(dataval, attr)

        option_ret_type = types.optional(ret_type)
        if(isinstance(ret_type, (Fact,))):
            # If a fact member is Null then return None
            def cast_obj(x):
                if(_raw_ptr_from_struct(x) != 0):
                    return _cast_structref(ret_type, x)
                return None
            ret = context.compile_internal(builder, cast_obj, option_ret_type(field_type,), (ret,))
            ret_type = option_ret_type
            
        elif(isinstance(ret_type, (ListType,))):
            # List members should always be non-null
            def cast_obj(x):
                return _cast_list(ret_type, x)
            ret = context.compile_internal(builder, cast_obj, option_ret_type(field_type,), (ret,))
            ret_type = option_ret_type
            
        return imputils.impl_ret_borrowed(context, builder, ret_type, ret)

    @lower_setattr_generic(struct_typeclass)
    def struct_setattr_impl(context, builder, sig, args, attr):
        
        dataval = fact_setattr_codegen(context, builder, sig, args, attr, mutability_protected=True)
        

        # [inst_type, val_type] = sig.args
        # [instance, val] = args
        # utils = _Utils(context, builder, inst_type)
        # dataval = utils.get_data_struct(instance)
        
        # field_type = inst_type.field_dict[attr]
        # if(isinstance(field_type, (ListType,Fact))):
        #     if(isinstance(field_type, Fact)):
        #         def cast_obj(x):
        #             if(x is None):
        #                 return _struct_from_ptr(BaseFact,0)
        #             return _cast_structref(BaseFact, _nonoptional(x))
        #     else:
        #         def cast_obj(x):
        #             return _cast_structref(field_type,x) 

        #     casted = context.compile_internal(builder, cast_obj, field_type(val_type,), (val,))
        # else:
        #     casted = context.cast(builder, val, val_type, field_type)

        # pyapi = context.get_python_api(builder)

        
            
        # # read old
        # old_value = getattr(dataval, attr)
        # # incref new value
        # context.nrt.incref(builder, field_type, casted)
        # # decref old value (must be last in case new value is old value)
        # context.nrt.decref(builder, field_type, old_value)
        # # write new
        # setattr(dataval, attr, casted)


def _register_fact_structref(fact_type):
    from cre.cre_object import CREObjModel
    if fact_type is types.StructRef:
        raise ValueError(f"cannot register {types.StructRef}")
    default_manager.register(fact_type, CREObjModel)
    define_attributes(fact_type)
    return fact_type


# @intrinsic
# def _fact_get_chr_mbrs_infos(typingctx, fact_type):
#     from cre.fact import base_fact_fields, base_fact_field_dict

#     t_ids = [_resolve_t_id_helper(x) for a,x in fact_type._fields if a not in base_fact_fields]
#     member_infos_out_type = types.UniTuple(member_info_type, len(t_ids))


#     def codegen(context, builder, sig, args):
#         [fact,] = args
#         utils = _Utils(context, builder, fact_type)

#         baseptr = utils.get_data_pointer(fact)
#         baseptr_val = builder.ptrtoint(baseptr, cgutils.intp_t)
#         dataval = utils.get_data_struct(fact)

#         member_infos = []
#         i = 0
#         for attr, typ in fact_type._fields:
#             # print("<<", attr, base_fact_field_dict)
#             if(attr not in base_fact_field_dict):
#                 index_of_member = datamodel.get_field_position(attr)
#                 member_ptr = builder.gep(baseptr, [cgutils.int32_t(0), cgutils.int32_t(index_of_member)], inbounds=False)
#                 member_ptr = builder.ptrtoint(member_ptr, cgutils.intp_t)
#                 offset = builder.trunc(builder.sub(member_ptr, baseptr_val), cgutils.ir.IntType(16))
#                 t_id = context.get_constant(u2, t_ids[i])
#                 member_infos.append(context.make_tuple(builder, member_info_type, (t_id, offset)))
#                 i += 1

#         ret = context.make_tuple(builder,member_infos_out_type, member_infos)
#         return ret

#     sig = member_infos_out_type(fact_type_ref)
#     return sig, codegen



@intrinsic
def _fact_get_chr_mbrs_infos(typingctx, fact_type):
    from cre.fact import base_fact_field_dict
    '''get the base address of the struct pointed to by structref 'inst' '''
    
    # members_type = [v for k,v in fact_type._fields if k == 'members'][0]
    t_ids = [_resolve_t_id_helper(x) for a,x in fact_type._fields if a not in base_fact_field_dict and a != "chr_mbrs_infos"]

    # count = members_type.count
    member_infos_out_type = types.UniTuple(member_info_type, len(t_ids))

    
    def codegen(context, builder, sig, args):
        [fact,] = args

        utils = _Utils(context, builder, fact_type)

        baseptr = utils.get_data_pointer(fact)
        baseptr_val = builder.ptrtoint(baseptr, cgutils.intp_t)
        dataval = utils.get_data_struct(fact)
        # index_of_members = dataval._datamodel.get_field_position("members")

        member_infos = []
        i = 0
        for attr, typ in fact_type._fields:
            if(attr not in base_fact_field_dict and attr != "chr_mbrs_infos"):
                index_of_member = dataval._datamodel.get_field_position(attr)
                member_ptr = builder.gep(baseptr, [cgutils.int32_t(0), cgutils.int32_t(index_of_member)], inbounds=True)
                member_ptr = builder.ptrtoint(member_ptr, cgutils.intp_t)
                offset = builder.trunc(builder.sub(member_ptr, baseptr_val), cgutils.ir.IntType(16))
                t_id = context.get_constant(u2, t_ids[i])
                member_infos.append(context.make_tuple(builder, member_info_type, (t_id, offset) ))
                i += 1
                # print("<<", attr, typ)
        ret = context.make_tuple(builder,member_infos_out_type, member_infos)
        return ret

    sig = member_infos_out_type(fact_type)
    return sig, codegen
