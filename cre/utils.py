
import numba
from numba import types, njit, u1,u2,u4,u8, i8,i2,f8, literally
from numba.core.imputils import impl_ret_borrowed
from numba.types import Tuple, void, ListType
from numba.typed import List
from numba.experimental.structref import _Utils, imputils
from numba.extending import intrinsic, overload_method, overload
from numba.core import cgutils, utils as numba_utils
from llvmlite.ir import types as ll_types
from llvmlite import ir
import inspect
import numpy as np 
import numba
from numba.typed.typedobjectutils import _container_get_data, _nonoptional
from numba.core.datamodel import default_manager, models
from numba.core.typeconv import Conversion
from numba.np.arrayobj import _getitem_array_single_int, make_array
from numba.core.datamodel.models import IntegerModel, register_default
from numba.core.imputils import (lower_cast)
import operator

#### Ptr Type #####

class Ptr(types.Integer):
    pass

# Pointer Type, typically equivalent to i8 but will decref when dtor of parent struct is called
ptr_t = Ptr("ptr_t", numba_utils.MACHINE_BITS)

#Weak Pointer Type
wptr_t = i8

@register_default(Ptr)
class PtrModel(IntegerModel):
    _ptr_type = ir.IntType(8).as_pointer()

    def get_nrt_meminfo(self, builder, value):
        """
        Returns the MemInfo object or None if it is not tracked.
        It is only defined for types.meminfo_pointer
        """
        return builder.inttoptr(value, cgutils.voidptr_t)

    def has_nrt_meminfo(self):
        return True

@lower_cast(Ptr, types.Integer)
def b_ind_set(context, builder, fromty, toty, val):
    return val

# class PtrCompare(AbstractTemplate):
#     def generic(self, args, kws):
#         [lhs, rhs] = args
#         if isinstance(lhs, types.Integer) and isinstance(rhs, types.Integer):

from numba.core.imputils import (builtin_registry, lower_builtin, lower_getattr,
                                    lower_getattr_generic, lower_cast,
                                    lower_constant, impl_ret_borrowed,
                                    impl_ret_untracked)

from numba.cpython.numbers import lower_builtin, int_eq_impl, int_ne_impl


@overload(operator.eq)
def _ptr_eq(a,b):
    if(isinstance(a,types.Integer) and isinstance(b,types.Integer)):
        def impl(a,b):
            return i8(a) == i8(b)
        return impl


# overload(operator.eq)(_fact_eq)
# print("<<", list(builtin_registry.functions))
# lower_builtin(operator.eq, Ptr, types.Integer)(int_eq_impl)
# lower_builtin(operator.eq, types.Integer, Ptr)(int_eq_impl)
# lower_builtin(operator.ne, Ptr, types.Integer)(int_ne_impl)






#### CastFriendlyMixin ####


class CastFriendlyMixin():
    def can_convert_to(self, typingctx, other):
        """
        Convert this Record to the *other*.
        This method only implements width subtyping for records.
        """
        from numba.core.errors import NumbaExperimentalFeatureWarning
        # print(other.__dict__)
        if issubclass(type(self), type(other)):
            if len(other._fields) > len(self._fields):
                return
            for other_fd, self_fd in zip(other._fields,
                                         self._fields):
                if not other_fd == self_fd and other_fd[1] != types.Any and self_fd[1] != types.Any:
                    return
            # warnings.warn(f"{self} has been considered a subtype of {other} "
            #               f" This is an experimental feature.",
            #               category=NumbaExperimentalFeatureWarning)
            return Conversion.safe

    def can_convert_from(self, typingctx, other):
        return other.can_convert_to(typingctx, other)

    def unify(self, typingctx, other):
        if(self.can_convert_to(typingctx, other)):
            return other


#### deref_info_type ####

np_deref_info_type = np.dtype([
    # Enum for ATTR or LIST
    ('type', np.uint8),
    ('a_id', np.uint32),
    ('t_id', np.uint16),
    ('offset', np.int32)
    # NOTE: Should pad to make 64-bit aligned
])

deref_info_type = numba.from_dtype(np_deref_info_type)

DEREF_TYPE_ATTR = 0
DEREF_TYPE_LIST = 1 

#### idrec encoding ####

@njit(Tuple([u2,u8,u1])(u8),cache=True)
def decode_idrec(idrec):
    t_id = idrec >> 48
    f_id = (idrec >> 8) & 1099511627775 #i.e. ((~u8(0)) >> 24) 
    a_id = idrec & 0xFF
    return (t_id, f_id, a_id)


@njit(u8(u2,u8,u1),cache=True)
def encode_idrec(t_id, f_id, a_id):
    return (t_id << 48) | (f_id << 8) | a_id

meminfo_type = types.MemInfoPointer(types.voidptr)
@lower_cast(meminfo_type, i8)
def cast_meminfo_to_ptr(context, builder, fromty, toty, val):
    return builder.ptrtoint(val, cgutils.intp_t)

@intrinsic
def _ptr_to_meminfo(typingctx, ptr_type):    

    mip_type = types.MemInfoPointer(types.voidptr)
    def codegen(context, builder, signature, args):
        raw_ptr = args[0]
        meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)
        context.nrt.incref(builder, mip_type, meminfo)

        return meminfo
    sig = mip_type(ptr_type)
    return sig, codegen



@njit(types.MemInfoPointer(types.voidptr)(i8),cache=True)
def ptr_to_meminfo(ptr):
    return _ptr_to_meminfo(ptr)


@intrinsic
def lower_setattr(typingctx, inst_type, attr_type, val_type):
    # print("KK", isinstance(inst_type, types.StructRef), inst_type, attr_type)
    if (isinstance(attr_type, types.Literal) and 
        isinstance(inst_type, types.StructRef)):
        
        attr = attr_type.literal_value
        def codegen(context, builder, sig, args):
            [instance, attr_v, val] = args

            utils = _Utils(context, builder, inst_type)
            dataval = utils.get_data_struct(instance)
            # cast val to the correct type
            field_type = inst_type.field_dict[attr]
            casted = context.cast(builder, val, val_type, field_type)
            # read old
            old_value = getattr(dataval, attr)
            # incref new value
            context.nrt.incref(builder, val_type, casted)
            # decref old value (must be last in case new value is old value)
            context.nrt.decref(builder, val_type, old_value)
            # write new
            setattr(dataval, attr, casted)
        sig = types.void(inst_type, types.literal(attr), val_type)
        return sig, codegen


@intrinsic
def lower_getattr(typingctx, inst_type, attr_type):
    if (isinstance(attr_type, types.Literal) and 
        isinstance(inst_type, types.StructRef)):
        
        attr = attr_type.literal_value
        fieldtype = inst_type.field_dict[attr]
        def codegen(context, builder, sig, args):
            [instance, attr_v] = args

            utils = _Utils(context, builder, inst_type)
            dataval = utils.get_data_struct(instance)
            ret = getattr(dataval, attr)
            return imputils.impl_ret_borrowed(context, builder, fieldtype, ret)


        sig = fieldtype(inst_type, types.literal(attr))
        return sig, codegen


@intrinsic
def _struct_from_meminfo(typingctx, struct_type, meminfo):
    inst_type = struct_type.instance_type

    def codegen(context, builder, sig, args):
        _, meminfo = args

        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo
        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        return st._getvalue()

    sig = inst_type(struct_type, types.MemInfoPointer(types.voidptr))
    return sig, codegen


@intrinsic
def _meminfo_from_struct(typingctx, val):
    def codegen(context, builder, sig, args):
        [td] = sig.args
        [d] = args

        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder, value=d)
        meminfo = dstruct.meminfo
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)
        # Returns the plain MemInfo
        return meminfo
        
    sig = meminfo_type(val,)
    return sig, codegen

def _obj_cast_codegen(context, builder, val, frmty, toty, incref=True):
    ctor = cgutils.create_struct_proxy(frmty)
    
    dstruct = ctor(context, builder, value=val)
    meminfo = dstruct.meminfo
    if(incref and context.enable_nrt):
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

    st = cgutils.create_struct_proxy(toty)(context, builder)
    st.meminfo = meminfo
    
    return st._getvalue()


@intrinsic
def _cast_structref(typingctx, cast_type_ref, inst_type):
    cast_type = cast_type_ref.instance_type
    if(isinstance(inst_type, types.Optional)):
        inst_type = inst_type.type
    def codegen(context, builder, sig, args):
        _,d = args

        return _obj_cast_codegen(context, builder, d, inst_type, cast_type)
    sig = cast_type(cast_type_ref, inst_type)
    return sig, codegen


def _list_cast_codegen(context, builder, val, frmty, toty, incref=True):
    ctor = cgutils.create_struct_proxy(frmty)
    frm_st = ctor(context, builder, value=val)
    
    to_st = cgutils.create_struct_proxy(toty)(context, builder)
    meminfo, data = frm_st.meminfo, frm_st.data
    if(incref and context.enable_nrt):
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

    to_st.meminfo = meminfo
    to_st.data = data

    return to_st._getvalue()



@intrinsic
def _cast_list(typingctx, cast_type_ref, frmty):
    toty = cast_type_ref.instance_type
    if(isinstance(frmty, types.Optional)):
        frmty = frmty.type
    def codegen(context, builder, sig, args):
        _,val = args
        return _list_cast_codegen(context, builder, val, frmty, toty)

        

        # return _obj_cast_codegen(context, builder, d, inst_type, cast_type)
    sig = toty(cast_type_ref, frmty)
    return sig, codegen


    




#Seems to also work for lists
# _cast_list = _cast_structref

@njit(cache=True)
def cast_structref(typ,inst):
    return _cast_structref(typ,inst)


def _struct_from_ptr_codegen(context, builder, raw_ptr, raw_ptr_ty, inst_type):

    if(not isinstance(raw_ptr_ty, types.MemInfoPointer)):
        meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)
    else:
        meminfo = raw_ptr

    st = cgutils.create_struct_proxy(inst_type)(context, builder)
    st.meminfo = meminfo

    return impl_ret_borrowed(
        context,
        builder,
        inst_type,
        st._getvalue()
    )


@intrinsic
def _struct_from_ptr(typingctx, struct_type, raw_ptr):
    inst_type = struct_type.instance_type

    def codegen(context, builder, sig, args):
        _, raw_ptr = args
        _, raw_ptr_ty = sig.args
        return _struct_from_ptr_codegen(context, builder, raw_ptr, raw_ptr_ty, inst_type)

    sig = inst_type(struct_type, raw_ptr)
    return sig, codegen


def _list_from_ptr_codegen(context, builder, raw_ptr, list_type):
    mi = builder.inttoptr(raw_ptr, cgutils.voidptr_t)

    ctor = cgutils.create_struct_proxy(list_type)
    dstruct = ctor(context, builder)

    data_pointer = context.nrt.meminfo_data(builder, mi)
    data_pointer = builder.bitcast(data_pointer, cgutils.voidptr_t.as_pointer())

    dstruct.data = builder.load(data_pointer)
    dstruct.meminfo = mi

    return impl_ret_borrowed(
        context,
        builder,
        list_type,
        dstruct._getvalue(),
    )


@intrinsic
def _list_from_ptr(typingctx, listtyperef, raw_ptr_ty):
    """Recreate a list from a MemInfoPointer
    """
    
    list_type = listtyperef.instance_type
    
    def codegen(context, builder, sig, args):
        [_, raw_ptr] = args
        return _list_from_ptr_codegen(context, builder, raw_ptr, list_type)

    sig = list_type(listtyperef, raw_ptr_ty)
    return sig, codegen

_dict_from_ptr = _list_from_ptr





def _ptr_from_struct_codegen(context, builder, val, td, incref=True):
    ctor = cgutils.create_struct_proxy(td)
    dstruct = ctor(context, builder, value=val)
    meminfo = dstruct.meminfo

    if(incref):
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

    return builder.ptrtoint(dstruct.meminfo, cgutils.intp_t)


@intrinsic
def _raw_ptr_from_struct(typingctx, val_ty):
    if(isinstance(val_ty, types.Optional)):
        val_ty = val_ty.type

    def codegen(context, builder, sig, args):
        return _ptr_from_struct_codegen(context, builder, args[0], sig.args[0], False)

        # return res
        
    sig = i8(val_ty)
    return sig, codegen

@intrinsic
def _raw_ptr_from_struct_incref(typingctx, val_ty):
    if(isinstance(val_ty, types.Optional)):
        val_ty = val_ty.type

    def codegen(context, builder, sig, args):
        return _ptr_from_struct_codegen(context, builder, args[0], sig.args[0], True)
        
    sig = i8(val_ty)
    return sig, codegen


# @njit
# def pointer_from_struct(self):
#     return _raw_ptr_from_struct(self) 


@intrinsic
def _ptr_from_struct_incref(typingctx, val_ty):
    if(isinstance(val_ty, types.Optional)):
        val_ty = val_ty.type

    def codegen(context, builder, sig, args):
        return _ptr_from_struct_codegen(context, builder, args[0], sig.args[0], True)
        
    sig = ptr_t(val_ty)
    return sig, codegen

@intrinsic
def _ptr_to_data_ptr(typingctx, raw_ptr):
    def codegen(context, builder, sig, args):
        raw_ptr,  = args
        raw_ptr_ty,  = sig.args

        meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)
        data_ptr = context.nrt.meminfo_data(builder, meminfo)
        ret = builder.ptrtoint(data_ptr, cgutils.intp_t)

        return ret

    sig = i8(raw_ptr, )
    return sig, codegen

@intrinsic
def cast(typctx, val_typ, _typ):
    typ = _typ.instance_type
    if(isinstance(val_typ, types.Optional)):
        val_typ = val_typ.type
    codegen = None
    if(isinstance(typ, Ptr)):
        raise ValueError("Cannot use cast() to cast to ptr_t")

    if(isinstance(val_typ, (types.Integer,types.MemInfoPointer))):
        if(isinstance(typ, types.StructRef)):
            def codegen(context, builder, sig, args):
                return _struct_from_ptr_codegen(context, builder, args[0], sig.args[0], typ)

        elif(isinstance(typ, (types.ListType, types.DictType))):
            def codegen(context, builder, sig, args):
                return _list_from_ptr_codegen(context, builder, args[0], typ)

        elif(isinstance(typ, types.Integer)): # Allows ptr_t -> i8
            def codegen(context, builder, sig, args):
                return args[0]

    elif(isinstance(val_typ, types.StructRef)):
        if(isinstance(typ, types.StructRef)):
            def codegen(context, builder, sig, args):
                return _obj_cast_codegen(context, builder, args[0], sig.args[0], typ)

        elif(isinstance(typ, types.Integer)):
            def codegen(context, builder, sig, args):
                return _ptr_from_struct_codegen(context, builder, args[0], sig.args[0], False)

    elif(isinstance(val_typ, (types.ListType, types.DictType))):
        if(isinstance(typ, types.Integer)):        
            def codegen(context, builder, sig, args):
                return _ptr_from_struct_codegen(context, builder, args[0], sig.args[0], False)



    if(codegen is None):
        raise ValueError(f"No cast() from {val_typ} to {typ}.")
    return typ(val_typ,_typ), codegen


#### memcpy ####

def _meminfo_copy_unsafe(builder, nrt, meminfo):
    mod = builder.module
    fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.voidptr_t, cgutils.voidptr_t])
    fn = cgutils.get_or_insert_function(mod, fnty, "meminfo_copy_unsafe")
    fn.return_value.add_attribute("noalias")
    return builder.call(fn, [builder.bitcast(nrt, cgutils.voidptr_t), builder.bitcast(meminfo, cgutils.voidptr_t)])

@intrinsic
def _memcpy_structref(typingctx, inst_type):    
    def codegen(context, builder, signature, args):
        val = args[0]
        # print("___", inst_type)
        ctor = cgutils.create_struct_proxy(inst_type)
    
        dstruct = ctor(context, builder, value=val)
        meminfo = dstruct.meminfo
        nrt = context.nrt.get_nrt_api(builder)
        new_meminfo = _meminfo_copy_unsafe(builder, nrt, meminfo)

        inst_struct = context.make_helper(builder, inst_type)
        inst_struct.meminfo = new_meminfo
        # context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), new_meminfo)
        # context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), new_meminfo)

        return impl_ret_borrowed(context, builder, inst_type, inst_struct._getvalue())
        # return inst_struct._getvalue()

    sig = inst_type(inst_type)
    return sig, codegen


#### Dtor ####

def _meminfo_call_dtor(builder, meminfo):
    mod = builder.module
    fnty = ir.FunctionType(ir.VoidType(), [cgutils.voidptr_t])
    fn = cgutils.get_or_insert_function(mod, fnty, "NRT_MemInfo_call_dtor")
    # fn.return_value.add_attribute("noalias")
    return builder.call(fn, [builder.bitcast(meminfo, cgutils.voidptr_t)])

@intrinsic
def _call_dtor(typingctx, inst_type):
    def codegen(context, builder, signature, args):
        val = args[0]
        st = cgutils.create_struct_proxy(inst_type)(context, builder, value=val)
        # print(st)
        meminfo = st.meminfo
        _meminfo_call_dtor(builder, meminfo)
    sig = types.void(inst_type)

    return sig, codegen

#### Refcounting Utils #### 


#TODO just make _incref
@intrinsic
def _incref_structref(typingctx, inst_type):
    '''Increments the refcount '''
    def codegen(context, builder, sig, args):
        val, = args

        # try:
        # ctor = cgutils.create_struct_proxy(inst_type)
        # dstruct = ctor(context, builder, value=d)
        # context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), dstruct.meminfo)
        # except TypeError:

        context.nrt.incref(builder, inst_type, val)

        #     pass 
        

    sig = void(inst_type)
    return sig, codegen



# @njit(cache=True)
# def incref(x):
#     _incref_structref(x)


@intrinsic
def _decref_structref(typingctx,inst_type):
    def codegen(context, builder, sig, args):
        d, = args

        ctor = cgutils.create_struct_proxy(inst_type)
        dstruct = ctor(context, builder, value=d)
        meminfo = dstruct.meminfo
        context.nrt.decref(builder, types.MemInfoPointer(types.voidptr), dstruct.meminfo)

    sig = void(inst_type)
    return sig, codegen


@intrinsic
def _decref_ptr(typingctx, raw_ptr):
    def codegen(context, builder, sig, args):
        raw_ptr, = args
        meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)
        context.nrt.decref(builder, types.MemInfoPointer(types.voidptr), meminfo)

    sig = void(raw_ptr)
    return sig, codegen

@intrinsic
def _incref_ptr(typingctx, raw_ptr):
    def codegen(context, builder, sig, args):
        raw_ptr, = args
        meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)


    sig = void(raw_ptr)
    return sig, codegen

@intrinsic
def _decref_meminfo(typingctx, meminfo_type):
    def codegen(context, builder, sig, args):
        meminfo, = args
        # meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)
        context.nrt.decref(builder, types.MemInfoPointer(types.voidptr), meminfo)


    sig = void(meminfo_type)
    return sig, codegen

@intrinsic
def _incref_meminfo(typingctx, raw_ptr):
    def codegen(context, builder, sig, args):
        meminfo, = args
        # meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)


    sig = void(meminfo_type)
    return sig, codegen

@njit(cache=True)
def decref_meminfo(mi):
    _decref_meminfo(mi)

@njit(cache=True)
def incref_meminfo(mi):
    _incref_meminfo(mi)



#### Anonymous Structrefs Member Access #### 

@intrinsic
def _struct_get_attr_offset(typingctx, inst, attr):
    '''Get the offset of the attribute 'attr' from the base address of the struct
        pointed to by structref 'inst'
    '''
    attr_literal = attr.literal_value
    def codegen(context, builder, sig, args):
        inst_type,_ = sig.args
        val,_ = args

        if(not isinstance(inst_type, types.StructRef)):
            #If we get just get the type and not an instance make a dummy instance
            inst_type = inst_type.instance_type
            val = context.make_helper(builder, inst_type)._getvalue()

        #Get the base address of the struct data
        utils = _Utils(context, builder, inst_type)
        baseptr = utils.get_data_pointer(val)
        baseptr = builder.ptrtoint(baseptr, cgutils.intp_t)

        #Get the address of member for 'attr'
        dataval = utils.get_data_struct(val)
        attrptr = dataval._get_ptr_by_name(attr_literal)
        attrptr = builder.ptrtoint(attrptr, cgutils.intp_t)

        #Subtract them to get the offset
        offset = builder.sub(attrptr,baseptr)
        return offset

    sig = i8(inst,attr)
    return sig, codegen

@njit(cache=False)
def struct_get_attr_offset(inst,attr):
    return _struct_get_attr_offset(inst,literally(attr))


@intrinsic
def _struct_get_attr_ptr(typingctx, inst, attr):
    '''Get the data ptr of the attribute 'attr' in 'inst'
    '''
    attr_literal = attr.literal_value
    def codegen(context, builder, sig, args):
        inst_type,_ = sig.args
        val,_ = args

        if(not isinstance(inst_type, types.StructRef)):
            #If we get just get the type and not an instance make a dummy instance
            inst_type = inst_type.instance_type
            val = context.make_helper(builder, inst_type)._getvalue()

        #Get the base address of the struct data
        utils = _Utils(context, builder, inst_type)
        baseptr = utils.get_data_pointer(val)
        baseptr = builder.ptrtoint(baseptr, cgutils.intp_t)

        #Get the address of member for 'attr'
        dataval = utils.get_data_struct(val)
        attrptr = dataval._get_ptr_by_name(attr_literal)
        attrptr = builder.ptrtoint(attrptr, cgutils.intp_t)
        return attrptr

    sig = i8(inst,attr)
    return sig, codegen

@intrinsic
def _struct_get_data_ptr(typingctx, inst_type):
    '''get the base address of the struct pointed to by structref 'inst' '''
    def codegen(context, builder, sig, args):
        val_ty, = sig.args
        val, = args

        utils = _Utils(context, builder, val_ty)
        dataptr = utils.get_data_pointer(val)
        ret = builder.ptrtoint(dataptr, cgutils.intp_t)
        return ret

    sig = i8(inst_type,)
    return sig, codegen

@intrinsic
def _load_ptr(typingctx, typ, ptr):
    '''Get the value pointed to by 'ptr' assuming it has type 'typ' 
    '''
    inst_type = typ.instance_type
    def codegen(context, builder, sig, args):
        _,ptr = args
        llrtype = context.get_value_type(inst_type)
        ptr = builder.inttoptr(ptr, ll_types.PointerType(llrtype))

        # dm_item = context.data_model_manager[inst_type]
        # ret = dm_item.load_from_data_pointer(builder, ptr)

        ret = builder.load(ptr)
        # if context.enable_nrt:
        #     context.nrt.incref(builder, llrtype, ret)
        return imputils.impl_ret_borrowed(context, builder, inst_type, ret)

    sig = inst_type(typ,ptr)
    return sig, codegen

@intrinsic
def _store(typingctx, typ, ptr, val):
    '''Get the value pointed to by 'ptr' assuming it has type 'typ' 
    '''
    inst_type = typ.instance_type
    def codegen(context, builder, sig, args):
        _,ptr,val = args
        llrtype = context.get_value_type(inst_type)
        ptr = builder.inttoptr(ptr, ll_types.PointerType(llrtype))
        builder.store(val, ptr)
        
        # if context.enable_nrt:
        #     context.nrt.incref(builder, llrtype, val)
        

    sig = types.void(typ,ptr,val)
    return sig, codegen

@intrinsic
def _store_safe(typingctx, typ, _ptr, _val):
    '''Get the value pointed to by 'ptr' assuming it has type 'typ' 
    '''
    inst_type = typ.instance_type
    def codegen(context, builder, sig, args):
        _,ptr,val = args
        llrtype = context.get_value_type(inst_type)
        ptr = builder.inttoptr(ptr, ll_types.PointerType(llrtype))

        old_val = builder.load(ptr)
        builder.store(val, ptr)
        
        if context.enable_nrt:
            context.nrt.incref(builder, inst_type, val)
            context.nrt.decref(builder, inst_type, old_val)
        
    sig = types.void(typ, _ptr, _val)
    return sig, codegen

@intrinsic
def _memcpy(typingctx, dst, src, nbytes):
    def codegen(context, builder, sig, args):
        # print("RECOMPILE")
        dst, src, nbytes = args
        dst = builder.inttoptr(dst, cgutils.voidptr_t)
        src = builder.inttoptr(src, cgutils.voidptr_t)
        # one = context.get_constant(types.intp, 1)

        # cgutils.raw_memcpy(builder, dst, src, one, nbytes, align)
        # return context.get_dummy_value()

        if isinstance(nbytes, int):
            nbytes = ir.Constant(cgutils.intp_t, nbytes)

        memcpy = builder.module.declare_intrinsic('llvm.memcpy',
                                                  [cgutils.voidptr_t, cgutils.voidptr_t, cgutils.intp_t])
        is_volatile = cgutils.true_bit
        builder.call(memcpy, [dst,
                              src,
                              nbytes,
                              is_volatile])


    sig = types.void(i8,i8,i8)
    return sig, codegen


@intrinsic
def _nullify_attr(typingctx, struct_type, _attr):
    # print(struct_type, _attr)
    attr = _attr.literal_value
    def codegen(context, builder, sig, args):
        [st,_] = args
        utils = _Utils(context, builder, struct_type)
        dataval = utils.get_data_struct(st)
        val_data_ptr = dataval._get_ptr_by_name(attr)
        val_type = dataval._datamodel.get_type(attr)
        val_type = context.get_value_type(val_type)
        builder.store(cgutils.get_null_value(val_type), val_data_ptr)

        # if context.enable_nrt:
        #     context.nrt.decref(builder, val_type, dataval)

    sig = types.void(struct_type, _attr)
    return sig, codegen

@intrinsic
def _attr_is_null(typingctx, struct_type, _attr):
    # print(struct_type, _attr)
    attr = _attr.literal_value
    def codegen(context, builder, sig, args):
        [st,_] = args
        utils = _Utils(context, builder, struct_type)
        dataval = utils.get_data_struct(st)
        val_data_ptr = dataval._get_ptr_by_name(attr)
        val_data_ptr = builder.bitcast(val_data_ptr, cgutils.intp_t.as_pointer())
        val = builder.load(val_data_ptr)
        return cgutils.is_null(builder,val)
        
    sig = types.boolean(struct_type, _attr)
    return sig, codegen


@intrinsic
def _func_from_address(typingctx, func_type_ref, addr):
    '''Recovers a function from it's signature and address '''
    
    func_type = func_type_ref.instance_type
    def codegen(context, builder, sig, args):
        _, addr = args

        sfunc = cgutils.create_struct_proxy(func_type)(context, builder)

        llty = context.get_value_type(types.voidptr)
        addr_ptr = builder.inttoptr(addr,llty)

        sfunc.addr = addr_ptr
        return sfunc._getvalue()



    sig = func_type(func_type_ref, addr)
    return sig, codegen


@intrinsic
def _address_from_func(typingctx, func_type_ref, func):
    '''Gets the address of a function_type'''
    func_type = func_type_ref.instance_type
    def codegen(context, builder, sig, args):
        _, func = args
        sfunc = cgutils.create_struct_proxy(func_type)(context, builder, func)
        addr = builder.ptrtoint(sfunc.addr, cgutils.intp_t)
        return addr

    sig = i8(func_type_ref, func_type)
    return sig, codegen


#### List Intrisics ####

ll_list_type = cgutils.voidptr_t
ll_listiter_type = cgutils.voidptr_t
ll_voidptr_type = cgutils.voidptr_t
ll_status = cgutils.int32_t
ll_ssize_t = cgutils.intp_t
ll_bytes = cgutils.voidptr_t


def base_ptr_from_container_data(builder,cd):
    fnty = ir.FunctionType(
        ll_voidptr_type,
        [ll_list_type],
    )
    fname = 'numba_list_base_ptr'
    fn = cgutils.get_or_insert_function(builder.module, fnty, fname)
    fn.attributes.add('alwaysinline')
    fn.attributes.add('nounwind')
    fn.attributes.add('readonly')

    base_ptr = builder.call(
            fn,
            [cd,],
        )
    return base_ptr

# def filter_l_ty(l_ty):


@intrinsic
def _list_base(typingctx, l_ty):
    is_none = isinstance(l_ty.item_type, types.NoneType)
    sig = i8(l_ty,)
    def codegen(context, builder, sig, args):
        [l, ] = args
        #The type can be anything for the sake of getting the base ptr.
        tl = ListType(i8) 
        lp = _container_get_data(context, builder, tl, l)

        base_ptr = base_ptr_from_container_data(builder, lp)

        out = builder.ptrtoint(base_ptr, cgutils.intp_t)

        return out
    return sig, codegen

@intrinsic
def _list_base_from_ptr(typingctx, ptr_ty):
    sig = i8(ptr_ty,)
    def codegen(context, builder, sig, args):
        [ptr, ] = args

        mi = builder.inttoptr(ptr, cgutils.voidptr_t)

        data_pointer = context.nrt.meminfo_data(builder, mi)
        data_pointer = builder.bitcast(data_pointer, cgutils.voidptr_t.as_pointer())
        data = builder.load(data_pointer)

        base_ptr = base_ptr_from_container_data(builder, data)
        out = builder.ptrtoint(base_ptr, cgutils.intp_t)
        return out
    return sig, codegen

@intrinsic
def _listtype_sizeof_item(typingctx, l_ty):
    sig = i8(l_ty,)
    tl = l_ty.instance_type
    def codegen(context, builder, sig, args):
        llty = context.get_data_type(tl.item_type)
        return cgutils.sizeof(builder,llty.as_pointer())
    return sig, codegen


@njit
def listtype_sizeof_item(lt):
    return _listtype_sizeof_item(lt)

### Array Intrinsics ###

from numba.np.arrayobj import make_array

def _get_array_data_ptr_codegen(context, builder, sig, args, incref=True):
    [arr_typ] = sig.args
    [arr] = args
    # does create_struct_proxy plus some other stuff
    arr_st = make_array(arr_typ)(context, builder, arr)
    # arr_st = cgutils.create_struct_proxy(arr_typ)(context, builder, arr)
    if context.enable_nrt and incref:
        context.nrt.incref(builder, arr_typ, arr)
    return builder.ptrtoint(arr_st.data, cgutils.intp_t)


@intrinsic
def _get_array_data_ptr_incref(typingctx, arr_typ):
    def codegen(context, builder, sig, args):
        return _get_array_data_ptr_codegen(context, builder, sig, args, True)
                
    sig = ptr_t(arr_typ)
    return sig, codegen

@intrinsic
def _get_array_raw_data_ptr(typingctx, arr_typ):
    def codegen(context, builder, sig, args):
        return _get_array_data_ptr_codegen(context, builder, sig, args, False)

    sig = i8(arr_typ)
    return sig, codegen

@intrinsic
def _get_array_raw_data_ptr_incref(typingctx, arr_typ):
    def codegen(context, builder, sig, args):
        return _get_array_data_ptr_codegen(context, builder, sig, args, True)
                
    sig = i8(arr_typ)
    return sig, codegen


@intrinsic
def _as_void(typingctx, src):
    """ returns a void pointer from a given memory address """
    from numba.core import types, cgutils
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen

@njit(cache=True)
def _arr_from_data_ptr(d_ptr, shape, dtype):
    return numba.carray(_as_void(d_ptr),shape,dtype=dtype)


#### Automatic Variable Aliasing ####

def assign_to_alias_in_parent_frame(x,alias):
    if(alias is not None): 
        # Binds this instance globally in the calling python context 
        #  so that it is bound to a variable named whatever alias was set to
        inspect.stack()[2][0].f_globals[alias] = x


##### Helpful context for timing code ####
import time
class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.6f} ms')

##### Stuff for finding memleaks #####
from numba.core.runtime import rtsys, _nrt_python
import gc

class NRTStatsEnabledMeta(type):
    def __enter__(self):
        _nrt_python.memsys_enable_stats()

    def __exit__(self, *args):
        _nrt_python.memsys_disable_stats()        

class NRTStatsEnabled(metaclass=NRTStatsEnabledMeta):
    def __new__(cls):
        return cls

def used_bytes(garbage_collect=True):
    if(garbage_collect): gc.collect()
    stats = rtsys.get_allocation_stats()
    return stats.alloc-stats.free


##### Cacheable Version of Typed List Generation / Iteration ####

@njit(cache=True)
def _new_list(typ):
    return List.empty_list(typ)

@njit(cache=True)
def _append_to_list(l, x):
    return l.append(x)

def as_typed_list(typ, it):
    l = _new_list(typ)
    for x in it:
        _append_to_list(l,x)
    return l

@njit(cache=True)
def len_typed_list(l):
    return len(l)

@njit(cache=True)
def getitem_typed_list(l,i):
    return l[i]

def iter_typed_list(lst):
    if(isinstance(lst, (List,))):
        for i in range(len_typed_list(lst)):
            yield getitem_typed_list(lst,i)
    else:
         return iter(lst)
            
##### Make Tuple of StructRefs from array of ptrs ####

# @intrinsic
# def _struct_tuple_from_pointer_arr(typingctx, struct_types, ptr_arr):
#     ''' Takes a tuple of fact types and a ptr_array i.e. an i8[::1] and outputs 
#         the facts pointed to, casted to the appropriate types '''
#     # print(">>",struct_types)
#     _struct_types = struct_types
#     if(isinstance(struct_types, types.TypeRef )): _struct_types = struct_types.instance_type

#     typs = tuple([x.instance_type for x in _struct_types])
#     print("<<", typs)
#     # if(isinstance(_struct_types, types.UniTuple)):
#     #     typs = tuple([_struct_types.dtype.instance_type] * _struct_types.count)
#     #     out_type =  types.UniTuple(_struct_types.dtype.instance_type,_struct_types.count)
#     # else:
#     #     # print(struct_types.__dict__)
#     #     typs = tuple([x.instance_type for x in _struct_types.types])
#     #     out_type =  types.Tuple(typs)
#     # print(typs)
#     # print(out_type)
    
#     sig = out_type(struct_types,i8[::1])
#     def codegen(context, builder, sig, args):
#         _,ptrs = args

#         vals = []
#         ary = make_array(i8[::1])(context, builder, value=ptrs)
#         for i, inst_type in enumerate(typs):
#             i_val = context.get_constant(types.intp, i)

#             # Same as _struct_from_ptr
#             raw_ptr = _getitem_array_single_int(context,builder,i8,i8[::1],ary,i_val)
#             meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)

#             context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

#             st = cgutils.create_struct_proxy(inst_type)(context, builder)
#             st.meminfo = meminfo

            

#             vals.append(st._getvalue())


#         ret = context.make_tuple(builder,out_type,vals)
#         return ret#impl_ret_borrowed(context, builder, sig.return_type, ret)
#         # return 

#     return sig,codegen

@intrinsic
def _tuple_setitem(typingctx, tup, i, val):
    #NOTE: Haven't properly tested... 'i' must be constant 
    def codegen(context, builder, sig, args):
        tup,i,val = args
        return builder.insert_value(tup, val, i)
        # print(llty.__dict__)
        
    return tup(tup, i, val), codegen

@intrinsic
def _sizeof_type(typingctx, typ_ref):
    typ = typ_ref.instance_type
    def codegen(context, builder, sig, args):
        llty = context.get_data_type(typ)
        return context.get_constant(types.intp, context.get_abi_sizeof(llty))
        # print(llty.__dict__)
        
    return i8(typ_ref,), codegen


@intrinsic
def _get_member_offsets(typingctx, typ_ref):
    typ = typ_ref.instance_type
    n_members = len(typ._fields)
    def codegen(context, builder, sig, args):
        llty = context.get_data_type(typ)

        return context.get_constant(types.intp, context.get_abi_sizeof(llty))
        # print(llty.__dict__)
        
    return types.UniTuple(u2,n_members)(typ_ref,), codegen



@intrinsic
def _get_member_offset(typingctx, struct_type, attr_literal):
    if(not isinstance(attr_literal,types.Literal)):
        return

    attr = attr_literal.literal_value

    # print("ATTTR",attr)

    def codegen(context, builder, sig, args):
        [st,_] = args
        # print("C")

        utils = _Utils(context, builder, struct_type)
        # print("A")
        baseptr = utils.get_data_pointer(st)
        baseptr_val = builder.ptrtoint(baseptr, cgutils.intp_t)
        # print("B")
        dataval = utils.get_data_struct(st)
        index_of_member = dataval._datamodel.get_field_position(attr)

        # print(index_of_member)

        member_ptr = builder.gep(baseptr, [cgutils.int32_t(0), cgutils.int32_t(index_of_member)], inbounds=True)
        member_ptr = builder.ptrtoint(member_ptr, cgutils.intp_t)
        offset = builder.trunc(builder.sub(member_ptr, baseptr_val), cgutils.ir.IntType(16))
        # print(offset)

        return offset

    sig = u2(struct_type, attr_literal)
    return sig, codegen

@intrinsic
def _struct_tuple_from_pointer_arr(typingctx, struct_types, ptr_arr):
    ''' Takes a tuple of fact types and a ptr_array i.e. an i8[::1] and outputs 
        the facts pointed to, casted to the appropriate types '''
    # print(">>",struct_types)
    _struct_types = struct_types
    if(isinstance(struct_types,types.TypeRef)): struct_types = struct_types.instance_type
    if(isinstance(struct_types, types.UniTuple)):
        typs = tuple([struct_types.dtype.instance_type] * struct_types.count)
        out_type =  types.UniTuple(struct_types.dtype.instance_type,struct_types.count)
    else:
        # print(struct_types.__dict__)
        typs = tuple([x.instance_type for x in struct_types.types])
        out_type =  Tuple(typs)
    # print(out_type)
    
    sig = out_type(_struct_types,i8[::1])
    def codegen(context, builder, sig, args):
        _,ptrs = args

        vals = []
        ary = make_array(i8[::1])(context, builder, value=ptrs)
        for i, inst_type in enumerate(typs):
            i_val = context.get_constant(types.intp, i)

            # Same as _struct_from_ptr
            raw_ptr = _getitem_array_single_int(context,builder,i8,i8[::1],ary,i_val)
            meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)

            st = cgutils.create_struct_proxy(inst_type)(context, builder)
            st.meminfo = meminfo

            context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

            vals.append(st._getvalue())


        
        return context.make_tuple(builder,out_type,vals)

    return sig,codegen

#TODO: Might need to add an incref in here to handle objects
@intrinsic
def _tuple_from_data_ptrs(typingctx, member_types, ptr_arr):
    if(isinstance(member_types,types.TypeRef)): member_types = member_types.instance_type
    if(isinstance(member_types, types.UniTuple)):
        typs = tuple([member_types.dtype.instance_type] * member_types.count)
        out_type =  types.UniTuple(member_types.dtype.instance_type,member_types.count)
    else:
        # print(struct_types.__dict__)
        typs = tuple([x.instance_type for x in struct_types.types])
        out_type =  Tuple(typs)
    
    print(member_types)
    # out_type = member_types.instance_type
    # out_type = tuple([x.instance_type for x in member_types])
    # out_type =  Tuple(out_type)
    def codegen(context, builder, sig, args):
        _,ptrs = args
        vals = []
        ary = make_array(i8[::1])(context, builder, value=ptrs)
        for i, inst_type in enumerate(out_type):
            i_val = context.get_constant(types.intp, i)

            raw_ptr = _getitem_array_single_int(context,builder,i8,i8[::1],ary,i_val)

            llrtype = context.get_value_type(inst_type)
            data_pointer = builder.inttoptr(raw_ptr, ll_types.PointerType(llrtype))
            vals.append(builder.load(data_pointer))

        return context.make_tuple(builder, out_type, vals)
    sig = out_type(member_types, ptr_arr)
    return sig, codegen


# Check for https://github.com/numba/numba/issues/6993
def check_issue_6993():
    @njit
    def foo(b):
        c = 0
        for i in range(len(b)):
            yield i
            c += 1

    @njit(cache=True)
    def bar(b):
        c = 0 
        for i in foo(b):
            c += i
        return c

    @njit(cache=True)
    def get_list():
        return List([1,3,4,5])

    lst = get_list() #BOOP("A",5)
    bar(lst)
    return lst._opaque.refcount > 1


# ----------------------------------
# : new_w_del

def imp_dtor_w_del(context, module, instance_type, del_fn_symbol_name):
    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = ir.FunctionType(ir.VoidType(),
                                     [llvoidptr, llsize, llvoidptr])

    fname = "_DelDtor.{0}".format(instance_type.name)
    dtor_fn = cgutils.get_or_insert_function(module, dtor_ftype, fname)
    if dtor_fn.is_declaration:
        # Define
        builder = ir.IRBuilder(dtor_fn.append_basic_block())

        alloc_fe_type = instance_type.get_data_type()
        alloc_type = context.get_value_type(alloc_fe_type)

        ptr = builder.bitcast(dtor_fn.args[0], alloc_type.as_pointer())
        data = context.make_helper(builder, alloc_fe_type, ref=ptr)

        extra_fn_type = ir.FunctionType(ir.VoidType(),[llvoidptr])
        extra_fn = cgutils.get_or_insert_function(module, extra_fn_type, del_fn_symbol_name)
        builder.call(extra_fn, [builder.bitcast(dtor_fn.args[0], cgutils.voidptr_t)])

        context.nrt.decref(builder, alloc_fe_type, data._getvalue())

        builder.ret_void()

    return dtor_fn

@intrinsic
def new_w_del(typingctx, struct_type, _del_fn_symbol_name):
    inst_type = struct_type.instance_type
    del_fn_symbol_name = _del_fn_symbol_name.literal_value
    def codegen(context, builder, signature, args):
        model = context.data_model_manager[inst_type.get_data_type()]
        alloc_type = model.get_value_type()
        alloc_size = context.get_abi_sizeof(alloc_type)

        meminfo = context.nrt.meminfo_alloc_dtor(
            builder,
            context.get_constant(types.uintp, alloc_size),
            imp_dtor_w_del(context, builder.module, inst_type, del_fn_symbol_name),
        )
        data_pointer = context.nrt.meminfo_data(builder, meminfo)
        data_pointer = builder.bitcast(data_pointer, alloc_type.as_pointer())

        # Nullify all data
        builder.store(cgutils.get_null_value(alloc_type), data_pointer)

        inst_struct = context.make_helper(builder, inst_type)
        inst_struct.meminfo = meminfo

        return inst_struct._getvalue()

    sig = inst_type(struct_type, _del_fn_symbol_name)
    return sig, codegen


#--------------------------------------------------------
#: get/set Globals

def get_or_make_global(context, builder, typ,name):
    mod = builder.module
    try:
        # Search for existing global
        gv = mod.get_global(name)
    except KeyError:
        # Inject the symbol if not already exist.
        # gv = ir.GlobalVariable(mod, context.get_value_type(typ), name=name)
        lltyp = context.get_value_type(typ) 
        gv = ir.GlobalVariable(mod, lltyp, name=name)
        gv.linkage = 'common'
        gv.initializer = cgutils.get_null_value(gv.type.pointee)
    return gv

@intrinsic
def _get_global(typingctx, _typ, _name):
    typ = _typ.instance_type
    name = _name.literal_value
    def codegen(context, builder, sig, args):
        gv = get_or_make_global(context,builder,typ, name)        
        v = builder.load(gv)
        context.nrt.incref(builder, typ, v)
        return v
    sig = typ(_typ, _name)
    return sig, codegen

@intrinsic
def _set_global(typingctx, _typ, _name, val):
    typ = _typ.instance_type
    name = _name.literal_value
    def codegen(context, builder, sig, args):
        val = args[-1]
        gv = get_or_make_global(context,builder,typ, name)
        builder.store(val, gv)
    sig = types.void(_typ, _name, val)
    return sig, codegen


# --------------------------------
# : _call_fast implementation
@intrinsic
def _call_fast(typingctx, func, args):
    '''Calls a first-class FunctionType with attrs=("nounwind",'readnone') '''    
    func_type = func
    def codegen(context, builder, sig, _args):        
        _, inp_types = sig.args
        arg_types = func_type.signature.args
        func, args = _args

        # Unpack args and cast types to the types specified in the func signature
        args = cgutils.unpack_tuple(builder, args, len(inp_types))
        args = [context.cast(builder, a, it, at) for a, it, at in zip(args, inp_types, arg_types)]
            
        # Grab the function address
        sfunc = cgutils.create_struct_proxy(func_type)(context, builder, func)
        llty = context.get_value_type(func_type.ftype)
        fn_addr = builder.bitcast(sfunc.addr, llty)        

        # Call the function with special attributes
        ret = builder.call(fn_addr, args, cconv=func_type.cconv, attrs=("nounwind",'readnone'))
        return ret 
    sig = func.signature.return_type(func, args)
    return sig, codegen

@intrinsic
def _call_nounwind(typingctx, func, args):
    '''Calls a first-class FunctionType with attrs=("nounwind",'readnone') '''    
    func_type = func
    def codegen(context, builder, sig, _args):        
        _, inp_types = sig.args
        arg_types = func_type.signature.args
        func, args = _args

        # Unpack args and cast types to the types specified in the func signature
        args = cgutils.unpack_tuple(builder, args, len(inp_types))
        args = [context.cast(builder, a, it, at) for a, it, at in zip(args, inp_types, arg_types)]
            
        # Grab the function address
        sfunc = cgutils.create_struct_proxy(func_type)(context, builder, func)
        llty = context.get_value_type(func_type.ftype)
        fn_addr = builder.bitcast(sfunc.addr, llty)        

        # Call the function with special attributes
        ret = builder.call(fn_addr, args, cconv=func_type.cconv, attrs=("nounwind"))
        return ret 
    sig = func.signature.return_type(func, args)
    return sig, codegen


@intrinsic
def _tuple_getitem(typingctx, tup, i):
    tup_type = tup
    ind = i.literal_value
    print(tup, i, tup.types[ind])
    def codegen(context, builder, sig, args):
        tup, i = args
        return builder.extract_value(tup, ind)
    return tup.types[ind](tup, i), codegen
        # args = cgutils.unpack_tuple(builder, args, len(inp_types))


### Timing --- Only works for Linux ###
from sys import platform
if('linux' in platform):
    import numpy as np
    import ctypes
    import time

    CLOCK_MONOTONIC = 0x1
    clock_gettime_proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_long))
    pybind = ctypes.CDLL(None)
    clock_gettime_addr = pybind.clock_gettime
    clock_gettime_fn_ptr = clock_gettime_proto(clock_gettime_addr)


    @njit(cache=True)
    def timenow():
        timespec = np.zeros(2, dtype=np.int64)
        clock_gettime_fn_ptr(CLOCK_MONOTONIC, timespec.ctypes)
        ts = timespec[0]
        tns = timespec[1]
        return np.float64(ts) + 1e-9 * np.float64(tns)
