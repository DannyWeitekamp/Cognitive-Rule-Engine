from numba import types, njit, u1,u2,u4,u8, i8,i2
from numba.types import Tuple, void
from numba.experimental.structref import _Utils, imputils
from numba.extending import intrinsic
from numba.core import cgutils
from llvmlite.ir import types as ll_types


#### idrec encoding ####

@njit(Tuple([u2,u8,u1])(u8),cache=True)
def decode_idrec(idrec):
    t_id = idrec >> 48
    f_id = (idrec >> 8) & 0x000FFFFF
    a_id = idrec & 0xF
    return (t_id, f_id, a_id)


@njit(u8(u2,u8,u1),cache=True)
def encode_idrec(t_id, f_id, a_id):
    return (t_id << 48) | (f_id << 8) | a_id

meminfo_type = types.MemInfoPointer(types.voidptr)

@intrinsic
def lower_setattr(typingctx, inst_type, attr_type, val_type):
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


@intrinsic
def _cast_structref(typingctx, cast_type_ref, inst_type):
    # inst_type = struct_type.instance_type
    cast_type = cast_type_ref.instance_type
    def codegen(context, builder, sig, args):
        # [td] = sig.args
        _,d = args

        ctor = cgutils.create_struct_proxy(inst_type)
        dstruct = ctor(context, builder, value=d)
        meminfo = dstruct.meminfo
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        st = cgutils.create_struct_proxy(cast_type)(context, builder)
        st.meminfo = meminfo
        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        # context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        return st._getvalue()
    sig = cast_type(cast_type_ref, inst_type)
    return sig, codegen


@intrinsic
def _struct_from_pointer(typingctx, struct_type, raw_ptr):
    inst_type = struct_type.instance_type

    def codegen(context, builder, sig, args):
        _, raw_ptr = args
        _, raw_ptr_ty = sig.args

        meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)

        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo
        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        return st._getvalue()

    sig = inst_type(struct_type, raw_ptr)
    return sig, codegen

@intrinsic
def _pointer_from_struct(typingctx, val):
    def codegen(context, builder, sig, args):
        [td] = sig.args
        [d] = args

        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder, value=d)
        meminfo = dstruct.meminfo

        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        # context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        res = builder.ptrtoint(dstruct.meminfo, cgutils.intp_t)

        return res
        
    sig = i8(val,)
    return sig, codegen

@intrinsic
def _pointer_from_struct_incref(typingctx, val):
    def codegen(context, builder, sig, args):
        [td] = sig.args
        [d] = args

        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder, value=d)
        meminfo = dstruct.meminfo

        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        res = builder.ptrtoint(dstruct.meminfo, cgutils.intp_t)

        return res
        
    sig = i8(val,)
    return sig, codegen


#### Refcounting Utils #### 


@intrinsic
def _incref_structref(typingctx,inst_type):
    '''Increments the refcount '''
    def codegen(context, builder, sig, args):
        d, = args

        ctor = cgutils.create_struct_proxy(inst_type)
        dstruct = ctor(context, builder, value=d)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), dstruct.meminfo)

    sig = void(inst_type)
    return sig, codegen


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
def _decref_pointer(typingctx, raw_ptr):
    def codegen(context, builder, sig, args):
        raw_ptr, = args
        meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)
        context.nrt.decref(builder, types.MemInfoPointer(types.voidptr), meminfo)


    sig = void(raw_ptr)
    return sig, codegen


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

@intrinsic
def _struct_get_data_pointer(typingctx, inst_type):
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
def _load_pointer(typingctx, typ, ptr):
    '''Get the value pointed to by 'ptr' assuming it has type 'typ' 
    '''
    inst_type = typ.instance_type
    def codegen(context, builder, sig, args):
        _,ptr = args
        llrtype = context.get_value_type(inst_type)
        ptr = builder.inttoptr(ptr, ll_types.PointerType(llrtype))
        ret = builder.load(ptr)
        return imputils.impl_ret_borrowed(context, builder, inst_type, ret)

    sig = inst_type(typ,ptr)
    return sig, codegen
