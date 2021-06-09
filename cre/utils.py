from numba import types, njit, u1,u2,u4,u8, i8,i2, literally
from numba.types import Tuple, void, ListType
from numba.experimental.structref import _Utils, imputils
from numba.extending import intrinsic
from numba.core import cgutils
from llvmlite.ir import types as ll_types
from llvmlite import ir
import inspect
import numpy as np 
import numba
from numba.typed.typedobjectutils import _container_get_data

#### deref_type ####

_deref_type = np.dtype([('type', np.uint8), ('offset', np.int64)])
deref_type = numba.from_dtype(_deref_type)

OFFSET_TYPE_ATTR = 0
OFFSET_TYPE_LIST = 1 

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
    # if(isinstance(frmty,types.Optional)):
    #     val = val.data
    # print(frmty, ":::",toty)
    dstruct = ctor(context, builder, value=val)
    meminfo = dstruct.meminfo
    if(incref):
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

    st = cgutils.create_struct_proxy(toty)(context, builder)
    st.meminfo = meminfo
    
    return st._getvalue()


@intrinsic
def _cast_structref(typingctx, cast_type_ref, inst_type):
    cast_type = cast_type_ref.instance_type
    def codegen(context, builder, sig, args):
        _,d = args

        # ctor = cgutils.create_struct_proxy(inst_type)
        # dstruct = ctor(context, builder, value=d)
        # meminfo = dstruct.meminfo
        # context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        # st = cgutils.create_struct_proxy(cast_type)(context, builder)
        # st.meminfo = meminfo
        
        # return st._getvalue()
        return _obj_cast_codegen(context, builder, d, inst_type, cast_type)
    sig = cast_type(cast_type_ref, inst_type)
    return sig, codegen

#Seems to also work for lists
_cast_list = _cast_structref

@njit(cache=True)
def cast_structref(typ,inst):
    return _cast_structref(typ,inst)


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



def _pointer_from_struct_codegen(context, builder, val, td,incref=True):
    ctor = cgutils.create_struct_proxy(td)
    dstruct = ctor(context, builder, value=val)
    meminfo = dstruct.meminfo

    if(incref):
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

    return builder.ptrtoint(dstruct.meminfo, cgutils.intp_t)


@intrinsic
def _pointer_from_struct(typingctx, val):
    def codegen(context, builder, sig, args):
        [td] = sig.args
        [d] = args

        return _pointer_from_struct_codegen(context, builder, d, td, False)

        # return res
        
    sig = i8(val,)
    return sig, codegen

@intrinsic
def _pointer_from_struct_incref(typingctx, val):
    def codegen(context, builder, sig, args):
        [td] = sig.args
        [d] = args

        return _pointer_from_struct_codegen(context, builder, d, td, True)
        
    sig = i8(val,)
    return sig, codegen

@intrinsic
def _pointer_to_data_pointer(typingctx, raw_ptr):
    def codegen(context, builder, sig, args):
        raw_ptr,  = args
        raw_ptr_ty,  = sig.args

        meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)
        data_ptr = context.nrt.meminfo_data(builder, meminfo)
        ret = builder.ptrtoint(data_ptr, cgutils.intp_t)

        return ret

    

    sig = i8(raw_ptr, )
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

@intrinsic
def _incref_pointer(typingctx, raw_ptr):
    def codegen(context, builder, sig, args):
        raw_ptr, = args
        meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)


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

@njit(cache=False)
def struct_get_attr_offset(inst,attr):
    return _struct_get_attr_offset(inst,literally(attr))

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


@intrinsic
def _func_from_address(typingctx, func_type_ref, addr):
    '''Recovers a function from it's signature and address '''
    
    func_type = func_type_ref.instance_type
    print(func_type)
    def codegen(context, builder, sig, args):
        _, addr = args

        pyapi = context.get_python_api(builder)
        sfunc = cgutils.create_struct_proxy(func_type)(context, builder)

        llty = context.get_value_type(types.voidptr)
        addr_ptr = builder.inttoptr(addr,llty)

        sfunc.addr = addr_ptr
        return sfunc._getvalue()



    sig = func_type(func_type_ref, addr)
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
    fn = builder.module.get_or_insert_function(fnty, fname)
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
def _list_base(typingctx, l_ty):#, w_ty, index_ty):
    is_none = isinstance(l_ty.item_type, types.NoneType)
    sig = i8(l_ty,)
    def codegen(context, builder, sig, args):
        # [tl,] = sig.args
        [l, ] = args
        #The type can be anything for the sake of getting the base ptr
        tl = ListType(i8) 
        lp = _container_get_data(context, builder, tl, l)

        base_ptr = base_ptr_from_container_data(builder, lp)

        out = builder.ptrtoint(base_ptr, cgutils.intp_t)

        return out
    return sig, codegen

@intrinsic
def _list_base_from_ptr(typingctx, ptr_ty):#, w_ty, index_ty):
    # is_none = isinstance(l_ty.item_type, types.NoneType)
    sig = i8(i8,)
    def codegen(context, builder, sig, args):
        # [tl,] = sig.args
        [ptr, ] = args

        typ = ListType(i8) 

        mi = builder.inttoptr(ptr, cgutils.voidptr_t)

        ctor = cgutils.create_struct_proxy(typ)
        dstruct = ctor(context, builder)
# 
        data_ptr = context.nrt.meminfo_data(builder, mi)
        data_ptr = builder.bitcast(data_ptr, ll_list_type.as_pointer())

        data = builder.load(data_ptr)
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


#### Automatic Variable Aliasing ####

def assign_to_alias_in_parent_frame(x,alias):
    if(alias is not None): 
        # Binds this instance globally in the calling python context 
        #  so that it is bound to a variable named whatever alias was set to
        inspect.stack()[2][0].f_globals[alias] = x

