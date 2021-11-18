import numba
from numba import types, njit, u1,u2,u4,u8, i8,i2, literally
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
from numba.typed.typedobjectutils import _container_get_data
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
def downcast(context, builder, fromty, toty, val):
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


#### deref_type ####

_deref_type = np.dtype([('type', np.uint8),  ('a_id', np.uint8), ('fact_num', np.int64), ('offset', np.int64)])
deref_type = numba.from_dtype(_deref_type)

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
    if(incref and context.enable_nrt):
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
def _struct_from_ptr(typingctx, struct_type, raw_ptr):
    inst_type = struct_type.instance_type

    def codegen(context, builder, sig, args):
        _, raw_ptr = args
        _, raw_ptr_ty = sig.args

        meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)

        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo


        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        # context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        return impl_ret_borrowed(
            context,
            builder,
            inst_type,
            st._getvalue()
        )

    sig = inst_type(struct_type, raw_ptr)
    return sig, codegen




@intrinsic
def _list_from_ptr(typingctx, listtyperef, raw_ptr_ty):
    """Recreate a list from a MemInfoPointer
    """
    
    list_type = listtyperef.instance_type
    
    def codegen(context, builder, sig, args):
        # [tdref, _] = sig.args
        # td = tdref.instance_type
        [_, raw_ptr] = args

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
        ret = builder.load(ptr)
        return imputils.impl_ret_borrowed(context, builder, inst_type, ret)

    sig = inst_type(typ,ptr)
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
    sig = i8(ptr_ty,)
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

### Array Intrinsics ###

from numba.np.arrayobj import make_array
@intrinsic
def _get_array_data_ptr(typingctx, arr_typ):
    def codegen(context, builder, sig, args):
        [arr_typ] = sig.args
        [arr] = args
        # does create_struct_proxy plus some other stuff
        arr_st = make_array(arr_typ)(context, builder, arr)
        # arr_st = cgutils.create_struct_proxy(arr_typ)(context, builder, arr)
        if context.enable_nrt:
            context.nrt.incref(builder, arr_typ, arr)
        return builder.ptrtoint(arr_st.data, cgutils.intp_t)
        
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
        print(f'{self.name}: {self.t1-self.t0:.2f} ms')


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
    if(isinstance(lst, (List))):
        for i in range(len_typed_list(lst)):
            yield getitem_typed_list(lst,i)
    else:
         return iter(lst)
            
##### Make Tuple of StructRefs from array of ptrs ####

@intrinsic
def _struct_tuple_from_pointer_arr(typingctx, struct_types, ptr_arr):
    ''' Takes a tuple of fact types and a ptr_array i.e. an i8[::1] and outputs 
        the facts pointed to, casted to the appropriate types '''
    # print(">>",struct_types)
    _struct_types = struct_types
    if(isinstance(struct_types, types.TypeRef )): _struct_types = struct_types.instance_type

    if(isinstance(_struct_types, types.UniTuple)):
        typs = tuple([_struct_types.dtype.instance_type] * _struct_types.count)
        out_type =  types.UniTuple(_struct_types.dtype.instance_type,_struct_types.count)
    else:
        # print(struct_types.__dict__)
        typs = tuple([x.instance_type for x in _struct_types.types])
        out_type =  types.Tuple(typs)
    # print(typs)
    # print(out_type)
    
    sig = out_type(struct_types,i8[::1])
    def codegen(context, builder, sig, args):
        _,ptrs = args

        vals = []
        ary = make_array(i8[::1])(context, builder, value=ptrs)
        for i, inst_type in enumerate(typs):
            i_val = context.get_constant(types.intp, i)

            # Same as _struct_from_ptr
            raw_ptr = _getitem_array_single_int(context,builder,i8,i8[::1],ary,i_val)
            meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)

            context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

            st = cgutils.create_struct_proxy(inst_type)(context, builder)
            st.meminfo = meminfo

            

            vals.append(st._getvalue())


        ret = context.make_tuple(builder,out_type,vals)
        return ret#impl_ret_borrowed(context, builder, sig.return_type, ret)
        # return 

    return sig,codegen


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
    print("TRY")
    if(not isinstance(attr_literal,types.Literal)):
        print("FAIL")
        return

    attr = attr_literal.literal_value

    print("ATTTR",attr)

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

