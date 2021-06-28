import numba
from numba import njit, i8, i4, u4, types
from numba.types import unicode_type, Tuple
from numba.cpython.unicode import UnicodeModel, _kind_to_byte_width, _empty_string, _strncpy
from numba.core.datamodel.registry import default_manager
from numba.core.imputils import impl_ret_borrowed
from numba.extending import intrinsic, register_jitable
from numba.core import cgutils
from llvmlite.ir import types as ll_types
import numpy as np
# print(UnicodeModel.__dict__)
# print(default_manager[unicode_type]._members)


# @njit(cache=True)
# def foo():
#     s = "HI"
#     return s
#     # print(i8(s._length+0))

# s = foo()
# print(s.__slots__)


uni_str_fields = [
    ('data', np.int64),
    ('length', np.int64),
    ('kind', np.int32),
    ('is_ascii', np.uint32),
    ('hash', np.int64),
    ('meminfo', np.int64),
    ('parent', np.int64),
     ]
uni_str_dtype = np.dtype(uni_str_fields)
uni_str_type = numba.from_dtype(uni_str_dtype)




@intrinsic
def _str_from_members(typingctx, data, length, kind, is_ascii, _hash, meminfo, parent):
    '''Recovers a function from it's signature and address '''
    rec_types = (i8,i8,i4,u4,i8,i8,i8)
    sig = unicode_type(*rec_types)
    def codegen(context, builder, sig, args):
        [data, length, kind, is_ascii, _hash, meminfo, parent] = args
        uni_str_ctor = cgutils.create_struct_proxy(types.unicode_type)
        uni_str = uni_str_ctor(context, builder)
        # char_width = _kind_to_byte_width(kind)

        model_typs = default_manager[unicode_type]._members

        uni_str.data = builder.inttoptr(data, cgutils.voidptr_t)
        uni_str.length = context.cast(builder, length, rec_types[1], model_typs[1])
        uni_str.kind = context.cast(builder, kind, rec_types[2], model_typs[2])
        uni_str.is_ascii = context.cast(builder, is_ascii, rec_types[3], model_typs[3])
        uni_str.hash = context.cast(builder, _hash, rec_types[4], model_typs[4])
        uni_str.meminfo = builder.inttoptr(meminfo, cgutils.voidptr_t)
        uni_str.parent = builder.inttoptr(parent, cgutils.voidptr_t)

        # context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        return impl_ret_borrowed(context, builder, unicode_type,uni_str._getvalue())
        # return uni_str._getvalue()
    return sig, codegen

@intrinsic
def _str_rip_members(typingctx, str_type):
    '''Recovers a function from it's signature and address '''
    if(str_type != unicode_type): return
    rec_types = (i8,i8,i4,u4,i8,i8,i8)
    sig =  Tuple(tuple(rec_types))(unicode_type,)
    def codegen(context, builder, sig, args):
        src, = args
        uni_str = cgutils.create_struct_proxy(unicode_type)(context, builder, value=src)

        model_typs = default_manager[unicode_type]._members
        data = builder.ptrtoint(uni_str.data, cgutils.intp_t)
        length = context.cast(builder, uni_str.length, model_typs[1], i8)
        kind = context.cast(builder, uni_str.kind, model_typs[2], i4)
        is_ascii = context.cast(builder, uni_str.is_ascii, model_typs[3], u4)
        _hash = context.cast(builder, uni_str.hash, model_typs[4], i8)
        meminfo = builder.ptrtoint(uni_str.meminfo, cgutils.intp_t)
        parent = builder.ptrtoint(uni_str.parent, cgutils.intp_t)
        tup = (data, length, kind, is_ascii, _hash,meminfo, parent)
        res = context.make_tuple(builder, sig.return_type, tup)

        # Increfing makes sure that the string ends up on the heap
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), uni_str.meminfo)

        return impl_ret_borrowed(context, builder, sig.return_type, res)
    return sig, codegen

@register_jitable
def insert_str_record(arr,index,val):
    data, length, kind, is_ascii, _hash, meminfo, parent = \
        _str_rip_members(val)
    print(data, length, kind, is_ascii, _hash, meminfo, parent)
    arr_i = arr[index]
    arr_i.data = data
    arr_i.length = length
    arr_i.kind = kind
    arr_i.is_ascii = is_ascii
    arr_i.hash = _hash
    arr_i.meminfo = meminfo
    arr_i.parent = parent


@register_jitable
def extract_str_from_record(rec):
    return _str_from_members(rec.data, rec.length,
     rec.kind, rec.is_ascii, rec.hash, rec.meminfo, rec.parent)


#     char_width = _kind_to_byte_width(kind)
#     s = _malloc_string(kind, char_width, length, is_ascii)
#     _set_code_point(s, length, np.uint32(0))    # Write NULL character
#     return s


@njit(cache=False)
def foo():
    arr = np.empty(16,dtype=uni_str_type)
    for i, a in enumerate(["A","B","C","D"]):
        for j, b in enumerate(["A","B","C","D"]):
            v = a+b
            print(hash(v))
            print(i,j,v)
            insert_str_record(arr,i*4 + j, v)
    return arr

@njit(cache=False)
def bar(arr):
    for i in range(4):
        for j in range(4):
            print(extract_str_from_record(arr[i*4+j]))
            # return 


    # data, length, kind, is_ascii, _hash, meminfo, parent = \
    #     _str_rip_members("BOOOP")
    # s = _str_from_members(data, length, kind,
    #          is_ascii, _hash, meminfo, parent)
    # print(s)

arr = foo()
print(arr)
print(bar(arr))
