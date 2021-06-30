import numba
from numba import njit, i8, i4, u4, types
from numba.types import unicode_type, Tuple
from numba.typed import Dict
from numba.cpython.unicode import UnicodeModel, _kind_to_byte_width, _empty_string, _strncpy
from numba.core.datamodel.registry import default_manager
from numba.core.imputils import impl_ret_borrowed
from numba.extending import intrinsic, register_jitable
from numba.core import cgutils
from numba import prange
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

import time
class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.2f} ms')



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
def insert_str_record(rec,val):
    data, length, kind, is_ascii, _hash, meminfo, parent = \
        _str_rip_members(val)
    # print(data, length, kind, is_ascii, _hash, meminfo, parent)
    # arr_i = arr[index]
    rec.data = data
    rec.length = length
    rec.kind = kind
    rec.is_ascii = is_ascii
    rec.hash = _hash
    rec.meminfo = meminfo
    rec.parent = parent


@register_jitable
def extract_str_from_record(rec):
    return _str_from_members(rec.data, rec.length,
     rec.kind, rec.is_ascii, rec.hash, rec.meminfo, rec.parent)


#     char_width = _kind_to_byte_width(kind)
#     s = _malloc_string(kind, char_width, length, is_ascii)
#     _set_code_point(s, length, np.uint32(0))    # Write NULL character
#     return s


@njit(cache=True,parallel=True)
def parallel_insert():
    arr = np.empty((10,95,95),dtype=uni_str_type)
    for p in prange(0,10):
        for i in range(0,95):
            for j in range(0,95):
                v = chr(10+p)+chr(32+i)+chr(32+j)
                rec = arr[p,i,j]
                insert_str_record(arr[p,i,j], v)
                # rec.hash = hash(v)
    d = Dict.empty(unicode_type,i8)
    k = 0
    for i, rec in enumerate(arr.flatten()):
        v = extract_str_from_record(rec)
        if(v not in d):
            d[v] = k; k += 1
    return d

@njit(cache=True)
def seq_insert():
    # arr = np.empty((95,95),dtype=uni_str_type)
    d = Dict.empty(unicode_type,i8)
    k = 0
    for p in range(0,10):
        for i in range(0,95):
            for j in range(0,95):
                v = chr(10+p)+chr(32+i)+chr(32+j)
                if(v not in d):
                    d[v] = k; k += 1
            # d[v] = i*95+j
            # rec = arr[i,j]
            # insert_str_record(arr[i,j], v)
            # rec.hash = hash(v)
    
    # for i, rec in enumerate(arr.flatten()):
    #     d[extract_str_from_record(rec)] = i
    return d

# @njit(cache=True)
# def bar(arr):
#     v = ""
#     for i in range(0,95):
#         for j in range(0,95):
#             v = extract_str_from_record(arr[i,j])
#     print(v)
            # return 


    # data, length, kind, is_ascii, _hash, meminfo, parent = \
    #     _str_rip_members("BOOOP")
    # s = _str_from_members(data, length, kind,
    #          is_ascii, _hash, meminfo, parent)
    # print(s)

parallel_insert()
seq_insert()
with PrintElapse("parallel_insert"):
    arr = parallel_insert()

with PrintElapse("seq_insert"):
    arr = seq_insert()


# print(arr)
# print(bar(arr))
