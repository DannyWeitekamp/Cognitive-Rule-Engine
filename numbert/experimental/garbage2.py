from numba import types, njit, u1,u2,u4,u8, i8, carray
from numba.types import Tuple
from numba.typed import List
from numba.experimental.structref import _Utils, imputils
from numba.extending import intrinsic
from numba.core import cgutils
from numbert.experimental.fact import define_fact
from llvmlite import ir
import numpy as np

spec1 = {"A" : "string", "B" : "number"}
BOOP, BOOPType = define_fact("BOOP", spec1, context="test__merge_spec_inheritance")

import timeit
N=100
def time_ms(f):
    f() #warm start
    return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))


@intrinsic
def _struct_from_pointer(typingctx, struct_type, raw_ptr):
    inst_type = struct_type.instance_type

    def codegen(context, builder, sig, args):
        _, raw_ptr = args
        _, raw_ptr_ty = sig.args

        meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo
        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        

        return st._getvalue()

    sig = inst_type(struct_type, raw_ptr)
    return sig, codegen

@intrinsic
def _pointer_from_struct(typingctx, val):
    def codegen(context, builder, sig, args):
        [td] = sig.args
        [d] = args



        model = context.data_model_manager[td.get_data_type()]
        alloc_type = model.get_value_type()

        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder, value=d)
        meminfo = dstruct.meminfo

        # context.nrt.incref(builder, alloc_type, dstruct)

        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        res = builder.ptrtoint(dstruct.meminfo, cgutils.intp_t)

        return res
        
    sig = i8(val,)
    return sig, codegen





@njit()
def foo(st):
    ptr = _pointer_from_struct(st)
    print(ptr)
    b = _struct_from_pointer(BOOPType, ptr)
    print(b.A,b.B)
    ptr = _pointer_from_struct(b)
    print(ptr)
    return b

@njit()
def bar():
    st = BOOP("A",1)
    ptr = _pointer_from_struct(st)
    print(ptr)
    b = _struct_from_pointer(BOOPType, ptr)
    print(b.A,b.B)
    ptr = _pointer_from_struct(b)
    print(ptr)
    return b


b = BOOP("A",1)
# print(b._meminfo.data)

b = foo(b)
print(b._meminfo.data)

bar()

# print(foo())
# print(b._meminfo.data)

@intrinsic
def _pointer_from_arr(typingctx, val):
    def codegen(context, builder, sig, args):
        [val] = args
        [typ] = sig.args

        ctinfo = context.make_helper(builder, typ, value=val)
    # res = array.shape
        ptr = ctinfo.data

        #Incref the array so that it isn't garbage collected
        context.nrt.incref(builder, typ, val)

        # Convert it to an integer
        ptr = builder.ptrtoint(ptr, context.get_value_type(types.intp))

        return ptr

    sig = i8(val,)
    return sig, codegen




u8_ptr = types.CPointer(types.int64)

@intrinsic
def _pointer_from_uint(typingctx, val):
    def codegen(context, builder, sig, args):
        # context.get_value_type(types.intp)
        res = builder.inttoptr(args[0], cgutils.intp_t.as_pointer())
        return res
    sig = u8_ptr(u8,)
    return sig, codegen


# @generated_jit

@njit(cache=True)
def _arr_to_data(arr):
    ptr = _pointer_from_arr(arr)
    size = arr.size
    return ptr, size

@njit(cache=True)
def _arr_from_data(data):
    ptr,size = data[0],data[1]
    return carray(_pointer_from_uint(ptr), size)




@njit(cache=True)
def get_fact(facts,t_id,f_id):
    print(_arr_from_data(facts[t_id]))
    ptr = _arr_from_data(facts[t_id])[f_id]
    print("ptr",ptr)
    return _struct_from_pointer(BOOPType,ptr)



@njit(cache=True)
def gen_facts():
    # keep_around_list = List()
    keep_around_list2 = List()
    facts = np.empty((8,2),dtype=np.int64)
    for i in range(8):
        fact_ptr_arr = np.empty(10,dtype=np.int64)
        
        
        if(i == 1):
            for j in range(5):
                b = BOOP("A",j)
                fact_ptr_arr[j] = _pointer_from_struct(b)
                # c = _struct_from_pointer(BOOPType,fact_ptr_arr[j])
                # print(fact_ptr_arr[j], c.B)
                # keep_around_list.append(c)


        facts[i] = _arr_to_data(fact_ptr_arr)
        keep_around_list2.append(fact_ptr_arr)
        print(facts[i])

    return facts, keep_around_list2

@njit
def print_ptrs(l):
    for st in l:
        ptr = _pointer_from_struct(st)
        b = _struct_from_pointer(BOOPType,ptr)
        print(ptr, b.B)


print("BE")
# gen_facts()
facts, keep_around_list2 = gen_facts()
print("AF")
print("-----")
# print_ptrs(keep_around_list)


@njit
def internal_get_facts():
    b = get_fact(facts,1,0)
    print("AQUI",b.A, b.B)

print("HERE",internal_get_facts())






@njit()
def doo():
    a = np.ones(5,dtype=np.int64) * 7
    print(a)
    ptr = _pointer_from_arr(a)#a.ctypes.data

    print(a.ctypes.data)
    print(a.shape)
    print(a.dtype)

    b = carray(_pointer_from_uint(ptr), (5,)) #np.ctypeslib.as_array(ptr,shape=(5,))
    print(b.ctypes.data)
    print(b.shape)
    print(b.dtype)

    print(b)
    # print(a)
    # _pointer_from_arr(a)

print("----------")


doo()


boop_list = types.ListType(BOOPType)
@njit
def setup_list():
    facts = List.empty_list(boop_list)
    for i in range(8):
        facts.append(List.empty_list(BOOPType))
    print(len(facts))

    return facts

list_facts = setup_list()

@njit
def declare_list(facts):
    t_f = facts[1]
    fact = BOOP("A",7)
    for i in range(10000):
        t_f.append(fact)

def d_list():
    declare_list(list_facts)


@njit
def setup_arr():
    l = List()
    facts = np.empty((8,2),dtype=np.int64)
    for i in range(8):
        fact_ptrs = np.empty(10000,dtype=np.int64)
        facts[i] = _arr_to_data(fact_ptrs)
        l.append(fact_ptrs)
        # facts.append(List.empty_list(BOOPType))
    # print(len(facts))

    return facts

@njit
def dec_ij(facts,t_id,f_id, fact):
    # print(_arr_from_data(facts[t_id]))
     _arr_from_data(facts[t_id])[f_id] = _pointer_from_struct(fact)
    # print("ptr",ptr)
    # return _struct_from_pointer(BOOPType,ptr)


@njit
def declare_arr(facts):
    fact = BOOP("A",7)
    for i in range(10000):
        # _arr_from_data(facts[1])[i] = _pointer_from_struct(fact)
        dec_ij(facts,1,i,fact)


arr_facts = setup_arr()

def d_arr():
    declare_arr(arr_facts)





print("declare_list: ", time_ms(d_list)) 
print("declare_arr: ", time_ms(d_arr)) 

