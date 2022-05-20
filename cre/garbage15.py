import numba
from llvmlite import ir
from llvmlite import binding as ll
from numba.core import types, cgutils, errors
from numba import njit, i8, generated_jit
from numba.core.unsafe.nrt import NRT_get_api
from numba.extending import intrinsic, overload_method, overload
from numba.core.imputils import impl_ret_borrowed
import cre_cfuncs
# from cre_cfuncs import meminfo_copy_unsafe

print(cre_cfuncs.c_helpers)
from cre.fact import define_fact
from cre.utils import _raw_ptr_from_struct, _struct_from_ptr, _store, _cast_structref, decode_idrec, encode_idrec, _incref_ptr, _load_ptr

numba_path = numba.extending.include_path()
print(numba_path)
    # cre_c_funcs = Extension(
    #     name='cre_cfuncs', 
    #     sources=['cre/cfuncs/test.c'],
    #     include_dirs=[numba_path]
    # )

for name, c_addr in cre_cfuncs.c_helpers.items():
    ll.add_symbol(name, c_addr)


# from cffi import FFI
# ffibuilder = FFI()
# ffibuilder.cdef('''
# long NRT_MemInfo_copy_unsafe(void* nrt, long mi);
# ''')
# ffibuilder.set_source("cre_extras",
#     '#include "cfuncs/extras.h"',
#     sources=["cfuncs/extras.c"],
#     include_dirs=[numba_path])
# ffibuilder.compile()

# import cre_extras
# numba.core.typing.cffi_utils.register_module(cre_extras)

# from cre_extras.lib import NRT_MemInfo_copy_unsafe


def _meminfo_copy_unsafe(builder, nrt, meminfo):
    mod = builder.module
    fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.voidptr_t, cgutils.voidptr_t])
    fn = cgutils.get_or_insert_function(mod, fnty, "meminfo_copy_unsafe")
    fn.return_value.add_attribute("noalias")
    return builder.call(fn, [builder.bitcast(nrt, cgutils.voidptr_t), builder.bitcast(meminfo, cgutils.voidptr_t)])


@intrinsic
def memcopy_fact(typingctx, inst_type):    
    def codegen(context, builder, signature, args):
        val = args[0]
        ctor = cgutils.create_struct_proxy(inst_type)
    
        dstruct = ctor(context, builder, value=val)
        meminfo = dstruct.meminfo
        nrt = context.nrt.get_nrt_api(builder)
        new_meminfo = _meminfo_copy_unsafe(builder, nrt, meminfo)

        inst_struct = context.make_helper(builder, inst_type)
        inst_struct.meminfo = new_meminfo

        return impl_ret_borrowed(
            context,
            builder,
            inst_type,
            inst_struct._getvalue()
        )

    sig = inst_type(inst_type)
    return sig, codegen

from cre.cre_object import cre_obj_iter_t_id_item_ptrs, CREObjType
@generated_jit(cache=True)
def copy_fact(fact):
    fact_type = fact
    def impl(fact):
        # nrt = NRT_get_api()
        new_fact = memcopy_fact(fact)
        a,b =   _cast_structref(CREObjType,fact), _cast_structref(CREObjType,new_fact)

        t_id, _, a_id = decode_idrec(a.idrec)
        b.idrec = encode_idrec(t_id,0,a_id)

        for info_a, info_b in zip(cre_obj_iter_t_id_item_ptrs(a),cre_obj_iter_t_id_item_ptrs(b)):
            t_id_a, m_id_a, data_ptr_a = info_a
            t_id_b, m_id_b, data_ptr_b = info_b

            if(m_id_b != 0):
                obj_ptr = _load_ptr(i8, data_ptr_a)
                _incref_ptr(obj_ptr)

        return new_fact
    print("END")
    return impl








BOOP = define_fact("BOOP", {"A":i8,"B":i8})
MOOP = define_fact("MOOP", {"boop1":"BOOP","boop2":"BOOP"})

b1 = BOOP(1,2)
b2 = BOOP(2,4)
b1_copy = copy_fact(b1)
m = MOOP(b1,b2)
m_copy = copy_fact(m)
print(m_copy.boop1)
print(m_copy.boop2)


@njit(cache=True)
def foo():
    b1 = BOOP(1,2)
    b2 = BOOP(2,4)
    b1_copy = copy_fact(b1)
    m = MOOP(b1,b2)
    m_copy = copy_fact(m)
    print(m_copy.boop1)
    print(m_copy.boop2)
    

foo()
