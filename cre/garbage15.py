import numba
from llvmlite import ir
from llvmlite import binding as ll
from numba.core import types, cgutils, errors
from numba import njit, i8, generated_jit
from numba.extending import intrinsic
from numba.core.imputils import impl_ret_borrowed
import cre_cfuncs
# Make methods in cre_cfuncs.c_helpers available to LLVM
for name, c_addr in cre_cfuncs.c_helpers.items():
    ll.add_symbol(name, c_addr)





from cre.fact import define_fact
# Just a shortcut for defining structrefs
BOOP = define_fact("BOOP", {"A":i8,"B":i8})

@njit(cache=True)
def foo():
    a = BOOP(1,2)
    b = _memcpy_structref(a)
    a.A,a.B = 7,7
    print(a, b)

foo()




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
