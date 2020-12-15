from numbert.experimental.structref import gen_structref_code, define_structref
from numba import optional, u8, njit

base_fact_fields = [
    ("idrec", u8),
    ("kb", optional(u8))
]
BaseFact, BaseFactType = define_structref("TestFact", base_fact_fields)

b = BaseFact(1,None)
@njit
def foo(b):
    print(b.kb)

foo(b)

from numbert.experimental.fact import _gen_fact_ctor


fields = [
    ("X", u8),
    ("Y", u8),
]
print(_gen_fact_ctor(base_fact_fields, fields))
