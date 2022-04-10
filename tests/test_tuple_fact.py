# from cre.fact import _fact_from_spec, _standardize_spec, _merge_spec_inheritance, \
#      define_fact, cast_fact, _cast_structref, BaseFact, BaseFactType, DeferredFactRefType
from cre.utils import _cast_structref
from cre.context import cre_context
from cre.memory import Memory
from cre.tuple_fact import TupleFact, TF#TupleFact, assert_cre_obj, TupleFact
from cre.primitive import Primitive, StringPrimitiveType
from cre.cre_object import CREObjType
from cre.utils import PrintElapse

from numba import njit, u8, i8
from numba.typed import List, Dict
import pytest
import numpy as np

import cre.dynamic_exec



@njit(cache=True)
def as_cre_obj(x):
    return _cast_structref(CREObjType,x)

@njit(cache=True)
def init_prim():
    return Primitive(2)

@njit(cache=True)
def eq(a,b):
    return a == b

def test_eq():
    # hi = Primitive("HI")
    # a1 = as_cre_obj(Primitive(True))
    # a2 = as_cre_obj(Primitive(True))
    # b = as_cre_obj(Primitive(False))
    
    # assert eq(a1, a2) == True
    # assert eq(a1, b) == False

    # a1 = as_cre_obj(Primitive(2))
    # a2 = as_cre_obj(Primitive(2))
    # b = as_cre_obj(Primitive(3))

    # assert eq(a1, a2) == True
    # assert eq(a1, b) == False

    # a1 = as_cre_obj(Primitive(1.0))
    # a2 = as_cre_obj(Primitive(1.0))
    # b = as_cre_obj(Primitive(2.0))
    
    # assert eq(a1, a2) == True
    # assert eq(a1, b) == False

    # a1 = as_cre_obj(Primitive("A"))
    # a2 = as_cre_obj(Primitive("A"))
    # b = as_cre_obj(Primitive("B"))
    
    # assert eq(a1, a2) == True
    # assert eq(a1, b) == False

    a1 = as_cre_obj(TF("HI",TF("HI",2)))
    a2 = as_cre_obj(TF("HI",TF("HI",2)))
    b1 = as_cre_obj(TF("HI",TF("HI",3)))
    b2 = as_cre_obj(TF("HI",TF("HO",2)))
    b3 = as_cre_obj(TF("HO",TF("HI",2)))
    b4 = as_cre_obj(TF("HO",TF("HI",2, 0)))

    # print(eq(a1, a2))
    assert eq(a1, a2)
    assert not eq(a1, b1)
    assert not eq(a1, b2)
    assert not eq(a1, b3)
    assert not eq(a1, b4)

@njit(cache=True)
def hsh(x):
    return hash(x)


def test_hash():
    # a1 = as_cre_obj(Primitive(True))
    # print(hsh(a1))    

    # a1 = as_cre_obj(Primitive(2))
    # print(hsh(a1))

    # a1 = as_cre_obj(Primitive(1.0))
    # print(hsh(a1))
    
    # a1 = as_cre_obj(Primitive("A"))
    # print(hsh(a1))

    # a1 = as_cre_obj(TupleFact("HI",2))
    # print(hsh(a1))

    a1 = as_cre_obj(TF("HI",TF("HI",2)))
    a2 = as_cre_obj(TF("HI",TF("HI",2)))
    b1 = as_cre_obj(TF("HI",TF("HI",3)))
    b2 = as_cre_obj(TF("HI",TF("HO",2)))
    b3 = as_cre_obj(TF("HO",TF("HI",2)))
    b4 = as_cre_obj(TF("HO",TF("HI",2, 0)))

    print(hsh(a1),hsh(a2))
    print(hsh(a1),hsh(b1))
    print(hsh(a1),hsh(b2))
    print(hsh(a1),hsh(b3))
    print(hsh(a1),hsh(b4))


    assert hsh(a1) == hsh(a2)
    assert hsh(a1) != hsh(b1)
    assert hsh(a1) != hsh(b2)
    assert hsh(a1) != hsh(b3)
    assert hsh(a1) != hsh(b4)

N = 10000



@njit(cache=True)
def _b_dict_insert_tup_fact():
    d = Dict.empty(TupleFact, i8)
    for i in range(N):
        hash(TF(str(i),i))
        # d[TupleFact(str(i),i)] = i
    return d

def test_b_dict_insert_tup_fact(benchmark):
    _b_dict_insert_tup_fact()
    benchmark.pedantic(_b_dict_insert_tup_fact,warmup_rounds=1,rounds=20)

def _b_dict_insert_tup():
    d = {}
    for i in range(N):
        hash((str(i),i))
        # d[(str(i),i)] = i
    return d

def test_b_dict_insert_py_tup(benchmark):
    _b_dict_insert_tup()
    benchmark.pedantic(_b_dict_insert_tup,warmup_rounds=1,rounds=20)
    

@njit(cache=True)
def _b_make_tup_fact():
    d = Dict.empty(TupleFact, i8)
    for i in range(N):
        TF(str(i),i)
        # d[TupleFact(str(i),i)] = i
    return d

def test_b_make_TupleFact(benchmark):
    _b_make_tup_fact()
    benchmark.pedantic(_b_make_tup_fact,warmup_rounds=1,rounds=20)    


@njit(cache=True)
def _b_make_py_tup():
    d = Dict.empty(TupleFact, i8)
    for i in range(N):
        (str(i),i)
        # d[TupleFact(str(i),i)] = i
    return d

def test_b_make_py_tup(benchmark):
    _b_make_py_tup()
    benchmark.pedantic(_b_make_py_tup,warmup_rounds=1,rounds=20)    





if __name__ == "__main__":
    # test_eq()
    test_hash()
    # p = init_prim()
    # print(type(p), p)
    # print(assert_cre_obj(2))
    # _b_make_tup_fact()
    _b_dict_insert_tup_fact()
    # _b_make_py_tup()

