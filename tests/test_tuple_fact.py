# from cre.fact import _fact_from_spec, _standardize_spec, _merge_spec_inheritance, \
#      define_fact, cast_fact, _cast_structref, BaseFact, BaseFactType, DeferredFactRefType
from cre.utils import _cast_structref
from cre.context import cre_context
from cre.memory import Memory
from cre.tuple_fact import TupleFact, TF#TupleFact, assert_cre_obj, TupleFact
from cre.fact import define_fact#TupleFact, assert_cre_obj, TupleFact
# from cre.primitive import Primitive, StringPrimitiveType
from cre.cre_object import CREObjType
from cre.utils import PrintElapse, decode_idrec

from numba import njit, f8, u8, i8, bool_
from numba.types import unicode_type
from numba.typed import List, Dict
import pytest
import numpy as np

import cre.dynamic_exec



@njit(cache=True)
def as_cre_obj(x):
    return _cast_structref(CREObjType,x)

# @njit(cache=True)
# def init_prim():
#     return Primitive(2)

@njit(bool_(CREObjType, CREObjType),cache=True)
def eq(a,b):
    return a == b


def test_init():
    @njit(cache=True)
    def make_it():
        return TF("HI",TF("HI",2))

    tf = make_it.py_func()
    tf = make_it()








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

    a1 = TF("HI",TF("HI",2))
    a2 = TF("HI",TF("HI",2))
    b1 = TF("HI",TF("HI",3))
    b2 = TF("HI",TF("HO",2))
    b3 = TF("HO",TF("HI",2))
    b4 = TF("HO",TF("HI",2, 0))

    print(decode_idrec(a1.idrec))
    print(decode_idrec(a2.idrec))
    print(decode_idrec(b1.idrec))
    print(decode_idrec(b2.idrec))
    print(decode_idrec(b3.idrec))
    print(decode_idrec(b4.idrec))

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

    a1 = TF("HI",TF("HI",2))
    a2 = TF("HI",TF("HI",2))
    b1 = TF("HI",TF("HI",3))
    b2 = TF("HI",TF("HO",2))
    b3 = TF("HO",TF("HI",2))
    b4 = TF("HO",TF("HI",2, 0))

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

def test_hash_obj_builtin_members():
    from cre.default_ops import Equals
    from cre.var import Var
    eq_f8 = Equals(f8,f8)
    eq_str = Equals(unicode_type,unicode_type)
    a_f8 = Var(f8,'a')
    b_f8 = Var(f8,'b')
    a_str = Var(unicode_type,'a')
    b_str = Var(unicode_type,'b')
    
    a1 = TF(eq_f8, a_f8, b_f8)
    a2 = TF(eq_f8, a_f8, b_f8)
    b1 = TF(eq_str, a_str, b_str)

    a1_hash = hsh(a1)
    a2_hash = hsh(a1)

    print(hsh(a1), hsh(a2))
    print(hsh(a1), hsh(b1))

    assert hsh(a1) == hsh(a2)
    assert hsh(a1) != hsh(b1)

def test_eq_obj_builtin_members():
    from cre.default_ops import Equals
    from cre.var import Var
    eq_f8 = Equals(f8,f8)
    eq_str = Equals(unicode_type,unicode_type)
    a_f8 = Var(f8,'a')
    b_f8 = Var(f8,'b')
    a_str = Var(unicode_type,'a')
    b_str = Var(unicode_type,'b')
    
    a1 = TF(eq_f8, a_f8, b_f8)
    a2 = TF(eq_f8, a_f8, b_f8)
    b1 = TF(eq_str, a_str, b_str)

    a1_hash = hsh(a1)
    a2_hash = hsh(a1)

    assert eq(a1, a2)
    assert not eq(a1, b1)
    

    # print(a1, a2)


N = 10000


def test_str():
    with cre_context("test_str"):
        a1 = TF("HI", TF("HI", 2))
        # assert str(at) == 'TF("HI", TF("HI", 2))'
        print(a1)

        a1_typ = TF(unicode_type, TF(unicode_type, i8))
        print(a1_typ)

        # assert str(a1_typ) == 'TF(unicode_type, TF(unicode_type, i8))'

        BOOP = define_fact("BOOP", {"A" : a1_typ, "B" : "number"})

        b = BOOP(A=a1, B=1)

        # assert str(b) == "BOOP(A=TupleFact('HI', TupleFact('HI', 2)), B=1.0)"
        print(b)

        print(str(TF(b, 1)))


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
    test_init()
    test_eq()
    test_hash()
    test_hash_obj_builtin_members()
    test_eq_obj_builtin_members()
    test_str()
    # p = init_prim()
    # print(type(p), p)
    # print(assert_cre_obj(2))
    # _b_make_tup_fact()
    # _b_dict_insert_tup_fact()

    
    # _b_make_py_tup()

