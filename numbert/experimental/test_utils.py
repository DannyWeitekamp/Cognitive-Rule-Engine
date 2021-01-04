from numbert.experimental.fact import define_fact
from numbert.experimental.structref import define_structref
from numba import njit
from numba.types import unicode_type, f8, i8
from numba.experimental.structref import new
import logging
import numpy as np
import pytest
from numbert.experimental.utils import _incref_structref, _decref_structref, \
         _meminfo_from_struct, _struct_from_meminfo, _pointer_from_struct, \
         _struct_from_pointer, encode_idrec, decode_idrec
from numba.core.runtime.nrt import rtsys

BOOP, BOOPType = define_structref("BOOP", [("A", unicode_type), ("B", i8)])


##### test_encode_decode #####

def test_encode_decode():
    id_rec = encode_idrec(7,8,9)
    assert isinstance(id_rec,int)
    assert decode_idrec(id_rec) == (7,8,9)


#### test_structref_to_meminfo ####

@njit(cache=True)
def new_boop():
    return BOOP("?", 1)

@njit(cache=True)
def rip_meminfo_inc(st):
    # _incref_structref(st)
    return _meminfo_from_struct(st)

@njit(cache=True)
def restore_struct_from_meminfo(m):
    return _struct_from_meminfo(BOOPType,m)

def test_structref_to_meminfo():
    b1 = new_boop()
    b2 = new_boop.py_func()

    assert b1._meminfo.refcount == 1
    assert b2._meminfo.refcount == 1

    meminfo1 = rip_meminfo_inc(b1)
    meminfo2 = rip_meminfo_inc(b2)

    assert b1._meminfo.refcount == 2
    assert b2._meminfo.refcount == 2

    b1 = restore_struct_from_meminfo(meminfo1)
    b2 = restore_struct_from_meminfo(meminfo2)

    meminfo1, meminfo2 = None, None

    assert b1.A == "?" and b1.B == 1
    assert b2.A == "?" and b2.B == 1


#### test_structref_to_pointer ####

@njit(cache=True)
def clear_incref(st):
    _decref_structref(st)

@njit(cache=True)
def rip_pointer_inc(st):
    _incref_structref(st)
    return _pointer_from_struct(st)

@njit(cache=True)
def restore_struct_from_pointer(p):
    return _struct_from_pointer(BOOPType,p)


def test_structref_to_pointer():
    b1 = new_boop()
    b2 = new_boop.py_func()

    assert b1._meminfo.refcount == 1
    assert b2._meminfo.refcount == 1

    ptr1 = rip_pointer_inc(b1)
    ptr2 = rip_pointer_inc(b2)

    assert b1._meminfo.refcount == 2
    assert b2._meminfo.refcount == 2

    b1 = restore_struct_from_pointer(ptr1)
    b2 = restore_struct_from_pointer(ptr2)

    assert b1.A == "?" and b1.B == 1
    assert b2.A == "?" and b2.B == 1

    clear_incref(b1)
    clear_incref(b2)

    assert b1._meminfo.refcount == 1
    assert b2._meminfo.refcount == 1


if __name__ == "__main__":
    test_structref_to_meminfo()
    test_structref_to_pointer()









    
