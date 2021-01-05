from numbert.experimental.fact import define_fact
from numbert.experimental.structref import define_structref
from numba import njit,literally
from numba.types import unicode_type, f8, i8
from numba.experimental.structref import new
import logging
import numpy as np
import pytest

from numbert.experimental.utils import _incref_structref, _decref_structref, \
         _meminfo_from_struct, _struct_from_meminfo, _pointer_from_struct, \
         _struct_from_pointer, encode_idrec, decode_idrec, _struct_get_attr_offset, \
         _struct_get_data_pointer, _load_pointer
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

@njit(cache=True)
def get_offset_inst(b,attr):
    return _struct_get_attr_offset(b,literally(attr))

@njit(cache=True)
def load_offset(typ, b ,offset):
    return _load_pointer(typ,_struct_get_data_pointer(b)+offset)


BEEP1, BEEP1Type = define_structref("BEEP1", [("A", i8), ("B", i8),("C", i8)])
BEEP2, BEEP2Type = define_structref("BEEP2", [("A", unicode_type), ("B", i8),("C", i8)])

def test_direct_member_access():
    b1 = BEEP1(1,2,1)
    b2 = BEEP1(1,2,3)

    assert get_offset_inst(b1,"A") == 0
    assert get_offset_inst(b1,"B") == 8
    assert get_offset_inst(b1,"C") == 16

    offset = get_offset_inst(b1,"C")
    cls_offset = get_offset_inst(BEEP1Type,"C")

    assert offset == cls_offset

    assert load_offset(i8,b2,offset) == 3

    b1 = BEEP2("b1",2,1)
    b2 = BEEP2("b2",2,3)

    offset = get_offset_inst(b1,"C")
    cls_offset = get_offset_inst(BEEP2Type,"C")
    assert offset == cls_offset

    offset = get_offset_inst(b1,"A")
    cls_offset = get_offset_inst(BEEP2Type,"A")
    assert offset == cls_offset

    load_offset(unicode_type,b2,offset) == "b2"





    # _load_pointer(i8,_struct_get_data_pointer(b2)+offset) == 3


    




if __name__ == "__main__":
    # test_structref_to_meminfo()
    # test_structref_to_pointer()
    test_direct_member_access()









    
