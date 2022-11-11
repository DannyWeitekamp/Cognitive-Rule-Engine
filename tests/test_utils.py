from cre.fact import define_fact
from cre.structref import define_structref
from numba import njit,literally
from numba.types import unicode_type, f8, i8, ListType
from numba.typed import List
from numba.experimental.structref import new
import logging
import numpy as np
import pytest

from cre.utils import (_incref_structref, _decref_structref, 
         _meminfo_from_struct, _struct_from_meminfo,  _raw_ptr_from_struct,
         _struct_from_ptr, encode_idrec, decode_idrec, _struct_get_attr_offset, 
         _struct_get_data_ptr, _load_ptr, struct_get_attr_offset, 
         _listtype_sizeof_item, _ptr_from_struct_incref, _list_base_from_ptr)
from numba.core.runtime.nrt import rtsys

BOOP, BOOPType = define_structref("BOOP", [("A", unicode_type), ("B", i8)])


##### test_encode_decode #####

def test_encode_decode():
    idrec = encode_idrec(7,8,9)
    assert isinstance(idrec,int)
    assert decode_idrec(idrec) == (7,8,9)

    idrec = encode_idrec(0x0000,1099511627775,255)
    assert isinstance(idrec,int)
    assert decode_idrec(idrec) == (0x0000,1099511627775,255)

    idrec = encode_idrec(0xFFFF,1099511627775,255)
    assert isinstance(idrec,int)
    assert decode_idrec(idrec) == (0xFFFF,1099511627775,255)


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
    return _raw_ptr_from_struct(st)

@njit(cache=True)
def restore_struct_from_ptr(p):
    return _struct_from_ptr(BOOPType,p)


def test_structref_to_pointer():
    b1 = new_boop()
    b2 = new_boop.py_func()

    assert b1._meminfo.refcount == 1
    assert b2._meminfo.refcount == 1

    ptr1 = rip_pointer_inc(b1)
    ptr2 = rip_pointer_inc(b2)

    assert b1._meminfo.refcount == 2
    assert b2._meminfo.refcount == 2

    b1 = restore_struct_from_ptr(ptr1)
    b2 = restore_struct_from_ptr(ptr2)

    assert b1.A == "?" and b1.B == 1
    assert b2.A == "?" and b2.B == 1

    clear_incref(b1)
    clear_incref(b2)

    assert b1._meminfo.refcount == 1
    assert b2._meminfo.refcount == 1

# @njit(cache=True)
# def struct_get_attr_offset(b,attr):
#     return _struct_get_attr_offset(b,literally(attr))

@njit(cache=True)
def load_offset(typ, b ,offset):
    return _load_ptr(typ,_struct_get_data_ptr(b)+offset)



BEEP1, BEEP1Type = define_structref("BEEP1", [("A", i8), ("B", i8),("C", i8)])
BEEP2, BEEP2Type = define_structref("BEEP2", [("A", unicode_type), ("B", i8),("C", i8)])

def test_direct_member_access():
    b1 = BEEP1(1,2,1)
    b2 = BEEP1(1,2,3)

    assert struct_get_attr_offset(b1,"A") == 0
    assert struct_get_attr_offset(b1,"B") == 8
    assert struct_get_attr_offset(b1,"C") == 16

    offset = struct_get_attr_offset(b1,"C")
    cls_offset = struct_get_attr_offset(BEEP1Type,"C")

    assert offset == cls_offset

    assert load_offset(i8,b2,offset) == 3

    b1 = BEEP2("b1",2,1)
    b2 = BEEP2("b2",2,3)

    offset = struct_get_attr_offset(b1,"C")
    cls_offset = struct_get_attr_offset(BEEP2Type,"C")
    assert offset == cls_offset

    offset = struct_get_attr_offset(b1,"A")
    cls_offset = struct_get_attr_offset(BEEP2Type,"A")
    assert offset == cls_offset

    load_offset(unicode_type,b2,offset) == "b2"


@njit
def listtype_sizeof(lst_typ):
    return _listtype_sizeof_item(lst_typ)

@njit
def get_3(ptr,typ,lst_typ):
    item_size = _listtype_sizeof_item(lst_typ)
    base_ptr = _list_base_from_ptr(ptr)
    a = _load_ptr(typ,base_ptr)
    b = _load_ptr(typ,base_ptr+item_size*1)
    c = _load_ptr(typ,base_ptr+item_size*2)
    return a,b,c

unc_lst = ListType(unicode_type)
@njit
def _test_list_getitem_intrinsic_unicode_type():
    l = List.empty_list(unicode_type)
    l.append("A")
    l.append("B")
    l.append("C")
    ptr = _ptr_from_struct_incref(l)
    a,b,c = get_3(ptr,unicode_type,unc_lst)
    return a, b, c

f8_lst = ListType(f8)
@njit
def _test_list_getitem_intrinsic_f8():
    l = List.empty_list(f8)
    l.append(1)
    l.append(2)
    l.append(3)
    ptr = _ptr_from_struct_incref(l)
    a,b,c = get_3(ptr,f8,f8_lst)
    return a, b, c

BOOP_lst = ListType(BOOPType)
@njit
def _test_list_getitem_intrinsic_BOOP():
    l = List.empty_list(BOOPType)
    l.append(BOOP("A",1))
    l.append(BOOP("B",2))
    l.append(BOOP("C",3))
    ptr = _ptr_from_struct_incref(l)
    a,b,c = get_3(ptr,BOOPType,BOOP_lst)
    return a, b, c
    
def test_list_intrinsics():
    assert listtype_sizeof(unc_lst) == 48 
    assert listtype_sizeof(f8_lst) == 8 
    assert listtype_sizeof(BOOP_lst) == 8 #should be ptr width

    assert _test_list_getitem_intrinsic_unicode_type() == ("A","B","C")
    assert _test_list_getitem_intrinsic_f8() == (1,2,3)
    assert [x.A for x in _test_list_getitem_intrinsic_BOOP()] == ["A","B","C"]


###################### BENCHMARKS ########################

#### b_encode_idrec ####

def gen_rand_nums():
    return (np.random.randint(1000,size=(10000,3)),), {}

@njit(cache=True)
def _b_encode_idrec(rand_nums):
    for x in rand_nums:
        encode_idrec(x[0],x[1],x[2])

@pytest.mark.benchmark(group="utils")
def test_b_encode_idrec(benchmark):
    benchmark.pedantic(_b_encode_idrec,setup=gen_rand_nums, warmup_rounds=1)


#### b_decode_idrec ####

def gen_rand_idrecs():
    return (np.random.randint(0xFFFFFFFF,size=(10000,),dtype=np.uint64),), {}

@njit(cache=True)
def _b_decode_idrec(rand_idrecs):
    for x in rand_idrecs:
        decode_idrec(x)

@pytest.mark.benchmark(group="utils")
def test_b_decode_idrec(benchmark):
    benchmark.pedantic(_b_decode_idrec,setup=gen_rand_idrecs, warmup_rounds=1)



# print(_test_list_intrinsics_unicode())
# print(_test_list_intrinsics_f8())






    # _load_ptr(i8,_struct_get_data_pointer(b2)+offset) == 3


    




if __name__ == "__main__":
    # test_structref_to_meminfo()
    # test_structref_to_pointer()
    # test_direct_member_access()
    test_list_intrinsics()









    
