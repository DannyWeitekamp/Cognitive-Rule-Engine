import numpy as np
from numba import njit, f8, i8
from numba.typed import List, Dict
from numba.types import DictType, ListType, unicode_type
from cre.op import Op
from cre.sc_planner import (gen_apply_multi_source,
                     apply_multi, SetChainingPlanner, insert_record,
                     join_records_of_type, forward_chain_one)
from cre.utils import _pointer_from_struct_incref, _list_from_ptr, _dict_from_ptr

import time
class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.2f} ms')

class Add(Op):
    signature = f8(f8,f8)        
    short_hand = '({0}+{1})'
    def check(a, b):
        return a > 0
    def call(a, b):
        return a + b

class Multiply(Op):
    signature = f8(f8,f8)
    short_hand = '({0}*{1})'
    def check(a, b):
        return b != 0
    def call(a, b):
        return a * b  

class Concatenate(Op):
    signature = unicode_type(unicode_type,unicode_type)
    short_hand = '({0}+{1})'
    def call(a, b):
        return a + b  

def setup_float(planner=None,n=5):
    if(planner is None):
        planner = SetChainingPlanner()

    @njit(cache=True)
    def inject_float_data(planner,n):
        l = List.empty_list(f8,n)
        for x in np.arange(n,dtype=np.float64):
            l.append(x)

        planner.flat_vals_ptr_dict[('float64',0)] = _pointer_from_struct_incref(l)

    inject_float_data(planner,n)

    return planner

def setup_str(planner=None,n=5):
    if(planner is None):
        planner = SetChainingPlanner()

    @njit(cache=True)
    def inject_str_data(planner,n):
        l = List.empty_list(unicode_type,n)
        for x in range(65,n+65):
            l.append(chr(x))

        planner.flat_vals_ptr_dict[('unicode_type',0)] = _pointer_from_struct_incref(l)

    inject_str_data(planner,n)

    return planner

# def test_setup():
#     planner = setup_float()
#     l_typ = ListType(f8)
#     @njit(cache=True)
#     def print_boop(planner):
#         ptr = planner.flat_vals_ptr_dict[('float64',0)]
#         print(_list_from_ptr(l_typ,ptr))
#     print_boop(planner)


def test_apply_multi():
    planner = setup_float()
    rec = apply_multi(Add, planner, 0)
    assert np.array_equal(rec.hist.reshape(rec.hist_shape),
             np.array([[0,0,0,0,0], #0 implied failed check()
                       [1,2,3,4,5],
                       [2,3,4,5,6],
                       [3,4,5,6,7],
                       [4,5,6,7,8]]))
    
    d_typ = DictType(f8,i8)
    @njit(cache=True)
    def summary_vals_to_uid(rec):
        d = _dict_from_ptr(d_typ, rec.vals_to_uid_ptr)
        return len(d), min(d), max(d)

    assert summary_vals_to_uid(rec) == (8,1,8)

def test_insert_record():
    planner = setup_float()
    rec = apply_multi(Add, planner, 0)
    insert_record(planner, rec, 'float64', 1)
    @njit(cache=True)
    def len_f_recs(planner,typ_name,depth):
        return len(planner.forward_records[depth][typ_name])

    assert len_f_recs(planner,'float64',1) == 1

def test_join_records_of_type():
    planner = setup_float()
    rec = apply_multi(Add, planner, 0)
    insert_record(planner, rec, 'float64', 1)
    rec = apply_multi(Multiply, planner, 0)
    insert_record(planner, rec, 'float64', 1)

    d_typ = DictType(f8, i8)
    l_typ = ListType(f8)
    join_records_of_type(planner,1,'float64',f8, d_typ)

    @njit(cache=True)
    def summary_stats(planner, typ_name, depth):
        l = _list_from_ptr(l_typ, planner.flat_vals_ptr_dict[(typ_name,depth)])
        d = _dict_from_ptr(d_typ, planner.vals_to_depth_ptr_dict[typ_name])
        return len(l), min(l),max(l),len(d), min(d),max(d)

    assert summary_stats(planner,'float64', 1) == (12, 0.0, 16.0, 12, 0.0, 16.0)


def test_forward_chain_one():
    planner = setup_float()
    planner = setup_str(planner)
    forward_chain_one(planner, [Add,Multiply,Concatenate])
    forward_chain_one(planner, [Add,Multiply,Concatenate])

    fd_typ = DictType(f8, i8)
    fl_typ = ListType(f8)
    sd_typ = DictType(unicode_type, i8)
    sl_typ = ListType(unicode_type)
    @njit(cache=True)
    def summary_stats(planner, typ_name, depth, d_typ, l_typ):
        l = _list_from_ptr(l_typ, planner.flat_vals_ptr_dict[(typ_name,depth)])
        d = _dict_from_ptr(d_typ, planner.vals_to_depth_ptr_dict[typ_name])
        print(l)
        print(d)
        return len(l), min(l),max(l),len(d), min(d),max(d)

    print(summary_stats(planner,'float64',1,fd_typ,fl_typ))
    print(summary_stats(planner,'float64',2,fd_typ,fl_typ))

    print(summary_stats(planner,'unicode_type',1,sd_typ,sl_typ))
    print(summary_stats(planner,'unicode_type',2,sd_typ,sl_typ))








def benchmark_apply_multi():
    planner = setup_float(n=1000)

    apply_multi(Add, planner, 0)
    with PrintElapse("test_apply_multi"):
        for i in range(10):
            apply_multi(Add, planner, 0)





if __name__ == "__main__":
    # test_apply_multi()
    # test_insert_record()
    # test_join_records_of_type()
    test_forward_chain_one()
    # benchmark_apply_multi()
        # test_apply_multi()
# from numba import njit, i8
# from numba.typed import Dict
# from numba.types import ListType
# import numpy as np
# import dill
# from cre.utils import _struct_from_pointer, _pointer_from_struct_incref
# from cre.condensed_chainer import CondensedRecord
# from cre_cache.Add._5e24697b8e500d3d837dca80591bde623483d2322c5204e56fd36c79ddc2ed7d import call, check

# typ0, = dill.loads(b'\x80\x04\x95\xd5\x00\x00\x00\x00\x00\x00\x00\x8c\x19numba.core.types.abstract\x94\x8c\x13_type_reconstructor\x94\x93\x94\x8c\x07copyreg\x94\x8c\x0e_reconstructor\x94\x93\x94\x8c\x18numba.core.types.scalars\x94\x8c\x05Float\x94\x93\x94\x8c\ndill._dill\x94\x8c\n_load_type\x94\x93\x94\x8c\x06object\x94\x85\x94R\x94N\x87\x94}\x94(\x8c\x04name\x94\x8c\x07float64\x94\x8c\x08bitwidth\x94K@\x8c\x05_code\x94K\x17u\x87\x94R\x94\x85\x94.')
# ret_typ = dill.loads(b'\x80\x04\x95\xd3\x00\x00\x00\x00\x00\x00\x00\x8c\x19numba.core.types.abstract\x94\x8c\x13_type_reconstructor\x94\x93\x94\x8c\x07copyreg\x94\x8c\x0e_reconstructor\x94\x93\x94\x8c\x18numba.core.types.scalars\x94\x8c\x05Float\x94\x93\x94\x8c\ndill._dill\x94\x8c\n_load_type\x94\x93\x94\x8c\x06object\x94\x85\x94R\x94N\x87\x94}\x94(\x8c\x04name\x94\x8c\x07float64\x94\x8c\x08bitwidth\x94K@\x8c\x05_code\x94K\x17u\x87\x94R\x94.')

# l_typ0 = ListType(typ0)
# @njit(cache=True)
# def Add_apply_multi(planner, depth, start0=0, start1=0):
#     tup0 = ('float64',depth)
#     print("START")
#     if(tup0 in planner.flat_vals_ptr_dict):
#         iter_ptr0 = planner.flat_vals_ptr_dict[tup0]
#         iter0 = _list_from_ptr(l_typ0, iter_ptr0)
#         print(iter0)
#     else:
#         return None
#     print("End")
#     for x in iter0:
#         print(x)
#     print("?",len(iter0))
#     l0, l1 = len(iter0)-start0, len(iter0)-start1
#     hist_shape = (l0, l1)
#     print(hist_shape )
#     hist = np.zeros(hist_shape, dtype=np.uint64)
#     vals_to_uid = Dict.empty(ret_typ, i8)
#     print(hist)
#     uid=1
#     for i0 in range(start0,len(iter0)):
#         for i1 in range(start1,len(iter0)):
#             print(i0, i1)
#             a0,a1 = iter0[i0],iter0[i1]
#             if(not check(a0,a1)): continue
#             v = call(a0,a1)
#             if(v in vals_to_uid):
#                 hist[i0,i1] = vals_to_uid[v]
#             else:
#                 hist[i0,i1] = uid
#                 vals_to_uid[v] = uid; uid+=1
#     vals_to_uid_ptr = _pointer_from_struct_incref(vals_to_uid)
#     return CondensedRecord(hist.flatten(), hist_shape, vals_to_uid_ptr)
    

# print(apply_multi(Add,planner, 0))
# print(
