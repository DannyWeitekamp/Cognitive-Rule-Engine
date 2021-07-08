import numpy as np
from numba import njit, f8, i8
from numba.typed import List, Dict
from numba.types import DictType, ListType, unicode_type
from cre.op import Op
from cre.sc_planner2 import (gen_apply_multi_source,
                     apply_multi, SetChainingPlanner, insert_record,
                     join_records_of_type, forward_chain_one, next_rec_entry,
                    rec_entry_from_ptr, SC_Record_EntryType, retrace_goals_back_one, expl_tree_ctor,
                    build_explanation_tree, ExplanationTreeType)
from cre.utils import _pointer_from_struct_incref, _list_from_ptr, _dict_from_ptr, _struct_from_pointer

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

def test_apply_multi():
    planner = setup_float()
    rec = apply_multi(Add, planner, 0)
    d_typ = DictType(f8,i8)
    @njit(cache=True)
    def summary_vals_map(planner,target=6.0):
        d = _dict_from_ptr(d_typ, planner.val_map_ptr_dict['float64'])
        return len(d), min(d), max(d)

    @njit(cache=True)
    def args_for(planner,target=6.0):
        d = _dict_from_ptr(d_typ, planner.val_map_ptr_dict['float64'])
        l = List()
        re = rec_entry_from_ptr(d[target])
        while(re is not None):
            print(re.args)
            l.append(re.args)
            re = next_rec_entry(re)            
        return l

    assert summary_vals_map(planner) == (8,1,8)
    assert np.array_equal(np.array(args_for(planner,6)),
                 np.array([[4, 2],[3, 3],[2, 4]]))

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
        d = _dict_from_ptr(d_typ, planner.val_map_ptr_dict[typ_name])
        return len(l), min(l),max(l),len(d), min(d),max(d)

    assert summary_stats(planner,'float64', 1) == (12, 0.0, 16.0, 12, 0.0, 16.0)


@njit(cache=True)
def summary_stats(planner, typ_name, depth, d_typ, l_typ):
    print("----",typ_name, depth,"-----")
    l = _list_from_ptr(l_typ, planner.flat_vals_ptr_dict[(typ_name,depth)])
    d = _dict_from_ptr(d_typ, planner.val_map_ptr_dict[typ_name])
    # print(l)
    # print(d)
    return len(l), min(l),max(l),len(d), min(d),max(d)

def test_forward_chain_one():
    fd_typ = DictType(f8, i8)
    fl_typ = ListType(f8)
    sd_typ = DictType(unicode_type, i8)
    sl_typ = ListType(unicode_type)
    
    planner = setup_float()
    planner = setup_str(planner)
    forward_chain_one(planner, [Add,Multiply,Concatenate])

    assert summary_stats(planner,'float64',1,fd_typ,fl_typ) == \
        (12, 0.0, 16.0, 12, 0.0, 16.0)

    assert summary_stats(planner,'unicode_type',1,sd_typ,sl_typ) == \
        (25, 'AA', 'EE', 25, 'AA', 'EE')


    forward_chain_one(planner, [Add,Multiply,Concatenate])


    print(summary_stats(planner,'float64',1,fd_typ,fl_typ))
    assert summary_stats(planner,'float64',2,fd_typ,fl_typ) == \
        (53, 0.0, 256.0, 53, 0.0, 256.0)

    print(summary_stats(planner,'unicode_type',1,sd_typ,sl_typ))
    assert summary_stats(planner,'unicode_type',2,sd_typ,sl_typ) == \
        (650, 'AA', 'EEEE', 650, 'AA', 'EEEE')


def setup_retrace(n=5):
    planner = setup_float(n=n)
    planner = setup_str(planner,n=n)
    forward_chain_one(planner, [Add,Multiply,Concatenate])
    forward_chain_one(planner, [Add,Multiply,Concatenate])
    return planner


@njit(cache=True)
def tree_str(root,ind=0):
    if(len(root.children) == 0): return "?"
    s = ''
    for child in root.children:
        op, child_arg_ptrs = child.op, child.child_arg_ptrs

        s += op.name + "("
        print(child_arg_ptrs)
        for ptr in child_arg_ptrs:
            for i in range(ind):
                s += " "
            ch_expl = _struct_from_pointer(ExplanationTreeType, ptr)
            print("str",tree_str(ch_expl, ind+1))
            s += tree_str(ch_expl, ind+1)
            s += ","
        s += ")\n"
    return s
        




def test_build_explanation_tree():
    planner = setup_retrace()
    root = build_explanation_tree(planner, f8, 36.0)

    print(tree_str(root))
    # for child in root.children:
    #     op, args = child
    #     print(op.name)



    # goals = Dict.empty(f8,ExplanationTreeType)
    # goals[36.0] = expl_tree_ctor()
    # retrace_goals_back_one(planner, goals)
    # goals = List([36.0])
    # e_trees = List([expl_tree_ctor()])
    # print(retrace_back_one(planner, DictType(f8,i8),'float64', goals, e_trees))

    # goals = List(["AABC"])
    # e_trees = List([expl_tree_ctor()])
    # print(retrace_back_one(planner, DictType(unicode_type,i8),'unicode_type', goals, e_trees))

    # build_explanation_tree(planner,36.0, f8)


def benchmark_apply_multi():
    planner = setup_float(n=1000)

    apply_multi(Add, planner, 0)
    with PrintElapse("benchmark_apply_multi"):
        for i in range(10):
            apply_multi(Add, planner, 0)

def benchmark_retrace_goals_back_one():
    planner = setup_retrace()
    goals = List([36.0])

    apply_multi(Add, planner, 0)
    with PrintElapse("benchmark_retrace_back_one"):
        for i in range(10):
            retrace_goals_back_one(planner, DictType(f8,i8),'float64', goals)




if __name__ == "__main__":
    # test_apply_multi()
    # test_insert_record()
    # test_join_records_of_type()
    # test_forward_chain_one()
    test_build_explanation_tree()
    # benchmark_apply_multi()
    # benchmark_retrace_back_one()
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


