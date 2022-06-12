from cre.context import cre_context
from cre.fact import define_fact
from cre.tuple_fact import TF, TupleFact
from cre.memset import MemSet, MemSetType, decode_idrec, encode_idrec, next_empty_f_id, make_f_id_empty, retracted_f_ids_for_t_id
from numba import njit
from numba.types import unicode_type, NamedTuple
from numba.typed import List
from numba.core.errors import TypingError
from numba.experimental.structref import new
import logging
import numpy as np
import pytest
from collections import namedtuple
from cre.subscriber import BaseSubscriberType, init_base_subscriber
from cre.utils import _struct_from_meminfo
import gc
from numba.core.runtime.nrt import rtsys
from weakref import WeakKeyDictionary


tf_spec = {"value" : "string",
        "above" : "string",
        "below" : "string",
        "to_left" : "string",
        "to_right" : "string",
        }

##### test_declare_retract #####

with cre_context("test_declare_retract"):
    TextField = define_fact("TextField",tf_spec)

@njit(cache=True)
def declare_retract(ms, t_id):
    for i in range(100):
        i_s = "A" + str(i)
        ms.declare(TextField(i_s,i_s,i_s,i_s,i_s),i_s)

    for i in range(0,100,10):
        i_s = "A" + str(i)
        ms.retract(i_s)

    # print(ms.empty_f_id_heads)
    # t_id = ms.context_data.fact_to_t_id["TextField"]
    return retracted_f_ids_for_t_id(ms,t_id).head

@njit(cache=True)
def declare_again(ms,t_id):
    for i in range(0,100,10):
        i_s = "B" + str(i)
        ms.declare(TextField(i_s,i_s,i_s,i_s,i_s),i_s)


    # t_id = ms.context_data.fact_to_t_id["TextField"]
    return retracted_f_ids_for_t_id(ms,t_id).head#ms.empty_f_id_heads[t_id]





def test_declare_retract():
    with cre_context("test_declare_retract"):
        #NRT version
        ms = MemSet()
        assert declare_retract(ms,TextField.t_id) == 10
        assert declare_again(ms,TextField.t_id) == 0
        print("A")
        #Python version
        ms = MemSet()
        assert declare_retract.py_func(ms,TextField.t_id) == 10
        assert declare_again.py_func(ms,TextField.t_id) == 0

def test_declare_retract_tuple_fact():
    with cre_context("test_declare_retract_tuple_fact"):
        #NRT version
        ms = MemSet()
        idrec1 = ms.declare(("A",1))
        idrec2 = ms.declare(TF("A",1))
        print(decode_idrec(idrec1))
        print(decode_idrec(idrec2))

##### test_modify #####
@njit(cache=True)
def modify_right(ms,fact,v):
    ms.modify(fact,"to_right",v)

@njit(cache=True)
def bad_modify_type(ms):
    ms.modify("???","to_right","???")
    

def test_modify():
    with cre_context("test_modify"):
        TextField = define_fact("TextField",tf_spec)
        ms = MemSet()
        fact = TextField("A","B","C","D","E")

        modify_right(ms,fact,"nb")
        assert fact.to_right == "nb"

        modify_right.py_func(ms,fact,"py")
        assert fact.to_right == "py"

        with pytest.raises(TypingError):
            bad_modify_type(ms)

        with pytest.raises(TypingError):
            bad_modify_type.py_func(ms)
        

##### test_declare_overloading #####

with cre_context("test_declare_overloading"):
    TextField = define_fact("TextField",tf_spec)

@njit(cache=True)
def declare_unnamed(ms):
    return ms.declare(TextField("A","B","C","D","E"))

def test_declare_overloading():
    with cre_context("test_declare_overloading"):
        ms = MemSet()
        idrec1 = declare_unnamed(ms)
        idrec2 = declare_unnamed.py_func(ms)
        assert idrec1 != idrec2


##### test_retract_keyerror #####
with cre_context("test_retract_keyerror"):
    TextField = define_fact("TextField",tf_spec)

@njit(cache=True)
def retract_keyerror(ms):
    ms.declare(TextField("A","B","C","D","E"),"A")
    ms.retract("A")
    ms.retract("A")

def test_retract_keyerror():
    with cre_context("test_retract_keyerror"):
        #NRT version
        ms = MemSet()
        with pytest.raises(KeyError):
            retract_keyerror(ms)

        #Python version
        ms = MemSet()
        with pytest.raises(KeyError):
            retract_keyerror.py_func(ms)

##### test_get_facts #####

# @njit(cache=True)
# def all_of_type(ms):
#     return ms.all_facts_of_type(TextField)
from itertools import product
def test_get_facts():
    with cre_context("test_get_facts"):
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3 = define_fact("BOOP3", spec3)

        ms = MemSet()
        ms.declare(BOOP1("A",1))
        ms.declare(BOOP1("B",2))
        ms.declare(BOOP1("C",3))

        @njit(cache=True)
        def iter_b1(ms):
            l = List()
            for x in ms.get_facts(BOOP1):
                l.append(x)
            return l

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            assert isinstance(all_tf[0], BOOP1._fact_proxy)
            assert len(all_tf) == 3
        
        # ms = MemSet()
        ms.declare(BOOP2("D",4))
        ms.declare(BOOP2("E",5))
        ms.declare(BOOP2("F",6))        

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            print(all_tf)
            assert isinstance(all_tf[0], BOOP1._fact_proxy)
            assert len(all_tf) == 6


        # ms = MemSet()
        ms.declare(BOOP3("G",7))
        ms.declare(BOOP3("H",8))
        ms.declare(BOOP3("I",9))  

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            print(all_tf)
            assert isinstance(all_tf[0], BOOP1._fact_proxy)
            assert len(all_tf) == 9


def test_retroactive_register():
    with cre_context("test_context_retroactive_register") as context:
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3 = define_fact("BOOP3", spec3)
    # Check that retroactive registration works fine for declare()
    with cre_context("other_context") as context:
        with pytest.raises(ValueError):
            context.get_t_id(name="BOOP1")

        ms = MemSet()
        ms.declare(BOOP1("A",1))
        ms.declare(BOOP1("A",2))
        ms.declare(BOOP2("B",2, 3))
        ms.declare(BOOP2("B",3, 3))
        ms.declare(BOOP3("C",3, 4, 5))
        ms.declare(BOOP3("C",4, 4, 5))

        b1_t_id = context.get_t_id(_type=BOOP1)
        b2_t_id = context.get_t_id(_type=BOOP2)
        b3_t_id = context.get_t_id(_type=BOOP3)

        assert b1_t_id != b2_t_id and b2_t_id != b3_t_id

        # c = context
        # assert np.array_equal(c.get_parent_t_ids(t_id=b3_t_id),[b1_t_id,b2_t_id])
        # assert np.array_equal(c.get_parent_t_ids(t_id=b2_t_id),[b1_t_id])
        # assert np.array_equal(c.get_parent_t_ids(t_id=b1_t_id),[])
        
        # assert np.array_equal(c.get_child_t_ids(t_id=b3_t_id),[b3_t_id])
        # assert np.array_equal(c.get_child_t_ids(t_id=b2_t_id),[b2_t_id,b3_t_id])
        # assert np.array_equal(c.get_child_t_ids(t_id=b1_t_id),[b1_t_id,b2_t_id,b3_t_id])

       

# from itertools import product

# NOTE: Something funny going on here, getting errors like:
#    "Invalid use of getiter with parameters (cre.FactIterator[BOOP1])"
def _test_iter_facts():
    with cre_context("test_iter_facts"):
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3 = define_fact("BOOP3", spec3)

        ms = MemSet()
        ms.declare(BOOP1("A",1))
        ms.declare(BOOP1("B",2))
        ms.declare(BOOP1("C",3))

        @njit(cache=True)
        def iter_b1(ms):
            l = List()
            for x in ms.iter_facts(BOOP1):
                l.append(x)
            return l

        iter_b1(ms)
        raise ValueError()

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            assert isinstance(all_tf[0], BOOP1)
            assert len(all_tf) == 3
        
        # ms = MemSet()
        ms.declare(BOOP2("D",4))
        ms.declare(BOOP2("E",5))
        ms.declare(BOOP2("F",6))        

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            print(all_tf)
            assert isinstance(all_tf[0], BOOP1)
            assert len(all_tf) == 6


        # ms = MemSet()
        ms.declare(BOOP3("G",7))
        ms.declare(BOOP3("H",8))
        ms.declare(BOOP3("I",9))  

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            print(all_tf)
            assert isinstance(all_tf[0], BOOP1)
            assert len(all_tf) == 9

        # for i in range(2):
        #     all_tf = iter_b1(ms)
        #     print(all_tf)
        #     assert isinstance(all_tf[0], BOOP1)
        #     assert len(all_tf) == 3

        # all_tf = list(ms.get_facts(BOOP1))
        # assert isinstance(all_tf[0], BOOP1)
        # assert len(all_tf) == 3


##### test_subscriber #####

# @njit(cache=True)
# def dummy_subscriber_ctor():
#     st = new(BaseSubscriberType)
#     init_base_subscriber(st)

#     return st

# def test_grow_change_queues():
#     with cre_context("test_grow_change_queues"):
#         TextField = define_fact("TextField",tf_spec)
#         #NRT version
#         ms = MemSet()
#         dummy_subscriber = dummy_subscriber_ctor() 
#         ms.add_subscriber(dummy_subscriber)

#         idrec = ms.declare(TextField("A","B","C","D","E"))

#         ch_q = ms.change_queue
#         assert ch_q.data[ch_q.head-1] == idrec

#         t_id, f_id, _ = decode_idrec(idrec)

#         # assert ms.subscribers[0].grow_queue.data[0] == idrec
#         # gr_q = ms.grow_queue
#         # assert gr_q.data[gr_q.head-1] == idrec

#         ms.retract(idrec)

#         # assert ms.subscribers[0].change_queue.data[0] == idrec
#         # ch_q = ms.change_queue
#         assert ch_q.data[ch_q.head-1] == encode_idrec(t_id, f_id, 0xFF)

with cre_context("test_mem_leaks"):
    TextField = define_fact("TextField",tf_spec)
    BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})

def used_bytes():
    stats = rtsys.get_allocation_stats()
    print(stats)
    return stats.alloc-stats.free


def test_mem_leaks():
    ''' Test for MemSet leaks in mem. This test might fail if other tests fail
        even if there is nothing wrong '''
    with cre_context("test_mem_leaks"):
        init_used = used_bytes()

        # Empty Mem
        ms = MemSet()
        ms = None; gc.collect()
        print(used_bytes()-init_used)
        assert used_bytes()-init_used <= 0

        # Declare a bunch of stuff
        ms = MemSet()
        for i in range(100):
            tf = TextField()
            ms.declare(tf, str(i))
        tf, ms = None, None; gc.collect()
        assert used_bytes()-init_used <= 0

        # Declare More than one kind of stuff
        ms = MemSet()
        for i in range(100):
            tf = TextField(value=str(i))
            b = BOOP(A=str(i), B=i)
            ms.declare(tf, str(i))
            ms.declare(b, "B"+str(i))
        tf, ms, b = None, None, None; gc.collect()
        assert used_bytes()-init_used <= 0

        # Declare More than one kind of stuff and retract some
        ms = MemSet()
        for i in range(100):
            tf = TextField(value=str(i))
            b = BOOP(A=str(i), B=i)
            ms.declare(tf, str(i))
            ms.declare(b, "B"+str(i))
        for i in range(0,100,10):
            ms.retract(str(i))
            ms.retract("B"+str(i))
        tf, ms, b = None, None, None; gc.collect()
        # print(used_bytes()-init_used)
        assert used_bytes()-init_used <= 0





        # print([x._meminfo.refcount for x in w.keys()])
        
        # print([x._meminfo.refcount for x in w.keys()])
        
        


        
        # print()
        # print(rtsys.get_allocation_stats())
    # print(rtsys.get_allocation_stats())
        




###################### BENCHMARKS ########################


#### b_encode_idrec ####

def gen_rand_nums():
    return (np.random.randint(1000,size=(10000,3)),), {}

@njit(cache=True)
def _b_encode_idrec(rand_nums):
    for x in rand_nums:
        encode_idrec(x[0],x[1],x[2])

def test_b_encode_idrec(benchmark):
    benchmark.pedantic(_b_encode_idrec,setup=gen_rand_nums, warmup_rounds=1)


#### b_decode_idrec ####

def gen_rand_idrecs():
    return (np.random.randint(0xFFFFFFFF,size=(10000,),dtype=np.uint64),), {}

@njit(cache=True)
def _b_decode_idrec(rand_idrecs):
    for x in rand_idrecs:
        decode_idrec(x)

def test_b_decode_idrec(benchmark):
    benchmark.pedantic(_b_decode_idrec,setup=gen_rand_idrecs, warmup_rounds=1)

#### helper funcs #####

with cre_context("test_memset"):
    BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})


def _benchmark_setup():
    with cre_context("test_memset"):
        mem = MemSet()
    return (mem,), {}

#### declare_10000 ####



@njit(cache=True)
def _delcare_10000(ms):
    out = np.empty((10000,),dtype=np.uint64)
    for i in range(10000):
        out[i] = ms.declare(BOOP("?",i))
    return out

def test_b_declare_10000(benchmark):
    benchmark.pedantic(_delcare_10000,setup=_benchmark_setup, warmup_rounds=1)

#### retract_10000 ####

def _retract_setup():
    (ms,),_ = _benchmark_setup()
    idrecs = _delcare_10000(ms)
    return (ms,idrecs), {}

@njit(cache=True)
def _retract_10000(ms,idrecs):
    for idrec in idrecs:
        ms.retract(idrec)

def test_b_retract_10000(benchmark):
    benchmark.pedantic(_retract_10000,setup=_retract_setup, warmup_rounds=1)


#### get_facts_10000 ####

def get_facts_setup():
    with cre_context("test_memset"):
        ms = MemSet()
        _delcare_10000(ms)
    return (ms,), {}

@njit(cache=True)
def _get_facts_10000(ms):
    for x in ms.get_facts(BOOP):
        pass

def test_b_get_facts_10000(benchmark):
    benchmark.pedantic(_get_facts_10000,setup=get_facts_setup, warmup_rounds=1)



if __name__ == "__main__":
    import faulthandler; faulthandler.enable()
    # test_declare_retract()
    # test_retroactive_register()
    # test_declare_retract_tuple_fact()
    # test_declare_overloading()
    # test_modify()
    
    # test_retract_keyerror()
    # test_subscriber()
    # test_get_facts()
    # test_mem_leaks()
    # test_get_facts()
    # _test_iter_facts()

    _delcare_10000(MemSet())
