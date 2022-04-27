from cre.context import cre_context
from cre.fact import define_fact
from cre.tuple_fact import TF, TupleFact
from cre.memory import Memory, MemoryType, decode_idrec, encode_idrec, next_empty_f_id, make_f_id_empty, retracted_f_ids_for_t_id
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
def declare_retract(mem):
    for i in range(100):
        i_s = "A" + str(i)
        mem.declare(TextField(i_s,i_s,i_s,i_s,i_s),i_s)

    for i in range(0,100,10):
        i_s = "A" + str(i)
        mem.retract(i_s)

    # print(mem.mem_data.empty_f_id_heads)
    t_id = mem.context_data.fact_to_t_id["TextField"]
    return retracted_f_ids_for_t_id(mem.mem_data,t_id).head

@njit(cache=True)
def declare_again(mem):
    for i in range(0,100,10):
        i_s = "B" + str(i)
        mem.declare(TextField(i_s,i_s,i_s,i_s,i_s),i_s)


    t_id = mem.context_data.fact_to_t_id["TextField"]
    return retracted_f_ids_for_t_id(mem.mem_data,t_id).head#mem.mem_data.empty_f_id_heads[t_id]





def test_declare_retract():
    with cre_context("test_declare_retract"):
        #NRT version
        mem = Memory()
        assert declare_retract(mem) == 10
        assert declare_again(mem) == 0
        print("A")
        #Python version
        mem = Memory()
        assert declare_retract.py_func(mem) == 10
        assert declare_again.py_func(mem) == 0

def test_declare_retract_tuple_fact():
    with cre_context("test_declare_retract_tuple_fact"):
        #NRT version
        mem = Memory()
        idrec1 = mem.declare(("A",1))
        idrec2 = mem.declare(TF("A",1))
        print(decode_idrec(idrec1))
        print(decode_idrec(idrec2))

##### test_modify #####
@njit(cache=True)
def modify_right(mem,fact,v):
    mem.modify(fact,"to_right",v)

@njit(cache=True)
def bad_modify_type(mem):
    mem.modify("???","to_right","???")
    

def test_modify():
    with cre_context("test_modify"):
        TextField = define_fact("TextField",tf_spec)
        mem = Memory()
        fact = TextField("A","B","C","D","E")

        modify_right(mem,fact,"nb")
        assert fact.to_right == "nb"

        modify_right.py_func(mem,fact,"py")
        assert fact.to_right == "py"

        with pytest.raises(TypingError):
            bad_modify_type(mem)

        with pytest.raises(TypingError):
            bad_modify_type.py_func(mem)
        

##### test_declare_overloading #####

with cre_context("test_declare_overloading"):
    TextField = define_fact("TextField",tf_spec)

@njit(cache=True)
def declare_unnamed(mem):
    return mem.declare(TextField("A","B","C","D","E"))

def test_declare_overloading():
    with cre_context("test_declare_overloading"):
        mem = Memory()
        idrec1 = declare_unnamed(mem)
        idrec2 = declare_unnamed.py_func(mem)
        assert idrec1 != idrec2


##### test_retract_keyerror #####
with cre_context("test_retract_keyerror"):
    TextField = define_fact("TextField",tf_spec)

@njit(cache=True)
def retract_keyerror(mem):
    mem.declare(TextField("A","B","C","D","E"),"A")
    mem.retract("A")
    mem.retract("A")

def test_retract_keyerror():
    with cre_context("test_retract_keyerror"):
        #NRT version
        mem = Memory()
        with pytest.raises(KeyError):
            retract_keyerror(mem)

        #Python version
        mem = Memory()
        with pytest.raises(KeyError):
            retract_keyerror.py_func(mem)

##### test_get_facts #####

# @njit(cache=True)
# def all_of_type(mem):
#     return mem.all_facts_of_type(TextField)
from itertools import product
def test_get_facts():
    with cre_context("test_get_facts"):
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3 = define_fact("BOOP3", spec3)

        mem = Memory()
        mem.declare(BOOP1("A",1))
        mem.declare(BOOP1("B",2))
        mem.declare(BOOP1("C",3))

        @njit(cache=True)
        def iter_b1(mem):
            l = List()
            for x in mem.get_facts(BOOP1):
                l.append(x)
            return l

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(mem) if t else iter_b1(mem)
            assert isinstance(all_tf[0], BOOP1._fact_proxy)
            assert len(all_tf) == 3
        
        # mem = Memory()
        mem.declare(BOOP2("D",4))
        mem.declare(BOOP2("E",5))
        mem.declare(BOOP2("F",6))        

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(mem) if t else iter_b1(mem)
            print(all_tf)
            assert isinstance(all_tf[0], BOOP1._fact_proxy)
            assert len(all_tf) == 6


        # mem = Memory()
        mem.declare(BOOP3("G",7))
        mem.declare(BOOP3("H",8))
        mem.declare(BOOP3("I",9))  

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(mem) if t else iter_b1(mem)
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

        mem = Memory()
        mem.declare(BOOP1("A",1))
        mem.declare(BOOP1("A",2))
        mem.declare(BOOP2("B",2, 3))
        mem.declare(BOOP2("B",3, 3))
        mem.declare(BOOP3("C",3, 4, 5))
        mem.declare(BOOP3("C",4, 4, 5))

        b1_t_id = context.get_t_id(fact_type=BOOP1)
        b2_t_id = context.get_t_id(fact_type=BOOP2)
        b3_t_id = context.get_t_id(fact_type=BOOP3)

        assert b1_t_id != b2_t_id and b2_t_id != b3_t_id

        assert context.get_t_id(fact_num=BOOP1._fact_num) == b1_t_id
        assert context.get_t_id(fact_num=BOOP2._fact_num) == b2_t_id
        assert context.get_t_id(fact_num=BOOP3._fact_num) == b3_t_id
        

        # NOTE: Inheritance tracking doesn't work in this case... would require some refactoring
        #  issue is that assign_name_num_to_t_id() requires a name and the fact_num of the inherited type
        # cd = context.context_data
        # print(cd.parent_t_ids)
        # print(cd.parent_t_ids[b1_t_id])
        # print(cd.parent_t_ids[b2_t_id])
        # print(cd.parent_t_ids[b3_t_id])
        # print(cd.child_t_ids[b1_t_id])
        # print(cd.child_t_ids[b2_t_id])
        # print(cd.child_t_ids[b3_t_id])
        # assert np.array_equal(cd.parent_t_ids[b1_t_id],[])
        # assert np.array_equal(cd.parent_t_ids[b2_t_id],[b1_t_id])
        # assert np.array_equal(cd.parent_t_ids[b3_t_id],[b1_t_id,b2_t_id])
        # assert np.array_equal(cd.child_t_ids[b1_t_id],[b1_t_id,b2_t_id,b3_t_id])
        # assert np.array_equal(cd.child_t_ids[b2_t_id],[b2_t_id,b3_t_id])
        # assert np.array_equal(cd.child_t_ids[b3_t_id],[b3_t_id])


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

        mem = Memory()
        mem.declare(BOOP1("A",1))
        mem.declare(BOOP1("B",2))
        mem.declare(BOOP1("C",3))

        @njit(cache=True)
        def iter_b1(mem):
            l = List()
            for x in mem.iter_facts(BOOP1):
                l.append(x)
            return l

        iter_b1(mem)
        raise ValueError()

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(mem) if t else iter_b1(mem)
            assert isinstance(all_tf[0], BOOP1)
            assert len(all_tf) == 3
        
        # mem = Memory()
        mem.declare(BOOP2("D",4))
        mem.declare(BOOP2("E",5))
        mem.declare(BOOP2("F",6))        

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(mem) if t else iter_b1(mem)
            print(all_tf)
            assert isinstance(all_tf[0], BOOP1)
            assert len(all_tf) == 6


        # mem = Memory()
        mem.declare(BOOP3("G",7))
        mem.declare(BOOP3("H",8))
        mem.declare(BOOP3("I",9))  

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(mem) if t else iter_b1(mem)
            print(all_tf)
            assert isinstance(all_tf[0], BOOP1)
            assert len(all_tf) == 9

        # for i in range(2):
        #     all_tf = iter_b1(mem)
        #     print(all_tf)
        #     assert isinstance(all_tf[0], BOOP1)
        #     assert len(all_tf) == 3

        # all_tf = list(mem.get_facts(BOOP1))
        # assert isinstance(all_tf[0], BOOP1)
        # assert len(all_tf) == 3


##### test_subscriber #####

@njit(cache=True)
def dummy_subscriber_ctor():
    st = new(BaseSubscriberType)
    init_base_subscriber(st)

    return st

def test_grow_change_queues():
    with cre_context("test_grow_change_queues"):
        TextField = define_fact("TextField",tf_spec)
        #NRT version
        mem = Memory()
        dummy_subscriber = dummy_subscriber_ctor() 
        mem.add_subscriber(dummy_subscriber)

        idrec = mem.declare(TextField("A","B","C","D","E"))

        ch_q = mem.mem_data.change_queue
        assert ch_q.data[ch_q.head-1] == idrec

        t_id, f_id, _ = decode_idrec(idrec)

        # assert mem.mem_data.subscribers[0].grow_queue.data[0] == idrec
        # gr_q = mem.mem_data.grow_queue
        # assert gr_q.data[gr_q.head-1] == idrec

        mem.retract(idrec)

        # assert mem.mem_data.subscribers[0].change_queue.data[0] == idrec
        # ch_q = mem.mem_data.change_queue
        assert ch_q.data[ch_q.head-1] == encode_idrec(t_id, f_id, 0xFF)

with cre_context("test_mem_leaks"):
    TextField = define_fact("TextField",tf_spec)
    BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})

def used_bytes():
    stats = rtsys.get_allocation_stats()
    print(stats)
    return stats.alloc-stats.free


def test_mem_leaks():
    ''' Test for memory leaks in mem. This test might fail if other tests fail
        even if there is nothing wrong '''
    with cre_context("test_mem_leaks"):
        init_used = used_bytes()

        # Empty Mem
        mem = Memory()
        mem = None; gc.collect()
        print(used_bytes()-init_used)
        assert used_bytes()-init_used == 0

        # Declare a bunch of stuff
        mem = Memory()
        for i in range(100):
            tf = TextField()
            mem.declare(tf, str(i))
        tf, mem = None, None; gc.collect()
        assert used_bytes()-init_used == 0

        # Declare More than one kind of stuff
        mem = Memory()
        for i in range(100):
            tf = TextField(value=str(i))
            b = BOOP(A=str(i), B=i)
            mem.declare(tf, str(i))
            mem.declare(b, "B"+str(i))
        tf, mem, b = None, None, None; gc.collect()
        assert used_bytes()-init_used == 0

        # Declare More than one kind of stuff and retract some
        mem = Memory()
        for i in range(100):
            tf = TextField(value=str(i))
            b = BOOP(A=str(i), B=i)
            mem.declare(tf, str(i))
            mem.declare(b, "B"+str(i))
        for i in range(0,100,10):
            mem.retract(str(i))
            mem.retract("B"+str(i))
        tf, mem, b = None, None, None; gc.collect()
        # print(used_bytes()-init_used)
        assert used_bytes()-init_used == 0





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
    return (np.random.randint(0xFFFFFFFF,size=(10000,)),), {}

@njit(cache=True)
def _b_decode_idrec(rand_idrecs):
    for x in rand_idrecs:
        decode_idrec(x)

def test_b_decode_idrec(benchmark):
    benchmark.pedantic(_b_decode_idrec,setup=gen_rand_idrecs, warmup_rounds=1)

#### helper funcs #####

with cre_context("test_mem"):
    BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})


def _benchmark_setup():
    with cre_context("test_mem"):
        mem = Memory()
    return (mem,), {}

#### declare_10000 ####



@njit(cache=True)
def _delcare_10000(mem):
    out = np.empty((10000,),dtype=np.uint64)
    for i in range(10000):
        out[i] = mem.declare(BOOP("?",i))
    return out

def test_b_declare_10000(benchmark):
    benchmark.pedantic(_delcare_10000,setup=_benchmark_setup, warmup_rounds=1)

#### retract_10000 ####

def _retract_setup():
    (mem,),_ = _benchmark_setup()
    idrecs = _delcare_10000(mem)
    return (mem,idrecs), {}

@njit(cache=True)
def _retract_10000(mem,idrecs):
    for idrec in idrecs:
        mem.retract(idrec)

def test_b_retract_10000(benchmark):
    benchmark.pedantic(_retract_10000,setup=_retract_setup, warmup_rounds=1)


#### get_facts_10000 ####

def get_facts_setup():
    with cre_context("test_mem"):
        mem = Memory()
        _delcare_10000(mem)
    return (mem,), {}

@njit(cache=True)
def _get_facts_10000(mem):
    for x in mem.get_facts(BOOP):
        pass

def test_b_get_facts_10000(benchmark):
    benchmark.pedantic(_get_facts_10000,setup=get_facts_setup, warmup_rounds=1)



if __name__ == "__main__":
    import faulthandler; faulthandler.enable()
    test_retroactive_register()
    # test_declare_retract_tuple_fact()
    # test_declare_overloading()
    # test_modify()
    # test_declare_retract()
    # test_retract_keyerror()
    # test_subscriber()
    # test_get_facts()
    # test_mem_leaks()
    # test_get_facts()
    # _test_iter_facts()
