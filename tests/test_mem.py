from cre.context import cre_context
from cre.fact import define_fact
from cre.memory import Memory, MemoryType, decode_idrec, encode_idrec, next_empty_f_id, make_f_id_empty, retracted_f_ids_for_t_id
from numba import njit
from numba.types import unicode_type, NamedTuple
from numba.core.errors import TypingError
from numba.experimental.structref import new
import logging
import numpy as np
import pytest
from collections import namedtuple
from cre.subscriber import BaseSubscriberType, init_base_subscriber
from cre.utils import _struct_from_meminfo


tf_spec = {"value" : "string",
        "above" : "string",
        "below" : "string",
        "to_left" : "string",
        "to_right" : "string",
        }

##### test_declare_retract #####

with cre_context("test_declare_retract"):
    TextField, TextFieldType = define_fact("TextField",tf_spec)

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

@njit(cache=True)
def bad_declare_type(mem):
    mem.declare("Bad")

@njit(cache=True)
def bad_retract_type(mem):
    mem.retract(["A",1])

def test_declare_retract():
    with cre_context("test_declare_retract"):
        #NRT version
        mem = Memory()
        assert declare_retract(mem) == 10
        assert declare_again(mem) == 0

        #Python version
        mem = Memory()
        assert declare_retract.py_func(mem) == 10
        assert declare_again.py_func(mem) == 0

        with pytest.raises(TypingError):
            bad_declare_type(mem)

        with pytest.raises(TypingError):
            bad_declare_type.py_func(mem)

        with pytest.raises(TypingError):
            bad_retract_type(mem)

        with pytest.raises(TypingError):
            bad_retract_type.py_func(mem)

##### test_modify #####
@njit(cache=True)
def modify_right(mem,fact,v):
    mem.modify(fact,"to_right",v)

@njit(cache=True)
def bad_modify_type(mem):
    mem.modify("???","to_right","???")
    

def test_modify():
    with cre_context("test_modify"):
        TextField, TextFieldType = define_fact("TextField",tf_spec)
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
    TextField, TextFieldType = define_fact("TextField",tf_spec)

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
    TextField, TextFieldType = define_fact("TextField",tf_spec)

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

##### test_all_facts_of_type #####

with cre_context("test_iter_facts_of_type"):
    TextField, TextFieldType = define_fact("TextField",tf_spec)

# @njit(cache=True)
# def all_of_type(mem):
#     return mem.all_facts_of_type(TextFieldType)

def test_iter_facts_of_type():
    with cre_context("test_iter_facts_of_type"):
        #NRT version
        mem = Memory()
        declare_retract(mem)
        all_tf = list(mem.iter_facts_of_type(TextField))
        # all_tf = all_of_type(mem)
        assert isinstance(all_tf[0],TextField)
        assert len(all_tf) == 90

        all_tf = list(mem.iter_facts_of_type(TextField))
        assert isinstance(all_tf[0],TextField)
        assert len(all_tf) == 90


##### test_subscriber #####

@njit(cache=True)
def dummy_subscriber_ctor():
    st = new(BaseSubscriberType)
    init_base_subscriber(st)

    return st

def test_grow_change_queues():
    with cre_context("test_grow_change_queues"):
        TextField, TextFieldType = define_fact("TextField",tf_spec)
        #NRT version
        mem = Memory()
        dummy_subscriber = dummy_subscriber_ctor() 
        mem.add_subscriber(dummy_subscriber)

        idrec = mem.declare(TextField("A","B","C","D","E"))

        # assert mem.mem_data.subscribers[0].grow_queue.data[0] == idrec
        # gr_q = mem.mem_data.grow_queue
        # assert gr_q.data[gr_q.head-1] == idrec

        mem.retract(idrec)

        # assert mem.mem_data.subscribers[0].change_queue.data[0] == idrec
        ch_q = mem.mem_data.change_queue
        assert ch_q.data[ch_q.head-1] == idrec



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
    BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})


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

def test_b_declare10000(benchmark):
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

def test_b_retract10000(benchmark):
    benchmark.pedantic(_retract_10000,setup=_retract_setup, warmup_rounds=1)







if __name__ == "__main__":
    # test_declare_overloading()
    # test_modify()
    # test_encode_decode()
    # test_declare_retract()
    # test_retract_keyerror()
    # test_subscriber()
    test_iter_facts_of_type()
