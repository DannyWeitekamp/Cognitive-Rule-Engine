from numbert.experimental.context import kb_context
from numbert.experimental.fact import define_fact
from numbert.experimental.kb import KnowledgeBase, KnowledgeBaseType, decode_idrec, encode_idrec, next_empty_f_id, make_f_id_empty, retracted_f_ids_for_t_id
from numba import njit
from numba.types import unicode_type, NamedTuple
from numba.core.errors import TypingError
from numba.experimental.structref import new
import logging
import numpy as np
import pytest
from collections import namedtuple
from numbert.experimental.subscriber import BaseSubscriberType, init_base_subscriber
from numbert.experimental.utils import _struct_from_meminfo


tf_spec = {"value" : "string",
        "above" : "string",
        "below" : "string",
        "to_left" : "string",
        "to_right" : "string",
        }

##### test_declare_retract #####

with kb_context("test_declare_retract"):
    TextField, TextFieldType = define_fact("TextField",tf_spec)

@njit(cache=True)
def declare_retract(kb):
    for i in range(100):
        i_s = "A" + str(i)
        kb.declare(TextField(i_s,i_s,i_s,i_s,i_s),i_s)

    for i in range(0,100,10):
        i_s = "A" + str(i)
        kb.retract(i_s)

    # print(kb.kb_data.empty_f_id_heads)
    t_id = kb.context_data.fact_to_t_id["TextField"]
    return retracted_f_ids_for_t_id(kb.kb_data,t_id).head

@njit(cache=True)
def declare_again(kb):
    for i in range(0,100,10):
        i_s = "B" + str(i)
        kb.declare(TextField(i_s,i_s,i_s,i_s,i_s),i_s)


    t_id = kb.context_data.fact_to_t_id["TextField"]
    return retracted_f_ids_for_t_id(kb.kb_data,t_id).head#kb.kb_data.empty_f_id_heads[t_id]

@njit(cache=True)
def bad_declare_type(kb):
    kb.declare("Bad")

@njit(cache=True)
def bad_retract_type(kb):
    kb.retract(["A",1])

def test_declare_retract():
    with kb_context("test_declare_retract"):
        #NRT version
        kb = KnowledgeBase()
        assert declare_retract(kb) == 10
        assert declare_again(kb) == 0

        #Python version
        kb = KnowledgeBase()
        assert declare_retract.py_func(kb) == 10
        assert declare_again.py_func(kb) == 0

        with pytest.raises(TypingError):
            bad_declare_type(kb)

        with pytest.raises(TypingError):
            bad_declare_type.py_func(kb)

        with pytest.raises(TypingError):
            bad_retract_type(kb)

        with pytest.raises(TypingError):
            bad_retract_type.py_func(kb)

##### test_modify #####
@njit(cache=True)
def modify_right(kb,fact,v):
    kb.modify(fact,"to_right",v)

@njit(cache=True)
def bad_modify_type(kb):
    kb.modify("???","to_right","???")
    

def test_modify():
    with kb_context("test_modify"):
        TextField, TextFieldType = define_fact("TextField",tf_spec)
        kb = KnowledgeBase()
        fact = TextField("A","B","C","D","E")

        modify_right(kb,fact,"nb")
        assert fact.to_right == "nb"

        modify_right.py_func(kb,fact,"py")
        assert fact.to_right == "py"

        with pytest.raises(TypingError):
            bad_modify_type(kb)

        with pytest.raises(TypingError):
            bad_modify_type.py_func(kb)
        

##### test_declare_overloading #####

with kb_context("test_declare_overloading"):
    TextField, TextFieldType = define_fact("TextField",tf_spec)

@njit(cache=True)
def declare_unnamed(kb):
    return kb.declare(TextField("A","B","C","D","E"))

def test_declare_overloading():
    with kb_context("test_declare_overloading"):
        kb = KnowledgeBase()
        idrec1 = declare_unnamed(kb)
        idrec2 = declare_unnamed.py_func(kb)
        assert idrec1 != idrec2


##### test_retract_keyerror #####
with kb_context("test_retract_keyerror"):
    TextField, TextFieldType = define_fact("TextField",tf_spec)

@njit(cache=True)
def retract_keyerror(kb):
    kb.declare(TextField("A","B","C","D","E"),"A")
    kb.retract("A")
    kb.retract("A")

def test_retract_keyerror():
    with kb_context("test_retract_keyerror"):
        #NRT version
        kb = KnowledgeBase()
        with pytest.raises(KeyError):
            retract_keyerror(kb)

        #Python version
        kb = KnowledgeBase()
        with pytest.raises(KeyError):
            retract_keyerror.py_func(kb)

##### test_all_facts_of_type #####

with kb_context("test_all_facts_of_type"):
    TextField, TextFieldType = define_fact("TextField",tf_spec)

@njit(cache=True)
def all_of_type(kb):
    return kb.all_facts_of_type(TextFieldType)

def test_all_facts_of_type():
    with kb_context("test_all_facts_of_type"):
        #NRT version
        kb = KnowledgeBase()
        declare_retract(kb)
        all_tf = all_of_type(kb)
        assert isinstance(all_tf[0],TextField)
        assert len(all_tf) == 90

        all_tf = all_of_type.py_func(kb)
        assert isinstance(all_tf[0],TextField)
        assert len(all_tf) == 90


##### test_subscriber #####

@njit(cache=True)
def dummy_subscriber_ctor():
    st = new(BaseSubscriberType)
    init_base_subscriber(st)

    return st

def test_grow_change_queues():
    with kb_context("test_grow_change_queues"):
        TextField, TextFieldType = define_fact("TextField",tf_spec)
        #NRT version
        kb = KnowledgeBase()
        dummy_subscriber = dummy_subscriber_ctor() 
        kb.add_subscriber(dummy_subscriber)

        idrec = kb.declare(TextField("A","B","C","D","E"))

        # assert kb.kb_data.subscribers[0].grow_queue.data[0] == idrec
        gr_q = kb.kb_data.grow_queue
        assert gr_q.data[gr_q.head-1] == idrec

        kb.retract(idrec)

        # assert kb.kb_data.subscribers[0].change_queue.data[0] == idrec
        ch_q = kb.kb_data.change_queue
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

with kb_context("test_kb"):
    BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})


def _benchmark_setup():
    with kb_context("test_kb"):
        kb = KnowledgeBase()
    return (kb,), {}

#### declare_10000 ####



@njit(cache=True)
def _delcare_10000(kb):
    out = np.empty((10000,),dtype=np.uint64)
    for i in range(10000):
        out[i] = kb.declare(BOOP("?",i))
    return out

def test_b_declare10000(benchmark):
    benchmark.pedantic(_delcare_10000,setup=_benchmark_setup, warmup_rounds=1)

#### retract_10000 ####

def _retract_setup():
    (kb,),_ = _benchmark_setup()
    idrecs = _delcare_10000(kb)
    return (kb,idrecs), {}

@njit(cache=True)
def _retract_10000(kb,idrecs):
    for idrec in idrecs:
        kb.retract(idrec)

def test_b_retract10000(benchmark):
    benchmark.pedantic(_retract_10000,setup=_retract_setup, warmup_rounds=1)







if __name__ == "__main__":
    test_modify()
    # test_encode_decode()
    test_declare_retract()
    test_retract_keyerror()
    # test_subscriber()
    test_all_facts_of_type()
