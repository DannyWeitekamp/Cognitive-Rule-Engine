import numpy as np
from numba import njit
from numbert.experimental.utils import _meminfo_from_struct, _struct_from_meminfo
from numbert.experimental.subscriber import BaseSubscriberType
from numbert.experimental.fact import define_fact
from numbert.experimental.kb import KnowledgeBase
from numbert.experimental.context import kb_context
from numbert.experimental.predicate_node import get_alpha_predicate_node, get_beta_predicate_node
from numbert.experimental.test_kb import _delcare_10000, _retract_10000
import pytest

@njit
def njit_update(pt):
    meminfo = _meminfo_from_struct(pt)
    subscriber = _struct_from_meminfo(BaseSubscriberType,meminfo)
    subscriber.update_func(meminfo)


def test_alpha_predicate_node():
    with kb_context("test_alpha_predicate_node"):
        BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

        kb = KnowledgeBase()

        pn = get_alpha_predicate_node(BOOPType,"B", "<",9)

        kb.add_subscriber(pn)

        x = BOOP("x",7)
        y = BOOP("y",11)
        z = BOOP("z",8)

        assert len(pn.truth_values) == 0        

        kb.declare(x)
        kb.declare(y)
        kb.declare(z)

        njit_update(pn)
        assert all(pn.truth_values[:3] == [1,0,1])

        kb.modify(x,"B", 100)
        kb.modify(y,"B", 3)
        kb.modify(z,"B", 88)

        #Checks doesn't change before update
        assert all(pn.truth_values[:3] == [1,0,1])
        njit_update(pn)
        assert all(pn.truth_values[:3] == [0,1,0])

        kb.retract(x)
        kb.retract(y)
        kb.retract(z)

        #Checks doesn't change before update
        assert all(pn.truth_values[:3] == [0,1,0])

        njit_update(pn)
        # print(pn.truth_values)
        #Checks that retracted facts show up as u1.nan = 0XFF
        assert all(pn.truth_values[:3] == [0xFF,0xFF,0xFF])

        kb.declare(x)
        kb.declare(y)
        kb.declare(z)
        kb.modify(z,"A","Z")
        kb.modify(x,"B",0)
        kb.modify(y,"B",0)
        kb.modify(z,"B",0)

        njit_update(pn)

        assert all(pn.truth_values[:3] == [1,1,1])

def test_beta_predicate_node():
    with kb_context("test_beta_predicate_node"):
        BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

        kb = KnowledgeBase()

        pn = get_beta_predicate_node(BOOPType,"B", "<", BOOPType,"B")

        kb.add_subscriber(pn)

        x = BOOP("x",7)
        y = BOOP("y",11)
        z = BOOP("z",8)

        assert len(pn.truth_values) == 0        

        kb.declare(x)
        kb.declare(y)
        kb.declare(z)

        njit_update(pn)

        print(pn.truth_values)
        assert all(pn.truth_values[0,:3] == [0,1,1])
        assert all(pn.truth_values[1,:3] == [0,0,0])
        assert all(pn.truth_values[2,:3] == [0,1,0])




# import time

with kb_context("test_predicate_node"):
    BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

#### get_alpha_predicate_node ####

def _get_alpha_predicate_node():
    with kb_context("test_predicate_node"):
        pn = get_alpha_predicate_node(BOOPType,"B", "<",50)
        pn = get_alpha_predicate_node(BOOPType,"B", "<",49)

def test_b_get_alpha_predicate_node(benchmark):
    benchmark.pedantic(_get_alpha_predicate_node, iterations=1)


#### setup ####

def _benchmark_setup():
    with kb_context("test_predicate_node"):
        kb = KnowledgeBase()
        pn = get_alpha_predicate_node(BOOPType,"B", "<",50)
        kb.add_subscriber(pn)
        return (kb, pn), {}

def test_b_setup(benchmark):
    benchmark.pedantic(_benchmark_setup, iterations=1)


#### alpha_update_post_declare ####

def _alpha_update_post_declare_setup():
    (kb,pn),_ = _benchmark_setup()
    idrecs = _delcare_10000(kb)
    return (kb,pn,idrecs), {}


@njit(cache=True)
def _alpha_update_post_declare_10000(kb,pn,idrecs):
    njit_update(pn)

def test_b_alpha_update_post_declare_10000(benchmark):
    benchmark.pedantic(_alpha_update_post_declare_10000,setup=_alpha_update_post_declare_setup, warmup_rounds=1)

#### alpha_update_post_retract ####

def _alpha_update_post_retract_setup():
    (kb,pn),_ = _benchmark_setup()
    idrecs = _delcare_10000(kb)
    njit_update(pn) #<-- Handle update after declare to time just change set 
    _retract_10000(kb,idrecs)

    return (kb,pn), {}

@njit(cache=True)
def _alpha_update_post_retract_10000(kb,pn):
    njit_update(pn)

def test_b_alpha_update_post_retract_10000(benchmark):
    benchmark.pedantic(_alpha_update_post_retract_10000,setup=_alpha_update_post_retract_setup, warmup_rounds=1)

#### alpha_update_10000_times ####

@njit(cache=True)
def _alpha_update_10000_times(kb,pn):
    for i in range(10000):
        idrec = kb.declare(BOOP("?",i))
        njit_update(pn)
        kb.retract(idrec)
        # njit_update(pn)

def test_b_alpha_update_10000_times(benchmark):
    benchmark.pedantic(_alpha_update_10000_times,setup=_benchmark_setup, warmup_rounds=1)
    



if __name__ == "__main__":
    # test_alpha_predicate_node()
    test_beta_predicate_node()

