import numpy as np
from numba import njit, f8
from numbert.experimental.utils import _meminfo_from_struct, _struct_from_meminfo, _cast_structref, struct_get_attr_offset
from numbert.experimental.subscriber import BaseSubscriberType
from numbert.experimental.fact import define_fact
from numbert.experimental.kb import KnowledgeBase
from numbert.experimental.context import kb_context
from numbert.experimental.predicate_node import get_alpha_predicate_node, get_beta_predicate_node, BasePredicateNodeType, get_linked_instance
from numbert.experimental.predicate_node import get_alpha_predicate_node_definition, get_beta_predicate_node_definition
from numbert.experimental.test_kb import _delcare_10000, _retract_10000
import pytest

@njit
def njit_update(pt):
    meminfo = _meminfo_from_struct(pt)
    subscriber = _struct_from_meminfo(BaseSubscriberType,meminfo)
    subscriber.update_func(meminfo)

@njit
def filter_alpha(pn,link_data, inds):
    # print(pn.filter_func)
    mi = _meminfo_from_struct(pn)
    return pn.filter_func(mi, link_data, inds)

@njit
def filter_beta(pn,link_data, left_inds, right_inds):
    # print(pn.filter_func)
    mi = _meminfo_from_struct(pn)
    return pn.filter_func(mi, link_data, left_inds, right_inds)

@njit
def cast(pn,ty):
    return _cast_structref(ty, pn)

# @njit
# def offset(inst,attr):
#     return _struct_get_attr_offset(inst,attr)
from numbert.experimental.predicate_node import base_subscriber_fields, basepredicate_node_fields, alpha_predicate_node_fields, beta_predicate_node_fields, GenericAlphaPredicateNodeType, GenericBetaPredicateNodeType

def test_predicate_node_sanity():
    with kb_context("test_predicate_node_sanity"):
        BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})
        kb = KnowledgeBase()

        pn = get_alpha_predicate_node(BOOPType,"B", "<",9)
        pnc = cast(pn, BasePredicateNodeType)
        png = cast(pn, GenericAlphaPredicateNodeType)
        attrs_pn  = np.array([struct_get_attr_offset(pnc,x[0]) for x in base_subscriber_fields+basepredicate_node_fields])
        attrs_pnc = np.array([struct_get_attr_offset(pn, x[0]) for x in base_subscriber_fields+alpha_predicate_node_fields])
        attrs_png = np.array([struct_get_attr_offset(png,x[0]) for x in base_subscriber_fields+alpha_predicate_node_fields])
        assert np.array_equal(attrs_pn,attrs_pnc[:len(attrs_pn)])
        assert np.array_equal(attrs_pn,attrs_png[:len(attrs_pn)])
        assert np.array_equal(attrs_pnc,attrs_png)
        
        ld = get_linked_instance(pnc, kb)

        assert len(filter_alpha(png, ld, np.arange(5)))==0

        pn = get_beta_predicate_node(BOOPType,"B", "<", BOOPType,"B")
        pnc = cast(pn, BasePredicateNodeType)
        png = cast(pn, GenericBetaPredicateNodeType)

        ld = get_linked_instance(pnc, kb)

        attrs_pn  = np.array([struct_get_attr_offset(pnc,x[0]) for x in base_subscriber_fields+basepredicate_node_fields])
        attrs_pnc = np.array([struct_get_attr_offset(pn, x[0]) for x in base_subscriber_fields+beta_predicate_node_fields])
        attrs_png = np.array([struct_get_attr_offset(png,x[0]) for x in base_subscriber_fields+beta_predicate_node_fields])

        assert np.array_equal(attrs_pn,attrs_pnc[:len(attrs_pn)])
        assert np.array_equal(attrs_pn,attrs_png[:len(attrs_pn)])
        assert np.array_equal(attrs_pnc,attrs_png)
        assert len(filter_beta(png, ld, np.arange(5), np.arange(5))) == 0
        # print('l')
        # pn.filter(np.arange(5))
        # pnn = get_linked_instance(cast(pn, BasePredicateNodeType), kb)




def test_alpha_predicate_node():
    with kb_context("test_alpha_predicate_node"):
        BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

        kb = KnowledgeBase()
        pn = get_alpha_predicate_node(BOOPType,"B", "<",9)
        ld = get_linked_instance(pn, kb)

        # kb.add_subscriber(pn)

        x = BOOP("x",7)
        y = BOOP("y",11)
        z = BOOP("z",8)

        assert len(ld.truth_values) == 0        

        kb.declare(x)
        kb.declare(y)
        kb.declare(z)

        # njit_update(pn)
        inds = filter_alpha(pn, ld, np.arange(3))
        assert np.array_equal(inds, [0,2]) 
        assert all(ld.truth_values[:3,0] == [1,0,1])

        kb.modify(x,"B", 100)
        kb.modify(y,"B", 3)
        kb.modify(z,"B", 88)

        #Checks doesn't change before update
        assert all(ld.truth_values[:3,0] == [1,0,1])
        inds = filter_alpha(pn, ld, np.arange(3))
        assert np.array_equal(inds, [1]) 
        assert all(ld.truth_values[:3,0] == [0,1,0])

        kb.retract(x)
        kb.retract(y)
        kb.retract(z)

        #Checks doesn't change before update
        assert all(ld.truth_values[:3,0] == [0,1,0])

        inds = filter_alpha(pn, ld, np.arange(3))
        print(inds)
        assert np.array_equal(inds, []) 

        # print(ld.truth_values)
        #Checks that retracted facts show up as u1.nan = 0XFF
        assert all(ld.truth_values[:3,0] == [0xFF,0xFF,0xFF])

        kb.declare(x)
        kb.declare(y)
        kb.declare(z)
        kb.modify(z,"A","Z")
        kb.modify(x,"B",0)
        kb.modify(y,"B",0)
        kb.modify(z,"B",0)

        inds = filter_alpha(pn, ld, np.arange(3))
        assert np.array_equal(inds, [0,1,2]) 

        assert all(ld.truth_values[:3,0] == [1,1,1])

def test_beta_predicate_node_1_typed():
    with kb_context("test_beta_predicate_node_1_typed"):
        BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

        kb = KnowledgeBase()
        pn = get_beta_predicate_node(BOOPType,"B", "<", BOOPType,"B")
        # png = cast(pn, GenericBetaPredicateNodeType)
        ld = get_linked_instance(pn, kb)
        # kb.add_subscriber(pn)

        x = BOOP("x",7)
        y = BOOP("y",11)
        z = BOOP("z",8)

        assert len(ld.truth_values) == 0        

        kb.declare(x)
        kb.declare(y)
        kb.declare(z)

        # njit_update(pn)
        inds = filter_beta(pn, ld, np.arange(3), np.arange(3))
        # print("B")
        # print(inds)

        assert np.array_equal(inds, [[0, 1], [0, 2], [2, 1]])
        # print(ld.truth_values)
        assert all(ld.truth_values[0,:3] == [0,1,1])
        assert all(ld.truth_values[1,:3] == [0,0,0])
        assert all(ld.truth_values[2,:3] == [0,1,0])

        q = BOOP("q",-7)
        r = BOOP("r",-11)
        t = BOOP("t",-8)
        kb.declare(q)
        kb.declare(r)
        kb.declare(t)

        inds = filter_beta(pn, ld, np.arange(6), np.arange(6))
        # print(inds)
        assert np.array_equal(inds,[[0,1],[0,2],[2,1],[3,0],[3,1],
                [3,2],[4,0],[4,1],[4,2],[4,3],[4,5],[5,0],[5,1],[5,2],[5,3]])
        # njit_update(pn)
        # print(ld.truth_values[:6,:6])
        assert all(ld.truth_values[0,:6] == [0,1,1,0,0,0])
        assert all(ld.truth_values[1,:6] == [0,0,0,0,0,0])
        assert all(ld.truth_values[2,:6] == [0,1,0,0,0,0])
        assert all(ld.truth_values[3,:6] == [1,1,1,0,0,0])
        assert all(ld.truth_values[4,:6] == [1,1,1,1,0,1])
        assert all(ld.truth_values[5,:6] == [1,1,1,1,0,0])

        kb.modify(r,"B", 0)
        kb.modify(y,"B", 0)

        inds = filter_beta(pn, ld, np.arange(6), np.arange(6))

        # njit_update(pn)
        # print(ld.truth_values[:6,:6])
        assert all(ld.truth_values[0,:6] == [0,0,1,0,0,0])
        assert all(ld.truth_values[1,:6] == [1,0,1,0,0,0])
        assert all(ld.truth_values[2,:6] == [0,0,0,0,0,0])
        assert all(ld.truth_values[3,:6] == [1,1,1,0,1,0])
        assert all(ld.truth_values[4,:6] == [1,0,1,0,0,0])
        assert all(ld.truth_values[5,:6] == [1,1,1,1,1,0])

        kb.retract(r)
        kb.retract(y)

        inds = filter_beta(pn, ld, np.arange(6), np.arange(6))

        # njit_update(pn)
        # print(ld.truth_values[:6,:6])
        assert all(ld.truth_values[0,:6] == [0,0,1,0,0,0])
        assert all(ld.truth_values[1,:6] == [0,0,0,0,0,0])
        assert all(ld.truth_values[2,:6] == [0,0,0,0,0,0])
        assert all(ld.truth_values[3,:6] == [1,0,1,0,0,0])
        assert all(ld.truth_values[4,:6] == [0,0,0,0,0,0])
        assert all(ld.truth_values[5,:6] == [1,0,1,1,0,0])


def test_beta_predicate_node_2_typed():
    with kb_context("test_beta_predicate_node_2_typed"):
        BOOP1, BOOP1Type = define_fact("BOOP1",{"A": "number", "B" : "string"})
        BOOP2, BOOP2Type = define_fact("BOOP2",{"A": "string", "B" : "number"})

        kb = KnowledgeBase()

        pn = get_beta_predicate_node(BOOP1Type,"A", "<", BOOP2Type,"B")
        ld = get_linked_instance(pn, kb)
        # kb.add_subscriber(pn)

        x1,x2 = BOOP1(7,"x"),  BOOP2("x",7.5) #<- slightly different
        y1,y2 = BOOP1(11,"y"), BOOP2("y",11)
        z1,z2 = BOOP1(8,"z"),  BOOP2("z",8)

        assert len(ld.truth_values) == 0        

        kb.declare(x1); kb.declare(x2)
        kb.declare(y1); kb.declare(y2)
        kb.declare(z1); kb.declare(z2)
        
        inds = filter_beta(pn, ld, np.arange(3), np.arange(3))
        # print(inds)

        # print(ld.truth_values)
        assert np.array_equal(inds, [[0,0],[0, 1], [0, 2], [2, 1]])
        assert all(ld.truth_values[0,:3] == [1,1,1])
        assert all(ld.truth_values[1,:3] == [0,0,0])
        assert all(ld.truth_values[2,:3] == [0,1,0])

        q1, q2 = BOOP1(-7,"q"),  BOOP2("q",-7)
        r1, r2 = BOOP1(-11,"r"), BOOP2("r",-11)
        t1, t2 = BOOP1(-8,"t"),  BOOP2("t",-8)
        kb.declare(q1); kb.declare(q2)
        kb.declare(r1); kb.declare(r2)
        kb.declare(t1); kb.declare(t2)

        inds = filter_beta(pn, ld, np.arange(6), np.arange(6))
        # print(inds)
        # print(ld.truth_values[:6,:6])
        assert np.array_equal(inds,[[0,0],[0,1],[0,2],[2,1],[3,0],[3,1],
                [3,2],[4,0],[4,1],[4,2],[4,3],[4,5],[5,0],[5,1],[5,2],[5,3]])
        assert all(ld.truth_values[0,:6] == [1,1,1,0,0,0])
        assert all(ld.truth_values[1,:6] == [0,0,0,0,0,0])
        assert all(ld.truth_values[2,:6] == [0,1,0,0,0,0])
        assert all(ld.truth_values[3,:6] == [1,1,1,0,0,0])
        assert all(ld.truth_values[4,:6] == [1,1,1,1,0,1])
        assert all(ld.truth_values[5,:6] == [1,1,1,1,0,0])

        kb.modify(r1,"A", 0); kb.modify(r2,"B", 0)
        kb.modify(y1,"A", 0); kb.modify(y2,"B", 0)

        inds = filter_beta(pn, ld, np.arange(6), np.arange(6))
        # print(ld.truth_values[:6,:6])
        assert all(ld.truth_values[0,:6] == [1,0,1,0,0,0])
        assert all(ld.truth_values[1,:6] == [1,0,1,0,0,0])
        assert all(ld.truth_values[2,:6] == [0,0,0,0,0,0])
        assert all(ld.truth_values[3,:6] == [1,1,1,0,1,0])
        assert all(ld.truth_values[4,:6] == [1,0,1,0,0,0])
        assert all(ld.truth_values[5,:6] == [1,1,1,1,1,0])

        kb.retract(r1); kb.retract(r2)
        kb.retract(y1); kb.retract(y2)
        
        inds = filter_beta(pn, ld, np.arange(6), np.arange(6))
        # print(ld.truth_values[:6,:6])
        assert all(ld.truth_values[0,:6] == [1,0,1,0,0,0])
        assert all(ld.truth_values[1,:6] == [0,0,0,0,0,0])
        assert all(ld.truth_values[2,:6] == [0,0,0,0,0,0])
        assert all(ld.truth_values[3,:6] == [1,0,1,0,0,0])
        assert all(ld.truth_values[4,:6] == [0,0,0,0,0,0])
        assert all(ld.truth_values[5,:6] == [1,0,1,1,0,0])





# import time

with kb_context("test_predicate_node"):
    BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

#### get_alpha_predicate_node ####

def _get_alpha_predicate_node():
    with kb_context("test_predicate_node"):
        pn = get_alpha_predicate_node(BOOPType,"B", "<",50)
        pn = get_alpha_predicate_node(BOOPType,"B", "<",49)

@pytest.mark.benchmark(group="setup")
def test_b_get_alpha_predicate_node(benchmark):
    benchmark.pedantic(_get_alpha_predicate_node, iterations=1)


#### setup ####

def _benchmark_setup():
    with kb_context("test_predicate_node"):
        kb = KnowledgeBase()
        pn = get_alpha_predicate_node(BOOPType,"B", "<",50)
        ld = get_linked_instance(pn,kb)
        # kb.add_subscriber(pn)
        return (kb, pn, ld), {}

@pytest.mark.benchmark(group="setup")
def test_b_setup(benchmark):
    benchmark.pedantic(_benchmark_setup, iterations=1)


#### alpha_update_post_declare ####

def _alpha_update_post_declare_setup():
    (kb,pn,ld),_ = _benchmark_setup()
    idrecs = _delcare_10000(kb)
    return (kb,pn,ld, np.arange(10000), idrecs), {}


@njit(cache=True)
def _alpha_update_post_declare_10000(kb,pn,ld,inds,idrecs):
    # njit_update(pn)
    filter_alpha(pn,ld,inds)

def test_b_alpha_update_post_declare_10000(benchmark):
    benchmark.pedantic(_alpha_update_post_declare_10000,setup=_alpha_update_post_declare_setup, warmup_rounds=1)

#### alpha_update_post_retract ####

def _alpha_update_post_retract_setup():
    (kb,pn,ld),_ = _benchmark_setup()
    idrecs = _delcare_10000(kb)
    inds = np.arange(10000)
    filter_alpha(pn,ld,inds)
    # njit_update(pn) #<-- Handle update after declare to time just change set 
    _retract_10000(kb,idrecs)

    return (kb,pn,ld, inds, idrecs), {}

@njit(cache=True)
def _alpha_update_post_retract_10000(kb,pn,ld, inds, idrecs):
    filter_alpha(pn,ld,inds)

def test_b_alpha_update_post_retract_10000(benchmark):
    benchmark.pedantic(_alpha_update_post_retract_10000,setup=_alpha_update_post_retract_setup, warmup_rounds=1)

#### alpha_update_10000_times ####

def _alpha_setup():
    with kb_context("test_predicate_node"):
        kb = KnowledgeBase()
        pn = get_alpha_predicate_node(BOOPType,"B", "<", 50)
        ld = get_linked_instance(pn, kb)
        # kb.add_subscriber(pn)
        idrecs = np.empty((10000,),dtype=np.int64)
        for i in range(10000):
            idrecs[i] = kb.declare(BOOP("?",i))
        # njit_update(pn)
        return (kb, pn, ld, np.arange(10000), idrecs), {}


@njit(cache=True)
def _alpha_update_10000_times(kb,pn,ld, inds,idrecs):
    filter_alpha(pn,ld,inds)
    for i in range(10000):
        # idrec = kb.declare(BOOP("?",i))
        kb.retract(idrecs[i])
        filter_alpha(pn,ld,inds)
        # njit_update(pn)
        # njit_update(pn)

def test_b_alpha_update_10000_times(benchmark):
    benchmark.pedantic(_alpha_update_10000_times,setup=_alpha_setup, warmup_rounds=1)



def _beta_setup():
    with kb_context("test_predicate_node"):
        kb = KnowledgeBase()
        pn = get_beta_predicate_node(BOOPType,"B", "<", BOOPType,"B")
        # kb.add_subscriber(pn)
        ld = get_linked_instance(pn, kb)
        idrecs = np.empty((100,),dtype=np.int64)
        for i in range(100):
            idrecs[i] = kb.declare(BOOP("?",i))
        return (kb, pn, ld, np.arange(100), idrecs), {}


@njit(cache=True)
def _beta_update_100_times(kb,pn,ld, inds,idrecs):
    filter_beta(pn,ld,inds,inds)
    for i in range(100):
        kb.retract(idrecs[i])
        filter_beta(pn,ld,inds,inds)
        # njit_update(pn)

def test_b_beta_update_100x100(benchmark):
    benchmark.pedantic(_beta_update_100_times,setup=_beta_setup, warmup_rounds=1)




if __name__ == "__main__":
    test_predicate_node_sanity()
    # test_alpha_predicate_node()
    # test_beta_predicate_node_1_typed()
    # test_beta_predicate_node_2_typed()

