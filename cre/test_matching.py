import numpy as np
from numba import njit, f8
from numba.typed import List
from cre.condition_node import *
from cre.kb import KnowledgeBase
from cre.context import kb_context
from cre.matching import get_ptr_matches,_get_matches
from cre.utils import _struct_from_pointer, _pointer_from_struct

# with kb_context("test_matching"):
    

def match_names(c,kb=None):
    out = []
    for m in c.get_matches(kb):
        print(m)
        out.append([x.name for x in m])
        # print("X")
    return out

@njit(cache=True)
def boop_Bs_from_ptrs(ptr_matches):
    out = np.empty(ptr_matches.shape,dtype=np.float64)
    for i, match in enumerate(ptr_matches):
        for j, ptr in enumerate(match):
            boop = _struct_from_pointer(BOOPType, ptr)
            out[i,j] = _struct_from_pointer(BOOPType, ptr).B
    return out

@njit(cache=True)
def get_ptr(fact):
    return _pointer_from_struct(fact)


def kb_w_n_boops(n):
    kb = KnowledgeBase()
    for i in range(n):
        boop = BOOP(str(i), i)
        kb.declare(boop)
    return kb


def test_matching():
    with kb_context("test_matching_unconditioned"):
        BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})
        # with kb_context("test_link"):
        # BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})
        kb = kb_w_n_boops(5)


        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

        c = (l1.B > 0) & (l1.B != 3) & (l1.B < 4) & (l2.B != 3) | \
            (l1.B == 3) 

        print(c)

        cl = get_linked_conditions_instance(c, kb)

        Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
        print("Bs", Bs)

        print("----------")

        c = (l1.B <= 1) & (l1.B < l2.B) & (l2.B <= r1.B)

        cl = get_linked_conditions_instance(c, kb)

        Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
        print("Bs", Bs)


def test_matching_unconditioned():
    with kb_context("test_matching_unconditioned"):
        BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})
        kb = kb_w_n_boops(5)


        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        c = l1 & l2

        cl = get_linked_conditions_instance(c, kb)
        Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
        print("Bs", Bs)


        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        c = l1 & (l2.B < 3)

        cl = get_linked_conditions_instance(c, kb)
        Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
        print("Bs", Bs)

def test_ref_matching():
    with kb_context("test_ref_matching"):
        TestLL, TestLLType = define_fact("TestLL",{"name": "string", "B" :'number', "nxt" : "TestLL"})
        kb = KnowledgeBase()
        a = TestLL("A", B=0)
        b = TestLL("B", B=1, nxt=a)
        c = TestLL("C", B=2, nxt=b)
        kb.declare(a)
        kb.declare(b)
        kb.declare(c)

        print(a,b,c)

        x1, x2 = Var(TestLL,"x1"), Var(TestLL,"x2")
        c = (x1.nxt == x2)
        # assert str(c) == '(x1.nxt == x2)'
        print(c)
        # print(match_names(c,kb))

        assert sorted(match_names(c,kb)) == [['B','A'],['C','B']]
        # cl = get_linked_conditions_instance(c, kb)
        # print(get_ptr_matches(cl))
        # Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))

        # print(Bs)
        c = x1.nxt == 0
        # assert str(c) == '(x1.nxt == None)'
        print(c)

        assert match_names(c,kb) == [['A']]

        # cl = get_linked_conditions_instance(c, kb)
        # print(get_ptr_matches(cl))


        c = x1.nxt == None
        # assert str(c) == '(x1.nxt == None)'
        print(c)

        assert match_names(c,kb) == [['A']]

        # cl = get_linked_conditions_instance(c, kb)
        # print(get_ptr_matches(cl))

def test_multiple_deref():
    with kb_context("test_multiple_deref"):
        TestLL, TestLLType = define_fact("TestLL",{"name": "string", "B" :'number', "nxt" : "TestLL"})
        kb = KnowledgeBase()

        #    c1   c2
        #     |   |
        #    b1   b2
        #       V
        #       a
        a = TestLL("A", B=0)
        b1 = TestLL("B1", B=1, nxt=a)
        c1 = TestLL("C1", B=2, nxt=b1)
        b2 = TestLL("B2", B=1, nxt=a)
        c2 = TestLL("C2", B=2, nxt=b2)

        kb.declare(a)
        kb.declare(b1)
        kb.declare(c1)
        kb.declare(b2)
        kb.declare(c2)

        v1 = Var(TestLL,'v1')
        v2 = Var(TestLL,'v2')

        # One Deep check same fact instance
        c = (v1.nxt != 0) & (v1.nxt == v2.nxt) & (v1 != v2)
        assert match_names(c, kb) == [['B1', 'B2'], ['B2', 'B1']]

        # One Deep check same B value
        c = (v1.nxt != 0) & (v1.nxt.B == v2.nxt.B) & (v1 != v2)
        names = match_names(c, kb) 
        print(names)
        assert names == [['B1', 'B2'], ['C1', 'C2'], ['B2', 'B1'], ['C2', 'C1']]

        # Two Deep w/ Permutions
        c = (v1.nxt.nxt != 0) & (v1.nxt.nxt == v2.nxt.nxt) & (v1 != v2)
        assert match_names(c, kb) == [['C1', 'C2'], ['C2', 'C1']]

        # Two Deep w/o Permutions. 
        # NOTE: v1 < v2 compares ptrs, can't guarentee order 
        c = (v1.nxt.nxt != 0) & (v1.nxt.nxt == v2.nxt.nxt) & (v1 < v2)
        names = match_names(c, kb)
        assert names == [['C1', 'C2']] or names == [['C2', 'C1']]

        kb.declare(TestLL("D1", B=3, nxt=c1))
        kb.declare(TestLL("D2", B=3, nxt=c2))

        # Three Deep (use None) -- helps check that dereference errors 
        #  are treated internally as errors instead evaluating to 0.
        c = (v1.nxt.nxt.nxt != 0) & (v1.nxt.nxt.nxt == v2.nxt.nxt.nxt) & (v1 != v2)
        names = match_names(c, kb) 
        print(names)
        assert names == [['D1', 'D2'], ['D2', 'D1']]



def test_NOT():
    with kb_context("test_NOT"):
        BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})

        kb = kb_w_n_boops(3)

        x1,x2,x3 = Var(BOOP,'x1'), Var(BOOP,'x2'), Var(BOOP,'x3')
        c = (x1.B > 1) & (x2.B < 1) & NOT(x3.B > 9000) 

        assert match_names(c,kb) == [['2','0']]

        over_9000 = BOOP("over_9000", 9001)
        kb.declare(over_9000)

        assert match_names(c,kb) == []

        kb.retract(over_9000)

        assert match_names(c,kb) == [['2','0']]

        #TODO: Make sure NOT() works on betas

def test_list():
    with kb_context("test_list"):
        TList, TListType = define_fact("TList",{"name" : "string", "items" : "ListType(string)"})
        v1 = Var(TList,"v1")
        v2 = Var(TList,"v2")

        kb = KnowledgeBase()
        kb.declare(TList("A", List(["x","a"])))
        kb.declare(TList("B", List(["x","b"])))

        c = (v1 != v2) & (v1.items[0] == v2.items[0])
        assert match_names(c, kb) == [['A','B'],['B','A']]

        c = (v1 != v2) & (v1.items[1] != v2.items[1])
        assert match_names(c, kb) == [['A','B'],['B','A']]

        #TODO: Self-Beta-like conditions
        c = v1.items[0] != v1.items[1]
        match_names(c, kb)






with kb_context("test_matching_benchmarks"):
    BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})



@njit(cache=True)
def apply_it(kb,l1,l2,r1):
    print(l1,l2,r1)
    kb.declare(BOOP("??",1000+r1.B))

@njit(cache=True)
def apply_all_matches(c, f, kb):
    cl = get_linked_conditions_instance(c, kb)
    ptr_matches = get_ptr_matches(cl)
    for match in ptr_matches:
        arg0 = _struct_from_pointer(BOOPType,match[0]) 
        arg1 = _struct_from_pointer(BOOPType,match[1]) 
        arg2 = _struct_from_pointer(BOOPType,match[2]) 
        f(kb,arg0,arg1,arg2)


def test_applying():
    with kb_context("test_matching_benchmarks"):
        kb = kb_w_n_boops(5)
        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")
        c = (l1.B <= 1) & (l1.B < l2.B) & (l2.B <= r1.B)
        apply_all_matches(c,apply_it,kb)
        apply_all_matches(c,apply_it,kb)



def matching_1_t_4_lit_setup():
    with kb_context("test_matching_benchmarks"):
        kb = kb_w_n_boops(100)

        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

        c = (l1.B > 0) & (l1.B != 3) & (l1.B < 4) & (l2.B != 3) 

        cl = get_linked_conditions_instance(c, kb)

        # Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
        return (cl,), {}

@njit(cache=True)
def check_twice(cl):
    get_ptr_matches(cl)
    # get_ptr_matches(cl)


def test_b_matching_1_t_4_lit(benchmark):
    benchmark.pedantic(check_twice,setup=matching_1_t_4_lit_setup, warmup_rounds=1)


if(__name__ == "__main__"):
    # test_applying()
    # test_matching()
    # test_matching_unconditioned()
    # test_ref_matching()
    # test_multiple_deref()
    test_list()
    # import pytest.__main__.benchmark
    # matching_1_t_4_lit_setup()
    # test_NOT()
