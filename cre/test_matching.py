import numpy as np
from numba import njit, f8
from numba.typed import List
from cre.condition_node import *
from cre.kb import KnowledgeBase
from cre.context import kb_context
from cre.matching import get_pointer_matches_from_linked
from cre.utils import _struct_from_pointer, _pointer_from_struct

BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

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
    # with kb_context("test_link"):
    # BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})
    kb = kb_w_n_boops(5)


    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
    r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

    c = (l1.B > 0) & (l1.B != 3) & (l1.B < 4) & (l2.B != 3) | \
        (l1.B == 3) 

    print(c)

    cl = get_linked_conditions_instance(c, kb)

    Bs = boop_Bs_from_ptrs(get_pointer_matches_from_linked(cl))
    print("Bs", Bs)

    print("----------")

    c = (l1.B <= 1) & (l1.B < l2.B) & (l2.B <= r1.B)

    cl = get_linked_conditions_instance(c, kb)

    Bs = boop_Bs_from_ptrs(get_pointer_matches_from_linked(cl))
    print("Bs", Bs)





@njit(cache=True)
def apply_it(kb,l1,l2,r1):
    print(l1,l2,r1)
    kb.declare(BOOP("??",1000+r1.B))

@njit(cache=True)
def apply_all_matches(c, f, kb):
    cl = get_linked_conditions_instance(c, kb)
    ptr_matches = get_pointer_matches_from_linked(cl)
    for match in ptr_matches:
        arg0 = _struct_from_pointer(BOOPType,match[0]) 
        arg1 = _struct_from_pointer(BOOPType,match[1]) 
        arg2 = _struct_from_pointer(BOOPType,match[2]) 
        f(kb,arg0,arg1,arg2)


def test_applying():
    kb = kb_w_n_boops(5)
    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
    r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")
    c = (l1.B <= 1) & (l1.B < l2.B) & (l2.B <= r1.B)
    apply_all_matches(c,apply_it,kb)
    apply_all_matches(c,apply_it,kb)



def matching_1_t_4_lit_setup():
    kb = kb_w_n_boops(100)

    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
    r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

    c = (l1.B > 0) & (l1.B != 3) & (l1.B < 4) & (l2.B != 3) 

    cl = get_linked_conditions_instance(c, kb)

    # Bs = boop_Bs_from_ptrs(get_pointer_matches_from_linked(cl))
    return (cl,), {}

@njit(cache=True)
def check_twice(cl):
    get_pointer_matches_from_linked(cl)
    # get_pointer_matches_from_linked(cl)


def test_b_matching_1_t_4_lit(benchmark):
    benchmark.pedantic(check_twice,setup=matching_1_t_4_lit_setup, warmup_rounds=1)













if(__name__ == "__main__"):
    test_applying()
    # test_matching()
    # test_b_matching_1_t_4_lit()
