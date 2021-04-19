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



def test_matching():
    # with kb_context("test_link"):
    # BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})
    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
    r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

    c = (l1.B > 0) & (l1.B != 3) & (l1.B < 4) & (l2.B != 3) | \
        (l1.B == 3) 

    kb = KnowledgeBase()

    for i in range(5):
        boop = BOOP(str(i), i)
        # print(i, ":", get_ptr(boop))
        kb.declare(boop)
    
    cl = get_linked_conditions_instance(c, kb)

    Bs = boop_Bs_from_ptrs(get_pointer_matches_from_linked(cl))
    print("Bs", Bs)

    print("----------")

    c = (l1.B <= 1) & (l1.B < l2.B) & (l2.B <= r1.B)

    cl = get_linked_conditions_instance(c, kb)

    Bs = boop_Bs_from_ptrs(get_pointer_matches_from_linked(cl))
    print("Bs", Bs)

















if(__name__ == "__main__"):
    test_matching()
