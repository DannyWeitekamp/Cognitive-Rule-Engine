import numpy as np
from numba import njit, f8
from numba.typed import List
from cre.conditions import *
from cre.memory import Memory
from cre.context import cre_context
# from cre.matching import get_ptr_matches,_get_matches
from cre.utils import _struct_from_pointer, _pointer_from_struct

# with cre_context("test_matching"):
    

def match_names(c,mem=None):
    out = []
    for m in c.get_matches(mem):
        # print(m)
        out.append([x.name for x in m])
        # print("X")
    return out

def set_is_same(a,b):
    setA = set([tuple(x) for x in a])
    setB = set([tuple(x) for x in b])
    return setA == setB


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


def mem_w_n_boops(n,BOOP):
    mem = Memory()
    for i in range(n):
        boop = BOOP(str(i), i)
        mem.declare(boop)
    return mem


# def test_matching():
#     with cre_context("test_matching_unconditioned"):
#         BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})
#         # with cre_context("test_link"):
#         # BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})
#         mem = mem_w_n_boops(5)


#         l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
#         r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

#         c = (l1.B > 0) & (l1.B != 3) & (l1.B < 4) & (l2.B != 3) | \
#             (l1.B == 3) 

#         print(c)

#         cl = get_linked_conditions_instance(c, mem)

#         Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
#         print("Bs", Bs)

#         print("----------")

#         c = (l1.B <= 1) & (l1.B < l2.B) & (l2.B <= r1.B)

#         cl = get_linked_conditions_instance(c, mem)

#         Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
#         print("Bs", Bs)


# def test_matching_unconditioned():
#     with cre_context("test_matching_unconditioned"):
#         BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})
#         mem = mem_w_n_boops(5)


#         l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
#         c = l1 & l2

#         cl = get_linked_conditions_instance(c, mem)
#         Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
#         print("Bs", Bs)


#         l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
#         c = l1 & (l2.B < 3)

#         cl = get_linked_conditions_instance(c, mem)
#         Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
#         print("Bs", Bs)

def test_ref_matching():
    with cre_context("test_ref_matching"):
        TestLL, TestLLType = define_fact("TestLL",{"name": "string", "B" :'number', "nxt" : "TestLL"})
        mem = Memory()
        a = TestLL("A", B=0)
        b = TestLL("B", B=1, nxt=a)
        c = TestLL("C", B=2, nxt=b)
        mem.declare(a)
        mem.declare(b)
        mem.declare(c)

        print(a,b,c)

        x1, x2 = Var(TestLL,"x1"), Var(TestLL,"x2")
        c = (x1.nxt == x2)
        # assert str(c) == '(x1.nxt == x2)'
        print(c)
        # print(match_names(c,mem))

        assert sorted(match_names(c,mem)) == [['B','A'],['C','B']]
        # cl = get_linked_conditions_instance(c, mem)
        # print(get_ptr_matches(cl))
        # Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))

        # print(Bs)
        c = x1.nxt == None
        # assert str(c) == '(x1.nxt == None)'
        print(c)

        assert match_names(c,mem) == [['A']]

        # cl = get_linked_conditions_instance(c, mem)
        # print(get_ptr_matches(cl))


        c = x1.nxt == None
        # assert str(c) == '(x1.nxt == None)'
        print(c)

        assert match_names(c,mem) == [['A']]

        # cl = get_linked_conditions_instance(c, mem)
        # print(get_ptr_matches(cl))



def test_multiple_deref():
    with cre_context("test_multiple_deref"):
        TestLL, TestLLType = define_fact("TestLL",{"name": "string", "B" :'number', "nxt" : "TestLL"})
        mem = Memory()

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

        print([(i,pointer_from_struct(x)) for i,x in enumerate([a,b1,c1,b2,c2])])

        mem.declare(a)
        mem.declare(b1)
        mem.declare(c1)
        mem.declare(b2)
        mem.declare(c2)

        v1 = Var(TestLL,'v1')
        v2 = Var(TestLL,'v2')

        # One Deep check same fact instance
        c = (v1.nxt != None) & (v1.nxt == v2.nxt) & (v1 != v2)
        names = match_names(c, mem) 
        assert set_is_same(names, [['B1', 'B2'], ['B2', 'B1']])

        # One Deep check same B value
        c = (v1.nxt != None) & (v1.nxt.B == v2.nxt.B) & (v1 != v2)
        names = match_names(c, mem) 
        assert set_is_same(names, [['B1', 'B2'], ['C1', 'C2'], ['B2', 'B1'], ['C2', 'C1']])

        # Two Deep w/ Permutions
        c = (v1.nxt.nxt != None) & (v1.nxt.nxt == v2.nxt.nxt) & (v1 != v2)
        names = match_names(c, mem)
        assert set_is_same(names, [['C1', 'C2'], ['C2', 'C1']])

        # Two Deep w/o Permutions. 
        # NOTE: v1 < v2 compares ptrs, can't guarentee order 
        c = (v1.nxt.nxt != None) & (v1.nxt.nxt == v2.nxt.nxt) & (v1 < v2)
        names = match_names(c, mem)
        assert names == [['C1', 'C2']] or names == [['C2', 'C1']]

        mem.declare(TestLL("D1", B=3, nxt=c1))
        mem.declare(TestLL("D2", B=3, nxt=c2))

        # Three Deep (use None) -- helps check that dereference errors 
        #  are treated internally as errors instead evaluating to 0.
        c = (v1.nxt.nxt.nxt != None) & (v1.nxt.nxt.nxt == v2.nxt.nxt.nxt) & (v1 != v2)
        names = match_names(c, mem) 
        assert set_is_same(names, [['D1', 'D2'], ['D2', 'D1']])



def test_NOT():
    with cre_context("test_NOT"):
        BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})

        mem = mem_w_n_boops(3,BOOP)

        x1,x2,x3 = Var(BOOP,'x1'), Var(BOOP,'x2'), Var(BOOP,'x3')
        c = (x1.B > 1) & (x2.B < 1) & NOT(x3.B > 9000) 

        assert match_names(c,mem) == [['2','0']]

        over_9000 = BOOP("over_9000", 9001)
        mem.declare(over_9000)

        assert match_names(c,mem) == []

        mem.retract(over_9000)

        assert match_names(c,mem) == [['2','0']]

        #TODO: Make sure NOT() works on betas

def test_list():
    with cre_context("test_list"):
        TList, TListType = define_fact("TList",{"name" : "string", "items" : "ListType(string)"})
        v1 = Var(TList,"v1")
        v2 = Var(TList,"v2")

        mem = Memory()
        mem.declare(TList("A", List(["x","a"])))
        mem.declare(TList("B", List(["x","b"])))

        c = (v1 != v2) & (v1.items[0] == v2.items[0])
        names = match_names(c, mem)
        assert set_is_same(names, [['A','B'],['B','A']])

        c = (v1 != v2) & (v1.items[1] != v2.items[1])

        names = match_names(c, mem)
        assert set_is_same(names,  [['A','B'],['B','A']])

        mem.declare(TList("C", List(["x","c"])))
        mem.declare(TList("D", List(["x","x"])))
        #TODO: Self-Beta-like conditions
        c = v1.items[0] != v1.items[1]
        names = match_names(c, mem)
        assert set_is_same(names, [["A"],["B"],["C"]])

def test_multiple_types():
    with cre_context("test_multiple_types"):
        BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})
        TList, TListType = define_fact("TList",{"name" : "string", "items" : "ListType(string)"})

        b = Var(BOOP,"b")
        t = Var(TList,"t")

        mem = Memory()
        mem.declare(BOOP("A", 0))
        mem.declare(BOOP("B", 1))
        mem.declare(TList("A", List(["x","a"])))
        mem.declare(TList("B", List(["x","b"])))

        c = t & b & (t.name == b.name)
        assert match_names(c, mem) == [["A","A"], ["B","B"]]

        c = t & b & (b.name == t.name)
        assert match_names(c, mem) == [["A","A"], ["B","B"]]













with cre_context("test_matching_benchmarks"):
    BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})



@njit(cache=True)
def apply_it(mem,l1,l2,r1):
    print(l1,l2,r1)
    mem.declare(BOOP("??",1000+r1.B))

# @njit(cache=True)
# def apply_all_matches(c, f, mem):
#     cl = get_linked_conditions_instance(c, mem)
#     ptr_matches = get_ptr_matches(cl)
#     for match in ptr_matches:
#         arg0 = _struct_from_pointer(BOOPType,match[0]) 
#         arg1 = _struct_from_pointer(BOOPType,match[1]) 
#         arg2 = _struct_from_pointer(BOOPType,match[2]) 
#         f(mem,arg0,arg1,arg2)


# def test_applying():
#     with cre_context("test_matching_benchmarks"):
#         mem = mem_w_n_boops(5)
#         l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
#         r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")
#         c = (l1.B <= 1) & (l1.B < l2.B) & (l2.B <= r1.B)
#         apply_all_matches(c,apply_it,mem)
#         apply_all_matches(c,apply_it,mem)



def matching_1_t_4_lit_setup():
    with cre_context("test_matching_benchmarks") as ctxt:
        BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})
        mem = mem_w_n_boops(100,BOOP)

        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

        c = (l1.B > 0) & (l1.B != 3) & (l1.B < 4) & (l2.B != 3) 

        c.get_matches(mem)

        # cl = get_linked_conditions_instance(c, mem)
# 
        # Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
        return (c,mem), {}

from cre.rete import get_match_iter, update_graph, ReteGraphType, parse_mem_change_queue, update_node
@njit(cache=True)
def check_twice(c,mem):
    graph = _struct_from_pointer(ReteGraphType, c.matcher_inst_ptr)
    parse_mem_change_queue(graph)

    for lst in graph.nodes_by_nargs:
        for node in lst:
            update_node(node)        
    # get_match_iter(mem,c)

    # with ctxt:
    #     for x in c.get_matches(mem):
    #         pass
        
        # c.get_matches(mem)
        


def test_b_matching_1_t_4_lit(benchmark):
    benchmark.pedantic(check_twice,setup=matching_1_t_4_lit_setup, warmup_rounds=1, rounds=100)


if(__name__ == "__main__"):
    pass

    check_twice(*matching_1_t_4_lit_setup()[0])
    # test_applying()
    # test_matching()
    # test_matching_unconditioned()
    # test_ref_matching()
    # test_multiple_deref()
    # test_list()
    # test_list()
    # test_multiple_types()
    # test_ref_matching()
    # import pytest.__main__.benchmark
    # matching_1_t_4_lit_setup()
    # test_NOT()
    # test_b_matching_1_t_4_lit()
    # test_multiple_types()
