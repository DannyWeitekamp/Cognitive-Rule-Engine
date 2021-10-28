import numpy as np
from numba import njit, f8
from numba.typed import List
from cre.conditions import *
from cre.memory import Memory
from cre.context import cre_context
# from cre.matching import get_ptr_matches,_get_matches
from cre.utils import _struct_from_ptr
from numba.core.runtime.nrt import rtsys
import gc


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
            boop = _struct_from_ptr(BOOPType, ptr)
            out[i,j] = _struct_from_ptr(BOOPType, ptr).B
    return out

@njit(cache=True)
def get_ptr(fact):
    return _raw_ptr_from_struct(fact)


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

        # print([(i,pointer_from_struct(x)) for i,x in enumerate([a,b1,c1,b2,c2])])

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


def used_bytes():
    stats = rtsys.get_allocation_stats()
    print(stats)
    return stats.alloc-stats.free













with cre_context("test_matching_benchmarks"):
    BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})


from weakref import WeakValueDictionary
def test_mem_leaks():
    # with cre_context("test_matching_benchmarks"):

    # Lead with these because in principle when an Op is typed a singleton inst is alloced
    (c,mem),_ = matching_alphas_setup()
    (c,mem),_ = matching_betas_setup()
    c,mem = None,None; gc.collect()

    init_used = used_bytes()

    #Do this to avoid the global ref from auto_aliasing stuff
    


    # print(l1)
    # print(globals()['l1'])
    # print(locals())
    # print(l1,l2)
    w = WeakValueDictionary()
    # print(l1._meminfo.refcount, l2._meminfo.refcount)
    # print([x._meminfo.refcount for x in w.keys()])
    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
    l1, l2 = None,None; gc.collect()
    print(used_bytes()-init_used)

    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
    c = (l1.B > 0)
    c = None; gc.collect()
    print(type(c),l1._meminfo.refcount,l2._meminfo.refcount)
    c, l1, l2 = None, None,None; gc.collect()
    print(used_bytes()-init_used)

    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
    c = (l1.B > 0) & (l2.B != 3) 
    c, l1, l2 = None, None,None; gc.collect()
    print(used_bytes()-init_used)




    (c,mem),_ = matching_alphas_setup()
    print(c._meminfo.refcount, mem._meminfo.refcount)
    
    mem = None; gc.collect()
    print("aft_mem",used_bytes()-init_used)
    c = None; gc.collect()
    print("aft_c",used_bytes()-init_used)
    # w[0] = c
    # w[1] = mem
    c,mem = None,None; gc.collect()
    print(used_bytes()-init_used)

    (c,mem),_ = matching_betas_setup()
    c,mem = None,None; gc.collect()
    print(used_bytes()-init_used)




@njit(cache=True)
def apply_it(mem,l1,l2,r1):
    print(l1,l2,r1)
    mem.declare(BOOP("??",1000+r1.B))

# @njit(cache=True)
# def apply_all_matches(c, f, mem):
#     cl = get_linked_conditions_instance(c, mem)
#     ptr_matches = get_ptr_matches(cl)
#     for match in ptr_matches:
#         arg0 = _struct_from_ptr(BOOPType,match[0]) 
#         arg1 = _struct_from_ptr(BOOPType,match[1]) 
#         arg2 = _struct_from_ptr(BOOPType,match[2]) 
#         f(mem,arg0,arg1,arg2)


# def test_applying():
#     with cre_context("test_matching_benchmarks"):
#         mem = mem_w_n_boops(5)
#         l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
#         r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")
#         c = (l1.B <= 1) & (l1.B < l2.B) & (l2.B <= r1.B)
#         apply_all_matches(c,apply_it,mem)
#         apply_all_matches(c,apply_it,mem)

# with cre_context("test_matching_benchmarks") as ctxt:
#     BOOP, BOOPType = define_fact("BOOP",{"name": "string", "B" : "number"})

def matching_alphas_setup():
    with cre_context("test_matching_benchmarks") as ctxt:
        mem = mem_w_n_boops(500,BOOP)

        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

        c = (l1.B > 0) & (l2.B != 3) 

        return (c,mem), {}

def matching_betas_setup():
    with cre_context("test_matching_benchmarks") as ctxt:
        mem = mem_w_n_boops(500,BOOP)

        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

        c = l1 & l2 & (l1.B > l2.B) #& ((l1.B % 2) == (l2.B + 1) % 2)

        return (c,mem), {}

from cre.rete import get_match_iter, update_graph, ReteGraphType, parse_mem_change_queue, update_node, build_rete_graph, new_match_iter, restitch_match_iter
def apply_get_matches(c,mem):
    # rete_graph = build_rete_graph(mem, c)
    # update_graph(rete_graph)
    # m_iter = new_match_iter(rete_graph)
    # restitch_match_iter(m_iter, -1)
    with cre_context("test_matching_benchmarks") as ctxt:
        c.get_matches(mem)
    # graph = _struct_from_ptr(ReteGraphType, c.matcher_inst_ptr)
    # parse_mem_change_queue(graph)

    # for lst in graph.nodes_by_nargs:
    #     for node in lst:
    #         update_node(node)        
    # get_match_iter(mem,c)

    # with ctxt:
    #     for x in c.get_matches(mem):
    #         pass
        
        # c.get_matches(mem)

@njit(cache=True)
def do_update_graph(c,mem):
    # print("BUILD")
    rete_graph = build_rete_graph(mem, c)
    # print("UDPATE")
    update_graph(rete_graph)
    # print("new iter")
    m_iter = new_match_iter(rete_graph)


def test_b_matching_alphas_lit(benchmark):
    # with cre_context("test_matching_benchmarks") as ctxt:
    benchmark.pedantic(apply_get_matches,setup=matching_alphas_setup, warmup_rounds=1, rounds=20)
    alloc_stats = rtsys.get_allocation_stats()
    gc.collect()
    assert(alloc_stats.free==alloc_stats.alloc), f'{alloc_stats}'

def test_b_matching_betas_lit(benchmark):
    # with cre_context("test_matching_benchmarks") as ctxt:
    # alloc_stats1 = rtsys.get_allocation_stats()
    benchmark.pedantic(do_update_graph,setup=matching_betas_setup, warmup_rounds=1, rounds=20)
    alloc_stats = rtsys.get_allocation_stats()
    gc.collect()
    assert(alloc_stats.free==alloc_stats.alloc), f'{alloc_stats}'


# def diff_increases()

if(__name__ == "__main__"):
    pass
    test_mem_leaks()
    # dat = matching_alphas_setup()[0]
    # dat = matching_betas_setup()[0]

    # with cre_context("test_matching_benchmarks") as ctxt:
    #     # c.get_matches(mem)

    #     mem = mem_w_n_boops(80, BOOP)
    #     print(rtsys.get_allocation_stats())
    #     mem = None
    #     print(rtsys.get_allocation_stats())

   
    # gc.collect(); alloc_stats0 = rtsys.get_allocation_stats()
    # do_update_graph(*dat)

    # # dat = None
    
    # gc.collect(); alloc_stats1 = rtsys.get_allocation_stats()
    # print(alloc_stats0.alloc-alloc_stats0.free, alloc_stats1.alloc-alloc_stats1.free)

    # do_update_graph(*dat)
    
    # gc.collect(); alloc_stats2 = rtsys.get_allocation_stats()
    # print(alloc_stats1.alloc-alloc_stats1.free, alloc_stats2.alloc-alloc_stats2.free)


    # test_ref_matching()
    # test_multiple_deref()
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
