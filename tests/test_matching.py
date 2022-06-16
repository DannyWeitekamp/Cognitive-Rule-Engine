import numpy as np
from numba import njit, f8
from numba.typed import List
from cre.conditions import *
from cre.memset import MemSet
from cre.context import cre_context
# from cre.matching import get_ptr_matches,_get_matches
from cre.utils import _struct_from_ptr, _list_base,_list_base_from_ptr,_load_ptr, _incref_structref, _raw_ptr_from_struct
from numba.core.runtime.nrt import rtsys
import gc
from cre.rete import repr_match_iter_dependencies

# with cre_context("test_matching"):
    

def match_names(c,ms=None):
    out = []
    m_iter = c.get_matches(ms)
    # print(repr_match_iter_dependencies(m_iter))
    for m in m_iter:
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
            boop = _struct_from_ptr(BOOP, ptr)
            out[i,j] = _struct_from_ptr(BOOP, ptr).B
    return out

@njit(cache=True)
def get_ptr(fact):
    return _raw_ptr_from_struct(fact)

@njit(cache=True)
def declare_n_BOOPS(n,BOOP,ms):
    for i in range(n):
        boop = BOOP(str(i), i)
        ms.declare(boop)

def ms_w_n_boops(n,BOOP):
    ms = MemSet()
    declare_n_BOOPS(n,BOOP,ms)
    return ms


# def test_matching():
#     with cre_context("test_matching_unconditioned"):
#         BOOP = define_fact("BOOP",{"name": "string", "B" : "number"})
#         # with cre_context("test_link"):
#         # BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})
#         mem = mem_w_n_boops(5)


#         l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
#         r1, r2 = Var(BOOP,"r1"), Var(BOOP,"r2")

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
#         BOOP = define_fact("BOOP",{"name": "string", "B" : "number"})
#         mem = mem_w_n_boops(5)


#         l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
#         c = l1 & l2

#         cl = get_linked_conditions_instance(c, mem)
#         Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
#         print("Bs", Bs)


#         l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
#         c = l1 & (l2.B < 3)

#         cl = get_linked_conditions_instance(c, mem)
#         Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))
#         print("Bs", Bs)

def test_ref_matching():
    ''' This mostly tests PtrOps '''
    with cre_context("test_ref_matching"):
        TestDLL = define_fact("TestDLL", {"name": "string", "prev" : "TestDLL", "next" : "TestDLL"})
        ms = MemSet()

        #a -> b -> c
        a = TestDLL("A")
        b = TestDLL("B")
        c = TestDLL("C")
        a.next = b
        b.prev = a
        b.next = c
        c.prev = b
        ms.declare(a)
        ms.declare(b)
        ms.declare(c)

        print(a,b,c)

        x1, x2 = Var(TestDLL,"x1"), Var(TestDLL,"x2")
        c = (x1.next == x2)
        # assert str(c) == '(x1.next == x2)'
        # print(c)
        # print(match_names(c,ms))

        assert sorted(match_names(c,ms)) == [['A','B'],['B','C']]
        # cl = get_linked_conditions_instance(c, ms)
        # print(get_ptr_matches(cl))
        # Bs = boop_Bs_from_ptrs(get_ptr_matches(cl))

        # print(Bs)
        c = x1.next == None
        # assert str(c) == '(x1.next == None)'
        print(c)

        assert match_names(c,ms) == [['C']]

        # cl = get_linked_conditions_instance(c, ms)
        # print(get_ptr_matches(cl))


        c = x1.next == None
        # assert str(c) == '(x1.next == None)'
        print(c)

        assert match_names(c,ms) == [['C']]


        c = x1 == x1.next.prev

        assert match_names(c,ms) == [['A'],['B']]

        # cl = get_linked_conditions_instance(c, ms)
        # print(get_ptr_matches(cl))



def test_multiple_deref():
    with cre_context("test_multiple_deref"):
        TestLL = define_fact("TestLL",{"name": "string", "B" :'number', "nxt" : "TestLL"})
        ms = MemSet()

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

        ms.declare(a)
        ms.declare(b1)
        ms.declare(c1)
        ms.declare(b2)
        ms.declare(c2)

        v1 = Var(TestLL,'v1')
        v2 = Var(TestLL,'v2')

        # One Deep check same fact instance
        c = (v1.nxt != None) & (v1.nxt == v2.nxt) & (v1 != v2)
        names = match_names(c, ms) 
        print(names)
        assert set_is_same(names, [['B1', 'B2'], ['B2', 'B1']])

        # One Deep check same B value
        c = (v1.nxt != None) & (v1.nxt.B == v2.nxt.B) & (v1 != v2)
        names = match_names(c, ms) 
        assert set_is_same(names, [['B1', 'B2'], ['C1', 'C2'], ['B2', 'B1'], ['C2', 'C1']])

        # Two Deep w/ permutations
        c = (v1.nxt.nxt != None) & (v1.nxt.nxt == v2.nxt.nxt) & (v1 != v2)
        names = match_names(c, ms)
        assert set_is_same(names, [['C1', 'C2'], ['C2', 'C1']])

        # Two Deep w/o permutations. 
        # NOTE: v1 < v2 compares ptrs, can't guarentee order 
        c = (v1.nxt.nxt != None) & (v1.nxt.nxt == v2.nxt.nxt) & (v1 < v2)
        names = match_names(c, ms)
        assert names == [['C1', 'C2']] or names == [['C2', 'C1']]

        ms.declare(TestLL("D1", B=3, nxt=c1))
        ms.declare(TestLL("D2", B=3, nxt=c2))

        # Three Deep (use None) -- helps check that dereference errors 
        #  are treated internally as errors instead evaluating to 0.
        c = (v1.nxt.nxt.nxt != None) & (v1.nxt.nxt.nxt == v2.nxt.nxt.nxt) & (v1 != v2)
        names = match_names(c, ms) 
        assert set_is_same(names, [['D1', 'D2'], ['D2', 'D1']])



def _test_NOT():
    '''TODO: FIXME'''
    with cre_context("test_NOT"):
        BOOP = define_fact("BOOP",{"name": "string", "B" : "number"})

        ms = ms_w_n_boops(3,BOOP)

        x1,x2,x3 = Var(BOOP,'x1'), Var(BOOP,'x2'), Var(BOOP,'x3')
        c = (x1.B > 1) & (x2.B < 1) & NOT(x3.B > 9000) 

        assert match_names(c,ms) == [['2','0']]

        over_9000 = BOOP("over_9000", 9001)
        ms.declare(over_9000)

        assert match_names(c,ms) == []

        ms.retract(over_9000)

        assert match_names(c,ms) == [['2','0']]

        #TODO: Make sure NOT() works on betas
# @njit(cache=True)
# def get_list_base(lst):
#     # _incref_structref(lst)

#     base = _list_base(lst)
#     inst_ptr = _raw_ptr_from_struct(lst)
#     print("inst:", inst_ptr, "base:", _list_base_from_ptr(inst_ptr), base, lst)
#     # print("HALLOOOO", _load_ptr(unicode_type,base))
#     # print()
#     return len(_load_ptr(unicode_type,base))

def test_list():
    with cre_context("test_list"):
        TList = define_fact("TList",{"name" : "string", "items" : "ListType(string)"})
        # print(TList.name)
        v1 = Var(TList,"v1")
        v2 = Var(TList,"v2")

        # print("--len(LIST[0])", get_list_base(l1), get_list_base(l2))
        # print("--len(LIST[0])", get_list_base(l1), get_list_base(l2))

        # print("FACT BASES", a.get_ptr(), b.get_ptr())
        ms = MemSet()
        ms.declare(TList("A", List(["x","a"])),)
        ms.declare(TList("B", List(["x","b"] )))

        # print("head_type", v1.items[0].head_type)
        # print((v1.items[0] == v2.items[0]).call_sig)

        # print("MOOOO", v2.items[0].deref_offsets)

        c = (v1 != v2) & (v1.items[0] == v2.items[0])
        names = match_names(c, ms)
        assert set_is_same(names, [['A','B'],['B','A']])

        # Checks that empty matches work fine
        c = (v1 != v2) & (v1.items[0] != v2.items[0])
        names = match_names(c, ms)
        assert set_is_same(names, [])

        
        # raise ValueError()

        c = (v1 != v2) & (v1.items[1] != v2.items[1])

        names = match_names(c, ms)
        assert set_is_same(names,  [['A','B'],['B','A']])

        ms.declare(TList("C", List(["x","c"])))
        ms.declare(TList("D", List(["x","x"])))
        #TODO: Self-Beta-like conditions
        c = v1.items[0] != v1.items[1]
        names = match_names(c, ms)
        assert set_is_same(names, [["A"],["B"],["C"]])


        


        # print(l1,l2)

def test_multiple_types():
    with cre_context("test_multiple_types") as c:
        BOOP = define_fact("BOOP",{"name": "string", "B" : "number"})
        TList = define_fact("TList",{"name" : "string", "items" : "ListType(string)"})

        # print("THE ACTUAL T_IDS",BOOP.t_id,TList.t_id)

        b = Var(BOOP,"b")
        t = Var(TList,"t")

        # print(c.context_data.fact_to_t_id["BOOP"], c.context_data.fact_to_t_id["TList"])

        ms = MemSet()
        ms.declare(BOOP("A", 0))
        ms.declare(BOOP("B", 1))
        ms.declare(TList("A", List(["x","a"])))
        ms.declare(TList("B", List(["x","b"])))

        c = t & b & (t.name == b.name)
        assert match_names(c, ms) == [["A","A"], ["B","B"]]

        c = t & b & (b.name == t.name)
        assert match_names(c, ms) == [["A","A"], ["B","B"]]



def test_same_parents():
    with cre_context("test_same_parents"):
        BOOP = define_fact("BOOP",{"name": str, "mod3": float, "mod5": float, "mod7": float, "val" : float})    

        ms = MemSet()
        for i in range(106):
            ms.declare(BOOP(str(i),i%3,i%5, i%7,i))

        a = Var(BOOP,"a")
        b = Var(BOOP,"b")
        c = Var(BOOP,"c")

        # Aligned case
        conds = (a.val < b.val) & (a.mod3 == b.mod3) & (a.mod5 == b.mod5) & (a.mod7 == b.mod7)
        assert sorted(match_names(conds, ms)) == [['0', '105']]

        # Unaligned case
        conds = (a.val < b.val) & (b.mod3 == a.mod3) & (a.mod5 == b.mod5) & (b.mod7 == a.mod7)
        assert sorted(match_names(conds, ms)) == [['0', '105']]

        conds = (a & b & c &
                 (a.val < b.val) & (b.mod3 == a.mod3) & (a.mod5 == b.mod5) & (b.mod7 == a.mod7) &
                 (a.val < c.val) & (c.mod3 == a.mod7) & (c.mod3 == b.mod5) & (c.mod3 == a.mod7) & 
                 (c.val < 12))

        assert sorted(match_names(conds, ms)) == [['0', '105', '3'], ['0', '105', '6'], ['0', '105', '9']]
        
        



def used_bytes():
    stats = rtsys.get_allocation_stats()
    # print(stats)
    return stats.alloc-stats.free













with cre_context("test_matching_benchmarks"):
    BOOP = define_fact("BOOP",{"name": "string", "B" : "number"})


from weakref import WeakValueDictionary
from cre.default_ops import LessThan, ObjEquals
def test_mem_leaks():
    # with cre_context("test_matching_benchmarks"):

    # Lead with these because in principle when an Op is typed a singleton inst is alloced
    (c,ms),_ = matching_alphas_setup()
    (c,ms),_ = matching_betas_setup()
    c,ms = None,None; gc.collect()

    init_used = used_bytes()

    # Vars    
    for i in range(2):
        l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
        l1, l2 = None,None; gc.collect()
        # print(used_bytes()-init_used)
        assert used_bytes()==init_used

    # print()
    # Explicit op 1 literal
    for i in range(2):
        l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
        op = LessThan(l1.B, l1.B)
        op, l1, l2 = None, None,None; gc.collect()
        if(i==0): init_used = used_bytes()
        # print(used_bytes()-init_used)
        assert used_bytes()==init_used

    # print()
    # Explicit ptrop 1 literal
    for i in range(2):
        l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
        op = ObjEquals(l1, l2)
        op, l1, l2 = None, None,None; gc.collect()
        if(i==0): init_used = used_bytes()
        # print(used_bytes()-init_used)
        assert used_bytes()==init_used


    # Shorthand 1 literal
    for i in range(2):
        l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
        c = (l1.B > 0)
        c = None; gc.collect()
        # print(type(c),l1._msinfo.refcount,l2._msinfo.refcount)
        c, l1, l2 = None, None,None; gc.collect()
        if(i==0): init_used = used_bytes()
        # print(used_bytes()-init_used)
        assert used_bytes()==init_used
    # print()

    # AND
    for i in range(2):
        l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
        c = (l1.B > 0) & (l2.B != 3) 
        c, l1, l2 = None, None,None; gc.collect()
        if(i==0): init_used = used_bytes()
        # print(used_bytes()-init_used)
        assert used_bytes()==init_used



    # Alphas setup 
    (c,ms),_ = matching_alphas_setup()    
    c, ms = None, None; gc.collect()
    assert used_bytes()==init_used

    # Betas setup
    (c,ms),_ = matching_betas_setup()
    c,ms = None,None; gc.collect()
    assert used_bytes()==init_used

    (c,ms),_ = matching_alphas_setup()
    print("c", c._meminfo.refcount)

    # from cre.rete import update_graph, build_rete_graph
    # rete_graph = build_rete_graph(ms, c)
    # update_graph(rete_graph)
    with cre_context("test_matching_benchmarks") as ctxt:
        matches = c.get_matches(ms)

    # distr_dnf = c.distr_dnf
    # rete_graph = c.rete_graph
    # print("c", c._meminfo.refcount, 'rete_graph', rete_graph._meminfo.refcount, "matches", matches._meminfo.refcount)
    c,matches,ms,ctxt = None,None,None,None; gc.collect()

    # print('rete_graph', rete_graph._meminfo.refcount)


    # print(used_bytes(),init_used)
    assert used_bytes()==init_used









@njit(cache=True)
def apply_it(ms,l1,l2,r1):
    print(l1,l2,r1)
    ms.declare(BOOP("??",1000+r1.B))

# @njit(cache=True)
# def apply_all_matches(c, f, ms):
#     cl = get_linked_conditions_instance(c, ms)
#     ptr_matches = get_ptr_matches(cl)
#     for match in ptr_matches:
#         arg0 = _struct_from_ptr(BOOP,match[0]) 
#         arg1 = _struct_from_ptr(BOOP,match[1]) 
#         arg2 = _struct_from_ptr(BOOP,match[2]) 
#         f(ms,arg0,arg1,arg2)


# def test_applying():
#     with cre_context("test_matching_benchmarks"):
#         ms = ms_w_n_boops(5)
#         l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
#         r1, r2 = Var(BOOP,"r1"), Var(BOOP,"r2")
#         c = (l1.B <= 1) & (l1.B < l2.B) & (l2.B <= r1.B)
#         apply_all_matches(c,apply_it,ms)
#         apply_all_matches(c,apply_it,ms)

# with cre_context("test_matching_benchmarks") as ctxt:
#     BOOP = define_fact("BOOP",{"name": "string", "B" : "number"})

def matching_alphas_setup():
    with cre_context("test_matching_benchmarks") as ctxt:
        ms = ms_w_n_boops(500,BOOP)

        l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
        r1, r2 = Var(BOOP,"r1"), Var(BOOP,"r2")

        c = (l1.B > 0) & (l2.B != 3) 

        return (c,ms), {}

def matching_betas_setup():
    with cre_context("test_matching_benchmarks") as ctxt:
        ms = ms_w_n_boops(500,BOOP)

        l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
        r1, r2 = Var(BOOP,"r1"), Var(BOOP,"r2")

        c = l1 & l2 & (l1.B > l2.B) #& ((l1.B % 2) == (l2.B + 1) % 2)

        return (c,ms), {}

from cre.rete import get_match_iter, update_graph, ReteGraphType, parse_change_queue, update_node, build_rete_graph, new_match_iter#, restitch_match_iter
def apply_get_matches(c,ms):
    # rete_graph = build_rete_graph(ms, c)
    # update_graph(rete_graph)
    # m_iter = new_match_iter(rete_graph)
    # restitch_match_iter(m_iter, -1)
    with cre_context("test_matching_benchmarks") as ctxt:
        c.get_matches(ms)
    # graph = _struct_from_ptr(ReteGraphType, c.matcher_inst_ptr)
    # parse_ms_change_queue(graph)

    # for lst in graph.nodes_by_nargs:
    #     for node in lst:
    #         update_node(node)        
    # get_match_iter(ms,c)

    # with ctxt:
    #     for x in c.get_matches(ms):
    #         pass
        
        # c.get_matches(ms)

@njit(cache=True)
def do_update_graph(c,ms):
    # print("BUILD")
    rete_graph = build_rete_graph(ms, c)
    # print("UDPATE")
    update_graph(rete_graph)
    # print("new iter")
    m_iter = new_match_iter(rete_graph)


def test_b_matching_alphas_lit(benchmark):
    # with cre_context("test_matching_benchmarks") as ctxt:
    benchmark.pedantic(apply_get_matches,setup=matching_alphas_setup, warmup_rounds=1, rounds=10)
    # alloc_stats = rtsys.get_allocation_stats()
    # gc.collect()
    # print(alloc_stats.alloc-alloc_stats.free)
    # assert(alloc_stats.free==alloc_stats.alloc), f'{alloc_stats}'

def test_b_matching_betas_lit(benchmark):
    # with cre_context("test_matching_benchmarks") as ctxt:
    # alloc_stats1 = rtsys.get_allocation_stats()
    benchmark.pedantic(do_update_graph,setup=matching_betas_setup, warmup_rounds=1, rounds=10)
    # alloc_stats = rtsys.get_allocation_stats()
    # gc.collect()
    # print(alloc_stats.alloc-alloc_stats.free)
    # assert(alloc_stats.free==alloc_stats.alloc), f'{alloc_stats}'


# def diff_increases()

if(__name__ == "__main__"):
    import faulthandler; faulthandler.enable()
    
    # test_mem_leaks()
    # dat = matching_alphas_setup()[0]
    # dat = matching_betas_setup()[0]

    # with cre_context("test_matching_benchmarks") as ctxt:
    #     # c.get_matches(ms)

    #     ms = ms_w_n_boops(80, BOOP)
    #     print(rtsys.get_allocation_stats())
    #     ms = None
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
    # test_matching_unconditioned()
    # test_list()
    # test_multiple_types()
    # import pytest.__main__.benchmark
    # matching_1_t_4_lit_setup()
    # _test_NOT()
    # test_b_matching_1_t_4_lit()
    # test_multiple_types()
    test_same_parents()
    # test_mem_leaks()

