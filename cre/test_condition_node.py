from cre.condition_node import *
from cre.kb import KnowledgeBase
from cre.context import kb_context
from time import time_ns
from cre.utils import  _pointer_from_struct

BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

def test_aliasing():
    pass

@njit(cache=True)
def first_alpha(c):
    return c.dnf[0][0][0].is_alpha

@njit(cache=True)
def first_beta(c):
    return c.dnf[0][1][0].is_alpha 

def test_literal():
    with kb_context("test_literal"):
        # BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})
        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        c1 = l1.B < 1
        print(first_alpha(c1))
        # print(c1.dnf[0][0][0].is_alpha)
        c2 = l1.B < l2.B
        print(first_beta(c2))
        # print(c1.dnf[0][1][0].is_alpha)




# @njit(cache=True)
def test_build_conditions():
    with kb_context("test_build_conditions"):
        # BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")


        c1 = l1.B < 1
        c2 = l1.B < l2.B

        assert str(c1) == "(l1.B < ?)"
        assert str(c2) == "(l1.B < l2.B)"

        ### LT + AND/OR ###

        c3 = (l1.B < 1) & (l1.B > 7) & (l1.B < r1.B) & (l1.B < r2.B) |\
             (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (l2.B < r2.B)

        c3_str = \
'''(l1.B < ?) & (l1.B > ?) & (l1.B < r1.B) & (l1.B < r2.B) |\\
(l2.B < ?) & (l2.B > ?) & (l2.B < r1.B) & (l2.B < r2.B)'''

        assert str(c3) == c3_str

        ### NOT ###

        nc3_str = \
'''~(l1.B < ?) & ~(l2.B < ?) |\\
~(l1.B < ?) & ~(l2.B > ?) |\\
~(l1.B < ?) & ~(l2.B < r1.B) |\\
~(l1.B < ?) & ~(l2.B < r2.B) |\\
~(l1.B > ?) & ~(l2.B < ?) |\\
~(l1.B > ?) & ~(l2.B > ?) |\\
~(l1.B > ?) & ~(l2.B < r1.B) |\\
~(l1.B > ?) & ~(l2.B < r2.B) |\\
~(l2.B < ?) & ~(l1.B < r1.B) |\\
~(l2.B > ?) & ~(l1.B < r1.B) |\\
~(l1.B < r1.B) & ~(l2.B < r1.B) |\\
~(l1.B < r1.B) & ~(l2.B < r2.B) |\\
~(l2.B < ?) & ~(l1.B < r2.B) |\\
~(l2.B > ?) & ~(l1.B < r2.B) |\\
~(l1.B < r2.B) & ~(l2.B < r1.B) |\\
~(l1.B < r2.B) & ~(l2.B < r2.B)'''
        assert str(~c3) == nc3_str

        ### EQ / NEQ ###

        c4 = (l1.B == 5) & (l1.B == 5) & (l1.B == l2.B) & (l1.B != l2.B)

        c4_str = \
'''(l1.B == ?) & (l1.B == ?) & (l1.B == l2.B) & ~(l1.B == l2.B)'''
        assert str(c4) == c4_str    

        nc4_str = \
'''~(l1.B == ?) |\\
~(l1.B == ?) |\\
~(l1.B == l2.B) |\\
(l1.B == l2.B)'''
        assert str(~c4) == nc4_str

        ### AND / OR btw DNFS ### 

        c3_and_c4_str = \
'''(l1.B < ?) & (l1.B > ?) & (l1.B == ?) & (l1.B == ?) & (l1.B < r1.B) & (l1.B < r2.B) & (l1.B == l2.B) & ~(l1.B == l2.B) |\\
(l2.B < ?) & (l2.B > ?) & (l1.B == ?) & (l1.B == ?) & (l2.B < r1.B) & (l2.B < r2.B) & (l1.B == l2.B) & ~(l1.B == l2.B)'''
        assert str(c3 & c4) == c3_and_c4_str
        c3_or_c4_str = \
'''(l1.B < ?) & (l1.B > ?) & (l1.B < r1.B) & (l1.B < r2.B) |\\
(l2.B < ?) & (l2.B > ?) & (l2.B < r1.B) & (l2.B < r2.B) |\\
(l1.B == ?) & (l1.B == ?) & (l1.B == l2.B) & ~(l1.B == l2.B)'''
        assert str(c3 | c4) == c3_or_c4_str

    # l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")


list_i8 = ListType(i8)
list_list_i8 = ListType(ListType(i8))

@njit(cache=True)
def get_init_cond_sizes(conds):
    alpha_sizes = List.empty_list(list_i8)
    beta_sizes = List.empty_list(list_i8)
    for alpha_conjucts, beta_conjucts, beta_inds in conds.distr_dnf:
        alpha_sizes.append(List([len(conjunct) for conjunct in alpha_conjucts]))
        beta_sizes.append(List([len(conjunct) for conjunct in beta_conjucts]))

    return alpha_sizes, beta_sizes


@njit(cache=True)
def var_get_ptr(var):
    return _pointer_from_struct(var)

@njit(cache=True)
def cond_get_vars(cond):
    return cond.vars

def test_initialize():
    with kb_context("test_initialize"):
        # BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

        c = (l1.B < 1) & (l1.B > 7) & (l2.B < r1.B) & (r2.B < l1.B) |\
             (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (r1.B < r2.B) |\
             (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (l1.B < l2.B)

        assert [x.alias for x in cond_get_vars(c)] == ['l1','l2','r1','r2']

        initialize_conditions(c)
        print("DONE")
        alpha_sizes, beta_sizes = get_init_cond_sizes(c)

        
        print(alpha_sizes)
        print(beta_sizes)
        assert [list(x) for x in alpha_sizes] == [[2, 0, 0, 0], [0, 2, 0, 0], [0, 2, 0, 0]]
        assert [list(x) for x in beta_sizes] == [[1, 1], [1, 1], [1, 1]]


@njit(cache=True)
def get_pointer(st):
    return _pointer_from_struct(st)

def test_link():
    with kb_context() as context:
        print(context.fact_types)
        # BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})
        
        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

        c = (l1.B < 1) & (l1.B > 7) & (l2.B < r1.B) & (r2.B < l1.B) |\
             (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (r1.B < r2.B) |\
             (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (l1.B < l2.B)


        kb = KnowledgeBase()
        cl = get_linked_conditions_instance(c, kb)

        assert get_pointer(cl) == get_pointer(c)

        cl = get_linked_conditions_instance(c, kb, copy=True)

        assert get_pointer(cl) != get_pointer(c)

        

def test_unconditioned():
    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")

    # c = var_and(l1,l2)

    print(conditions_repr(l1 & l2,"c"))
    print(conditions_repr((l1.B < 1) & l2, "c"))
    print(conditions_repr(l1 & (l2.B > 1), "c"))
    
def test_multiple_deref():
    with kb_context("test_ref_matching"):
        TestLL, TestLLType = define_fact("TestLL",{"name": "string", "B" :'number', "nxt" : "TestLL"})

        v1 = Var(TestLL,'v1')
        v2 = Var(TestLL,'v2')
        c = v1.nxt.nxt == v2.nxt.nxt
        print(c)

def test_existential_not():
    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
    # print(l1.B)
    # print(~l1)
    # print(NOT(l1).B)
    # print(NOT(l1.B))
    a = (l1.B < 1)
    # print(repr(a))
    c = a & (l2.B > 1)
    c_n = NOT(c)
    # print("c.vars",c.vars)
    # print("c_n.vars", c_n.vars)
    # print(repr(c))
    # print(repr(c_n))
    assert repr(c) == 'l1, l2 = Var(BOOP), Var(BOOP)\n(l1.B < ?) & (l2.B > ?)'
    assert repr(c_n) == 'l1, l2 = NOT(BOOP), NOT(BOOP)\n(l1.B < ?) & (l2.B > ?)'

    c2 = NOT(l1.B < l2.B)
    assert repr(c2) == 'l1, l2 = NOT(BOOP), NOT(BOOP)\n(l1.B < l2.B)'
    # print(repr(c2))


if(__name__ == "__main__"):
    # test_link()
    # test_initialize()
    # for i in range(10):
    #     t0 = time_ns()
    #     test_build_conditions()
    #     print(f'{(time_ns()-t0)/1e6} ms')
    for i in range(10):
        t0 = time_ns()
        test_unconditioned()
        print(f'{(time_ns()-t0)/1e6} ms')
    # test_multiple_deref()
    # test_existential_not()
# # bar.py_func()
    # bar()

    
    # exit()
