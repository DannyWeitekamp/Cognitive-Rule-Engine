from numba import f8
from cre.conditions import *
from cre.memset import MemSet
from cre.context import cre_context
from cre.cre_object import CREObjType
from time import time_ns
from cre.utils import  _raw_ptr_from_struct, _cast_structref

import  cre.dynamic_exec

BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})

def test_aliasing():
    pass

# @njit(cache=True)
# def first_alpha(c):
#     return c.dnf[0][0][0].is_alpha

# @njit(cache=True)
# def first_beta(c):
#     return c.dnf[0][1][0].is_alpha 

# def test_literal():
#     with cre_context("test_literal"):
#         # BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})
#         l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
#         c1 = l1.B < 1
#         print(first_alpha(c1))
#         # print(c1.dnf[0][0][0].is_alpha)
#         c2 = l1.B < l2.B
#         print(first_beta(c2))
#         # print(c1.dnf[0][1][0].is_alpha)




# @njit(cache=True)
def test_build_conditions():
    with cre_context("test_build_conditions"):
        # BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})

        l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
        r1, r2 = Var(BOOP,"r1"), Var(BOOP,"r2")


        c1 = l1.B < 1
        c2 = l1.B < l2.B

        assert str(c1) == "(l1.B < 1)"
        assert str(c2) == "(l1.B < l2.B)"

        ### LT + AND/OR ###

        c3 = (l1.B < 1) & (l1.B > 7) & (l1.B < r1.B) & (l1.B < r2.B) |\
             (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (l2.B < r2.B)

        c3_str = \
'''(l1.B < 1) & (l1.B > 7) & (l1.B < r1.B) & (l1.B < r2.B) |\\
(l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (l2.B < r2.B)'''

        assert str(c3) == c3_str

        ### NOT ###
        # print(str(~c3))

        nc3_str = \
'''~(l1.B < 1) & ~(l2.B < 1) |\\
~(l1.B < 1) & ~(l2.B > 7) |\\
~(l1.B < 1) & ~(l2.B < r1.B) |\\
~(l1.B < 1) & ~(l2.B < r2.B) |\\
~(l1.B > 7) & ~(l2.B < 1) |\\
~(l1.B > 7) & ~(l2.B > 7) |\\
~(l1.B > 7) & ~(l2.B < r1.B) |\\
~(l1.B > 7) & ~(l2.B < r2.B) |\\
~(l1.B < r1.B) & ~(l2.B < 1) |\\
~(l1.B < r1.B) & ~(l2.B > 7) |\\
~(l1.B < r1.B) & ~(l2.B < r1.B) |\\
~(l1.B < r1.B) & ~(l2.B < r2.B) |\\
~(l1.B < r2.B) & ~(l2.B < 1) |\\
~(l1.B < r2.B) & ~(l2.B > 7) |\\
~(l1.B < r2.B) & ~(l2.B < r1.B) |\\
~(l1.B < r2.B) & ~(l2.B < r2.B)'''
        # assert str(~c3) == nc3_str

        ### EQ / NEQ ###

        c4 = (l1.B == 5) & (l1.B == 5) & (l1.B == l2.B) & (l1.B != l2.B)

        c4_str = \
'''(l1.B == 5) & (l1.B == 5) & (l1.B == l2.B) & ~(l1.B == l2.B)'''
        assert str(c4) == c4_str    

        nc4_str = \
'''~(l1.B == 5) |\\
~(l1.B == 5) |\\
~(l1.B == l2.B) |\\
(l1.B == l2.B)'''
        assert str(~c4) == nc4_str

        ### AND / OR btw DNFS ### 
        c3_and_c4_str = \
'''(l1.B < 1) & (l1.B > 7) & (l1.B < r1.B) & (l1.B < r2.B) & (l1.B == 5) & (l1.B == 5) & (l1.B == l2.B) & ~(l1.B == l2.B) |\\
(l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (l2.B < r2.B) & (l1.B == 5) & (l1.B == 5) & (l1.B == l2.B) & ~(l1.B == l2.B)'''
        assert str(c3 & c4) == c3_and_c4_str

        c3_or_c4_str = \
'''(l1.B < 1) & (l1.B > 7) & (l1.B < r1.B) & (l1.B < r2.B) |\\
(l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (l2.B < r2.B) |\\
(l1.B == 5) & (l1.B == 5) & (l1.B == l2.B) & ~(l1.B == l2.B)'''
        assert str(c3 | c4) == c3_or_c4_str

    # l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")


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
    return _raw_ptr_from_struct(var)

@njit(cache=True)
def cond_get_vars(cond):
    return cond.vars

# def test_initialize():
#     with cre_context("test_initialize"):
#         # BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})

#         l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
#         r1, r2 = Var(BOOP,"r1"), Var(BOOP,"r2")

#         c = (l1.B < 1) & (l1.B > 7) & (l2.B < r1.B) & (r2.B < l1.B)# |\
#              # (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (r1.B < r2.B) |\
#              # (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (l1.B < l2.B)

#         assert [x.alias for x in cond_get_vars(c)] == ['l1','l2','r1','r2']

#         initialize_conditions(c)
#         print("DONE")
#         alpha_sizes, beta_sizes = get_init_cond_sizes(c)

        
#         print(alpha_sizes)
#         print(beta_sizes)
#         assert [list(x) for x in alpha_sizes] == [[2, 0, 0, 0], [0, 2, 0, 0], [0, 2, 0, 0]]
#         assert [list(x) for x in beta_sizes] == [[1, 1], [1, 1], [1, 1]]


@njit(cache=True)
def get_pointer(st):
    return _raw_ptr_from_struct(st)

def _test_link():
    '''TODO: REWRITE'''
    print("START TEST LINK")
    with cre_context() as context:
        # BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})
        
        l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
        r1, r2 = Var(BOOP,"r1"), Var(BOOP,"r2")

        c = (l1.B < 1) & (l1.B > 7) & (l2.B < r1.B) & (r2.B < l1.B) |\
             (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (r1.B < r2.B) |\
             (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (l1.B < l2.B)

        print(c)
        ms = MemSet()
        cl = get_linked_conditions_instance(c, ms)

        assert get_pointer(cl) == get_pointer(c)

        cl = get_linked_conditions_instance(c, ms, copy=True)

        assert get_pointer(cl) != get_pointer(c)

        

def test_unconditioned():
    l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")

    # c = var_and(l1,l2)

    print(conditions_repr(l1 & l2,"c"))
    print(conditions_repr((l1.B < 1) & l2, "c"))
    print(conditions_repr(l1 & (l2.B > 1), "c"))
    
def test_multiple_deref():
    with cre_context("test_ref_matching"):
        TestLL = define_fact("TestLL",{"name": "string", "B" :'number', "nxt" : "TestLL"})

        v1 = Var(TestLL,'v1')
        v2 = Var(TestLL,'v2')
        c = v1.nxt.nxt == v2.nxt.nxt
        print(c)

def _test_existential_not():
    l1, l2 = Var(BOOP,"l1"), Var(BOOP,"l2")
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
    print(repr(c))
    print(repr(c_n))
    assert repr(c) == 'l1, l2 = Var(BOOP), Var(BOOP)\n(l1.B < ?) & (l2.B > ?)'
    assert repr(c_n) == 'l1, l2 = NOT(BOOP), NOT(BOOP)\n(l1.B < ?) & (l2.B > ?)'

    c2 = NOT(l1.B < l2.B)
    assert repr(c2) == 'l1, l2 = NOT(BOOP), NOT(BOOP)\n(l1.B < l2.B)'
    # print(repr(c2))

def test_list_operations():
    with cre_context("test_list_operations"):
        TList = define_fact("TList",{"name" : "string", "items" : "ListType(string)"})
        v = Var(TList,"v")
        print(v.items[0])
        print(v.items[0] != v.items[1])
        assert str(v.items[0]) == "Var(TList,'v').items[0]"

        c = v.items[0] != v.items[1]
        assert str(c) == "~(v.items[0] == v.items[1])"



@njit(cache=True)
def hsh(x):
    return hash(x)


def test_hash():
    ''' Tests hashing for Var, Literal, and Conditions '''
    with cre_context("test_hash"):
        TestLL = define_fact("TestLL",{"name": "string", "B" :'number', "nxt" : "TestLL"})

        ### VAR ### 
        a1 = Var(TestLL)
        a2 = a1
        b1 = Var(TestLL)

        assert hsh(a1) == hsh(a2)
        assert hsh(a1) != hsh(b1)

        a1 = Var(TestLL)
        b1 = Var(TestLL).nxt
        b2 = Var(TestLL).nxt.nxt.B

        assert hsh(a1) != hsh(b1)
        assert hsh(a1) != hsh(b2)

        ### LITERAL ### 
        x, y, z = Var(f8,'x'), Var(f8,'y'), Var(f8,'z')
        x2, y2, z2 = Var(f8,'x'), Var(f8,'y'), Var(f8,'z')

        a1 = literal_ctor((x + z) + (y + z))
        a2 = literal_ctor((x + z) + (y + z))
        b1 = literal_not(literal_ctor((x + z) + (y + z)))
        b2 = literal_ctor((x + z) + (y + x))
        b3 = literal_ctor((x2 + z2) + (y2 + z2))

        assert hsh(a1) == hsh(a2)
        assert hsh(a1) != hsh(b1)
        assert hsh(a1) != hsh(b2)
        assert hsh(a1) != hsh(b3)

        ### CONDITIONS ### 
        X,Y = Var(TestLL,"X"), Var(TestLL,"Y")
        a1 = ( (X.B == 0) &
               ((X.nxt.B == Y.B) | (X.nxt.name == Y.name)) )
        a2 = ( (X.B == 0) &
               ((X.nxt.B == Y.B) | (X.nxt.name == Y.name)) )

        b1 = ( (A:=Var(TestLL)) & (A.B == 0) &
               (B:=Var(TestLL)) & ((A.nxt.B == B.B) | (A.nxt.name == B.name)) )
        b2 = ( (X.B < 0) &
               ((X.nxt.B == Y.B) | (X.nxt.name == Y.name)) )
        b3 = ( (X.B == 0) &
               ((X.nxt.B == Y.B)) )

        print(hsh(a1), hsh(a2), hsh(b1), hsh(b2), hsh(b3))

        assert hsh(a1) == hsh(a2)
        assert hsh(a1) != hsh(b1)
        assert hsh(a1) != hsh(b2)
        assert hsh(a1) != hsh(b3)




        


@njit(cache=True)
def eq(a,b):
    return _cast_structref(CREObjType, a)==_cast_structref(CREObjType, b)

def test_eq():
        TestLL = define_fact("TestLL",{"name": "string", "B" :'number', "nxt" : "TestLL"})

        ### VAR ### 
        a1 = Var(TestLL)
        a2 = a1
        b1 = Var(TestLL)

        assert eq(a1,a2)
        assert not eq(a1,b1)

        a1 = Var(TestLL)
        b1 = Var(TestLL).nxt
        b2 = Var(TestLL).nxt.nxt.B

        assert not eq(a1,b1)
        assert not eq(a1,b2)

        ### LITERAL ### 
        x, y, z = Var(f8,'x'), Var(f8,'y'), Var(f8,'z')
        x2, y2, z2 = Var(f8,'x'), Var(f8,'y'), Var(f8,'z')

        a1 = literal_ctor((x + z) + (y + z))
        a2 = literal_ctor((x + z) + (y + z))
        b1 = literal_not(literal_ctor((x + z) + (y + z)))
        b2 = literal_ctor((x + z) + (y + x))
        b3 = literal_ctor((x2 + z2) + (y2 + z2))

        assert eq(a1,a2)
        assert not eq(a1,b1)
        assert not eq(a1,b2)
        assert not eq(a1,b3)

        ### CONDITIONS ### 
        X,Y = Var(TestLL,"X"), Var(TestLL,"Y")
        a1 = ( (X.B == 0) &
               ((X.nxt.B == Y.B) | (X.nxt.name == Y.name)) )
        a2 = ( (X.B == 0) &
               ((X.nxt.B == Y.B) | (X.nxt.name == Y.name)) )

        b1 = ( (A:=Var(TestLL)) & (A.B == 0) &
               (B:=Var(TestLL)) & ((A.nxt.B == B.B) | (A.nxt.name == B.name)) )
        b2 = ( (X.B < 0) &
               ((X.nxt.B == Y.B) | (X.nxt.name == Y.name)) )
        b3 = ( (X.B == 0) &
               ((X.nxt.B == Y.B)) )

        assert eq(a1,a2)
        assert not eq(a1,b1)
        assert not eq(a1,b2)
        assert not eq(a1,b3)

from cre.conditions import conds_to_lit_sets, best_intersection, make_base_ptrs_to_inds, score_remaps
def test_anti_unify():
    x, y, z = Var(f8,'x'), Var(f8,'y'), Var(f8,'z')
    a, b, c, d = Var(f8,'a'), Var(f8,'b'), Var(f8,'c'), Var(f8,'d')

    c1 = (x < y) & (y < z) & (y < z) & (z != x) & (y != 0) 
    c2 = (a < b) & (b < c) & (b < c) & (b < c) & (c != a) & (b != 0) & (d != 0)


    conds_antiunify(c1,c2)


    c1 = ((x < y) & (z != x) & (y != 0) |
          (x < y) & (z == x) & (y != 7) | 
          (x > y) & (z != x) & (y != 2)
         )

    c2 = ((a < b) & (c == a) & (b != 7) & (d > 0) |
          (a < b) & (c != a) & (b != 0) |
          (a > b) & (c != a) & (b != 0) & (d != 7)
         )

    conds_antiunify(c1,c2)

    
    # score_remaps()

    # For ? < ?
    # x -> [1,1,0,0]
    # y -> [1,1,1,0]
    # z -> [0,1,1,0]
    # test_frzn_ind_arr_type()
    # raise ValueError()

    # lsl1 = conds_to_lit_sets(c1)
    # lsl2 = conds_to_lit_sets(c2)

    # # print(lsl1[0].keys())
    # bpti1 = make_base_ptrs_to_inds(c1)
    # bpti2 = make_base_ptrs_to_inds(c2)
    # score_remaps(lsl1[0], lsl2[0], bpti1, bpti2)
    # score_remaps(lsl2[0], lsl1[0], bpti2, bpti1)





    



# import logging
# print([name for name in logging.root.manager.loggerDict])
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

if(__name__ == "__main__"):
    # test_anti_unify()
    # test_unconditioned()
    # test_build_conditions()
    test_list_operations()
    # test_link()
    # test_initialize()
    # for i in range(10):
    #     t0 = time_ns()
    #     test_build_conditions()
    #     print(f'{(time_ns()-t0)/1e6} ms')
    # for i in range(10):
    #     t0 = time_ns()
    #     test_unconditioned()
    #     print(f'{(time_ns()-t0)/1e6} ms')
    # test_multiple_deref()
    # test_existential_not()
# bar.py_func()
    # bar()
    # test_hash()

    # test_eq()
    # 
    # exit()

