from numbert.experimental.condition_node import *
from time import time_ns
from numbert.experimental.utils import  _pointer_from_struct

BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

def test_aliasing():
    pass

@njit(cache=True)
def first_alpha(c):
    return c.dnf[0][0][0].is_alpha

@njit(cache=True)
def first_beta(c):
    return c.dnf[0][1][0].is_alpha 

def test_term():
    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
    c1 = l1.B < 1
    print(first_alpha(c1))
    # print(c1.dnf[0][0][0].is_alpha)
    c2 = l1.B < l2.B
    print(first_beta(c2))
    # print(c1.dnf[0][1][0].is_alpha)




# @njit(cache=True)
def test_build_conditions():
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
~(l1.B < r1.B) & ~(l2.B < ?) |\\
~(l1.B < r1.B) & ~(l2.B > ?) |\\
~(l1.B < r1.B) & ~(l2.B < r1.B) |\\
~(l1.B < r1.B) & ~(l2.B < r2.B) |\\
~(l1.B < r2.B) & ~(l2.B < ?) |\\
~(l1.B < r2.B) & ~(l2.B > ?) |\\
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
'''(l1.B < ?) & (l1.B > ?) & (l1.B < r1.B) & (l1.B < r2.B) & (l1.B == ?) & (l1.B == ?) & (l1.B == l2.B) & ~(l1.B == l2.B) |\\
(l2.B < ?) & (l2.B > ?) & (l2.B < r1.B) & (l2.B < r2.B) & (l1.B == ?) & (l1.B == ?) & (l1.B == l2.B) & ~(l1.B == l2.B)'''
    
    assert str(c3 & c4) == c3_and_c4_str
    c3_or_c4_str = \
'''(l1.B < ?) & (l1.B > ?) & (l1.B < r1.B) & (l1.B < r2.B) |\\
(l2.B < ?) & (l2.B > ?) & (l2.B < r1.B) & (l2.B < r2.B) |\\
(l1.B == ?) & (l1.B == ?) & (l1.B == l2.B) & ~(l1.B == l2.B)'''
    assert str(c3 | c4) == c3_or_c4_str

def test_multiple_deref():
    pass
    # l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")


list_i8 = ListType(i8)
list_list_i8 = ListType(ListType(i8))

@njit(cache=True)
def get_init_cond_sizes(cond):
    alpha_sizes = List.empty_list(list_i8)
    beta_sizes = List.empty_list(list_i8)
    for dnf in cond.alpha_dnfs:
        alpha_sizes.append(List([len(conjunct) for conjunct in dnf]))

    for dnf in cond.beta_dnfs:
        beta_sizes.append(List([len(conjunct) for conjunct in dnf]))

    return alpha_sizes, beta_sizes


@njit(cache=True)
def var_get_ptr(var):
    return _pointer_from_struct(var)

@njit(cache=True)
def cond_get_vars(cond):
    return cond.vars

def test_initialize():
    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
    r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

    c = (l1.B < 1) & (l1.B > 7) & (l2.B < r1.B) & (r2.B < l1.B) |\
         (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (r1.B < r2.B) |\
         (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (l1.B < l2.B)

    assert [x.alias for x in cond_get_vars(c)] == ['l1','l2','r1','r2']

    initialize_condition(c)

    alpha_sizes, beta_sizes = get_init_cond_sizes(c)

    

    assert [list(x) for x in alpha_sizes] == [[2], [2,2], [], []]
    assert [list(x) for x in beta_sizes] == [[1], [1,1,1], [1], [1]]
    

if(__name__ == "__main__"):
    test_initialize()
    # test_term()
    for i in range(10):
        t0 = time_ns()
        test_build_conditions()
        print(f'{(time_ns()-t0)/1e6} ms')
# # bar.py_func()
    # bar()

    
    # exit()