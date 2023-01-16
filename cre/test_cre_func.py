from numba import generated_jit, njit, i8, f8
from numba.types import unicode_type, FunctionType
from cre.cre_func import CREFunc
from cre.cre_object import _get_chr_mbrs_infos_from_attrs, _iter_mbr_infos
from cre.fact import define_fact
from cre.var import Var
from cre.utils import PrintElapse

from numba.core.runtime.nrt import rtsys
def used_bytes():
    stats = rtsys.get_allocation_stats()
    return stats.alloc-stats.free


def test_numerical():
    @CREFunc(signature=i8(i8,i8,i8,i8))
    def Add(a, b, c, d):
        return a + b + c +d

    a = Var(i8,'a')
    b = Var(i8,'b')
    c = Var(i8,'c')

    z = Add(a,b,c,c)
    # print("<<", str(z))
    assert z(1,2,3) == 9
    assert str(z) == "Add(a, b, c, c)"
    q = z(c,c,c)
    # print("<<", q)
    assert(q(7)==28)
    assert str(q) == "Add(c, c, c, c)"

    assert Add(a,Add(a,b,c,a),c,c)(1,2,3) == 14
    assert Add(Add(Add(2,1,1,a),2,1,Add(2,1,1,b)),1,1,c)(1,2,3) == 19
    z = Add(Add(Add(2,1,1,a),2,1,Add(2,1,1,b)),1,1,c)
    assert str(z) == "Add((Add((Add(2, 1, 1, a)), 2, 1, (Add(2, 1, 1, b)))), 1, 1, c)"

    @CREFunc(signature=f8(f8,f8),
            shorthand='{0}+{1}')
    def Add(a, b):
        return a + b

    assert Add(9.0, 7.0) == 16.0

    Incr = Add(Var(f8,'a'),1.0)
    assert Incr(3.0) == 4.0
    assert str(Incr) == "a+1.0"



def test_string():
    @CREFunc(signature=unicode_type(unicode_type,unicode_type),
            shorthand='{0}+{1}')
    def Concat(a, b):
        return a + b
    
    a = Var(unicode_type,'a')
    b = Var(unicode_type,'b')
    c = Var(unicode_type,'c')

    for i in range(2):
        z = Concat("|", Concat(Concat(b,c),"|" ))

        assert z("X","Y") == "|XY|"
        assert str(z) == "'|'+((b+c)+'|')"

        if(i == 0):
            init_bytes = used_bytes()
        else:
            assert used_bytes() == init_bytes


def test_obj():
    print("--START OBJ--")
    BOOP = define_fact("BOOP", {"A" :unicode_type, "B" :i8})

    @CREFunc(signature=unicode_type(unicode_type,unicode_type),
            shorthand='{0}+{1}')
    def Concat(a, b):
        return a + b

    @CREFunc(signature=BOOP(BOOP,BOOP),
            shorthand='{0}+{1}')
    def CatBOOPs(a, b):
        return BOOP(a.A + b.A, a.B + b.B)

    a = Var(BOOP,'a')
    b = Var(BOOP,'b')
    c = Var(BOOP,'c')
    
    for i in range(5):
        ba, bb = BOOP("A",1), BOOP("B",2)

        z = Concat(a.A, b.A)

        assert z(ba,bb) == "AB"
        assert str(z) == "a.A+b.A"

        zboop = CatBOOPs(a,b)

        assert zboop(ba,bb) == BOOP(A="AB", B=3)
        assert str(zboop) == "a+b"

        z = None
        zboop = None
        if(i == 0):
            init_bytes = used_bytes()
        else:
            assert used_bytes() == init_bytes
            # print(used_bytes(), init_bytes)



if __name__ == "__main__":
    import faulthandler; faulthandler.enable()

    test_numerical()
    test_string()
    test_obj()


    @CREFunc(signature=i8(i8,i8))
    def Add(a, b):
        return a + b

    from cre.cre_func import (set_base_arg_val_impl, _func_from_address,
        cre_func_call_self, get_str_return_val_impl, call_self_f_type)


    N = 100

    @njit(cache=True)
    def baz(op):
        z = 0
        for i in range(N):
            for j in range(N):
                z += op(i, j)
        return z

    a = Var(i8,'a')
    b = Var(i8,'b')
    comp = Add(Add(0,b),a)

    print(Add._type.name)
    print(comp._type.name)

    baz(Add)
    baz(comp)

    with PrintElapse("baz"):
        print(baz(Add))

    with PrintElapse("baz comp"):
        print(baz(comp))

    # _func_from_address(call_self_f_type, self.call_self_addr)(self)

    i8_ft = FunctionType(i8(i8,i8))

    set_base = set_base_arg_val_impl(0)
    ret_impl = get_str_return_val_impl(i8)


    @njit(cache=True)
    def foo(op):
        z = 0
        for i in range(N):
            for j in range(N):
                set_base(op, 0, i)
                set_base(op, 1, j)
                cre_func_call_self(op)
                z += ret_impl(op)
        return z

    @njit(cache=True)
    def foo_fast(op):
        z = 0
        # f = _func_from_address(i8_ft, op.call_heads_addr)
        for i in range(N):
            for j in range(N):
                f = _func_from_address(i8_ft,op.call_heads_addr)
                z += f(i,j)
                
        return z

    @njit(i8(i8,i8), cache=True)
    def njit_add(a,b):
        return a + b

    @njit(i8(FunctionType(i8(i8,i8))), cache=True)
    def bar(op):
        z = 0
        for i in range(N):
            for j in range(N):
                z += op(i, j)
        return z

    @njit(i8(), cache=True)
    def inline():
        z = 0
        for i in range(N):
            for j in range(N):
                z += njit_add(i, j)
        return z

    foo(Add)
    foo_fast(Add)
    bar(njit_add)
    inline()

    with PrintElapse("foo"):
        foo(Add)

    with PrintElapse("foo fast"):
        foo_fast(Add)

    with PrintElapse("foo python"):
        foo.py_func(Add)
        

    with PrintElapse("bar"):
        bar(njit_add)

    with PrintElapse("bar  python"):
        bar.py_func(njit_add)


    with PrintElapse("inline"):
        inline()




    








