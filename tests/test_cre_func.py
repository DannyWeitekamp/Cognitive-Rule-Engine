from numba import generated_jit, njit, i8, f8
from numba.types import unicode_type, FunctionType
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.func import CREFunc, set_func_arg, set_var_arg, reinitialize, CREFuncTypeClass, cre_func_copy
from cre.obj import _get_chr_mbrs_infos_from_attrs, _iter_mbr_infos
from cre.fact import define_fact
from cre.var import Var
from cre.utils import PrintElapse, _func_from_address, _cast_structref, NRTStatsEnabled, used_bytes
from cre.context import cre_context
import cre.type_conv
import pytest




def test_numerical():
    with NRTStatsEnabled:
        @CREFunc(signature=i8(i8,i8,i8,i8))
        def Add(a, b, c, d):
            return a + b + c +d

        a = Var(i8,'a')
        b = Var(i8,'b')
        c = Var(i8,'c')

        z = Add(a,b,c,c)
        assert z.depth == 1

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
        assert str(z) == "Add(Add(Add(2, 1, 1, a), 2, 1, Add(2, 1, 1, b)), 1, 1, c)"
        assert z.depth == 3

        @CREFunc(signature=f8(f8,f8),
                shorthand='{0}+{1}')
        def Add(a, b):
            return a + b

        assert Add(9.0, 7.0) == 16.0

        for i in range(2):
            a = Var(f8,'a')
            Incr = Add(Var(f8,'a'),1.0)
            assert Incr(3.0) == 4.0
            assert str(Incr) == "a+1"

            if(i == 0):
                init_bytes = used_bytes()
            else:
                assert used_bytes() == init_bytes



def test_string():
    with NRTStatsEnabled:
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
            assert z.depth == 3

            if(i == 0):
                init_bytes = used_bytes()
            else:
                assert used_bytes() == init_bytes



def test_obj():
    with NRTStatsEnabled:
        with cre_context("test_obj"):
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
            
            for i in range(2):
                ba, bb = BOOP("A",1), BOOP("B",2)

                z = Concat(a.A, a.A)
                print(z(ba), str(z))

                assert z(ba) == "AA"
                assert str(z) == "a.A+a.A"

                z = Concat(a.A, b.A)

                assert z(ba,bb) == "AB"
                assert str(z) == "a.A+b.A"

                z = Concat(Concat("|", Concat(a.A, b.A)),"|")
                assert z(ba,bb) == "|AB|"
                

                zboop = CatBOOPs(a,b)

                assert zboop(ba,bb) == BOOP(A="AB", B=3)
                assert str(zboop) == "a+b"

                # Make sure "head_var_ptrs" actually grabs vars w/ derefs
                assert not any(z.base_var_ptrs == z.head_var_ptrs)

                z = None
                zboop = None
                if(i == 0):
                    init_bytes = used_bytes()
                else:
                    assert used_bytes() == init_bytes
                    # print(used_bytes(), init_bytes)

def test_mixed_types():
    from cre.default_funcs import Identity
    @CREFunc(signature=i8(i8,i8),
            shorthand='{0}+{1}')
    def Add(a, b):
        return a + b

    @CREFunc(signature=unicode_type(unicode_type,unicode_type),
                shorthand='{0}+{1}')
    def Concat(a, b):
        return a + b

    @CREFunc(signature=unicode_type(i8),
            shorthand='str({0})')
    def ToStr(a):
        return str(a)

    @CREFunc(signature=i8(unicode_type),
            shorthand='int({0})')
    def ToInt(a):
        return float(a)

    a, b, c = Var(i8,'a'), Var(i8,'b'), Var(i8,'c')

    z = Concat(Concat(ToStr(a),ToStr(b)),ToStr(c))
    assert z(1,2,3) == "123"
    # print("ret_val:", ret_val)
    q = ToInt(z)
    assert q(1,2,3) == 123    

    z = ToStr(Add(Add(a,b),c))
    assert z(1,2,3) == "6"
    




new_f_type = CREFuncTypeClass(f8,(f8,f8),is_composed=True,name="Composed")
@njit(cache=True)
def compose_hardcoded(f, g):
    _f = cre_func_copy(f)
    g1 = cre_func_copy(g)
    g2 = cre_func_copy(g)
    set_var_arg(g1, 0, Var(f8,"x"))
    reinitialize(g1)
    set_var_arg(g2, 0, Var(f8,"y"))
    reinitialize(g2)
    set_func_arg(_f, 0, g1)
    set_func_arg(_f, 1, g2)
    reinitialize(_f)
    return _cast_structref(new_f_type, _f)

@njit(cache=True)
def compose_overloaded(f, g):
    
    x,y = Var(f8,"x"), Var(f8,"y")
    h = f(g(x),g(y))
    return h


def test_njit_compose():    
    @CREFunc(signature=f8(f8,f8), shorthand="{0}*{1}")
    def Multiply(a, b):
        return a * b

    @CREFunc(signature=f8(f8), shorthand="{0}+1")
    def Increment(a):
        return a + 1

    f = compose_hardcoded(Multiply, Increment)

    assert f(1,2) == 6
    assert str(f) == "(x+1)*(y+1)"

    f = compose_overloaded(Multiply, Increment)

    assert f(1,2) == 6
    assert str(f) == "(x+1)*(y+1)"

from cre.func import cre_func_deep_copy_generic
def test_no_mutate_on_compose():
    a,b,c = Var(f8,'a'), Var(f8,'b'), Var(f8,'c')
    c0 = a + b + b
    c0_ = cre_func_deep_copy_generic(c0)
    print("c0_", c0_)
    s0 = str(c0) 
    print(c0)
    c1 = c0(a,c+b)
    s1 = str(c0)
    print(c1)
    print(c0)
    assert str(c0) == s0

def test_compose_deref_bases():
    with cre_context("test_compose_deref_bases"):
        BOOP = define_fact("BOOP", {"A" :unicode_type, "B" : f8})

        @CREFunc(signature=f8(f8,f8), shorthand="{0}*{1}")
        def Multiply(a, b):
            return a * b

        a,b = Var(BOOP,'a'), Var(BOOP,'b')

        c0 = Multiply(a.B, Multiply(a.B, b.B))

        a0, a1 =BOOP("A",7),BOOP("B",2) 
        print(c0(a0,a1))
        assert c0(BOOP("A",7),BOOP("B",2)) == 98

        print(c0)

        x,y = Var(BOOP,'x'), Var(BOOP,'y')

        c1 = c0(x,y)

        assert c1(BOOP("A",7),BOOP("B",2)) == 98

        print(c1)




def test_commutes():
    @CREFunc(signature=f8(f8,f8,f8),
            shorthand="{0}*{1}*{2}", commutes=True)
    def Add3(a, b, c):
        return a + b + c

    print(Add3.commutes)
    assert Add3.commutes == [[0,1,2]]
    print(Add3.right_commutes)
    rc_str = str(Add3.right_commutes)
    assert "2: array([0, 1]" in rc_str
    assert "1: array([0]" in rc_str

    with pytest.raises(AssertionError):
        @CREFunc(signature=f8(f8,f8, unicode_type),commutes=[[0,1,2]])
        def Floop(a,b,c):
            return a + b

def test_not_jittable():
    from cre.default_funcs import Add

    with pytest.warns(NumbaPerformanceWarning):
        @CREFunc(signature=f8(f8))
        def Map(a):
            l = []
            d = {1.0: 10.0, 2.0: 20.0}
            for i in range(int(a)):
                l.append(str(i+1))
                l.append(float(i+1))

            return d[l[-1]]


    assert Map(1.0)==10.0
    # assert Map.call(1.0)==10.0
    # assert Map.check(1.0)==1


    op = Add(Map(Var(float,'x')),Map(Var(float,'y')))
    assert op(1,2)==30.0


def test_returns_object():
    with cre_context('test_returns_object'):
        spec = {"A" : "string", "B" : "number"}
        BOOP = define_fact("BOOP", spec)


        @CREFunc(signature=BOOP(BOOP))
        def RetBOOP(a):
            return BOOP(a.A, a.B+9.0)
            
        assert RetBOOP(BOOP("A",1.0)).B == 10.0

        with pytest.warns(NumbaPerformanceWarning):        
            @CREFunc(signature=BOOP(BOOP))
            def BOOPMap(a):
                l = []
                d = {1.0: 10.0, 2.0: 20.0}
                for i in range(int(a.B)):
                    l.append(str(i+1))
                    l.append(float(i+1))

                return BOOP("A", d[l[-1]])

        assert BOOPMap(BOOP("A",1.0)).B == 10.0

def test_constant():
    @CREFunc(signature=unicode_type(), shorthand="'X'")
    def X():
        return "X"
    assert str(X) == "'X'"
    assert X() == "X"


def test_var_cmp_overloads():
    x, y = Var(f8,'x'), Var(f8,'y')

    #--beta cmp--
    op = x < y
    assert str(op) == "x < y"
    op = x <= y
    assert str(op) == "x <= y"
    op = x > y
    assert str(op) == "x > y"
    op = x >= y
    assert str(op) == "x >= y"
    op = x == y
    assert str(op) == "x == y"

    #--left cmp--
    op = 1 < y
    assert str(op) == "y > 1"
    op = 1 <= y
    assert str(op) == "y >= 1"
    op = 1 > y
    assert str(op) == "y < 1"
    op = 1 >= y
    assert str(op) == "y <= 1"
    op = 1 == y
    assert str(op) == "y == 1"

    #--right cmp--
    op = x < 1
    assert str(op) == "x < 1"
    op = x <= 1
    assert str(op) == "x <= 1"
    op = x > 1
    assert str(op) == "x > 1"
    op = x >= 1
    assert str(op) == "x >= 1"
    op = x == 1
    assert str(op) == "x == 1"

def test_var_arith_overloads():
    x, y = Var(f8,'x'), Var(f8,'y')

    #--beta arith--
    op = x + y
    assert str(op) == "x + y"
    op = x - y
    assert str(op) == "x - y"
    op = x * y
    assert str(op) == "x * y"
    op = x / y
    assert str(op) == "x / y"
    op = x // y
    assert str(op) == "x // y"
    op = x ** y
    assert str(op) == "x ** y"

    #--left arith--
    op = 1 + y
    assert str(op) == "1 + y"
    op = 1 - y
    assert str(op) == "1 - y"
    op = 1 * y
    assert str(op) == "1 * y"
    op = 1 / y
    assert str(op) == "1 / y"
    op = 1 // y
    assert str(op) == "1 // y"
    op = 1 ** y
    assert str(op) == "1 ** y"

    #--right arith--
    op = x + 1
    assert str(op) == "x + 1"
    op = x - 1
    assert str(op) == "x - 1"
    op = x * 1
    assert str(op) == "x * 1"
    op = x / 1
    assert str(op) == "x / 1"
    op = x // 1
    assert str(op) == "x // 1"
    op = x ** 1
    assert str(op) == "x ** 1"



def test_op_cmp_overloads():
    x, y, z = Var(f8,'x'), Var(f8,'y'), Var(f8,'z')

    #--beta cmp--
    op = (x + z) < (y + z)
    assert str(op) == "(x + z) < (y + z)"
    op = (x + z) <= (y + z)
    assert str(op) == "(x + z) <= (y + z)"
    op = (x + z) > (y + z)
    assert str(op) == "(x + z) > (y + z)"
    op = (x + z) >= (y + z)
    assert str(op) == "(x + z) >= (y + z)"
    op = (x + z) == (y + z)
    assert str(op) == "(x + z) == (y + z)"

    # #--left cmp--
    op = 1 < (y + z)
    assert str(op) == "(y + z) > 1"
    op = 1 <= (y + z)
    assert str(op) == "(y + z) >= 1"
    op = 1 > (y + z)
    assert str(op) == "(y + z) < 1"
    op = 1 >= (y + z)
    assert str(op) == "(y + z) <= 1"
    op = 1 == (y + z)
    assert str(op) == "(y + z) == 1"

    # #--right cmp--
    op = (x + z) < 1
    assert str(op) == "(x + z) < 1"
    op = (x + z) <= 1
    assert str(op) == "(x + z) <= 1"
    op = (x + z) > 1
    assert str(op) == "(x + z) > 1"
    op = (x + z) >= 1
    assert str(op) == "(x + z) >= 1"
    op = (x + z) == 1
    assert str(op) == "(x + z) == 1"

def test_op_arith_overloads ():
    x, y, z = Var(f8,'x'), Var(f8,'y'), Var(f8,'z')

    #--beta cmp--
    op = (x + z) + (y + z)
    assert str(op) == "(x + z) + (y + z)"
    op = (x + z) - (y + z)
    assert str(op) == "(x + z) - (y + z)"
    op = (x + z) * (y + z)
    assert str(op) == "(x + z) * (y + z)"
    op = (x + z) / (y + z)
    assert str(op) == "(x + z) / (y + z)"
    op = (x + z) // (y + z)
    assert str(op) == "(x + z) // (y + z)"
    op = (x + z) ** (y + z)
    assert str(op) == "(x + z) ** (y + z)"

    # #--left cmp--
    op = 1 + (y + z)
    assert str(op) == "1 + (y + z)"
    op = 1 - (y + z)
    assert str(op) == "1 - (y + z)"
    op = 1 * (y + z)
    assert str(op) == "1 * (y + z)"
    op = 1 / (y + z)
    assert str(op) == "1 / (y + z)"
    op = 1 // (y + z)
    assert str(op) == "1 // (y + z)"
    op = 1 ** (y + z)
    assert str(op) == "1 ** (y + z)"

    # #--right cmp--
    op = (x + z) + 1
    assert str(op) == "(x + z) + 1"
    op = (x + z) - 1
    assert str(op) == "(x + z) - 1"
    op = (x + z) * 1
    assert str(op) == "(x + z) * 1"
    op = (x + z) / 1
    assert str(op) == "(x + z) / 1"
    op = (x + z) // 1
    assert str(op) == "(x + z) // 1"
    op = (x + z) ** 1
    assert str(op) == "(x + z) ** 1"

def test_ptr_ops():
    from cre.default_funcs import ObjIsNone, ObjEquals
    with cre_context("test_ptr_ops"):
        BOOP = define_fact("BOOP", {"val" : f8,"nxt" : "BOOP"})

        a,b,c = Var(BOOP,"a"), Var(BOOP,"b"), Var(BOOP,"c")    

        l1 = ObjIsNone(a.nxt)
        l2 = ObjEquals(a.nxt, b.nxt)

        assert str(l2) == 'a.nxt == b.nxt'
        # print(repr(l2))
        # assert repr(l2) == 'ObjEquals([a:CREObjType].nxt, [b:CREObjType].nxt)'

        l3 = ObjEquals(a, a.nxt)
        assert str(l3) == 'a == a.nxt'

        x0 = BOOP(1.0)
        x1 = BOOP(1.0)
        y0 = BOOP(2.0,x0)
        assert ObjEquals(x0,x0)
        assert not ObjEquals(x0,y0)
        assert not ObjEquals(x0,x1)
        assert ObjEquals(a, b.nxt)(x0, y0)

def test_bad_compose():
    a, b = Var(f8,'a'), Var(f8,'b')
    x, y = Var(unicode_type,'x'), Var(unicode_type,'y')

    # Var case
    q = a + b
    with pytest.raises(TypeError):
        q(x,y)

    # CREFunc case
    q_s = x + y
    with pytest.raises(TypeError):
        q_s(q,q)

    # Constant case
    with pytest.raises(TypeError):
        q("A","B")

        
def test_minimal_str():
    from cre.default_funcs import CastFloat, CastStr
    with cre_context("test_minimal_str"):
        BOOP = define_fact("BOOP", 
            {"id": unicode_type, "val" : f8}
        )

        a,b = Var(BOOP,"a"), Var(BOOP,"b")

        c = CastStr(CastFloat(a.id) + (b.val))

        assert str(c) == "str(float(a.id) + b.val)"
        assert c.minimal_str(ignore_funcs=[CastFloat, CastStr]) == "a + b"

        c = CastStr(CastFloat(a.id) + (b.val) + 100)        

        assert str(c) == "str((float(a.id) + b.val) + 100)"
        assert c.minimal_str(ignore_funcs=[CastFloat, CastStr]) == "(a + b) + 100"


# ---------------------------------------------------
# : Performance Benchmarks

def setup_call():
    @CREFunc(signature=i8(i8,i8))
    def Add(a, b):
        return a + b

    @CREFunc(signature=i8(i8,i8,i8))
    def Add3(a, b, c):
        return a + b + c

    a = Var(i8,'a')
    b = Var(i8,'b')
    comp = Add3(0,a,b)
    return (Add, comp), {}

N = 100

@njit(cache=True)
def apply_100x100(op):
    z = 0
    for i in range(N):
        for j in range(N):
            z += op(i, j)
    return z

@pytest.mark.benchmark(group="cre_func")
def test_b_call_inlined_uncomposed_100x100(benchmark):
    benchmark.pedantic(lambda op,comp: apply_100x100(op),
        setup=setup_call, warmup_rounds=1, rounds=10)

@pytest.mark.benchmark(group="cre_func")
def test_b_call_composed_100x100(benchmark):
    benchmark.pedantic(lambda op,comp: apply_100x100(comp),
        setup=setup_call, warmup_rounds=1, rounds=10)

i8_ft = FunctionType(i8(i8,i8))
@njit(cache=True)
def call_heads_100x100(op):
    z = 0
    for i in range(N):
        for j in range(N):
            f = _func_from_address(i8_ft,op.call_heads_addr)
            z += f(i,j)   
    return z    

@pytest.mark.benchmark(group="cre_func")
def test_b_call_dynamic_100x100(benchmark):
    benchmark.pedantic(lambda op,comp: call_heads_100x100(op),
        setup=setup_call, warmup_rounds=1, rounds=10)


def setup_compose():
    @CREFunc(signature=i8(i8,i8,i8))
    def Add3(a, b, c):
        return a + b + c

    a = Var(i8,'a')
    b = Var(i8,'b')
    return (Add3, a, b), {}

@njit(cache=True)
def compose(Add3, a, b):
    for i in range(100):
        Add3(i,a,b)

@pytest.mark.benchmark(group="cre_func")
def test_compose_py_100(benchmark):
    benchmark.pedantic(compose.py_func,
        setup=setup_compose, warmup_rounds=1, rounds=10)

@pytest.mark.benchmark(group="cre_func")
def test_compose_nb_100(benchmark):
    benchmark.pedantic(compose,
        setup=setup_compose, warmup_rounds=1, rounds=10)



# class ShouldRaise():
#     def __enter__(self):



if __name__ == "__main__":
    import faulthandler; faulthandler.enable()
    import sys

    # @CREFunc(signature=i8(i8,i8,i8))
    # def Add3(a, b, c):
    #     return a + b + c

    # from cre.utils import _tuple_getitem
    # @njit
    # def bar():
    #     return _tuple_getitem((1,"2",False), 1)

    # print(bar())
    # # raise ValueError()

    
    # @njit
    # def foo(f,a,b):
    #     f(7, a, b)

    # a = Var(i8,'a')
    # b = Var(i8,'b')
    # # foo(Add3,a,b)
    # with PrintElapse("Compose 100"):
    #     for i in range(100):
    #         Add3(i,a,b)

    

    # test_commutes()
    # test_no_mutate_on_compose()
    # test_numerical()
    # test_string()
    # test_obj()
    # test_mixed_types()
    # test_njit_compose()
    # test_compose_deref_bases()
    test_not_jittable()
    # test_returns_object()
    # test_var_cmp_overloads()
    # test_var_arith_overloads()
    # test_op_cmp_overloads()
    # test_op_arith_overloads()
    # test_ptr_ops()
    # test_constant()
    # test_bad_compose()
    # test_minimal_str()

    sys.exit()
    # @njit(f8(f8,f8),cache=True)
    with PrintElapse("DEFINE DIVIDE"):
        @CREFunc(signature=f8(f8,f8),
                shorthand='{0}/{1}')
        def Divide(a, b):
            if(a == 0):
                raise ValueError("Bad a")
            else:
                return a / b

    with PrintElapse("DEFINE QDIVIDE"):
        @CREFunc(signature=f8(f8,f8),
                shorthand='{0}/{1}')
        def ZDivide(a, b):
            if(a == 0):
                raise ValueError("Bad a")
            else:
                return a / b

    sys.exit() # Stuff below still has issues


    with pytest.raises(ValueError):
        Divide(0,2)
    
    with pytest.raises(ValueError):
        Divide(0,2)    

    print("::")
    # with pytest.raises(ZeroDivisionError):
    print(Divide(1,0))

    @njit(cache=True)
    def boop(op,a,b):
        op(a,b)

    boop(Divide, 0,2)
    # boop(Divide, 1,0)



    f8_ft = FunctionType(f8(f8,f8))
    @njit(cache=True)
    def boop(cf,a,b):
        fn = _func_from_address(f8_ft, cf.call_heads_addr)
        fn(a,b)

    boop(Divide, 0,2)
    # boop(Divide, 1,0)

    # print(list(Divide.overloads.values())[0].type_annotation.__dict__)
    # print(list(Divide.overloads.values())[0].entry_point)

    








