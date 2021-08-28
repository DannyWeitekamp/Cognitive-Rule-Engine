from numba import f8, njit
from numba.core.errors import NumbaPerformanceWarning
from numba.types import  FunctionType, unicode_type
import numpy as np
from cre.op import Op
from cre.var import Var
from cre.utils import _func_from_address
from cre.context import cre_context
from cre.fact import define_fact
import re
import pytest

def test_op_singleton():
    class Add(Op):
        signature = f8(f8,f8)        
        def call(x,y):
            return x + y

    assert isinstance(Add,Op), "Op subclasses should be singleton instances."

    # Test that we can actually pass the Op singleton as a parameter
    #  and reconstruct it.
    ftype = FunctionType(Add.signature)
    @njit(cache=True)
    def foo(op):
        f = _func_from_address(ftype, op.call_addr)
        return f(1,2)

    assert foo(Add) == 3

def test_define_apply_op():
    with pytest.raises(AssertionError):
        class Add(Op):
            pass
    with pytest.raises(AssertionError):
        class Add(Op):
            signature = f8(f8,f8)

    class Add(Op):
        signature = f8(f8,f8)        
        def call(a,b):
            return a + b

    class AddWCheck(Op):
        signature = f8(f8,f8)        
        def call(a,b):
            return a + b
        def check(a,b):
            return a > 0

    assert Add(1,2) == 3
    assert AddWCheck.check(1,2)
    assert not AddWCheck.check(-1,2)

def test_compose_op():
    class Add(Op):
        signature = f8(f8,f8)        
        def call(a,b):
            return a + b    

    AddPlusOne = Add(Var(float),Add(Var(float),1))
    assert AddPlusOne(1,2) == 4

    Add3PlusOne = Add(Var(float),Add(Var(float),Add(Var(float),1)))
    assert Add3PlusOne(7,1,2) == 11

    x = Var(float)
    Double = Add(x,x)
    assert Double(5) == 10

    WeirdDouble = Add(x,Add(x,0))
    assert WeirdDouble(5) == 10

@njit(cache=True)
def extract_var_ptrs(op):
    '''Defined just to reduce runtime on test below'''
    out =np.empty(len(op.var_map),dtype=np.int64)
    for i,x in enumerate(op.var_map):
        out[i] =x 
    return out


def test_var_propagation():
    class Add3(Op):
        signature = f8(f8,f8,f8)
        def call(a, b, c):
            return a + b + c
    x,y,z = Var(float,'x'),Var(float,'y'),Var(float,'z')
    op = Add3(x,y,z)
    assert str(op) == 'Add3(x,y,z)'
    assert [x.get_ptr(),y.get_ptr(),z.get_ptr()] == [*extract_var_ptrs(op)]

def test_order():
    class Subtract(Op):
        signature = f8(f8,f8)
        def call(x, y):
            return x - y

    assert Subtract(1,2) == -1
    assert Subtract(2,1) == 1
    a,b = Var(float,'a'), Var(float,'b')
    s2 = Subtract(b,Subtract(a,1))
    assert s2(3,2) == 2

def test_auto_aliasing():
    class Add3(Op):
        signature = f8(f8,f8,f8)
        def call(x, y, z):
            return x + y + z

    assert str(Add3)=='Add3(x,y,z)'
    a,b,c = Var(float),Var(float),Var(float)
    assert str(Add3(a,b,c)) == 'Add3(a,b,c)'

def test_source_gen():
    class Add(Op):
        signature = f8(f8,f8)        
        short_hand = '({0}+{1})'
        def check(a, b):
            return a > 0
        def call(a, b):
            return a + b

    class Multiply(Op):
        signature = f8(f8,f8)
        short_hand = '({0}*{1})'
        def check(a, b):
            return b != 0
        def call(a, b):
            return a * b    

    Double = Multiply(Var(float,'x'), 2)
    DoublePlusOne = Add(Double,1)
    TimesDoublePlusOne = Multiply(DoublePlusOne,Var(float,'y'))

    assert str(DoublePlusOne) == "Add(Multiply(x,2),1)"
    assert str(TimesDoublePlusOne) == "Multiply(Add(Multiply(x,2),1),y)"
    assert TimesDoublePlusOne.check(-1,1) == False
    assert TimesDoublePlusOne.check(1,0) == False
    assert TimesDoublePlusOne.check(1,1) == True

    assert TimesDoublePlusOne.gen_expr(use_shorthand=True) == '(((x*2)+1)*y)'
    assert TimesDoublePlusOne.gen_expr(use_shorthand=False) == 'Multiply(Add(Multiply(x,2),1),y)'
    assert TimesDoublePlusOne.gen_expr(lang='javascript',use_shorthand=True) == '(((x*2)+1)*y)'

    class IntegerDivision(Op):
        signature = f8(f8,f8)
        short_hand = {
            '*' : '({0}//{1})',
            'js' : 'Math.floor({0}/{1})',
        }
        call_body = {
            "js" : '''{ind}return Math.floor({0}/{1});'''
        }
        def call(a, b):
            return a // b

    assert IntegerDivision.gen_expr(lang='python',use_shorthand=True) == '(a//b)'
    assert IntegerDivision.gen_expr(lang='javascript',use_shorthand=True) == 'Math.floor(a/b)'

    # Python source will just copy it's own definition
    assert "a // b\n" in IntegerDivision.make_source()

    print(IntegerDivision.make_source('js'))
    # For other languages by default look in call_body 
    assert "Math.floor(a/b);\n" in IntegerDivision.make_source('js')

    # Otherwise fall back on any defined short_hands
    IntegerDivision.call_body = {}
    assert "Math.floor(a/b)\n" in IntegerDivision.make_source('js')

    # OpComp built Ops should really only be rendered in the then() of a rule 
    # print(TimesDoublePlusOne.make_source(""))


def test_commutes():
    class Add3(Op):
        signature = f8(f8,f8,f8)        
        commutes = True
        def call(a, b, c):
            return a + b + c

    print(Add3.commutes)
    assert Add3.commutes == [[0,1,2]]
    print(Add3.right_commutes)
    assert str(Add3.right_commutes) == '{2: array([0, 1]), 1: array([0])}'

    with pytest.raises(AssertionError):
        class Floop(Op):
            signature = f8(f8,f8, unicode_type)        
            commutes = [[0,1,2]]
            def call(a, b, c):
                return a + b

def test_fact_args():
    class Add(Op):
        signature = f8(f8,f8)        
        commutes = True
        def call(a, b):
            return a + b

    with cre_context('test_fact_args'):
        spec = {"A" : "string", "B" : "number"}
        BOOP, BOOPType = define_fact("BOOP", spec)

        op = Add(Var(BOOP,'x').B, Var(BOOP,'y').B)
        assert str(op) == "Add(x.B,y.B)"
        assert op(BOOP("A",1),BOOP("B",2)) == 3.0

        vb = Var(BOOP,'v').B
        op = Add(vb,vb)
        assert str(op) == 'Add(v.B,v.B)'
        assert op(BOOP("A",1)) == 2.0

        op = Add(vb,Add(vb,Var(BOOP, 'u').B))
        assert str(op) == 'Add(v.B,Add(v.B,u.B))'
        assert op(BOOP("A",1), BOOP("B",2)) == 4.0


def not_jit_compilable():
    class Add(Op):
        signature = f8(f8,f8)        
        commutes = True
        def call(a, b):
            return a + b

    with pytest.warns(NumbaPerformanceWarning):
        class Map(Op):
            signature = f8(f8)
            def check(a):
                l = []
                d = {1.0: 10.0, 2.0: 20.0}
                for i in range(int(a)):
                    l.append(str(i+1))
                    l.append(float(i+1))

                return d[l[-1]] > 0
            def call(a):
                l = []
                d = {1.0: 10.0, 2.0: 20.0}
                for i in range(int(a)):
                    l.append(str(i+1))
                    l.append(float(i+1))

                return d[l[-1]]


    assert Map(1.0)==10.0
    assert Map.call(1.0)==10.0
    assert Map.check(1.0)==1


    op = Add(Map(Var(float,'x')),Map(Var(float,'y')))
    assert op(1,2)==30.0

    with cre_context('test_fact_args'):
        spec = {"A" : "string", "B" : "number"}
        BOOP, BOOPType = define_fact("BOOP", spec)

        class BOOPMap(Op):
            signature = BOOPType(BOOPType)
            def call(a):
                l = []
                d = {1.0: 10.0, 2.0: 20.0}
                for i in range(int(a.B)):
                    l.append(str(i+1))
                    l.append(float(i+1))

                return BOOP("A", d[l[-1]])

        assert BOOPMap(BOOP("A",1.0)).B == 10.0
        assert BOOPMap.call(BOOP("A",1.0)).B == 10.0


import time
class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.2f} ms')


if __name__ == "__main__":
    # with PrintElapse("test_op_singleton"):
    #     test_op_singleton()
    # # with PrintElapse("test_define_apply_op"):
    #     test_define_apply_op()
    # # with PrintElapse("test_op_singleton"):
    #     test_compose_op()
    # # with PrintElapse("test_var_propagation"):
    #     test_var_propagation()
    #     test_order()
    # # with PrintElapse("test_auto_aliasing"):
    #     test_auto_aliasing()
    # # with PrintElapse("test_source_gen"):
    #     test_source_gen()

        # test_commutes()
        # test_fact_args()
    not_jit_compilable()
            

