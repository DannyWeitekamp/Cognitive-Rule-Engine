from numba import f8, njit
import numpy as np
from cre.op import Op
from cre.var import Var
import pytest

def test_op_singleton():
    class Add(Op):
        signature = f8(f8,f8)        
        def call(x,y):
            return x + y

    assert isinstance(Add,Op), "Op subclasses should be singleton instances."

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

    print(str(DoublePlusOne))
    assert str(DoublePlusOne) == "Add(Multiply(x,2),1)"
    assert str(TimesDoublePlusOne) == "Multiply(Add(Multiply(x,2),1),y)"
    assert TimesDoublePlusOne.check(-1,1) == False
    assert TimesDoublePlusOne.check(1,0) == False
    assert TimesDoublePlusOne.check(1,1) == True

    assert TimesDoublePlusOne.gen_expr(use_shorthand=True) == '(((x*2)+1)*y)'
    assert TimesDoublePlusOne.gen_expr(use_shorthand=False) == 'Multiply(Add(Multiply(x,2),1),y)'


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
        test_op_singleton()
    # with PrintElapse("test_define_apply_op"):
        test_define_apply_op()
        test_define_apply_op()
    # with PrintElapse("test_op_singleton"):
        test_compose_op()
    # with PrintElapse("test_var_propagation"):
        test_var_propagation()
    # with PrintElapse("test_auto_aliasing"):
        test_auto_aliasing()
    # with PrintElapse("test_source_gen"):
        test_source_gen()
            

