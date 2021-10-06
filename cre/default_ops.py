from numba import i8
from cre.op import Op
from cre.ptrop import PtrOp
from cre.utils import _load_pointer


@Op(shorthand = '({0} == {1})', commutes=True)
def Equals(a, b):
    return a == b

@Op(shorthand = '({0} < {1})')
def LessThan(a, b):
    return a < b

@Op(shorthand = '({0} <= {1})')
def LessThanEq(a, b):
    return a <= b


@Op(shorthand = '({0} > {1})')
def GreaterThan(a, b):
    return a > b

@Op(shorthand = '({0} >= {1})')
def GreaterThanEq(a, b):
    return a >= b

@Op(shorthand = '({0} + {1})', commutes=True)
def Add(a, b):
    return a + b

@Op(shorthand = '({0} - {1})')
def Subtract(a, b):
    return a - b

@Op(shorthand = '({0} * {1})', commutes=True)
def Multiply(a, b):
    return a * b

def denom_not_zero(a,b):
    return b != 0

@Op(shorthand = '({0} / {1})', check=denom_not_zero)
def Divide(a, b):
    return a / b

@Op(shorthand = '({0} // {1})', check=denom_not_zero)
def FloorDivide(a, b):
    return a // b

@Op(shorthand = '({0} ** {1})', check=denom_not_zero)
def Power(a, b):
    return a ** b


@PtrOp(nargs=2, shorthand = '({0} == {1})')
def ObjEquals(ptrs):
    '''From two head_ptrs see if the underlying pointers to objects are the same'''
    objptr0 = _load_pointer(i8,ptrs[0])
    objptr1 = _load_pointer(i8,ptrs[1])
    return objptr0 == objptr1

@PtrOp(nargs=1, shorthand = '({0} == None)')
def ObjIsNone(ptrs):
    '''From a head_ptr see if the underlying object pointer is the NULL pointer'''
    return _load_pointer(i8,ptrs[0]) == 0


