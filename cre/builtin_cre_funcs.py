from numba import i8, njit
from numba.types import unicode_type
from cre.cre_func import CREFunc
from cre.utils import _load_ptr


@CREFunc(shorthand = '{0}', no_raise=True)
def Identity(a):
    return a

@CREFunc(shorthand = '{0} == {1}', commutes=True, no_raise=True)
def Equals(a, b):
    # print("BEF", a, b)
    # print("##", a,b, a==b)
    # print("START Equals", a == b, len(a), len(b))
    return a == b

@CREFunc(shorthand = '{0} < {1}', no_raise=True)
def LessThan(a, b):
    print("LT", a,"<",b)
    return a < b

@CREFunc(shorthand = '{0} <= {1}', no_raise=True)
def LessThanEq(a, b):
    return a <= b


@CREFunc(shorthand = '{0} > {1}', no_raise=True)
def GreaterThan(a, b):
    return a > b

@CREFunc(shorthand = '{0} >= {1}', no_raise=True)
def GreaterThanEq(a, b):
    return a >= b

@CREFunc(shorthand = '{0} + {1}', commutes=True, no_raise=True)
def Add(a, b):
    return a + b

@CREFunc(shorthand = '{0} - {1}', no_raise=True)
def Subtract(a, b):
    return a - b

@CREFunc(shorthand = '{0} * {1}', commutes=True, no_raise=True)
def Multiply(a, b):
    return a * b

def denom_not_zero(a,b):
    return b != 0

@CREFunc(shorthand = '{0} / {1}', check=denom_not_zero)
def Divide(a, b):
    return a / b

@CREFunc(shorthand = '{0} // {1}', check=denom_not_zero)
def FloorDivide(a, b):
    return a // b

@CREFunc(shorthand = '{0} ** {1}')
def Power(a, b):
    return a ** b

@CREFunc(shorthand = '{0} % {1}', no_raise=True)
def Modulus(a, b):
    return a % b

@CREFunc(shorthand = '{0} < {1}', no_raise=True)
def FactIdrecsLessThan(a, b):
    return a.idrec < b.idrec

@CREFunc(shorthand = '{0} + {1}', 
    signature = unicode_type(unicode_type,unicode_type),
    commutes=False, no_raise=True)
def Concatenate(a, b):
    return a + b


@CREFunc(shorthand = '{0} == {1}', ptr_args=True)
def ObjEquals(a, b):
    '''From two head_ptrs see if the underlying pointers to objects are the same'''
    # objptr0 = _load_ptr(i8,a)
    # objptr1 = _load_ptr(i8,b)
    # return objptr0 == objptr1
    return a == b

    
@CREFunc(shorthand = '{0} == None', ptr_args=True)
def ObjIsNone(a):
    '''From a head_ptr see if the underlying object pointer is the NULL pointer'''
    # return _load_ptr(i8,a) == 0
    return a == 0


def check_cast_float(a):
    if(a == ""): return False
    try:
        float(a)
    except:
        return False
    return True

# @CREFunc(shorthand = 'float({0}', check=check_cast_float)
@CREFunc(shorthand = 'float({0})')
def CastFloat(a):
    return float(a)

# print(CastFloat(unicode_type).check('0'))
# print(CastFloat(unicode_type).check('A'))

def check_cast_str(a):
    try:
        str(a)
    except:
        return False
    return True

# @CREFunc(shorthand = 'str({0}', check=check_cast_str)
@CREFunc(shorthand = 'str({0})')
def CastStr(a):
    return str(a)

# print("------------")
# print(CastStr)
# print(CastStr.__class__)
# print("------------")
