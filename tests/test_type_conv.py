import numpy as np
from numba import types, jit,njit
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.types import ListType, unicode_type
from cre.type_conv import str_to_float, str_to_int, int_to_str, float_to_str
from cre.op import Op
from cre.var import Var


import timeit
N=1000
def time_ms(f):
    f() #warm start
    return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))

def test_str_to_float():
    assert str_to_float("-12312.001345e-2")==float("-12312.001345e-2")
    assert str_to_float("12312.01e2")==float("12312.01e2")
    assert str_to_float("12312")==float("12312")
    assert str_to_float("1.3452e-12")==float("1.3452e-12")
    assert str_to_float("1.3452e12")==float("1.3452e12")
    # This one is harder to get exact, but it is close
    assert np.isclose(str_to_float("1.3452e25"),float("1.3452e25"))
    assert np.isclose(str_to_float("1.3452e-25"),float("1.3452e-25"))
    assert str_to_float("1.3452")==float("1.3452")
    assert str_to_float("-1000.9999")==float("-1000.9999")
    assert str_to_float("-0.009999")==float("-0.009999")
    assert str_to_float("0")==float("0")

def test_str_to_int():

    assert str_to_int("123") == int("123")
    assert str_to_int("-123") == int("-123")
    assert str_to_int("-1000023") == int("-1000023")
    # print(str_to_int("123.0") == int("123.0"))
    # print(str_to_int("-123.01") == int("-123.01"))
    # print(str_to_int("0.01") == int("0.01"))
    # print(str_to_int("-0.01") == int("-0.01"))

def to_str_close(x):
    # print(int_to_str(x), str(x))
    return np.isclose(float(float_to_str(x)), float(str(x)))


def test_to_str():
    # assert to_str_close(147)
    # assert to_str_close(-147) 
    assert to_str_close(146.78) 
    assert to_str_close(-146.78)
    assert to_str_close(0.78) 
    assert to_str_close(-0.78) 
    assert to_str_close(4.0)
    assert to_str_close(-4.0) 
    assert to_str_close(1.34e24) 
    assert to_str_close(-1.34e-24)
    assert to_str_close(2e24)
    assert to_str_close(2e-24)
    assert to_str_close(-2e24)
    assert to_str_close(-2e-24)
    assert to_str_close(9.9999999999e24)
    assert to_str_close(9.99999999999999e24)
    assert to_str_close(9.9999999999e-24)
    assert to_str_close(9.99999999999999e-24)
    assert to_str_close(-9.9999999999e24)
    assert to_str_close(-9.99999999999999e24)
    assert to_str_close(-9.9999999999e-24)
    assert to_str_close(-9.99999999999999e-24)
    assert to_str_close(-9.99999999999999e-200)
    assert to_str_close(9.99999999999999e+200)


@njit(cache=True)
def _str_test():
  return str(1.2)

@njit(cache=True)
def _float_test():
  return float("1.2")

def test_str_float_overloaded():
    assert _float_test() == 1.2
    assert _str_test() == '1.2'


def test_ops():
    class StrToFloat(Op):
        signature = f8(unicode_type,)
        def call(x):
            return float(x)

    class Add3(Op):
        signature = f8(f8,f8,f8)        
        commutes = True
        def call(a, b, c):
            return a + b + c

    x,y,z = Var(unicode_type,'x'), Var(unicode_type,'y'), Var(unicode_type,'z')
    op = Add3(StrToFloat(x),StrToFloat(y),StrToFloat(z))
    assert str(op) == 'Add3(StrToFloat(x), StrToFloat(y), StrToFloat(z))'
    assert op('1','2','3') == 6.0

if __name__ == "__main__":
    print(str_to_float("0"))
    # test_str_to_float()
    # test_to_str()
    # test_str_to_int()
    # test_ops()

    # @njit
    # def test_nb_float():
    #     str_to_float('-12312.0')

    # def test_py_float():
    #     float("-12312.0")   

    # @njit
    # def test_nb_int():
    #     str_to_int("-123")  

    # def test_py_int():
    #     int("-123") 


    # print("nb_float\t",time_ms(test_nb_float))
    # print("py_float\t",time_ms(test_py_float))
    # print("nb_int\t",time_ms(test_nb_int))
    # # print("nb_int_core\t",time_ms(test_nb_int_core))
    # print("py_int\t",time_ms(test_py_int))
    # print("float_to_str", time_ms(lambda : float_to_str(146.78)))

