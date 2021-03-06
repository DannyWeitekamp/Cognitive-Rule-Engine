import numba 
import numpy as np
import numba
from numba import types, jit,njit, generated_jit
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.types import ListType, unicode_type, UnicodeType
from numba.cpython.unicode import _empty_string, _set_code_point, _get_code_point, PY_UNICODE_1BYTE_KIND
from numba.extending import overload, overload_method
# from numba.cpython.unicode import _empty_string, _set_code_point, _get_code_point, PY_UNICODE_1BYTE_KIND


DIGITS_START = 48
DIGITS_END = 58
DASH = 45
DOT = 46
PLUS = 43
E_CHAR = 101

## FROM STRING ##

# @njit(i8(unicode_type,),cache=True)
@njit(cache=True)
def str_to_posint(s):
    final_index, result = len(s) - 1, 0
    for i,v in enumerate(s):
        char = ord(v)
        if(char < DIGITS_START or char >= DIGITS_END):
            raise ValueError("Could not convert string to numeric type.")
        result += (char - 48) * (10 ** (final_index - i))
    return result

# @njit(i8(unicode_type,), cache=True)
@njit(cache=True)
def str_to_int(s):
    neg = (s[0] == "-")
    if(neg): s = s[1:]
    result = str_to_posint(s)
    return -result if neg else result

@njit(cache=True)
def str_to_float(s):
    neg = (s[0] == "-")
    if(neg): s = s[1:]

    exp, dec_loc, exp_loc = 1,-1, len(s)
    for i,c in enumerate(s):
        if(c == "."): 
            dec_loc=i
        elif(c == "e"):
            exp_loc=i
            exp = str_to_int(s[exp_loc+1:])

    if(dec_loc != -1):
        result = str_to_posint(s[:dec_loc])
        result += float(str_to_posint(s[dec_loc+1:exp_loc])) / np.power(10,(exp_loc-dec_loc-1))
    else:
        result = str_to_posint(s)

    if(exp != 1):
        result *= 10.**exp

    if(neg): result = -result
    return result


## TO STRING ##

@njit(cache=True)
def int_to_str(x):
    isneg = int(x < 0.0)
    x = np.abs(x)
    l = 0 
    _x = x
    while _x > 0:
        _x = _x // 10
        l += 1
    s = _empty_string(PY_UNICODE_1BYTE_KIND,l+isneg)
    if(isneg): _set_code_point(s,0,DASH)
    for i in range(l):
        digit = x % 10
        _set_code_point(s,isneg+l-i-1,digit + DIGITS_START)
        x = x // 10
    return s

# if(_x < 1e-128): _x = (_x * 1e128) % 10; l2 += 128; print(_x,"??128")
    # if(_x < 1e-64): _x = (_x * 1e64) % 10; l2 += 64; print(_x,"??64")
    # if(_x < 1e-32): _x = (_x * 1e32) % 10; l2 += 32; print(_x,"??32")
    # if(_x < 1e-16): _x = (_x * 1e16) % 10; l2 += 16; print(_x,"??16")
    # if(_x < 1e-8): _x = (_x * 1e8) % 10; l2 += 8; print(_x,"??8")
    # if(_x < 1e-4): _x = (_x * 1e4) % 10; l2 += 4; print(_x,"??4")
    # if(_x < 1e-2): _x = (_x * 1e2) % 10; l2 += 2; print(_x,"??2")
    # if(_x < 1e-1): _x = (_x * 1e1) % 10; l2 += 1; print(_x,"??1")
    

@njit(cache=True)
def get_n_digits(x):
    l1,l2 = 0,-1
    _x = x
    while _x > 0:
        _x = _x // 10
        l1 += 1

    _x = x % 10     
    while _x > 1e-10:
        _x = (_x * 10) % 10
        l2 += 1
        if(l2 >= 16): break
    return l1, l2


@njit(cache=True)
def float_to_str(x):
    if(x == np.inf):
        return 'inf'
    elif(x == -np.inf):
        return '-inf'

    isneg = int(x < 0.0)
    x = np.abs(x)
    
    if(x != 0.0):
        # There is probably a more efficient way to do this
        e = np.floor(np.log10(x))
        if(10**e - x > 0): e -= 1
    else:
        e = 0
    
    is_exp, is_neg_exp = e >= 16, e <= -16

    exp_chars = 0
    if (is_exp or is_neg_exp):
        exp_chars = 4
        if(e >= 100 or e <= -100): exp_chars = 5


    if(is_exp):
        offset_x = np.around(x * (10.0**-(e)),15)
        l1, l2 = get_n_digits(offset_x)
    elif(is_neg_exp):
        offset_x = np.around(x * (10**-(e)),15)
        l1, l2 = get_n_digits(offset_x)
    else:
        offset_x = x
        l1,l2 = get_n_digits(x)
        l2 = max(1,l2) # Will have at least .0 
    
    use_dec = l2 > 0

    # print("<<", e, offset_x, l2)

    l = l1+l2+use_dec
    length = l+isneg+exp_chars
    s = _empty_string(PY_UNICODE_1BYTE_KIND,length)
    if(isneg): _set_code_point(s,0,DASH)

    _x = offset_x
    for i in range(l1):
        digit = int(_x % 10)
        _set_code_point(s,(isneg+l1)-i-1,digit + DIGITS_START)
        _x = _x // 10

    if(use_dec):
        _set_code_point(s,l1+isneg,DOT)

    _x = offset_x % 10
    for i in range(l2):
        _x = (_x * 10) % 10
        digit = int(_x)  
        
        _set_code_point(s,(isneg+l1)+i+use_dec,digit + DIGITS_START)

    if(is_exp or is_neg_exp):
        i = (isneg+l1+use_dec+l2)
        _set_code_point(s,i,E_CHAR)        
        if(is_exp):
            _set_code_point(s,i+1,PLUS)     
        if(is_neg_exp):
            _set_code_point(s,i+1,DASH)           

        i = length-1
        exp = np.abs(e)
        while(exp > 0):
            digit = exp % 10
            _set_code_point(s,i,digit + DIGITS_START)
            exp = exp // 10
            i -= 1

    return s

# Monkey Patch type conversions so that float(str) and str(float)
# work like in normal python

@overload(str)
def overload_float_to_str(x):
    if(x is types.bool_):
        def impl(x):
            return "True" if x else "False"
    elif(x in types.real_domain): 
        def impl(x):
            return float_to_str(x)
    elif(x in types.integer_domain):
        def impl(x):
            return int_to_str(x)
    else:
        return
    return impl

@overload(float)
def overload_str_to_float(x):
    if(x != unicode_type): return
    def impl(x):
        return str_to_float(x)
    return impl

@overload(int)
def overload_str_to_int(x):
    if(x != unicode_type): return
    def impl(x):
        return str_to_int(x)
    return impl


@njit(cache=True,inline="never")
def format_str(s, args):
    strs = List.empty_list(unicode_type)
    start = 0
    end = 0
    for i,c in enumerate(s):
        if(c == "{"):
            end = i 
            strs.append(s[start:end])
        elif(c == "}"):
            start = i+1
            ind = int(s[end+1:i])
            if(ind >= len(args)): raise ValueError()
            strs.append(args[ind])

    strs.append(s[start:])
    return "".join(strs)

@generated_jit(cache=True,nopython=True)
@overload_method(UnicodeType,'format')
def overload_format(s, *args):
    zero_type = args[0]
    if(isinstance(zero_type, types.BaseTuple)):
        zero_type = zero_type[0]
    if(isinstance(zero_type,types.ListType)):
        def impl(s, *args):
            return format_str(s,args[0])
    else:
        def impl(s, *args):
            return format_str(s,args)
    return impl



from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                         AbstractTemplate, infer_global, infer,
                                         infer_getattr, signature,
                                         bound_function, make_callable_template)


@infer_global(float)
class Float(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws

        [arg] = args

        if(arg == unicode_type):
            #For unicode_type fall back on @overload 
            return 

        if arg not in types.number_domain:
            raise TypeError("float() only support for numbers")

        if arg in types.complex_domain:
            raise TypeError("float() does not support complex")

        if arg in types.integer_domain:
            return signature(types.float64, arg)

        elif arg in types.real_domain:
            return signature(arg, arg)






# @njit
# def failed_int():
#   int("1.2")

# print(repr(failed_str()))
# print(repr(failed_float()))
# failed_int()
