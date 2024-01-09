import numba
from numba import njit, u8,i8
import numpy as np

WN_VAR_TYPE = u8(1)
WN_BAD_DEREF = u8(2)
WN_INFER_UNPROVIDED = u8(3)
WN_FAIL_MATCH = u8(4)
WN_NOT_MAP_LITERAL = u8(5)
WN_NOT_MAP_VAR = u8(6)

np_why_not_type = np.dtype([
    # Enum for ATTR or LIST
    ('ptr', np.int64),
    ('var_ind0', np.int64),
    ('var_ind1', np.int64),
    ('d_ind', np.int64),
    ('c_ind', np.int64),
    ('kind', np.uint64),
])

why_not_type = numba.from_dtype(np_why_not_type)

@njit
def new_why_not(ptr, var_ind0, var_ind1=-1, d_ind=-1, c_ind=-1, kind=0):
    arr = np.empty(1,dtype=why_not_type)
    arr[0].ptr = i8(ptr)
    arr[0].var_ind0 = i8(var_ind0)
    arr[0].var_ind1 = i8(var_ind1)
    arr[0].d_ind = i8(d_ind)
    arr[0].c_ind = i8(c_ind)
    arr[0].kind = u8(kind)
    return arr[0]
