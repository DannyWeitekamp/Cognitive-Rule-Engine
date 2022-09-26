from numba import njit, u8
from numba.cpython.hashing import _Py_hash_t, _Py_uhash_t, _PyHASH_XXROTATE, _PyHASH_XXPRIME_1, _PyHASH_XXPRIME_2, _PyHASH_XXPRIME_5, process_return
@njit(_Py_uhash_t(_Py_uhash_t,_Py_uhash_t), cache=True)
def accum_item_hash(acc,lane):
    if lane == _Py_uhash_t(-1):
        return _Py_uhash_t(1546275796)
    acc += lane * _PyHASH_XXPRIME_2
    acc = _PyHASH_XXROTATE(acc)
    acc *= _PyHASH_XXPRIME_1
    return acc
