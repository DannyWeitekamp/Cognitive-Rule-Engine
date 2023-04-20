import numpy as np
from numba import njit, u8, types
from numba.cpython.hashing import (_Py_hash_t, _Py_uhash_t, _PyHASH_XXROTATE,
        _PyHASH_XXPRIME_1, _PyHASH_XXPRIME_2, _PyHASH_XXPRIME_5, process_return,
        _siphash24)
from numba.core.extending import register_jitable

# ------------------------------------------------------------
# : accum_item_hash

@njit(_Py_uhash_t(_Py_uhash_t,_Py_uhash_t), cache=True)
def accum_item_hash(acc,lane):
    if lane == _Py_uhash_t(-1):
        return _Py_uhash_t(1546275796)
    acc += lane * _PyHASH_XXPRIME_2
    acc = _PyHASH_XXROTATE(acc)
    acc *= _PyHASH_XXPRIME_1
    return acc


# ------------------------------------------------------------
# : unicode_hash_noseed

from numba.core.unsafe.bytes import grab_byte, grab_uint64_t
from numba.cpython.unicode import _kind_to_byte_width
import sys

# Set a biggish hash-cutoff for using fast DJBX33A, instead of slow siphash 
#  should speed things up appreciably, security sensitive things shouldn't
#  be calling the no_seed hash implementation anyway.
_HASH_CUTOFF = 10

# Constants instead of using Python's random seed
const_siphash_k0 = u8(7823388449531244853)
const_siphash_k1 = u8(2452911659039219235)

@register_jitable(locals={'_hash': _Py_uhash_t})
def _Py_HashBytes_noseed(val, _len):
    if (_len == 0):
        return process_return(0)

    if (_len < _HASH_CUTOFF):
        # TODO: this branch needs testing, needs a CPython setup for it!
        # /* Optimize hashing of very small strings with inline DJBX33A. */
        _hash = _Py_uhash_t(5381)  # /* DJBX33A starts with 5381 */
        for idx in range(_len):
            _hash = ((_hash << 5) + _hash) + np.uint8(grab_byte(val, idx))

        _hash ^= _len
    else:
        tmp = _siphash24(types.uint64(const_siphash_k0),
                         types.uint64(const_siphash_k1),
                         val, _len)
        _hash = process_return(tmp)
    return process_return(_hash)

# Hashes a unicode string, but without using python's hash seed.
#  Ensures that the hash is is the same between executions. 
@njit(cache=True)
def unicode_hash_noseed(val):    
    kindwidth = _kind_to_byte_width(val._kind)
    _len = len(val)
    return _Py_HashBytes_noseed(val._data, kindwidth * _len)
