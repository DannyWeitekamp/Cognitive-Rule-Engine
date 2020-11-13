from numba import types, njit, guvectorize,vectorize,prange, generated_jit
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple 
from numba.cpython.unicode import  _set_code_point
from numbert.utils import cache_safe_exec
from numbert.core import TYPE_ALIASES, REGISTERED_TYPES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from numbert.gensource import assert_gen_source
from numbert.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache
from collections import namedtuple
import numpy as np
import timeit
import itertools
import types as pytypes
import sys
import __main__

# from .context import _BaseContextful
# from .transform import infer_type

@generated_jit()
def foo(x):
	def out(x):
		return Dict.empty(x,i8)
	return out


@njit
def bar(x):
	return Dict.empty(x,i8)

print(foo(f8))
print(foo(unicode_type))
print(bar(f8)._numba_type_)


