#From here: https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
import numba
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.core.types import unicode_type
from numba.core.dispatcher import Dispatcher
import numpy as np

import numba.typed.typedlist as tl_mod 
import numba.typed.typeddict as td_mod
import os
from numbert.caching import cache_dir

os.environ['NUMBA_CACHE_DIR'] = os.path.join(os.path.split(cache_dir)[0], "numba_cache")

#Monkey Patch Numba so that the builtin functions for List() and Dict() cache between runs 
def monkey_patch_caching(mod,exclude=[]):
	for name, val in mod.__dict__.items():
		if(isinstance(val,Dispatcher) and name not in exclude):
			val.enable_caching()

#They promised to fix this by 0.51.0, so we'll only run it if an earlier release
if(tuple([int(x) for x in numba.__version__.split('.')]) < (0,51,0)):
	monkey_patch_caching(tl_mod,['_sort'])
	monkey_patch_caching(td_mod)


#These will be filled in if the user registers a new type
TYPE_ALIASES = {
	"float" : 'f8',
	"flt" : 'f8',
	"number" : 'f8',
	"string" : 'unicode_type',
	"str" : 'unicode_type'
}

REGISTERED_TYPES = {'f8': f8,
					 'unicode_type' : unicode_type}

numba_type_map = {
	"f8" : f8,
	"unicode_type" : unicode_type,
	"string" : unicode_type,
	"number" : f8,	
}

py_type_map = {
	"f8" : float,
	"unicode_type" : str,
	"string" : str,
	"number" : float,	
}

numpy_type_map = {
	"string" : '|U%s',
	"number" : np.float64,	
}
