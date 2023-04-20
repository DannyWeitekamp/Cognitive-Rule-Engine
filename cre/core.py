#From here: https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
import numba
from numba import types, njit
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16,types, boolean
from numba.core.types import unicode_type, float64
from numba.core.dispatcher import Dispatcher
from numba.extending import intrinsic
import numpy as np

import numba.typed.typedlist as tl_mod 
import numba.typed.typeddict as td_mod
import os
from cre.caching import cache_dir, get_cache_path, import_from_cached

os.environ['NUMBA_CACHE_DIR'] = os.path.join(os.path.split(cache_dir)[0], "numba_cache")

from numba.core.dispatcher import Dispatcher
import numba.typed.typedlist as tl_mod 
import numba.typed.typeddict as td_mod
import cloudpickle
import warnings
import threading
#Monkey Patch Numba so that the builtin functions for List() and Dict() cache between runs 
def monkey_patch_caching(mod,exclude=[]):
    for name, val in mod.__dict__.items():
        if(isinstance(val,Dispatcher) and name not in exclude):
            val.enable_caching()

# monkey_patch_caching(tl_mod,['_sort'])
# monkey_patch_caching(td_mod)

#They promised to fix this by 0.51.0, so we'll only run it if an earlier release
# if(tuple([int(x) for x in numba.__version__.split('.')]) < (0,55,0)):



#These will be filled in if the user registers a new type
TYPE_ALIASES = {
    "boolean" : 'boolean',
    "bool" : 'boolean',
    "float" : 'float64',
    "flt" : 'float64',
    "number" : 'float64',
    "string" : 'unicode_type',
    "str" : 'unicode_type',
    'unicode_type' : 'unicode_type',
    'float64' : 'float64',
}

JITSTRUCTS = {}                  

numba_type_map = {
    "boolean" : boolean,
    "float64" : float64,
    "unicode_type" : unicode_type,
    "string" : unicode_type,
    "number" : float64, 
}

py_type_map = {
    "float64" : float,
    "unicode_type" : str,
    "string" : str,
    "number" : float,   
    "bool" : bool,   
}

numpy_type_map = {
    "string" : '|U%s',
    "number" : np.float64,  
}

STRING_DTYPE = np.dtype("U50")



def standardize_type(typ, context, name='', attr=''):
    '''Takes in a string or type and returns the standardized type'''
    if(isinstance(typ, type)):
        typ = typ.__name__
    if(isinstance(typ,str)):
        typ_str = typ
        is_list = typ_str.lower().startswith("list")
        if(is_list): typ_str = typ_str.split("(")[1][:-1]

        if(typ_str.lower() in TYPE_ALIASES): 
            typ = numba_type_map[TYPE_ALIASES[typ_str.lower()]]
        # elif(typ_str == name):
        #     typ = context.get_deferred_type(name)# DeferredFactRefType(name)
        elif(typ_str in context.name_to_type):
            typ = context.name_to_type[typ_str]
        else:
            typ = context.get_deferred_type(typ_str)
            # raise TypeError(f"Attribute type {typ_str!r} not recognized in spec" + 
            #     f" for attribute definition {attr!r}." if attr else ".")

        if(is_list): typ = ListType(typ)

    if(hasattr(typ, "_fact_type")): typ = typ._fact_type
    return typ

# -----------------------------------------------------------------------
# : Type Registry (part of disk cache)

registry_lock = threading.Lock()

# The fact registry is used to give a unique 't_id' number to each type definition
#  in CRE it is just a text file with <t_id> <Type Name> <Hash Code> on each line
#  This is necessary so that, for instance, custom Fact types can 
def load_type_registry():
    type_registry, type_registry_map = [], {}
    i = 0
    has_warned = False
    path = get_cache_path("type_registry",suffix='')
    if(os.path.exists(path)):
        with open(path,'r') as f:
            for i, line in enumerate(f):
                tokens = line.split()
                c, name, hash_code = tokens[0], tokens[1], tokens[2]
                if(not has_warned and c != str(i)):
                    warnings.warn("CRE cache's type registry is non-contiguous. This may indicate a filesystem race condition in a previous CRE session. This can cause mistyping errors which can produce segmentation faults.", RuntimeWarning)
                    has_warned = True
                type_registry.append((name, hash_code))
                type_registry_map[(name, hash_code)] = i
    return type_registry, type_registry_map


# These are supposed to be synced with a disk resource (i.e. the 'type_registry' file) 
#  so they are globally defined outside of a particular cre_context(), and should
#  be shared across different threads running CRE on the same filesystem.
TYPE_REGISTRY, TYPE_REGISTRY_MAP = load_type_registry()

def lines_in_type_registry():
    return len(TYPE_REGISTRY)

from cre.utils import PrintElapse
def add_type_pickle(typ, t_id):
    ''' Creates a pickled cache for a type object. Note: add_to_type_registry
        can be be called with typ=None, in which case a call to this should follow.
        This pattern allows a type to have access to its t_id before it is defined.
    '''
    pickle_dir = get_cache_path("type_pickles",suffix='')
    os.makedirs(pickle_dir, exist_ok=True)
    with open(os.path.join(pickle_dir,f'{t_id}.pkl'), 'wb') as f:
        cloudpickle.dump(typ, f)

def add_to_type_registry(name, hash_code, typ=None):
    ''' Adds a type name and hash to the type registry and returns a new t_id.
        If the 'typ' object is given then the type is also pickled and placed in the cache.'''
    tup = (name,hash_code)
    if(tup not in TYPE_REGISTRY_MAP):
        with registry_lock:
            count = len(TYPE_REGISTRY)
            TYPE_REGISTRY.append(tup)
        TYPE_REGISTRY_MAP[tup] = count
        with open(get_cache_path("type_registry",suffix=''),'a') as f:
            f.write(f"{count} {name} {hash_code}\n")
        if(typ is not None): add_type_pickle(typ, count)
    return count

def unpickle_type_from_t_id(t_id):
    if(t_id < len(DEFAULT_REGISTERED_TYPES)):
        return list(DEFAULT_REGISTERED_TYPES.values())[t_id]
    name, hash_code = None, None
    pickle_dir = get_cache_path("type_pickles",suffix='')
    with open(os.path.join(pickle_dir,f'{t_id}.pkl'), 'rb') as f:
        typ = cloudpickle.load(f)
    return typ

def t_id_from_type_name(name, hash_code):
    return TYPE_REGISTRY_MAP.get((name, hash_code), -1)
            


DEFAULT_REGISTERED_TYPES = {
    'undefined': types.undefined,
    'bool' : types.bool_,
    'int' : i8,
    'float' : f8,
    'str' : unicode_type,

    # These types filled w/ add_to_type_registry() in source files.
    'CREObj' : types.undefined,
    'Fact' : types.undefined,
    'TupleFact' : types.undefined,
    'Var': types.undefined,
    'CREFunc' : types.undefined,
    'Literal' : types.undefined,
    'Conditions' : types.undefined,
    'Rule' : types.undefined
}

if(not os.path.exists(get_cache_path("type_registry",suffix=''))):
    for name in DEFAULT_REGISTERED_TYPES:
        add_to_type_registry(name,"builtin")


DEFAULT_TYPE_T_IDS = {}
for i, (name, typ) in enumerate(DEFAULT_REGISTERED_TYPES.items()):
    DEFAULT_TYPE_T_IDS[name] = i
    if(typ is not types.undefined):
        DEFAULT_TYPE_T_IDS[typ] = i

DEFAULT_T_ID_TYPES = {v:k for k,v in DEFAULT_TYPE_T_IDS.items()}           


T_ID_UNDEFINED = DEFAULT_TYPE_T_IDS['undefined']
T_ID_BOOL = DEFAULT_TYPE_T_IDS['bool']
T_ID_INT = DEFAULT_TYPE_T_IDS['int'] 
T_ID_FLOAT = DEFAULT_TYPE_T_IDS['float']
T_ID_STR = DEFAULT_TYPE_T_IDS['str']
T_ID_CRE_OBJ = DEFAULT_TYPE_T_IDS['CREObj']
T_ID_FACT = DEFAULT_TYPE_T_IDS['Fact']
T_ID_TUPLE_FACT = DEFAULT_TYPE_T_IDS['TupleFact']
T_ID_VAR = DEFAULT_TYPE_T_IDS['Var']
T_ID_FUNC = DEFAULT_TYPE_T_IDS['CREFunc']
T_ID_LITERAL = DEFAULT_TYPE_T_IDS['Literal']
T_ID_CONDITIONS = DEFAULT_TYPE_T_IDS['Conditions']
T_ID_RULE = DEFAULT_TYPE_T_IDS['Rule']


SHORT_NAMES = {
    types.undefined : "undf",
    types.bool_ : "bool",
    i8 : "i8",
    f8 : "f8",
    unicode_type : "str",
}

def short_name(x):
    return SHORT_NAMES[x]


def register_global_default(name, typ):
    '''Not for external use'''
    assert name in DEFAULT_REGISTERED_TYPES, f"{name} is not preregistered as a global type."
    DEFAULT_REGISTERED_TYPES[name] = typ
    DEFAULT_TYPE_T_IDS[typ] = DEFAULT_TYPE_T_IDS[name]
    DEFAULT_T_ID_TYPES[DEFAULT_TYPE_T_IDS[name]] = typ







