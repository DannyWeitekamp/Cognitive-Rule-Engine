#From here: https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
import numba
from numba import types, njit
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16,types
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

GLOBAL_TYPE_COUNT = -1

# The fact registry is used to give a unique number to each fact definition
#  it is just a text file with <Fact Name> <Hash Code> on each line
def lines_in_type_registry():
    global GLOBAL_TYPE_COUNT
    if(GLOBAL_TYPE_COUNT == -1):
        try:
            with open(get_cache_path("type_registry",suffix=''),'r') as f:
                GLOBAL_TYPE_COUNT = len([1 for line in f])
        except FileNotFoundError:
            GLOBAL_TYPE_COUNT = 0
    return GLOBAL_TYPE_COUNT

def add_to_type_registry(name, hash_code):
    global GLOBAL_TYPE_COUNT
    if(GLOBAL_TYPE_COUNT == -1): lines_in_type_registry()
    count = GLOBAL_TYPE_COUNT
    with open(get_cache_path("type_registry",suffix=''),'a') as f:
        f.write(f"{name} {hash_code}\n")
    GLOBAL_TYPE_COUNT += 1
    return count

def type_from_t_id(t_id):
    if(t_id < len(DEFAULT_REGISTERED_TYPES)):
        return list(DEFAULT_REGISTERED_TYPES.values())[t_id]
    name, hash_code = None, None
    with open(get_cache_path("type_registry",suffix=''),'r') as f:
        for i, line in enumerate(f):
            if(i == t_id):
                tokens = line.split()
                name, hash_code = tokens[0], tokens[1]
                break

    fact_type = import_from_cached(name, hash_code, ['fact_type'])['fact_type']
    return fact_type

def t_id_from_type_name(typ_name, hash_code=None):
    if(hash_code is None): hash_code = hash(typ)
    name = str(typ)
    with open(get_cache_path("type_registry",suffix=''),'r') as f:
        for i, line in enumerate(f):
            tokens = line.split()
            if(tokens[0] == name and tokens[1] == hash_code):
                return i
    return -1
            




# @intrinsic
# def _instrinstic_get_null_meminfo(typingctx):
#   def codegen(context, builder, sig, args):
#       null_meminfo = context.get_constant_null(types.MemInfoPointer(types.voidptr))
#       context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), null_meminfo)
#       return null_meminfo
        
#   sig = types.MemInfoPointer(types.voidptr)()

#   return sig, codegen

# @njit(cache=True)
# def _get_null_meminfo():
#   return _instrinstic_get_null_meminfo()

# NULL_MEMINFO = _get_null_meminfo()

CRE_TYPE_EXECUTABLE    =int('10000000', 2)
CRE_TYPE_VAR           =int('10010000', 2)
CRE_TYPE_OP            =int('10100000', 2)
CRE_TYPE_CONDITIONS    =int('10110000', 2)
CRE_TYPE_RULE          =int('11000000', 2)

CRE_TYPE_FACT          = int('00001000', 2)
CRE_TYPE_ATOM          = int('00001001', 2)
CRE_TYPE_PRED          = int('00001010', 2)


DEFAULT_REGISTERED_TYPES = {
                            'undefined': types.undefined,
                            'bool' : types.bool_,
                            'int' : i8,
                            'float' : f8,
                            'str' : unicode_type,
                            'CREObj' : types.undefined,
                            'Fact' : types.undefined,
                            'TupleFact' : types.undefined,
                            'Var': types.undefined,
                            'Op' : types.undefined,
                            'Literal' : types.undefined,
                            'Conditions' : types.undefined
                            }

if(not os.path.exists(get_cache_path("type_registry",suffix=''))):
    for t in DEFAULT_REGISTERED_TYPES:
        add_to_type_registry(t,"builtin")


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
T_ID_OP = DEFAULT_TYPE_T_IDS['Op']
T_ID_LITERAL = DEFAULT_TYPE_T_IDS['Literal']
T_ID_CONDITIONS = DEFAULT_TYPE_T_IDS['Conditions']


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







