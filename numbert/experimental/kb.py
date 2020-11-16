from numba import types, njit, guvectorize,vectorize,prange
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple, Tuple, Array
from numba.cpython.unicode import  _set_code_point
from numba.experimental import structref
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

from numbert.experimental.context import _BaseContextful
from numbert.experimental.transform import infer_type


from numbert.experimental.struct_gen import gen_struct_code
from numbert.caching import import_from_cached, source_in_cache, source_to_cache

   
i8_arr = i8[:]
u1_arr = u1[:]
str_to_bool_dict = DictType(unicode_type,u1_arr)
two_str = UniTuple(unicode_type,2)
two_str_set = DictType(two_str,u1)

kb_data_fields = [
    ("enum_data" , DictType(unicode_type,i8_arr)),
    ("enum_consistency" , DictType(two_str,u1)),
    ("consistency_listeners" , DictType(i8, two_str_set)),
    ("consistency_map_counter" , Array(i8, 0, "C")),
    ("unnamed_counter" , Array(i8, 0, "C")),
]

if(not source_in_cache("KnowledgeBaseData",'KnowledgeBaseData')):
    source = gen_struct_code("KnowledgeBaseData",kb_data_fields)
    source_to_cache("KnowledgeBaseData",'KnowledgeBaseData',source)
    
KnowledgeBaseData, KnowledgeBaseDataTypeTemplate = import_from_cached("KnowledgeBaseData",
    "KnowledgeBaseData",["KnowledgeBaseData","KnowledgeBaseDataTypeTemplate"]).values()
print(KnowledgeBaseData, KnowledgeBaseDataTypeTemplate)

KnowledgeBaseDataType = KnowledgeBaseDataTypeTemplate(fields=kb_data_fields)


@njit(cache=True)
def init_store_data(
        NB_Type # Template attr
    ):
    data = Dict.empty(unicode_type,NB_Type)
    return data

@njit(cache=True)
def declare(store_data, kb_data, name, obj):
    store_data[name] = obj
    signal_inconsistent(kb_data.consistency_listeners,name,"*")

@njit(cache=True)
def declare_unnamed(store_data, kb_data, obj):
    name = "%" + str(kb_data.unnamed_counter.item())
    kb_data.unnamed_counter += 1
    store_data[name] = obj
    signal_inconsistent(kb_data.consistency_listeners,name,"*")


@njit(cache=True)
def modify_attr(store_data, kb_data, name, attr, value):
    if(name in store_data):
        #This probably requires mutable types
        raise NotImplemented()
        # data[name].attr = value
    else:
        raise ValueError()

    signal_inconsistent_attr(kb_data.consistency_listeners,name,attr)

@njit(cache=True)
def retract(store_data, kb_data, name):
    del store_data[name]
    signal_inconsistent(kb_data.consistency_listeners,name,"")  




def gen_knowledge_store_aot_funcs(cc,typ,NB_Type):
    '''Wraps jitted functions into an ahead of time compiled module
       called from generated source code for each fact type
    '''
    store_data_type = DictType(unicode_type, NB_Type)

    @cc.export('init_store_data',store_data_type())
    @njit(nogil=True, fastmath=True, cache=True)
    def _init_store_data():
      return init_store_data(NB_Type)

    cc.export('declare',(store_data_type, KnowledgeBaseDataType, unicode_type, NB_Type))(declare)
    cc.export('declare_unnamed',(store_data_type, KnowledgeBaseDataType, NB_Type))(declare_unnamed)
    # cc.export('modify_attr',?(??))(modify_attr)
    cc.export('retract',(store_data_type, KnowledgeBaseDataType, unicode_type))(retract)
   




        
class KnowledgeStore(_BaseContextful):
    ''' Stores KnowledgeBase data for a particular type of fact'''
    def __init__(self, typ, kb, context=None):
        super().__init__(context)

        
        self.kb = kb
        self.kb_data = kb.kb_data
        # self.enum_data, self.enum_consistency, self.consistency_listeners, \
        # self.consistency_map_counter, self.unnamed_counter = self.kb_data
        # print(self.kb_data)

        # spec = self.context.registered_specs[typ]
        # print(self.context.jitstructs)
        struct = self.context.jitstructs[typ]
        out = import_from_cached(typ,struct.hash,[
            'init_store_data', 'declare', 'declare_unnamed', 
            'retract'
            ],typ).values()
        self._init_store_data, self._declare, self._declare_unnamed, self._retract = out 

        self.store_data = self._init_store_data()


    def declare(self,*args):
        if(len(args) == 2):
            # print(self.kb_data)
            self._declare(self.store_data,self.kb_data,args[0],args[1])
        else:
            self._declare_unnamed(self.store_data,self.kb_data,args[0])

    def modify(self,name,obj):
        raise NotImplemented()

    def retract(self,name):
        self._retract(name)


@njit(cache=True)
def init_kb_data():
    enum_data = Dict.empty(unicode_type,i8_arr)

    consistency_listeners = Dict.empty(i8, two_str_set)

    enum_consistency = Dict.empty(two_str,u1)
    consistency_map_counter = np.zeros(1,dtype=np.int64) 
    consistency_listeners[0] = enum_consistency
    consistency_map_counter += 1
    
    unnamed_counter = np.zeros(1,dtype=np.int64)

    return KnowledgeBaseData(enum_data, enum_consistency, consistency_listeners,
                             consistency_map_counter, unnamed_counter)


@njit(cache=True)
def signal_inconsistent(consistency_listeners, name, attr):
    for _,cm in consistency_listeners.items():
        cm[(name,attr)] = True


@njit(cache=True)
def add_consistency_map(kb_data, c_map):
    '''Adds a new consitency map, returns it's index in the knowledgebase'''
    _,_, consistency_listeners, consistency_map_counter = kb_data
    consistency_map_counter += 1
    consistency_listeners[consistency_map_counter[0]] = c_map
    return consistency_map_counter

@njit(cache=True)
def remove_consistency_map(kb_data, index):
    '''Adds a new consitency map, returns it's index in the knowledgebase'''
    _, _,_, consistency_listeners, _ = kb_data
    del consistency_listeners[index]

class KnowledgeBase(structref.StructRefProxy):
    ''' '''
    # class KnowledgeBaseData(structref.StructRefProxy):
    def __new__(cls, context=None):
        self = structref.StructRefProxy.__new__(cls)
        # self.super().__init__(context)
        _BaseContextful.__init__(self,context)
        self.stores = {}


        self.kb_data = init_kb_data()

        return self
    # def __init__(self, context=None):
        
    #     # self.enum_data = self.enum_data, self.enum_consistency, self.consistency_listeners, \
    #     # self.consistency_map_counter, self.unnamed_counter = self.kb_data
    #     # self.

    def _get_fact_type(self,x):
        x_t = type(x)
        assert hasattr(x_t, 'name'), "Can only declare namedtuples built w/ numbert.define_fact()"
        return x_t.name

    def declare(self,name,x):
        typ = self._get_fact_type(x)
        if(typ not in self.stores):
            self.stores[typ] = KnowledgeStore(typ,self)
        self.stores[typ].declare(name,x)

    def modify():

        raise NotImplemented()

    def retract(self,name):

        raise NotImplemented()


@structref.register
class KnowledgeBaseTypeTemplate(types.StructRef):
    pass
    

structref.define_proxy(KnowledgeBase, KnowledgeBaseTypeTemplate, [])
KnowledgeBaseType = KnowledgeBaseTypeTemplate(fields=[])


def _get_fact_type(x):
    x_t = type(x)
    assert hasattr(x_t, 'name'), "Can only declare namedtuples built w/ numbert.define_fact()"
    return x_t.name

# print(ks.enum_consistency)

from numba.extending import overload_method
@overload_method(KnowledgeBaseTypeTemplate, "declare")
def kb_declare(self, name, x):
    print("HERE", x, name)
    typ = _get_fact_type(x)
    print("HERE", x, typ, name)
    if(typ not in self.stores):
        self.stores[typ] = KnowledgeStore(typ,self)
    store_data = self.stores[typ].store_data
    def impl(self, name,x):
        return declare(store_data, name, x)
    return impl




# @overload_method(TypeRef, 'empty')
# def typeddict_empty(cls, key_type, value_type):
#     if cls.instance_type is not DictType:
#         return

#     def impl(cls, key_type, value_type):
#         return dictobject.new_dict(key_type, value_type)

#     return impl



