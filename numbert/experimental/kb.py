from numba import types, njit, guvectorize,vectorize,prange
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

from numbert.experimental.context import _BaseContextful
from numbert.experimental.transform import infer_type





@njit(cache=True)
def init_store_data(
		typ # Template attr
	):
	data = Dict.empty(unicode_type,typ)
	return data

@njit(cache=True)
def signal_inconsistent(
		consistency_listeners,
		name,
		n_attrs # Template attr
	):
	for cm in consistency_listeners:
		cm[(name,"*")] = True

@njit(cache=True)
def signal_inconsistent_attr(consistency_listeners, name, attr):
	for cm in consistency_listeners:
		cm[(name,attr)] = True

@njit(cache=True)
def declare(store_data, kb_data, name, obj):
	_,_, consistency_listeners,_,_ = kb_data
	store_data[name] = obj
	signal_inconsistent(consistency_listeners,name)

@njit(cache=True)
def declare_unnamed(store_data, kb_data, obj):
	_,_, consistency_listeners,_,_ = kb_data
	name = "%" + str(unnamed_counter[0])
	unnamed_counter[0] += 1
	store_data[name] = obj
	signal_inconsistent(consistency_listeners,name)


@njit(cache=True)
def modify_attr(store, name, attr, value):
	data, _,_, consistency_listeners,_,_ = store
	if(name in data):
		#This probably requires mutable types
		raise NotImplemented()
		# data[name].attr = value
	else:
		raise ValueError()

	signal_inconsistent_attr(consistency_listeners,name,attr)

@njit(cache=True)
def retract(store, name):
	data, _,_, consistency_listeners,_,_ = store
	del data[name]
	signal_inconsistent(consistency_listeners,name)	


@njit(cache=True)
def add_consistency_map(store, c_map):
	'''Adds a new consitency map, returns it's index in the knowledgebase'''
	data, _,_, consistency_listeners, consistency_map_counter = store
	consistency_map_counter[0] += 1
	consistency_listeners[consistency_map_counter[0]] = c_map
	return consistency_map_counter[0]

@njit(cache=True)
def remove_consistency_map(store, index):
	'''Adds a new consitency map, returns it's index in the knowledgebase'''
	_, _,_, consistency_listeners, _ = store
	del consistency_listeners[index]

		
class KnowledgeStore(_BaseContextful):
	''' Stores KnowledgeBase data for a particular type of fact'''
	def __init__(self, typ, kb, context=None):
		super().__init__(context)

		self.store_data = init_store_data(typ)
		self.kb = kb
		self.kb_data = kb.kb_data
		self.enum_data, self.enum_consistency, self.consistency_listeners, \
		self.consistency_map_counter, self.unnamed_counter = self.kb_data

	def declare(self,*args):
		if(len(args) == 2):
			declare(self.store_data,self.kb_data,args[0],args[1])
		else:
			declare_unnamed(self.store_data,self.kb_data,args[0])

	def modify(self,name,obj):
		raise NotImplemented()

	def retract(self,name):
		retract(name)


i8_arr = i8[:]
u1_arr = u1[:]
str_to_bool_dict = DictType(unicode_type,u1_arr)
two_str = UniTuple(unicode_type,2)
two_str_set = DictType(two_str,u1)


@njit(cache=True)
def init_kb_data():
	enum_data = Dict.empty(unicode_type,i8_arr)

	consistency_listeners = Dict.empty(i8, two_str_set)

	enum_consistency = Dict.empty(two_str,u1)
	consistency_map_counter = np.zeros(1,dtype=np.int64) 
	consistency_listeners[0] = enum_consistency
	consistency_map_counter += 1
	
	unnamed_counter = np.zeros(1,dtype=np.int64)

	return enum_data, enum_consistency, consistency_listeners, consistency_map_counter, unnamed_counter


class KnowledgeBase(_BaseContextful):
	''' '''
	def __init__(self, context=None):
		super().__init__(context)
		self.stores = {}

		self.kb_data = init_kb_data()
		self.enum_data, self.enum_consistency, self.consistency_listeners, \
		self.consistency_map_counter, self.unnamed_counter = self.kb_data
		# self.

	def declare(x,name,typ):
		raise NotImplemented()

	def modify():
		raise NotImplemented()

	def retract():
		raise NotImplemented()


kb = KnowledgeBase()

ks = KnowledgeStore(unicode_type,kb)
ks.declare("A","A")
print(ks.enum_consistency)

	
# @overload_method(KnowledgeBase.declare)
# def typeddict_empty(self, x):

#     def impl(cls, key_type, value_type):
#         return dictobject.new_dict(key_type, value_type)

#     return impl



# @overload_method(TypeRef, 'empty')
# def typeddict_empty(cls, key_type, value_type):
#     if cls.instance_type is not DictType:
#         return

#     def impl(cls, key_type, value_type):
#         return dictobject.new_dict(key_type, value_type)

#     return impl



