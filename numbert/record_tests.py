from numba import types, njit, jit, prange
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, DictType, unicode_type, Array, Tuple, NamedTuple
from numba.experimental import jitclass
from numba.core.dispatcher import Dispatcher
from numba.core import sigutils 
import numba
from numba.core.dispatcher import Dispatcher
import numpy as np
import timeit
from pprint import pprint
from numbert.utils import cache_safe_exec
from numbert.numbalizer import infer_type, infer_nb_type
from collections import namedtuple
from numbert.core import TYPE_ALIASES, numba_type_map, py_type_map, REGISTERED_TYPES
from numbert.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, gen_import_str
from numbert.gensource import gen_source_standard_imports


def gen_source_inf_hist_types(typ,hsh,ind='   '):
	s = "from numba.pycc import CC\n\n"
	s += "cc = CC('InfHistory_{}')\n\n".format(hsh)
	s += "record_type = Tuple([i8, i8[::1], i8[::1],\n"
	s += ind + "ListType(unicode_type),\n"
	s += ind + 	"DictType({},i8)])\n".format(typ)
	s += "record_list_type = ListType(record_type)\n"
	# s += "nt = {}_InfHistory = namedtuple('{}_InfHistory',\n".format(typ,typ)
	# s += ind + "['u_vds','u_vs','dec_u_vs','records'])\n".format(typ)
	s += "nb_nt  = Tuple((\n"
	s += ind + "DictType({},i8),\n".format(typ)
	if(typ == 'f8'):
		s += ind + "f8[::1],\n"
		s += ind + "f8[::1],\n"
	else:
		s += ind + "ListType({}),\n".format(typ)
		s += ind + "ListType({}),\n".format(typ)
	s += ind + "DictType(i8,record_list_type)\n"
	# s += "],nt)\n"
	s += "))\n\n"
	return s

def gen_source_declare(typ,ind='   '):
	header = "@cc.export('declare',(nb_nt,{}))\n".format(typ)
	header += "@njit((nb_nt,{}),cache=True,fastmath=True,nogil=True)\n".format(typ)
	header += "def declare(history, x):\n"
	body =  ind + "dec_record = history[-1][0][0]\n"
	body += ind + "_,_,_,_, vmap = dec_record\n"
	body += ind + "if(x not in vmap):\n"
	body += ind*2 + "vmap[x] = len(vmap)\n\n"
	
	return header + body

	
def gen_source_process_declared(typ,ind='   '):
	header = "@cc.export('process_declared',nb_nt(nb_nt))\n"
	header += "@njit(cache=True,fastmath=True,nogil=True)\n"
	header += "def process_declared(history):\n"

	body =  ind + "records = history[-1]\n"
	body += ind + "dec_record = records[0][0]\n"
	body += ind + "_,_,_,_, vmap = dec_record\n"
	body += ind + "u_vds =  Dict.empty({},i8)\n".format(typ)
	body += ind + "for x in vmap:\n"
	body += ind*2 +  "u_vds[x] = 0\n"

	if(typ == 'f8'):
		body += ind + "u_vs = np.empty(len(u_vds))\n"
		body += ind + "for i,v in enumerate(u_vds):\n"
		body += ind*2 +  "u_vs[i] = v\n"
	else:
		body += ind + "u_vs = List.empty_list({})\n".format(typ)
		body += ind + "for i,v in enumerate(u_vds):\n"
		body += ind*2 +  "u_vs.append(v)\n"
	body += ind + "dec_u_vs = u_vs.copy()\n"
	body += ind + "return (u_vds,u_vs,dec_u_vs,records)\n\n"
	return header + body

def gen_source_empty_inf_history(typ,ind='   '):
	header = "@cc.export('empty_inf_history',nb_nt())\n"
	header += "@njit(cache=True,fastmath=True,nogil=True)\n"
	header += "def empty_inf_history():\n"

	body =  ind + "records = Dict.empty(i8,record_list_type)\n"
	body += ind + "records[0] = List.empty_list(record_type)\n"
	body += ind + "tl = List.empty_list(unicode_type);tl.append('{}');\n".format(typ)
	body += ind + "vmap = Dict.empty({},i8)\n".format(typ)
	#Type : (0 (i.e. no-op), _hist, shape, arg_types, vmap)
	body += ind + "records[0].append(\n" 
	body += ind*2 + "(0, np.empty((0,),dtype=np.int64),\n"
	body += ind*2 + "np.empty((0,),dtype=np.int64),\n"
	body += ind*2 + "tl,vmap))\n"
		
	body += ind + "u_vds = Dict.empty({},i8)\n".format(typ)
	if(typ == 'f8'):
		body += ind + "u_vs = np.empty(0)\n"
		body += ind + "dec_u_vs = np.empty(0)\n"
	else:
		body += ind + "u_vs = List.empty_list({})\n".format(typ)
		body += ind + "dec_u_vs = List.empty_list({})\n".format(typ)
	body += ind + "return (u_vds,u_vs,dec_u_vs,records)\n\n"
	return header + body


d = Dict.empty(unicode_type,i8)
d['boop'] = 1
# print(gen_source_inf_hist_types('f8'))
# print(gen_source_process_declared('f8'))
# print(gen_source_empty_inf_history('f8'))


# a =foo()
# print(a)

class InferenceHistory():
	def __init__(self,history,
					  declare,
					  process_declared,
					  record_type
					  ):
		self.history = history
		self._declare = declare
		self._process_declared = process_declared
		self.record_type = record_type
		self.declared_processed = True

	def assert_declared_processed(self):
		if(not self.declared_processed):
			#TODO: History should be a StructRef so that not redefining
			st = time.time()
			self.history = self._process_declared(self.history)		
			print("_process_declared: ",time.time()-st)
		self.declared_processed = True

	def declare(self,x):
		self._declare(self.history,x)
		self.declared_processed = False



class DummyKB():
	inf_histories = {}
	curr_infer_depth = 0
	
	# self.hist_consistent = True
	# self.declared_consistent = True
	def get_inf_history(self,x,typ=None,force_regen=False):
		if(typ is None): typ = TYPE_ALIASES[infer_type(x)]
		# struct_typ = self._assert_record_type(typ)
		# typ_store = self.hists[typ]
		if(typ not in self.inf_histories):
			hash_code = x.hash if hasattr(x,'hash') else unique_hash([typ])
			if(not source_in_cache(typ,hash_code) or force_regen):
				source =  gen_source_standard_imports()
				source += gen_source_inf_hist_types(typ,hash_code)
				source += gen_source_declare(typ)
				source += gen_source_process_declared(typ)
				source += gen_source_empty_inf_history(typ)
				source_to_cache(typ,hash_code,source,True)
				print(source)
			
			#Import functions from the aot compiled stuff
			out1 = import_from_cached(typ,hash_code,[
				'declare', 'process_declared', 'empty_inf_history'
				],'InfHistory').values()

			#Import types from the rest
			out2 = [None]# out2 = import_from_cached(typ,hash_code,['record_type']).values()

			declare, process_declared, empty_inf_history, record_type = tuple([*out1,*out2])
			self.inf_histories[typ] = InferenceHistory(
				empty_inf_history(),
				declare,
				process_declared,
				record_type
			)
		return self.inf_histories[typ]


	def _assert_declared_processed(self,types=None,force_regen=False):
		if(not self.declared_consistent):
			for typ in (types if types else REGISTERED_TYPES.keys()):
				st = time.time()
				inf_hist = self.get_inf_history(typ,force_regen=force_regen)
				print("get_inf_hist",time.time()-st)
				inf_hist.assert_declared_processed()


	def declare(self,x,typ=None):
		'''Takes a whole state conforming to the format output by Numbalizer.state_to_nb_objects()
		   or individual items
		'''
		if(isinstance(x,dict)):
			pass #TODO: need to rethink
			# for typ, nb_objects_of_type in x.items():
			# 	assert typ in REGISTERED_TYPES, "Type is not registered %r." % typ
			# 	self._assert_declare_store(typ)
			# 	declare_nb_objects(self.hists[typ][0][0],nb_objects_of_type)
		else:
			inf_hist = self.get_inf_history(x,typ)
			inf_hist.declare(x)

import time
start_time = time.time()
print("START")
kb = DummyKB()
for x in [0,1,2,3]:
	kb.declare(x)

for x in [str(x) for x in range(4)]:
	kb.declare(x)



u_vds, u_vs, dec_u_vs, records = kb.inf_histories['f8'].history
print(u_vs)
kb.inf_histories['f8'].assert_declared_processed()
u_vds, u_vs, dec_u_vs, records = kb.inf_histories['f8'].history

kb.inf_histories['unicode_type'].assert_declared_processed()
u_vds, u_vs, dec_u_vs, records = kb.inf_histories['f8'].history
print(u_vs)

# print(kb.inf_histories['f8'].history)

print("ELAPSE", time.time() - start_time)
# 
