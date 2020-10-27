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
from numbert.gensource import gen_source_standard_imports, gen_source_inf_hist_types, \
							  gen_source_empty_inf_history, gen_source_insert_record





d = Dict.empty(unicode_type,i8)
d['boop'] = 1

class InferenceHistory():
    def __init__(self,history,
                      declare,
                      make_contiguous,
                      record_type
                      ):
        self.history = history
        self._declare = declare
        self._make_contiguous = make_contiguous
        self.record_type = record_type
        self.declared_processed = True

    def assert_declared_processed(self):
        if(not self.declared_processed):
            #TODO: History should be a StructRef so that not redefining
            st = time.time()
            self.history = self._make_contiguous(self.history,0)     
            # print("_process_declared: ",time.time()-st)
        self.declared_processed = True

    def declare(self,x):
        self._declare(self.history,x)
        self.declared_processed = False



class DummyKB():
    inf_histories = {}
    curr_infer_depth = 0

    def get_inf_history(self,x,typ=None,force_regen=False):
        if(typ is None): typ = TYPE_ALIASES[infer_type(x)]
        if(typ not in self.inf_histories):
            hash_code = x.hash if hasattr(x,'hash') else unique_hash([typ])
            if(not source_in_cache(typ,hash_code) or force_regen):
                d_hsh = x.hash if hasattr(x,'hash') else None
                source =  gen_source_standard_imports()
                source += gen_source_inf_hist_types(typ,hash_code)#,d_hsh)
                source += gen_source_empty_inf_history(typ)
                source += gen_source_insert_record(typ)
                source_to_cache(typ,hash_code,source,True)
            
            #Import functions from the aot compiled stuff
            out1 = import_from_cached(typ,hash_code,[
                'declare', 'make_contiguous', 'empty_inf_history'
                ],'InfHistory').values()

            #Import types from the rest
            out2 = [None]# out2 = import_from_cached(typ,hash_code,['record_type']).values()
            declare, make_contiguous, empty_inf_history, record_type = tuple([*out1,*out2])
            self.inf_histories[typ] = InferenceHistory(
                empty_inf_history(),
                declare,
                make_contiguous,
                record_type
            )
        return self.inf_histories[typ]


    def _assert_declared_processed(self,types=None,force_regen=False):
        if(not self.declared_consistent):
            for typ in (types if types else REGISTERED_TYPES.keys()):
                st = time.time()
                inf_hist = self.get_inf_history(typ,force_regen=force_regen)
                inf_hist.assert_declared_processed()


    def declare(self,x,typ=None):
        '''Takes a whole state conforming to the format output by Numbalizer.state_to_nb_objects()
           or individual items
        '''
        if(isinstance(x,dict)):
            pass #TODO: need to rethink
            # for typ, nb_objects_of_type in x.items():
            #   assert typ in REGISTERED_TYPES, "Type is not registered %r." % typ
            #   self._assert_declare_store(typ)
            #   declare_nb_objects(self.hists[typ][0][0],nb_objects_of_type)
        else:
            inf_hist = self.get_inf_history(x,typ)
            inf_hist.declare(x)

import time
start_time = time.time()
print("START")
kb = DummyKB()
for x in [0,1,2,3]:
    kb.declare(x)


u_vds, u_vs, dec_u_vs, records = kb.inf_histories['f8'].history
print(u_vs)
kb.inf_histories['f8'].assert_declared_processed()
u_vds, u_vs, dec_u_vs, records = kb.inf_histories['f8'].history
print(u_vs)

print("ELAPSE f8", time.time() - start_time)


start_time = time.time()

for x in [str(x) for x in range(4)]:
    kb.declare(x)

kb.inf_histories['unicode_type'].assert_declared_processed()
u_vds, u_vs, dec_u_vs, records = kb.inf_histories['unicode_type'].history
print(u_vs)

# print(kb.inf_histories['f8'].history)

print("ELAPSE str", time.time() - start_time)
# 
start_time = time.time()

from numbert.numbalizer import Numbalizer
numbalizer = Numbalizer()
ie_spec = {
    "id" : "number",
    "value" : "number"
}

numbalizer.register_specification("P",ie_spec)
# numbalizer.object_to_nb_object({"type": "TF", "id": "moo", "value": "poo",})

for x in range(4):
    # name = "e"+x
    kb.declare(numbalizer.object_to_nb_object(str(x),{"type": "P", "id": x, "value": x}))

print("ELAPSE TF", time.time() - start_time)

