from numba import types, njit, jit, prange
from numba import deferred_type, optional
from numba import void, b1, u1, u2, u4, u8, i1, i2, i4, i8, f4, f8, c8, c16
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
from numbert.caching import unique_hash, source_to_cache, import_from_cached, \
                             source_in_cache, gen_import_str
from numbert.gensource import assert_gen_source





d = Dict.empty(unicode_type,i8)
d['boop'] = 1

class InferenceHistory():
    ''' 
    This class is a wrapper around an inference history tuple of type 'typ':
        (u_vds, u_vs, dec_u_vs, records) where:
            u_vds : DictType(typ,i8) - maps instanes of typ to unique integers
            u_vs : ListType(typ) - keys of u_vds in contiguous array/List
            dec_u_vs : ListType(typ) - same as u_vs, but only for declared items
            records : DictType(i8,ListType(record_type)) - all records by depth

        record_type : tuple of (op_uid, btsr_flat, bstr_shape, arg_types, vmap)
            op_uid : an integer that identifies the op for the record
            btsr_flat : flattened bstr (bstr is multidimensional array of output uids
                with one dimension for each argument to the operator)
            bstr_shape : the shape of bstr
            arg_types : strings for each argument type
            vmap : DictType(typ, i8) maps items to their uid
    '''
    def __init__(self,history,
                      declare,
                      make_contiguous,
                      insert_record,
                      backtrace_goals,
                      backtrace_selection,
                      record_type,
                      ):
        self.history = history
        self._declare = declare
        self._make_contiguous = make_contiguous
        self._insert_record = insert_record
        self._backtrace_goals = backtrace_goals
        self._backtrace_selection = backtrace_selection
        self._insert_record = insert_record
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

    def insert_record(self,depth,op, btsr, vmap):
        btsr_flat = btsr.reshape(-1)
        btsr_shape = np.array(btsr.shape,np.int64)
        self._insert_record(self.history,depth,op.uid,btsr_flat,btsr_shape,op.arg_types,vmap)

    def backtrace_goals(goals, hist_elems, max_depth, max_solutions=1):
        return self._backtrace_goals(goals, self.history, hist_elems, max_depth, max_solutions=1)


    def backtrace_selection():
        return self._backtrace_selection(sel, self.history, hist_elems, max_depth, max_solutions=1)






class DummyKB():
    inf_histories = {}
    curr_infer_depth = 0

    def get_inf_history(self,x,typ=None,force_regen=False):
        if(typ is None): typ = TYPE_ALIASES[infer_type(x)]
        if(typ not in self.inf_histories):
            hash_code = x.hash if hasattr(x,'hash') else unique_hash([typ])
            assert_gen_source(typ,hash_code)

            out1 = import_from_cached(typ,hash_code,[
                'declare', 'make_contiguous', 'empty_inf_history', 'insert_record', 'backtrace_goals', 'backtrace_selection'
                ],'InfHistory').values()

            #Import types from the rest
            out2 = [None]# out2 = import_from_cached(typ,hash_code,['record_type']).values()
            declare, make_contiguous, empty_inf_history, insert_record, \
             backtrace_goals, backtrace_selection, record_type  = tuple([*out1,*out2])
            self.inf_histories[typ] = InferenceHistory(
                empty_inf_history(),
                declare,
                make_contiguous,
                insert_record,
                backtrace_goals,
                backtrace_selection,
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


start_time = time.time()

h = kb.inf_histories['f8']
vmap = Dict.empty(f8,i8)
vmap[7.0] = 1
btsr = np.ones((2,2),np.int64)
op = type('op', (), {})()
op.uid = 1
op.arg_types = List(['f8','f8'])
print("ELAPSE Insert", time.time() - start_time)
h.insert_record(1,op,btsr,vmap)

print("ELAPSE Insert", time.time() - start_time)

start_time = time.time()
h.insert_record(1,op,btsr,vmap)
print("ELAPSE Insert", time.time() - start_time)
print(h.history[-1][1])



# @njit(cache=True)
# def select_from_list(lst,sel):
#     out = List()
#     for s in sel:
#         out.append(lst[s])
#     return out

# def select_from_collection(col,sel):
#     if(isinstance(col,np.ndarray)):
#         return col[sel]
#     elif(isinstance(col,List)):
#         return select_from_list(col,sel)


# def backtrace_selection(sel,history,hist_elems,max_depth, max_solutions=1):
#     '''Same as backtrace_goals except takes a set of indicies into u_vs'''
#     _,u_vs,_,_ = history
#     #f8
#     goals = u_vs[sel]
#     #other
#     goals = List()
#     for s in sel:
#         goals.append(u_vs[s])
#     return backtrace_goals(goals,history,hist_elems,max_depth,max_solutions=max_solutions)


