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
import itertools
from pprint import pprint
from numbert.utils import cache_safe_exec
from numbert.numbalizer import infer_type, infer_nb_type
from collections import namedtuple
from numbert.core import TYPE_ALIASES, numba_type_map, py_type_map, REGISTERED_TYPES
from numbert.caching import unique_hash, source_to_cache, import_from_cached, \
                             source_in_cache, gen_import_str
from numbert.gensource import assert_gen_source
from numbert.operator import BaseOperator, BaseOperatorMeta, Var, OperatorComposition


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
    def __init__(self,typ, hash_code):
        '''Constructed via typ string and hash_code string'''

        assert_gen_source(typ,hash_code)

        out1 = import_from_cached(typ,hash_code,[
            'inf_declare', 'make_contiguous', 'empty_inf_history',
             'insert_record', 'backtrace_goals', 'backtrace_selection'
            ],'InfHistory').values()

        out2 = [None]# out2 = import_from_cached(typ,hash_code,['record_type']).values()
        declare, make_contiguous, empty_inf_history, insert_record, \
         backtrace_goals, backtrace_selection, record_type  = tuple([*out1,*out2])

        self.history = empty_inf_history()
        self._declare = declare
        self._make_contiguous = make_contiguous
        self._insert_record = insert_record
        self._backtrace_goals = backtrace_goals
        self._backtrace_selection = backtrace_selection
        self._insert_record = insert_record
        self.record_type = record_type
        self.declared_processed = True

    @property
    def u_vds(self):
        return self.history[0]

    @property
    def u_vs(self):
        return self.history[1]

    @property
    def dec_u_vs(self):
        return self.history[2]

    @property
    def records(self):
        return self.history[3]



    def assert_declared_processed(self):
        if(not self.declared_processed):
            #TODO: History should be a StructRef so that not redefining
            st = time.time()
            self.history = self._make_contiguous(self.history,0)     
            # print("_process_declared: ",time.time()-st)
        self.declared_processed = True

    def make_contiguous(self,depth):
        self.history = self._make_contiguous(self.history,depth)     

    def declare(self,x):
        self._declare(self.history,x)
        self.declared_processed = False

    def insert_record(self,depth,op, btsr, vmap):
        btsr_flat = btsr.reshape(-1)
        btsr_shape = np.array(btsr.shape,dtype=np.int64)
        self._insert_record(self.history,depth,op.uid,btsr_flat,btsr_shape,List(op.arg_types),vmap)

    def backtrace_goals(self, goals, hist_elems, max_depth, max_solutions=1):
        return self._backtrace_goals(goals, self.history, hist_elems, max_depth, max_solutions)


    def backtrace_selection(self, sel,hist_elems, max_depth, max_solutions=1):
        return self._backtrace_selection(sel, self.history, hist_elems, max_depth, max_solutions)






class DummyKB():
    inf_histories = {}
    curr_infer_depth = 0
    declared_processed = False

    def get_inf_history(self,typ=None,x=None,force_regen=False):
        if(typ is None):
            if(x is None): raise ValueError("typ and x cannot be None")
            typ = TYPE_ALIASES[infer_type(x)]
        if(typ not in self.inf_histories):
            hash_code = x.hash if x and hasattr(x,'hash') else unique_hash([typ])
            self.inf_histories[typ] = InferenceHistory(typ,hash_code)
            
        return self.inf_histories[typ]


    def _assert_declared_processed(self,types=None,force_regen=False):
        if(not self.declared_processed):
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
            inf_hist = self.get_inf_history(typ=typ,x=x)
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


# start_time = time.time()

# h = kb.inf_histories['f8']
# vmap = Dict.empty(f8,i8)
# vmap[7.0] = 1
# btsr = np.ones((2,2),np.int64)
# op = type('op', (), {})()
# op.uid = 1
# op.arg_types = List(['f8','f8'])
# print("ELAPSE Insert", time.time() - start_time)
# h.insert_record(1,op,btsr,vmap)

# print("ELAPSE Insert", time.time() - start_time)

# start_time = time.time()
# h.insert_record(1,op,btsr,vmap)
# print("ELAPSE Insert", time.time() - start_time)
# print(h.history[-1][1])


import time
start_time = time.time()
def forward(kb,ops):
    kb._assert_declared_processed()

    output_types = set()
    new_records = {typ:[] for typ in output_types}
    depth = kb.curr_infer_depth = kb.curr_infer_depth+1
    
    for op in ops:
        if(not all([t in kb.inf_histories for t in op.arg_types])): continue
        if(isinstance(op,BaseOperatorMeta)):
            args = [kb.inf_histories[t].u_vs for t in op.u_arg_types]
            btsr, vmap = op.broadcast_forward(*args)
        elif(isinstance(op,OperatorComposition)):
            btsr, vmap = broadcast_forward_op_comp(kb,op)
        kb.get_inf_history(op.out_type).insert_record(depth, op, btsr, vmap)
        output_types.add(op.out_type)
    for typ in output_types:
        kb.get_inf_history(typ).make_contiguous(depth)


class Add(BaseOperator):
    commutes = True
    signature = 'float(float,float)'

    def forward(x, y):
        return x + y

forward(kb,[Add])
forward(kb,[Add])

from numbert.aot_template_funcs import backtrace_goals, HE
he_list = ListType(HE)
@njit(cache=True)
def HistElmListList():
    return List.empty_list(he_list)

def _retrace_goal_history(kb,ops,goal,g_typ, max_solutions):
    '''Backtraces the operators that produced the intermediate values at each
      depth leading up to the goal value. Output is a list of dictionaries
      (one for each timestep). Dictionaries are keyed by type string and have
      values of type ListType(ListType(HE)) where HE is a History Element signifiying
      the application of an operator and indicies of it's arguments. Each List in the 
      outer List corresponds to a unique value for that depth that has it's own List of 
      HistElements that produced it.
    '''
    h = kb.get_inf_history(g_typ)
    
    goals = List.empty_list(REGISTERED_TYPES[g_typ])
    goals.append(goal)

    hist_elems = HistElmListList()

    arg_inds = h.backtrace_goals(goals, hist_elems, max_depth=kb.curr_infer_depth, max_solutions=1)

    out = [{g_typ: hist_elems}]
    i = 1
    while(True):
        nxt = {}
        new_arg_inds = None
        for typ in arg_inds:
            hist_elems = HistElmListList()
            
            typ_new_inds = h.backtrace_selection(arg_inds[typ],hist_elems, kb.curr_infer_depth-i, max_solutions=max_solutions)
            if(new_arg_inds is None):
                new_arg_inds = typ_new_inds
            else:
                for typ,inds in typ_new_inds.items():
                    if(typ not in new_arg_inds):
                        new_arg_inds[typ] = inds
                    else:
                        new_arg_inds[typ] = np.append(new_arg_inds[typ],inds)

            nxt[typ] = hist_elems
        out.append(nxt)
        if(new_arg_inds is None or len(new_arg_inds) == 0):
            break
        assert i <= kb.curr_infer_depth, "Retrace has persisted past current infer depth."
        i += 1
        arg_inds = new_arg_inds
    print(out)
    return list(reversed(out))

def retrace_solutions(kb,ops,goal,g_typ,max_solutions=1000):
    ''' Calls _retrace_goal_history() to get hist elements leading up to the 
        production of the goal value, then uses these hist elements to compose
        a set nested tuples that can be used to instantiate an operator composition
    '''

    goal_history = _retrace_goal_history(kb,ops,goal,g_typ,max_solutions)
    pprint(goal_history)

    tups = []
    for depth in range(len(goal_history)):
        tups_depth = {}
        for typ in goal_history[depth].keys():
            dec_u_vs = kb.get_inf_history(typ).dec_u_vs

            tups_depth_typ = tups_depth[typ] = []
            for j,l in enumerate(goal_history[depth][typ]):
                tups_depth_j = []
                for he in l:
                    op_uid, args = he
                    if(op_uid == 0):
                        tups_depth_j.append(Var(binding=dec_u_vs[args[0].item()],type=typ))
                    else:
                        op = BaseOperator.operators_by_uid[op_uid]
                        prod_lists = [[op]] +[tups[depth-1][a_typ][args[k]] for k,a_typ in enumerate(op.arg_types)]
                        for t in itertools.product(*prod_lists):
                            # if(isinstance(t[0],OperatorComposition)):
                            #   op_comp = t[0]
                            #   t = OperatorCompositiondeepcopy(op_comp.tup)
                            #   t.bind(*op_comp.args)

                            #   raise ValueError("POOP")

                            tups_depth_j.append(t)
                tups_depth_typ.append(tups_depth_j)
        tups.append(tups_depth)

    out = [OperatorComposition(t) for t in tups[-1][g_typ][0][:max_solutions]]

    return out



print(retrace_solutions(kb, [Add],8.0,'f8'))

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


