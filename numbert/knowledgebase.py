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
import itertools
import math
from pprint import pprint
from numbert.utils import cache_safe_exec
from numbert.numbalizer import infer_type, infer_nb_type
from collections import namedtuple
from numbert.core import TYPE_ALIASES, numba_type_map, py_type_map, REGISTERED_TYPES, JITSTRUCTS
from numbert.caching import unique_hash, source_to_cache, import_from_cached, \
                             source_in_cache, gen_import_str
from numbert.gensource import assert_gen_source
from numbert.operator import BaseOperator, BaseOperatorMeta, Var, OperatorComposition






@njit(nogil=True,fastmath=True,cache=True) 
def join_new_vals(vd,new_ds,depth):
    for d in new_ds:
        for v in d:
            if(v not in vd):
                vd[v] = depth
    return vd

# @njit(nogil=True,fastmath=True,cache=True) 
def array_from_dict(d):
    out = np.empty(len(d))
    for i,v in enumerate(d):
        out[i] = v
    return out

# @njit(nogil=True,fastmath=True,cache=True) 
def list_from_dict(d):
    out = List.empty_list(d._dict_type.key[0])
    for i,v in enumerate(d):
        out.append(v)
    return out


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
            'declare', 'declare_nb_objects', 'make_contiguous', 
            'empty_inf_history', 'insert_record', 'backtrace_goals',
            'backtrace_selection'
            ],'InfHistory').values()

        out2 = [None]# out2 = import_from_cached(typ,hash_code,['record_type']).values()
        declare, declare_nb_objects, make_contiguous, empty_inf_history, insert_record, \
         backtrace_goals, backtrace_selection, record_type  = tuple([*out1,*out2])

        # from numbert.aot_template_funcs import backtrace_goals, backtrace_selection
        self.history = empty_inf_history()
        self._declare = declare
        self._declare_nb_objects = declare_nb_objects
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
            # st = time.time()
            self.history = self._make_contiguous(self.history,0)     
            # print("_process_declared: ",time.time()-st)
        self.declared_processed = True

    def make_contiguous(self,depth):
        self.history = self._make_contiguous(self.history,depth)     

    def declare(self,x):
        self._declare(self.history,x)
        self.declared_processed = False

    def declare_nb_objects(self,nb_objects):
        self._declare_nb_objects(self.history,nb_objects)
        self.declared_processed = False        

    def insert_record(self,depth,op, btsr, vmap):
        btsr_flat = btsr.reshape(-1)
        btsr_shape = np.array(btsr.shape,dtype=np.int64)
        self._insert_record(self.history,depth,op.uid,btsr_flat,btsr_shape,List(op.arg_types),vmap)

    def backtrace_goals(self, goals, hist_elems, max_depth, max_solutions=1):
        return self._backtrace_goals(goals, self.history, hist_elems, max_depth, max_solutions)


    def backtrace_selection(self, sel,hist_elems, max_depth, max_solutions=1):
        return self._backtrace_selection(sel, self.history, hist_elems, max_depth, max_solutions)




class NBRT_KnowledgeBase(object):
    def __init__(self):
        self.inf_histories = {}
        self.curr_infer_depth = 0
        self.declared_processed = False

    def get_inf_history(self,typ=None,x=None,force_regen=False):
        if(typ is None):
            if(x is None): raise ValueError("typ and x cannot both be None")
            if(isinstance(x,Dict)): 
                print("ALOOOHA", repr(x))
                #TODO: Need to extract type name from Dict
                raise NotImplementedError()
            else:
                typ = TYPE_ALIASES[infer_type(x)]
        if(typ not in self.inf_histories):
            if(hasattr(x,'hash')):
                hash_code = x.hash
            elif(typ in JITSTRUCTS):
                hash_code = JITSTRUCTS[typ].hash
            else:
                hash_code = unique_hash([typ])
            # hash_code = x.hash if x and  else 
            self.inf_histories[typ] = InferenceHistory(typ,hash_code)
            
        return self.inf_histories[typ]


    def _assert_declared_processed(self,types=None,force_regen=False):
        if(not self.declared_processed):
            for typ in (types if types else REGISTERED_TYPES.keys()):
                # st = time.time()
                inf_hist = self.get_inf_history(typ,force_regen=force_regen)
                inf_hist.assert_declared_processed()

    

    def declare(self,x,typ=None):
        '''Takes a whole state conforming to the format output by Numbalizer.state_to_nb_objects()
           or individual items
        '''
        if(isinstance(x,dict)):
            # raise NotImplementedError()
            # pass #TODO: need to rethink
            for typ, nb_objects_of_type in x.items():
                assert typ in REGISTERED_TYPES, "Type is not registered %r." % typ
                inf_hist = self.get_inf_history(typ=typ,x=nb_objects_of_type)
                inf_hist.declare_nb_objects(nb_objects_of_type)
        else:
            inf_hist = self.get_inf_history(typ=typ,x=x)
            inf_hist.declare(x)

    def _assert_record_type(self,typ):
        if(typ not in self.hist_structs):
            typ_cls = REGISTERED_TYPES[typ]


            #Type : (op_id, _hist, shape, arg_types, vmap)
            struct_typ = self.hist_structs[typ] = Tuple([i8,
                                         i8[::1], i8[::1], ListType(unicode_type),
                                         DictType(typ_cls,i8)])
            self.hists[typ] = self.hists.get(typ,Dict.empty(i8,ListType(struct_typ)))
        return self.hist_structs[typ]

    # def _assert_declare_store(self,typ):
    #   print('\t',"%.02f"%(time.time()-start_time),"ROOP1")
    #   struct_typ = self._assert_record_type(typ)
    #   print('\t',"%.02f"%(time.time()-start_time),"ROOP2")
    #   typ_store = self.hists[typ]
    #   if(0 not in typ_store):
    #       typ_cls = REGISTERED_TYPES[typ]
    #       tsd = typ_store[0] = typ_store.get(0, List.empty_list(struct_typ))
    #       tl = List();tl.append(typ);
    #       vmap = Dict.empty(typ_cls,i8)
    #       #Type : (0 (i.e. no-op), _hist, shape, arg_types, vmap)
    #       tsd.append( tuple([0, np.empty((0,),dtype=np.int64),
    #                  np.empty((0,),dtype=np.int64), tl,vmap]) )
    #   print('\t',"%.02f"%(time.time()-start_time),"ROOP3")


    # def _assert_declared_values(self):
    #   if(not self.declared_consistent):
    #       for typ in REGISTERED_TYPES.keys():
    #           print("%.02f"%(time.time()-start_time),"NOOP1")
    #           self._assert_declare_store(typ)
    #           print("%.02f"%(time.time()-start_time),"NOOP2.2")
    #           record = self.hists[typ][0][0]
    #           print("%.02f"%(time.time()-start_time),"NOOP2.3")
    #           _,_,_,_, vmap = record
    #           print("%.02f"%(time.time()-start_time),"NOOP2.5")
    #           typ_cls = REGISTERED_TYPES[typ]
    #           print("%.02f"%(time.time()-start_time),"NOOP2.6")
    #           d = self.u_vds[typ] = Dict.empty(typ_cls,i8)
    #           print("%.02f"%(time.time()-start_time),"NOOP2.7")
    #           for x in vmap:
    #               d[x] = 0
    #           print("%.02f"%(time.time()-start_time),"NOOP3")
    #           if(typ == TYPE_ALIASES['float']):
    #               self.u_vs[typ] = array_from_dict(d)
    #           else:
    #               self.u_vs[typ] = list_from_dict(d)
    #           print("%.02f"%(time.time()-start_time),"NOOP4")

    #           self.dec_u_vs[typ] = self.u_vs[typ].copy()
    #           print("%.02f"%(time.time()-start_time),"NOOP5")
    #           # print(self.u_vs)
    #       self.declared_consistent = True




    # def declare(self,x,typ=None):
    #   '''Takes a whole state conforming to the format output by Numbalizer.state_to_nb_objects()
    #      or individual items
    #   '''
    #   if(isinstance(x,dict)):
    #       for typ, nb_objects_of_type in x.items():
    #           assert typ in REGISTERED_TYPES, "Type is not registered %r." % typ
    #           self._assert_declare_store(typ)
    #           declare_nb_objects(self.hists[typ][0][0],nb_objects_of_type)
    #   else:
    #       if(typ is None): typ = TYPE_ALIASES[infer_type(x)]
    #       self._assert_declare_store(typ)
    #       record = self.hists[typ][0][0]
    #       _,_,_,_, vmap = record

    #       if(x not in vmap):
    #           vmap[x] = len(vmap)
    #   self.hist_consistent = False
    #   self.declared_consistent = False

    def how_search(self,ops,goal,search_depth=1,max_solutions=1):
        return how_search(self,ops,goal,search_depth=search_depth,max_solutions=max_solutions)

    def unify_op(self,op,goal):
        return unify_op(self,op,goal)

    def check_produce_goal(self,ops,goal):
        return check_produce_goal(self,ops,goal)

    def forward(self,ops):
        forward(self,ops)

    
        
@njit(cache=True)
def declare_nb_objects(dec_record, nb_objects):
    _,_,_,_, vmap = dec_record
    for uid,obj in nb_objects.items():
        if(obj not in vmap):
            vmap[obj] = len(vmap)



# @njit(nogil=True,fastmath=True,parallel=False) 

def insert_record(kb,depth,op, btsr, vmap):
    # print('is')
    typ = op.out_type
    struct_typ = kb._assert_record_type(typ)
    # if(typ not in kb.hist_structs):
    #   typ_cls = kb.REGISTERED_TYPES[typ]
    #   kb.hist_structs[typ] = Tuple([i8,
    #                                i8[::1], i8[::1], ListType(unicode_type),
    #                                DictType(typ_cls,i8)])

    typ_store = kb.hists[typ] = kb.hists.get(typ,Dict.empty(i8,ListType(struct_typ)))
    tsd = typ_store[depth] = typ_store.get(depth, List.empty_list(struct_typ))
    tsd.append(tuple([op.uid,
                      btsr.reshape(-1), np.array(btsr.shape,np.int64), List(op.arg_types),
                      vmap]))
    # print('istop')
    return tsd

# @njit(cache=True):
# def extract_vmaps():

def broadcast_forward_op_comp(kb,op_comp):
    if(op_comp.out_type == None): raise ValueError("Only typed outputs work with this function.")


    arg_sets = [kb.get_inf_history(t).u_vs if t in kb.inf_histories else [] for t in op_comp.arg_types]
    # arg_sets = [kb.u_vs.get(t,[]) for t in op_comp.arg_types]
    lengths = tuple([len(x) for x in arg_sets])
    out = np.empty(lengths,dtype=np.int64)
    d = Dict.empty(numba_type_map[op_comp.out_type],i8)
    arg_ind_combinations = itertools.product(*[np.arange(l) for l in lengths])
    uid = 1
    for arg_inds in arg_ind_combinations:
        try:
            v = op_comp(*[arg_set[i] for i,arg_set in zip(arg_inds,arg_sets)])
            if(v not in d):
                d[v] = uid; uid += 1;
            out[tuple(arg_inds)] = d[v]
        except ValueError:
            out[tuple(arg_inds)] = 0
    return out, d





# Add.broadcast_forward = Add_forward
# Subtract.broadcast_forward = Subtract_forward
# Concatenate.broadcast_forward = cat_forward
# import time
# start_time = time.time()
# def forward(kb,ops):
#   print("%.02f"%(time.time()-start_time),"BEFORE")
#   kb._assert_declared_values()
#   print("%.02f"%(time.time()-start_time),"AFTER")

#   output_types = set()
#   # output_types = set([op.out_type for op in ops])
#   new_records = {typ:[] for typ in output_types}
#   depth = kb.curr_infer_depth = kb.curr_infer_depth+1
    
#   for op in ops:
#       print("%.02f"%(time.time()-start_time),"SHLOOP1", op)
#       if(not all([t in kb.u_vs for t in op.arg_types])): continue
#       typ = op.out_type
#       if(isinstance(op,BaseOperatorMeta)):
#           args = [kb.u_vs[t] for t in op.u_arg_types]
#           btsr, vmap = op.broadcast_forward(*args)
#       elif(isinstance(op,OperatorComposition)):
#           btsr, vmap = broadcast_forward_op_comp(kb,op)

#       records = insert_record(kb, depth, op, btsr, vmap)
#       new_records[typ] = records
#       output_types.add(op.out_type)
        
#   for typ in output_types:
#       if(typ in new_records):
#           vmaps = List([rec[4] for rec in new_records[typ]])
#           kb.u_vds[typ] = join_new_vals(kb.u_vds[typ],vmaps,depth)

#           if(typ == TYPE_ALIASES['float']):
#               kb.u_vs[typ] = array_from_dict(kb.u_vds[typ])
#           else:
#               kb.u_vs[typ] = list_from_dict(kb.u_vds[typ])
    # print("F_end")
def forward(kb,ops):
    kb._assert_declared_processed()

    output_types = set()
    new_records = {typ:[] for typ in output_types}
    depth = kb.curr_infer_depth = kb.curr_infer_depth+1
    
    for op in ops:
        # print(op)
        if(not all([t in kb.inf_histories for t in op.arg_types])): continue
        if(isinstance(op,BaseOperatorMeta)):
            args = [kb.inf_histories[t].u_vs for t in op.u_arg_types]
            # if(kb.curr_infer_depth == 1):
                # print(args)
            # print('shape',[x.shape if hasattr(x,'shape') else None for x in args], kb.curr_infer_depth)
            btsr, vmap = op.broadcast_forward(*args)
        elif(isinstance(op,OperatorComposition)):
            btsr, vmap = broadcast_forward_op_comp(kb,op)
        kb.get_inf_history(op.out_type).insert_record(depth, op, btsr, vmap)
        output_types.add(op.out_type)
    for typ in output_types:
        kb.get_inf_history(typ).make_contiguous(depth)
    # for typ in kb.inf_histories:
    # h = kb.get_inf_history("TF").history[3]
    # print('len(h)',kb.get_inf_history("TF").history)
    # if(len(h) > 1):
    # print([x for x in h.values()])
    # print(typ,kb.curr_infer_depth)
    # print(kb.get_inf_history(typ).history[1])
        # print()



# HE_deffered = deferred_type()
# @jitclass([('op_uid', i8),
#          ('args', i8[:])])
# class HistElm(object):
#   def __init__(self,op_uid,args):
#       self.op_uid = op_uid
#       self.args = args
#   # def __repr__(self):
#   #   return str(self.op_uid) + ":" + str(self.args) 
# HE = HistElm.class_type.instance_type
# HE_deffered.define(HE)


from numbert.aot_template_funcs import backtrace_goals, HE
he_list = ListType(HE)
@njit(cache=True)
def HistElmListList():
    return List.empty_list(he_list)

def _retrace_goal_history(kb,ops,goal,g_typ, max_solutions):
    '''Backtraces the operators that produced the intermediate values at each
      infer depth leading up to the goal value. Output is a list of dictionaries
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

    arg_inds = h.backtrace_goals(goals, hist_elems, max_depth=kb.curr_infer_depth, max_solutions=max_solutions)

    out = [{g_typ: hist_elems}]
    i = 1
    while(True):
        nxt = {}
        new_arg_inds = None
        for typ in arg_inds:
            hist_elems = HistElmListList()
            
            # print('ai', arg_inds)
            h = kb.get_inf_history(typ)
            typ_new_inds = h.backtrace_selection(arg_inds[typ],hist_elems, kb.curr_infer_depth-i, max_solutions=max_solutions)
            # print('tni', typ_new_inds)
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
    # print(out)
    return list(reversed(out))

def retrace_solutions(kb,ops,goal,g_typ,max_solutions=1000):
    ''' Calls _retrace_goal_history() to get hist elements leading up to the 
        production of the goal value, then uses these hist elements to compose
        a set of nested tuples that can be used to instantiate an operator composition
    '''
    # print("RETRACE", goal)
    goal_history = _retrace_goal_history(kb,ops,goal,g_typ,max_solutions)
    # pprint(goal_history)

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
    # print(out)
    return out



#Adapted from here: https://gitter.im/numba/numba?at=5dc1f9d13d669b28a0408463
@njit(nogil=True,fastmath=True,cache=True) 
def unravel_indicies(indicies, shape):
    sizes = np.zeros(len(shape), dtype=np.int64)
    result = np.zeros(len(shape), dtype=np.int64)
    sizes[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * shape[i + 1]

    out = np.empty((len(indicies),len(shape)), dtype=np.int64)
    for j,index in enumerate(indicies):
        remainder = index
        for i in range(len(shape)):
            out[j,i] = remainder // sizes[i]
            remainder %= sizes[i]
    return out


i8_i8_dict = DictType(i8,i8)
i8_arr = i8[:]
@njit(nogil=True,fastmath=True,cache=True, locals={"arg_ind":i8[::1],"arg_uids":i8[::1]}) 
def retrace_back_one(goals, records, u_vds, hist_elems, max_depth, max_solutions):
    unq_arg_inds = Dict.empty(unicode_type, i8_i8_dict)
    pos_by_typ = Dict.empty(unicode_type, i8)

    solution_quota = float(max_solutions-len(goals)) if max_solutions else np.inf
    #Go through each goal in goals, and find applications of operations 
    #   which resulted in each subgoal. Add the indicies of args of each 
    #   goal satisficing operation application to the set of subgoals for 
    #   the next iteration.
    for goal in goals:
        n_goal_solutions = 0
        hist_elems_k = List.empty_list(HE)
        hist_elems.append(hist_elems_k)


        #Determine the shallowest infer depth where the goal was encountered
        shallowest_depth = u_vds[goal]
        # print("SHALLOW MAX",shallowest_depth,max_depth)
        for depth in range(shallowest_depth,max_depth+1):
            # print("depth:",depth)
        # depth = shallowest_depth

        #If the goal was declared (i.e. it is present at depth 0) then
        #    make a no-op history element for it
            if(depth == 0):
                _,_,_,_, vmap = records[0][0]
                if(goal in vmap):
                    arg_ind = np.array([vmap[goal]],dtype=np.int64)
                    hist_elems_k.append(HistElm(0,arg_ind))

            #Otherwise the goal was infered from the declared values
            else:
                #For every record (i.e. history of an inference with a particular op) 
                if(depth >= len(records)): continue
                _records = records[depth]
                for record in _records:
                    needs_more_solutions = True
                    op_uid, _hist, shape, arg_types, vmap = record

                    #Make a dictionary for each type to collect unique arg values
                    for typ in arg_types:
                        if(typ not in unq_arg_inds):
                            unq_arg_inds[typ] = Dict.empty(i8,i8)
                            pos_by_typ[typ] = 0

                    #If the record shows that the goal was produced by the op associated
                    #   with record. 
                    if(goal in vmap):
                        #Then find any set of arguments used to produce it
                        wher = np.where(_hist == vmap[goal])[0]
                        inds = unravel_indicies(wher,shape)

                        
                        #For every such combination of arguments
                        for i in range(inds.shape[0]):
                            #Build a mapping from each argument's index to a unique id
                            arg_uids = np.empty(inds.shape[1],np.int64)
                            for j in range(inds.shape[1]):
                                d = unq_arg_inds[arg_types[j]]
                                v = inds[i,j]
                                if(v not in d):
                                    d[v] = pos_by_typ[typ]
                                    pos_by_typ[typ] += 1
                                arg_uids[j] = d[v]
                            
                            #Redundant Arguments not allowed 
                            # if(len(np.unique(arg_uids)) == len(arg_uids)):
                                #Store the op_uid and argument unique ids in a HistElm
                            hist_elems_k.append(HistElm(op_uid,arg_uids))
                            n_goal_solutions += 1
                            if(n_goal_solutions >= 1 and solution_quota <= 0):
                                needs_more_solutions = False
                                break
                    if(not needs_more_solutions): break


    #Consolidate the dictionaries of unique arg indicies into arrays.
    #   These will be used with select_from_collection to slice out goals
    #   for the next iteration.
    out_arg_inds = Dict.empty(unicode_type,i8_arr)
    for typ in unq_arg_inds:
        u_vals = out_arg_inds[typ] = np.empty((len(unq_arg_inds[typ])),np.int64)
        for i, v in enumerate(unq_arg_inds[typ]):
            u_vals[i] = v

    return out_arg_inds



def _infer_goal_type(goal):
    if(isinstance(goal, (int,float))):
        return TYPE_ALIASES['float']
    elif(isinstance(goal, (str))):
        return TYPE_ALIASES['string']
    elif(type(goal).__name__ in REGISTERED_TYPES):
        return TYPE_ALIASES[type(goal).__name__]
    else:
        ValueError("Goal type is not registered: %s" % type(goal))



def unify_op(kb,op,goal):
    g_typ = _infer_goal_type(goal)
    kb._assert_declared_processed()
    #Handle Copy/No-op right up front
    if(isinstance(op,OperatorComposition) and op.depth == 0):
        if(g_typ in kb.inf_histories):
            h = kb.get_inf_history(g_typ)
            if(goal in h.u_vds):
                return [[goal]]
        else:
            return []
    if(op.out_type != g_typ): return []
    
    if(not all([t in kb.inf_histories for t in op.arg_types])):return []

    if(isinstance(op,BaseOperatorMeta)):
        args = [kb.get_inf_history(t).u_vs for t in op.u_arg_types]
        _hist, vmap = op.broadcast_forward(*args)
    elif(isinstance(op,OperatorComposition)):
        _hist, vmap = broadcast_forward_op_comp(kb,op)
        
    arg_sets = [kb.get_inf_history(t).u_vs if t in kb.inf_histories else [] for t in op.arg_types]
    if(goal in vmap):
        inds = np.stack(np.where(_hist == vmap[goal])).T

        return [[arg_sets[i][j] for i,j in enumerate(ind)] for ind in inds]
    return []


def how_search(kb,ops,goal,search_depth=1,max_solutions=10,min_stop_depth=-1):
    if(min_stop_depth == -1): min_stop_depth = search_depth
    # print("%.02f"%(time.time()-start_time),"BOOP1")
    kb._assert_declared_processed()
    # print("%.02f"%(time.time()-start_time),"BOOP2.5")
    g_typ = _infer_goal_type(goal)
    # print(g_typ)
    # print("%.02f"%(time.time()-start_time),"BOOP2")
    for depth in range(1,search_depth+1):
        # print("depth:",depth, "/", search_depth,kb.curr_infer_depth)
        if(depth < kb.curr_infer_depth): continue
    # while():
        
        if(g_typ in kb.inf_histories and kb.curr_infer_depth > min_stop_depth):
            h = kb.get_inf_history(g_typ)
            if((goal in h.u_vds)):
                break
        
        # print("DEPTH",depth)
        # for k,v in kb.inf_histories.items():
        #     print(k)
        #     print(v.history[1])
        # print([(k,v.history) for ])
        forward(kb,ops)
        
        # print(kb.u_vds[g_typ])
        # print(kb.hists)



    if(g_typ in kb.inf_histories):
        h = kb.get_inf_history(g_typ)
        # print(goal, h.u_vds)
        if(goal in h.u_vds):
            return retrace_solutions(kb,ops,goal,g_typ,max_solutions=max_solutions)
    return []
