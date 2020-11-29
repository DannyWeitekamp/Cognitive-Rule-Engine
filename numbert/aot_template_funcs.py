import numpy as np
from numba import types, njit, jit, prange
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, DictType, unicode_type, Array, Tuple, NamedTuple

@njit(cache=True,fastmath=True,nogil=True)
def declare(history, x):
   dec_record = history[-1][0][0]
   _,_,_,_, vmap = dec_record
   if(x not in vmap):
      vmap[x] = len(vmap)


@njit(cache=True)
def declare_nb_objects(history, nb_objects):
    dec_record = history[-1][0][0]
    _,_,_,_, vmap = dec_record
    for uid,obj in nb_objects.items():
        if(obj not in vmap):
            vmap[obj] = len(vmap)


@njit(cache=True,fastmath=True,nogil=True)
def join_u_vds(u_vds,records,depth):
   for _,_,_,_, vmap in records[depth]:
      for x in vmap:
         pd = u_vds.get(x,100000)
         if(pd > depth):
            u_vds[x] = depth

@njit(nogil=True,fastmath=True,cache=True) 
def array_from_dict(d):
   out = np.empty(len(d))
   for i,v in enumerate(d):
      out[i] = v
   return out

@njit(nogil=True,fastmath=True,cache=True) 
def list_from_dict(d):
   out = List()
   for i,v in enumerate(d):
      out.append(v)
   return out


@njit(cache=True,fastmath=True,nogil=True)
def make_contiguous_f8(history,depth):
   u_vds,_,_,records = history
   for _,_,_,_, vmap in records[depth]:
      for x in vmap:
         pd = u_vds.get(x,100000)
         if(pd > depth):
            u_vds[x] = depth

   u_vs = np.empty(len(u_vds))
   for i,v in enumerate(u_vds):
      u_vs[i] = v
   dec_u_vs = u_vs.copy()
   return (u_vds,u_vs,dec_u_vs,records)


@njit(cache=True,fastmath=True,nogil=True)
def make_contiguous(history,depth):
   u_vds,_,_,records = history
   for _,_,_,_, vmap in records[depth]:
      for x in vmap:
         pd = u_vds.get(x,100000)
         if(pd > depth):
            u_vds[x] = depth

   u_vs = List()
   for i,v in enumerate(u_vds):
      u_vs.append(v)
   dec_u_vs = u_vs.copy()
   return (u_vds,u_vs,dec_u_vs,records)

@njit
def insert_record(history, depth, op_uid, arg_types, btsr_flat, btsr_shape, vmap):
   _,_,_,records = history
   # btsr_flat = btsr.reshape(-1)
   # btsr_shape = np.array(btsr.shape,np.int64)
   arg_types = arg_types
   r_d = records.get(depth, List.empty_list())
   r_d.append((op_uid, btsr_flat, btsr_shape, arg_types, vmap)) 
   records[depth] = r_d


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


HE = Tuple([i8,i8[::1]])

i8_i8_dict = DictType(i8,i8)
i8_arr = i8[:]
@njit(nogil=True,fastmath=True,cache=True, locals={"arg_ind":i8[::1],"arg_uids":i8[::1]}) 
def backtrace_goals(goals, history, hist_elems, max_depth, max_solutions=1):
    '''Takes a set of goal values (float,str, or custom object) and a history
        object for that type. Appends hist_elem tuples (op_uid, arg_inds) into 
        hist_elems. Returns a DictType(unicode_type,i8[:]) keyed by type strings with 
        values an array of argument indicies that will pick out new subgoals in the next
        backtrace step.
    '''
    u_vds,_,_,records = history
    
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
        # print("SHALLOW MAX",shallowest_depth,max_depth, goal)
        for depth in range(shallowest_depth,max_depth+1):
            # print("depth:",depth)
        # depth = shallowest_depth

        #If the goal was declared (i.e. it is present at depth 0) then
        #    make a no-op history element for it
            if(depth == 0):
                _,_,_,_, vmap = records[0][0]
                if(goal in vmap):
                    arg_ind = np.array([vmap[goal]],dtype=np.int64)
                    hist_elems_k.append((0,arg_ind))

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
                            hist_elems_k.append((op_uid,arg_uids))
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

@njit(nogil=True,fastmath=True,cache=True) 
def backtrace_selection(sel,history,hist_elems,max_depth, max_solutions=1):
   _,u_vs,_,_ = history
   goals = List()
   for s in sel:
      goals.append(u_vs[s])
   return backtrace_goals(goals,history,hist_elems,max_depth,max_solutions=max_solutions)


@njit(nogil=True,fastmath=True,cache=True) 
def backtrace_selection_f8(sel,history,hist_elems,max_depth, max_solutions=1):
   _,u_vs,_,_ = history
   goals = u_vs[sel]
   return backtrace_goals(goals,history,hist_elems,max_depth,max_solutions=max_solutions)


# @cc.export('insert_record',(hist_type,i8,i8, i8[::1], i8[::1], ListType(unicode_type), DictType(NB_TextField,i8)))
@jit(nogil=True, fastmath=True, cache=True)
def insert_record(record_type,
                  #End Template Args
                  history, depth, op_uid, btsr_flat, btsr_shape, arg_types, vmap):
   _,_,_,records = history
   r_d = records.get(depth, List.empty_list(record_type))
   r_d.append((op_uid, btsr_flat, btsr_shape, arg_types, vmap))
   records[depth] = r_d

@njit(cache=True,fastmath=True,nogil=True)
def empty_inf_history(typ,NB_Type,record_type, record_list_type  #End Template Args
   ):
   records = Dict.empty(i8,record_list_type)
   records[0] = List.empty_list(record_type)
   tl = List.empty_list(unicode_type);tl.append(typ);
   vmap = Dict.empty(NB_Type,i8)
   records[0].append(
      (0, np.empty((0,),dtype=np.int64),
      np.empty((0,),dtype=np.int64),
      tl,vmap))
   u_vds = Dict.empty(NB_Type,i8)
   u_vs = List.empty_list(NB_Type)
   dec_u_vs = List.empty_list(NB_Type)
   return (u_vds,u_vs,dec_u_vs,records)


def gen_inf_history_aot_funcs(cc,typ,NB_Type):
   record_type = Tuple([i8, i8[::1], i8[::1],
   ListType(unicode_type),
   DictType(NB_Type,i8)])
   record_list_type = ListType(record_type)
   hist_type  = Tuple((
      DictType(NB_Type,i8),
      ListType(NB_Type),
      ListType(NB_Type),
      DictType(i8,record_list_type)
   ))

   cc.export('inf_declare',(hist_type,NB_Type))(declare)
   cc.export('inf_declare_nb_objects',(hist_type,DictType(unicode_type,NB_Type)))(declare_nb_objects)
   cc.export('make_contiguous',hist_type(hist_type,i8))(make_contiguous)
   cc.export('backtrace_goals',DictType(unicode_type,i8[:])(ListType(NB_Type),hist_type,ListType(ListType(HE)),i8,i8))(backtrace_goals)
   cc.export('backtrace_selection',DictType(unicode_type,i8[:])(i8[:],hist_type,ListType(ListType(HE)),i8,i8))(backtrace_selection)

   import numbert.aot_template_funcs
   @cc.export('insert_record',(hist_type,i8,i8, i8[::1], i8[::1], ListType(unicode_type), DictType(NB_Type,i8)))
   @jit(nogil=True, fastmath=True, cache=True)
   def _insert_record(history, depth, op_uid, btsr_flat, btsr_shape, arg_types, vmap):
      return insert_record(record_type, history, depth, op_uid, btsr_flat, btsr_shape, arg_types, vmap)

   @cc.export('empty_inf_history',hist_type())
   def _empty_inf_history():
      return empty_inf_history(typ,NB_Type,record_type,record_list_type)
   



  


# @njit(cache=True,fastmath=True,nogil=True)
# def empty_inf_history(typ):
#    records = Dict()
#    records[0] = List()
#    tl = List.empty_list(unicode_type);tl.append(typ);
#    vmap = Dict()
#    records[0].append(
#       (0, np.empty((0,),dtype=np.int64),
#       np.empty((0,),dtype=np.int64),
#       tl,vmap))
#    u_vds = Dict()
#    u_vs = List.empty_list(unicode_type)
#    dec_u_vs = List.empty_list(unicode_type)
#    return (u_vds,u_vs,dec_u_vs,records)

# @njit(cache=True,fastmath=True,nogil=True)
# def empty_inf_history_f8(typ):
#    records = Dict()
#    records[0] = List()
#    tl = List.empty_list(unicode_type);tl.append(typ);
#    vmap = Dict()
#    records[0].append(
#       (0, np.empty((0,),dtype=np.int64),
#       np.empty((0,),dtype=np.int64),
#       tl,vmap))
#    u_vds = Dict()
#    u_vs = np.empty(0)
#    dec_u_vs = np.empty(0)
#    return (u_vds,u_vs,dec_u_vs,records)
