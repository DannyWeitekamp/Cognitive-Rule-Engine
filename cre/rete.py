import numpy as np
from numba import types, njit, i8, u8, i4, u1, u2, i8, f8, literally, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple
from numba.experimental.structref import new, define_boxing
import numba.experimental.structref as structref
from cre.utils import (_dict_from_ptr, _pointer_from_struct,
         _pointer_from_struct_incref, _struct_from_pointer, decode_idrec,
        encode_idrec, deref_type, OFFSET_TYPE_ATTR, OFFSET_TYPE_LIST,
         _pointer_to_data_pointer, _list_base_from_ptr, _load_pointer,
         _decref_structref, _decref_pointer)
from cre.structref import define_structref
from cre.memory import MemoryType
from cre.vector import VectorType
from cre.var import GenericVarType
from cre.op import GenericOpType
from cre.conditions import LiteralType, build_distributed_dnf
from cre.vector import VectorType, new_vector

RETRACT = u1(0xFF)# u1(0)
DECLARE = u1(0)

deref_record_field_dict = {
    # Pointers to the Dict(i8,u1)s inside deref_depends
    "parent_ptrs" : i8[::1], 
    "arg_ind" : i8,
    "base_idrec" : u8,
    "was_successful" : u1,
}

DerefRecord, DerefRecordType = define_structref("DerefRecord", deref_record_field_dict)



node_memory_field_dict = {
    # If it is a root then we should
    # check the facts of the mem linked to the graph 
    "is_root" : types.boolean,

    #
    "change_set" : DictType(u8,u1), 
    # Maps f_ids -> sets of f_ids
    "matches" : DictType(u8, i8), 

}

NodeMemory, NodeMemoryType = define_structref("NodeMemory", node_memory_field_dict)

@njit(cache=True)
def new_node_mem():
    st = new(NodeMemoryType)
    st.change_set = Dict.empty(u8,u1)
    st.matches = Dict.empty(u8,i8)
    st.is_root = False
    return st

@njit(cache=True)
def new_root_node_mem():
    st = new(NodeMemoryType)
    st.change_set = Dict.empty(u8,u1)
    st.is_root = True
    return st





dict_i8_u1_type = DictType(i8,u1)
dict_u8_i8_arr_type = DictType(u8,i8[::1])
deref_dep_typ = DictType(u8,DictType(i8,u1))


# outputs_item_typ = DictType(u8,DictType(u8,u1))
dict_u8_u1_type = DictType(u8,u1)
outputs_typ = DictType(u8,DictType(u8,u1))

base_rete_node_field_dict = {
    # Pointers to the Dict(i8,u1)s inside deref_depends
    "mem" : MemoryType, 
    "lit" : LiteralType,
    "op" : GenericOpType,
    "deref_depends" : deref_dep_typ, 
    "relevant_global_diffs" : VectorType,

    "var_inds" : i8[::1],
    "t_ids" : u2[::1],
    # "vars" : ListType(GenericVarType),
    "head_ptrs" : ListType(DictType(u8,i8[::1])),
    "inputs" : ListType(NodeMemoryType),
    "outputs" : ListType(NodeMemoryType), #DictType(u8,DictType(u8,u1))
}

BaseReteNode, BaseReteNodeType = define_structref("BaseReteNode", base_rete_node_field_dict, define_constructor=False)


i8_arr_typ = i8[::1]

@njit(cache=True)
def node_ctor(mem, lit, t_ids, var_inds):
    st = new(BaseReteNodeType)
    st.mem = mem
    st.deref_depends = Dict.empty(u8,dict_i8_u1_type)
    st.relevant_global_diffs = new_vector(4)
    st.var_inds = var_inds
    st.t_ids = t_ids
    st.lit = lit
    st.op = op = lit.op
    # st.vars = _vars

    head_ptrs = List.empty_list(dict_u8_i8_arr_type)
    for i in range(len(op.base_var_map)):
        head_ptrs.append(Dict.empty(u8, i8_arr_typ))
    st.head_ptrs = head_ptrs

    outputs = List.empty_list(NodeMemoryType)
    for i in range(len(op.base_var_map)):
        # outputs.append(Dict.empty(u8,dict_u8_u1_type))
        outputs.append(new_node_mem())

    st.outputs = outputs

    return st


@njit(Tuple((i8,u8[::1]))(i8, deref_type[::1]), cache=True,locals={"data_ptr":i8, "inst_ptr":i8})
def deref_head_and_relevant_idrecs(inst_ptr, deref_offsets):
    ''' '''
    relevant_idrecs = np.zeros((len(deref_offsets)-1)*2+1, dtype=np.uint64)
    k = -1

    ok = True
    for deref in deref_offsets[:-1]:
        if(inst_ptr == 0): ok = False; break;
        if(deref.type == u1(OFFSET_TYPE_ATTR)):
            data_ptr = _pointer_to_data_pointer(inst_ptr)
        else:
            data_ptr = _list_base_from_ptr(inst_ptr)
        t_id, f_id, _ = decode_idrec(u8(_load_pointer(u8, data_ptr)))
        if(k >= 0):
            relevant_idrecs[k] = encode_idrec(t_id, f_id, RETRACT);
        k += 1
        relevant_idrecs[k] = encode_idrec(t_id, f_id, deref.a_id); k += 1

        inst_ptr = _load_pointer(i8, data_ptr+deref.offset)
        
    if(inst_ptr != 0):
        deref = deref_offsets[-1]
        if(deref.type == u1(OFFSET_TYPE_ATTR)):
            data_ptr = _pointer_to_data_pointer(inst_ptr)
        else:
            data_ptr = i8(_list_base_from_ptr(inst_ptr))
        t_id, f_id, _ = decode_idrec(u8(_load_pointer(u8, data_ptr)))
        relevant_idrecs[k] = encode_idrec(t_id, f_id, RETRACT); k += 1
        relevant_idrecs[k] = encode_idrec(t_id, f_id, deref.a_id); k += 1

        head_ptr = data_ptr+deref.offset
        return head_ptr, relevant_idrecs
    else:
        return 0, relevant_idrecs[:k]


#Example Deref a.B.B.B.A
# dep_idrecs = [(1, MOD[a.B]), (2, RETRACT[a.B]),
#               (2, MOD[a.B.B]), (3, RETRACT[a.B.B]),
#                (3, MOD[a.B.B.B]), (4, RETRACT[a.B.B.B]),
#                (4, MOD[a.B.B.B.A])
#                     ]



@njit(cache=True)
def invalidate_deref_rec(rec):
    r_ptr = _pointer_from_struct(rec)
    for ptr in rec.parent_ptrs:
        parent = _dict_from_ptr(dict_i8_u1_type, ptr)
        del parent[r_ptr]
        _decref_structref(rec)

@njit(i8(deref_dep_typ, u8, i8),cache=True)
def make_deref_record_parent(deref_depends, idrec, r_ptr):
    p = deref_depends.get(idrec,None)
    if(p is None): 
        p = deref_depends[idrec] = Dict.empty(i8,u1)
    p[r_ptr] = u1(1)
    return _pointer_from_struct(p)

# @njit(i8(BaseReteNodeType, u2, u8, deref_type[::1]),cache=True)
@njit(cache=True)
def validate_deref(self, arg_ind, base_t_id, f_id, deref_offsets):
    '''Try to get the head_ptr of 'f_id' in input 'k'. Inject a DerefRecord regardless of the result '''
    # t_id = self.t_ids[k]
    facts = _struct_from_pointer(VectorType, self.mem.mem_data.facts[base_t_id])
    base_ptr = facts.data[f_id]
    # deref_offsets = self.vars[k].deref_offsets
    head_ptr, rel_idrecs  = deref_head_and_relevant_idrecs(base_ptr,deref_offsets)
    was_successful = (head_ptr != 0)
    parent_ptrs = np.empty(len(rel_idrecs), dtype=np.int64)
    r = DerefRecord(parent_ptrs, arg_ind, encode_idrec(base_t_id,f_id,0), was_successful)
    r_ptr = _pointer_from_struct_incref(r)
    for i, idrec in enumerate(rel_idrecs):
        ptr = make_deref_record_parent(self.deref_depends, idrec, r_ptr)
        parent_ptrs[i] = ptr

    return head_ptr

# @njit(void(BaseReteNodeType, i8,u8,u1),locals={"f_id" : u8}, cache=True)
@njit(locals={"f_id" : u8, "a_id" : u8}, cache=True)
def validate_head_or_retract(self, arg_ind, idrec, change_set):
    '''Update the head_ptr dictionaries by following the deref
     chains of DECLARE/MODIFY changes, and make retractions
    for an explicit RETRACT or a failure in the deref chain.'''
    _, f_id, a_id = decode_idrec(idrec)
    if(a_id != RETRACT):
        base_t_id = self.t_ids[arg_ind]
        r = self.op.head_ranges[arg_ind]
        head_ptrs = np.zeros((r.length,),dtype=np.int64)
        okay = True
        # For each head_var try to deref all the way to the head_ptr and put it in head_ptrs
        for i in range(r.length):
            head_var = _struct_from_pointer(GenericVarType,self.op.head_var_ptrs[r.start+i])
            deref_offsets = head_var.deref_offsets
            head_ptr = validate_deref(self, arg_ind, base_t_id, f_id, deref_offsets)
            if(head_ptr == 0): 
                okay=False;
                del change_set[idrec];
                break; 
            head_ptrs[i] = head_ptr

        if(okay):
            self.head_ptrs[arg_ind][f_id] = head_ptrs
            return
            
    # # At this point we are definitely RETRACT
    # to_clear = self.outputs[arg_ind].matches
    # for x in to_clear:
    #     for i in range(len(self.outputs)):
    #         if(i == arg_ind): continue
    #         other_out_matches = self.outputs[i].matches
    #         del other_out_matches[x]

    # this_out_matches = self.outputs[arg_ind].matches
    # del this_out_matches[a_id] 
    # del self.head_ptrs[arg_ind][f_id]  

@njit(cache=True)
def update_changes_deref_dependencies(self, arg_change_sets):
     ### 'relevant_global_diffs' is the set of self.mem.change_queue
    # items relevant to intermediate derefs computed for this literal,
    # and modification of the head attribute. Shouldn't happen frequently ###
    for i in range(self.relevant_global_diffs.head):
        idrec = self.relevant_global_diffs[i]
        if(idrec in self.deref_depends):
            deref_records = self.deref_depends[idrec]

            # Invalidate old DerefRecords
            for r_ptr in deref_records:
                r = _struct_from_pointer(DerefRecordType,r_ptr)
                invalidate_deref_rec(r)

                # Any change in the deref chain counts as a MODIFY
                # TODO: This is for sure wrong
                arg_change_sets[r.arg_ind][r.base_idrec] = u1(1) #MODIFY

@njit(cache=True)
def update_changes_from_inputs(self, arg_change_sets):
    for i, inp in enumerate(self.inputs):
        arg_change_sets_i = arg_change_sets[i]
        for idrec in inp.change_set:
            arg_change_sets_i[idrec] = u1(1)

        # if(len(arg_change_sets_i) > 0):
             
        # else:
        #     arg_change_sets[i] = inp.change_set

@njit(cache=True)
def update_head_ptrs(self, arg_change_sets):
    for i,change_set in enumerate(arg_change_sets):
        for idrec in change_set:
            # _, f_id, a_id = decode_idrec(idrec)
            validate_head_or_retract(self, i, idrec, change_set)


def foo(lengths):
    n = len(lengths)
    iters = np.zeros(n,dtype=np.int64)
    # lens = np.zeros(n,dtype=np.int64)
    i = end = n-1
    while(i >= 0):
        iters[i] += 1
        if(iters[i] >= lengths[i]): 
            i -= 1
            iters[i] = 0
        elif(i != end):
            i += 1
        print(iters)
            
from cre.utils import _func_from_address
match_heads_f_type = types.FunctionType(u1(i8[::1],))

u8_i8_dict_type = DictType(u8,i8)


@njit(cache=True)
def insert_alpha_match(self,f_id):
    was_match = self.outputs[0].matches.get(f_id,0) != 0
    if(not was_match):
        self.outputs[0].change_set[encode_idrec(0,f_id,DECLARE)] = u1(1)
        self.outputs[0].matches[f_id] = 1

@njit(cache=True)
def invalidate_alpha_match(self,f_id):
    was_match = self.outputs[0].matches.get(f_id,0) != 0
    if(was_match):
        self.outputs[0].change_set[encode_idrec(0,f_id,RETRACT)] = u1(1)

@njit(cache=True)
def _ld_dict(ptr):
    return _dict_from_ptr(u8_i8_dict_type, ptr)     

@njit(cache=True,locals={'d0_ptr':i8, 'd1_ptr':i8})
def insert_beta_match(self,match_f_ids):
    f_id0,f_id1 = match_f_ids[0], match_f_ids[1]
    outputs = self.outputs
    was_match = True
    d0_ptr = outputs[0].matches.get(f_id0, 0)
    if(d0_ptr == 0):
        d0 = Dict.empty(u8,i8)
        d0_ptr = _pointer_from_struct_incref(d0)
        outputs[0].matches[f_id0] = d0_ptr
        was_match = False
    else:
        _d0_ptr = d0_ptr
        d0 = _ld_dict(_d0_ptr)
    d1_ptr = outputs[1].matches.get(f_id1, 0)
    if(d1_ptr == 0):
        d1 = Dict.empty(u8,i8)
        d1_ptr = _pointer_from_struct_incref(d1)
        outputs[1].matches[f_id1] = d1_ptr
        was_match = False
    else:
        _d1_ptr = d1_ptr
        d1 = _ld_dict(_d1_ptr)

    d0[f_id1] = 1
    d1[f_id0] = 1

    if(not was_match):
        outputs[0].change_set[encode_idrec(0,f_id0,0)] = u1(1)
        outputs[1].change_set[encode_idrec(0,f_id1,1)] = u1(1)
    return was_match

@njit(cache=True, locals={'ptr0':i8,'ptr1':i8,})
def invalidate_beta_match(self,match_f_ids):
    f_id0,f_id1 = match_f_ids[0], match_f_ids[1]
    outputs = self.outputs
    was_match = False
    ptr0 = outputs[0].matches.get(f_id0,0)
    if(ptr0 != 0):
        d = _ld_dict(ptr0)
        if(d.get(f_id1,0) != 0):
            was_match = True
            d[f_id1] = 0
    ptr1 = outputs[1].matches.get(f_id1,0)
    if(ptr1 != 0):
        d = _ld_dict(ptr1)
        if(d.get(f_id0,0) != 0):
            was_match = True
            d[f_id0] = 0
    return was_match



@njit(cache=True)
def invalidate_f_id(self, arg_ind, f_id, decref=False):
    outputs = self.outputs
    ptr0 = outputs[0].matches.get(f_id0,0)
    if(ptr0 != 0):
        if(decref): _decref_pointer(ptr0)
        outputs[0].matches[f_id0] = 0


@njit(cache=True)
def update_node(self):
    
    arg_change_sets = List.empty_list(dict_u8_u1_type)
    for j in range(len(self.var_inds)):
         arg_change_sets.append(Dict.empty(u8,u1))

    update_changes_deref_dependencies(self, arg_change_sets)
    update_changes_from_inputs(self, arg_change_sets)
    update_head_ptrs(self, arg_change_sets)

    match_head_ptrs = _func_from_address(match_heads_f_type, self.op.match_head_ptrs_addr)

    head_ranges = self.op.head_ranges
    match_inp_ptrs = np.zeros((len(self.op.head_var_ptrs),),dtype=np.int64)
    match_f_ids = np.zeros((len(self.var_inds),),dtype=np.uint64)
    n_vars = len(head_ranges)

    print("---", self.op, "---")

    for i, out in enumerate(self.outputs):
        out.change_set = Dict.empty(u8,u1)
    
    # print(arg_change_sets)
   
    #Note: They way these loops currently work there is some
    for i, change_set in enumerate(arg_change_sets):
        head_ptrs_i = self.head_ptrs[i]
        i_strt, i_len = head_ranges[i][0], head_ranges[i][1]

        if(n_vars > 1):
            j = 1 if i == 0 else 0
            j_strt, j_len = head_ranges[j][0], head_ranges[j][1]        
            for idrec_i in change_set:
                _, f_id_i, a_id_i = decode_idrec(idrec_i)
                match_inp_ptrs[i_strt:i_strt+i_len] = head_ptrs_i[f_id_i]
                match_f_ids[i] = f_id_i
            
            
                for f_id_j, h_ptrs_j in self.head_ptrs[j].items():
                    print(encode_idrec(self.t_ids[j],f_id_j,0), arg_change_sets[j])
                    if(j > i and encode_idrec(0,f_id_j,0) in arg_change_sets[j]):
                        continue
                    match_inp_ptrs[j_strt:j_strt+j_len] = h_ptrs_j
                    match_f_ids[j] = f_id_j



                    is_match = match_head_ptrs(match_inp_ptrs) ^ self.lit.negated
                    was_match = False
                    if(is_match):
                        print("INSER", match_f_ids, self.lit.negated)
                        was_match = insert_beta_match(self, match_f_ids)
                    else:
                        was_match = invalidate_beta_match(self, match_f_ids)
                    # print(f_id_i, f_id_j, is_match, was_match)
        else:
            for idrec_i in change_set:
                _, f_id_i, a_id_i = decode_idrec(idrec_i)
                match_inp_ptrs[i_strt:i_strt+i_len] = head_ptrs_i[f_id_i]

                is_match = match_head_ptrs(match_inp_ptrs) ^ self.lit.negated
                
                if(is_match):
                    insert_alpha_match(self,f_id_i)
                else:
                    invalidate_alpha_match(self,f_id_i)

    print("<<", [x.matches for x in self.outputs])
















# dict_u8_u1_type = DictType(u8,u1)

# @njit(cache=True)
# def filter_beta(self):
#     arg_change_sets = List.empty_list(dict_u8_u1_type)
#     for i in range(len(self.var_inds)):
#          arg_change_sets.append(Dict.empty(u8,u1))

#     update_deref_dependencies(self, arg_change_sets)
#     update_changes_from_inputs(self, arg_change_sets)
   


#     ### Make sure the arg_change_sets are up to date
   

#     # Update the head_ptr dictionaries by following the deref chains of DECLARE/MODIFY 
#     # changes, and make retractions for an explicit RETRACT or a failure in the deref chain.
#     for i,change_set in enumerate(arg_change_sets):
#         for idrec in change_set:
#             _, f_id, a_id = decode_idrec(idrec)
#             validate_head_or_retract(self, i, f_id, a_id)
#     # for idrec0 in arg_change_sets[0].items():
#     #     _,f_id0, a_id0 = 
#     #     validate_head_or_retract(self, f_id0, a_id0)

#     # for f_id1, a_id1 in arg_change_sets[1].items():
#     #     validate_head_or_retract(self, f_id1, a_id1)

#     ### Check all pairs at this point we should only be dealing with DECLARE/MODIFY changes
#     for f_id0, a_id0 in arg_change_sets[0].items():
#         h_ptr0 = self.head_ptrs[0][f_id0]
#         if(a_id0 != RETRACT):
#             for h_ptr1 in self.head_ptrs[1]:
#                 check_pair(h_ptr0, h_ptr1, a_id0)
            
#     for f_id1, a_id1 in arg_change_sets[1].items():
#         if(a_id1 != RETRACT):
#             h_ptr1 = self.head_ptrs[1][f_id1]
#             for f_id0, h_ptr0 in self.head_ptrs[0].items():
#                 if(f_id0 not in arg_change_sets[0]):
#                     check_pair(h_ptr0, h_ptr1, a_id1)

# def check_pair(self, h_ptr0, h_ptr1, chg_typ):
#     passes = self.call_head_ptrs(h_ptr0, h_ptr1)

#     # For any MODIFY type of change
    # if(not passes and chg_typ != DECLARE):

    # else:
node_list_type = ListType(BaseReteNodeType)
node_mem_list_type = ListType(NodeMemoryType)
rete_graph_field_dict = {
    "change_head" : i8,
    "mem" : MemoryType,
    "nodes_by_nargs" : ListType(ListType(BaseReteNodeType)),
    "var_root_nodes" : DictType(i8,BaseReteNodeType),
    "var_end_nodes" : DictType(i8,BaseReteNodeType),
    "global_deref_idrec_map" : DictType(u8, node_list_type),
    "global_base_t_id_map" : DictType(u2, NodeMemoryType),
    "match_iter_prototype_ptr" : i8, #NOTE: Should really use deferred type
}

@njit(cache=True)
def rete_graph_ctor(mem,nodes_by_nargs, var_root_nodes, var_end_nodes,
                    global_deref_idrec_map, global_base_t_id_map):
    st = new(ReteGraphType)
    st.change_head = 0
    st.mem = mem
    st.nodes_by_nargs = nodes_by_nargs
    st.var_root_nodes = var_root_nodes
    st.var_end_nodes = var_end_nodes
    st.global_deref_idrec_map = global_deref_idrec_map
    st.global_base_t_id_map = global_base_t_id_map
    st.match_iter_prototype_ptr = 0
    return st


ReteGraph, ReteGraphType = define_structref("ReteGraph", rete_graph_field_dict, define_constructor=False)



@njit(cache=True)
def _global_map_insert(idrec, g_map, node):
    if(idrec not in g_map):
        g_map[idrec] = List.empty_list(BaseReteNodeType)
    g_map[idrec].append(node)


@njit(cache=True,locals={})
def _make_rete_nodes(mem, c, index_map):
    nodes_by_nargs = List()
    global_deref_idrec_map = Dict.empty(u8, node_list_type)
    # global_base_t_id_map = Dict.empty(u8, node_list_type)

    fn_to_t_id = mem.context_data.fact_num_to_t_id

    for distr_conjuct in c.distr_dnf:
        for i, var_conjuct in enumerate(distr_conjuct):
            for lit in var_conjuct:
                nargs = len(lit.var_base_ptrs)
                while(len(nodes_by_nargs) <= nargs-1):
                    nodes_by_nargs.append(List.empty_list(BaseReteNodeType))

                t_ids = np.empty((nargs,),dtype=np.uint16)
                var_inds = np.empty((nargs,),dtype=np.int64)
                for i,base_var_ptr in enumerate(lit.var_base_ptrs):
                    var_inds[i] = index_map[base_var_ptr]
                    base_var = _struct_from_pointer(GenericVarType, base_var_ptr)
                    t_id = mem.context_data.fact_to_t_id.get(base_var.base_type_name,None)
                    if(t_id is None): raise ValueError("Base Vars of Conditions() must inherit from Fact")
                    t_ids[i] = t_id

                # print("t_ids", t_ids)
                node = node_ctor(mem, lit, t_ids, var_inds)
                nodes_by_nargs[nargs-1].append(node)
                # print("<< aft", lit.op.head_var_ptrs)
                for i, head_var_ptr in enumerate(lit.op.head_var_ptrs):
                    head_var = _struct_from_pointer(GenericVarType, head_var_ptr)
                    ind = np.min(np.nonzero(lit.var_base_ptrs==head_var.base_ptr)[0])
                    t_id = t_ids[ind]
                    # print("START")
                    for d_offset in head_var.deref_offsets:
                        idrec1 = encode_idrec(u2(t_id),0,u1(d_offset.a_id))
                        _global_map_insert(idrec1, global_deref_idrec_map, node)
                        # print("-idrec1", decode_idrec(idrec1))
                        fn = d_offset.fact_num
                        if(fn >= 0 and fn < len(fn_to_t_id)):
                            t_id = fn_to_t_id[d_offset.fact_num]
                            idrec2 = encode_idrec(u2(t_id),0,0)
                            _global_map_insert(idrec2, global_deref_idrec_map, node)
                            # print("--idrec2", decode_idrec(idrec2))
                        else:
                            break
                                

    return nodes_by_nargs, global_deref_idrec_map

optional_node_mem_type = types.optional(NodeMemoryType)
# optional_node_type = types.optional(BaseReteNodeType)





@njit(cache=True)
def build_rete_graph(mem, c):
    index_map = Dict.empty(i8, i8)
    for i, v in enumerate(c.vars):
        index_map[v.base_ptr] = i

    if(not c.has_distr_dnf):
        build_distributed_dnf(c,index_map)

    nodes_by_nargs, global_deref_idrec_map = \
         _make_rete_nodes(mem, c, index_map)

    global_base_t_id_map = Dict.empty(u2, NodeMemoryType)
    var_end_nodes = Dict.empty(i8,BaseReteNodeType)
    var_root_nodes = Dict.empty(i8,BaseReteNodeType)

    # Link nodes together. 'nodes_by_nargs' should already be ordered
    # so that alphas are before 2-way, 3-way, etc. betas. 
    for i, nodes in enumerate(nodes_by_nargs):
        for node in nodes:
            inputs = List.empty_list(NodeMemoryType)
            for j, ind in enumerate(node.var_inds):
                if(ind in var_end_nodes):
                    # Extract the appropriate NodeMemory for this input
                    e_node = var_end_nodes[ind]
                    om = e_node.outputs[np.min(np.nonzero(e_node.var_inds==ind)[0])]
                    inputs.append(om)
                else:
                    t_id = node.t_ids[j]
                    if(t_id not in global_base_t_id_map):
                        root = new_root_node_mem()
                        global_base_t_id_map[t_id] = root
                    root = global_base_t_id_map[t_id]


                    inputs.append(root)
                    var_root_nodes[ind] = node

                    
                    
                        
                    # global_base_t_id_map[t_id].append(root)
                    # idrec = encode_idrec(, 0, 0)
                    # _global_map_insert(, node, global_base_t_id_map)

            # Make this node the new end node for the vars it takes as inputs
            for ind in node.var_inds:
                var_end_nodes[ind] = node
            
            node.inputs = inputs
            # print("<<",len(inputs), len(node.var_inds))


    return rete_graph_ctor(mem, nodes_by_nargs, var_root_nodes, var_end_nodes, global_deref_idrec_map, global_base_t_id_map)


@njit(cache=True,locals={"t_id" : u2, "f_id":u8, "a_id":u1})
def parse_mem_change_queue(r_graph):
    
    global_deref_idrec_map = r_graph.global_deref_idrec_map
    global_base_t_id_map = r_graph.global_base_t_id_map

    # for idrec in global_deref_idrec_map:
    #     print("<< global_deref_idrec_map", decode_idrec(idrec))
    # print(global_deref_idrec_map)

    change_queue = r_graph.mem.mem_data.change_queue
    # print("**:", change_queue.data[:change_queue.head])
    # print("**:", global_base_t_id_map)
    for i in range(r_graph.change_head, change_queue.head):
        idrec = u8(change_queue[i])
        t_id, f_id, a_id = decode_idrec(idrec)
        # print(t_id, f_id, a_id)

        # Add this idrec to change_set of root nodes
        # root_node_mem = global_base_t_id_map.get(t_id,None)
        if(t_id in global_base_t_id_map):
            # print("T_ID", t_id)

            global_base_t_id_map[t_id].change_set[encode_idrec(0,f_id,a_id)] = u1(1)
        # else:
            # print("~T_ID", t_id, List(global_base_t_id_map.keys()))

        # Add this idrec to relevant deref idrecs
        # idrec_pattern = encode_idrec(t_id, 0, a_id)
        idrec_pattern = encode_idrec(t_id, 0, a_id)
        nodes = global_deref_idrec_map.get(idrec_pattern,None)
        if(nodes is not None):
            for node in nodes:
                node.relevant_global_diffs.add(idrec)



       
        # print(decode_idrec(idrec))

    # print(r_graph.global_deref_idrec_map)
    # print(r_graph.global_base_t_id_map)
    r_graph.change_head = change_queue.head

    # print(r_graph.var_root_nodes)
    # for root_node in r_graph.var_root_nodes.values():
    #     print("R")
    #     for i, inp in enumerate(root_node.inputs):
    #         print(inp.change_set)
        # for idrec in root_node.relevant_global_diffs.data:
        #     print(decode_idrec(idrec))
            # print(_pointer_from_struct(inp.change_set))
        # print(root_node.inputs[0].change_set)




    # r_graph.mem.change_queue


match_iterator_node_field_dict = {
    "graph" : ReteGraphType,
    "node" : BaseReteNodeType,
    "associated_output" : NodeMemoryType,
    "depends_on_var_ind" : i8,
    "var_ind" : i8,

    # "is_exhausted": boolean,
    "curr_ind": i8,
    "f_ids" : u8[::1],
    "other_f_ids" : u8[::1],
}



MatchIterNode, MatchIterNodeType = define_structref("MatchIterNode", match_iterator_node_field_dict, define_constructor=False)



class MatchIterator(structref.StructRefProxy):
    ''' '''
    def __new__(cls, context=None):
        raise ValueError("MatchIterator should not be instantiated directly. Use Conditions.get_matches().")
    def __next__(self):
        return match_iter_next(self)
    def __iter__(self):
        return self

match_iterator_field_dict = {

    "iter_nodes" : ListType(MatchIterNodeType),
    "is_empty" : types.boolean,
    # "curr_match" : u8[::1],
}
match_iterator_fields = [(k,v) for k,v, in match_iterator_field_dict.items()]


@structref.register
class MatchIteratorTypeTemplate(types.StructRef):
    def __str__(self):
        return "cre.MatchIterator"
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


define_boxing(MatchIteratorTypeTemplate, MatchIterator)
MatchIteratorType = MatchIteratorTypeTemplate(match_iterator_fields)



# MatchIter, MatchIteratorType = define_structref("MatchIter", match_iterator_field_dict)


@njit(cache=True)
def copy_match_iter(m_iter):
    m_iter_nodes = List.empty_list(MatchIterNodeType)
    for i,m_node in enumerate(m_iter.iter_nodes):
        new_m_node = new(MatchIterNodeType)
        new_m_node.graph = m_node.graph
        new_m_node.node = m_node.node
        new_m_node.var_ind = m_node.var_ind
        new_m_node.associated_output = m_node.associated_output
        new_m_node.depends_on_var_ind = m_node.depends_on_var_ind
        new_m_node.curr_ind = m_node.curr_ind
        if(m_node.curr_ind != -1):
            new_m_node.f_ids = m_node.f_ids
        m_iter_nodes.append(new_m_node)

    m_iter = new(MatchIteratorType)
    m_iter.iter_nodes = m_iter_nodes 
    m_iter.is_empty = m_iter.is_empty

    return m_iter

@njit(cache=True)
def new_match_iter(graph):
    print("HERE")
    if(graph.match_iter_prototype_ptr == 0):
        m_iter_nodes = List.empty_list(MatchIterNodeType)
        handled_vars = Dict.empty(i8,MatchIterNodeType)

        # print("START", len(graph.var_end_nodes))

        for i in range(len(graph.var_end_nodes)-1,-1,-1):
            node = graph.var_end_nodes[i]

            m_node = new(MatchIterNodeType)
            m_node.graph = graph
            m_node.node = node
            m_node.var_ind = i
            m_node.associated_output = node.outputs[np.argmax(node.var_inds==i)]

            if(i in handled_vars):
                m_node.depends_on_var_ind = handled_vars[i].var_ind
            else:
                m_node.depends_on_var_ind = -1

            m_node.curr_ind = -1;
            # m_node.is_exhausted = True

            for j in node.var_inds:
                if(j not in handled_vars):
                    handled_vars[j] = m_node

            m_iter_nodes.append(m_node)

        # Reverse the order since we should iterate from upstream to downstream
        rev_m_iter_nodes = List.empty_list(MatchIterNodeType)
        for i in range(len(m_iter_nodes)-1,-1,-1):
            rev_m_iter_nodes.append(m_iter_nodes[i])

        m_iter = new(MatchIteratorType)
        m_iter.iter_nodes = rev_m_iter_nodes 
        m_iter.is_empty = False
        graph.match_iter_prototype_ptr = _pointer_from_struct_incref(m_iter)
    
    prototype = _struct_from_pointer(MatchIteratorType, graph.match_iter_prototype_ptr)
    m_iter = copy_match_iter(prototype)
    
    return m_iter


@njit(cache=True)
def repr_match_iter_dependencies(m_iter):
    rep = ""
    for i, m_node in enumerate(m_iter.iter_nodes):
        s = f'({str(m_node.var_ind)}'
        if(m_node.depends_on_var_ind != -1):
            # dep = _struct_from_pointer(MatchIterNodeType, )
            s += f",{str(m_node.depends_on_var_ind)})"
        else:
             s += f")"
        rep += s
        if(i < len(m_iter.iter_nodes)-1): rep += " "

    return rep



    # return MatchIter(rev_m_iter_nodes)

@njit(cache=True)
def update_other_f_ids(m_node):
    # print("START UPDATE OTHER")
    if(len(m_node.node.var_inds) > 1):
        # print("A")
        matches = m_node.associated_output.matches
        f_id = m_node.f_ids[m_node.curr_ind]
        # print("B", matches, m_node.f_ids, m_node.curr_ind)
        other_f_ids_d = _dict_from_ptr(dict_i8_u1_type, matches[f_id]) 
        # print("C", other_f_ids_d)
        # print("C", )
        # print("C", len(other_f_ids_d))
        other_f_ids = np.empty((len(other_f_ids_d),), dtype=np.uint64)
        # print("F")
        cnt = 0
        for j, (o_f_id,v) in enumerate(other_f_ids_d.items()):
            if(v == 0): continue
            other_f_ids[j] = o_f_id; cnt += 1;
        m_node.other_f_ids = other_f_ids[:cnt]
        # print("S UPDATE OTHER", m_node.other_f_ids)
        # print("Z")
    # print("END UPDATE OTHER")



@njit(cache=True)
def restitch_match_iter(m_iter, start_from=-1):
    if(start_from == -1): start_from = len(m_iter.iter_nodes)-1
    for i in range(start_from,-1,-1):
        # print("I", i)
        m_node = m_iter.iter_nodes[i]
        if(m_node.curr_ind == -1):
            if(m_node.depends_on_var_ind == -1):
                # print("A")
                matches = m_node.associated_output.matches
                f_ids = np.empty((len(matches)),dtype=np.uint64)
                cnt = 0
                # print("B", matches, len(m_node.node.var_inds))
                # print(matches)
                for j, (f_id, v) in enumerate(matches.items()):
                    if(v == 0): continue
                    f_ids[j] = f_id; cnt += 1;
                m_node.f_ids = f_ids[:cnt]
                # print("C", cnt)
                # print(m_node.f_ids)
                
                
            else:
                dep_node = m_iter.iter_nodes[m_node.depends_on_var_ind]
                m_node.f_ids = dep_node.other_f_ids
                # m_node.curr_ind = 0
            m_node.curr_ind = 0
        if(i > 0):
            # print("SS")
            update_other_f_ids(m_node)
            # print("S UPDATE OTHER", i, m_node.other_f_ids)
        # else:
        #     print("CURRIND", m_node.curr_ind)
            # m_node.curr_ind = 0

@njit(cache=True)
def match_iter_next(m_iter):
    n_vars = len(m_iter.iter_nodes)
    if(m_iter.is_empty or n_vars == 0): raise StopIteration()

    f_ids = np.empty(n_vars,dtype=np.uint64)
    most_downstream_overflow = -1
    for i, m_node in enumerate(m_iter.iter_nodes):
        # print("RESTICH")
        # print(i, ":", m_node.curr_ind, m_node.f_ids)
        f_ids[m_node.var_ind] = m_node.f_ids[m_node.curr_ind]

        update_other_needed = False
        if(i == 0):
            m_node.curr_ind += 1
        elif(most_downstream_overflow == i-1):
            m_node.curr_ind += 1
            update_other_needed = True

        if(m_node.curr_ind >= len(m_node.f_ids)):
            if(i == n_vars-1):
                m_iter.is_empty = True
                return f_ids
            m_node.curr_ind = -1
            most_downstream_overflow = i
        else:
            update_other_f_ids(m_node)


    if(most_downstream_overflow != -1):
        restitch_match_iter(m_iter, most_downstream_overflow)

    return f_ids

@njit(cache=True)
def update_graph(graph):
    parse_mem_change_queue(graph)
    for lst in graph.nodes_by_nargs:
        for node in lst:
            update_node(node)        


@njit(cache=True)
def get_matches(mem, conds):
    if(conds.matcher_inst_ptr == 0):
        rete_graph = build_rete_graph(mem, conds)
        conds.matcher_inst_ptr = _pointer_from_struct_incref(rete_graph)
    rete_graph = _struct_from_pointer(ReteGraphType, conds.matcher_inst_ptr)
    update_graph(rete_graph)
    m_iter = new_match_iter(rete_graph)
    # for i, m_node in enumerate(m_iter.iter_nodes):
    #     print("<<", m_node.curr_ind)
    # print("INITIAL RESTICH")
    restitch_match_iter(m_iter)
    # print("RESTRICT ED")
    return m_iter





if __name__ == "__main__":
    pass
    # deref_depends = Dict.empty(i8, DictType(i8,u1))

    # node_ctor()







"""

Notes: 

What is needed by 

    









"""
