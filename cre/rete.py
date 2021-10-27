import numpy as np
import numba
from numba import types, njit, i8, u8, i4, u1, u2, i8, f8, literally, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple
from numba.experimental.structref import new, define_boxing
import numba.experimental.structref as structref
from cre.utils import (wptr_t, ptr_t, _dict_from_ptr, _raw_ptr_from_struct, _get_array_data_ptr,
         _ptr_from_struct_incref, _struct_from_ptr, decode_idrec, CastFriendlyMixin,
        encode_idrec, deref_type, DEREF_TYPE_ATTR, DEREF_TYPE_LIST, _obj_cast_codegen,
         _ptr_to_data_ptr, _list_base_from_ptr, _load_ptr, PrintElapse, meminfo_type,
         _decref_structref, _decref_ptr, cast_structref, _struct_tuple_from_pointer_arr, _meminfo_from_struct)
from cre.structref import define_structref
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.memory import MemoryType
from cre.vector import VectorType
from cre.var import GenericVarType
from cre.op import GenericOpType
from cre.conditions import LiteralType, build_distributed_dnf, ConditionsType
from cre.vector import VectorType, new_vector
from cre.fact import BaseFactType 
import cloudpickle

from numba.core.imputils import (lower_cast)


RETRACT = u1(0xFF)# u1(0)
DECLARE = u1(0)

deref_record_field_dict = {
    # Weak Pointers to the Dict(i8,u1)s inside deref_depends
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
    "change_buffer" : u8[::1],#DictType(u8,u1), 
    "change_set" : u8[::1],#DictType(u8,u1), 
    
    # Maps f_ids -> sets of f_ids
    "match_idrecs_buffer" : u8[::1], 
    "match_idrecs" : u8[::1], 

    "match_inds_buffer" : i8[::1], 
    "match_inds" : i8[::1], 

    # "match_holes" : VectorType,

    # Weak pointer to the node that owns this memory
    "parent_node_ptr" : i8,

}

NodeMemory, NodeMemoryType = define_structref("NodeMemory", node_memory_field_dict)


@njit(cache=True)
def new_node_mem():
    # print("NEW NODE")
    st = new(NodeMemoryType)
    #Placeholders 
    st.change_buffer = np.empty(8,dtype=np.uint64)#Dict.empty(u8,u1)
    st.change_set = st.change_buffer
    
    st.match_idrecs_buffer = np.empty(8,dtype=np.uint64)
    st.match_idrecs = st.match_idrecs_buffer[:0]
    st.match_inds_buffer = np.empty(8,dtype=np.int64)
    st.match_inds = st.match_inds_buffer[:0]
    # st.match_holes = new_vector(2)#Dict.empty(u8,i8)
    st.is_root = False
    st.parent_node_ptr = 0
    return st

@njit(cache=True)
def new_root_node_mem():
    st = new_node_mem()
    st.is_root = True
    return st


_input_state_type = np.dtype([
    ('idrec', np.int64),
    ('true_count', np.int64),
    ('head_was_valid', np.uint8), 
    ('is_changed', np.uint8),

    # The input had a match in the previous match cycle 
    ('true_was_nonzero', np.uint8), 

    # The input has ever had a match. Needed to keep track of holes in the match set. 
    #  holes are necessary to ensure match iterators are valid on backtracks.
    ('true_ever_nonzero', np.uint8),
])
    
input_state_type = numba.from_dtype(_input_state_type)


dict_i8_u1_type = DictType(i8,u1)
dict_u8_i8_arr_type = DictType(u8,i8[::1])
deref_dep_typ = DictType(u8,DictType(ptr_t,u1))


# outputs_item_typ = DictType(u8,DictType(u8,u1))
dict_u8_u1_type = DictType(u8,u1)
# outputs_typ = DictType(u8,DictType(u8,u1))

_idrec_ind_pair = np.dtype([('idrec', np.uint64),  ('ind', np.int64)])
idrec_ind_pair_type = numba.from_dtype(_idrec_ind_pair)

base_rete_node_field_dict = {
    # Pointers to the Dict(i8,u1)s inside deref_depends
    "mem" : MemoryType, 
    "has_op" : types.boolean,
    "lit" : types.optional(LiteralType),
    "op" : types.optional(GenericOpType),
    "deref_depends" : deref_dep_typ, 
    "relevant_global_diffs" : VectorType,
    "n_vars" : i8,

    "var_inds" : i8[::1],
    "t_ids" : u2[::1],
    # "vars" : ListType(GenericVarType),
    "idrecs_to_inds" : ListType(DictType(u8,i8)),
    "retracted_inds" : ListType(VectorType),
    "widths" : i8[::1],
    "head_ptr_buffers" : ListType(i8[:,::1]),
    "input_state_buffers" : ListType(input_state_type[::1]),
    


    # "idrecs_change_buffers" : ListType(u8[::1]),
    "inds_change_buffers" : ListType(i8[::1]),
    # "changed_idrecs" : ListType(u8[::1]),
    # "unchanged_idrecs" : ListType(u8[::1]),
    "changed_inds" : ListType(i8[::1]),
    "unchanged_inds" : ListType(i8[::1]),

    "idrecs_match_buffers" : ListType(u8[::1]),
    "inds_match_buffers" : ListType(i8[::1]),

    # "head_ptr_buffers" : i8[:,::1],

    # "head_ptrs" : ListType(DictType(u8,i8[::1])),
    "inputs" : ListType(NodeMemoryType),
    "outputs" : ListType(NodeMemoryType), #DictType(u8,DictType(u8,u1))
    "truth_table" : u1[:, ::1],

    # True if the inputs to this node both come from the same beta node
    # e.g. like in the case of (a.v < b.v) & (a.v != b.v)
    "inputs_same_parent" : u1,

    # A weak pointer to the node upstream to this one
    "upstream_node_ptr" : i8,

    # A temporary record array of with inds and idrecs of changes
    "change_pairs" : ListType(idrec_ind_pair_type[::1])

}

BaseReteNode, BaseReteNodeType = define_structref("BaseReteNode", base_rete_node_field_dict, define_constructor=False)


u8_arr_typ = u8[::1]
i8_arr_typ = i8[::1]
i8_x2_arr_typ = i8[:,::1]
input_state_arr_type = input_state_type[::1]
idrec_ind_pair_arr_type = idrec_ind_pair_type[::1]

@njit(cache=True)
def node_ctor(mem, t_ids, var_inds,lit=None):
    # print("NEW RETE NODE")
    st = new(BaseReteNodeType)
    st.mem = mem
    st.deref_depends = Dict.empty(u8,dict_i8_u1_type)
    st.relevant_global_diffs = new_vector(4)
    st.var_inds = var_inds
    st.t_ids = t_ids

    # if(lit is not None):
    st.head_ptr_buffers = List.empty_list(i8_x2_arr_typ)
    st.input_state_buffers = List.empty_list(input_state_arr_type)

    # st.idrecs_change_buffers = List.empty_list(u8_arr_typ)
    # st.changed_idrecs = List.empty_list(u8_arr_typ)
    # st.unchanged_idrecs = List.empty_list(u8_arr_typ)
    st.inds_change_buffers = List.empty_list(i8_arr_typ)
    st.changed_inds = List.empty_list(i8_arr_typ)
    st.unchanged_inds = List.empty_list(i8_arr_typ)

    # st.idrecs_match_buffers = List.empty_list(u8_arr_typ)
    # st.inds_match_buffers = List.empty_list(i8_arr_typ)

    st.lit = lit
    if(lit is not None):
        st.op = op = lit.op
        n_vars = st.n_vars = len(op.base_var_map)
        for i in range(n_vars):
            l = op.head_ranges[i].length
            st.head_ptr_buffers.append(np.empty((8,l),dtype=np.int64))
            st.input_state_buffers.append(np.zeros(8, dtype=input_state_type))

            # idrec_change_buff = np.empty(8, dtype=np.uint64)
            ind_change_buff = np.empty(8, dtype=np.int64)
            # st.idrecs_change_buffers.append(idrec_change_buff)
            # st.changed_idrecs.append(idrec_change_buff)
            # st.unchanged_idrecs.append(idrec_change_buff)
            st.inds_change_buffers.append(ind_change_buff)
            st.changed_inds.append(ind_change_buff)
            st.unchanged_inds.append(ind_change_buff)

            # idrec_buff = np.empty(8, dtype=np.uint64)
            # ind_buff = np.empty(8, dtype=np.int64)
            # st.idrecs_match_buffers.append(idrec_buff)
            # st.inds_match_buffers.append(ind_buff)
    else:
        st.op = None
        n_vars = 1

    # st.head_ptr_buffers = head_ptr_buffers
    # st.input_state_buffers = input_state_buffers
    # st.idrecs_change_buffers = idrecs_change_buffers
    # st.changed_idrecs = changed_idrecs
    # st.unchanged_idrecs = unchanged_idrecs
    # st.inds_change_buffers = inds_change_buffers
    # st.changed_inds = changed_inds
    # st.unchanged_inds = unchanged_inds

    outputs = List.empty_list(NodeMemoryType)
    

    self_ptr = _raw_ptr_from_struct(st)
    for i in range(n_vars):
        # outputs.append(Dict.empty(u8,dict_u8_u1_type))
        node_mem = new_node_mem()
        node_mem.parent_node_ptr = self_ptr
        outputs.append(node_mem)
        
    st.outputs = outputs

    st.truth_table = np.zeros((8,8), dtype=np.uint8)
    st.idrecs_to_inds = List.empty_list(u8_i8_dict_type) 
    st.retracted_inds = List.empty_list(VectorType) 
    st.change_pairs = List.empty_list(idrec_ind_pair_arr_type) 
    st.widths = np.zeros(2,dtype=np.int64)
    #  "idrecs_to_inds" : ListType(DictType(u8,i8)),
    # "retracted_inds" : ListType(VectorType),
    # "widths" : i8[::1],

    for i in range(n_vars):
        st.idrecs_to_inds.append(Dict.empty(u8,i8))
        st.retracted_inds.append(new_vector(8))
        st.change_pairs.append(np.empty(0,dtype=idrec_ind_pair_type))

    # Just make False by default, can end up being True after linking
    st.inputs_same_parent = False
    st.upstream_node_ptr = 0


    return st


@njit(i8(i8, deref_type[::1]), cache=True,locals={"data_ptr":i8, "inst_ptr":i8})
def deref_head_and_relevant_idrecs(inst_ptr, deref_offsets):
    ''' '''
    
    # relevant_idrecs = np.zeros((max((len(deref_offsets)-1)*2+1,0),), dtype=np.uint64)
    # print("N", len(relevant_idrecs))
    k = -1

    # ok = True

    for deref in deref_offsets[:-1]:
        # print("ENTERED")
        if(inst_ptr == 0): break;
        if(deref.type == u1(DEREF_TYPE_ATTR)):
            data_ptr = _ptr_to_data_ptr(inst_ptr)
        else:
            data_ptr = _list_base_from_ptr(inst_ptr)
        t_id, f_id, _ = decode_idrec(u8(_load_ptr(u8, data_ptr)))
        # if(k >= 0):
        #     relevant_idrecs[k] = encode_idrec(t_id, f_id, RETRACT);
        # k += 1
        # relevant_idrecs[k] = encode_idrec(t_id, f_id, deref.a_id); k += 1

        inst_ptr = _load_ptr(i8, data_ptr+deref.offset)
    
    # print(inst_ptr)
    if(inst_ptr != 0):
        deref = deref_offsets[-1]
        if(deref.type == u1(DEREF_TYPE_ATTR)):
            data_ptr = _ptr_to_data_ptr(inst_ptr)
        else:
            data_ptr = _list_base_from_ptr(inst_ptr)
        t_id, f_id, _ = decode_idrec(u8(_load_ptr(u8, data_ptr)))
        # relevant_idrecs[k] = encode_idrec(t_id, f_id, RETRACT); k += 1
        # relevant_idrecs[k] = encode_idrec(t_id, f_id, deref.a_id); k += 1

        head_ptr = data_ptr+deref.offset
        # print("head_ptr",head_ptr, relevant_idrecs)
        return head_ptr#, relevant_idrecs
    else:
        # print("nope")
        return 0#, relevant_idrecs[:k]


#Example Deref a.B.B.B.A
# dep_idrecs = [(1, MOD[a.B]), (2, RETRACT[a.B]),
#               (2, MOD[a.B.B]), (3, RETRACT[a.B.B]),
#                (3, MOD[a.B.B.B]), (4, RETRACT[a.B.B.B]),
#                (4, MOD[a.B.B.B.A])
#                     ]



@njit(cache=True)
def invalidate_head_ptr_rec(rec):
    r_ptr = _raw_ptr_from_struct(rec)
    for ptr in rec.parent_ptrs:
        parent = _dict_from_ptr(dict_i8_u1_type, ptr)
        del parent[r_ptr]
        _decref_structref(rec)

@njit(wptr_t(deref_dep_typ, u8, ptr_t),cache=True)
def make_deref_record_parent(deref_depends, idrec, r_ptr):
    p = deref_depends.get(idrec,None)
    if(p is None): 
        p = deref_depends[idrec] = Dict.empty(ptr_t,u1)
    p[r_ptr] = u1(1)
    return _raw_ptr_from_struct(p)

@njit(i8(deref_type,i8),inline='always',cache=True)
def deref_once(deref,inst_ptr):
    if(deref.type == u1(DEREF_TYPE_ATTR)):
        return _ptr_to_data_ptr(inst_ptr)
    else:
        return _list_base_from_ptr(inst_ptr)


# @njit(i8(BaseReteNodeType, u2, u8, deref_type[::1]),cache=True)
@njit(cache=True)
def resolve_head_ptr(self, arg_ind, base_t_id, f_id, deref_offsets):
    '''Try to get the head_ptr of 'f_id' in input 'arg_ind'. Inject a DerefRecord regardless of the result 
        Keep in mind that a head_ptr is the pointer to the address where the data is stored not the data itself.
    '''
    facts = _struct_from_ptr(VectorType, self.mem.mem_data.facts[base_t_id])
    if(len(deref_offsets) > 0):
        inst_ptr = facts.data[f_id]
        if(len(deref_offsets) > 1):
            rel_idrecs = np.empty(len(deref_offsets)-1, dtype=np.uint64)
            for k in range(len(deref_offsets)-1):
                if(inst_ptr == 0): break;
                deref = deref_offsets[k]
                data_ptr = deref_once(deref,inst_ptr)
                rel_idrecs[k] = _load_ptr(u8, data_ptr)
                inst_ptr = _load_ptr(i8, data_ptr+deref.offset)

            # Inject a deref record so that we can track changes to intermediate facts
            parent_ptrs = rel_idrecs.astype(i8) 
            r = DerefRecord(parent_ptrs, arg_ind, encode_idrec(base_t_id,f_id,0), inst_ptr != 0) #7us
            r_ptr = _ptr_from_struct_incref(r)
            for i, idrec in enumerate(rel_idrecs): #21 us
                parent_ptrs[i] = i8(make_deref_record_parent(self.deref_depends, idrec, r_ptr))            

        if(inst_ptr != 0):
            deref = deref_offsets[-1]
            data_ptr = deref_once(deref,inst_ptr)
            return data_ptr+deref.offset
        else:
            return 0
    else:
        return _get_array_data_ptr(facts.data) + (f_id * 8) #assuming 8 byte ptrs

# @njit(void(BaseReteNodeType, i8,u8,u1),locals={"f_id" : u8}, cache=True)
@njit(locals={"f_id" : u8, "a_id" : u8}, cache=True)
def validate_head_or_retract(self, arg_ind, idrec, head_ptrs, r):
    '''Update the head_ptr dictionaries by following the deref
     chains of DECLARE/MODIFY changes, and make retractions
    for an explicit RETRACT or a failure in the deref chain.'''

    t_id, f_id, a_id = decode_idrec(idrec)
    is_valid = True
    if(a_id != RETRACT):
        # base_t_id = self.t_ids[arg_ind]
        # r = self.op.head_ranges[arg_ind]
        # print("r.length", r.length)
        # return r.length == 77
        # start = (k*r.length)
        # end = start + r.length
        # head_ptrs = head_ptr_buffer[k] #np.empty(r.length,dtype=np.int64)
        # return False
        # okay = True
        # For each head_var try to deref all the way to the head_ptr and put it in head_ptrs
        for i in range(r.length):
            # continue 
            head_var = _struct_from_ptr(GenericVarType,self.op.head_var_ptrs[r.start+i])
            deref_offsets = head_var.deref_offsets
            # continue
            # print("--start resolve_head_ptr",)
            head_ptr = resolve_head_ptr(self, arg_ind, t_id, f_id, deref_offsets)
            # print("resolve_head_ptr", f_id, head_ptr)
            if(head_ptr == 0): 
                is_valid=False;
                # del change_set[idrec];
                break; 
            head_ptrs[i] = head_ptr

        # if(is_valid):
            # new_idrec = encode_idrec(t_id, f_id, 0)
            # self.head_ptrs[arg_ind][new_idrec] = head_ptrs
    else:
        is_valid = False
    return is_valid
            
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
                r = _struct_from_ptr(DerefRecordType,r_ptr)
                invalidate_head_ptr_rec(r)

                # Any change in the deref chain counts as a MODIFY
                # TODO: This is for sure wrong
                arg_change_sets[r.arg_ind][r.base_idrec] = 1 #MODIFY

# for i,change_set in enumerate(arg_change_sets):
#         idrecs_to_inds_i = self.idrecs_to_inds[i]
#         retracted_inds_i = self.retracted_inds[i]
#         for idrec in change_set:
#             ind = idrecs_to_inds_i.get(idrec, -1)
#             if(ind == -1):
#                 if(len(retracted_inds_i) > 0):
#                     ind = retracted_inds_i.pop()
#                 else:
#                     ind = self.widths[i]
#                     self.widths[i] += 1
#                 idrecs_to_inds_i[idrec] = ind






# @njit(cache=True)
# def input_states_update(input_state_buffer, idrec, ind):


@njit(cache=True)
def update_changes_from_inputs(self):
    # print("Q")
    # num_changes = np.zeros(self.n_vars,dtype=np.int64)
    for i, inp in enumerate(self.inputs):
        num_changes = 0
        # arg_change_sets_i = arg_change_sets[i]
        w_i = self.widths[i]
        idrecs_to_inds_i = self.idrecs_to_inds[i]
        retracted_inds_i = self.retracted_inds[i]
        head_ptr_buffers_i = self.head_ptr_buffers[i]
        input_state_buffers_i = self.input_state_buffers[i]
        # idrecs_change_buffers_i = self.idrecs_change_buffers[i]

        # idrecs_match_buffers_i = self.idrecs_match_buffers[i]
        inds_change_buffers_i = self.inds_change_buffers[i]
        # inds_match_buffers_i = self.inds_match_buffers_i[i]
        head_range_i = self.op.head_ranges[i]
        # return
        # print("--", len(inp.change_set))
        # Designate inds for each idrec and expand widths (14 us) 
        if(not self.inputs_same_parent):
            change_pairs = self.change_pairs[i] = np.empty(len(inp.change_set),dtype=idrec_ind_pair_type)
            for k, idrec in enumerate(inp.change_set):
                # If the inputs have the same parent then the node should just share
                #  the same idrecs_to_inds and widths
                ind = idrecs_to_inds_i.get(idrec,-1)
                if(ind == -1): 
                    if(len(retracted_inds_i) > 0):
                        ind = retracted_inds_i.pop()
                    else:
                        ind = w_i
                        w_i += 1
                    idrecs_to_inds_i[idrec] = ind
                # else:
                #     ind = idrecs_to_inds_i.get(idrec,-1)
                #     assert ind != -1

                change_pairs[k].idrec = idrec
                change_pairs[k].ind = ind

        self.widths[i] = w_i
        # print("<<",idrecs_to_inds_i)
        # print("A")
        
        # Make sure various buffers that must be the length of the input set are long enough (3 us)
        curr_len, curr_w = head_ptr_buffers_i.shape
        if(self.widths[i] > curr_len):
            expand = max(self.widths[i]-curr_len, curr_len)
            new_head_ptr_buff = np.empty((curr_len+expand,curr_w),dtype=np.int64)
            new_head_ptr_buff[:curr_len] = head_ptr_buffers_i
            head_ptr_buffers_i = self.head_ptr_buffers[i] = new_head_ptr_buff

            new_input_state_buff = np.zeros((curr_len+expand,),dtype=input_state_type)
            new_input_state_buff[:curr_len] = input_state_buffers_i
            input_state_buffers_i = self.input_state_buffers[i] = new_input_state_buff

            # new_idrec_chng_buff = np.empty((curr_len+expand,),dtype=np.uint64)
            # new_idrec_chng_buff[:curr_len] = idrecs_change_buffers_i
            # idrecs_change_buffers_i = self.idrecs_change_buffers[i] = new_idrec_chng_buff
            # new_idrec_match_buff = np.empty((curr_len+expand,),dtype=np.uint64)
            # new_idrec_match_buff[:curr_len] = idrecs_match_buffers_i
            # idrecs_match_buffers_i = self.idrecs_match_buffers[i] = new_idrec_match_buff

            new_inds_chng_buff = np.empty((curr_len+expand,),dtype=np.int64)
            new_inds_chng_buff[:curr_len] = inds_change_buffers_i
            inds_change_buffers_i = self.inds_change_buffers[i] = new_inds_chng_buff
            
            # new_inds_match_buff = np.empty((curr_len+expand,),dtype=np.int64)
            # new_inds_match_buff[:curr_len] = inds_match_buffers_i
            # inds_match_buffers_i = self.inds_match_buffers_i[i] = new_inds_match_buff
        # print("B", idrec_ind_pairs)
        
        # # Try to deref, mark any changes (10 us)
        # print(self.change_pairs[i])
        for pair in self.change_pairs[i]:
            idrec, ind = pair.idrec, pair.ind

            head_ptrs = head_ptr_buffers_i[ind]
            input_state = input_state_buffers_i[ind]

            is_valid = validate_head_or_retract(self, i, idrec, head_ptrs, head_range_i)
            is_changed = is_valid ^ input_state.head_was_valid

            input_state.idrec = idrec
            input_state.is_changed = is_changed
            input_state.head_was_valid = is_valid

            num_changes += is_changed
        # print("C")
        # print("num changes:", num_changes)
        # changed_idrecs_i = self.changed_idrecs[i] = self.idrecs_change_buffers[i][:num_changes]
        # unchanged_idrecs_i = self.unchanged_idrecs[i] = self.idrecs_change_buffers[i][num_changes:]
        changed_inds_i = self.changed_inds[i] = self.inds_change_buffers[i][:num_changes]
        unchanged_inds_i = self.unchanged_inds[i] = self.inds_change_buffers[i][num_changes:w_i]

        # Fill the changed and unchanged idrecs/inds arrays (<1 us)
        c, u = 0, 0
        for j in range(self.widths[i]):
            # print(j, c, u, self.widths[i])
            input_state = input_state_buffers_i[j]
            if(input_state.is_changed):
                # changed_idrecs_i[c] = input_state.idrec
                changed_inds_i[c] = j
                c += 1
            else:
                # unchanged_idrecs_i[u] = input_state.idrec
                unchanged_inds_i[u] = j
                u += 1
        # print(changed_inds_i, unchanged_inds_i)




@njit(cache=True)
def resize_truth_table(self):
    s0,s1 = self.truth_table.shape
    expand0 = max(0,self.widths[0]-s0)
    expand1 = max(0,self.widths[1]-s1)
    if(expand0 and expand0):
        expand0 = max(s0,expand0)
        expand1 = max(s1,expand1)

        new_truth_table = np.empty((s0+expand0, s1+expand1),dtype=np.uint8)
        new_truth_table[:s0,:s1] = self.truth_table
        new_truth_table[s0:] = u1(0)
        new_truth_table[:, s1:] = u1(0)
        self.truth_table = new_truth_table

    elif(expand0):
        expand0 = max(s0,expand0)

        new_truth_table = np.empty((s0+expand0, s1),dtype=np.uint8)
        new_truth_table[:s0,:s1] = self.truth_table
        new_truth_table[s0:] = u1(0)
        self.truth_table = new_truth_table

    elif(expand1):
        expand1 = max(s1,expand1)
        new_truth_table = np.empty((s0,s1+expand1),dtype=np.uint8)
        new_truth_table[:s0,:s1] = self.truth_table
        new_truth_table[:, s1:] = u1(0)
        self.truth_table = new_truth_table

            # print("::", i, idrec)

        # if(len(arg_change_sets_i) > 0):
             
        # else:
        #     arg_change_sets[i] = inp.change_set

# @njit(cache=True)
# def update_head_ptrs(self, arg_change_sets):
#     for i,change_set in enumerate(arg_change_sets):
#         # print(change_set)
#         for idrec in change_set:
#             # _, f_id, a_id = decode_idrec(idrec)
#             # print("<<",idrec)
#             validate_head_or_retract(self, i, idrec, change_set)



# def foo(lengths):
#     n = len(lengths)
#     iters = np.zeros(n,dtype=np.int64)
#     # lens = np.zeros(n,dtype=np.int64)
#     i = end = n-1
#     while(i >= 0):
#         iters[i] += 1
#         if(iters[i] >= lengths[i]): 
#             i -= 1
#             iters[i] = 0
#         elif(i != end):
#             i += 1
#         print(iters)
            
from cre.utils import _func_from_address
match_heads_f_type = types.FunctionType(u1(i8[::1],))

u8_i8_dict_type = DictType(u8,i8)


# @njit(cache=True)
# def insert_alpha_match(self, idrec):
#     was_match = self.outputs[0].matches.get(idrec,0) != 0
#     if(not was_match):
#         self.outputs[0].change_set[idrec] = u1(1)
#         self.outputs[0].matches[idrec] = 1

# @njit(cache=True)
# def invalidate_alpha_match(self, idrec):
#     was_match = self.outputs[0].matches.get(idrec,0) != 0
#     if(was_match):
#         t_id, f_id, _ = decode_idrec(idrec)
#         self.outputs[0].change_set[encode_idrec(t_id, f_id, RETRACT)] = u1(1)

@njit(cache=True)
def insert_alpha_match(self, ind):
    self.truth_table[ind,:] |= u1(1)    
    # was_match = self.outputs[0].matches.get(idrec,0) != 0
    # if(not was_match):
    #     self.outputs[0].change_set[idrec] = u1(1)
    #     self.outputs[0].matches[idrec] = 1

# @njit(cache=True)
# def invalidate_alpha_match(self, idrec):
    # was_match = self.outputs[0].matches.get(idrec,0) != 0
    # if(was_match):
    #     t_id, f_id, _ = decode_idrec(idrec)
    #     self.outputs[0].change_set[encode_idrec(t_id, f_id, RETRACT)] = u1(1)

@njit(cache=True)
def _ld_dict(ptr):
    return _dict_from_ptr(u8_i8_dict_type, ptr)     

@njit(cache=True,locals={'d0_ptr':i8, 'd1_ptr':i8})
def insert_beta_match(self, match_inds, match_idrecs):
    # pass
    ind0, ind1 = match_inds[0], match_inds[1]
    # idrec0, idrec1 = match_idrecs[0], match_idrecs[1]
    # was_match = (self.truth_table[ind0,ind1] & 1) != 0
    # print(ind0,ind1)
    self.truth_table[ind0,ind1] |= u1(1)    

    # if(not was_match):
    #     self.outputs[0].change_set[idrec0] = u1(1)
    #     self.outputs[1].change_set[idrec1] = u1(1)
    #     self.outputs[0].matches[idrec0] = u1(1)
    #     self.outputs[1].matches[idrec1] = u1(1)
    # outputs = self.outputs
    # was_match = True
    # d0_ptr = outputs[0].matches.get(idrec0, 0)
    # if(d0_ptr == 0):
    #     d0 = Dict.empty(u8,i8)
    #     d0_ptr = _ptr_from_struct_incref(d0)
    #     outputs[0].matches[idrec0] = d0_ptr
    #     was_match = False
    # else:
    #     _d0_ptr = d0_ptr
    #     d0 = _ld_dict(_d0_ptr)
    # d1_ptr = outputs[1].matches.get(idrec1, 0)
    # if(d1_ptr == 0):
    #     d1 = Dict.empty(u8,i8)
    #     d1_ptr = _ptr_from_struct_incref(d1)
    #     outputs[1].matches[idrec1] = d1_ptr
    #     was_match = False
    # else:
    #     _d1_ptr = d1_ptr
    #     d1 = _ld_dict(_d1_ptr)

    # d0[idrec1] = 1
    # d1[idrec0] = 1


    # return was_match

@njit(cache=True, locals={'ptr0':i8,'ptr1':i8,})
def invalidate_beta_match(self, match_idrecs):
    idrec0,idrec1 = match_idrecs[0], match_idrecs[1]
    outputs = self.outputs
    was_match = False
    ptr0 = outputs[0].matches.get(idrec0,0)
    if(ptr0 != 0):
        d = _ld_dict(ptr0)
        if(d.get(idrec1,0) != 0):
            was_match = True
            d[idrec1] = 0
    ptr1 = outputs[1].matches.get(idrec1,0)
    if(ptr1 != 0):
        d = _ld_dict(ptr1)
        if(d.get(idrec0,0) != 0):
            was_match = True
            d[idrec0] = 0
    return was_match



# @njit(cache=True)
# def invalidate_f_id(self, arg_ind, f_id, decref=False):
#     outputs = self.outputs
#     ptr0 = outputs[0].matches.get(f_id0,0)
#     if(ptr0 != 0):
#         if(decref): _decref_ptr(ptr0)
#         outputs[0].matches[f_id0] = 0

@njit(cache=True)
def beta_matches_to_str(matches):
    # for x in self.outputs
    s = ""
    for match0, others_ptr in matches.items():
        others_str = ""
        t_id0, f_id0, a0 = decode_idrec(match0)
        others = _ld_dict(others_ptr)
        for match1, ok in others.items():
            t_id1, f_id1, _ = decode_idrec(match1)
            if(ok): others_str += f"({t_id1},{f_id1}),"
            
        s +=f"({t_id0},{f_id0}) : [{others_str[:-1] if len(others_str) > 0 else ''}]\n"

    return s


@njit(cache=True)
def alpha_matches_to_str(matches):
    # for x in self.outputs
    s = "["
    for match0, others_ptr in matches.items():
        # others_str = ""
        t_id0, f_id0, a0 = decode_idrec(match0)
        # others = _ld_dict(others_ptr)
        # for match1, ok in others.items():
        #     t_id1, f_id1, _ = decode_idrec(match1)
        #     if(ok): others_str += f"({t_id1},{f_id1}),"
            
        s +=f"({t_id0},{f_id0}),"

    return s[:-1] + "]"
    # print("<<", [x.matches ])



@njit(cache=True, inline='always')
def _handle_beta_match(tt, negated, inp_state_i, inp_state_j, match_head_ptrs_func, match_inp_ptrs, match_inds, match_idrecs):    
    ind0, ind1 = match_inds[0], match_inds[1]
    if(not inp_state_j.head_was_valid): return
    
    # print("L", match_inds, match_inp_ptrs)
    was_match = tt[ind0, ind1]
    is_match = match_head_ptrs_func(match_inp_ptrs) ^ negated
    tt[ind0, ind1] = u1(is_match)
    count_diff = i8(is_match) - i8(was_match)
    inp_state_i.true_count += count_diff
    inp_state_j.true_count += count_diff

    # print("Beta",is_match, match_inp_ptrs, match_inds, decode_idrec(inp_state_i.idrec)[1], decode_idrec(inp_state_j.idrec)[1])

    #     tt[0, 0] |= u1(1)
        # print("INSER", match_f_ids, self.lit.negated)
        # ind0, ind1 = 
    # idrec0, idrec1 = match_idrecs[0], match_idrecs[1]
    # was_match = (self.truth_table[ind0,ind1] & 1) != 0
    # print(ind0,ind1)
        
        # was_match = insert_beta_match(self, match_inds, match_idrecs)
    # else:
        # was_match = invalidate_beta_match(self, match_inds)
    # print(f_id_i, decode_idrec(idrec_j)[1], is_match, was_match)

@njit(cache=True)
def update_node(self):
    # return
    # print("START")
    # If the node is an identity node then skip since the input is hardwired to the output
    if(self.op is None): return
    # print("UPDATE_NODE", self.lit)    
        
    # NOTE: Before this could be 0 for declare , 1 for modify...
# arg_change_sets = List.empty_list(u8_i8_dict_type)
    # for j in range(len(self.var_inds)):
    #      arg_change_sets.append(Dict.empty(u8,i8))
    
        # self.change_set = self.inputs.outputs

    #TODO will need to add back in
    #update_changes_deref_dependencies(self, arg_change_sets)
    # print("A")
    # return
    update_changes_from_inputs(self)
    # print("B")
    # update_head_ptrs(self, arg_change_sets)
    # return 
    

    
    # print("!UPDATE_NODE", self.op)    
    head_ranges = self.op.head_ranges
    n_vars = len(head_ranges)    
    negated = self.lit.negated

    # for i,change_set in enumerate(arg_change_sets):
    #     idrecs_to_inds_i = self.idrecs_to_inds[i]
    #     retracted_inds_i = self.retracted_inds[i]
    #     for idrec in change_set:
    #         ind = idrecs_to_inds_i.get(idrec, -1)
    #         if(ind == -1):
    #             if(len(retracted_inds_i) > 0):
    #                 ind = retracted_inds_i.pop()
    #             else:
    #                 ind = self.widths[i]
    #                 self.widths[i] += 1
    #             idrecs_to_inds_i[idrec] = ind


    if(n_vars > 1):
        resize_truth_table(self)

    # print("C")

    # return

    match_head_ptrs_func = _func_from_address(match_heads_f_type, self.op.match_head_ptrs_addr)


    match_inp_ptrs = np.zeros(len(self.op.head_var_ptrs),dtype=np.int64)
    match_idrecs = np.zeros(len(self.var_inds),dtype=np.uint64)
    match_inds = np.zeros(len(self.var_inds),dtype=np.int64)
    tt = self.truth_table

    # print("---", "~" if self.lit.negated else "", self.op, "---")
    # print(self.inputs[0].change_set)
    # print(self.inputs[0].change_set)
    # return 
    # for i, out in enumerate(self.outputs):
    #     out.change_set = Dict.empty(u8,u1)
    
    # print("WHATT")
    # print(arg_change_sets)
    # garbage = 0
    # return
    # (Beta 125 us)
    for i in range(n_vars):
        idrecs_to_inds_i = self.idrecs_to_inds[i]
        head_ptrs_i = self.head_ptr_buffers[i]
        i_strt, i_len = head_ranges[i][0], head_ranges[i][1]

        # if(j < i): continue
        # changed_idrecs_i = self.changed_idrecs[i]
        changed_inds_i = self.changed_inds[i]
        if(n_vars > 1):
            # print("BETA CASE", changed_inds_i)
            j = 1 if i == 0 else 0
            # p1, p2= self.inputs[i].parent_node_ptr, self.inputs[j].parent_node_ptr
            # inputs_same_parent = p1 != 0 and p1 == p2
            # BETA CASE 
            j_strt, j_len = head_ranges[j][0], head_ranges[j][1]        
            idrecs_to_inds_j = self.idrecs_to_inds[j]
            head_ptrs_j = self.head_ptr_buffers[j]
            input_state_buffers_i = self.input_state_buffers[i]
            input_state_buffers_j = self.input_state_buffers[j]

            # print("Z")

            for ind_i in changed_inds_i:
                inp_state_i  = input_state_buffers_i[ind_i]
                # if(not inp_state_i.head_was_valid): continue
                idrec_i = inp_state_i.idrec
                t_id_i, f_id_i, a_id_i = decode_idrec(idrec_i)
                idrec_i = encode_idrec(t_id_i, f_id_i, 0)
                # print("H", idrec_i, head_ptrs_i)
                match_inp_ptrs[i_strt:i_strt+i_len] = head_ptrs_i[ind_i]
                match_idrecs[i] = idrec_i
                match_inds[i] = ind_i#= idrecs_to_inds_i[idrec_i]

                # print("D",i, j)
                # print(self.widths[j], self.unchanged_inds[j])
                # if(i < j):
                #     for idrec_j,ind_j in arg_change_sets[j].items():
                #         tt[ind_i,ind_j] |= u1(0x2)
                
                # prev_matches_j = self.inputs[i].matches[idrec_i]
                # if(i < j)

                #NOTE...        

                # idrecs_j = self.unchanged_idrecs[j]
                # inds_j = self.unchanged_inds[j]

                # if(inputs_same_parent):
                #     # print("SAME PARENT", len(idrecs_to_inds_j))
                #     for idrec_j in _ld_dict(self.inputs[i].matches[idrec_i]):
                #         match_idrecs[j] = idrec_j
                #         ind_j = match_inds[j] = self.idrecs_to_inds[j][idrec_j]
                #         # if(i > j and tt[match_inds[0],match_inds[1]] & u1(0x2)): continue
                #         match_inp_ptrs[j_strt:j_strt+j_len] = head_ptrs_j[ind_j]
                #         _handle_beta_match(tt, negated, match_head_ptrs_func, match_inp_ptrs, match_inds, match_idrecs)
                # else:
                    # print("DIFF PARENT", len(idrecs_to_inds_j))
                # print(input_state_buffers_j)

                if(self.inputs_same_parent):
                    upstream_node = _struct_from_ptr(BaseReteNodeType, self.upstream_node_ptr)
                    u_tt = upstream_node.truth_table
                    if(j > i):
                        # Update the whole row/column
                        for ind_j in range(self.widths[j]):
                            inp_state_j = input_state_buffers_j[ind_j]
                            match_inds[j], match_idrecs[j] = ind_j, inp_state_j.idrec
                            match_inp_ptrs[j_strt:j_strt+j_len] = head_ptrs_j[ind_j]
                            # print(head_ptrs_j, ind_j)
                            if(not u_tt[match_inds[0],match_inds[1]]): continue
                            _handle_beta_match(tt, negated, inp_state_i, inp_state_j,
                                match_head_ptrs_func, match_inp_ptrs, match_inds, match_idrecs)
                    else:
                        # Check just the unchanged parts, so to avoid repeat checks 
                        for ind_j in self.unchanged_inds[j]:
                            inp_state_j = input_state_buffers_j[ind_j]
                            match_inds[j], match_idrecs[j] = ind_j, inp_state_j.idrec
                            match_inp_ptrs[j_strt:j_strt+j_len] = head_ptrs_j[ind_j]
                            if(not u_tt[match_inds[0],match_inds[1]]): continue
                            _handle_beta_match(tt, negated, inp_state_i, inp_state_j,
                                match_head_ptrs_func, match_inp_ptrs, match_inds, match_idrecs)
                else:
                    if(j > i):
                        # Update the whole row/column
                        for ind_j in range(self.widths[j]):
                            inp_state_j = input_state_buffers_j[ind_j]
                            match_inds[j], match_idrecs[j] = ind_j, inp_state_j.idrec
                            match_inp_ptrs[j_strt:j_strt+j_len] = head_ptrs_j[ind_j]
                            # print(head_ptrs_j, ind_j)
                            _handle_beta_match(tt, negated, inp_state_i, inp_state_j,
                                match_head_ptrs_func, match_inp_ptrs, match_inds, match_idrecs)
                    else:
                        # Check just the unchanged parts, so to avoid repeat checks 
                        for ind_j in self.unchanged_inds[j]:
                            inp_state_j = input_state_buffers_j[ind_j]
                            match_inds[j], match_idrecs[j] = ind_j, inp_state_j.idrec
                            match_inp_ptrs[j_strt:j_strt+j_len] = head_ptrs_j[ind_j]
                            _handle_beta_match(tt, negated, inp_state_i, inp_state_j,
                                match_head_ptrs_func, match_inp_ptrs, match_inds, match_idrecs)
                # if(i < j):
                #     for idrec_j,ind_j in arg_change_sets[j].items():
                #         tt[ind_i,ind_j] &= u1(0x1)
        else:
            # print("ALPHA CASE", changed_inds_i, len(input_state_buffers_i))
            input_state_buffers_i = self.input_state_buffers[i]
            # ALPHA CASE 
            for ind_i in changed_inds_i:
                inp_state_i=  input_state_buffers_i[ind_i]
                # if(not inp_state_i.head_was_valid): continue
                idrec_i = inp_state_i.idrec
                t_id_i, f_id_i, a_id_i = decode_idrec(idrec_i)
                idrec_i = encode_idrec(t_id_i, f_id_i, 0)

                match_inp_ptrs[i_strt:i_strt+i_len] = head_ptrs_i[ind_i]

                # print("alpha",match_inp_ptrs)
                is_match = match_head_ptrs_func(match_inp_ptrs) ^ negated

                inp_state_i.true_count = i8(is_match)
            # print("ALPHA CASE END")

                # ind0, ind1 = match_inds[0], match_inds[1]
                # was_match = tt[ind0, ind1]
                # is_match = match_head_ptrs_func(match_inp_ptrs) ^ negated
                # tt[ind0, ind1] = u1(is_match)
                # count_diff = i8(is_match) - i8(was_match)
                # inp_state_i.true_count += count_diff
                # inp_state_j.true_count += count_diff
                
                # if(is_match):
                #     insert_alpha_match(self,idrec_i)
                # else:
                #     invalidate_alpha_match(self,idrec_i)

        # for k, inp_s in enumerate(self.input_state_buffers[i]):
        #     print(i,k, inp_s.true_count)
            # print(alpha_matches_to_str(self.outputs[0].matches))

    # if(n_vars > 1):
    #     print(beta_matches_to_str(self.outputs[0].matches))
    # else:
    #     print(alpha_matches_to_str(self.outputs[0].matches))   
    # print("LOOP")
    # if(n_vars > 1):
    for i, out_i in enumerate(self.outputs):
        change_ind = 0
        match_ind = 0
        input_state_buffers_i = self.input_state_buffers[i]
        # idrecs_change_buffers_i = self.idrecs_change_buffers[i]

        # match_idrecs_buffer_i = out_i.match_idrecs_buffer
        # match_inds_buffer_i = out_i.match_inds_buffer
        # change_buffer_i = out_i.change_buffer

        # idrecs_match_buffers_i = self.idrecs_match_buffers[i]
        # inds_match_buffers_i = self.inds_match_buffers[i]
        
        for k in range(self.widths[i]):
            input_state = input_state_buffers_i[k]
            idrec_i = input_state.idrec
            t_id_i, f_id_i, a_id_i = decode_idrec(idrec_i)

            true_is_nonzero = (input_state.true_count != 0)
            input_state.true_ever_nonzero |= true_is_nonzero

            # print(k, input_state.true_ever_nonzero, match_ind, input_state.true_was_nonzero, true_is_nonzero, change_ind)

            if(input_state.true_ever_nonzero):
                # Insert match idrec or Leave a hole if ever_nonzero
                idrec = encode_idrec(t_id_i, f_id_i, DECLARE) if true_is_nonzero else u8(0)
                # idrecs_match_buffers_i[match_ind] = idrec

                # Expand match buffers if necessary
                # print("<<", match_ind, decode_idrec(idrec)[1], k)
                node_memory_insert_match_buffers(out_i, match_ind, idrec, k)
                match_ind += 1


                # buff_len = len(match_idrecs_buffer_i)
                # if(match_ind >= buff_len):
                #     new_idrecs_buffer = np.empty(buff_len*2, dtype=np.uint64)
                #     new_idrecs_buffer[:buff_len] = match_idrecs_buffer_i
                #     match_idrecs_buffer_i = out_i.match_idrecs_buffer = new_idrecs_buffer

                #     new_inds_buffer = np.empty(buff_len*2, dtype=np.int64)
                #     new_inds_buffer[:buff_len] = match_inds_buffer_i
                #     match_inds_buffer_i = out_i.match_inds_buffer = new_inds_buffer

                # match_idrecs_buffer_i[match_ind] = k
                

            # If true_was_nonzero flipped then insert into the change_set
            # print(i, input_state.true_count, input_state.true_was_nonzero)
            if(input_state.true_was_nonzero != true_is_nonzero):
                t_id_i, f_id_i, _ = decode_idrec(idrec_i)
                a_id = DECLARE if true_is_nonzero else RETRACT
                idrec_i = encode_idrec(t_id_i, f_id_i, a_id)
                # if(true_is_nonzero):
                #     idrec_i = encode_idrec(t_id_i, f_id_i, DECLARE)
                # else:
                #     idrec_i = encode_idrec(t_id_i, f_id_i, RETRACT)

                node_memory_insert_change_buffer(out_i, change_ind, idrec_i)
                change_ind += 1

                # Expand change buffer if necessary
                # buff_len = len(change_buffer_i)
                # if(change_ind >= buff_len):
                #     new_change_buffer = np.empty(buff_len*2, dtype=np.uint64)
                #     new_change_buffer[:buff_len] = change_buffer_i
                #     change_buffer_i = out_i.change_buffer = new_change_buffer

                # change_buffer_i[change_ind] = idrec_i
                # inds_change_buffers_i[change_ind] = k
                

            input_state.true_was_nonzero = true_is_nonzero
        
        out_i.match_idrecs = out_i.match_idrecs_buffer[:match_ind]
        out_i.match_inds = out_i.match_inds_buffer[:match_ind]
        out_i.change_set = out_i.change_buffer[:change_ind]

        # print("match_idrecs", out_i.match_idrecs)
        # print("match_inds", out_i.match_inds)
        # print("change_set", np.array([decode_idrec(x)[1] for x in out_i.change_set]))
            















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
    "conds" : ConditionsType,
    "nodes_by_nargs" : ListType(ListType(BaseReteNodeType)),
    "var_root_nodes" : DictType(i8,BaseReteNodeType),
    "var_end_nodes" : DictType(i8,BaseReteNodeType),
    "global_deref_idrec_map" : DictType(u8, node_list_type),
    "global_t_id_root_memory_map" : DictType(u2, NodeMemoryType),
    # "var_t_ids" : u2[::1], #NOTE: Something wrong with this... consider just not keeping
    # "match_iter_prototype_meminfo" : meminfo_type, #NOTE: Should really use deferred type
    "match_iter_prototype_ptr" : ptr_t, #NOTE: Should really use deferred type
}


ReteGraph, ReteGraphType = define_structref("ReteGraph", rete_graph_field_dict, define_constructor=False)

@njit(cache=True)
def rete_graph_ctor(mem, conds, nodes_by_nargs, var_root_nodes, var_end_nodes,
                    global_deref_idrec_map, global_t_id_root_memory_map):
    st = new(ReteGraphType)
    st.change_head = 0
    st.mem = mem
    st.conds = conds
    st.nodes_by_nargs = nodes_by_nargs
    st.var_root_nodes = var_root_nodes
    st.var_end_nodes = var_end_nodes
    st.global_deref_idrec_map = global_deref_idrec_map
    st.global_t_id_root_memory_map = global_t_id_root_memory_map
    # st.var_t_ids = var_t_ids 
    st.match_iter_prototype_ptr = 0
    
    return st


# @njit(void(ReteGraphType))
# def rete_graph_dtor(self):
#     if(self.match_iter_prototype_ptr != 0):
#         # NOTE we don't need to clean up anything else if we do the array thing
#         _decref_ptr(self.match_iter_prototype_ptr)


@njit(ConditionsType(ReteGraphType,), cache=True)
def rete_graph_get_conds(self):
    return self.conds


@njit(cache=True)
def _global_map_insert(idrec, g_map, node):
    if(idrec not in g_map):
        g_map[idrec] = List.empty_list(BaseReteNodeType)
    g_map[idrec].append(node)


@njit(cache=True)
def _get_var_t_id(mem, var):
    t_id = mem.context_data.fact_to_t_id.get(var.base_type_name,None)
    if(t_id is None): raise ValueError("Base Vars of Conditions() must inherit from Fact")
    return t_id

@njit(cache=True)
def _ensure_long_enough(nodes_by_nargs, nargs):
    while(len(nodes_by_nargs) <= nargs-1):
        nodes_by_nargs.append(List.empty_list(BaseReteNodeType))

ReteNode_List_type = ListType(BaseReteNodeType)

@njit(cache=True,locals={})
def _make_rete_nodes(mem, c, index_map):
    nodes_by_nargs = List.empty_list(ReteNode_List_type)
    nodes_by_nargs.append(List.empty_list(BaseReteNodeType))
    global_deref_idrec_map = Dict.empty(u8, node_list_type)
    # global_t_id_root_memory_map = Dict.empty(u8, node_list_type)

    fnum_to_t_id = mem.context_data.fact_num_to_t_id
    # fname_to_t_id = mem.context_data.fact_to_t_id

    # Make an identity node (i.e. lit,op=None) so there are always alphas
    for j in range(len(c.vars)):
        t_ids = np.empty((1,),dtype=np.uint16)
        var_inds = np.empty((1,),dtype=np.int64)
        base_var = c.vars[j]
        t_ids[0] = _get_var_t_id(mem, base_var)
        var_inds[0] = index_map[i8(base_var.base_ptr)]
        # _ensure_long_enough(nodes_by_nargs, 1)
        nodes_by_nargs[0].append(node_ctor(mem, t_ids, var_inds,lit=None))


    for distr_conjuct in c.distr_dnf:
        for j, var_conjuct in enumerate(distr_conjuct):

            # if(len(var_conjuct)==0):
                
                # continue

            for lit in var_conjuct:
                nargs = len(lit.var_base_ptrs)
                _ensure_long_enough(nodes_by_nargs, nargs)

                t_ids = np.empty((nargs,),dtype=np.uint16)
                var_inds = np.empty((nargs,),dtype=np.int64)
                for i, base_var_ptr in enumerate(lit.var_base_ptrs):
                    var_inds[i] = index_map[i8(base_var_ptr)]
                    base_var = _struct_from_ptr(GenericVarType, base_var_ptr)
                    t_id = _get_var_t_id(mem, base_var)
                    t_ids[i] = t_id

                # print("t_ids", t_ids)
                node = node_ctor(mem, t_ids, var_inds, lit)
                nodes_by_nargs[nargs-1].append(node)
                # print("<< aft", lit.op.head_var_ptrs)
                for i, head_var_ptr in enumerate(lit.op.head_var_ptrs):
                    head_var = _struct_from_ptr(GenericVarType, head_var_ptr)
                    ind = np.min(np.nonzero(lit.var_base_ptrs==i8(head_var.base_ptr))[0])
                    t_id = t_ids[ind]
                    # print("START")
                    for d_offset in head_var.deref_offsets:
                        idrec1 = encode_idrec(u2(t_id),0,u1(d_offset.a_id))
                        _global_map_insert(idrec1, global_deref_idrec_map, node)
                        # print("-idrec1", decode_idrec(idrec1))
                        fn = d_offset.fact_num
                        if(fn >= 0 and fn < len(fnum_to_t_id)):
                            t_id = fnum_to_t_id[d_offset.fact_num]
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
    # print("A")
    index_map = Dict.empty(i8, i8)
    # var_t_ids = np.zeros((len(c.vars),), dtype=np.uint16)

    for i, v in enumerate(c.vars):
        index_map[i8(v.base_ptr)] = i

        # base_var = _struct_from_ptr(GenericVarType, v.base_ptr)
        # if(base_var.base_type_name in mem.context_data.fact_to_t_id):
        #     var_t_ids[i] = u2(mem.context_data.fact_to_t_id[base_var.base_type_name])
        # if(_t_id is not None):
        #     t_id = u2(i8(_t_id))
        #     var_t_ids[i] = t_id
    # print("B")
    if(not c.has_distr_dnf):
        build_distributed_dnf(c,index_map)
    # print("C")
    nodes_by_nargs, global_deref_idrec_map = \
         _make_rete_nodes(mem, c, index_map)

    global_t_id_root_memory_map = Dict.empty(u2, NodeMemoryType)
    var_end_nodes = Dict.empty(i8,BaseReteNodeType)
    var_root_nodes = Dict.empty(i8,BaseReteNodeType)

    # print("D")
    # Link nodes together. 'nodes_by_nargs' should already be ordered
    # so that alphas are before 2-way, 3-way, etc. betas. 
    for i, nodes in enumerate(nodes_by_nargs):
        for node in nodes:
            inputs = List.empty_list(NodeMemoryType)
            for j, ind in enumerate(node.var_inds):
                # print("ind", ind)
                if(ind in var_end_nodes):
                    # Extract the appropriate NodeMemory for this input
                    e_node = var_end_nodes[ind]
                    om = e_node.outputs[np.min(np.nonzero(e_node.var_inds==ind)[0])]
                    inputs.append(om)
                    node.upstream_node_ptr = _raw_ptr_from_struct(e_node)
                else:
                    t_id = node.t_ids[j]
                    if(t_id not in global_t_id_root_memory_map):
                        root = new_root_node_mem()
                        global_t_id_root_memory_map[t_id] = root
                    root = global_t_id_root_memory_map[t_id]


                    inputs.append(root)
                    var_root_nodes[ind] = node

            node.inputs = inputs

            if(len(inputs) == 2):
                p1,p2 = inputs[0].parent_node_ptr, inputs[1].parent_node_ptr
                node.inputs_same_parent = (p1 != 0 and p1 == p2)
            else:
                node.inputs_same_parent = False

            # In 'inputs_same_parent' cases make nodes share idrec_to_inds, widths, change_pairs
            if(node.inputs_same_parent):
                assert node.upstream_node_ptr != 0
                upstream_node = _struct_from_ptr(BaseReteNodeType, node.upstream_node_ptr)
                node.idrecs_to_inds = upstream_node.idrecs_to_inds
                node.widths = upstream_node.widths
                node.change_pairs = upstream_node.change_pairs

            # Short circut the input to the output for identity nodes
            if(node.lit is None):
                node.outputs = node.inputs

            # Make this node the new end node for the vars it takes as inputs
            for ind in node.var_inds:
                var_end_nodes[ind] = node
            
            
            # print("<<",len(inputs), node.var_inds)

    # print(var_end_nodes)


    return rete_graph_ctor(mem, c, nodes_by_nargs, var_root_nodes,
             var_end_nodes, global_deref_idrec_map, global_t_id_root_memory_map)

@njit(cache=True)
def node_memory_insert_change_buffer(self, k, idrec):
    buff_len = len(self.change_buffer)
    if(k >= buff_len):
        expand = max(k-buff_len, buff_len)
        new_change_buffer = np.empty(expand+buff_len,dtype=np.uint64)
        new_change_buffer[:buff_len] = self.change_buffer
        self.change_buffer = new_change_buffer
    self.change_buffer[k] = idrec
    

@njit(cache=True)
def node_memory_insert_match_buffers(self, k, idrec, ind):
    buff_len = len(self.match_idrecs_buffer)
    if(k >= buff_len):
        expand = max(k-buff_len, buff_len)

        new_idrecs_buffer = np.empty(expand+buff_len, dtype=np.uint64)
        new_idrecs_buffer[:buff_len] = self.match_idrecs_buffer
        self.match_idrecs_buffer = new_idrecs_buffer

        new_inds_buffer = np.empty(expand+buff_len, dtype=np.int64)
        new_inds_buffer[:buff_len] = self.match_inds_buffer
        self.match_inds_buffer = new_inds_buffer

        
    self.match_idrecs_buffer[k] = idrec
    self.match_inds_buffer[k] = ind



@njit(cache=True,locals={"t_id" : u2, "f_id":u8, "a_id":u1})
def parse_mem_change_queue(r_graph):
    # print("START PARSE")
    
    global_deref_idrec_map = r_graph.global_deref_idrec_map
    global_t_id_root_memory_map = r_graph.global_t_id_root_memory_map
    mem_data = r_graph.mem.mem_data
    change_queue = mem_data.change_queue

    # for t_id, root_mem in global_t_id_root_memory_map.items():
    #     facts = _struct_from_ptr(VectorType, mem.mem_data.facts[t_id])
    #     if(len(root_mem.match_idrecs_buffer) < len(facts.data)):
    #         new_match_idrecs_buffer = np.empty(len(facts.data),dtype=np.uint64)
    #         new_match_idrecs_buffer[]
    #         for f_id in range(len(facts.data)):

    #             root_mem.match_idrecs_buffer[]
    #         root_mem.match_idrecs_buffer = new_match_idrecs_buffer






    # for idrec in global_deref_idrec_map:
    #     print("<< global_deref_idrec_map", decode_idrec(idrec))
    # print(global_deref_idrec_map)
    
    for t_id, root_mem in global_t_id_root_memory_map.items():
        root_mem.change_set = root_mem.change_buffer[:0]


    # print("**:", change_queue.data[:change_queue.head])
    # print("**:", global_t_id_root_memory_map)
    for i in range(r_graph.change_head, change_queue.head):
        idrec = u8(change_queue[i])
        t_id, f_id, a_id = decode_idrec(idrec)
        # print(t_id, f_id, a_id)

        # Add this idrec to change_set of root nodes
        # root_node_mem = global_t_id_root_memory_map.get(t_id,None)
        if(t_id in global_t_id_root_memory_map):
            # print("T_ID", t_id)
            root_mem = global_t_id_root_memory_map[t_id]

            k = len(root_mem.change_set)
            node_memory_insert_change_buffer(root_mem, k, encode_idrec(t_id,f_id,0))
            root_mem.change_set = root_mem.change_buffer[:k+1]


            idrec = encode_idrec(t_id,f_id,0) if(a_id != RETRACT) else u8(0)
            node_memory_insert_match_buffers(root_mem, i8(f_id), idrec, i8(f_id))
            if(i8(f_id) >= len(root_mem.match_idrecs)):
                root_mem.match_idrecs = root_mem.match_idrecs_buffer[:i8(f_id)+1]
                root_mem.match_inds = root_mem.match_inds_buffer[:i8(f_id)+1]

            # # Ensure change buffers are the right size
            # buff_len = len(root_mem.change_buffer)
            # if(buff_len == len(root_mem.change_set)):
            #     new_change_buffer = np.empty(buff_len*2,dtype=np.uint64)
            #     new_change_buffer[:buff_len] = root_mem.change_buffer
            #     root_mem.change_buffer = new_change_buffer

            # Ensure match buffers are the right size
            # buff_len = len(root_mem.match_idrecs_buffer)
            # if(buff_len == len(root_mem.change_set)):
            #     new_match_idrecs_buff = np.empty(buff_len*2,dtype=np.uint64)
            #     new_match_idrecs_buff[:buff_len] = root_mem.change_buffer
            #     root_mem.change_buffer = new_match_idrecs_buff

            # root_mem.change_buffer[len(root_mem.change_set)] = encode_idrec(t_id,f_id,0)





            # root_mem.change_set[idrec] = u1(1)
            # fact_idrec = encode_idrec(t_id,f_id,0)
            # if(a_id == DECLARE):
            #     root_mem.matches[fact_idrec] = 1
            # elif(a_id == RETRACT):
            #     root_mem.matches[fact_idrec] = 0

        # else:
            # print("~T_ID", t_id, List(global_t_id_root_memory_map.keys()))

        # Add this idrec to relevant deref idrecs
        # idrec_pattern = encode_idrec(t_id, 0, a_id)
        idrec_pattern = encode_idrec(t_id, 0, a_id)
        nodes = global_deref_idrec_map.get(idrec_pattern,None)
        if(nodes is not None):
            for node in nodes:
                node.relevant_global_diffs.add(idrec)



       
        # print(decode_idrec(idrec))

    # print(r_graph.global_deref_idrec_map)
    # print(r_graph.global_t_id_root_memory_map)
    r_graph.change_head = change_queue.head


    # for t_id, root_mem in global_t_id_root_memory_map.items():
    #     print("!",root_mem.change_set)
    #     print("!",root_mem.change_buffer)
    #     print("@",root_mem.match_idrecs)
    #     print("#",root_mem.match_inds)

    # print("PARSE DONE")

    # print(r_graph.var_root_nodes)
    # for root_node in r_graph.var_root_nodes.values():
    #     print("R")
    #     for i, inp in enumerate(root_node.inputs):
    #         print(inp.change_set)
        # for idrec in root_node.relevant_global_diffs.data:
        #     print(decode_idrec(idrec))
            # print(_raw_ptr_from_struct(inp.change_set))
        # print(root_node.inputs[0].change_set)




    # r_graph.mem.change_queue


match_iterator_node_field_dict = {
    "graph" : ReteGraphType,
    "node" : BaseReteNodeType,
    "associated_arg_ind" : i8,
    "depends_on_var_ind" : i8,
    "var_ind" : i8,
    # "is_exhausted": boolean,
    "curr_ind": i8,
    "idrecs" : u8[::1],
    "other_idrecs" : u8[::1],
}



MatchIterNode, MatchIterNodeType = define_structref("MatchIterNode", match_iterator_node_field_dict, define_constructor=False)





match_iterator_field_dict = {
    "graph" : ReteGraphType,
    "iter_nodes" : ListType(MatchIterNodeType),
    "is_empty" : types.boolean,
    "output_types" : types.Any
    # "curr_match" : u8[::1],
}
match_iterator_fields = [(k,v) for k,v, in match_iterator_field_dict.items()]


def gen_match_iter_source(output_types):
    return f'''import cloudpickle
from numba import njit
from cre.rete import GenericMatchIteratorType, MatchIteratorType, match_iterator_field_dict
from cre.utils import _cast_structref
output_types = cloudpickle.loads({cloudpickle.dumps(output_types)})
m_iter_type = MatchIteratorType([(k,v) for k,v in {{**match_iterator_field_dict ,"output_types": output_types}}.items()])

@njit(m_iter_type(GenericMatchIteratorType),cache=True)
def specialize_m_iter(self):
    return _cast_structref(m_iter_type,self)
    '''




class MatchIterator(structref.StructRefProxy):
    ''' '''
    # def __init__(self):
    #     super().__init__()
    m_iter_type_cache = {}
    def __new__(cls, mem, conds):
        # Make a generic MatchIterator (reuses graph if conds already has one)
        # with PrintElapse("get_match_iter"):
        # with mem.context as ctx:
        generic_m_iter = get_match_iter(mem, conds)

        #Cache 'output_types' and 'specialized_m_iter_type'
        var_base_types = conds.var_base_types


        if(var_base_types not in cls.m_iter_type_cache):
            hash_code = unique_hash([var_base_types])
            if(not source_in_cache('MatchIterator', hash_code)):
                output_types = types.TypeRef(types.Tuple([types.TypeRef(x) for x in conds.var_base_types]))
                source = gen_match_iter_source(output_types)
                source_to_cache('MatchIterator', hash_code, source)
            l = import_from_cached('MatchIterator', hash_code, ['specialize_m_iter', 'output_types'])
            # op_cls = self._generate_op_cls = l['MatchIterator']

            # specialized_m_iter_type = MatchIteratorType([(k,v) for k,v in {**match_iterator_field_dict ,"output_types": output_types}.items()])
            output_types, specialize_m_iter = cls.m_iter_type_cache[var_base_types] = l['output_types'], l['specialize_m_iter']
        else:
            output_types, specialize_m_iter  = cls.m_iter_type_cache[var_base_types]
        
        # Specialize the match iter so that it outputs conds.var_base_types 
        self = specialize_m_iter(generic_m_iter)
        self.output_types = output_types#tuple([types.TypeRef(x) for x in conds.var_base_types])
        return self
        
    def __next__(self):
        # ptrs = match_iter_next_ptrs(self)
        # print(ptrs)
        # return fact_ptrs_as_tuple(self.output_types, ptrs)
        # with PrintElapse("match_iter_next"):
        return match_iter_next(self)

        # return match_iter_next(self)
    def __iter__(self):
        return self



@structref.register
class MatchIteratorType(CastFriendlyMixin, types.StructRef):
    def __str__(self):
        return "cre.MatchIterator"
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


define_boxing(MatchIteratorType, MatchIterator)
GenericMatchIteratorType = MatchIteratorType(match_iterator_fields)

# Allow any specialization of MatchIteratorType to be downcast to GenericMatchIteratorType
@lower_cast(MatchIteratorType, GenericMatchIteratorType)
def downcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty)




# MatchIter, MatchIteratorType = define_structref("MatchIter", match_iterator_field_dict)

# @njit(cache=True)
# def copy_iter_nodes()


@njit(GenericMatchIteratorType(GenericMatchIteratorType),cache=True)
def copy_match_iter(m_iter):
    m_iter_nodes = List.empty_list(MatchIterNodeType)
    for i,m_node in enumerate(m_iter.iter_nodes):
        new_m_node = new(MatchIterNodeType)
        new_m_node.graph = m_node.graph
        new_m_node.node = m_node.node
        new_m_node.var_ind = m_node.var_ind
        new_m_node.associated_arg_ind = m_node.associated_arg_ind
        new_m_node.depends_on_var_ind = m_node.depends_on_var_ind
        new_m_node.curr_ind = m_node.curr_ind
        if(m_node.curr_ind != -1):
            new_m_node.idrecs = m_node.idrecs
        m_iter_nodes.append(new_m_node)

    new_m_iter = new(GenericMatchIteratorType)
    new_m_iter.graph = m_iter.graph 
    new_m_iter.iter_nodes = m_iter_nodes 
    new_m_iter.is_empty = m_iter.is_empty

    return new_m_iter

@njit(GenericMatchIteratorType(ReteGraphType), cache=True)
def new_match_iter(graph):
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
            m_node.associated_arg_ind = np.argmax(node.var_inds==i)#node.outputs[np.argmax(node.var_inds==i)]

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

        m_iter = new(GenericMatchIteratorType)
        m_iter.graph = graph 
        m_iter.iter_nodes = rev_m_iter_nodes 
        m_iter.is_empty = False
        # graph.match_iter_prototype_meminfo = _meminfo_from_struct(m_iter)
        graph.match_iter_prototype_ptr = _ptr_from_struct_incref(m_iter)
    
    prototype = _struct_from_ptr(GenericMatchIteratorType, graph.match_iter_prototype_ptr)
    m_iter = copy_match_iter(prototype)
    
    return m_iter


@njit(unicode_type(GenericMatchIteratorType),cache=True)
def repr_match_iter_dependencies(m_iter):
    rep = ""
    for i, m_node in enumerate(m_iter.iter_nodes):
        s = f'({str(m_node.var_ind)}'
        if(m_node.depends_on_var_ind != -1):
            # dep = _struct_from_ptr(MatchIterNodeType, )
            s += f",{str(m_node.depends_on_var_ind)})"
        else:
             s += f")"
        rep += s
        if(i < len(m_iter.iter_nodes)-1): rep += " "

    return rep



    # return MatchIter(rev_m_iter_nodes)

@njit(types.void(MatchIterNodeType),cache=True)
# @njit(cache=True)
def update_other_idrecs(m_node):
    # print("START UPDATE OTHER")
    if(len(m_node.node.var_inds) > 1):
        # print("A", m_node.idrecs, m_node.curr_ind)
        # matches = m_node.associated_output.matches_inds
        
        # matches_inds = m_node.outputs[arg_ind]

        # f_id = m_node.idrecs[m_node.curr_ind]
        # print("B", matches, m_node.f_ids, m_node.curr_ind)

        # other_idrecs_d = _dict_from_ptr(dict_i8_u1_type, matches[f_id]) 
        # print("C", other_f_ids_d)
        # print("C", )
        # print("C", len(other_f_ids_d))
        arg_ind = m_node.associated_arg_ind
        other_ind = 0 if arg_ind == 1 else 1
        node = m_node.node
        associated_output = node.outputs[arg_ind]
        this_internal_ind = associated_output.match_inds[m_node.curr_ind]

        truth_table = node.truth_table
        input_states = node.input_state_buffers[other_ind]
        
        other_idrecs = np.empty((len(associated_output.match_inds)), dtype=np.uint64)
        k = 0
        if(arg_ind == 0):
            for i, t in enumerate(truth_table[this_internal_ind, :]):
                if(t): other_idrecs[k] = input_states[i].idrec; k += 1
        else:        
            for i, t in enumerate(truth_table[:, this_internal_ind]):
                if(t): other_idrecs[k] = input_states[i].idrec; k += 1

        m_node.other_idrecs = other_idrecs[:k]
        # this_input_state = m_node.node.truth_table[arg_ind][]
        # # this_ind = 
        # other_idrecs = np.empty((len(other_input_state),), dtype=np.uint64)
        # print("F")
        # cnt = 0
        # for j, (o_f_id,v) in enumerate(other_input_states):
        #     if(v == 0): continue
        #     other_idrecs[j] = o_f_id; cnt += 1;
        # m_node.other_idrecs = other_idrecs[:cnt]
        # print("S UPDATE OTHER", m_node.other_f_ids)
        # print("Z")
    # print("END UPDATE OTHER")



@njit(types.void(GenericMatchIteratorType,i8), cache=True)
def restitch_match_iter(m_iter, start_from):
    if(start_from == -1): start_from = len(m_iter.iter_nodes)-1
    for i in range(start_from,-1,-1):
        # print("I", i)
        m_node = m_iter.iter_nodes[i]
        if(m_node.curr_ind == -1):
            if(m_node.depends_on_var_ind == -1):
                # print("A")
                # matches = m_node.associated_output.matches
                associated_output = m_node.node.outputs[m_node.associated_arg_ind]
                matches = associated_output.match_idrecs
                idrecs = np.empty((len(matches)),dtype=np.uint64)
                cnt = 0
                # print("B", matches, len(m_node.node.var_inds))
                # print(matches)
                # for j, (f_id, v) in enumerate(matches.items()):
                for j, idrec in enumerate(matches):
                    if(idrec == 0): continue
                    idrecs[j] = idrec; cnt += 1;
                m_node.idrecs = idrecs[:cnt]

                # print("N_IDRECS",m_node.idrecs)
                # if(i == start_from and len(m_node.idrecs) == 0):
                #     print("EMPTY")


                # print("C", cnt)
                # print(m_node.f_ids)
                
                
            else:
                dep_node = m_iter.iter_nodes[m_node.depends_on_var_ind]
                m_node.idrecs = dep_node.other_idrecs
                # m_node.curr_ind = 0
            m_node.curr_ind = 0
        if(i > 0):
            # print("SS")
            update_other_idrecs(m_node)
            # print("S UPDATE OTHER", i, m_node.other_idrecs)
        # else:
        #     print("CURRIND", m_node.curr_ind)
            # m_node.curr_ind = 0

@njit(u8[::1](GenericMatchIteratorType),cache=True)
def match_iter_next_idrecs(m_iter):
    n_vars = len(m_iter.iter_nodes)
    if(m_iter.is_empty or n_vars == 0): raise StopIteration()

    idrecs = np.empty(n_vars,dtype=np.uint64)
    most_downstream_overflow = -1
    for i, m_node in enumerate(m_iter.iter_nodes):
        # print("RESTICH")
        # print(i, ":", m_node.curr_ind, m_node.f_ids)
        idrecs[m_node.var_ind] = m_node.idrecs[m_node.curr_ind]

        update_other_needed = False
        if(i == 0):
            m_node.curr_ind += 1
        elif(most_downstream_overflow == i-1):
            m_node.curr_ind += 1
            update_other_needed = True

        if(m_node.curr_ind >= len(m_node.idrecs)):
            if(i == n_vars-1):
                m_iter.is_empty = True
                return idrecs
            m_node.curr_ind = -1
            most_downstream_overflow = i
        else:
            update_other_idrecs(m_node)


    if(most_downstream_overflow != -1):
        restitch_match_iter(m_iter, most_downstream_overflow)

    return idrecs

@njit(i8[::1](GenericMatchIteratorType), cache=True)
def match_iter_next_ptrs(m_iter):
    mem, graph = m_iter.graph.mem, m_iter.graph
    idrecs = match_iter_next_idrecs(m_iter)
    # print("^^", idrecs)
    ptrs = np.empty(len(idrecs),dtype=np.int64)
    for i, idrec in enumerate(idrecs):
        t_id, f_id, _  = decode_idrec(idrec)
        facts = _struct_from_ptr(VectorType, mem.mem_data.facts[t_id])
        ptrs[i] = facts.data[f_id]
    # print("SLOOP")
    return ptrs

@njit(cache=True)
def match_iter_next(m_iter):
    ptrs = match_iter_next_ptrs(m_iter)
    # print("PTRS", ptrs)
    tup = _struct_tuple_from_pointer_arr(m_iter.output_types, ptrs)
    # print("tup")
    # print(tup)
    return tup


@njit(cache=True)
def fact_ptrs_as_tuple(typs, ptr_arr):
    # print("BEFORE")
    tup = _struct_tuple_from_pointer_arr(typs, ptr_arr)
    # print("AFTER",tup)
    return tup

# @generated_jit(cache=True,nopython=True)
# def _get_matches(conds, struct_types, mem=None):
#     print(type(struct_types))
#     print(struct_types)
#     if(isinstance(struct_types, UniTuple)):
#         typs = tuple([struct_types.dtype.instance_type] * struct_types.count)
#         out_type =  UniTuple(struct_types.dtype.instance_type,struct_types.count)
#     else:
#         raise NotImplemented("Need to write intrinsic for multi-type ")

    # print(typs)
# @njit(cache=True, locals={"ptr_set" : i8[::1]})
# def match_iter_next(m_iter, struct_types):
#     ptrs = match_iter_next_ptrs(m_iter)
#     return _struct_tuple_from_pointer_arr(struct_types, ptr_set)


# @njit(cache=True)
# def match_iter_next(m_iter, types):
#     mem, graph = m_iter.graph.mem, m_iter.graph
#     idrecs = match_iter_next_idrecs(m_iter)
#     ptrs = np.empty(len(idrecs),dtype=np.int64)
#     for i, idrec in enumerate(idrecs):
#         t_id, f_id, _  = decode_idrec(idrec)
#         facts = _struct_from_ptr(VectorType, mem.mem_data.facts[t_id])
#         ptrs[i] = facts.data[f_id]
#     return ptrs


# @njit(cache=True)
# def match_iter_next(m_iter):
#     return match_iter_next_ptrs(m_iter)

@njit(cache=True)
def update_graph(graph):
    # print("START UP")
    parse_mem_change_queue(graph)
    for lst in graph.nodes_by_nargs:
        for node in lst:
            update_node(node)        
    # print("END UPDATE")


@njit(GenericMatchIteratorType(MemoryType, ConditionsType), cache=True)
def get_match_iter(mem, conds):
    # print("START")
    if(i8(conds.matcher_inst_ptr) == 0):
        rete_graph = build_rete_graph(mem, conds)
        # conds.matcher_inst_meminfo = _meminfo_from_struct(rete_graph)
        conds.matcher_inst_ptr = _ptr_from_struct_incref(rete_graph)
    # print("BUILT")
    rete_graph = _struct_from_ptr(ReteGraphType, conds.matcher_inst_ptr)
    update_graph(rete_graph)
    # print("UPDATED")
    m_iter = new_match_iter(rete_graph)
    # for i, m_node in enumerate(m_iter.iter_nodes):
    #     print("<<", m_node.curr_ind)
    # print("INITIAL RESTICH")
    restitch_match_iter(m_iter, -1)
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
