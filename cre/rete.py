import numpy as np
import numba
from numba import types, njit, i8, u8, i4, u1, u2, i8, f8, literally, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple
from numba.experimental.structref import new, define_boxing
import numba.experimental.structref as structref
from cre.utils import (wptr_t, ptr_t, _dict_from_ptr, _raw_ptr_from_struct, _get_array_raw_data_ptr,
         _ptr_from_struct_incref, _struct_from_ptr, decode_idrec, CastFriendlyMixin,
        encode_idrec, deref_info_type, DEREF_TYPE_ATTR, DEREF_TYPE_LIST, _obj_cast_codegen,
         _ptr_to_data_ptr, _list_base_from_ptr, _load_ptr, PrintElapse, meminfo_type,
         _decref_structref, _decref_ptr, cast_structref, _struct_tuple_from_pointer_arr, _meminfo_from_struct)
from cre.structref import define_structref
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.memset import MemSetType
from cre.vector import VectorType
from cre.var import GenericVarType
from cre.op import GenericOpType
from cre.conditions import LiteralType, build_distributed_dnf, ConditionsType
from cre.vector import VectorType, new_vector
from cre.fact import BaseFact 
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
    "insert_buffer" : u8[::1],#DictType(u8,u1), 
    "insert_set" : u8[::1],#DictType(u8,u1), 

    "remove_buffer" : u8[::1],#DictType(u8,u1), 
    "remove_set" : u8[::1],#DictType(u8,u1), 
    
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
    st.insert_buffer = np.empty(8,dtype=np.uint64)#Dict.empty(u8,u1)
    st.insert_set = st.insert_buffer

    st.remove_buffer = np.empty(8,dtype=np.uint64)#Dict.empty(u8,u1)
    st.remove_set = st.remove_buffer
    
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
    ('recently_inserted', np.uint8),
    ('recently_removed', np.uint8),
    ('is_removed', np.uint8),
    # The input had a match in the previous match cycle 
    ('true_was_nonzero', np.uint8), 

    # The input has ever had a match. Needed to keep track of holes in the match set. 
    #  holes are necessary to ensure match iterators are valid on backtracks.
    ('true_ever_nonzero', np.uint8),

    # Pad to align w/ i8[:3]
    ('_padding0', np.uint8),
    ('_padding1', np.uint8),
    # ('_padding2', np.uint8),
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
    "memset" : MemSetType, 
    "has_op" : types.boolean,
    "lit" : types.optional(LiteralType),
    "op" : types.optional(GenericOpType),
    "deref_depends" : deref_dep_typ, 
    
    "n_vars" : i8,

    "var_inds" : i8[::1],
    "t_ids" : u2[::1],
    # "vars" : ListType(GenericVarType),
    "idrecs_to_inds" : ListType(DictType(u8,i8)),
    "retracted_inds" : ListType(VectorType),
    "inp_widths" : i8[::1],
    "head_ptr_buffers" : ListType(i8[:,::1]),
    "input_state_buffers" : ListType(input_state_type[::1]),
    


    # "idrecs_insert_buffers" : ListType(u8[::1]),
    "inds_insert_buffers" : ListType(i8[::1]),
    "inds_remove_buffers" : ListType(i8[::1]),
    # "changed_idrecs" : ListType(u8[::1]),
    # "unchanged_idrecs" : ListType(u8[::1]),
    "inserted_inds" : ListType(i8[::1]),
    "unchanged_inds" : ListType(i8[::1]),
    "removed_inds" : ListType(i8[::1]),

    "idrecs_match_buffers" : ListType(u8[::1]),
    "inds_match_buffers" : ListType(i8[::1]),

    # "head_ptr_buffers" : i8[:,::1],

    # "head_ptrs" : ListType(DictType(u8,i8[::1])),
    "inputs" : ListType(NodeMemoryType),
    "outputs" : ListType(NodeMemoryType), #DictType(u8,DictType(u8,u1))
    "truth_table" : u1[:, ::1],

    # True if the inputs to this node both come from the same beta node
    # e.g. like in the case of (a.v < b.v) & (a.v != b.v)
    "upstream_same_parents" : u1,
    "upstream_aligned" : u1,

    # A weak pointer to the node upstream to this one
    "upstream_node_ptr" : i8,

    # # A temporary record array of inds and idrecs of upstream changes
    # "change_pairs" : ListType(idrec_ind_pair_type[::1])

    "modify_idrecs" : ListType(VectorType),

}

BaseReteNode, BaseReteNodeType = define_structref("BaseReteNode", base_rete_node_field_dict, define_constructor=False)


u8_arr_typ = u8[::1]
i8_arr_typ = i8[::1]
i8_x2_arr_typ = i8[:,::1]
input_state_arr_type = input_state_type[::1]
idrec_ind_pair_arr_type = idrec_ind_pair_type[::1]

@njit(cache=True)
def node_ctor(ms, t_ids, var_inds,lit=None):
    # print("NEW RETE NODE")
    st = new(BaseReteNodeType)
    st.memset = ms
    st.deref_depends = Dict.empty(u8,dict_i8_u1_type)
    st.modify_idrecs = List.empty_list(VectorType)
    st.var_inds = var_inds
    st.t_ids = t_ids

    # if(lit is not None):
    st.head_ptr_buffers = List.empty_list(i8_x2_arr_typ)
    st.input_state_buffers = List.empty_list(input_state_arr_type)

    # st.idrecs_insert_buffers = List.empty_list(u8_arr_typ)
    # st.changed_idrecs = List.empty_list(u8_arr_typ)
    # st.unchanged_idrecs = List.empty_list(u8_arr_typ)
    st.inds_insert_buffers = List.empty_list(i8_arr_typ)
    st.inds_remove_buffers = List.empty_list(i8_arr_typ)
    st.inserted_inds = List.empty_list(i8_arr_typ)
    st.unchanged_inds = List.empty_list(i8_arr_typ)
    st.removed_inds = List.empty_list(i8_arr_typ)

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
            ind_insert_buff = np.empty(8, dtype=np.int64)
            ind_remove_buff = np.empty(8, dtype=np.int64)
            # st.idrecs_insert_buffers.append(idrec_change_buff)
            # st.changed_idrecs.append(idrec_change_buff)
            # st.unchanged_idrecs.append(idrec_change_buff)
            st.inds_insert_buffers.append(ind_insert_buff)
            st.inds_remove_buffers.append(ind_remove_buff)
            st.inserted_inds.append(ind_insert_buff)
            st.removed_inds.append(ind_remove_buff)
            st.unchanged_inds.append(ind_insert_buff)
            st.modify_idrecs.append(new_vector(4))

            # idrec_buff = np.empty(8, dtype=np.uint64)
            # ind_buff = np.empty(8, dtype=np.int64)
            # st.idrecs_match_buffers.append(idrec_buff)
            # st.inds_match_buffers.append(ind_buff)
    else:
        st.op = None
        n_vars = 1

    # st.head_ptr_buffers = head_ptr_buffers
    # st.input_state_buffers = input_state_buffers
    # st.idrecs_insert_buffers = idrecs_insert_buffers
    # st.changed_idrecs = changed_idrecs
    # st.unchanged_idrecs = unchanged_idrecs
    # st.inds_insert_buffers = inds_insert_buffers
    # st.inserted_inds = inserted_inds
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
    # st.change_pairs = List.empty_list(idrec_ind_pair_arr_type) 
    st.inp_widths = np.zeros(2,dtype=np.int64)
    #  "idrecs_to_inds" : ListType(DictType(u8,i8)),
    # "retracted_inds" : ListType(VectorType),
    # "widths" : i8[::1],

    for i in range(n_vars):
        st.idrecs_to_inds.append(Dict.empty(u8,i8))
        st.retracted_inds.append(new_vector(8))
        # st.change_pairs.append(np.empty(0,dtype=idrec_ind_pair_type))

    # Just make False by default, can end up being True after linking
    st.upstream_same_parents = False
    st.upstream_aligned = False
    st.upstream_node_ptr = 0


    return st


@njit(i8(i8, deref_info_type[::1]), cache=True,locals={"data_ptr":i8, "inst_ptr":i8})
def deref_head_and_relevant_idrecs(inst_ptr, deref_infos):
    ''' '''
    
    # relevant_idrecs = np.zeros((max((len(deref_infos)-1)*2+1,0),), dtype=np.uint64)
    # print("N", len(relevant_idrecs))
    k = -1

    # ok = True

    for deref in deref_infos[:-1]:
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
        deref = deref_infos[-1]
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

@njit(i8(deref_info_type,i8),inline='never',cache=True)
def deref_once(deref,inst_ptr):
    if(deref.type == u1(DEREF_TYPE_ATTR)):
        return _ptr_to_data_ptr(inst_ptr)
    else:
        return _list_base_from_ptr(inst_ptr)


# @njit(i8(BaseReteNodeType, u2, u8, deref_info_type[::1]),cache=True)
@njit(cache=True)
def resolve_head_ptr(self, arg_ind, base_t_id, f_id, deref_infos):
    '''Try to get the head_ptr of 'f_id' in input 'arg_ind'. Inject a DerefRecord regardless of the result 
        Keep in mind that a head_ptr is the pointer to the address where the data is stored not the data itself.
    '''
    facts = _struct_from_ptr(VectorType, self.memset.facts[base_t_id])
    if(len(deref_infos) > 0):
        inst_ptr = facts.data[f_id]
        if(len(deref_infos) > 1):
            rel_idrecs = np.empty(len(deref_infos)-1, dtype=np.uint64)
            for k in range(len(deref_infos)-1):
                if(inst_ptr == 0): break;
                deref = deref_infos[k]
                data_ptr = deref_once(deref, inst_ptr)
                # print(f"{deref.type} inst_ptr {inst_ptr} -> {data_ptr+deref.offset}")
                # print("data_ptr", k, data_ptr)
                # Note: This won't retreive an idrec for lists 
                rel_idrecs[k] = _load_ptr(u8, data_ptr)
                inst_ptr = _load_ptr(i8, data_ptr+deref.offset)
                # print("inst_ptr", k, inst_ptr)

            # Inject a deref record so that we can track changes to intermediate facts
            parent_ptrs = rel_idrecs.astype(i8) 
            r = DerefRecord(parent_ptrs, arg_ind, encode_idrec(base_t_id,f_id,0), inst_ptr != 0) #7us
            r_ptr = _ptr_from_struct_incref(r)
            for i, idrec in enumerate(rel_idrecs): #21 us
                parent_ptrs[i] = i8(make_deref_record_parent(self.deref_depends, idrec, r_ptr))            

        if(inst_ptr != 0):
            deref = deref_infos[-1]
            data_ptr = deref_once(deref, inst_ptr)
            # print(f"{deref.type} inst_ptr {inst_ptr} -> {data_ptr+deref.offset}")
            # print("data_ptr", -1, data_ptr,data_ptr+deref.offset)
            return data_ptr+deref.offset
        else:
            return 0
    else:
        return _get_array_raw_data_ptr(facts.data) + (f_id * 8) #assuming 8 byte ptrs

# @njit(void(BaseReteNodeType, i8,u8,u1),locals={"f_id" : u8}, cache=True)
@njit(locals={"f_id" : u8, "a_id" : u8}, cache=True)
def validate_head_or_retract(self, arg_ind, idrec, head_ptrs, r):
    '''Update the head_ptr dictionaries by following the deref
     chains of DECLARE/MODIFY changes, and make retractions
    for an explicit RETRACT or a failure in the deref chain.'''

    t_id, f_id, a_id = decode_idrec(idrec)
    is_valid = True
    if(a_id != RETRACT):
        for i in range(r.length):
            head_var = _struct_from_ptr(GenericVarType,self.op.head_var_ptrs[r.start+i])
            deref_infos = head_var.deref_infos
            head_ptr = resolve_head_ptr(self, arg_ind, t_id, f_id, deref_infos)
            if(head_ptr == 0): 
                is_valid=False;
                break; 
            head_ptrs[i] = head_ptr

    else:
        is_valid = False
    return is_valid
            


# @njit(cache=True)
# def update_changes_deref_dependencies(self, arg_insert_sets):
#      ### 'modify_idrecs' is the set of self.mem.change_queue
#     # items relevant to intermediate derefs computed for this literal,
#     # and modification of the head attribute. Shouldn't happen frequently ###
#     for i in range(self.modify_idrecs.head):
#         idrec = self.modify_idrecs[i]
#         if(idrec in self.deref_depends):
#             deref_records = self.deref_depends[idrec]

#             # Invalidate old DerefRecords
#             for r_ptr in deref_records:
#                 r = _struct_from_ptr(DerefRecordType,r_ptr)
#                 invalidate_head_ptr_rec(r)

#                 # Any change in the deref chain counts as a MODIFY
#                 # TODO: This is for sure wrong
#                 arg_insert_sets[r.arg_ind][r.base_idrec] = 1 #MODIFY


# @njit(cache=True)
# def update_modify_changes(self):
#     print("modify_idrecs:",self.lit)
#     for arg_ind, modify_idrecs in enumerate(self.modify_idrecs):
#         arr = modify_idrecs.data[:modify_idrecs.head]
#         print("\t", arg_ind, modify_idrecs.head, np.array([decode_idrec(x)[1] for x in arr]))
#         modify_idrecs.head = 0


### STREAM OF CONCIOUSNESS NOTES --- REMOVE
'''
Inputs are fixed width. They are intended to have holes. Each node
has an input state for its inputs 

'''
####

@njit(cache=True)
def update_input_changes(self):
    '''Given upstream changes fills the changed and unchanged inds for a node.'''
    for i, inp in enumerate(self.inputs):

        # Extract values used below
        w_i = self.inp_widths[i]
        idrecs_to_inds_i = self.idrecs_to_inds[i]
        retracted_inds_i = self.retracted_inds[i]
        head_ptr_buffers_i = self.head_ptr_buffers[i]
        input_state_buffers_i = self.input_state_buffers[i]
        inds_insert_buffers_i = self.inds_insert_buffers[i]
        inds_remove_buffers_i = self.inds_remove_buffers[i]
        head_range_i = self.op.head_ranges[i]
        modify_idrecs_i = self.modify_idrecs[i]

        # Clear input_states of properties only meant to last one cycle
        # c = 0
        for k in range(w_i):
            input_state = input_state_buffers_i[k]
            input_state.recently_removed = False
            input_state.recently_inserted = False
                
        # Collection of various sources of changes as (idrec, ind) pairs.
        change_pairs = np.empty(len(inp.insert_set)+len(inp.remove_set)+
            len(modify_idrecs_i),dtype=idrec_ind_pair_type)
        print("i :", i)
        print("inp", np.array([decode_idrec(x.idrec)[1] for x in  input_state_buffers_i]))
        print("insrt", np.array([decode_idrec(x)[1] for x in  inp.insert_set]))
        print("remove", np.array([decode_idrec(x)[1] for x in  inp.remove_set]))
        print("modify", np.array([decode_idrec(x)[1] for x in  modify_idrecs_i.data[:modify_idrecs_i.head] ]))

        # Insert the insert_set of this input into change_pairs
        #  and add into idrecs_to_inds.
        c = 0
        for idrec in inp.insert_set:
            ind = idrecs_to_inds_i.get(idrec,-1)
            if(ind == -1): 
                if(len(retracted_inds_i) > 0):
                    ind = retracted_inds_i.pop()
                else:
                    ind = w_i
                    w_i += 1
                idrecs_to_inds_i[idrec] = ind

                change_pairs[c].idrec = idrec
                change_pairs[c].ind = ind
                c += 1

        # Insert any modify changes specifically routed to this node.
        mod_cutoff = c
        for k in range(modify_idrecs_i.head):
            t_id, f_id, a_id = decode_idrec(modify_idrecs_i.data[k])
            print(t_id, f_id, a_id)
            idrec = encode_idrec(t_id, f_id, 0)
            ind = idrecs_to_inds_i.get(idrec,-1)
            if(ind != -1):
                change_pairs[c].idrec = idrec
                change_pairs[c].ind = idrecs_to_inds_i[idrec]
                c += 1
        modify_idrecs_i.head = 0

        # Insert the remove_set of this input into change_pairs
        #  and set a -1 placeholder into idrecs_to_inds.
        rem_cutoff = c
        for idrec in inp.remove_set:
            t_id, f_id, a_id = decode_idrec(idrec)
            idrec = encode_idrec(t_id, f_id, 0)
            ind = idrecs_to_inds_i.get(idrec,-1)
            if(ind != -1): 
                retracted_inds_i.add(ind)
                idrecs_to_inds_i[idrec] = -1
                change_pairs[c].idrec = idrec
                change_pairs[c].ind = ind
                c += 1

        assert len(inp.match_idrecs) == w_i

        change_pairs = change_pairs[:c]
        # print("len(change_pairs): ", len(change_pairs))     
        
        # Ensure various buffers are large enough (3 us).
        curr_len, curr_w = head_ptr_buffers_i.shape
        self.inp_widths[i] = w_i
        if(self.inp_widths[i] > curr_len):
            expand = max(self.inp_widths[i]-curr_len, curr_len)
            new_head_ptr_buff = np.empty((curr_len+expand,curr_w),dtype=np.int64)
            new_head_ptr_buff[:curr_len] = head_ptr_buffers_i
            head_ptr_buffers_i = self.head_ptr_buffers[i] = new_head_ptr_buff

            new_input_state_buff = np.zeros((curr_len+expand,),dtype=input_state_type)
            new_input_state_buff[:curr_len] = input_state_buffers_i
            input_state_buffers_i = self.input_state_buffers[i] = new_input_state_buff

            new_inds_isrt_buff = np.empty((curr_len+expand,),dtype=np.int64)
            new_inds_isrt_buff[:curr_len] = inds_insert_buffers_i
            inds_insert_buffers_i = self.inds_insert_buffers[i] = new_inds_isrt_buff

            new_inds_rem_buff = np.empty((curr_len+expand,),dtype=np.int64)
            new_inds_rem_buff[:curr_len] = inds_remove_buffers_i
            inds_remove_buffers_i = self.inds_remove_buffers[i] = new_inds_rem_buff
        
        # For each fact in the change_pairs apply deref chains associated 
        #  with the base var for this input. Mark as a newly inserted/removed
        #  input as appropriate.
        # num_inserts = 0
        # num_removes = 0
        for k, pair in enumerate(change_pairs):
            idrec, ind = pair.idrec, pair.ind

            is_valid = False
            is_modify = (k >= mod_cutoff) & (k < rem_cutoff)
            input_state = input_state_buffers_i[ind]

            # If from upstream insert or a modify then check if the deref 
            #  chain(s) are valid to determine if is insert/remove/unchanged.
            if(k < rem_cutoff):
                head_ptrs = head_ptr_buffers_i[ind]
                was_valid = input_state.head_was_valid
                is_valid = validate_head_or_retract(self, i, idrec, head_ptrs, head_range_i)
                print(">> is_valid:", decode_idrec(idrec)[1], is_valid)
                is_removed = ~is_valid #& ~is_modify
                recently_inserted = (~was_valid & is_valid) | is_modify

            # Otherwise it is an upstream remove.
            else:
                recently_inserted, is_removed = False, True

            # print("change_pairs", decode_idrec(idrec), is_modify, recently_inserted, k, mod_cutoff)

            # Assign to the input_state struct
            input_state.idrec = idrec
            if(not input_state.is_removed and is_removed):
                input_state.recently_removed = True    
            input_state.recently_inserted = recently_inserted
            input_state.is_removed = is_removed
            input_state.head_was_valid = is_valid

            # if(not is_modify and recently_inserted): 
                # num_inserts += 1
            # num_removes += is_removed

        # inserted_inds_i = self.inserted_inds[i] = self.inds_insert_buffers[i][:num_inserts]
        # unchanged_inds_i = self.unchanged_inds[i] = self.inds_insert_buffers[i][num_inserts:w_i]
        # removed_inds_i = self.removed_inds[i] = self.inds_remove_buffers[i][:num_removes]

        # Fill the insert, remove, and unchanged inds arrays (<1 us).
        c, r, u = 0, 0, 0
        for j in range(self.inp_widths[i]):
            input_state = input_state_buffers_i[j]
            if(input_state.recently_inserted):
                inds_insert_buffers_i[c] = j; c += 1;
            elif(input_state.recently_removed):
                inds_remove_buffers_i[r] = j; r += 1;

        for j in range(self.inp_widths[i]):
            input_state = input_state_buffers_i[j]
            if(not input_state.recently_inserted and not input_state.recently_removed):
                inds_insert_buffers_i[c+u] = j; u += 1;

        self.inserted_inds[i] = inds_insert_buffers_i[:c]
        self.unchanged_inds[i] = inds_insert_buffers_i[c:c+u]
        self.removed_inds[i] = inds_remove_buffers_i[:r]

        
        print("removed_inds_i: ", self.removed_inds[i])
        print("inserted_inds_i: ", self.inserted_inds[i])
        print("unchanged_inds_i: ", self.unchanged_inds[i])
    print()
        # print(num_inserts, c)
        # print(num_removes, r)


@njit(cache=True)
def resize_truth_table(self):
    s0,s1 = self.truth_table.shape
    expand0 = max(0,self.inp_widths[0]-s0)
    expand1 = max(0,self.inp_widths[1]-s1)
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

            
from cre.utils import _func_from_address
match_heads_f_type = types.FunctionType(u1(i8[::1],))

u8_i8_dict_type = DictType(u8,i8)



@njit(cache=True)
def _ld_dict(ptr):
    return _dict_from_ptr(u8_i8_dict_type, ptr)     

# TODO: is dead?
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


@njit(cache=True)
def beta_matches_to_str(matches):
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
    s = "["
    for match0, others_ptr in matches.items():
        t_id0, f_id0, a0 = decode_idrec(match0)
        s +=f"({t_id0},{f_id0}),"

    return s[:-1] + "]"

@njit(cache=True, inline='always')
def _upstream_true(u_tt, aligned, pind_i, pind_j):
    if(aligned):
        return u_tt[pind_i, pind_j]
    else:
        return u_tt[pind_j, pind_i]

@njit(cache=True, inline='always')
def _check_beta(negated, j, j_strt, j_len, inp_buffers_j, 
         head_ptrs_j, ind_j, match_head_ptrs_func, match_inp_ptrs, match_inds):    
    inp_state_j = inp_buffers_j[ind_j]
    if(not inp_state_j.head_was_valid): return u1(0)
    match_inds[j] = ind_j
    match_inp_ptrs[j_strt:j_strt+j_len] = head_ptrs_j[ind_j]
    is_match = match_head_ptrs_func(match_inp_ptrs) ^ negated
    return is_match

@njit(cache=True, inline='always')
def _update_truth_table(tt, is_match, match_inds, inp_state_i, inp_state_j):
    # Set the truth table and modify 'true_count' as appropriate
    ind0, ind1 = match_inds[0], match_inds[1]
    was_match = tt[ind0, ind1]
    tt[ind0, ind1] = u1(is_match)
    count_diff = i8(is_match) - i8(was_match)
    inp_state_i.true_count += count_diff
    inp_state_j.true_count += count_diff



@njit(cache=True)
def update_node(self):
    # If the node is an identity node then skip since the input is hardwired to the output
    if(self.op is None): return

    # Note: Keep these print statements for debugging
    print("-----------------------")
    print("Update", self.lit)
    
    # Go through each idrec in the insert_set and identify changes in
    #  the set of candidate facts for this node's variables. 
    update_input_changes(self)
    # update_modify_changes(self)
    head_ranges = self.op.head_ranges
    n_vars = len(head_ranges)    
    negated = self.lit.negated

    # If is a beta node make sure the truth_table is big enough.
    if(n_vars > 1):
        resize_truth_table(self)

    # Load the 'match' function for this node's op.
    match_head_ptrs_func = _func_from_address(match_heads_f_type, self.op.match_head_ptrs_addr)

    # Buffers that will be loaded with ptrs, indrecs, or inds of candidates.
    match_inp_ptrs = np.zeros(len(self.op.head_var_ptrs),dtype=np.int64)
    match_inds = np.zeros(len(self.var_inds),dtype=np.int64)
    tt = self.truth_table

    # Loop through the (at most 2) variables for this node.
    for i in range(n_vars):
        # Extract various things used below
        idrecs_to_inds_i = self.idrecs_to_inds[i]
        head_ptrs_i = self.head_ptr_buffers[i]
        i_strt, i_len = head_ranges[i][0], head_ranges[i][1]
        inserted_inds_i = self.inserted_inds[i]

        # BETA CASE (i.e. n_vars = 2)
        if(n_vars > 1):
            # For every change in a candidate to the first variable we'll 
            #  update against every other possible candidate for the second.
            #  i.e. for i=0 we fill  whole rows in our truth table.
            #  [?--?--?]
            #  [X--X--X]
            #  [?--?--?]
            
            # For every change in a candidate to the second variable we'll 
            #  update against only the unchanged candidates for the first.
            #  i.e. for i=1 we fill columns in our truth table, but skip
            #   cells that were already updated.
            #  [?--X--?]
            #  [?--?--?]
            #  [?--X--?]

            # When i=1, j=0 and vise versa
            j = 1 if i == 0 else 0
            
            # Extract various things used below
            j_strt, j_len = head_ranges[j][0], head_ranges[j][1]        
            idrecs_to_inds_j = self.idrecs_to_inds[j]
            head_ptrs_j = self.head_ptr_buffers[j]
            inp_buffers_i = self.input_state_buffers[i]
            inp_buffers_j = self.input_state_buffers[j]
            pinds_i = self.inputs[i].match_inds
            pinds_j = self.inputs[j].match_inds
            aligned = self.upstream_aligned
            same_parent = self.upstream_same_parents
            
            # Go through all of the changed candidates
            for ind_i in inserted_inds_i:
                # Extract various things associated with 'ind_i'
                inp_state_i  = inp_buffers_i[ind_i]
                pind_i = pinds_i[ind_i]

                # Clear a_id from idrec_i
                idrec_i = inp_state_i.idrec
                t_id_i, f_id_i, a_id_i = decode_idrec(idrec_i)
                idrec_i = encode_idrec(t_id_i, f_id_i, 0)
                
                # Fill in the 'i' part of ptrs, indrecs, inds 
                match_inp_ptrs[i_strt:i_strt+i_len] = head_ptrs_i[ind_i]
                match_inds[i] = ind_i

                # NOTE: Sections below have lots of repeated code. Couldn't find a way to inline
                #  without incurring a ~4x slowdown.
                
                # If there is a beta node upstream of to this one that shares the same
                #  variables then we'll need to make sure we also check its truth table.
                if(self.upstream_same_parents):
                    upstream_node = _struct_from_ptr(BaseReteNodeType, self.upstream_node_ptr)
                    u_tt = upstream_node.truth_table

                    if(j > i):
                        # Update the whole row/column
                        for ind_j in range(self.inp_widths[j]):
                            input_state_j = inp_buffers_j[ind_j]
                            if(not _upstream_true(u_tt, aligned, pind_i, pinds_j[ind_j])):
                                match_inds[j] = ind_j
                                _update_truth_table(tt, u1(0), match_inds, inp_state_i, input_state_j)    
                                continue

                            if(input_state_j.head_was_valid):
                                match_inds[j] = ind_j
                                match_inp_ptrs[j_strt:j_strt+j_len] = head_ptrs_j[ind_j]
                                is_match = match_head_ptrs_func(match_inp_ptrs) ^ negated
                            else:
                                is_match = u1(0)
                            _update_truth_table(tt, is_match, match_inds, inp_state_i, input_state_j)
                    else:
                        # Check just the unchanged parts, so to avoid repeat checks 
                        for ind_j in self.unchanged_inds[j]:
                            input_state_j = inp_buffers_j[ind_j]
                            if(not _upstream_true(u_tt, aligned, pind_i, pinds_j[ind_j])):
                                match_inds[j] = ind_j
                                _update_truth_table(tt, u1(0), match_inds, inp_state_i, input_state_j)    
                                continue

                            if(input_state_j.head_was_valid):
                                match_inds[j] = ind_j
                                match_inp_ptrs[j_strt:j_strt+j_len] = head_ptrs_j[ind_j]
                                is_match = match_head_ptrs_func(match_inp_ptrs) ^ negated
                            else:
                                is_match = u1(0)
                            _update_truth_table(tt, is_match, match_inds, inp_state_i, input_state_j)

                # If no 'upstream_same_parents' then just update the truth table 
                #  with the match values for all relevant pairs.
                else:
                    if(j > i):
                        # Update the whole row/column
                        for ind_j in range(self.inp_widths[j]):
                            input_state_j = inp_buffers_j[ind_j]
                            if(input_state_j.head_was_valid):
                                match_inds[j] = ind_j
                                match_inp_ptrs[j_strt:j_strt+j_len] = head_ptrs_j[ind_j]
                                is_match = match_head_ptrs_func(match_inp_ptrs) ^ negated
                            else:
                                is_match = u1(0)
                            _update_truth_table(tt, is_match, match_inds, inp_state_i, input_state_j)
                    else:
                        # Check just the unchanged parts, so to avoid repeat checks 
                        for ind_j in self.unchanged_inds[j]:
                            input_state_j = inp_buffers_j[ind_j]
                            if(input_state_j.head_was_valid):
                                match_inds[j] = ind_j
                                match_inp_ptrs[j_strt:j_strt+j_len] = head_ptrs_j[ind_j]
                                is_match = match_head_ptrs_func(match_inp_ptrs) ^ negated
                            else:
                                is_match = u1(0)
                            _update_truth_table(tt, is_match, match_inds, inp_state_i, input_state_j)

        # ALPHA CASE (i.e. n_var = 1)
        else:
            input_state_buffers_i = self.input_state_buffers[i]
            
            # Go through all of the changed candidates 
            for ind_i in inserted_inds_i:
                inp_state_i = input_state_buffers_i[ind_i]

                # Check if the op matches this candidate
                match_inp_ptrs[i_strt:i_strt+i_len] = head_ptrs_i[ind_i]
                is_match = match_head_ptrs_func(match_inp_ptrs) ^ negated
                inp_state_i.true_count = i8(is_match)

    # Update each of (the at most 2) outputs (one for each Var).
    for i, out_i in enumerate(self.outputs):
        insert_ind = 0
        remove_ind = 0
        match_ind = 0
        input_state_buffers_i = self.input_state_buffers[i]
        
        # Update each slot k for match candidates for the ith Var.
        for k in range(self.inp_widths[i]):
            # Extract t_id, f_id, a_id for the input at k.
            input_state_k = input_state_buffers_i[k]
            idrec_k = input_state_k.idrec
            t_id_k, f_id_k, a_id_k = decode_idrec(idrec_k)

            # Determine if we've ever found a match for this candidate
            true_is_nonzero = (input_state_k.true_count != 0)
            input_state_k.true_ever_nonzero |= true_is_nonzero

            # If we have then we'll slot it into the output.
            print("??", f_id_k, true_is_nonzero, not input_state_k.is_removed)
            if(input_state_k.true_ever_nonzero):
                if(true_is_nonzero and not input_state_k.is_removed):
                    idrec = encode_idrec(t_id_k, f_id_k, DECLARE)
                else:
                    idrec = u8(0) 
                node_memory_insert_match_buffers(out_i, match_ind, idrec, k)
                match_ind += 1

            # print("<<", k, t_id_k, f_id_k, a_id_k, input_state_k.is_removed, input_state_k.true_was_nonzero, true_is_nonzero)

            # When 
            t_id_k, f_id_k, _ = decode_idrec(idrec_k)
            if( input_state_k.recently_removed or
               (input_state_k.true_was_nonzero and ~true_is_nonzero)):
                print("RETRACT", f_id_k, input_state_k.is_removed, input_state_k.true_was_nonzero, ~true_is_nonzero)
                out_i.remove_buffer = setitem_buffer(out_i.remove_buffer,
                    remove_ind, encode_idrec(t_id_k, f_id_k, RETRACT))
                remove_ind += 1

                # ind = np.nonzero(out_i.match_idrecs == idrec_k)[0][0]
                # node_memory_insert_match_buffers(out_i, ind, u8(0), -1)

            # When a candidate flips to matching add the insert_set.
            elif(input_state_k.true_was_nonzero != true_is_nonzero):
                out_i.insert_buffer = setitem_buffer(out_i.insert_buffer,
                    insert_ind, encode_idrec(t_id_k, f_id_k, DECLARE))
                insert_ind += 1

                # idrec = encode_idrec(t_id_k, f_id_k, DECLARE)
                # node_memory_insert_match_buffers(out_i, match_ind, idrec, k)
                # match_ind += 1

            input_state_k.true_was_nonzero = true_is_nonzero
        
        # To avoid reallocations the arrays in each output are slices of larger buffers.
        out_i.match_idrecs = out_i.match_idrecs_buffer[:match_ind]
        out_i.match_inds = out_i.match_inds_buffer[:match_ind]
        out_i.insert_set = out_i.insert_buffer[:insert_ind]
        out_i.remove_set = out_i.remove_buffer[:remove_ind]

        # Note: Keep these print statements for debugging
        print("i :", i)
        print("match_idrecs.f_id", np.array([decode_idrec(x)[1] for x in out_i.match_idrecs]))
        print("match_inds", out_i.match_inds)
        print("insert_set.f_id", np.array([decode_idrec(x)[1] for x in out_i.insert_set]))
        print("remove_set.f_id", np.array([decode_idrec(x)[1] for x in out_i.remove_set]))
        # print(self.truth_table)
            



node_arg_pair_type = Tuple((BaseReteNodeType,i8))
node_arg_list_type = ListType(node_arg_pair_type)
node_mem_list_type = ListType(NodeMemoryType)
rete_graph_field_dict = {
    # The change_head of the working memory at the last graph update.
    "change_head" : i8,

    # The working memory memset.
    "memset" : MemSetType,

    # All graph nodes organized by [[...alphas],[...betas],[...etc]]
    "nodes_by_nargs" : ListType(ListType(BaseReteNodeType)),

    # Maps a var_ind to its associated root node (i.e. the node that 
    #  holds all match candidates for a fact_type before filtering).
    "var_root_nodes" : DictType(i8,BaseReteNodeType),

    # TODO: Replace with list/array for speed?
    # The map a var_ind to the most downstream node that constrains that var.
    "var_end_nodes" : DictType(i8,BaseReteNodeType),

    # A matrix of size (n_var, n_var) with weak pointers to the most 
    #  downstream beta nodes connecting each pair of vars. 
    "var_end_join_ptrs" : i8[::,::1],

    # Maps (t_id, 0, a_id) idrec patterns to (node,arg_ind) that should be 
    #  rechecked based on that pattern.
    "global_modify_map" : DictType(u8, node_arg_list_type),

    # Maps t_ids to the root node memories associated with facts of that t_id.
    "global_t_id_root_memory_map" : DictType(u2, NodeMemoryType),

    # A strong pointer the prototype instance for match iterators on this graph.
    "match_iter_prototype_ptr" : ptr_t, #NOTE: Should really use deferred type
}


ReteGraph, ReteGraphType = define_structref("ReteGraph", rete_graph_field_dict, define_constructor=False)

@njit(cache=True)
def rete_graph_ctor(ms, conds, nodes_by_nargs, var_root_nodes, var_end_nodes,
                var_end_join_ptrs, global_modify_map, global_t_id_root_memory_map):
    st = new(ReteGraphType)
    st.change_head = 0
    st.memset = ms
    st.nodes_by_nargs = nodes_by_nargs
    st.var_root_nodes = var_root_nodes
    st.var_end_nodes = var_end_nodes
    st.var_end_join_ptrs = var_end_join_ptrs
    st.global_modify_map = global_modify_map
    st.global_t_id_root_memory_map = global_t_id_root_memory_map
    st.match_iter_prototype_ptr = 0
    
    return st


@njit(cache=False)
def conds_get_rete_graph(self):
    if(not self.matcher_inst_ptr == 0):
        return _struct_from_ptr(ReteGraphType,self.matcher_inst_ptr)
    return None


@njit(cache=True)
def _get_degree_order(c, index_map):
    ''' Order vars by decreasing beta degree --- the number of other 
        vars that they share beta literals with. Implements heuristic
        that the most constrained nodes are matched first. 
     '''
    has_pairs = np.zeros((len(c.vars),len(c.vars)),dtype=np.uint8)
    for distr_conjunct in c.distr_dnf:
        for j, var_conjuct in enumerate(distr_conjunct):
            for lit in var_conjuct:
                var_inds = np.empty((len(lit.var_base_ptrs),),dtype=np.int64)
                for i, base_var_ptr in enumerate(lit.var_base_ptrs):
                    var_inds[i] = index_map[i8(base_var_ptr)]
                if(len(var_inds) > 1):
                    has_pairs[var_inds[0],var_inds[1]] = 1
                    has_pairs[var_inds[1],var_inds[0]] = 1
    degree = has_pairs.sum(axis=1)
    degree_order = np.argsort(-degree)
    return degree_order


@njit(cache=True)
def _ensure_long_enough(nodes_by_nargs, nargs):
    while(len(nodes_by_nargs) <= nargs-1):
        nodes_by_nargs.append(List.empty_list(BaseReteNodeType))

@njit(cache=True)
def _mod_map_insert(idrec, mod_map, node, arg_ind):
    if(idrec not in mod_map):
        mod_map[idrec] = List.empty_list(node_arg_pair_type)
    mod_map[idrec].append((node, arg_ind))

ReteNode_List_type = ListType(BaseReteNodeType)

@njit(cache=True,locals={})
def _make_rete_nodes(mem, c, index_map):
    nodes_by_nargs = List.empty_list(ReteNode_List_type)
    nodes_by_nargs.append(List.empty_list(BaseReteNodeType))
    global_modify_map = Dict.empty(u8, node_arg_list_type)

    # Make an identity node (i.e. lit,op=None) so there are always alphas
    for j in range(len(c.vars)):
        t_ids = np.empty((1,),dtype=np.uint16)
        var_inds = np.empty((1,),dtype=np.int64)
        base_var = c.vars[j]
        t_ids[0] = base_var.base_t_id#_get_var_t_id(mem, base_var)
        var_inds[0] = index_map[i8(base_var.base_ptr)]
        nodes_by_nargs[0].append(node_ctor(mem, t_ids, var_inds,lit=None))

    degree_order = _get_degree_order(c, index_map)

    # print("degree_order", degree_order)

    for distr_conjunct in c.distr_dnf:
        for j, var_ind in enumerate(degree_order):
            var_conjuct = distr_conjunct[var_ind]

            for lit in var_conjuct:
                nargs = len(lit.var_base_ptrs)
                _ensure_long_enough(nodes_by_nargs, nargs)
                # print("A")
                t_ids = np.empty((nargs,),dtype=np.uint16)
                var_inds = np.empty((nargs,),dtype=np.int64)
                for i, base_var_ptr in enumerate(lit.var_base_ptrs):
                    var_inds[i] = index_map[i8(base_var_ptr)]
                    base_var = _struct_from_ptr(GenericVarType, base_var_ptr)
                    t_id = base_var.base_t_id#_get_var_t_id(mem, base_var)
                    t_ids[i] = t_id
                # print("B")
                # print("t_ids", t_ids)
                node = node_ctor(mem, t_ids, var_inds, lit)
                nodes_by_nargs[nargs-1].append(node)
                # print("<< aft", lit.op.head_var_ptrs)
                for i, head_var_ptr in enumerate(lit.op.head_var_ptrs):
                    head_var = _struct_from_ptr(GenericVarType, head_var_ptr)
                    arg_ind = np.min(np.nonzero(lit.var_base_ptrs==i8(head_var.base_ptr))[0])
                    t_id = t_ids[arg_ind]
                    # print("START")
                    for d_offset in head_var.deref_infos:
                        idrec1 = encode_idrec(u2(t_id),0,u1(d_offset.a_id))
                        _mod_map_insert(idrec1, global_modify_map, node, arg_ind)

                        t_id = d_offset.t_id
                        idrec2 = encode_idrec(u2(t_id),0,0)
                        _mod_map_insert(idrec2, global_modify_map, node, arg_ind)
                                
    return nodes_by_nargs, global_modify_map

optional_node_mem_type = types.optional(NodeMemoryType)

@njit(cache=True)
def arr_is_unique(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if(i != j and arr[i]==arr[j]):
                return False
    return True


@njit(cache=True)
def build_rete_graph(ms, c):
    # Build a map from base var ptrs to indicies
    index_map = Dict.empty(i8, i8)
    for i, v in enumerate(c.vars):
        index_map[i8(v.base_ptr)] = i

    # Reorganize the dnf into a distributed dnf with one dnf per var
    if(not c.has_distr_dnf):
        build_distributed_dnf(c,index_map)
    
    # Make all of the RETE nodes
    nodes_by_nargs, global_modify_map = \
         _make_rete_nodes(ms, c, index_map)

    global_t_id_root_memory_map = Dict.empty(u2, NodeMemoryType)
    var_end_join_ptrs = np.zeros((len(c.vars),len(c.vars)),dtype=np.int64)
    var_end_nodes = Dict.empty(i8,BaseReteNodeType)
    var_root_nodes = Dict.empty(i8,BaseReteNodeType)

    # Link nodes together. 'nodes_by_nargs' should already be ordered
    # so that alphas are before 2-way, 3-way, etc. betas. 
    for i, nodes in enumerate(nodes_by_nargs):
        for node in nodes:
            # print("node", node.lit)
            inputs = List.empty_list(NodeMemoryType)
            for j, var_ind in enumerate(node.var_inds):
                # print("ind", ind)
                if(var_ind in var_end_nodes):
                    # Set the .inputs of this node to be the NodeMemorys from the
                    #   .outputs of the last node in the graph to check against 'var_ind' 
                    e_node = var_end_nodes[var_ind]
                    om = e_node.outputs[np.min(np.nonzero(e_node.var_inds==var_ind)[0])]
                    inputs.append(om)
                    node.upstream_node_ptr = _raw_ptr_from_struct(e_node)
                    # print("wire", var_ind, e_node.lit, '[', np.min(np.nonzero(e_node.var_inds==var_ind)[0]),']', "->",  node.lit, "[", j, "]")
                else:

                    t_id = node.t_ids[j]
                    if(t_id not in global_t_id_root_memory_map):
                        root = new_root_node_mem()
                        global_t_id_root_memory_map[t_id] = root
                    root = global_t_id_root_memory_map[t_id]

                    inputs.append(root)
                    var_root_nodes[var_ind] = node

            node.inputs = inputs

            if(len(inputs) == 2):
                p1, p2 = inputs[0].parent_node_ptr, inputs[1].parent_node_ptr
                node.upstream_same_parents = (p1 != 0 and p1 == p2)
                if(node.upstream_same_parents):
                    parent = _struct_from_ptr(BaseReteNodeType, p1)
                    node.upstream_aligned = np.all(node.var_inds==parent.var_inds)   

                # Fill in end joins
                vi_a, vi_b = node.var_inds[0],node.var_inds[1]
                var_end_join_ptrs[vi_a, vi_b] = _raw_ptr_from_struct(node)
                var_end_join_ptrs[vi_b, vi_a] = _raw_ptr_from_struct(node)

                # print("Assign end join:", vi_a if vi_a < vi_b else vi_b,
                #      vi_b if vi_a < vi_b else vi_a, node.lit)
                

            # Short circut the input to the output for identity nodes
            if(node.lit is None):
                node.outputs = node.inputs

            # Make this node the new end node for the vars it takes as inputs
            for var_ind in node.var_inds:
                var_end_nodes[var_ind] = node
            
    return rete_graph_ctor(ms, c, nodes_by_nargs, var_root_nodes, var_end_nodes,
              var_end_join_ptrs, global_modify_map, global_t_id_root_memory_map)

@njit(cache=True)
def setitem_buffer(buffer, k, idrec):
    buff_len = len(buffer)
    if(k >= buff_len):
        expand = max(k-buff_len, buff_len)
        new_buffer = np.empty(expand+buff_len,dtype=np.uint64)
        new_buffer[:buff_len] = buffer
        buffer = new_buffer
    buffer[k] = idrec
    return buffer
    

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


from cre.processing.incr_processor import accumulate_change_events

@njit(cache=True,locals={"t_id" : u2, "f_id":u8, "a_id":u1})
def parse_change_queue(r_graph):

    # Extract values used below
    global_modify_map = r_graph.global_modify_map
    global_t_id_root_memory_map = r_graph.global_t_id_root_memory_map
    change_queue = r_graph.memset.change_queue

    
    for t_id, root_mem in global_t_id_root_memory_map.items():
        root_mem.insert_set = root_mem.insert_buffer[:0]
        root_mem.remove_set = root_mem.remove_buffer[:0]


    change_events = accumulate_change_events(change_queue, r_graph.change_head, -1)

    # print("**:", change_queue.data[:change_queue.head])
    # print("**:", global_t_id_root_memory_map)
    # for i in range(r_graph.change_head, change_queue.head):
    zeros = List.empty_list(u1)
    zeros.append(u1(0))

    for change_event in change_events:
        print("change_event", change_event)
        # idrec = u8(change_queue[i])
        t_id, f_id, _ = decode_idrec(change_event.idrec)

        
        # print("idrec", t_id, f_id, a_id)

        # Add this idrec to insert_set of root nodes
        if(t_id not in global_t_id_root_memory_map): continue
        root_mem = global_t_id_root_memory_map[t_id]

        if(change_event.was_modified):
            for a_id in change_event.a_ids:
                # Add this idrec to relevant deref idrecs
                idrec_pattern = encode_idrec(t_id, 0, a_id)
                node_arg_pairs = global_modify_map.get(idrec_pattern,None)
                if(node_arg_pairs is not None):
                    for (node,arg_ind) in node_arg_pairs:
                        print("added: ", node.lit, 't_id=', t_id, 'f_id=', f_id, 'a_id=', a_id)
                        node.modify_idrecs[arg_ind].add(encode_idrec(t_id, f_id, a_id))

        elif(not change_event.was_retracted):
            k = len(root_mem.insert_set)
            root_mem.insert_buffer = setitem_buffer(
                    root_mem.insert_buffer, k, encode_idrec(t_id,f_id,0))
            root_mem.insert_set = root_mem.insert_buffer[:k+1]

            idrec = encode_idrec(t_id,f_id,0)# if(a_id != RETRACT) else u8(0)
            node_memory_insert_match_buffers(root_mem, i8(f_id), idrec, i8(f_id))
            if(i8(f_id) >= len(root_mem.match_idrecs)):
                root_mem.match_idrecs = root_mem.match_idrecs_buffer[:i8(f_id)+1]
                root_mem.match_inds = root_mem.match_inds_buffer[:i8(f_id)+1]
        else:
            
            k = len(root_mem.remove_set)
            root_mem.remove_buffer = setitem_buffer(
                    root_mem.remove_buffer, k, encode_idrec(t_id,f_id,RETRACT))
            root_mem.remove_set = root_mem.remove_buffer[:k+1]

            print("RETRACT", t_id,f_id, root_mem.remove_set)

            node_memory_insert_match_buffers(root_mem, i8(f_id), u8(0), i8(0))
                


    r_graph.change_head = change_queue.head



# ----------------------------------------------------------------------
# : MatchIterator

match_iterator_node_field_dict = {
    # "graph" : ReteGraphType,
    "node" : BaseReteNodeType,
    "associated_arg_ind" : i8,
    "var_ind" : i8,
    "m_node_ind" : i8,
    # "is_exhausted": boolean,
    "curr_ind": i8,
    "idrecs" : u8[::1],
    "other_idrecs" : u8[::1],
    "dep_m_node_inds" : i8[::1],
    "dep_arg_inds" : i8[::1],
    "dep_node_ptrs" : i8[::1],
}



MatchIterNode, MatchIterNodeType = define_structref("MatchIterNode", match_iterator_node_field_dict, define_constructor=False)

match_iterator_field_dict = {
    "graph" : ReteGraphType,
    "iter_nodes" : ListType(MatchIterNodeType),
    "is_empty" : types.boolean,
    "iter_started" : types.boolean,
    "output_types" : types.Any
    # "curr_match" : u8[::1],
}
match_iterator_fields = [(k,v) for k,v, in match_iterator_field_dict.items()]


# NOTE: Probably uncessary now that Fact types recover their original type on unboxing.
def gen_match_iter_source(output_types):
    ''' Generates source code for '''
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
    m_iter_type_cache = {}
    def __new__(cls, ms, conds):
        # Make a generic MatchIterator (reuses graph if conds already has one)
        generic_m_iter = get_match_iter(ms, conds)
        #Cache 'output_types' and 'specialized_m_iter_type'
        var_base_types = conds.var_base_types

        if(var_base_types not in cls.m_iter_type_cache):
            hash_code = unique_hash([var_base_types])
            if(not source_in_cache('MatchIterator', hash_code)):
                output_types = types.TypeRef(types.Tuple([types.TypeRef(x) for x in conds.var_base_types]))
                source = gen_match_iter_source(output_types)
                source_to_cache('MatchIterator', hash_code, source)
            l = import_from_cached('MatchIterator', hash_code, ['specialize_m_iter', 'output_types'])
            output_types, specialize_m_iter = cls.m_iter_type_cache[var_base_types] = l['output_types'], l['specialize_m_iter']
        else:
            output_types, specialize_m_iter  = cls.m_iter_type_cache[var_base_types]

        # Specialize the match iter so that it outputs conds.var_base_types 
        self = specialize_m_iter(generic_m_iter)
        self.output_types = output_types#tuple([types.TypeRef(x) for x in conds.var_base_types])
        return self
        
    def __next__(self):
        # with PrintElapse("match_iter_next"):
        return match_iter_next(self)

    def __iter__(self):
        return self

    def __del__(self):
        pass


@structref.register
class MatchIteratorType(CastFriendlyMixin, types.StructRef):
    def __str__(self):
        return "cre.MatchIterator"
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


define_boxing(MatchIteratorType, MatchIterator)
GenericMatchIteratorType = MatchIteratorType(match_iterator_fields)

# Allow any specialization of MatchIteratorType to be upcast to GenericMatchIteratorType
@lower_cast(MatchIteratorType, GenericMatchIteratorType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty)


@njit(GenericMatchIteratorType(GenericMatchIteratorType, ReteGraphType),cache=True)
def copy_match_iter(m_iter, graph):
    '''Makes a copy a prototype match iterator'''
    m_iter_nodes = List.empty_list(MatchIterNodeType)
    for i,m_node in enumerate(m_iter.iter_nodes):
        new_m_node = new(MatchIterNodeType)
        # new_m_node.graph = m_node.graph
        new_m_node.node = m_node.node
        new_m_node.var_ind = m_node.var_ind
        new_m_node.m_node_ind = m_node.m_node_ind
        new_m_node.associated_arg_ind = m_node.associated_arg_ind
        new_m_node.dep_m_node_inds = m_node.dep_m_node_inds
        new_m_node.dep_arg_inds = m_node.dep_arg_inds
        new_m_node.dep_node_ptrs = m_node.dep_node_ptrs
        new_m_node.curr_ind = m_node.curr_ind
        if(m_node.curr_ind != -1):
            new_m_node.idrecs = m_node.idrecs
        m_iter_nodes.append(new_m_node)

    new_m_iter = new(GenericMatchIteratorType)
    new_m_iter.graph = graph 
    new_m_iter.iter_nodes = m_iter_nodes 
    new_m_iter.is_empty = m_iter.is_empty
    new_m_iter.iter_started = m_iter.iter_started

    return new_m_iter

@njit(GenericMatchIteratorType(ReteGraphType), cache=True)
def new_match_iter(graph):
    '''Produces a new MatchIterator for a graph.'''

    # Make an iterator prototype for this graph if one doesn't exist
    #  a copy of the prototype will be built when __iter__ is called.
    if(graph.match_iter_prototype_ptr == 0):
        m_iter_nodes = List.empty_list(MatchIterNodeType)
        handled_vars = Dict.empty(i8,MatchIterNodeType)

        #Make a list of var_inds
        var_inds = List.empty_list(i8)
        for i in range(len(graph.var_end_nodes)-1,-1,-1):
            var_inds.append(i) 

        # Loop downstream to upstream through the end nodes. Build a MatchIterNode
        #  for each end node to help us iterate over valid matches in the graph.
        # for var_ind in range(len(graph.var_end_nodes)-1,-1,-1):
        m_node_ind = 0;
        while(len(var_inds) > 0):
            print_ind = np.empty(len(handled_vars))
            for q, k in enumerate(handled_vars):
                print_ind[q] = k

            var_ind = var_inds.pop()
            node = graph.var_end_nodes[var_ind]

            # Instantiate a prototype m_node for the end node for this Var.
            m_node = new(MatchIterNodeType)
            m_node.node = node
            # The index the m_node will have after reversal
            m_node.m_node_ind = m_node_ind
            m_node.var_ind = var_ind
            m_node.associated_arg_ind = np.argmax(node.var_inds==var_ind)#node.outputs[np.argmax(node.var_inds==i)]
            m_node.curr_ind = -1;
            # m_node.dep_m_node_ind = -1
            # m_node.dep_arg_ind = -1
            # other_var_ind = node.var_inds[1 if m_node.associated_arg_ind==0 else 0]

            # print("END NODE", var_ind, m_node.m_node_ind, node.lit, m_node.associated_arg_ind)
            
            dep_m_node_inds = np.empty((len(graph.var_end_nodes)),dtype=np.int64)
            dep_arg_inds = np.empty((len(graph.var_end_nodes)),dtype=np.int64)
            dep_node_ptrs = np.empty((len(graph.var_end_nodes)),dtype=np.int64)
            c = 0
            for j, ptr in enumerate(graph.var_end_join_ptrs[m_node.var_ind]):
                if(ptr != 0 and j in handled_vars):
                    dep_m_node = handled_vars[j]
                    dep_m_node_inds[c] = dep_m_node.m_node_ind
                    dep_node_ptrs[c] = ptr

                    dep_node = _struct_from_ptr(BaseReteNodeType, ptr)
                    dep_arg_inds[c] = np.argmax(dep_node.var_inds==j)
                    # print(j, "Make", m_node.node.lit, m_node.associated_arg_ind, "dep on", dep_node.lit, dep_arg_inds[c])
                    c +=1
            m_node.dep_m_node_inds = dep_m_node_inds[:c]
            m_node.dep_arg_inds = dep_arg_inds[:c]
            m_node.dep_node_ptrs = dep_node_ptrs[:c]

            m_node_ind += 1

            # Mark 'var_ind' as being handled by m_node. This allows 
            #   downstream m_nodes to depend on it.
            if(var_ind not in handled_vars):
                handled_vars[var_ind] = m_node

            m_iter_nodes.append(m_node)

        # Instantiate the prototype and link it to the graph
        m_iter = new(GenericMatchIteratorType)
        m_iter.iter_nodes = m_iter_nodes 
        m_iter.is_empty = False
        m_iter.iter_started = False
        graph.match_iter_prototype_ptr = _ptr_from_struct_incref(m_iter)
    # print("END NEW MATCH ITER")
    # Return a copy of the prototype 
    prototype = _struct_from_ptr(GenericMatchIteratorType, graph.match_iter_prototype_ptr)
    m_iter = copy_match_iter(prototype,graph)
    return m_iter


@njit(unicode_type(GenericMatchIteratorType),cache=True)
def repr_match_iter_dependencies(m_iter):
    rep = ""
    for i, m_node in enumerate(m_iter.iter_nodes):
        s = f'({str(m_node.var_ind)}'
        for j, dep_m_node_ind in enumerate(m_node.dep_m_node_inds):
            dep_m_node = m_iter.iter_nodes[dep_m_node_ind]
            s += f",dep={str(dep_m_node.var_ind)}"
        s += f")"

        rep += s
        if(i < len(m_iter.iter_nodes)-1): rep += " "

    return rep

@njit(types.void(GenericMatchIteratorType, MatchIterNodeType),cache=True)
def update_from_upstream_match(m_iter, m_node):
    ''' Updates the list of `other_idrecs` for a beta m_node. If an 
          m_node's 'curr_ind' would have it yield a next match `A` for its 
          associated Var then the 'other_idrecs' are the idrecs
          for the matching facts of the other (non-associated) Var.
    '''
    multiple_deps = len(m_node.dep_m_node_inds) > 1
    if(multiple_deps):
        idrecs_set = Dict.empty(u8,u1)
    
    # print("-- UPDATE FROM UPSTREAM:", _struct_from_ptr(GenericVarType, m_node.node.lit.var_base_ptrs[m_node.associated_arg_ind]).alias, "--")

    for i, dep_m_node_ind in enumerate(m_node.dep_m_node_inds):
        # Each dep_node is the terminal beta node (i.e. a graph node not an iter node) 
        #  between the vars iterated by m_node and dep_m_node, and might not be the same
        #  as dep_m_node.node.
        dep_node = _struct_from_ptr(BaseReteNodeType, m_node.dep_node_ptrs[i])
        dep_arg_ind = m_node.dep_arg_inds[i]
        dep_m_node = m_iter.iter_nodes[dep_m_node_ind]
        assoc_arg_ind = 1 if dep_arg_ind == 0 else 0

        # print("\tdep on", dep_node.lit, dep_arg_ind)
        
        # Determine the index of the fixed downstream match within this node
        if(_raw_ptr_from_struct(m_node.node) == _raw_ptr_from_struct(dep_node)):
            # If they happen to represent the same graph node then get from match_inds
            other_output = m_node.node.outputs[dep_m_node.associated_arg_ind]
            fixed_intern_ind = other_output.match_inds[dep_m_node.curr_ind]

        # Otherwise we need to use the 'idrecs_to_inds' map
        else:
            # Extract the idrec for the current fixed dependency value in dep_m_node
            dep_idrec = dep_m_node.idrecs[dep_m_node.curr_ind]

            # Use idrecs_to_inds to find the index of the fixed value in dep_node. 
            fixed_intern_ind = dep_node.idrecs_to_inds[dep_arg_ind][dep_idrec]

        fixed_var = _struct_from_ptr(GenericVarType, dep_node.lit.var_base_ptrs[dep_arg_ind])
        fixed_idrec = dep_node.input_state_buffers[dep_arg_ind][fixed_intern_ind].idrec

        # print("\tFixed:",  fixed_var.alias, "==", m_iter.graph.memset.get_fact(fixed_idrec))
        
        # Consult the dep_node's truth_table to fill 'idrecs'.
        inp_states = dep_node.input_state_buffers[assoc_arg_ind]        
        truth_table = dep_node.truth_table
        k = 0
        idrecs = np.empty((truth_table.shape[assoc_arg_ind],), dtype=np.uint64)
        if(assoc_arg_ind == 1):
            for j, t in enumerate(truth_table[fixed_intern_ind, :]):
                if(t): idrecs[k] = inp_states[j].idrec; k += 1
        else:        
            for j, t in enumerate(truth_table[:, fixed_intern_ind]):
                if(t): idrecs[k] = inp_states[j].idrec; k += 1

        # Assign idrecs as just the parts of the buffer we filled. 
        idrecs = idrecs[:k]
        
        # If the var covered by m_node has multiple dependencies then 
        #  find the set of idrecs common between them. 
        if(multiple_deps):
            if(i == 0):
                for idrec in idrecs:
                    idrecs_set[idrec] = u1(1)
            else:
                # Intersect with previous idrecs_set
                new_idrecs_set = Dict.empty(u8,u1)
                for idrec in idrecs:
                    if(idrec in idrecs_set):
                        new_idrecs_set[idrec] = u1(1)
                idrecs_set = new_idrecs_set

            set_f_ids = np.empty(len(idrecs_set),dtype=np.int64)
            for i, x in enumerate(idrecs_set):
                set_f_ids[i] = decode_idrec(x)[1]

            # print("i", i, "set", set_f_ids, 'new', np.array([decode_idrec(x)[1] for x in idrecs]))
        else:
            m_node.idrecs = idrecs

    
    # If multiple dependencies copy remaining contents of idrecs_set into an array.
    if(multiple_deps):
        idrecs = np.empty((len(idrecs_set),), dtype=np.uint64)
        for i, idrec in enumerate(idrecs_set):
            idrecs[i] = idrec
        m_node.idrecs = idrecs
    # return idrecs

        # print("Update", m_node.node.lit, m_node.node.var_inds[assoc_arg_ind],
        #       "from", dep_m_node.node.lit, dep_m_node.node.var_inds[m_node.dep_arg_ind],
        #       ":", np.array([decode_idrec(x)[1] for x in m_node.idrecs]))
    # print("END DOWNSTREAM")


@njit(types.boolean(MatchIterNodeType))
def update_no_depends(m_node):
    # From the output associated with m_node make a copy of match_idrecs
    #  that omits all of the zeros. 
    print("UPDA TERMINAL")
    cnt = 0
    associated_output = m_node.node.outputs[m_node.associated_arg_ind]
    matches = associated_output.match_idrecs
    idrecs = np.empty((len(matches)),dtype=np.uint64)
    for j, idrec in enumerate(matches):
        if(idrec == 0): continue
        idrecs[cnt] = idrec; cnt += 1;
    print(m_node.node.lit, m_node.associated_arg_ind, idrecs)
    # If the output only had zeros then prematurely mark m_node as empty.
    if(cnt == 0):
        return False

    # Otherwise update the idrecs for the matches of m_node.
    m_node.idrecs = idrecs[:cnt]
    print("END UPDA TERMINAL")
    return True


@njit(u8[::1](GenericMatchIteratorType),cache=True)
def match_iter_next_idrecs(m_iter):
    print("START", m_iter.is_empty)
    n_vars = len(m_iter.iter_nodes)
    # Increment the m_iter nodes until satisfying all upstream
    while(not m_iter.is_empty):
        
        most_upstream_overflow = -1
        # On a fresh iterator skip incrementating and 
        #  just make sure upstream updates are applied
        if(not m_iter.iter_started):
            most_upstream_overflow = 0
            m_iter.iter_started = True

        # Otherwise increment from downstream to upstream
        else:
            most_upstream_overflow = -1

            # For each m_node from downstream to upstream
            for i in range(n_vars-1,-1,-1):
                m_node = m_iter.iter_nodes[i]
                # Increment curr_ind if m_node the most downstream or if
                #  a downstream m_node overflowed.
                if(i == n_vars-1 or most_upstream_overflow == i+1):
                    m_node.curr_ind += 1

                # Track whether incrementing also overflowed the ith m_node.
                if(m_node.curr_ind >= len(m_node.idrecs)):
                    m_node.curr_ind = 0
                    most_upstream_overflow = i

            # If the 0th m_node overflows then iteration is finished.
            if(most_upstream_overflow == 0):
                m_iter.is_empty = True

        # Starting with the most upstream overflow and moving downstream set m_node.idrecs
        #  to be the set of idrecs consistent with its upstream dependencies.
        idrec_sets_are_nonzero = True
        for i in range(most_upstream_overflow, n_vars):
            m_node = m_iter.iter_nodes[i]

            # Only Update if has dependencies
            if(len(m_node.dep_m_node_inds)):
                update_from_upstream_match(m_iter, m_node)

            if(len(m_node.idrecs) == 0): idrec_sets_are_nonzero = False;

        # Note: Keep these prints for debugging
        print("<< it: ", np.array([y.curr_ind for y in m_iter.iter_nodes]))        
        print("<< lens", np.array([len(m_iter.iter_nodes[i].idrecs) for i in range(n_vars)]))    

        # If each m_node has a non-zero idrec set we can yield a match
        #  otherwise we need to keep iterating
        if(idrec_sets_are_nonzero): break

    if(m_iter.is_empty or n_vars == 0): raise StopIteration()

    # Fill in the matched idrecs
    idrecs = np.empty(n_vars,dtype=np.uint64) 
    for i in range(n_vars-1,-1,-1):
        m_node = m_iter.iter_nodes[i]
        idrecs[m_node.var_ind] = m_node.idrecs[m_node.curr_ind]

    print("END")
    return idrecs

# @njit(u8[::1](GenericMatchIteratorType),cache=True)
# def match_iter_next_idrecs(m_iter):
#     n_vars = len(m_iter.iter_nodes)
#     end = len(m_iter.iter_nodes)-1
#     if(m_iter.is_empty or n_vars == 0): raise StopIteration()

#     # Build an array 'idrecs' for the idrecs of the next match set of Facts.
#     idrecs = np.empty(n_vars,dtype=np.uint64)

#     # For each 
#     most_upstream_overflow = -1
#     # for i, m_node in enumerate(m_iter.iter_nodes):
#     for i in range(end ,-1,-1):
#         m_node = m_iter.iter_nodes[i]
        

        

#         print(i, "v:", m_node.var_ind, m_node.curr_ind, np.array([decode_idrec(x)[1] for x in m_node.idrecs]) ,most_upstream_overflow)
#         # Fill the next idrec
#         # print("B")
#         # print(i, m_node.var_ind, decode_idrec(m_node.idrecs[m_node.curr_ind])[0])
#         idrecs[m_node.var_ind] = m_node.idrecs[m_node.curr_ind]

#         # Increment the current index for the m_node
#         if(i == end or most_upstream_overflow == i+1):
#             m_node.curr_ind += 1
        
#         # Track whether incrementing overflowed the m_node.
#         if(m_node.curr_ind >= len(m_node.idrecs)):
#             # If the last m_node overflows then iteration is finished.
#             if(i == 0):
#                 m_iter.is_empty = True
#                 return idrecs

#             if(m_node.dep_m_node_ind != -1):
#                 dep_m_node = m_iter.iter_nodes[m_node.dep_m_node_ind]
#                 update_from_upstream_match(m_node, dep_m_node)
                
#             m_node.curr_ind = 0
#             most_upstream_overflow = i

        

#         # print(i, most_upstream_overflow)
#     print("idrecs:", np.array([decode_idrec(x)[1] for x in idrecs]))
    # if(most_upstream_overflow != -1):
    #     # print("restich")
    #     restitch_match_iter(m_iter, most_upstream_overflow)

    return idrecs

@njit(i8[::1](GenericMatchIteratorType), cache=True)
def match_iter_next_ptrs(m_iter):
    ms, graph = m_iter.graph.memset, m_iter.graph
    idrecs = match_iter_next_idrecs(m_iter)
    # print("^^", idrecs)
    ptrs = np.empty(len(idrecs),dtype=np.int64)
    for i, idrec in enumerate(idrecs):
        t_id, f_id, _  = decode_idrec(idrec)
        facts = _struct_from_ptr(VectorType, ms.facts[t_id])
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
    parse_change_queue(graph)
    # print("PARSED")
    for lst in graph.nodes_by_nargs:
        for node in lst:
            # print(node)
            update_node(node)        


@njit(GenericMatchIteratorType(MemSetType, ConditionsType), cache=True)
def get_match_iter(ms, conds):
    # print("START", conds.matcher_inst_ptr)
    if(i8(conds.matcher_inst_ptr) == 0):
        rete_graph = build_rete_graph(ms, conds)
        # conds.matcher_inst_meminfo = _meminfo_from_struct(rete_graph)
        conds.matcher_inst_ptr = _ptr_from_struct_incref(rete_graph)
    # print("BUILT", conds.matcher_inst_ptr)
    rete_graph = _struct_from_ptr(ReteGraphType, conds.matcher_inst_ptr)
    # print("LOADED")
    update_graph(rete_graph)
    # print("UPDATED")
    m_iter = new_match_iter(rete_graph)
    # print("NEW MATCH ITER")
    for i in range(len(m_iter.iter_nodes)):
        m_node = m_iter.iter_nodes[i]
        m_node.curr_ind = 0
        if(len(m_node.dep_m_node_inds) == 0):
            ok = update_no_depends(m_node)
            if(not ok):
                m_iter.is_empty = True
                break
            # print("<<", m_node.var_ind, m_node.idrecs)
            # print(m_node.idrecs, m_node.node.lit)
        # else:
        #     update_from_upstream_match(m_iter,  m_node)
        
    # print("FINISH UPDATE")
    # print("<< lens BEG:", np.array([len(m_node.idrecs) for m_node in m_iter.iter_nodes]))
    # for i, m_node in enumerate(m_iter.iter_nodes):
    #     print("<<", m_node.curr_ind)
    # print("initial restich")
    # restitch_match_iter(m_iter, -1)
    return m_iter





if __name__ == "__main__":
    pass
    # deref_depends = Dict.empty(i8, DictType(i8,u1))

    # node_ctor()







"""

Notes: 

What is needed by 

    









"""
