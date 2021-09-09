import numpy as np
from numba import types, njit, i8, u8, i4, u1, u2, i8, f8, literally, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple
from numba.experimental.structref import new
from cre.utils import (_dict_from_ptr, _pointer_from_struct,
         _pointer_from_struct_incref, _struct_from_pointer, decode_idrec,
        encode_idrec, deref_type, OFFSET_TYPE_ATTR, OFFSET_TYPE_LIST,
         _pointer_to_data_pointer, _list_base_from_ptr, _load_pointer,
         _decref_structref)
from cre.structref import define_structref
from cre.memory import MemoryType
from cre.vector import VectorType
from cre.var import GenericVarType


RETRACT = u1(0xFF)# u1(0)
DECLARE = u1(0)

deref_record_field_dict = {
    # Pointers to the Dict(i8,u1)s inside deref_depends
    "parent_ptrs" : i8[::1], 
    "arg_ind" : i8,
    "was_successful" : u1,
}

DerefRecord, DerefRecordType = define_structref("DerefRecord", deref_record_field_dict)


dict_i8_u1_type = DictType(i8,u1)
dict_u8_i8_type = DictType(u8,i8)
deref_dep_typ = DictType(u8,DictType(i8,u1))


# outputs_item_typ = DictType(u8,DictType(u8,u1))
dict_u8_u1_type = DictType(u8,u1)
outputs_typ = DictType(u8,DictType(u8,u1))

base_rete_node_field_dict = {
    # Pointers to the Dict(i8,u1)s inside deref_depends
    "mem" : MemoryType, 
    "deref_depends" : deref_dep_typ, 
    "arg_inds" : i8[::1],
    "t_ids" : u2[::1],
    "vars" : ListType(GenericVarType),
    "inp_head_ptrs" : ListType(DictType(u8,i8)),
    "inputs" : types.Any,
    "outputs" : ListType(DictType(u8,DictType(u8,u1))),
}

BaseReteNode, BaseReteNodeType = define_structref("BaseReteNode", base_rete_node_field_dict, define_constructor=False)


@njit(cache=True)
def node_ctor(mem, _vars, t_ids, arg_inds ):
    st = new(BaseReteNodeType)
    st.mem = mem
    st.deref_depends = Dict.empty(u8,dict_i8_u1_type)
    st.arg_inds = arg_inds
    st.t_ids = t_ids
    st.vars = _vars

    inp_head_ptrs = List.empty_list(dict_u8_i8_type)
    for i in range(len(_vars)):
        inp_head_ptrs.append(Dict.empty(u8,i8))
    st.inp_head_ptrs = inp_head_ptrs

    outputs = List.empty_list(outputs_typ)
    for i in range(len(_vars)):
        outputs.append(Dict.empty(u8,dict_u8_u1_type))
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

@njit(i8(BaseReteNodeType, i8, u8),cache=True)
def validate_deref(self, k, f_id):
    '''Try to get the head_ptr of 'f_id' in input 'k'. Inject a DerefRecord regardless of the result '''
    t_id = self.t_ids[k]
    facts = _struct_from_pointer(VectorType,self.mem.mem_data.facts[t_id])
    base_ptr = facts.data[f_id]
    deref_offsets = self.vars[k].deref_offsets
    head_ptr, rel_idrecs  = deref_head_and_relevant_idrecs(base_ptr,deref_offsets)
    was_successful = (head_ptr != 0)
    parent_ptrs = np.empty(len(rel_idrecs), dtype=np.int64)
    r = DerefRecord(parent_ptrs, self.arg_inds[k], was_successful)
    r_ptr = _pointer_from_struct_incref(r)
    for i, idrec in enumerate(rel_idrecs):
        ptr = make_deref_record_parent(self.deref_depends, idrec, r_ptr)
        parent_ptrs[i] = ptr

    return head_ptr

@njit(void(BaseReteNodeType, i8,u8,u1),locals={"f_id" : u8}, cache=True)
def validate_head_or_retract(self, k, f_id, a_id):
    '''Update the head_ptr dictionaries by following the deref
     chains of DECLARE/MODIFY changes, and make retractions
    for an explicit RETRACT or a failure in the deref chain.'''
    if(a_id != RETRACT):
        head_ptr = validate_deref(self, k, f_id)
        print(head_ptr)
        if(head_ptr != 0): 
            self.inp_head_ptrs[k][f_id] = head_ptr
            return
        
    # At this point we are definitely RETRACT
    to_clear = self.outputs[k][a_id]
    for x in to_clear:
        other_out = self.outputs[1 if k else 0][x]
        del other_out[a_id]
    del self.outputs[k][a_id]
    del self.inp_head_ptrs[k][f_id]


def filter_beta(self):
    arg_change_sets = List([Dict(i8,u1), Dict(i8,u1)])

    ### 'relevant_global_diffs' is the set of self.mem.change_queue items relevant to intermediate derefs computed for this literal, and modification of the head attribute. Shouldn't happen frequently ###
    for idrec in self.relevant_global_diffs:
        if(idrec in self.deref_depends and len(self.deref_depends) > 0):
            deref_records = self.deref_depends[idrec]

            # Invalidate old DerefRecords
            for r in deref_records:
                r.invalidate()
                _, base_f_id, _ = decode_idrec(r.dep_idrecs[0])

                # Any change in the deref chain counts as a MODIFY
                arg_change_sets[r.arg_ind][base_f_id] = MODIFY


    ### Make sure the arg_change_sets are up to date
    for i, inp in enumerate(self.inputs.change_set):
        arg_change_sets_i = arg_change_sets[i]
        if(len(arg_change_sets_i) > 0):
             for f_id in inp.change_set:
                arg_change_sets_i.add(idrec)
        else:
            arg_change_sets[i] = inp.change_set


    ### Update the head_ptr dictionaries by following the deref chains of DECLARE/MODIFY changes, and make retractions for an explicit RETRACT or a failure in the deref chain.
    for f_id0, a_id0 in arg_change_sets[0].items():
        validate_head_or_retract(self, f_id0, a_id0)

    for f_id1, a_id1 in arg_change_sets[1].items():
        validate_head_or_retract(self, f_id1, a_id1)

    ### Check all pairs at this point we should only be dealing with DECLARE/MODIFY changes
    for f_id0, a_id0 in arg_change_sets[0].items():
        h_ptr0 = self.head_ptrs[0][f_id0]
        if(a_id0 != RETRACT):
            for h_ptr1 in self.head_ptrs[1]:
                check_pair(h_ptr0, h_ptr1, a_id0)
            
    for f_id1, a_id1 in arg_change_sets[1].items():
        if(a_id1 != RETRACT):
            h_ptr1 = self.head_ptrs[1][f_id1]
            for f_id0, h_ptr0 in self.head_ptrs[0].items():
                if(f_id0 not in arg_change_sets[0]):
                    check_pair(h_ptr0, h_ptr1, a_id1)

def check_pair(self, h_ptr0, h_ptr1, chg_typ):
    passes = self.call_head_ptrs(h_ptr0, h_ptr1)

    # For any MODIFY type of change
    # if(not passes and chg_typ != DECLARE):

    # else:

if __name__ == "__main__":
    pass
    # deref_depends = Dict.empty(i8, DictType(i8,u1))

    # node_ctor()







"""



    









"""