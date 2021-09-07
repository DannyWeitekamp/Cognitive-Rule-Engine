

rete_node_fields = {
    "inputs" : ListType(??),
    "change_set" : akd(i8),
    "output" : akd(i8),
    "arg_indicies" : i8[::1],
    "literal" : LiteralType
}


DEC, RET = 1, 0

def concat_f_ids(f_ids1, f_ids2):
    out = np.empty(len(f_ids1)+len(f_ids2), dtype=np.int64)
    c = 0
    for x in f_ids1: out[c] = x; c += 1;
    for x in f_ids2: out[c] = x; c += 1;
    return out
       

def update_node_output(self, f_ids, passes):
    is_in_out = f_ids in self.output
    if(passes):
        if(not is_in_out): 
            self.change_set[f_ids] = DEC
            self.output[f_ids] = DEC
    else:
        if(is_in_out):
            self.change_set[f_ids] = RET
            del self.output[f_ids]
    

def update_alpha(self):
    self.change_set = Dict.empty(i8[::1],u1)
    if(self.inp == None):
        for f_id, dec_or_ret  in sefl.mem.change_queue.items():
            fact = mem.get_by_f_id(f_id)
            passes = self.passes(fact)
            update_node_output(self, [f_id,])
    else:
        for f_ids, dec_or_ret in self.inps[0].change_set.items():
            fact = mem.get_by_f_id(f_ids[0])
            passes = self.passes(fact)
            update_node_output(self, f_ids)

    return len(self.change_set) > 0


def update_beta(self):
    self.change_set = Dict.empty(i8[::1],u1)
    check_set = Dict.empty(i8[::1],u1)
    for f_ids0, dec_or_ret0 in self.inps[0].change_set.items():
        for f_ids1, dec_or_ret0 in self.inps[1].output.items():
            check_set[concat_f_ids(f_ids0, f_ids1)] = 1

    for f_ids1, dec_or_ret1 in self.inps[1].change_set.items():
        for f_ids0, dec_or_ret0 in self.inps[0].output.items():
            check_set[concat_f_ids(f_ids0, f_ids1)] = 1

    l_ind = self.arg_indicies[0]
    r_ind = self.arg_indicies[1]
    for f_ids in check_set:
        left_fact = mem.get_by_f_id(f_ids[l_ind])
        right_fact = mem.get_by_f_id(f_ids[r_ind])
        passes = self.passes(left_fact, right_fact)
        update_node_output(self, f_ids, passes)

    return len(self.change_set) > 0




########### V2 ##################


Psuedo code round 2:

ConditionNode
    "inputs" : ListType(??),
    "output" : Vector(*OutputEntryType),
    "arg_indicies" : i8[::1],
    "literal" : LiteralType,
    "deref_t_ids" : i8[::1],
    "depends" : DictType(i8, ListType(OutputEntryType))
    "deref_depends" : List(DictType(i8, ListType(DerefRecord)))
    "deref_val_ptrs" : List(vector(i8))


OutputEntry = {
    "is_valid" : u1,
    "index" : i8,
    "f_ids" : u8[::1], #maybe u4[::1]
}

DerefRecord = {
    "is_valid" : u1,
    "ptr_index" : u1,
    "f_ids" : u8[::1],
}


# inp_change_sets = List()
for i in range(n_args):
    inp_change_set = Dict.empty(i8,u1)
    for f_id, chg_typ in self.inps[i].change_set.items():
        inp_change_set[f_id] = chg_typ

    deref_depends_i, deref_val_ptrs_i = deref_depends[i], deref_val_ptrs[i]
    for chg_typ, f_id in self.mem.change_queue:
        if(f_id in deref_depends_i):
            if(chg_typ == RET or chg_typ == MOD):
                deref_record = deref_depends_i[f_id]
                deref_record.is_valid = False
                deref_val_ptrs_i.remove(deref_record.ptr_index)

            if(chg_typ == DEC or chg_typ == MOD):
                deref_record = new_deref_record(...)
                ...deref_val_ptrs needs to get filled here
                for f_id in deref_record.f_ids:
                    if(f_id not in deref_depends_i): deref_depends_i[f_id] = List()
                    deref_depends_i[f_id].append(deref_record)


            # or something like that 
            inp_change_set[f_id] = chg_typ
    



# So now we need to make the appropriate updates using depends







