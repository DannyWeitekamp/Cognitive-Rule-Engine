

rete_node_fields = {
    "inputs" : ListType(??),
    "change_set" : akd(i8),
    "output" : akd(i8),
    "arg_indicies" : i8[::1],
    "literal" : LiteralType,
}


DEC, RET = 1, 0


def update_node_output(self, fids, passes):
    is_in_out = fids in self.output
    if(passes):
        if(not is_in_out): 
            self.change_set[fids] = DEC
            self.output[fids] = DEC
    else:
        if(is_in_out):
            self.change_set[fids] = RET
            del self.output[fids]
    
    

def update_alpha(self):
    self.change_set = Dict.empty(i8[::1],u1)
    if(self.inp == None):
        for fid, dec_or_ret  in sefl.mem.change_queue.items():
            fact = mem.get_by_fid(fid)
            passes = self.passes(fact)
            update_node_output(self, [fid,])
    else:
        for fids, dec_or_ret in self.inps[0].change_set.items():
            fact = mem.get_by_fid(fids[0])
            passes = self.passes(fact)
            update_node_output(self, fids)

    return len(self.change_set) > 0


def update_beta(self):
    self.change_set = Dict.empty(i8[::1],u1)
    check_set = Dict.empty(i8[::1],u1)
    for fids0, dec_or_ret0 in self.inps[0].change_set.items():
        for fids1, dec_or_ret0 in self.inps[1].output.items():
            check_set[[*fids0, fids1]] = 1

    for fids1, dec_or_ret1 in self.inps[1].change_set.items():
        for fids0, dec_or_ret0 in self.inps[0].output.items():
            check_set[[*fids0, fids1]] = 1

    l_ind = self.arg_indicies[0]
    r_ind = self.arg_indicies[1]
    for fids in check_set:
        left_fact = mem.get_by_fid(fids[l_ind])
        right_fact = mem.get_by_fid(fids[r_ind])
        passes = self.passes(left_fact, right_fact)
        update_node_output(self, fids, passes)

    return len(self.change_set) > 0




