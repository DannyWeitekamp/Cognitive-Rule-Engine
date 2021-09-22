import numpy as np
from numba import types, njit, i8, u8, i4, u1, i8, f8, literally, generated_jit
from numba.typed import Dict, List
from numba.types import DictType, ListType
from cre.memory import Memory
from cre.fact import define_fact
from cre.var import Var, GenericVarType
from cre.utils import pointer_from_struct, decode_idrec, encode_idrec, _struct_from_pointer, _pointer_from_struct, _load_pointer
from cre.rete import (deref_head_and_relevant_idrecs, RETRACT,
     DerefRecord, make_deref_record_parent, invalidate_deref_rec,
     validate_deref, validate_head_or_retract)
from cre.context import cre_context
from cre.default_ops import Add


def setup_deref_tests(ctx=None):
    with cre_context(ctx):
        BOOP, BOOPType = define_fact("BOOP",{"nxt" : "BOOP", "val" : f8})
        mem = Memory()

        a5 = BOOP(nxt=None, val=5)
        a4 = BOOP(nxt=a5, val=5)
        a3 = BOOP(nxt=a4, val=5)
        a2 = BOOP(nxt=a3, val=5)
        a1 = BOOP(nxt=a2, val=5)
        mem.declare(a1)
        mem.declare(a2)
        mem.declare(a3)
        mem.declare(a4)
        mem.declare(a5)
        return (mem, BOOP, BOOPType), (a1,a2,a3,a4,a5)



def test_deref_to_head_and_gen_relevant_idrecs():
    f = deref_head_and_relevant_idrecs
    
    (mem, BOOP, BOOPType), (a1,a2,a3,a4,a5) = setup_deref_tests()
    a1_ptr = pointer_from_struct(a1)
    a3_ptr = pointer_from_struct(a3)

    v = Var(BOOP,"a").nxt.val
    a_n, a_v = [x[1] for x in v.deref_offsets]
    a_r = RETRACT
    t = decode_idrec(a1.idrec)[0]
    f_a1 = decode_idrec(a1.idrec)[1]
    f_a2 = decode_idrec(a2.idrec)[1]
    f_a3 = decode_idrec(a3.idrec)[1]
    f_a4 = decode_idrec(a4.idrec)[1]
    f_a5 = decode_idrec(a5.idrec)[1]
    

    # print(BOOPType.field_dict)
    v = Var(BOOP,"a").nxt.val
    head_ptr, rel_idrecs = f(a1_ptr, v.deref_offsets)
    assert head_ptr > 0
    assert [decode_idrec(x) for x in rel_idrecs] == \
        [(t, f_a1, a_n), (t, f_a2, a_r), (t, f_a2, a_v)]
    # print(head_ptr, [decode_idrec(x) for x in rel_idrecs])

    v = Var(BOOP,"a").nxt.nxt.val
    head_ptr, rel_idrecs = f(a1_ptr, v.deref_offsets)
    assert head_ptr > 0
    assert [decode_idrec(x) for x in rel_idrecs] == \
        [(t, f_a1, a_n), (t, f_a2, a_r), (t, f_a2, a_n), (t, f_a3, a_r), (t, f_a3, a_v)]
    # print(head_ptr, [decode_idrec(x) for x in rel_idrecs])

    v = Var(BOOP,"a").nxt.nxt.nxt.val
    head_ptr, rel_idrecs = f(a1_ptr, v.deref_offsets)
    assert head_ptr > 0
    assert [decode_idrec(x) for x in rel_idrecs] == \
        [(t, f_a1, a_n), (t, f_a2, a_r), (t, f_a2, a_n), (t, f_a3, a_r), (t, f_a3, a_n), (t, f_a4, a_r), (t, f_a4, a_v)]

    v = Var(BOOP,"a").nxt.nxt.nxt.val
    head_ptr, rel_idrecs = f(a3_ptr, v.deref_offsets)
    assert head_ptr == 0
    assert len(rel_idrecs) == 5

def test_deref_record_parent():
    (mem, BOOP, BOOPType), (a1,a2,a3,a4,a5) = setup_deref_tests()

    #Make deref_depends
    deref_depends = Dict.empty(i8, DictType(i8,u1))

    #Make idrecs
    v = Var(BOOP,"a").nxt.val
    a_n, a_v = [x[1] for x in v.deref_offsets]

    t_id, f_id, _ = decode_idrec(a1.idrec)
    idrec1_n = encode_idrec(t_id, f_id, a_n)
    t_id, f_id, _ = decode_idrec(a2.idrec)
    idrec2_r = encode_idrec(t_id, f_id, RETRACT)
    t_id, f_id, _ = decode_idrec(a2.idrec)
    idrec2_v = encode_idrec(t_id, f_id, a_v)

    #Make DerefRecord
    parent_ptrs = np.zeros((3,), dtype=np.int64)
    r = DerefRecord(parent_ptrs, 0, 1)
    r_ptr = pointer_from_struct(r)


    p_ptr = make_deref_record_parent(deref_depends, idrec1_n, r_ptr)
    parent_ptrs[0] = p_ptr
    print(p_ptr)
    p_ptr = make_deref_record_parent(deref_depends, idrec2_r, r_ptr)
    parent_ptrs[1] = p_ptr
    print(p_ptr)
    p_ptr = make_deref_record_parent(deref_depends, idrec2_v, r_ptr)
    parent_ptrs[2] = p_ptr
    print(p_ptr)

    print(parent_ptrs)

    print(deref_depends)
    assert idrec1_n in deref_depends
    assert idrec2_r in deref_depends
    assert idrec2_v in deref_depends

    invalidate_deref_rec(r)
    print(deref_depends)


def _test_validate_deref():
    from cre.conditions import Literal
    from cre.rete import node_ctor
    from cre.var import GenericVarType
    (mem, BOOP, BOOPType), (a1,a2,a3,a4,a5) = setup_deref_tests()


    v = Var(BOOP,"a").nxt.val
    deref_offsets = v.deref_offsets
    a_n, a_v = [x[1] for x in v.deref_offsets]

    op = (v < 1)

    t_id, f_id, _ = decode_idrec(a1.idrec)

    node = node_ctor(mem,Literal(op),np.array([t_id],dtype=np.uint16), np.arange(1,dtype=np.int64))
    print(node)

    _, f_id, _ = decode_idrec(a1.idrec)
    head_ptr = validate_deref(node,t_id,f_id, deref_offsets)
    assert head_ptr != 0 
    print(head_ptr)

    _, f_id, _ = decode_idrec(a2.idrec)
    head_ptr = validate_deref(node,t_id,f_id, deref_offsets)
    assert head_ptr != 0
    print(head_ptr)

    _, f_id, _ = decode_idrec(a3.idrec)
    head_ptr = validate_deref(node,t_id,f_id, deref_offsets)
    assert head_ptr != 0
    print(head_ptr)

    _, f_id, _ = decode_idrec(a4.idrec)
    head_ptr = validate_deref(node,t_id,f_id, deref_offsets)
    assert head_ptr != 0
    print(head_ptr)

    _, f_id, _ = decode_idrec(a5.idrec)
    head_ptr = validate_deref(node,t_id,f_id, deref_offsets)
    assert head_ptr == 0
    print(head_ptr)


def _test_validate_head_or_retract():
    from cre.conditions import Literal
    from cre.var import GenericVarType
    from cre.rete import node_ctor
    (mem, BOOP, BOOPType), (a1,a2,a3,a4,a5) = setup_deref_tests()


    v = Var(BOOP,"a").nxt.val
    a_n, a_v = [x[1] for x in v.deref_offsets]

    t_id, f_id, _ = decode_idrec(a1.idrec)

    op = (v < 1)

    node = node_ctor(mem,Literal(op),np.array([t_id],dtype=np.uint16), np.arange(1,dtype=np.int64))
    print(node)

    _, f_id, _ = decode_idrec(a1.idrec)
    head_ptr = validate_head_or_retract(node,0,f_id,0)
    print(node.inp_head_ptrs)

    _, f_id, _ = decode_idrec(a2.idrec)
    head_ptr = validate_head_or_retract(node,0,f_id,0)
    print(node.inp_head_ptrs)

    _, f_id, _ = decode_idrec(a3.idrec)
    head_ptr = validate_head_or_retract(node,0,f_id,0)
    print(node.inp_head_ptrs)

    _, f_id, _ = decode_idrec(a4.idrec)
    head_ptr = validate_head_or_retract(node,0,f_id,0)
    print(node.inp_head_ptrs)

    # _, f_id, _ = decode_idrec(a5.idrec)
    # head_ptr = validate_head_or_retract(node,0,f_id,0)
    # print(head_ptr)

from cre.conditions import LiteralType

distr_conj_type = ListType(ListType(LiteralType))
literal_list_type = ListType(LiteralType)












# @njit(cache=True)
# def print_stuff(c):

def list_to_str(x):
    ''' Helper function that makes a copy of a nested list but with 
        all non list-like 'x' turned into str(x) '''
    if(isinstance(x,(list,List))):
        return [list_to_str(y) for y in x]
    return str(x)


def test_distr_dnf_and():
    from cre.rete import node_ctor
    from cre.var import GenericVarType
    from cre.conditions import as_distr_dnf_list
    with cre_context("test_distr_dnf_and"):
        BOOP, BOOPType = define_fact("BOOP",{"nxt" : "BOOP", "val" : f8})
    
        a,b,c = Var(BOOP,"a"), Var(BOOP,"b"), Var(BOOP,"c")

        ideal_conds = (
            a & (a.val > 1) & (a.nxt.val < a.val) & 
            b & (b.val > 1) & (b.val != 0) & (b.val != 0) & (a.val > b.val) & (b.val != a.val) & 
            c & (c.val == b.nxt.val) & (c.val != 0)
        )

        mixup_conds = (
            a & b & c &
            (c.val == b.nxt.val) & (c.val != 0)  &
            (a.val > 1) & (a.nxt.val < a.val) & (a.val > b.val) &
            (b.val != a.val) & (b.val > 1) & (b.val != 0) & (b.val != 0)
        )
        mixup_conds.distr_dnf
        ideal_conds.distr_dnf
        assert str(mixup_conds.distr_dnf) == str(ideal_conds.distr_dnf)

        py_list_distr_dnf = as_distr_dnf_list(mixup_conds.distr_dnf)
        assert list_to_str(py_list_distr_dnf) == list_to_str(mixup_conds.distr_dnf)


from cre.rete import (dict_u8_u1_type, update_changes_deref_dependencies,
     update_changes_from_inputs, validate_head_or_retract, update_node)
# @njit(cache=True)
def filter_first(graph):
    print("HEY")
    # for i,self in graph.var_root_nodes.items():
    #     print("HEY",i)
    #     update_node(self)

    for lst in graph.nodes_by_nargs:
        for node in lst:
            update_node(node)        

    #     arg_change_sets = List.empty_list(dict_u8_u1_type)
    #     for i in range(len(self.var_inds)):
    #          arg_change_sets.append(Dict.empty(u8,u1))

    #     update_changes_deref_dependencies(self, arg_change_sets)
    #     update_changes_from_inputs(self, arg_change_sets)

    #     print("<<", i)
    #     for idrec in arg_change_sets[0]:
    #         print(decode_idrec(idrec))
    #     # print(i, arg_change_sets)


    #      ### Make sure the arg_change_sets are up to date

    # for i,change_set in enumerate(arg_change_sets):
    #     for idrec in change_set:
    #         _, f_id, a_id = decode_idrec(idrec)
    #         validate_head_or_retract(self, i, f_id, a_id)

    # # head_ptrs = np.empty((len(self.inp_head_ptrs),),dtype=np.float64)

    # print()

    # print(self.inp_head_ptrs)
    # for i, ptrs in self.inp_head_ptrs[0].items():
    #     print("VAL",_load_pointer(f8,ptrs[0]))

@njit(cache=True)
def iter_matches(self):
    print("START ITR")

    prev_var_ind = -1
    for i, node in self.var_end_nodes.items():
        ind_i = np.min(np.nonzero(node.var_inds == i)[0])
        output_i = node.outputs[ind_i]

        if(prev_var_ind != -1):
            print("<<", prev_var_ind, node.var_inds)
            prev_inds = np.nonzero(node.var_inds == prev_var_ind)[0]
            if(len(prev_inds) > 0):
                prev_ind = np.min(prev_inds)
                output_prev = node.outputs[prev_ind]
                print(output_i.matches, output_prev.matches)
                continue
        # else:
        print(output_i.matches)
        prev_var_ind = i
        # print(i, node.var_inds, j)
        # print(i, "---", node.op, "---")
        # print([x.matches for x in node.outputs])

def test_build_rete_graph():
    from cre.rete import (node_ctor, build_rete_graph, parse_mem_change_queue,
                            make_match_iter,repr_match_iter_dependencies, copy_match_iter,
                            restitch_match_iter, match_iter_next)
    from cre.var import GenericVarType
    from cre.conditions import as_distr_dnf_list
    with cre_context("test_distr_dnf_and"):
        BOOP, BOOPType = define_fact("BOOP",{"nxt" : "BOOP", "val" : f8})
    
        a,b,c = Var(BOOP,"a"), Var(BOOP,"b"), Var(BOOP,"c")    

        # mixup_conds = (
        #     a & b & c & (c.val != 0)  &
        #     (a.val > 1) & (a.nxt.val > a.val) & (a.val > b.val) &
        #     (b.val != a.val) & (b.val > 1) & (b.val != 0) & (b.val != 0)
        # )

        mixup_conds = (
            a & b & c & (a.val > 2) #& (b.val != 0) & (b.val < a.val) &
            # (c.val == a.val)
        )

        mem = Memory()

        graph = build_rete_graph(mem, mixup_conds)
        print([[str(x.op) for x in y] for y in graph.nodes_by_nargs])
        print({i:str(x.op) for i,x in graph.var_end_nodes.items()})
        print({i:str(x.op) for i,x in graph.var_root_nodes.items()})

        a4 = BOOP(nxt=None, val=4)
        a3 = BOOP(nxt=a4, val=3)
        a2 = BOOP(nxt=a3, val=2)
        a1 = BOOP(nxt=a2, val=1)
        a0 = BOOP(nxt=a1, val=0)
        mem.declare(a0)
        mem.declare(a1)
        mem.declare(a2)
        mem.declare(a3)
        mem.declare(a4)

        parse_mem_change_queue(graph)

        filter_first(graph)


        m_iter = make_match_iter(graph)
        print(repr_match_iter_dependencies(m_iter))
        m_iter_copy = copy_match_iter(m_iter)
        print(repr_match_iter_dependencies(m_iter_copy))

        print(restitch_match_iter(m_iter_copy))

        while(True):
            try:
                print(match_iter_next(m_iter_copy))        
            except StopIteration:
                print("STOP ITER")
                break

        

        # iter_matches(graph)
        # update_deref_dependencies(self, arg_change_sets)
        # update_changes_from_inputs(self, arg_change_sets)

        # print(a,b,c)

        # print(nodes_by_nargs, var_end_nodes)








if __name__ == "__main__":

    # _test_validate_deref()
    # _test_validate_head_or_retract()
    # test_deref_to_head_and_gen_relevant_idrecs()
    # test_deref_record_parent()
    # _test_validate_deref()
    # _test_head_map()
    # _test_validate_head_or_retract()
    # test_distr_dnf_and()
    test_build_rete_graph()

