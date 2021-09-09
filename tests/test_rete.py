import numpy as np
from numba import types, njit, i8, u8, i4, u1, i8, f8, literally, generated_jit
from numba.typed import Dict, List
from numba.types import DictType, ListType
from cre.memory import Memory
from cre.fact import define_fact
from cre.var import Var
from cre.utils import pointer_from_struct, decode_idrec, encode_idrec
from cre.rete import (deref_head_and_relevant_idrecs, RETRACT,
     DerefRecord, make_deref_record_parent, invalidate_deref_rec,
     validate_deref, validate_head_or_retract)


def setup_deref_tests():
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
    from cre.rete import node_ctor
    from cre.var import GenericVarType
    (mem, BOOP, BOOPType), (a1,a2,a3,a4,a5) = setup_deref_tests()


    v = Var(BOOP,"a").nxt.val
    a_n, a_v = [x[1] for x in v.deref_offsets]

    t_id, f_id, _ = decode_idrec(a1.idrec)

    _vars = List.empty_list(GenericVarType)
    _vars.append(v)


    node = node_ctor(mem,_vars,np.array([t_id],dtype=np.uint16), np.arange(1,dtype=np.int64))
    print(node)

    _, f_id, _ = decode_idrec(a1.idrec)
    head_ptr = validate_deref(node,0,f_id)
    print(head_ptr)

    _, f_id, _ = decode_idrec(a2.idrec)
    head_ptr = validate_deref(node,0,f_id)
    print(head_ptr)

    _, f_id, _ = decode_idrec(a3.idrec)
    head_ptr = validate_deref(node,0,f_id)
    print(head_ptr)

    _, f_id, _ = decode_idrec(a4.idrec)
    head_ptr = validate_deref(node,0,f_id)
    print(head_ptr)

    _, f_id, _ = decode_idrec(a5.idrec)
    head_ptr = validate_deref(node,0,f_id)
    print(head_ptr)


def _test_validate_head_or_retract():
    from cre.rete import node_ctor
    from cre.var import GenericVarType
    (mem, BOOP, BOOPType), (a1,a2,a3,a4,a5) = setup_deref_tests()


    v = Var(BOOP,"a").nxt.val
    a_n, a_v = [x[1] for x in v.deref_offsets]

    t_id, f_id, _ = decode_idrec(a1.idrec)

    _vars = List.empty_list(GenericVarType)
    _vars.append(v)


    node = node_ctor(mem,_vars,np.array([t_id],dtype=np.uint16), np.arange(1,dtype=np.int64))
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











if __name__ == "__main__":
    # test_deref_to_head_and_gen_relevant_idrecs()
    # test_deref_record_parent()
    # _test_validate_deref()
    _test_validate_head_or_retract()