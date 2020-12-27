import numpy as np
from numba import types, njit, i8, u8, i4, u1, literally, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type, void
from numba.experimental.structref import new
from numba.extending import overload_method, intrinsic
from numbert.experimental.structref import define_structref, define_structref_template
from numbert.experimental.kb import KnowledgeBaseType, KnowledgeBase
from numbert.experimental.fact import define_fact
from numbert.experimental.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, decode_idrec, lower_getattr
from numbert.experimental.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from copy import copy

meminfo_type = types.MemInfoPointer(types.voidptr)

@njit(cache=True)
def lt(a,b):
    return a < b

@njit(cache=True)
def lte(a,b):
    return a <= b

@njit(cache=True)
def gt(a,b):
    return a > b

@njit(cache=True)
def gte(a,b):
    return a >= b

@njit(cache=True)
def eq(a,b):
    return a == b

@njit(cache=True)
def exec_op(op_str,a,b):
    if(op_str == "<"):
        return lt(a,b)
    elif(op_str == "<="):
        return lte(a,b)
    elif(op_str == ">"):
        return gt(a,b)
    elif(op_str == ">="):
        return gte(a,b)
    elif(op_str == ">="):
        return gte(a,b)
    raise ValueError()

op_str_map = {
    "<" : lt,
    "<=" : lte,
    ">" : gt,
    ">=" : gte,
    "==" : eq
}

def resolve_predicate_op(op):
    if(isinstance(op,str)):
        return op_str_map[op]
    return op



predicate_node_field_dict = {
    "truth_values" : u1[:],
    "ndim" : u1,
    "left_type" : types.Any,
    "left_attr" : types.Any,
    "op_str" : types.Any,
    "right_type" : types.Any,
    "right_attr" : types.Any,
    "right_val" : types.Any,
    # "update_func" : types.FunctionType(void(types.Any,meminfo_type))
}
predicate_node_fields = [(k,v) for k,v, in predicate_node_field_dict.items()]
PredicateNode, PredicateNodeTemplate = define_structref_template("PredicateNode", base_subscriber_fields + predicate_node_fields)


def define_alpha_predicate_node(typ, attr, op, literal_val):
    field_dict = copy(predicate_node_field_dict)
    field_dict["left_type"] = types.TypeRef(typ)
    field_dict["left_attr"] = types.literal(attr)
    field_dict["op_str"] = types.literal(op)
    field_dict["right_val"] = types.literal(literal_val)

    fields = base_subscriber_fields + [(k,v) for k,v, in field_dict.items()]

    pnode_type = PredicateNodeTemplate(fields=fields)

    @njit(cache=True)
    def eval_truth(kb,t_id,f_id):
        inst = _cast_structref(typ, kb.kb_data.facts[i8(t_id)][i8(f_id)])
        if(inst.idrec != u8(-1)):
            val = lower_getattr(inst, attr)
            return exec_op(op,val,literal_val)
        else:
            return 0xFF


    @njit(cache=True,locals={'new_size':u8})
    def update_func(pred_meminfo):
        if(pred_meminfo is None): return
        pred_node = _struct_from_meminfo(pnode_type, pred_meminfo)
        grw_s = pred_node.grow_queue
        chg_s = pred_node.change_queue
        kb = _struct_from_meminfo(KnowledgeBaseType, pred_node.kb_meminfo)

        new_size = 0
        if len(grw_s) > 0:
            new_size = max([decode_idrec(idrec)[1] for idrec in grw_s])+1

        if(new_size > 0):
            new_truth_values = np.empty((new_size,),dtype=np.uint8)
            for i,b in enumerate(pred_node.truth_values):
                new_truth_values[i] = b
        else:
            new_truth_values = pred_node.truth_values

        if(len(pred_node.grow_queue) > 0):
            for idrec in pred_node.grow_queue:
                t_id,f_id,_ = decode_idrec(idrec)
                new_truth_values[f_id] = eval_truth(kb,t_id,f_id)

                for child_meminfo in pred_node.children:
                    child = _struct_from_meminfo(BaseSubscriberType,child_meminfo)
                    child.grow_queue.append(idrec)
            pred_node.grow_queue = List.empty_list(u8)
            pred_node.truth_values = new_truth_values

        if(len(pred_node.change_queue) > 0):
            for idrec in pred_node.change_queue:
                t_id,f_id,_ = decode_idrec(idrec)
                truth = eval_truth(kb,t_id,f_id)
                new_truth_values[f_id] = truth
                if(truth != pred_node.truth_values[f_id]):
                    for child_meminfo in pred_node.children:
                        child = _struct_from_meminfo(BaseSubscriberType, child_meminfo)
                        child.change_queue.append(idrec)
            pred_node.change_queue = List.empty_list(u8)
            pred_node.truth_values = new_truth_values


    @njit(cache=True)
    def ctor():
        st = new(pnode_type)
        init_base_subscriber(st)
        st.update_func = update_func

        st.truth_values = np.empty((0,),dtype=np.uint8)
        st.ndim = 1
        st.left_type = typ
        st.left_attr = attr
        st.op_str = op
        st.right_val = literal_val
        return st
    return ctor, pnode_type

# @njit
# def set_update_func(pred_node,update_func):
#     pred_node.update_func = update_func

def get_alpha_predicate_node(typ, attr, op, literal_val):
    ctor, pnode_type = define_alpha_predicate_node(typ, attr, op, literal_val)

    out = ctor()
    # set_update_func(out, update_func)
    # out.kb = kb
    return out

def define_beta_predicate_node(left_type, left_attr, op, right_type, right_attr):
    # op = resolve_predicate_op(op)
    field_dict = copy(predicate_node_field_dict)
    field_dict["left_type"] = types.TypeRef(left_type)
    field_dict["left_attr"] = types.literal(left_attr)
    field_dict["op_str"] = types.literal(op)
    field_dict["right_type"] = types.TypeRef(right_type)
    field_dict["right_attr"] = types.literal(right_attr)
    fields = base_subscriber_fields + [(k,v) for k,v, in field_dict.items()]

    pnode_type = PredicateNodeTemplate(fields=fields)
    @njit(cache=True)
    def ctor(kb_meminfo):
        st = new(pnode_type)
        # st.kb = _struct_from_meminfo(KnowledgeBaseType,kb_meminfo)
        st.truth_values = np.empty((0,),dtype=np.uint8)
        st.ndim = 2
        st.left_type = left_type
        st.left_attr = left_attr
        st.op_str = op
        st.right_type = right_type
        st.right_attr = right_attr
        return st

    return ctor, pnode_type

def get_beta_predicate_node(kb, left_type, left_attr, op, right_type, right_attr):
    ctor, pnode_type = define_beta_predicate_node(left_type, left_attr, op, right_type, right_attr)    
    out = ctor(kb._meminfo)
    # out.kb = kb
    return out

# @overload_method(PredicateNodeTemplate, "update")
# def pred_update(self):
#     def impl(self):
#         pred_meminfo = _meminfo_from_struct(pred_node)
#         pred_node.update_func(pred_meminfo)






# BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

# @njit
# def njit_update(pt):
#     meminfo = _meminfo_from_struct(pt)
#     subscriber = _struct_from_meminfo(BaseSubscriberType,meminfo)
#     subscriber.update_func(meminfo)


# # from time import time_ns
# # ts = time_ns()
# # for i in range(1):
# #     PT = get_alpha_predicate_node(kb,BOOPType,"A", "<",i)
# #     njit_update(PT)

# #     PT.update_func(PT._meminfo)
#     # print(PT.left_attr)
#     # print(PT.right_val)

# kb = KnowledgeBase()

# pn = get_alpha_predicate_node(BOOPType,"B", "<",9)


# kb.add_subscriber(pn)

# kb.declare(BOOP("Q",7))
# kb.declare(BOOP("Z",11))

# njit_update(pn)


# # for i in range(1):
# #     PT = get_beta_predicate_node(kb,BOOPType,str(i)+"A", "<", BOOPType,"B")

# # print(float(time_ns()-ts)/1e6)


# @njit(cache=True)
# def dummy_grow(dummy_meminfo):
#     node = _struct_from_meminfo(BaseSubscriberType, dummy_meminfo)
#     for child_meminfo in node.children:
#         child = _struct_from_meminfo(BaseSubscriberType, child_meminfo)
#         for x in range(0,40):
#             child.grow_queue.append(x)

# @njit(cache=True)
# def dummy_change(dummy_meminfo):
#     node = _struct_from_meminfo(BaseSubscriberType, dummy_meminfo)
#     for child_meminfo in node.children:
#         child = _struct_from_meminfo(BaseSubscriberType, child_meminfo)
#         for x in range(0,40):
#             child.change_queue.append(x)


# @njit(cache=True)
# def dummy_ctor(kb_meminfo):
#     st = new(BaseSubscriberType)
#     init_base_subscriber(st)
#     # st.update_func = dumm
#     # st.change_queue = change_queue

#     return st








# exit()


# dummy_upstream = dummy_ctor(kb._meminfo)



# PT = get_alpha_predicate_node(kb,BOOPType,"B", "<",9)
# link_downstream(dummy_upstream,PT)

# dummy_grow(dummy_upstream._meminfo)
# njit_update(PT)
# dummy_change(dummy_upstream._meminfo)
# # njit_update(dummy_grow)
# # njit_update(dummy_change)


# njit_update(PT)

# print("----DONE-----")

# PT.update_func(PT._meminfo)



    # print(PT.left_attr)
    # print(PT.right_attr)
import timeit
N=100000
def time_ms(f):
    f() #warm start
    return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))

@njit
def re_lt(a,b):
    return exec_op(literally("<"),a,b)

@njit
def l():
    for i in range(100):
        lt(i,2)

@njit
def rl():
    for i in range(100):
        re_lt(i,2)

print(time_ms(l))
print(time_ms(rl))

# pt = PredicateNode(kb,
#     List.empty_list(meminfo_type),
#     List.empty_list(u8),
#     np.zeros(0,dtype=np.uint8),
#     1,
#     BOOPType,
#     "A",
#     None,





#### PLANNING PLANNING PLANNING ###

# We have Condtions they are structrefs
# there is a context.store() that keeps definitions of each predicate_node
# they are unique and every kb has (at most) exactly one predicate node of each type
# Conditions are defined with mixed alpha/beta bits 
# When the dnf is generated there are OR and AND collections that are condition nodes
# condition nodes are also unique
# 
