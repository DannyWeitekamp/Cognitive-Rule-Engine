import numpy as np
from numba import types, njit, i8, u8, i4, u1, literally, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type, void
from numba.experimental.structref import new
from numba.extending import overload_method, intrinsic
from numbert.experimental.structref import define_structref, define_structref_template
from numbert.experimental.kb import KnowledgeBaseType, KnowledgeBase
from numbert.experimental.fact import define_fact
from numbert.experimental.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref
from numbert.experimental.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType
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




# DefferedSubscriberType.define(BaseSubscriberType)
# print(BaseSubscriberType)
print("MOO")
kb = KnowledgeBase()
# BaseSubscriber(kb,List.empty_list(meminfo_type),List.empty_list(u8))

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
print("CHOO")
PredicateNode, PredicateNodeTemplate = define_structref_template("PredicateNode", base_subscriber_fields + predicate_node_fields)
print("DHOO")


# @njit(cache=True)
# def pred_update(typ,meminfo):
#     pred_node = _struct_from_meminfo(typ,meminfo)
#     print("BLLOOP",pred_node.ndim)

# # @njit(cache=True)
# # def gen_pred_update(pred_node):
# #     pred_node_literal = literally(pred_node)
# #     def impl():
# #         print("BLLOOP",pred_node_literal.ndim)
# #     return impl


# @generated_jit
# def gen_pred_update(pred_node):
#     pred_node_type = pred_node
#     def impl(pred_node):
#         def _impl(meminfo):
#             _pred_node = _struct_from_meminfo(type(pred_node),meminfo)
#             print("BLLOOP",_pred_node.ndim)
#         return _impl
#     return impl


np_u1 = np.uint8

def define_alpha_predicate_node(typ, attr, op, literal_val):
    print(typ.__dict__)

    field_dict = copy(predicate_node_field_dict)
    field_dict["left_type"] = types.TypeRef(typ)
    field_dict["left_attr"] = types.literal(attr)
    field_dict["op_str"] = types.literal(op)
    field_dict["right_val"] = types.literal(literal_val)

    fields = base_subscriber_fields + [(k,v) for k,v, in field_dict.items()]

    pnode_type = PredicateNodeTemplate(fields=fields)



    @njit(cache=True,locals={'new_size':u8})
    def update_func(pred_meminfo):
        if(pred_meminfo is None): return
        print("---UPDATE START----")
        pred_node = _struct_from_meminfo(pnode_type, pred_meminfo)
        grw_s = pred_node.grow_queue
        chg_s = pred_node.change_queue

        new_size = max(grw_s)+1 if len(grw_s) > 0 else 0
        # if(len(pred_node.grow_queue) > max_change): max_change
        print("new_size",new_size)
        if(new_size > 0):
            new_truth_values = np.empty((new_size,),dtype=np.uint8)
            for i,b in enumerate(pred_node.truth_values):
                new_truth_values[i] = b
        else:
            new_truth_values = pred_node.truth_values

        for x in pred_node.grow_queue:
            #TODO: needs to actually be based on fact values
            truth = exec_op(op,x,literal_val)
            print("grow:",x, truth)
            new_truth_values[x] = truth
            for child_meminfo in pred_node.children:
                child = _struct_from_meminfo(BaseSubscriberType,child_meminfo)
                child.grow_queue.append(x)
        pred_node.grow_queue = List.empty_list(u8)

        pred_node.truth_values = new_truth_values

        for x in pred_node.change_queue:
            #TODO: needs to actually be based on fact values
            truth = exec_op(op,x,literal_val)
            new_truth_values[x] = truth
            print("change",x, truth, "?=", pred_node.truth_values[x], "->", truth != pred_node.truth_values[x])
            if(truth != pred_node.truth_values[x]):
                for child_meminfo in pred_node.children:
                    child = _struct_from_meminfo(BaseSubscriberType,child_meminfo)
                    child.change_queue.append(x)
        pred_node.change_queue = List.empty_list(u8)

        pred_node.truth_values = new_truth_values

        print("---UPDATE DONE----", pred_node.ndim)

    @njit(cache=True)
    def ctor(kb_meminfo):
        st = new(pnode_type)
        # def update_func(pred_meminfo):
        #     pred_node = _struct_from_meminfo(pnode_type,pred_meminfo)
        #Use meminfo to get around casting error in numba 0.51.2
        
        # st.type = pnode_type
        init_base_subscriber(st,_struct_from_meminfo(KnowledgeBaseType,kb_meminfo) )
        st.update_func = update_func

        st.truth_values = np.empty((0,),dtype=np.uint8)
        st.ndim = 1
        st.left_type = typ
        st.left_attr = attr
        st.op_str = op
        st.right_val = literal_val
        return st
    return ctor, pnode_type, update_func

# @njit
# def set_update_func(pred_node,update_func):
#     pred_node.update_func = update_func

def get_alpha_predicate_node(kb, typ, attr, op, literal_val):
    ctor, pnode_type, update_func = define_alpha_predicate_node(typ, attr, op, literal_val)

    out = ctor(kb._meminfo)
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
        st.kb = _struct_from_meminfo(KnowledgeBaseType,kb_meminfo)
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






BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

@njit
def njit_update(pt):
    meminfo = _meminfo_from_struct(pt)
    subscriber = _struct_from_meminfo(BaseSubscriberType,meminfo)
    subscriber.update_func(meminfo)


from time import time_ns
ts = time_ns()
for i in range(1):
    PT = get_alpha_predicate_node(kb,BOOPType,"A", "<",i)
    njit_update(PT)

    PT.update_func(PT._meminfo)
    # print(PT.left_attr)
    # print(PT.right_val)



for i in range(1):
    PT = get_beta_predicate_node(kb,BOOPType,str(i)+"A", "<", BOOPType,"B")

print(float(time_ns()-ts)/1e6)


@njit(cache=True)
def dummy_grow(dummy_meminfo):
    node = _struct_from_meminfo(BaseSubscriberType, dummy_meminfo)
    for child_meminfo in node.children:
        child = _struct_from_meminfo(BaseSubscriberType, child_meminfo)
        for x in range(0,40):
            child.grow_queue.append(x)

@njit(cache=True)
def dummy_change(dummy_meminfo):
    node = _struct_from_meminfo(BaseSubscriberType, dummy_meminfo)
    for child_meminfo in node.children:
        child = _struct_from_meminfo(BaseSubscriberType, child_meminfo)
        for x in range(0,40):
            child.change_queue.append(x)


@njit(cache=True)
def dummy_ctor(kb_meminfo):
    st = new(BaseSubscriberType)
    init_base_subscriber(st,_struct_from_meminfo(KnowledgeBaseType,kb_meminfo) )
    # st.update_func = dumm
    # st.change_queue = change_queue

    return st

dummy_upstream = dummy_ctor(kb._meminfo)



PT = get_alpha_predicate_node(kb,BOOPType,"A", "<",9)
link_downstream(dummy_upstream,PT)

dummy_grow(dummy_upstream._meminfo)
njit_update(PT)
dummy_change(dummy_upstream._meminfo)
# njit_update(dummy_grow)
# njit_update(dummy_change)


njit_update(PT)

print("----DONE-----")

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
