import numpy as np
from numba import types, njit, i8, u8, i4, u1, literally, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type, void
from numba.experimental.structref import new
from numba.extending import overload_method, intrinsic
from numbert.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from numbert.experimental.context import kb_context
from numbert.experimental.structref import define_structref, define_structref_template
from numbert.experimental.kb import KnowledgeBaseType, KnowledgeBase, facts_for_t_id, fact_at_f_id
# <<<<<<< Updated upstream
# from numbert.experimental.fact import define_fact, BaseFactType
# =======
from numbert.experimental.fact import define_fact, BaseFactType, cast_fact
# >>>>>>> Stashed changes
from numbert.experimental.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, decode_idrec, lower_getattr, _struct_from_pointer
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



base_predicate_node_field_dict = {
    "truth_values" : u1[:],
    "left_t_id" : i8,
    "op_str" : unicode_type,
    "left_attr" : types.Any,
    "left_type" : types.Any,
}

basepredicate_node_fields = [(k,v) for k,v, in base_predicate_node_field_dict.items()]
BasePredicateNode, BasePredicateNodeType = define_structref("BasePredicateNode", base_subscriber_fields + basepredicate_node_fields)

predicate_node_field_dict = {
    **base_predicate_node_field_dict,
    "signature" : types.Any,
    "right_val" : types.Any,
    # "update_func" : types.FunctionType(void(types.Any,meminfo_type))
}
predicate_node_fields = [(k,v) for k,v, in predicate_node_field_dict.items()]
PredicateNode, PredicateNodeTemplate = define_structref_template("PredicateNode", base_subscriber_fields + predicate_node_fields)

@njit(cache=True)
def init_alpha(st,t_id, op_str,literal_val):
    st.truth_values = np.empty((0,),dtype=np.uint8)
    st.left_t_id = t_id
    # st.left_attr = left_attr
    st.op_str = op_str
    st.right_val = literal_val
    

@njit(cache=True)
def alpha_eval_truth(kb,facts,f_id, pred_node):
    # fact_ptr = facts.data[i8(f_id)]
    inst_ptr = facts.data[i8(f_id)]
    # inst = fact_at_f_id(pred_node.left_type,facts,i8(f_id))
    # inst = _cast_structref(pred_node.left_type, facts[i8(f_id)])
    if(inst_ptr != 0):
        inst = _struct_from_pointer(pred_node.left_type,inst_ptr)
        val = lower_getattr(inst, pred_node.left_attr)
        return exec_op(pred_node.op_str, val, pred_node.right_val)
    else:
        return 0xFF

@njit(cache=True,locals={'new_size':u8})
def alpha_update(pred_meminfo,pnode_type):
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

    facts = facts_for_t_id(kb.kb_data,i8(pred_node.left_t_id))

    if(len(pred_node.grow_queue) > 0):
        for idrec in pred_node.grow_queue:
            t_id, f_id,_ = decode_idrec(idrec)
            truth = alpha_eval_truth(kb,facts,f_id, pred_node)
            new_truth_values[f_id] = truth

            for child_meminfo in pred_node.children:
                child = _struct_from_meminfo(BaseSubscriberType,child_meminfo)
                child.grow_queue.append(idrec)
        pred_node.grow_queue = List.empty_list(u8)
        pred_node.truth_values = new_truth_values

    if(len(pred_node.change_queue) > 0):
        for idrec in pred_node.change_queue:
            t_id, f_id,_ = decode_idrec(idrec)
            truth = alpha_eval_truth(kb,facts,f_id, pred_node)

            new_truth_values[f_id] = truth
            if(truth != pred_node.truth_values[f_id]):
                for child_meminfo in pred_node.children:
                    child = _struct_from_meminfo(BaseSubscriberType, child_meminfo)
                    child.change_queue.append(idrec)
        pred_node.change_queue = List.empty_list(u8)
        pred_node.truth_values = new_truth_values


def gen_alpha_source(typ, attr, literal_val):
    typ_name = f'{typ._fact_name}Type'
    source = f'''
from numba import types, njit
from numba.experimental.structref import new
from numba.types import *
from numbert.experimental.predicate_node import PredicateNodeTemplate, init_alpha, alpha_update, predicate_node_field_dict
from numbert.experimental.subscriber import base_subscriber_fields, init_base_subscriber
{gen_import_str(typ._fact_name,typ._hash_code,[typ_name])}

specialization_dict = {{
    'left_type' : types.TypeRef({typ_name}),
    'left_attr' : types.literal('{attr}'),
    'right_val' : {types.literal(literal_val).literal_type}
}}

field_dict = {{**predicate_node_field_dict,**specialization_dict}}
if(isinstance(field_dict["right_val"],types.Integer)): field_dict["right_val"] = types.float64
fields = base_subscriber_fields + [(k,v) for k,v, in field_dict.items()]

pnode_type = PredicateNodeTemplate(fields=fields)


@njit(cache=True)
def update_func(pred_meminfo):
    alpha_update(pred_meminfo, pnode_type)        

@njit(cache=True)
def pre_ctor(t_id,op_str,literal_val):
    st = new(pnode_type)
    init_base_subscriber(st)
    init_alpha(st,t_id,op_str, literal_val)
    st.left_type = {typ_name}
    st.left_attr = '{attr}'
    # st.update_func = update_func
    return st

@njit
def ctor(t_id,op_str,literal_val):
    st = pre_ctor(t_id,op_str,literal_val)
    st.update_func = update_func
    return st
    '''
    return source


def define_alpha_predicate_node(typ, attr, literal_val):
    name = "AlphaPredicate"
    literal_type = types.literal(literal_val).literal_type
    hash_code = unique_hash([typ._fact_name,typ._hash_code, attr, literal_type])
    if(not source_in_cache(name,hash_code)):
        source = gen_alpha_source(typ, attr, literal_val)
        source_to_cache(name, hash_code, source)
        
    ctor, pnode_type = import_from_cached(name, hash_code,['ctor','pnode_type']).values()

    return ctor, pnode_type

# @njit
# def set_update_func(pred_node,update_func):
#     pred_node.update_func = update_func

def get_alpha_predicate_node(typ, attr, op, literal_val):
    context = kb_context()    
    t_id = context.fact_to_t_id[typ._fact_name]

    ctor, pnode_type = define_alpha_predicate_node(typ, attr, literal_val)
    out = ctor(t_id, op, literal_val)

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
def re(op,a,b):
    return exec_op(literally("<"),a,b)

@njit
def l():
    for i in range(10000):
        lt(i,2)

@njit
def rl():
    for i in range(10000):
        re_lt(i,2)



@njit
def r(op):
    out = np.empty(10000,dtype=np.uint8)
    for i in range(10000):
        out[i] = re(op,i,2)
    return out

@njit
def _r(op):
    out = np.empty(10000,dtype=np.uint8)
    for i in range(10000):
        out[i] = lt(i,2)
    return out

def foor():
    r("<")

def fool():
    _r("<")

print(time_ms(l))
print(time_ms(rl))
# print(time_ms(r))
print(time_ms(foor))
print(time_ms(fool))

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

# Conditions have arguments, an op_enum, truth_values and a signature
# Conditions with a signature of one argument type are essentially alpha conditions
# So OR/AND nodes need to know what the first beta argument index is or -1 for none
# Conditions are subscribers
