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
    # return a < b
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

@njit(cache=True)
def resolve_predicate_op(op_str):
    if(op_str == "<"):
        return lt
    elif(op_str == "<="):
        return lte
    elif(op_str == ">"):
        return gt
    elif(op_str == ">="):
        return gte
    elif(op_str == "=="):
        return eq

def resolve_predicate_op(op):
    if(isinstance(op,str)):
        return op_str_map[op]
    return op

#### Struct Definitions ####

base_predicate_node_field_dict = {
    "left_t_id" : i8,
    "left_attr" : types.Any,
    "left_type" : types.Any,
    "op_str" : unicode_type,
    # "op_func" : types.FunctionType(u1(types.Any,types.Any)),
}

basepredicate_node_fields = [(k,v) for k,v, in base_predicate_node_field_dict.items()]
BasePredicateNode, BasePredicateNodeType = define_structref("BasePredicateNode", base_subscriber_fields + basepredicate_node_fields)


alpha_predicate_node_field_dict = {
    **base_predicate_node_field_dict,
    "truth_values" : u1[:],
    "signature" : types.Any, #TOOO: Maybe don't need this 
    "right_val" : types.Any,
    # "update_func" : types.FunctionType(void(types.Any,meminfo_type))
}
alpha_predicate_node_fields = [(k,v) for k,v, in alpha_predicate_node_field_dict.items()]
AlphaPredicateNode, AlphaPredicateNodeTemplate = define_structref_template("AlphaPredicateNode", base_subscriber_fields + alpha_predicate_node_fields)


beta_predicate_node_field_dict = {
    **base_predicate_node_field_dict,
    "truth_values" : u1[:,:],
    "right_t_id" : i8,
    "right_attr" : types.Any,
    "right_type" : types.Any,
    "left_consistency" : u1[:],
    "right_consistency" : u1[:],
    # "update_func" : types.FunctionType(void(types.Any,meminfo_type))
}
beta_predicate_node_fields = [(k,v) for k,v, in beta_predicate_node_field_dict.items()]
BetaPredicateNode, BetaPredicateNodeTemplate = define_structref_template("BetaPredicateNode", base_subscriber_fields + beta_predicate_node_fields)


#### Alpha Predicate Nodes #####

@njit(cache=True)
def init_alpha(st,t_id,literal_val):
    st.truth_values = np.empty((0,),dtype=np.uint8)
    st.left_t_id = t_id
    # st.left_attr = left_attr
    # st.op_str = op_str
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
        # return pred_node.op_func(val, pred_node.right_val)#exec_op(pred_node.op_str, val, pred_node.right_val)
        return exec_op(pred_node.op_str, val, pred_node.right_val)
    else:
        return 0xFF

@njit(cache=True)
def expand_1d(truth_values,size,dtype):
    new_truth_values = np.empty((size,),dtype=dtype)
    for i,b in enumerate(truth_values):
        new_truth_values[i] = b
    return new_truth_values

@njit(cache=True)
def expand_2d(truth_values, n, m, dtype):
    new_truth_values = np.empty((n,m),dtype=dtype)
    for i in range(truth_values.shape[0]):
        for j in range(truth_values.shape[1]):
            new_truth_values[i,j] = truth_values[i,j]
    return new_truth_values


@njit(cache=True,locals={'new_size':u8})
def alpha_update(pred_meminfo,pnode_type):
    if(pred_meminfo is None): return
    pred_node = _struct_from_meminfo(pnode_type, pred_meminfo)
    kb = _struct_from_meminfo(KnowledgeBaseType, pred_node.kb_meminfo)
    grw_q = kb.kb_data.grow_queue
    chg_q = kb.kb_data.change_queue

    facts = facts_for_t_id(kb.kb_data,i8(pred_node.left_t_id))
    if(len(facts.data) > len(pred_node.truth_values)):
        pred_node.truth_values = expand_1d(pred_node.truth_values,len(facts.data),np.uint8)

    for i in range(pred_node.grow_head, grw_q.head):
        t_id, f_id, a_id = decode_idrec(grw_q[i])
        if(pred_node.left_t_id == t_id):
            truth = alpha_eval_truth(kb,facts,f_id, pred_node)
            pred_node.truth_values[f_id] = truth
            # pred_node.grow_queue.add(f_id)
    pred_node.grow_head = grw_q.head

    for i in range(pred_node.change_head, chg_q.head):
        t_id, f_id, a_id = decode_idrec(chg_q[i])
        if(pred_node.left_t_id == t_id):
            truth = alpha_eval_truth(kb,facts,f_id, pred_node)
            pred_node.truth_values[f_id] = truth
            # if(truth != pred_node.truth_values[f_id]):
            #     pred_node.change_queue.add(f_id)
    pred_node.change_head = chg_q.head


def gen_alpha_source(typ, attr, op_str, literal_val):
    typ_name = f'{typ._fact_name}Type'
    literal_type = types.literal(literal_val).literal_type
    if(isinstance(literal_type,types.Integer)): literal_type = types.float64
    fieldtype = typ.field_dict[attr]
    source = f'''
from numba import types, njit
from numba.experimental.structref import new
from numba.types import *
from numbert.experimental.predicate_node import AlphaPredicateNodeTemplate, init_alpha, alpha_update, alpha_predicate_node_field_dict, resolve_predicate_op
from numbert.experimental.subscriber import base_subscriber_fields, init_base_subscriber
{gen_import_str(typ._fact_name,typ._hash_code,[typ_name])}

specialization_dict = {{
    # 'op_func' : types.FunctionType(u1({fieldtype},{literal_type})),
    'op_str' : types.literal('{op_str}'),#types.FunctionType(u1({fieldtype},{literal_type})),
    'left_type' : types.TypeRef({typ_name}),
    'left_attr' : types.literal('{attr}'),
    'right_val' : {literal_type}
}}

field_dict = {{**alpha_predicate_node_field_dict,**specialization_dict}}
if(isinstance(field_dict["right_val"],types.Integer)): field_dict["right_val"] = types.float64
fields = base_subscriber_fields + [(k,v) for k,v, in field_dict.items()]

pnode_type = AlphaPredicateNodeTemplate(fields=fields)


@njit(cache=True)
def update_func(pred_meminfo):
    alpha_update(pred_meminfo, pnode_type)        

@njit(cache=True)
def pre_ctor(t_id,literal_val):
    st = new(pnode_type)
    init_base_subscriber(st)
    init_alpha(st,t_id, literal_val)
    st.left_type = {typ_name}
    st.left_attr = '{attr}'
    return st

@njit
def ctor(t_id,literal_val):
    st = pre_ctor(t_id,literal_val)
    st.update_func = update_func
    return st
    '''
    return source


def define_alpha_predicate_node(typ, attr, op_str, literal_val):
    name = "AlphaPredicate"
    literal_type = types.literal(literal_val).literal_type
    hash_code = unique_hash([typ._fact_name,typ._hash_code, attr, op_str, literal_type])
    if(not source_in_cache(name,hash_code)):
        source = gen_alpha_source(typ, attr, op_str, literal_val)
        source_to_cache(name, hash_code, source)
        
    ctor, pnode_type = import_from_cached(name, hash_code,['ctor','pnode_type']).values()

    return ctor, pnode_type

def get_alpha_predicate_node(typ, attr, op_str, literal_val):
    context = kb_context()    
    t_id = context.fact_to_t_id[typ._fact_name]

    ctor, pnode_type = define_alpha_predicate_node(typ, attr, op_str, literal_val)
    out = ctor(t_id, literal_val)

    return out


#### Beta Predicate Nodes ####

@njit(cache=True)
def init_beta(st, left_t_id, right_t_id):
    st.truth_values = np.empty((0,0),dtype=np.uint8)
    st.left_t_id = left_t_id
    st.right_t_id = right_t_id
    # st.op_str = '<'
    

@njit(cache=True)
def beta_eval_truth(kb,pred_node, left_facts, right_facts, i, j):
    left_ptr = left_facts.data[i]
    right_ptr = right_facts.data[j]

    if(left_ptr != 0 and right_ptr != 0):
        left_inst = _struct_from_pointer(pred_node.left_type,left_ptr)
        right_inst = _struct_from_pointer(pred_node.right_type,right_ptr)
        left_val = lower_getattr(left_inst, pred_node.left_attr)
        right_val = lower_getattr(right_inst, pred_node.right_attr)
        # return pred_node.op_func(left_val, right_val)
        return exec_op(pred_node.op_str, left_val, right_val)
    else:
        return 0#0xFF


@njit(cache=True,inline='always')
def update_pair(kb,pred_node, left_facts, right_facts, i, j):
    truth = beta_eval_truth(kb,pred_node, left_facts, right_facts, i, j)
    pred_node.truth_values[i,j] = truth



@njit(cache=True,locals={'new_size':u8})
def beta_update(pred_meminfo,pnode_type):
    if(pred_meminfo is None): return
    pred_node = _struct_from_meminfo(pnode_type, pred_meminfo)
    kb = _struct_from_meminfo(KnowledgeBaseType, pred_node.kb_meminfo)
    grw_q = kb.kb_data.grow_queue
    chg_q = kb.kb_data.change_queue

    left_facts = facts_for_t_id(kb.kb_data,i8(pred_node.left_t_id))
    right_facts = facts_for_t_id(kb.kb_data,i8(pred_node.right_t_id))


    if(len(left_facts.data) > pred_node.truth_values.shape[0] or
       len(right_facts.data) > pred_node.truth_values.shape[1]):
        pred_node.truth_values = expand_2d(pred_node.truth_values,
                                    len(left_facts.data),len(right_facts.data),np.uint8
                                 )
    if(len(left_facts.data) > len(pred_node.left_consistency)):
        pred_node.left_consistency = expand_1d(pred_node.left_consistency,
                                        len(left_facts.data),np.uint8)
    if(len(right_facts.data) > len(pred_node.right_consistency)):
        pred_node.right_consistency = expand_1d(pred_node.right_consistency,
                                        len(right_facts.data),np.uint8)

    for i in range(pred_node.grow_head, grw_q.head):
        t_id, f_id, a_id = decode_idrec(grw_q[i])
        if(pred_node.left_t_id == t_id):
            pred_node.left_consistency[f_id] = 0
        if(pred_node.right_t_id == t_id):
            pred_node.right_consistency[f_id] = 0
    pred_node.grow_head = grw_q.head

    for i in range(pred_node.change_head, chg_q.head):
        t_id, f_id, a_id = decode_idrec(chg_q[i])
        if(pred_node.left_t_id == t_id):
            pred_node.left_consistency[f_id] = 0
        if(pred_node.right_t_id == t_id):
            pred_node.right_consistency[f_id] = 0
    pred_node.change_head = chg_q.head


    #NOTE: This part might be sped up by running it only on request

    lc, rc = pred_node.left_consistency, pred_node.right_consistency
    # print(lc[:6], rc[:6])
    for i in range(left_facts.head):
        if(not lc[i]):
            for j in range(right_facts.head):
                update_pair(kb,pred_node,left_facts,right_facts,i,j)
    

    for j in range(right_facts.head):
        if(not rc[j]):
            for i in range(left_facts.head):
                if(lc[i]): update_pair(kb,pred_node,left_facts,right_facts,i,j)
    pred_node.left_consistency[:len(left_facts)] = 1
    pred_node.right_consistency[:len(right_facts)] = 1
                    

    # print("UPDATE OK")






def gen_beta_source(left_type, left_attr, op_str, right_type, right_attr):
    left_typ_name = f'{left_type._fact_name}Type'
    right_typ_name = f'{right_type._fact_name}Type'
    left_fieldtype = left_type.field_dict[left_attr]
    right_fieldtype = right_type.field_dict[right_attr]
    source = f'''
from numba import types, njit
from numba.experimental.structref import new
from numba.types import *
from numbert.experimental.predicate_node import BetaPredicateNodeTemplate, init_beta, beta_update, beta_predicate_node_field_dict
from numbert.experimental.subscriber import base_subscriber_fields, init_base_subscriber
{gen_import_str(left_type._fact_name,left_type._hash_code,[left_typ_name])}
{gen_import_str(right_type._fact_name,right_type._hash_code,[right_typ_name])}

specialization_dict = {{
    # 'op_func' : types.FunctionType(u1({left_fieldtype},{right_fieldtype})),
    'op_str' : types.literal('{op_str}'),
    'left_type' : types.TypeRef({left_typ_name}),
    'left_attr' : types.literal('{left_attr}'),
    'right_type' : types.TypeRef({right_typ_name}),
    'right_attr' : types.literal('{right_attr}'),
}}

field_dict = {{**beta_predicate_node_field_dict,**specialization_dict}}
fields = base_subscriber_fields + [(k,v) for k,v, in field_dict.items()]

pnode_type = BetaPredicateNodeTemplate(fields=fields)


@njit(cache=True)
def update_func(pred_meminfo):
    beta_update(pred_meminfo, pnode_type)        

@njit(cache=True)
def pre_ctor(left_t_id, right_t_id):
    st = new(pnode_type)
    init_base_subscriber(st)
    init_beta(st, left_t_id, right_t_id)
    st.left_type = {left_typ_name}
    st.left_attr = '{left_attr}'
    st.right_type = {right_typ_name}
    st.right_attr = '{right_attr}'
    return st

@njit
def ctor(*args):
    st = pre_ctor(*args)
    st.update_func = update_func
    return st
    '''
    return source


def define_beta_predicate_node(left_type, left_attr, op_str, right_type, right_attr):
    name = "BetaPredicate"
    hash_code = unique_hash([left_type._fact_name, left_type._hash_code, left_attr, op_str,
                             right_type._fact_name, right_type._hash_code, right_attr])
    if(not source_in_cache(name,hash_code)):
        source = gen_beta_source(left_type, left_attr, op_str, right_type, right_attr)
        source_to_cache(name, hash_code, source)
        
    ctor, pnode_type = import_from_cached(name, hash_code,['ctor','pnode_type']).values()

    return ctor, pnode_type

# @njit
# def set_update_func(pred_node,update_func):
#     pred_node.update_func = update_func

def get_beta_predicate_node(left_type, left_attr, op_str, right_type, right_attr):
    context = kb_context()    
    left_t_id = context.fact_to_t_id[left_type._fact_name]
    right_t_id = context.fact_to_t_id[right_type._fact_name]

    ctor, pnode_type = define_beta_predicate_node(left_type, left_attr, op_str, right_type, right_attr)
    print("PP", pnode_type)

    out = ctor(left_t_id, right_t_id)

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
