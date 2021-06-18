'''Utilities for defining alpha/beta predicate nodes

These subscriber nodes are roughly analagous to the alpha and beta nodes of the 
RETE matching algorithm, with a few important differences, including the fact
that there isn't a precompiled RETE graph, and the loose graph structure that does
exist does not pass around tuples. PredicateNodes are mostly independant of each other
and can be rearranged and resused in ConditionNodes as needed.

Alpha predicates correspond to statements of the form `fact.attribute < literal_value`, 
and Beta predicates correspond to satements of the form `fact1.attribute1 < fact2.attribute2`
where `<` can be any comparison operator. These nodes subscribe to changes
in a KnowledgeBase. When their update_func() is called each predicate nodes' internal 
truth_values are updated for each fact or pair of facts of the particular type(s) evaluated 
on the nodes' particular comparison statement.
'''

import numpy as np
from numba import types, njit, i8, u8, i4, u1, u4, literally, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type, void
from numba.core.types.misc import unliteral
from numba.experimental.structref import new
from numba.extending import overload_method, intrinsic, overload
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from cre.context import kb_context
from cre.structref import define_structref, define_structref_template
from cre.kb import KnowledgeBaseType, KnowledgeBase, facts_for_t_id, fact_at_f_id
from cre.fact import define_fact, BaseFactType, cast_fact
from cre.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, \
 decode_idrec, lower_getattr, _struct_from_pointer, struct_get_attr_offset, _struct_get_data_pointer, \
 _load_pointer, _pointer_to_data_pointer, _list_base_from_ptr
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType, new_vector
from cre.utils import deref_type, OFFSET_TYPE_ATTR, OFFSET_TYPE_LIST
from copy import copy
from operator import itemgetter

meminfo_type = types.MemInfoPointer(types.voidptr)



# @generated_jit(cache=True)
# def exec_op(op_str,a,b):
#     if(op_str.literal_value == "<"):
#         return lambda a,b : a < b
#     elif(op_str.literal_value == "<="):
#         return lambda a,b : a <= b
#     elif(op_str.literal_value == ">"):
#         return lambda a,b : b > a
#     elif(op_str.literal_value == ">="):
#         return lambda a,b : a >= b
#     elif(op_str.literal_value == "=="):
#         return lambda a,b : a == b
#     raise ValueError(f"Unrecognized op_str {op_str} {op_str.literal_value}")

# LT = types.literal("<")
# GT = types.literal("<")
@njit(cache=True)
def exec_op(op_str,a,b):
    '''Executes one of 5 boolean comparison operations. Since the predicate_node type
       fixes 'op_str' to a literal value. LLVM will compile out the switch case.
    '''
    # print("op_str", op_str, op_str == "<", op_str == ">")
    if(op_str == "<"):
        return a < b
    elif(op_str == "<="):
        return a <= b
    elif(op_str == ">"):
        return a > b
    elif(op_str == ">="):
        return a >= b
    elif(op_str == "=="):
        # print("equal", a,b, a==b)
        return a == b
    # GT, op_str == GT)
    raise ValueError("Unrecognized op_str.")

#### Link Data ####

predicate_node_link_data_field_dict = {
    "left_t_id" : u8,
    "right_t_id" : u8,
    "left_facts" : VectorType, #Vector<*Fact>
    "right_facts" : VectorType, #Vector<*Fact>
    
    "change_head": i8,
    "grow_head": i8,
    "change_queue": VectorType,
    "grow_queue": VectorType,
    "kb_grow_queue" : VectorType,
    "kb_change_queue" : VectorType,



    "truth_values" : u1[:,:],
    "left_consistency" : u1[:],
    "right_consistency" : u1[:],
}

predicate_node_link_data_fields = [(k,v) for k,v, in predicate_node_link_data_field_dict.items()]
PredicateNodeLinkData, PredicateNodeLinkDataType = define_structref("PredicateNodeLinkData", 
                predicate_node_link_data_fields, define_constructor=False)



@njit(cache=True)
def generate_link_data(pn, kb):
    '''Takes a prototype predicate node and a knowledge base and returns
        a link_data instance for that predicate node.
    '''
    link_data = new(PredicateNodeLinkDataType)
    # print(pn.left_fact_type_name)
    # print(kb.context_data.fact_to_t_id)
    # print(pn.left_fact_type_name)
    # print("Q")
    link_data.left_t_id = kb.context_data.fact_to_t_id[pn.left_fact_type_name]
    # print("Q2", link_data.left_t_id)
    link_data.left_facts = facts_for_t_id(kb.kb_data,i8(link_data.left_t_id)) 
    # print("Z")
    if(not pn.is_alpha):
        link_data.right_t_id = kb.context_data.fact_to_t_id[pn.right_fact_type_name]
        link_data.right_facts = facts_for_t_id(kb.kb_data,i8(link_data.right_t_id)) 
        link_data.left_consistency = np.empty((0,),dtype=np.uint8)
        link_data.right_consistency = np.empty((0,),dtype=np.uint8)
    else:
        link_data.right_t_id = -1

    # print("S")

    link_data.change_head = 0
    link_data.grow_head = 0
    link_data.change_queue = new_vector(8)
    link_data.grow_queue = new_vector(8)

    link_data.kb_grow_queue = kb.kb_data.grow_queue
    link_data.kb_change_queue = kb.kb_data.change_queue
    link_data.truth_values = np.empty((0,0),dtype=np.uint8)
        
    # print("DONE")

    # print(pn.is_alpha)
    # if(pn.is_alpha):
    #     a = _cast_structref(GenericAlphaPredicateNodeType, pn)
    #     new_a = new(GenericAlphaPredicateNodeType)
    #     new_a.filter_func = a.filter_func
    #     new_a.right_val = a.right_val

    #     new_pn = _cast_structref(BasePredicateNodeType, new_a)
    # else:
    #     b = _cast_structref(GenericBetaPredicateNodeType, pn)
    #     new_b = new(GenericBetaPredicateNodeType)
    #     new_b.filter_func = b.filter_func
    #     new_b.right_t_id = b.right_t_id
    #     new_b.right_facts = b.right_facts
        
    #     new_pn = _cast_structref(BasePredicateNodeType, new_b)

    
    return link_data


meminfo_type = types.MemInfoPointer(types.voidptr)
alpha_filter_func_type = types.FunctionType(i8[::1](meminfo_type, PredicateNodeLinkDataType, i8[::1], u1))
beta_filter_func_type = types.FunctionType(i8[:,::1](meminfo_type, PredicateNodeLinkDataType, i8[::1], i8[::1], u1))


#### Struct Definitions ####

base_predicate_node_field_dict = {
    #### Attributes filled in at definition time ###
    "id_str" : unicode_type,
    "is_alpha" : u1,
    
    "left_fact_type_name" : unicode_type,
    "right_fact_type_name" : unicode_type,

    
    "left_attr_offsets" : deref_type[::1],#types.Any,
    # "filter_func" : filter_func_type,
    "op_str" : types.literal('=='),
    
    
    

    # #### Attributes filled in at link time ###
    
    # "left_t_id" : u8,
    # "left_facts" : VectorType, #Vector<*Fact>
    # "truth_values" : u1[:,:],
    # "kb_grow_queue" : VectorType,
    # "kb_change_queue" : VectorType,
    
}

from pprint import pprint

basepredicate_node_fields = [(k,v) for k,v, in base_predicate_node_field_dict.items()]
# pprint(basepredicate_node_fields)
BasePredicateNode, BasePredicateNodeType = define_structref("BasePredicateNode", base_subscriber_fields + basepredicate_node_fields)


alpha_predicate_node_field_dict = {
    **base_predicate_node_field_dict,
    # "truth_values" : u1[:],

    #### Attributes filled in at definition time ###
    "filter_func" : alpha_filter_func_type,
    "left_type" : types.TypeRef(BaseFactType), #<- Filled in at definition
    "right_val" : types.float64, #<- Can be specialized to something else
    
}
alpha_predicate_node_fields = [(k,v) for k,v, in alpha_predicate_node_field_dict.items()]
# pprint(alpha_predicate_node_fields)
AlphaPredicateNode, AlphaPredicateNodeTemplate = define_structref_template("AlphaPredicateNode",
             base_subscriber_fields + alpha_predicate_node_fields, define_constructor=False)

GenericAlphaPredicateNodeType = AlphaPredicateNodeTemplate(base_subscriber_fields + alpha_predicate_node_fields)
# print(GenericAlphaPredicateNodeType)

beta_predicate_node_field_dict = {
    **base_predicate_node_field_dict,
    

    #### Attributes filled in at definition time ###
    "filter_func" : beta_filter_func_type,
    "right_attr_offsets" : deref_type[::1],
    "left_type" : types.TypeRef(BaseFactType), #<- Filled in at definition
    "right_type" : types.TypeRef(BaseFactType),
    
    
    #### Attributes filled in at link time ###

    
    # "right_t_id" : u8,
    # "right_facts" : VectorType, #Vector<*Fact>
    # "left_consistency" : u1[:],
    # "right_consistency" : u1[:],
}
beta_predicate_node_fields = [(k,v) for k,v, in beta_predicate_node_field_dict.items()]
BetaPredicateNode, BetaPredicateNodeTemplate = define_structref_template("BetaPredicateNode", 
                base_subscriber_fields + beta_predicate_node_fields, define_constructor=False)

GenericBetaPredicateNodeType = AlphaPredicateNodeTemplate(base_subscriber_fields + beta_predicate_node_fields)






#### Helper Array Expansion Functions ####

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


#### Alpha Predicate Nodes #####

@njit(cache=True,locals={'left_fact_type_name': unicode_type})
def init_alpha(st, left_fact_type_name, left_attr_offsets, right_val):
    '''Initializes an empty AlphaPredicateNode with t_id and right_val'''
    st.is_alpha = 1
    # st.truth_values = np.empty((0,0),dtype=np.uint8)
    # st.left_t_id = -1
    st.left_fact_type_name = left_fact_type_name
    st.left_attr_offsets = left_attr_offsets
    st.right_val = right_val
    

# @njit(cache=True)

@njit(cache=True,locals={"data_ptr":i8, "inst_ptr":i8})
def _deref_attrs(val_type, inst_ptr, attr_offsets):
    '''Helper function for deref_attrs'''

    for deref in attr_offsets[:-1]:
        if(inst_ptr == 0): raise Exception()
        if(deref.type == u1(OFFSET_TYPE_ATTR)):
            data_ptr = _pointer_to_data_pointer(inst_ptr)
        else:
            data_ptr = _list_base_from_ptr(inst_ptr)
        inst_ptr = _load_pointer(i8,data_ptr+deref.offset)
        
    if(inst_ptr == 0): raise Exception()
    deref = attr_offsets[-1]
    if(deref.type == u1(OFFSET_TYPE_ATTR)):
        data_ptr = _pointer_to_data_pointer(inst_ptr)
    else:
        data_ptr = i8(_list_base_from_ptr(inst_ptr))
    val = _load_pointer(val_type, data_ptr+deref.offset)

    return val

@generated_jit(cache=True)
def deref_attrs(val_type, inst_ptr, attr_offsets):
    if(val_type.instance_type == types.int64):
        # If val_type is an int64 then it might be a fact reference
        #  in which case we might be evaluating the fact itself
        #  not a memeber of the facts.
        def impl(val_type, inst_ptr, attr_offsets):
            if(len(attr_offsets) == 0): return inst_ptr
            return _deref_attrs(val_type, inst_ptr, attr_offsets)
    else:
        # If val_type is not an int64 then we don't need to check. 
        #  If we did consider returning inst_ptr then the return type
        #   would be ambiguous.
        def impl(val_type, inst_ptr, attr_offsets):
            return _deref_attrs(val_type, inst_ptr, attr_offsets)
    return impl




# def deref_attrs(val_type, inst_ptr,attr_offsets):
#     #TODO: Allow to deref arbitrary number of attributes
#     # print("attr_offsets", attr_offsets)
#     if(len(attr_offsets) == 0): return inst_ptr
#     data_ptr = _pointer_to_data_pointer(inst_ptr)
#     val = _load_pointer(val_type, data_ptr+attr_offsets[0])
#     return val
    


@njit(cache=True)
def alpha_eval_truth(facts, f_id, pred_node):
    '''Updates an AlphaPredicateNode with fact at f_id'''
    inst_ptr = facts.data[i8(f_id)]
    if(inst_ptr != 0):
        try:
            val = deref_attrs(pred_node.left_type, inst_ptr, pred_node.left_attr_offsets)
        except Exception:
            return 0xFF
        return exec_op(pred_node.op_str, val, pred_node.right_val)
    else:
        return 0xFF

@njit(cache=True,locals={'new_size':u8})
def alpha_filter(pnode_type, pred_meminfo, link_data, inds, negated):
    '''Implements update_func of AlphaPredicateNode subscriber'''

    # return inds
    # if(pred_meminfo is None): return
    # return inds
    # print("A")
    # if(pred_meminfo is None): return

    # Resolve this instance of the AlphaPredicateNode, it's KnowledgeBase, and 
    #   the fact pointer vector associated with this AlphaPredicateNode's t_id
    pred_node = _struct_from_meminfo(pnode_type, pred_meminfo)
    # kb = _struct_from_meminfo(KnowledgeBaseType, pred_node.kb_meminfo)
    grw_q = link_data.kb_grow_queue
    chg_q = link_data.kb_change_queue
    facts = link_data.left_facts#facts_for_t_id(kb.kb_data,i8(pred_node.left_t_id))
    # print("B")
    # Ensure that truth_values is the size of the fact pointer vector
    if(len(facts.data) > len(link_data.truth_values)):
        link_data.truth_values = expand_2d(link_data.truth_values,len(facts.data),1,np.uint8)
    # print("C")
    # Update from the grow head to the KnowledgeBase's grow head  
    for i in range(pred_node.grow_head, grw_q.head):
        t_id, f_id, a_id = decode_idrec(grw_q[i])
        if(link_data.left_t_id == t_id):
            truth = alpha_eval_truth(facts,f_id, pred_node)
            link_data.truth_values[f_id,0] = truth
            # pred_node.grow_queue.add(f_id)
    link_data.grow_head = grw_q.head
    # print("D")
    # Update from the change head to the KnowledgeBase's change head
    for i in range(pred_node.change_head, chg_q.head):
        t_id, f_id, a_id = decode_idrec(chg_q[i])
        if(link_data.left_t_id == t_id):
            truth = alpha_eval_truth(facts,f_id, pred_node)
            link_data.truth_values[f_id,0] = truth
            # if(truth != pred_node.truth_values[f_id]):
            #     pred_node.change_queue.add(f_id)
    link_data.change_head = chg_q.head

    # print(link_data.truth_values)

    new_inds = np.empty(len(inds),dtype=np.int64)
    n = 0
    for ind in inds:
        if((link_data.truth_values[ind,0] == (1 ^ negated)) ):
            new_inds[n] = ind
            n += 1

    # print(new_inds)
    return new_inds[:n]

@overload(AlphaPredicateNode, prefer_literal=True)
def alpha_ctor(left_fact_type, left_type, l_offsets, op_str, right_val):
    if(not isinstance(op_str, types.Literal)): return 
    if(not isinstance(l_offsets, types.Array)):
        raise ValueError(f"AlphaPredicateNode left_attr_offsets must be array, got {left_attr_offsets}.")

    # print("---------")
    # print(left_fact_type)
    # print(left_type) 
    # print(l_offsets)
    # print(op_str)
    # print(right_val)
    if(hasattr(right_val,"_fact_name")): raise ValueError("Alpha instantiated, but is a Beta")
    ctor, pnode_type = define_alpha_predicate_node(left_type.instance_type, op_str, right_val)
    left_fact_type_name = left_fact_type.instance_type._fact_name
    # left_fact_type_name = np.array([left_fact_type_name],dtype="U")
    # print(left_fact_type_name)
    # @njit(unicode_type())
    # def get_name():
    #     return unicode_type(left_fact_type_name)



    # left_fact_type_name._literal_type_cache = unicode_type
    # specialization_dict = {
    #     'op_str' : types.literal(op_str.literal_value),
    #     'left_type' : left_type,
    #     'right_val' : right_val
    # }

    # d = {**alpha_predicate_node_field_dict,**specialization_dict}
    # pnode_type = AlphaPredicateNodeTemplate(base_subscriber_fields+[(k,v) for k,v, in d.items()])
    # print("!!!!",struct_type)
    def impl(left_fact_type, left_type, l_offsets, op_str, right_val):
        # unq = get_name()
        # unq = 'floop'+"1"
        return ctor(str(left_fact_type_name), l_offsets, right_val)
        # st = new(pnode_type)
        # init_base_subscriber(st)
        # init_alpha(st, unq, l_offsets, right_val)
        # return st

    return impl



def gen_alpha_source(left_type, op_str, right_type):
    '''Produces source code for an AlphaPredicateNode that is specialized for the given types,
        attribute and comparison operator.'''
    # typ_name = f'{typ._fact_name}Type'
    # literal_type = types.literal(literal_val).literal_type
    if(isinstance(right_type,types.Integer)): right_type = types.float64
    
    # fieldtype = typ.field_dict[attr]
    source = f'''import numba
from numba import types, njit
from numba.experimental.structref import new
from numba.types import int64, float64, unicode_type
from cre.predicate_node import AlphaPredicateNodeTemplate, init_alpha, alpha_filter, alpha_predicate_node_field_dict
from cre.subscriber import base_subscriber_fields, init_base_subscriber


specialization_dict = {{
    'op_str' : types.literal('{op_str}'),
    'left_type' : types.TypeRef({left_type}),
    'right_val' : {right_type}
}}

field_dict = {{**alpha_predicate_node_field_dict,**specialization_dict}}
if(isinstance(field_dict["right_val"],types.Integer)): field_dict["right_val"] = types.float64
fields = base_subscriber_fields + [(k,v) for k,v, in field_dict.items()]

pnode_type = AlphaPredicateNodeTemplate(fields=fields)


@njit(cache=True)
def filter_func(pred_meminfo, link_data, inds, negated):
    return alpha_filter(pnode_type, pred_meminfo, link_data,  inds, negated)


@njit(cache=True)
def pre_ctor(left_fact_type_name, attr_offsets, literal_val):
    st = new(pnode_type)
    init_base_subscriber(st)
    init_alpha(st, left_fact_type_name, attr_offsets, literal_val)
    return st

@njit
def ctor(left_fact_type_name, attr_offsets, literal_val):
    st = pre_ctor(left_fact_type_name, attr_offsets, literal_val)
    st.filter_func = filter_func
    return st
    '''
    return source


def resolve_deref(typ,attr_chain):
    #NOTE: Does this acutally ever get used by the end user?
    print(type(deref_type.dtype),deref_type.dtype)
    offsets = np.empty((len(attr_chain),),dtype=deref_type.dtype)
    print(offsets)
    out_type = typ 
    for i, attr in enumerate(attr_chain):
        if(not hasattr(out_type,'field_dict')): 
            attr_chain_str = ".".join(attr_chain)
            raise AttributeError(f"Invalid dereference {typ}.{attr_chain_str}. {out_type} has no attribute '{attr}'.")
        fd = out_type.field_dict
        offsets[i][0] = OFFSET_TYPE_ATTR
        offsets[i][1] = out_type._attr_offsets[list(fd.keys()).index(attr)]  #struct_get_attr_offset(out_type,attr) #For some reason ~4.6ms
        out_type = fd[attr]

    return out_type, offsets

def define_alpha_predicate_node(left_type, op_str, right_type):
    '''Generates or gets the cached definition for an AlphaPredicateNode with the given 
        types, attributes, and comparison op. '''
    name = "AlphaPredicate"
    if(isinstance(left_type,types.StructRef)): 
        left_type = types.int64
        # Standardize so that it checks for null ptr instead of None
        if(right_type is None): right_type = types.int64
    hash_code = unique_hash([left_type, op_str, right_type])
    if(not source_in_cache(name,hash_code)):
        source = gen_alpha_source(left_type, op_str, right_type)
        source_to_cache(name, hash_code, source)
        
    ctor, pnode_type = import_from_cached(name, hash_code,['ctor','pnode_type']).values()

    return ctor, pnode_type

def get_alpha_predicate_node_definition(typ, attr_chain, op_str, right_type):
    '''Gets various definitions for an AlphaPredicateNode, returns a dict with 'ctor'
         'pnode_type', 'left_type', 'left_attr_offsets', 't_id' '''
    # context = kb_context()    
    # t_id = context.fact_to_t_id[typ._fact_name]

    if(not isinstance(attr_chain,list)): attr_chain = [attr_chain]
    
    left_type, left_attr_offsets = resolve_deref(typ, attr_chain)
    ctor, pnode_type = define_alpha_predicate_node(left_type, op_str, right_type)

    return locals()
    

def get_alpha_predicate_node(typ, attr_chain, op_str, literal_val):
    '''Gets a new instance of an AlphaPredicateNode that evals op(typ.attr, literal_val) '''
    right_type = types.literal(literal_val).literal_type

    dfn = get_alpha_predicate_node_definition(typ, attr_chain, op_str, right_type)
    ctor, left_attr_offsets = itemgetter('ctor', 'left_attr_offsets')(dfn)

    out = ctor(typ._fact_name, left_attr_offsets, literal_val)

    return out

# @overload_method(AlphaPredicateNodeTemplate, 'filter')
# def _impl_alpha_filter(self, inds):
#     def impl(self, inds):
#         return self.filter_func(_meminfo_from_struct(self),inds)
#     return impl


#### Beta Predicate Nodes ####

@njit(cache=True, locals={'left_fact_type_name': unicode_type, 'right_fact_type_name': unicode_type})
def init_beta(st, left_fact_type_name, left_attr_offsets, right_fact_type_name, right_attr_offsets):
    '''Initializes an empty BetaPredicateNode with left_t_id and right_t_id'''
    st.is_alpha = 0
    # st.truth_values = np.empty((0,0),dtype=np.uint8)
    # st.left_t_id = -1
    st.left_fact_type_name = left_fact_type_name
    st.left_attr_offsets = left_attr_offsets
    # st.right_t_id = -1
    st.right_fact_type_name = right_fact_type_name
    st.right_attr_offsets = right_attr_offsets
    # print(">>",st.left_fact_type_name,st.left_attr_offsets,st.right_fact_type_name, st.right_attr_offsets)
    

@njit(cache=True)
def beta_eval_truth(pred_node, left_facts, right_facts, i, j):
    '''Eval truth for BetaPredicateNode for facts i and j of left_facts and right_facts'''
    left_ptr = left_facts.data[i]
    right_ptr = right_facts.data[j]

    if(left_ptr != 0 and right_ptr != 0):
        try:
            left_val = deref_attrs(pred_node.left_type, left_ptr, pred_node.left_attr_offsets)
            right_val = deref_attrs(pred_node.right_type, right_ptr, pred_node.right_attr_offsets)
        except Exception:
            return 0xFF

        # If either dereference chain failed then return error byte 0xFF
        # if(left_val is None or right_val == None): return 0xFF
        # print(left_val, right_val)
        # left_inst = _struct_from_pointer(pred_node.left_type,left_ptr)
        # right_inst = _struct_from_pointer(pred_node.right_type,right_ptr)
        # left_val = lower_getattr(left_inst, pred_node.left_attr)
        # right_val = lower_getattr(right_inst, pred_node.right_attr)
        return exec_op(pred_node.op_str, left_val, right_val)
    else:
        return 0xFF


@njit(cache=True,inline='always')
def update_pair(pred_node, truth_values, left_facts, right_facts, i, j):
    '''Updates an BetaPredicateNode for facts i and j of left_facts and right_facts'''
    truth = beta_eval_truth(pred_node, left_facts, right_facts, i, j)
    truth_values[i,j] = truth



@njit(cache=True,locals={'new_size':u8})
def beta_filter(pnode_type, pred_meminfo, link_data, left_inds, right_inds, negated):
    '''Implements update_func of BetaPredicateNode subscriber'''

    # print(left_inds, right_inds)

    # return np.zeros((1,1))
    # if(pred_meminfo is None): return
    # Resolve this instance of the BetaPredicateNode, it's KnowledgeBase, and 
    #   the fact pointer vectors associated with this the left and right t_id.
    pred_node = _struct_from_meminfo(pnode_type, pred_meminfo)
    # kb = _struct_from_meminfo(KnowledgeBaseType, pred_node.kb_meminfo)
    grw_q = link_data.kb_grow_queue #kb.kb_data.grow_queue
    chg_q = link_data.kb_change_queue#kb.kb_data.change_queue
    left_facts = link_data.left_facts#facts_for_t_id(kb.kb_data,i8(pred_node.left_t_id))
    right_facts = link_data.right_facts#facts_for_t_id(kb.kb_data,i8(pred_node.right_t_id))
    #Expand the truth_values, and left and right consistencies to match fact vectors
    if(len(left_facts.data) > link_data.truth_values.shape[0] or
       len(right_facts.data) > link_data.truth_values.shape[1]):
        link_data.truth_values = expand_2d(link_data.truth_values,
                                    len(left_facts.data),len(right_facts.data),np.uint8
                                 )
    if(len(left_facts.data) > len(link_data.left_consistency)):
        link_data.left_consistency = expand_1d(link_data.left_consistency,
                                        len(left_facts.data),np.uint8)
    if(len(right_facts.data) > len(link_data.right_consistency)):
        link_data.right_consistency = expand_1d(link_data.right_consistency,
                                        len(right_facts.data),np.uint8)

    #Fill in inconsistencies, catching up to the KnoweldgeBases' grow_queue 
    for i in range(link_data.grow_head, grw_q.head):
        t_id, f_id, a_id = decode_idrec(grw_q[i])
        if(link_data.left_t_id == t_id):
            link_data.left_consistency[f_id] = 0
        if(link_data.right_t_id == t_id):
            link_data.right_consistency[f_id] = 0
    link_data.grow_head = grw_q.head

    #Fill in inconsistencies, catching up to the KnoweldgeBases' change_queue 
    for i in range(link_data.change_head, chg_q.head):
        t_id, f_id, a_id = decode_idrec(chg_q[i])
        if(link_data.left_t_id == t_id):
            link_data.left_consistency[f_id] = 0
        if(link_data.right_t_id == t_id):
            link_data.right_consistency[f_id] = 0
    link_data.change_head = chg_q.head

    #NOTE: This part might be sped up by running it only on request

    # Update all facts that are inconsistent, check inconsistent left_facts
    #   against all right facts.
    lc, rc = link_data.left_consistency, link_data.right_consistency
    for i in range(left_facts.head):
        if(not lc[i]):
            for j in range(right_facts.head):
                update_pair(pred_node, link_data.truth_values,left_facts,right_facts,i,j)
    
    # Check inconsistent right facts agains all left facts, except the ones that
    #   were already checked.
    for j in range(right_facts.head):
        if(not rc[j]):
            for i in range(left_facts.head):
                if(lc[i]): update_pair(pred_node, link_data.truth_values,left_facts,right_facts,i,j)

    # left and right inconsistencies all handled so fill in with 1s. 
    link_data.left_consistency[:len(left_facts)] = 1
    link_data.right_consistency[:len(right_facts)] = 1

    out = new_vector(8)
    for l_ind in left_inds:
        if(l_ind >= len(left_facts.data)): continue
        for r_ind in right_inds:
            if(r_ind >= len(left_facts.data)): continue
            if(link_data.truth_values[l_ind,r_ind] == (1 ^ negated)):
                out.add(l_ind)
                out.add(r_ind)
    return out.data[:out.head].reshape(out.head >> 1, 2)


@overload(BetaPredicateNode,prefer_literal=True)
def beta_ctor(left_fact_type, left_type, l_offsets, op_str, right_fact_type, right_type, r_offsets):
    if(not isinstance(op_str, types.Literal)): return 
    if(not isinstance(l_offsets, types.Array)):
        raise ValueError(f"BetaPredicateNode l_offsets must be array, got {left_attr_offsets}.")
    if(not isinstance(r_offsets, types.Array)):
        raise ValueError(f"BetaPredicateNode r_offsets must be array, got {left_attr_offsets}.")

    ctor, _ = define_beta_predicate_node(left_type.instance_type, op_str, right_type.instance_type)
    left_fact_type_name = left_fact_type.instance_type._fact_name
    right_fact_type_name = left_fact_type.instance_type._fact_name
    # specialization_dict = {
    #     'op_str' : types.literal(op_str.literal_value),
    #     'left_type' : left_type,
    #     'right_type' : right_type
    # }

    # d = {**beta_predicate_node_field_dict,**specialization_dict}
    # pnode_type = BetaPredicateNodeTemplate(base_subscriber_fields+[(k,v) for k,v, in d.items()])
    # print("!!!!",struct_type)
    def impl(left_fact_type, left_type, l_offsets, op_str, right_fact_type, right_type, r_offsets):

        # st = new(pnode_type)
        # init_base_subscriber(st)
        # init_beta(st, left_type._fact_name, left_attr_offsets, right_type._fact_name, right_attr_offsets)
        return ctor(str(left_fact_type_name), l_offsets, str(right_fact_type_name), r_offsets)

    return impl
                    

# def gen_beta_source(left_type, left_attr, op_str, right_type, right_attr):
def gen_beta_source(left_type, op_str, right_type):
    '''Generates or gets the cached definition for an BetaPredicateNode with the given 
        types, attributes, and comparison op. '''
    # left_typ_name = f'{left_type._fact_name}Type'
    # right_typ_name = f'{right_type._fact_name}Type'
    # left_fieldtype = left_type.field_dict[left_attr]
    # right_fieldtype = right_type.field_dict[right_attr]

    source = f'''
from numba import types, njit
from numba.experimental.structref import new
from numba.types import int64, float64, unicode_type
from cre.predicate_node import BetaPredicateNodeTemplate, init_beta, beta_filter, beta_predicate_node_field_dict
from cre.subscriber import base_subscriber_fields, init_base_subscriber

specialization_dict = {{
    'op_str' : types.literal('{op_str}'),
    'left_type' : types.TypeRef({left_type}),
    'right_type' : types.TypeRef({right_type}),
}}

field_dict = {{**beta_predicate_node_field_dict,**specialization_dict}}
fields = base_subscriber_fields + [(k,v) for k,v, in field_dict.items()]

pnode_type = BetaPredicateNodeTemplate(fields=fields)

@njit(cache=True)
def filter_func(pred_meminfo, link_data, left_inds, right_inds, negated):
    return beta_filter(pnode_type, pred_meminfo, link_data,  left_inds, right_inds, negated)        

@njit(cache=True)
def pre_ctor(left_fact_type_name, left_attr_offsets, right_fact_type_name, right_attr_offsets):
    st = new(pnode_type)
    init_base_subscriber(st)
    init_beta(st, left_fact_type_name, left_attr_offsets, right_fact_type_name, right_attr_offsets)
    return st

@njit
def ctor(left_fact_type_name, left_attr_offsets, right_fact_type_name, right_attr_offsets):
    st = pre_ctor(left_fact_type_name, left_attr_offsets, right_fact_type_name, right_attr_offsets)
    st.filter_func = filter_func
    return st
    '''
    return source


def define_beta_predicate_node(left_type, op_str, right_type):
    '''Generates or gets the cached definition for an AlphaPredicateNode with the given 
        types, attributes, and comparison op. '''
    name = "BetaPredicate"
    if(isinstance(left_type,types.StructRef)): left_type = types.int64
    if(isinstance(right_type,types.StructRef)): right_type = types.int64
    hash_code = unique_hash([left_type, op_str, right_type])
    if(not source_in_cache(name,hash_code)):
        source = gen_beta_source(left_type, op_str, right_type)
        source_to_cache(name, hash_code, source)
        
    ctor, pnode_type = import_from_cached(name, hash_code,['ctor','pnode_type']).values()

    return ctor, pnode_type

def get_beta_predicate_node_definition(left_fact_type, left_attr_chain, op_str, right_fact_type, right_attr_chain):
    '''Gets various definitions for an BetaPredicateNode, returns a dict with 'ctor'
         'pnode_type', 'left_type', 'left_attr_offsets', 'left_t_id', 'right_type', 'right_attr_offsets', 'right_t_id' '''
    context = kb_context()    
    left_t_id = context.fact_to_t_id[left_fact_type._fact_name]
    right_t_id = context.fact_to_t_id[right_fact_type._fact_name]

    if(not isinstance(left_attr_chain,list)): attr_chain = [left_attr_chain]
    if(not isinstance(right_attr_chain,list)): attr_chain = [right_attr_chain]
    
    left_type, left_attr_offsets = resolve_deref(left_fact_type, left_attr_chain)
    right_type, right_attr_offsets = resolve_deref(right_fact_type, right_attr_chain)
    ctor, pnode_type = define_beta_predicate_node(left_type, op_str, right_type)

    return locals()


def get_beta_predicate_node(left_fact_type, left_attr_chain, op_str, right_fact_type, right_attr_chain):
    '''Gets a new instance of an AlphaPredicateNode that evals 
        op(left_type.left_attr, right_type.right_attr).'''

    dfn = get_beta_predicate_node_definition(left_fact_type, left_attr_chain, op_str, right_fact_type, right_attr_chain)
    ctor, left_t_id, left_attr_offsets, right_t_id, right_attr_offsets = \
     itemgetter('ctor', 'left_t_id', 'left_attr_offsets', 'right_t_id', 'right_attr_offsets')(dfn)
    
    lft = left_fact_type._fact_name
    rft = right_fact_type._fact_name
    # print("LR<",lft, rft)
    out = ctor(lft, left_attr_offsets, rft, right_attr_offsets)

    return out


# @overload_method(BetaPredicateNodeTemplate, 'filter')
# def _impl_beta_filter(self, left_inds, right_inds):
#     def impl(self, left_inds, right_inds):
#         return self.filter_func(_meminfo_from_struct(self),left_inds, right_inds)
#     return impl


#### Linking ####





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

#  April 6, 2021
# Conditions are instantiated seperately from the knowledge base
# When they are linked to a knowledge base a copy might need to be made
#   because the creation of the conditions was a prototype but they
#   will have different cache info.
# When linked, predicate node in the conditions need to be initialized with
#    the fact vector for the knowledge base 
# Somehow predicate nodes need to be shared across Conditions objects
#   probably makes sense to have a predicate node cache in the knowledgebase
#   that is a dictionary of the predicate nodes' str (which needs to be precomputed)
#   as something like str(Var) + str(comp) + str(val/Var)
#   this should happen at link time if the node is cached then use that one
#   otherwise make a copy insert it into the dict and use that copy
#
# What is the speed tradeoff between passing indicies around vs masks around?
# does it even make sense to keep a big mask of the true comparisons?
#  if there are multiple conditions with the same nodes it may make sense
#  although the alternative is to pass around arrays of indicies in the 
#  alpha case... seems like the mask makes sense since there is a
#  dedicated slot for each item it only needs to recheck things that 
#  have changed..
#  ... so inside the conditions object, external to the predicate nodes
#  there can be a set of alpha and beta memory indicies which are sort
#  of the sparse versions of the masks
# alpha->alpha:
#  update() should pass in a set of indicies vetted by the previous node
#  and output the subset that passes then next alpha
# alpha->beta:
#  all beta nodes are updated independantly of one another 
# 



# The Plan:
# 1. [0%] Implement id_str from conditions object
# 2. [0%] Implement link_copy(node, kb) + tests which 
#  copies a prototype predicate node and links it to a knowledge_base
#  -attr_offsets, id_str, and op are kept
#  -left_facts, right_fact, kb_grow_queue, kb_change_queue are ripped
#  -truth values are reinstantiated (don't instantiate if attr_offsets > 1)
#  *By the end we should no longer need to import KnowledgeBase here
# 3. [20%] Implement filter() + tests as described above
# 4. Reimplement so that truth/consistency are densely bit packed 


# Small Hiccup, if the node can deref as much as it wants then
#  we need to account for this. We're not necessarily checking
#  for consistency with the index that comes in, so it can be
#  part of a different fact set. We can still use the original
#  fact set as the basis for the mask, but we can't really
#  trust the consistency, we need to recheck everything. Still
#  this is probably an improvement on RETE
