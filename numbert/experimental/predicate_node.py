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
from numba import types, njit, i8, u8, i4, u1, literally, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type, void
from numba.experimental.structref import new
from numba.extending import overload_method, intrinsic
from numbert.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from numbert.experimental.context import kb_context
from numbert.experimental.structref import define_structref, define_structref_template
from numbert.experimental.kb import KnowledgeBaseType, KnowledgeBase, facts_for_t_id, fact_at_f_id
from numbert.experimental.fact import define_fact, BaseFactType, cast_fact
from numbert.experimental.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, \
 decode_idrec, lower_getattr, _struct_from_pointer, struct_get_attr_offset, _struct_get_data_pointer, \
 _load_pointer, _pointer_to_data_pointer
from numbert.experimental.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from copy import copy
from operator import itemgetter

meminfo_type = types.MemInfoPointer(types.voidptr)


@njit(cache=True)
def exec_op(op_str,a,b):
    '''Executes one of 5 boolean comparison operations. Since the predicate_node type
       fixes 'op_str' to a literal value. LLVM will compile out the switch case.
    '''
    if(op_str == "<"):
        return a < b
    elif(op_str == "<="):
        return a <= b
    elif(op_str == ">"):
        return b > a
    elif(op_str == ">="):
        return a >= b
    elif(op_str == "=="):
        return a == b
    raise ValueError()


#### Struct Definitions ####

base_predicate_node_field_dict = {
    "left_t_id" : i8,
    "left_attr_offsets" : i8[:],#types.Any,
    "left_type" : types.Any,
    "op_str" : unicode_type,
    "truth_values" : u1[:,:],
}

basepredicate_node_fields = [(k,v) for k,v, in base_predicate_node_field_dict.items()]
BasePredicateNode, BasePredicateNodeType = define_structref("BasePredicateNode", base_subscriber_fields + basepredicate_node_fields)


alpha_predicate_node_field_dict = {
    **base_predicate_node_field_dict,
    # "truth_values" : u1[:],
    "right_val" : types.Any,
}
alpha_predicate_node_fields = [(k,v) for k,v, in alpha_predicate_node_field_dict.items()]
AlphaPredicateNode, AlphaPredicateNodeTemplate = define_structref_template("AlphaPredicateNode", base_subscriber_fields + alpha_predicate_node_fields)


beta_predicate_node_field_dict = {
    **base_predicate_node_field_dict,
    
    "right_t_id" : i8,
    "right_attr_offsets" : i8[:],
    "right_type" : types.Any,
    "left_consistency" : u1[:],
    "right_consistency" : u1[:],
}
beta_predicate_node_fields = [(k,v) for k,v, in beta_predicate_node_field_dict.items()]
BetaPredicateNode, BetaPredicateNodeTemplate = define_structref_template("BetaPredicateNode", base_subscriber_fields + beta_predicate_node_fields)


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

@njit(cache=True)
def init_alpha(st,t_id, attr_offsets, right_val):
    '''Initializes an empty AlphaPredicateNode with t_id and right_val'''
    st.truth_values = np.empty((0,0),dtype=np.uint8)
    st.left_t_id = t_id
    st.left_attr_offsets = attr_offsets
    st.right_val = right_val
    

@njit(cache=True)
def deref_attrs(val_type, inst_ptr,attr_offsets):
    #TODO: Allow to deref arbitrary number of attributes
    data_ptr = _pointer_to_data_pointer(inst_ptr)
    val = _load_pointer(val_type, data_ptr+attr_offsets[0])
    return val
    


@njit(cache=True)
def alpha_eval_truth(kb,facts,f_id, pred_node):
    '''Updates an AlphaPredicateNode with fact at f_id'''
    inst_ptr = facts.data[i8(f_id)]
    if(inst_ptr != 0):
        val = deref_attrs(pred_node.left_type, inst_ptr, pred_node.left_attr_offsets)
        return exec_op(pred_node.op_str, val, pred_node.right_val)
    else:
        return 0xFF

@njit(cache=True,locals={'new_size':u8})
def alpha_update(pred_meminfo,pnode_type):
    '''Implements update_func of AlphaPredicateNode subscriber'''
    if(pred_meminfo is None): return

    # Resolve this instance of the AlphaPredicateNode, it's KnowledgeBase, and 
    #   the fact pointer vector associated with this AlphaPredicateNode's t_id
    pred_node = _struct_from_meminfo(pnode_type, pred_meminfo)
    kb = _struct_from_meminfo(KnowledgeBaseType, pred_node.kb_meminfo)
    grw_q = kb.kb_data.grow_queue
    chg_q = kb.kb_data.change_queue
    facts = facts_for_t_id(kb.kb_data,i8(pred_node.left_t_id))

    # Ensure that truth_values is the size of the fact pointer vector
    if(len(facts.data) > len(pred_node.truth_values)):
        pred_node.truth_values = expand_2d(pred_node.truth_values,len(facts.data),1,np.uint8)

    # Update from the grow head to the KnowledgeBase's grow head  
    for i in range(pred_node.grow_head, grw_q.head):
        t_id, f_id, a_id = decode_idrec(grw_q[i])
        if(pred_node.left_t_id == t_id):
            truth = alpha_eval_truth(kb,facts,f_id, pred_node)
            pred_node.truth_values[f_id,0] = truth
            # pred_node.grow_queue.add(f_id)
    pred_node.grow_head = grw_q.head

    # Update from the change head to the KnowledgeBase's change head
    for i in range(pred_node.change_head, chg_q.head):
        t_id, f_id, a_id = decode_idrec(chg_q[i])
        if(pred_node.left_t_id == t_id):
            truth = alpha_eval_truth(kb,facts,f_id, pred_node)
            pred_node.truth_values[f_id,0] = truth
            # if(truth != pred_node.truth_values[f_id]):
            #     pred_node.change_queue.add(f_id)
    pred_node.change_head = chg_q.head


def gen_alpha_source(left_type, op_str, right_type):
    '''Produces source code for an AlphaPredicateNode that is specialized for the given types,
        attribute and comparison operator.'''
    # typ_name = f'{typ._fact_name}Type'
    # literal_type = types.literal(literal_val).literal_type
    if(isinstance(right_type,types.Integer)): right_type = types.float64
    # fieldtype = typ.field_dict[attr]
    source = f'''
from numba import types, njit
from numba.experimental.structref import new
from numba.types import *
from numbert.experimental.predicate_node import AlphaPredicateNodeTemplate, init_alpha, alpha_update, alpha_predicate_node_field_dict
from numbert.experimental.subscriber import base_subscriber_fields, init_base_subscriber

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
def update_func(pred_meminfo):
    alpha_update(pred_meminfo, pnode_type)        

@njit(cache=True)
def pre_ctor(t_id, attr_offsets, literal_val):
    st = new(pnode_type)
    init_base_subscriber(st)
    init_alpha(st,t_id, attr_offsets, literal_val)
    return st

@njit
def ctor(t_id, attr_offsets, literal_val):
    st = pre_ctor(t_id, attr_offsets, literal_val)
    st.update_func = update_func
    return st
    '''
    return source


def resolve_deref(typ,attr_chain):
    offsets = np.empty((len(attr_chain),),dtype=np.int64)
    out_type = typ 
    for i, attr in enumerate(attr_chain):
        fd = out_type.field_dict
        offsets[i] = out_type._attr_offsets[list(fd.keys()).index(attr)]  #struct_get_attr_offset(out_type,attr) #For some reason ~4.6ms
        out_type = fd[attr]

    return out_type, offsets

def define_alpha_predicate_node(left_type, op_str, right_type):
    '''Generates or gets the cached definition for an AlphaPredicateNode with the given 
        types, attributes, and comparison op. '''
    name = "AlphaPredicate"
    hash_code = unique_hash([left_type, op_str, right_type])
    if(not source_in_cache(name,hash_code)):
        source = gen_alpha_source(left_type, op_str, right_type)
        source_to_cache(name, hash_code, source)
        
    ctor, pnode_type = import_from_cached(name, hash_code,['ctor','pnode_type']).values()

    return ctor, pnode_type

def get_alpha_predicate_node_definition(typ, attr_chain, op_str, right_type):
    '''Gets various definitions for an AlphaPredicateNode, returns a dict with 'ctor'
         'pnode_type', 'left_type', 'left_attr_offsets', 't_id' '''
    context = kb_context()    
    t_id = context.fact_to_t_id[typ._fact_name]

    if(not isinstance(attr_chain,list)): attr_chain = [attr_chain]
    
    left_type, left_attr_offsets = resolve_deref(typ, attr_chain)
    ctor, pnode_type = define_alpha_predicate_node(left_type, op_str, right_type)

    return locals()
    

def get_alpha_predicate_node(typ, attr_chain, op_str, literal_val):
    '''Gets a new instance of an AlphaPredicateNode that evals op(typ.attr, literal_val) '''
    right_type = types.literal(literal_val).literal_type

    dfn = get_alpha_predicate_node_definition(typ, attr_chain, op_str, right_type)
    ctor, t_id, left_attr_offsets = itemgetter('ctor', 't_id','left_attr_offsets')(dfn)
        
    out = ctor(t_id, left_attr_offsets, literal_val)

    return out


#### Beta Predicate Nodes ####

@njit(cache=True)
def init_beta(st, left_t_id, left_attr_offsets, right_t_id, right_attr_offsets):
    '''Initializes an empty BetaPredicateNode with left_t_id and right_t_id'''
    st.truth_values = np.empty((0,0),dtype=np.uint8)
    st.left_t_id = left_t_id
    st.left_attr_offsets = left_attr_offsets
    st.right_t_id = right_t_id
    st.right_attr_offsets = right_attr_offsets
    

@njit(cache=True)
def beta_eval_truth(kb,pred_node, left_facts, right_facts, i, j):
    '''Eval truth for BetaPredicateNode for facts i and j of left_facts and right_facts'''
    left_ptr = left_facts.data[i]
    right_ptr = right_facts.data[j]

    if(left_ptr != 0 and right_ptr != 0):
        left_val = deref_attrs(pred_node.left_type, left_ptr, pred_node.left_attr_offsets)
        right_val = deref_attrs(pred_node.right_type, right_ptr, pred_node.right_attr_offsets)
        # left_inst = _struct_from_pointer(pred_node.left_type,left_ptr)
        # right_inst = _struct_from_pointer(pred_node.right_type,right_ptr)
        # left_val = lower_getattr(left_inst, pred_node.left_attr)
        # right_val = lower_getattr(right_inst, pred_node.right_attr)
        return exec_op(pred_node.op_str, left_val, right_val)
    else:
        return 0#0xFF


@njit(cache=True,inline='always')
def update_pair(kb,pred_node, left_facts, right_facts, i, j):
    '''Updates an BetaPredicateNode for facts i and j of left_facts and right_facts'''
    truth = beta_eval_truth(kb,pred_node, left_facts, right_facts, i, j)
    pred_node.truth_values[i,j] = truth



@njit(cache=True,locals={'new_size':u8})
def beta_update(pred_meminfo,pnode_type):
    '''Implements update_func of BetaPredicateNode subscriber'''
    if(pred_meminfo is None): return
    # Resolve this instance of the BetaPredicateNode, it's KnowledgeBase, and 
    #   the fact pointer vectors associated with this the left and right t_id.
    pred_node = _struct_from_meminfo(pnode_type, pred_meminfo)
    kb = _struct_from_meminfo(KnowledgeBaseType, pred_node.kb_meminfo)
    grw_q = kb.kb_data.grow_queue
    chg_q = kb.kb_data.change_queue
    left_facts = facts_for_t_id(kb.kb_data,i8(pred_node.left_t_id))
    right_facts = facts_for_t_id(kb.kb_data,i8(pred_node.right_t_id))

    #Expand the truth_values, and left and right consistencies to match fact vectors
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

    #Fill in inconsistencies, catching up to the KnoweldgeBases' grow_queue 
    for i in range(pred_node.grow_head, grw_q.head):
        t_id, f_id, a_id = decode_idrec(grw_q[i])
        if(pred_node.left_t_id == t_id):
            pred_node.left_consistency[f_id] = 0
        if(pred_node.right_t_id == t_id):
            pred_node.right_consistency[f_id] = 0
    pred_node.grow_head = grw_q.head

    #Fill in inconsistencies, catching up to the KnoweldgeBases' change_queue 
    for i in range(pred_node.change_head, chg_q.head):
        t_id, f_id, a_id = decode_idrec(chg_q[i])
        if(pred_node.left_t_id == t_id):
            pred_node.left_consistency[f_id] = 0
        if(pred_node.right_t_id == t_id):
            pred_node.right_consistency[f_id] = 0
    pred_node.change_head = chg_q.head


    #NOTE: This part might be sped up by running it only on request

    # Update all facts that are inconsistent, check inconsistent left_facts
    #   against all right facts.
    lc, rc = pred_node.left_consistency, pred_node.right_consistency
    for i in range(left_facts.head):
        if(not lc[i]):
            for j in range(right_facts.head):
                update_pair(kb,pred_node,left_facts,right_facts,i,j)
    
    # Check inconsistent right facts agains all left facts, except the ones that
    #   were already checked.
    for j in range(right_facts.head):
        if(not rc[j]):
            for i in range(left_facts.head):
                if(lc[i]): update_pair(kb,pred_node,left_facts,right_facts,i,j)

    # left and right inconsistencies all handled so fill in with 1s. 
    pred_node.left_consistency[:len(left_facts)] = 1
    pred_node.right_consistency[:len(right_facts)] = 1
                    

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
from numba.types import *
from numbert.experimental.predicate_node import BetaPredicateNodeTemplate, init_beta, beta_update, beta_predicate_node_field_dict
from numbert.experimental.subscriber import base_subscriber_fields, init_base_subscriber

specialization_dict = {{
    'op_str' : types.literal('{op_str}'),
    'left_type' : types.TypeRef({left_type}),
    'right_type' : types.TypeRef({right_type}),
}}

field_dict = {{**beta_predicate_node_field_dict,**specialization_dict}}
fields = base_subscriber_fields + [(k,v) for k,v, in field_dict.items()]

pnode_type = BetaPredicateNodeTemplate(fields=fields)


@njit(cache=True)
def update_func(pred_meminfo):
    beta_update(pred_meminfo, pnode_type)        

@njit(cache=True)
def pre_ctor(left_t_id, left_attr_offsets, right_t_id, right_attr_offsets):
    st = new(pnode_type)
    init_base_subscriber(st)
    init_beta(st, left_t_id, left_attr_offsets, right_t_id, right_attr_offsets)
    return st

@njit
def ctor(*args):
    st = pre_ctor(*args)
    st.update_func = update_func
    return st
    '''
    return source


def define_beta_predicate_node(left_type, op_str, right_type):
    '''Generates or gets the cached definition for an AlphaPredicateNode with the given 
        types, attributes, and comparison op. '''
    name = "BetaPredicate"
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

    out = ctor(left_t_id, left_attr_offsets, right_t_id, right_attr_offsets)

    return out




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
