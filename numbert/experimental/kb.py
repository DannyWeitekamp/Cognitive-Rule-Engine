from numba import types, njit, guvectorize,vectorize,prange, generated_jit, literally
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba import types
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple, Tuple, Array, optional
from numba.cpython.unicode import  _set_code_point
from numba.experimental import structref
from numba.experimental.structref import new
from numba.extending import overload_method, intrinsic
from numbert.utils import cache_safe_exec
from numbert.core import TYPE_ALIASES, REGISTERED_TYPES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from numba.core import types, cgutils
from numba.core.errors import TypingError

from numbert.gensource import assert_gen_source
from numbert.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache
from collections import namedtuple
import numpy as np
import timeit
import itertools
import types as pytypes
import sys
import __main__

from numbert.experimental.context import _BaseContextful, KnowledgeBaseContextDataType, KnowledgeBaseContext
from numbert.experimental.transform import infer_type


from numbert.experimental.subscriber import BaseSubscriberType
from numbert.experimental.structref import define_structref
from numbert.experimental.fact import BaseFact,BaseFactType, cast_fact
from numbert.experimental.utils import lower_setattr, _cast_structref, _meminfo_from_struct, decode_idrec, encode_idrec
from numbert.caching import import_from_cached, source_in_cache, source_to_cache

   
#### KB Data Definition ####

i8_arr = i8[:]
u1_arr = u1[:]
str_to_bool_dict = DictType(unicode_type,u1_arr)
two_str = UniTuple(unicode_type,2)
two_str_set = DictType(two_str,u1)

meminfo_type = types.MemInfoPointer(types.voidptr)
basefact_list = ListType(BaseFactType)
i8_list = ListType(i8)


kb_data_fields = [
    ("facts" , ListType(basefact_list)),
    ("empty_f_id_stacks" , ListType(i8_list)),
    ("empty_f_id_heads" , ListType(i8)),
    ("names_to_idrecs" , DictType(unicode_type,u8)),

    ("enum_data" , DictType(unicode_type,i8_arr)),
    ("enum_consistency" , DictType(two_str,u1)),
    ("subscribers" , ListType(BaseSubscriberType)), 

    # ("consistency_listeners" , DictType(i8, two_str_set)),
    # ("consistency_listener_counter" , Array(i8, 0, "C")),
    # ("unnamed_counter" , Array(i8, 0, "C")),    
    ("NULL_FACT", BaseFactType)
]

KnowledgeBaseData, KnowledgeBaseDataType = define_structref("KnowledgeBaseData",kb_data_fields)


@njit(cache=True)
def expand_kb_data_types(kb_data,n):
    for i in range(n):
        kb_data.facts.append(List.empty_list(BaseFactType))
        kb_data.empty_f_id_stacks.append(List.empty_list(i8))
        kb_data.empty_f_id_heads.append(0)

@njit(cache=True)
def init_kb_data(context_data):
    kb_data = new(KnowledgeBaseDataType)
    kb_data.facts = List.empty_list(basefact_list)
    kb_data.empty_f_id_stacks = List.empty_list(i8_list)
    kb_data.empty_f_id_heads = List.empty_list(i8)
    

    kb_data.names_to_idrecs = Dict.empty(unicode_type,u8)

    kb_data.enum_data = Dict.empty(unicode_type,i8_arr)

    # kb_data.consistency_listeners = Dict.empty(i8, two_str_set)

    kb_data.enum_consistency = Dict.empty(two_str,u1)
    # consistency_listener_counter = np.zeros(1,dtype=np.int64) 
    # consistency_listeners[0] = enum_consistency
    # consistency_listener_counter += 1
    kb_data.subscribers = List.empty_list(BaseSubscriberType) #FUTURE: Replace w/ resolved type
    
    # kb_data.unnamed_counter = np.zeros(1,dtype=np.int64)
    kb_data.NULL_FACT = BaseFact()
    # kb_data = KnowledgeBaseData(facts, empty_f_id_stacks, empty_f_id_heads, names_to_idrecs,
    #                         enum_data, enum_consistency, subscribers,
    #                         unnamed_counter, BaseFact() #Empty BaseFact is NULL_FACT
    #                          )
    L = max(len(context_data.attr_inds_by_type),1)
    expand_kb_data_types(kb_data,L)
    return kb_data

#### Consistency ####

@njit(cache=True)
def signal_inconsistent(consistency_listeners, name, attr):
    for _,cm in consistency_listeners.items():
        cm[(name,attr)] = True


@njit(cache=True)
def add_consistency_map(kb_data, c_map):
    '''Adds a new consitency map, returns it's index in the knowledgebase'''
    _,_, consistency_listeners, consistency_listener_counter = kb_data
    consistency_listener_counter += 1
    consistency_listeners[consistency_listener_counter[0]] = c_map
    return consistency_listener_counter

@njit(cache=True)
def remove_consistency_map(kb_data, index):
    '''Adds a new consitency map, returns it's index in the knowledgebase'''
    _, _,_, consistency_listeners, _ = kb_data
    del consistency_listeners[index]

#### Knowledge Base Definition ####

class KnowledgeBase(structref.StructRefProxy):
    ''' '''
    def __new__(cls, context=None):
        context_data = KnowledgeBaseContext.get_context(context).context_data
        kb_data = init_kb_data(context_data)
        self = structref.StructRefProxy.__new__(cls, kb_data, context_data)
        # _BaseContextful.__init__(self,context) #Maybe want this afterall
        self.kb_data = kb_data
        self.context_data = context_data
        return self
    
    def add_subscriber(self,subscriber):
        return add_subscriber(self,subscriber)

    def declare(self,fact,name=None):
        return declare(self,fact,name)

    def retract(self,identifier):
        return retract(self,identifier)

    def all_facts_of_type(self,typ):
        return all_facts_of_type(self,typ)

    def modify(self, fact, attr, val):
        return modify(self,fact, attr, val)



@structref.register
class KnowledgeBaseTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

structref.define_proxy(KnowledgeBase, KnowledgeBaseTypeTemplate, ["kb_data", "context_data"])
KnowledgeBaseType = KnowledgeBaseTypeTemplate(fields=[
    ("kb_data",KnowledgeBaseDataType),
    ("context_data" , KnowledgeBaseContextDataType)])




#### Helper Functions ####

@njit(cache=True)
def make_f_id_empty(kb_data, t_id, f_id):
    '''Adds adds tracking info for an empty f_id for when a fact is retracted'''
    es_s = kb_data.empty_f_id_stacks[t_id]
    es_h = kb_data.empty_f_id_heads[t_id]
    if(es_h < len(es_s)):
        es_s[es_h] = f_id
    else:
        es_s.append(f_id)
    kb_data.empty_f_id_heads[t_id] += 1
    kb_data.facts[t_id][f_id] = kb_data.NULL_FACT


@njit(cache=True)
def next_empty_f_id(kb_data,t_id):
    '''Gets the next dead f_id from retracting facts otherwise returns 
        a fresh one pointing to the end of the meminfo list'''
    es_s = kb_data.empty_f_id_stacks[t_id]
    es_h = kb_data.empty_f_id_heads[t_id]
    if(es_h <= 0):
        return len(kb_data.facts[t_id]) # a fresh f_id

    kb_data.empty_f_id_heads[t_id] = es_h = es_h - 1
    return es_s[es_h] # a recycled f_id

@generated_jit(cache=True)
def resolve_t_id(kb, fact):
    if(isinstance(fact,types.TypeRef)):
        fact = fact.instance_type
    fact_type_name = fact._fact_name
    def impl(kb, fact):
        t_id = kb.context_data.fact_to_t_id[fact_type_name]
        L = len(kb.kb_data.facts)
        if(t_id >= L):
            expand_kb_data_types(kb.kb_data, 1+L-t_id)
        return  t_id
    return impl

@njit(cache=True)
def name_to_idrec(kb,name):
    names_to_idrecs = kb.kb_data.names_to_idrecs
    if(name not in names_to_idrecs):
        raise KeyError("Fact not found.")
    return names_to_idrecs[name]

##### add_subscriber #####

@njit(cache=True)
def add_subscriber(kb, subscriber):
    l = len(kb.kb_data.subscribers)
    base_subscriber = _cast_structref(BaseSubscriberType,subscriber)
    kb.kb_data.subscribers.append(base_subscriber)
    if(subscriber.kb_meminfo is None):
        subscriber.kb_meminfo = _meminfo_from_struct(kb)
    else:
        raise RuntimeError("Subscriber can only be linked to one KnowledgeBase.")

    return l

##### subscriber signalling ####

@njit(cache=True)
def signal_subscribers_grow(kb, idrec):
    for sub in kb.kb_data.subscribers:
        sub.grow_queue.append(idrec)

@njit(cache=True)
def signal_subscribers_change(kb, idrec):
    for sub in kb.kb_data.subscribers:
        sub.change_queue.append(idrec)


##### declare #####

@njit(cache=True)
def declare_fact(kb,fact):
    t_id = resolve_t_id(kb,fact)
    f_id = next_empty_f_id(kb.kb_data,t_id)
    b_fact = cast_fact(BaseFactType,fact)
    facts = kb.kb_data.facts[t_id]

    idrec = encode_idrec(t_id,f_id,0)
    b_fact.idrec = idrec

    if(f_id < len(facts)):
        facts[f_id] = b_fact
        signal_subscribers_change(kb, idrec)
    else:
        facts.append(b_fact)
        signal_subscribers_grow(kb, idrec)
    
    return idrec

@njit(cache=True)
def declare_name(kb,name,idrec):
    kb.kb_data.names_to_idrecs[name] = idrec

@njit(cache=True)
def declare_fact_name(kb,fact,name):
    idrec = declare_fact(kb,fact)        
    declare_name(kb,name,idrec)
    return idrec

@njit(cache=True)
def declare(kb,fact,name):
    return kb.declare(fact,name)

##### retract #####

@njit(cache=True)
def retract_by_idrec(kb,idrec):
    t_id, f_id, a_id = decode_idrec(idrec)
    make_f_id_empty(kb.kb_data,i8(t_id), i8(f_id))
    signal_subscribers_change(kb, idrec)

@njit(cache=True)
def retract_by_name(kb,name):
    idrec = name_to_idrec(kb,name)
    retract_by_idrec(kb,idrec)
    del kb.kb_data.names_to_idrecs[name]

@njit(cache=True)
def retract(kb,identifier):
    return kb.retract(identifier)

##### modify #####

@njit(cache=True)
def modify_by_fact(kb,fact,attr,val):
    lower_setattr(fact,literally(attr),val)
    #TODO signal_subscribers w/ idrec w/ attr_ind
    signal_subscribers_change(kb, fact.idrec)

@njit(cache=True)
def modify_by_idrec(kb,fact,attr,val):

    raise NotImplemented()
    #lower_setattr(fact,literally(attr),val)
    #TODO signal_subscribers w/ idrec w/ attr_ind

@njit(cache=True)
def modify(kb,fact,attr,val):
    return kb.modify(fact,attr,val)


##### all_facts_of_type #####

@njit(cache=True)
def all_facts_of_type(kb,typ):
    t_id = resolve_t_id(kb,typ)
    out = List()
    for b_fact in kb.kb_data.facts[t_id]:
        if(b_fact.idrec != u8(-1)):
            out.append(cast_fact(typ,b_fact))
    return out

#### KnowledgeBase Overloading #####

@overload_method(KnowledgeBaseTypeTemplate, "add_subscriber")
def kb_declare(self, subscriber):
    if(not isinstance(subscriber,types.StructRef)): 
        raise TypingError(f"Cannot add subscriber of type '{type(fact)}'.")
    def impl(self, subscriber):
        return add_subscriber(self,subscriber)
    return impl


@overload_method(KnowledgeBaseTypeTemplate, "declare")
def kb_declare(self, fact, name=None):
    if(not isinstance(fact,types.StructRef)): 
        raise TypingError(f"Cannot declare fact of type '{type(fact)}'.")
    if(not name or isinstance(name, (types.NoneType,types.Omitted))):
        def impl(self, fact, name=None):
            return declare_fact(self,fact)
    else:
        def impl(self, fact, name=None):
            return declare_fact_name(self,fact,name)
    return impl

@overload_method(KnowledgeBaseTypeTemplate, "retract")
def kb_retract(self, identifier):
    if(identifier in (str,unicode_type)):
        def impl(self, identifier):
            return retract_by_name(self,identifier)
    elif(isinstance(identifier,types.Integer)):
        def impl(self, identifier):
            return retract_by_idrec(self,u8(identifier))
    elif(isinstance(identifier,types.StructRef)):
        def impl(self, identifier):
            return retract_by_idrec(self,identifier.idrec)
    else:
        raise TypingError(f"Cannot retract fact identifier of type '{type(identifier)}'." +
                        "kb.retract() accepts a valid fact idrec, name, or fact instance.")
    return impl

@overload_method(KnowledgeBaseTypeTemplate, "modify", prefer_literal=True)
def kb_modify(self, fact, attr, val):
    if(not isinstance(fact,types.StructRef)): 
        raise TypingError(f"Modify requires a fact instance, got instance of'{type(fact)}'.")
    def impl(self, fact, attr, val):
        return modify_by_fact(self, fact, attr, val)
    return impl

@overload_method(KnowledgeBaseTypeTemplate, "all_facts_of_type")
def kb_all_facts_of_type(self, typ):
    def impl(self, typ):
        return all_facts_of_type(self,typ)

    return impl
