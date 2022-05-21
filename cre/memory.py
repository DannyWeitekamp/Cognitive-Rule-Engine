
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
from numba.extending import overload_method, intrinsic, overload, lower_cast, lower_builtin, SentryLiteralArgs
from numba.core.errors import RequireLiteralValue
from cre.core import TYPE_ALIASES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from numba.core import types, cgutils
from numba.core.errors import TypingError

from cre.gensource import assert_gen_source
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache
from collections import namedtuple
import numpy as np
import timeit
import itertools
import types as pytypes
import sys
import cloudpickle
import __main__

from cre.context import _BaseContextful, CREContextDataType, CREContext, ensure_inheritance, cre_context
from cre.transform import infer_type


from cre.subscriber import BaseSubscriberType
from cre.structref import define_structref, define_boxing
from cre.fact import Fact, BaseFact,BaseFact, cast_fact, get_inheritance_t_ids
from cre.tuple_fact import TF, TupleFact
from cre.fact_intrinsics import fact_lower_setattr
from cre.utils import CastFriendlyMixin, lower_setattr, _cast_structref, _meminfo_from_struct, decode_idrec, encode_idrec, \
 _raw_ptr_from_struct, _ptr_from_struct_incref,  _struct_from_ptr, _decref_ptr, _decref_structref, _raw_ptr_from_struct_incref, _obj_cast_codegen
from cre.vector import new_vector, VectorType
from cre.caching import import_from_cached, source_in_cache, source_to_cache

BASE_T_ID_STACK_SIZE = 16
BASE_F_ID_STACK_SIZE = 64
BASE_FACT_SET_SIZE = 64
BASE_CHANGE_QUEUE_SIZE = 2048
   
#### mem Data Definition ####

i8_arr = i8[:]
u1_arr = u1[:]
str_to_bool_dict = DictType(unicode_type,u1_arr)
two_str = UniTuple(unicode_type,2)
two_str_set = DictType(two_str,u1)

meminfo_type = types.MemInfoPointer(types.voidptr)
basefact_list = ListType(BaseFact)
i8_arr = i8[:]


mem_data_fields = [
    #Vector<*Vector<*BaseFact>> i.e. 2D vector that holds pointers to facts
    ("facts" , VectorType), #<- will be arr of pointers to vectors
    #Vector<*Vector<i8>> i.e. 2D vector that holds retracted f_ids
    ("retracted_f_ids" , VectorType), 
    # ("empty_f_id_heads" , i8[:]),
    ("n_types" , i8),
    ("names_to_idrecs" , DictType(unicode_type,u8)),

    ("enum_data" , DictType(unicode_type,i8_arr)),
    ("enum_consistency" , DictType(two_str,u1)),
    ("subscribers" , ListType(BaseSubscriberType)), #<- Might not need 

    ("change_queue" , VectorType), 
    # ("grow_queue" , VectorType), 

    ("NULL_FACT", BaseFact)
]

MemoryData, MemoryDataType = define_structref("MemoryData",mem_data_fields)



@njit(cache=True)
def expand_mem_data_types(mem_data,n):
    old_n = mem_data.n_types
    mem_data.n_types += n

    for i in range(n):
        v = new_vector(BASE_F_ID_STACK_SIZE)    
        v_ptr = _raw_ptr_from_struct_incref(v)

        mem_data.retracted_f_ids.add(v_ptr)

        v = new_vector(BASE_FACT_SET_SIZE)    
        v_ptr = _raw_ptr_from_struct_incref(v)
        mem_data.facts.add(v_ptr)
        
        # mem_data.facts.append(List.empty_list(BaseFact))

    # return v
        

@njit(cache=True)
def init_mem_data(context_data):
    mem_data = new(MemoryDataType)
    mem_data.facts = new_vector(BASE_T_ID_STACK_SIZE)#List.empty_list(basefact_list)
    mem_data.retracted_f_ids = new_vector(BASE_T_ID_STACK_SIZE)
    # mem_data.empty_f_id_stacks = List.empty_list(i8_arr)
    # mem_data.empty_f_id_heads = np.zeros((BASE_T_ID_STACK_SIZE,),dtype=np.int64)
    mem_data.n_types = 0
    

    mem_data.names_to_idrecs = Dict.empty(unicode_type,u8)

    mem_data.enum_data = Dict.empty(unicode_type,i8_arr)

    # mem_data.consistency_listeners = Dict.empty(i8, two_str_set)

    mem_data.enum_consistency = Dict.empty(two_str,u1)
    # consistency_listener_counter = np.zeros(1,dtype=np.int64) 
    # consistency_listeners[0] = enum_consistency
    # consistency_listener_counter += 1
    mem_data.subscribers = List.empty_list(BaseSubscriberType) #FUTURE: Replace w/ resolved type

    mem_data.change_queue = new_vector(BASE_CHANGE_QUEUE_SIZE)
    # mem_data.grow_queue = new_vector(BASE_CHANGE_QUEUE_SIZE)
    
    # mem_data.unnamed_counter = np.zeros(1,dtype=np.int64)
    mem_data.NULL_FACT = BaseFact()
    # mem_data = MemoryData(facts, empty_f_id_stacks, empty_f_id_heads, names_to_idrecs,
    #                         enum_data, enum_consistency, subscribers,
    #                         unnamed_counter, BaseFact() #Empty BaseFact is NULL_FACT
    #                          )
    L = max(len(context_data.parent_t_ids)+1,1)
    expand_mem_data_types(mem_data,L)
    return mem_data

@njit(cache=True)
def mem_data_dtor(mem_data):
    '''Decref out data structures in mem_data that we explicitly incref'ed '''

    #Decref all declared facts and their container vectors 
    for i in range(mem_data.facts.head):
        facts_ptr = mem_data.facts.data[i]
        facts = _struct_from_ptr(VectorType, facts_ptr)
        for j in range(facts.head):
            fact_ptr = facts.data[j]
            _decref_ptr(fact_ptr)
        _decref_ptr(facts_ptr)

    #Decref the inner vectors of retracted_f_ids
    for i in range(mem_data.retracted_f_ids.head):
        ptr = mem_data.retracted_f_ids.data[i]
        _decref_ptr(ptr)





#### Consistency ####

@njit(cache=True)
def signal_inconsistent(consistency_listeners, name, attr):
    for _,cm in consistency_listeners.items():
        cm[(name,attr)] = True


@njit(cache=True)
def add_consistency_map(mem_data, c_map):
    '''Adds a new consitency map, returns it's index in the Memory'''
    _,_, consistency_listeners, consistency_listener_counter = mem_data
    consistency_listener_counter += 1
    consistency_listeners[consistency_listener_counter[0]] = c_map
    return consistency_listener_counter

@njit(cache=True)
def remove_consistency_map(mem_data, index):
    '''Adds a new consitency map, returns it's index in the Memory'''
    _, _,_, consistency_listeners, _ = mem_data
    del consistency_listeners[index]

@njit(cache=True)
def decref_fact(x):
    return _decref_structref(x)


#### Memory Definition ####

class Memory(structref.StructRefProxy):
    ''' '''
    def __new__(cls, context=None):
        context = cre_context(context)#CREContext.get_context(context)
        context_data = context.context_data
        mem_data = init_mem_data(context_data)
        self = mem_ctor(context_data,mem_data)
        # self = structref.StructRefProxy.__new__(cls, context_data, mem_data)
        # _BaseContextful.__init__(self,context) #Maybe want this afterall
        self.mem_data = mem_data
        self.context_data = context_data
        self.context = context
        return self
    
    def add_subscriber(self,subscriber):
        return add_subscriber(self,subscriber)

    def declare(self,fact,name=None):
        idrec = mem_declare(self,fact,name)
        # if(name is None):
        #     idrec = declare_fact(self, fact)
        # else:
        #     idrec = declare_fact_name(self, fact, name)            
        return idrec


    def retract(self,identifier):
        return mem_retract(self,identifier)

    def get_facts(self,typ=None, no_subtypes=False):
        self.context._ensure_retro_registers()
        if(isinstance(typ,str)):
            typ = self.context.name_to_type[typ]

        # name = str(typ)
        # t_id = self.context.context_data.fact_num_to_t_id[typ._fact_num]
        # t_ids = np.empty((1,), dtype=np.uint16)
        # t_ids[0] = t_id
        return get_facts(self, typ, no_subtypes)

    def iter_facts(self,typ):
        self.context._ensure_retro_registers()
        if(isinstance(typ,str)):
            typ = self.context.name_to_type[typ]
        # name = str(typ)
        # t_id = self.context.context_data.fact_num_to_t_id[typ._fact_num]
        # t_ids = np.empty((1,), dtype=np.uint16)
        # t_ids[0] = t_id
        return iter_facts(self, typ)


    def get_fact(self,idrec):
        return get_fact(self, idrec)
        # return iter_facts(self, typ)#all_facts_of_t_id(self, t_id)

    def modify(self, fact, attr, val):
        return mem_modify(self,fact, attr, val)#modify(self,fact, attr, val)

    @property
    def halt_flag(self):
        return get_halt_flag(self)

    @property
    def backtrack_flag(self):
        return get_backtrack_flag(self)

    def _repr_helper(self,rfunc,ind="    ",sep="\n",pad='\n'):
        strs = []
        # for typ in np.nonzero(self.mem_data.facts)[0]:
        #     if(isinstance(typ,Fact)):
        for fact in self.get_facts():
            strs.append(str(fact))
        nl = "\n"
        return f'''Memory(facts=({pad}{ind}{f"{sep}{ind}".join(strs)}{pad})'''

    def __str__(self,**kwargs):
        from cre.utils import PrintElapse
        with PrintElapse("__STR__"):
            return self._repr_helper(str,**kwargs)


    def __repr__(self,**kwargs):
        from cre.utils import PrintElapse
        with PrintElapse("__REPR__"):
            return self._repr_helper(repr,**kwargs)
    

    def __del__(self):        
        # NOTE: This definitely has bugs
        try:
            # pass
            if(hasattr(self,'mem_data')):
                mem_data_dtor(self.mem_data)
        except Exception as e:
            # If the process is ending then global variables can be sporatically None
            #   thus skip any TypeError of this sort.
            if(isinstance(e, ImportError)): return
            if(isinstance(e, TypeError) and "not callable" in str(e)): return
            
            print("An error occured when trying to clean a cre.Memory object:\n",e)

@njit(cache=True)
def get_halt_flag(self):
    return self.halt_flag

@njit(cache=True)
def get_backtrack_flag(self):
    return self.backtrack_flag

@structref.register
class MemoryTypeTemplate(types.StructRef):
    def __str__(self):
        return "cre.Memory"
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

mem_fields = [
    ("mem_data", MemoryDataType),
    ("context_data" , CREContextDataType),
    ("halt_flag", u1),
    ("backtrack_flag", u1)
    ]


define_boxing(MemoryTypeTemplate,Memory)

MemoryType = MemoryTypeTemplate(mem_fields)
# structref.define_proxy(Memory, MemoryTypeTemplate, [x[0] for x in mem_fields])
# MemoryType = MemoryTypeTemplate(fields=)

@njit(cache=True)
def mem_ctor(context_data, mem_data=None):
    st = new(MemoryType) 
    st.context_data = context_data
    st.mem_data = mem_data if(mem_data is not None) else init_mem_data(context_data)
    
    st.halt_flag = u1(0)
    st.backtrack_flag = u1(0)
    return st

# @njit(cache=True)
# def mem_dtor(mem):
#     fact_vecs = mem.mem_data.facts
#     for i in range(len(fact_vecs)):
#         facts_ptr = fact_vecs[i]
#         if(facts_ptr != 0):
#             facts = _struct_from_ptr(VectorType,facts_ptr)   
#             for j in range(len(facts)):
#                 ptr = facts[j]
#                 if(ptr != 0): _decref_ptr(ptr)
                    
#             _decref_ptr(facts_ptr)

    

@overload(Memory)
def overload_Memory(context_data=None, mem_data=None):
    if(context_data is None):
        print("WARNING: haven't figured out instantiating mem in njit context")
        glb_context_data = CREContext.get_context().context_data
        def impl(context_data=None, mem_data=None):
            return mem_ctor(glb_context_data, mem_data)
    else:
        def impl(context_data=None, mem_data=None):
            return mem_ctor(context_data, mem_data)
        # def impl(context_data=None, mem_data=None):
        #     return mem_ctor(context_data, mem_data)

    return impl


@njit(cache=True)
def facts_for_t_id(mem_data,t_id):
    L = len(mem_data.facts)
    if(t_id >= L):
        expand_mem_data_types(mem_data, (t_id+1)-L)
    # print("Declare", t_id, mem_data.facts[t_id])
    return _struct_from_ptr(VectorType, mem_data.facts[t_id])

@njit(cache=True)
def fact_at_f_id(typ, t_id_facts,f_id):
    ptr = t_id_facts.data[f_id]
    if(ptr != 0):
        return _struct_from_ptr(typ, ptr)
    else:
        return None

@njit(cache=True)
def retracted_f_ids_for_t_id(mem_data,t_id):
    return _struct_from_ptr(VectorType, mem_data.retracted_f_ids[t_id])

#### Helper Functions ####

@njit(cache=True)
def make_f_id_empty(mem_data, t_id, f_id):
    '''Adds adds tracking info for an empty f_id for when a fact is retracted'''
    
    retracted_f_ids_for_t_id(mem_data,t_id).add(f_id)
    fact_ptr = facts_for_t_id(mem_data,t_id)[f_id]
    if(fact_ptr != 0):
        _decref_ptr(fact_ptr)
    
    facts_for_t_id(mem_data,t_id)[f_id] = 0


@njit(cache=True)
def next_empty_f_id(mem_data,facts,t_id):
    '''Gets the next dead f_id from retracting facts otherwise returns 
        a fresh one pointing to the end of the meminfo list'''
    f_id_vec = retracted_f_ids_for_t_id(mem_data,t_id)
    if(f_id_vec.head <= 0):
        return len(facts) # a fresh f_id

    return f_id_vec.pop()#es_s[es_h] # a recycled f_id

# @njit(cache=True,inline='never')
# def _expand_for_new_t_id(mem_data,t_id):
#     L = len(mem_data.facts)
#     if(t_id >= L):
#         expand_mem_data_types(mem_data, 1+L-t_id)

# @generated_jit(cache=True)
# def resolve_t_id(mem, fact):
#     if(isinstance(fact,types.TypeRef)):
#         fact = fact.instance_type

#     fact_num = fact._fact_num
#     def impl(mem, fact):
#         t_id = mem.context_data.fact_num_to_t_id[fact_num]
#         L = len(mem.mem_data.facts)
#         if(t_id >= L):
#             expand_mem_data_types(mem.mem_data, 1+L-t_id)
#         return t_id
        
#     return impl
# from cre.context import grow_fact_num_to_t_id
@njit(u2(MemoryType, BaseFact),cache=True)
def resolve_t_id(mem, fact):
    '''Gets a t_id from a mem (which is was instantiated under some context) 
       and a fact_num. If the fact with 'fact_num' wasn't registered in the 
       mem's context then ensure that it gets registered retroactively.
    '''
    # fact_num = fact.fact_num
    cd = mem.context_data
    # Ensure that 'fact_num_to_t_id' for the mem's context is big enough.
    # fntt = cd.fact_num_to_t_id
    # if(fact_num >= len(fntt)):
    #     fntt = grow_fact_num_to_t_id(cd, fact_num*2)

    # Get the t_id for fact_num. If zero then register fact_num and
    #  mark the context as having an unhandled registration that needs
    #  to be processed on the python side.
    # t_id = cd.fact_num_to_t_id[fact_num]
    # if(t_id == 0):
        # t_id = mem.context_data.get_next_t_id()

    t_id, _, _ = decode_idrec(fact.idrec)
    # ensure_inheritance()


    if(t_id >= len(cd.child_t_ids) or len(cd.child_t_ids[t_id])==0):
        inh_t_ids = get_inheritance_t_ids(fact)
        parent_t_id = -1
        for _t_id in inh_t_ids:
            ensure_inheritance(cd, _t_id, parent_t_id)
            parent_t_id = i8(_t_id)

        cd.has_unhandled_retro_register = True

    # Ensure that the data structures in mem_data are long enough to index t_id.
    L = len(mem.mem_data.facts)
    if(t_id >= L):
        expand_mem_data_types(mem.mem_data, 1+L-t_id)

    return t_id

@njit(cache=True)
def name_to_idrec(mem,name):
    names_to_idrecs = mem.mem_data.names_to_idrecs
    if(name not in names_to_idrecs):
        raise KeyError("Fact not found.")
    return names_to_idrecs[name]






##### add_subscriber #####

@njit(cache=True)
def add_subscriber(mem, subscriber):
    l = len(mem.mem_data.subscribers)
    base_subscriber = _cast_structref(BaseSubscriberType,subscriber)
    mem.mem_data.subscribers.append(base_subscriber)
    if(subscriber.mem_meminfo is None):
        subscriber.mem_meminfo = _meminfo_from_struct(mem)
    else:
        raise RuntimeError("Subscriber can only be linked to one Memory.")

    return l

##### subscriber signalling ####

# @njit(cache=True)
# def signal_subscribers_grow(mem, idrec):
#     for sub in mem.mem_data.subscribers:
#         sub.grow_queue.add(idrec)

@njit(cache=True)
def signal_subscribers_change(mem, idrec):
    for sub in mem.mem_data.subscribers:
        sub.change_queue.add(idrec)


##### declare #####
# @njit(u8(MemoryType,i8,i8), cache=True)
# def declare_fact_from_ptr(mem, fact_ptr, fact_num):
    

@njit(u8(MemoryType,BaseFact),cache=True)
def declare_fact(mem, fact):
    # Incref so that the fact is not freed if this is the only reference
    fact_ptr = i8(_raw_ptr_from_struct_incref(fact)) #.4ms / 10000
    t_id = resolve_t_id(mem, fact)  #.1ms / 10000


    facts = facts_for_t_id(mem.mem_data, t_id) #negligible

    # Get the next empty f_id. Retracted f_ids are recycled for cache locality.
    f_id = next_empty_f_id(mem.mem_data, facts, t_id) # .5ms / 10000


    idrec = encode_idrec(t_id,f_id,0) #negligable
    fact.idrec = idrec #negligable


    if(f_id < len(facts)): # .2ms / 10000
        if(facts.data[f_id] != 0): _decref_ptr(facts.data[f_id])
        facts.data[f_id] = fact_ptr
        
        # signal_subscribers_change(mem, idrec)
    else:
        facts.add(fact_ptr)
        # mem.mem_data.grow_queue.add(idrec)
        # signal_subscribers_grow(mem, idrec)
    mem.mem_data.change_queue.add(idrec)

    return idrec


# from cre.core import T_ID_TUPLE_FACT
# @njit(u8(MemoryType,TupleFact),cache=True)
# def declare_tuple_fact(mem, fact):
#     #Incref so that the fact is not freed if this is the only reference
#     fact_ptr = i8(_raw_ptr_from_struct_incref(fact)) #.4ms / 10000

#     t_id = T_ID_TUPLE_FACT#resolve_t_id(mem, fact.fact_num)  #.1ms / 10000
#     facts = facts_for_t_id(mem.mem_data, t_id) #negligible
#     f_id = next_empty_f_id(mem.mem_data, facts, t_id) # .5ms / 10000


#     idrec = encode_idrec(t_id,f_id,0) #negligable
#     fact.idrec = idrec #negligable


#     if(f_id < len(facts)): # .2ms / 10000
#         if(facts.data[f_id] != 0): _decref_ptr(facts.data[f_id])
#         facts.data[f_id] = fact_ptr
        
#         # signal_subscribers_change(mem, idrec)
#     else:
#         facts.add(fact_ptr)
#         # mem.mem_data.grow_queue.add(idrec)
#         # signal_subscribers_grow(mem, idrec)
#     mem.mem_data.change_queue.add(idrec)
#     # return idrec
    
#     return idrec

@njit(cache=True)
def declare_name(mem,name,idrec):
    mem.mem_data.names_to_idrecs[name] = idrec

@njit(cache=True)
def declare_fact_name(mem,fact,name):
    idrec = declare_fact(mem,fact)        
    declare_name(mem,name,idrec)
    return idrec

# @njit(u8(MemoryType,BaseFactunicode_type,),cache=True)
# def declare(mem,fact,name):
#     return declare_fact_name(#mem.declare(fact,name)

##### retract #####

@njit(cache=True)
def retract_by_idrec(mem,idrec):
    t_id, f_id, _ = decode_idrec(idrec) #negligible
    make_f_id_empty(mem.mem_data,i8(t_id), i8(f_id)) #3.6ms
    mem.mem_data.change_queue.add(encode_idrec(t_id, f_id, u1(0xFF)))
    # signal_subscribers_change(mem, idrec) #.8ms

@njit(cache=True)
def retract_by_name(mem,name):
    idrec = name_to_idrec(mem,name)
    retract_by_idrec(mem,idrec)
    del mem.mem_data.names_to_idrecs[name]

# @njit(cache=True)
# def retract(mem,identifier):
#     return mem.retract(identifier)

##### modify #####

# @generated_jit(cache=True)
# def modify_by_fact(mem,fact,attr,val):
    
#     return impl

@njit(cache=True)
def modify_by_idrec(mem,fact,attr,val):

    raise NotImplemented()
    #lower_setattr(fact,literally(attr),val)
    #TODO signal_subscribers w/ idrec w/ attr_ind

@njit(cache=True)
def modify(mem,fact,attr,val):
    return mem.modify(fact,attr,val)


##### all_facts_of_type #####

# @njit(cache=True)
# def all_facts_of_t_id(mem,t_id):
#     # t_id = resolve_t_id(mem,typ)
#     out = List.empty_list(typ)
#     facts = facts_for_t_id(mem.mem_data,t_id)
#     for i in range(facts.head):
#         fact_ptr = facts.data[i]
#         if(fact_ptr != 0):#u8(-1)):
#             # out.append(cast_fact(typ,b_fact))
#             out.append(_struct_from_ptr(typ,fact_ptr))
#     return out

#### Memory Overloading #####
@overload_method(MemoryTypeTemplate, "halt")
def mem_halt(self):
    def impl(self):
        self.halt_flag = u1(1)
    return impl

@overload_method(MemoryTypeTemplate, "backtrack")
def mem_backtrack(self):
    def impl(self):
        self.backtrack_flag = u1(1)
    return impl

@generated_jit(cache=True)
@overload_method(MemoryTypeTemplate, "get_fact")
def mem_get_fact(self, identifier, typ=None):
    if(isinstance(typ,types.Omitted) or typ is None):
        return_typ = BaseFact    
    else:
        return_typ = typ.instance_type
    if(isinstance(identifier, types.Integer)):
        def impl(self, identifier, typ=None):
            t_id, f_id, _ =  decode_idrec(identifier)
            facts = facts_for_t_id(self.mem_data, t_id) #negligible
            fact_ptr = facts.data[f_id]
            return _struct_from_ptr(return_typ, fact_ptr)

    elif(isinstance(identifier, unicode_type)):
        def impl(self, identifier, typ=None):
            idrec = self.mem_data.names_to_idrecs[identifier]
            t_id, f_id, _ =  decode_idrec(idrec)
            facts = facts_for_t_id(self.mem_data, t_id) #negligible
            fact_ptr = facts.data[f_id]
            return _struct_from_ptr(return_typ, fact_ptr)

    return impl

@overload_method(MemoryTypeTemplate, "add_subscriber")
def mem_add_subscriber(self, subscriber):
    if(not isinstance(subscriber,types.StructRef)): 
        raise TypingError(f"Cannot add subscriber of type '{type(fact)}'.")
    def impl(self, subscriber):
        return add_subscriber(self,subscriber)
    return impl

@generated_jit(cache=True)
@overload_method(MemoryTypeTemplate, "declare")
def mem_declare(self, fact, name=None):    
    if(isinstance(fact, Fact)):
        if(not name or isinstance(name, (types.NoneType,types.Omitted))):
            def impl(self, fact, name=None):
                return declare_fact(self,fact)
        else:
            def impl(self, fact, name=None):
                return declare_fact_name(self,fact,name)
    elif(isinstance(fact, types.BaseTuple)):
        def impl(self, fact, name=None):
            return declare_fact(self,TF(*fact))
    else:
        raise TypingError(f"Cannot declare fact of type '{type(fact)}'.")
    
    return impl

@generated_jit(cache=True)
@overload_method(MemoryTypeTemplate, "retract")
def mem_retract(self, identifier):
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
                        "mem.retract() accepts a valid fact idrec, name, or fact instance.")
    return impl

@generated_jit(cache=True)
@overload_method(MemoryTypeTemplate, "modify")
def mem_modify(self, fact, attr, val):
    if(not isinstance(fact,types.StructRef)): 
        raise TypingError(f"Modify requires a fact instance, got instance of'{type(fact)}'.")

    SentryLiteralArgs(['attr']).for_function(mem_modify).bind(self, fact, attr, val) 
    a_id = u1(list(fact.field_dict.keys()).index(attr._literal_value))

    def impl(self, fact, attr, val):
        fact_lower_setattr(fact, attr, val)
        #TODO signal_subscribers w/ idrec w/ attr_ind
        # signal_subscribers_change(mem, fact.idrec)
        t_id, f_id, _ = decode_idrec(fact.idrec)
        self.mem_data.change_queue.add(encode_idrec(t_id, f_id, a_id))
    # if(isinstance(attr, types.Literal)):
    # def impl(self, fact, attr, val):
    #     modify_by_fact(self, fact, attr, val)
    #     # fact_lower_setattr(fact, attr, val)

    return impl









fact_iterator_field_dict = {
    "mem" : MemoryType,
    "t_ids": i8[::1],
    "curr_ind": i8 ,
    "curr_t_id_ind": i8,
    "fact_type": types.Any
}
fact_iterator_fields = [(k,v) for k,v, in fact_iterator_field_dict.items()]

# FactIterator, FactIteratorType = define_structref("FactIterator",fact_iter_fields)

def gen_fact_iter_source(fact_type):
    return f'''import cloudpickle
from numba import njit, u2, types
from numba.experimental.structref import new
from cre.memory import FactIterator, FactIteratorType, fact_iterator_field_dict, MemoryType, fact_iter_next_raw_ptr, GenericFactIteratorType
from cre.utils import _struct_from_ptr, _cast_structref
fact_type = cloudpickle.loads({cloudpickle.dumps(fact_type)})
f_iter_type = FactIteratorType([(k,v) for k,v in {{**fact_iterator_field_dict ,"fact_type": types.TypeRef(fact_type)}}.items()])
f_iter_type._fact_type = fact_type


# @njit(fact_type(GenericFactIteratorType),cache=True)
# def fact_iterator_next(it):
#     return _struct_from_ptr(fact_type, fact_iter_next_raw_ptr(it))
# '''




class FactIterator(structref.StructRefProxy):
    ''' '''
    f_iter_type_cache = {}
    def __new__(cls, mem, fact_type):
        # if(not isinstance(fact_type, types.StructRef)): fact_type = fact_type.fact_type
        # if(fact_type not in cls.f_iter_type_cache):
        #     hash_code = unique_hash([fact_type])
        #     if(not source_in_cache('FactIterator', hash_code)):
        #         source = gen_fact_iter_source(fact_type)
        #         source_to_cache('FactIterator', hash_code, source)
        #     l = import_from_cached('FactIterator', hash_code, ['fact_iterator_ctor', 'fact_iterator_next'])

        #     fact_iterator_next = cls.f_iter_type_cache[fact_type] = l['fact_iterator_next']
        # else:
        #     fact_iterator_next  = cls.f_iter_type_cache[fact_type]
        
        # self = fact_iterator_ctor(mem, t_ids)
        # self._fact_type = fact_type
        # self.fact_iterator_next = fact_iterator_next
        return iter_facts(mem, fact_type)
        
    def __next__(self):
        return fact_iter_next(self)#self.fact_iterator_next(self)

    def __iter__(self):
        return self

# @generated_jit(cache=True)
# def fact_iter_ctor()


from numba.types import (SimpleIteratorType,)

@structref.register
class FactIteratorType(CastFriendlyMixin, types.StructRef):
    def __str__(self):
        return f"cre.FactIterator[{self._fact_type}]"

    @property
    def iterator_type(self):
        return self

    @property
    def yield_type(self):
        return self._fact_type
    # def preprocess_fields(self, fields):
    #     return tuple((name, types.unliteral(typ)) for name, typ in fields)


define_boxing(FactIteratorType, FactIterator)
GenericFactIteratorType = FactIteratorType(fact_iterator_fields)
GenericFactIteratorType._fact_type = BaseFact

@lower_cast(FactIteratorType, GenericFactIteratorType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)

@njit(GenericFactIteratorType(MemoryType,i8[::1]),cache=True)
def generic_fact_iterator_ctor(mem, t_ids):
    st = new(GenericFactIteratorType)
    st.mem = mem
    st.t_ids = t_ids
    st.curr_ind = 0
    st.curr_t_id_ind = 0
    return st

# @generated_jit(cache=True)
# def fact_iterator_ctor(mem, typ):

@lower_builtin('getiter', FactIteratorType)
def getiter_fact_iter(context, builder, sig, args):
    return args[0].value

@overload_method(FactIteratorType,'__iter__')
def getiter(self):
    def impl(self):
        return self
    return impl

# @overload(FactIterator)
@generated_jit(cache=True)
@overload_method(MemoryTypeTemplate,'iter_facts')
def iter_facts(mem, fact_type, no_subtypes=False):
    assert isinstance(fact_type, types.TypeRef)

    fact_type = fact_type.instance_type
    fact_num = fact_type._fact_num

    hash_code = unique_hash([fact_type])
    if(not source_in_cache('FactIterator', hash_code)):
        source = gen_fact_iter_source(fact_type)
        source_to_cache('FactIterator', hash_code, source)
    it_type = import_from_cached('FactIterator', hash_code, ['f_iter_type'])['f_iter_type']
    def impl(mem, fact_type, no_subtypes=False):
        cd = mem.context_data
        if(no_subtypes):
            t_ids = np.array((cd.fact_num_to_t_id[fact_num],),dtype=np.int64)
        else:
            t_ids = cd.child_t_ids[cd.fact_num_to_t_id[fact_num]]
        _it = generic_fact_iterator_ctor(mem,t_ids)
        return _cast_structref(it_type, _it)
    return impl
    # context = mem.context



@njit(i8(GenericFactIteratorType),cache=True)
def fact_iter_next_raw_ptr(it):
    while(True):
        if(it.curr_t_id_ind >= len(it.t_ids)): raise StopIteration()
        t_id = it.t_ids[it.curr_t_id_ind]
        facts_ptr = it.mem.mem_data.facts[i8(t_id)]
        if(facts_ptr != 0):
            facts = _struct_from_ptr(VectorType, facts_ptr)
            if(it.curr_ind < len(facts)):
                ptr = facts[it.curr_ind]
                it.curr_ind +=1
                if(ptr != 0): return ptr
            else:
                it.curr_ind = 0
                it.curr_t_id_ind +=1
        else:
            it.curr_ind = 0
            it.curr_t_id_ind += 1


@generated_jit(cache=True, nopython=True)
@overload_method(MemoryTypeTemplate,'get_t_id')
def get_t_id(mem, fact_type, no_subtypes=False):
    def impl(mem, fact_type, no_subtypes=False):
        return mem.context_data.fact_num_to_t_id[fact_num]
    return impl

@generated_jit(cache=True)
@overload_method(MemoryTypeTemplate,'get_subtype_t_ids')
def get_subtype_t_ids(mem, fact_type, no_subtypes=False):
    def impl(mem, fact_type, no_subtypes=False):
        return mem.context_data.child_t_ids[cd.fact_num_to_t_id[fact_num]]
    return impl


@generated_jit(cache=True)
@overload_method(MemoryTypeTemplate,'get_facts')
def get_facts(mem, fact_type=None, no_subtypes=False):
    if(isinstance(fact_type, types.TypeRef)):
        _fact_type = fact_type.instance_type
    elif(isinstance(fact_type, types.NoneType)):
        _fact_type = BaseFact
    else:
        raise ValueError("fact_type of get_facts() must be a numba type or None.")
    
    # fact_num = getattr(_fact_type,'_fact_num', 0)

    # hash_code = unique_hash([fact_type])
    # if(not source_in_cache('FactIterator', hash_code)):
    #     source = gen_fact_iter_source(fact_type)
    #     source_to_cache('FactIterator', hash_code, source)
    # it_type = import_from_cached('FactIterator', hash_code, ['f_iter_type'])['f_iter_type']
    fact_t_id = _fact_type.t_id
    def impl(mem, fact_type=None, no_subtypes=False):
        cd = mem.context_data
        if(fact_type is None):
            t_ids = np.arange(mem.mem_data.facts.head, dtype=np.int64)
        else:
            # t_id = fact_type.t_id#cd.fact_num_to_t_id[fact_num]
            if(no_subtypes):
                t_ids = np.array((fact_t_id,),dtype=np.int64)
            else:                
                t_ids = cd.child_t_ids[fact_t_id]
        # print("<< t_ids", t_ids)
        out = List.empty_list(_fact_type)
        curr_t_id_ind = 0
        curr_ind = 0
        while(True):
            if(curr_t_id_ind >= len(t_ids)): break
            t_id = t_ids[curr_t_id_ind]

            if(t_id < len(mem.mem_data.facts)):                
                facts_ptr = mem.mem_data.facts[i8(t_id)]
                # print("facts_ptr", facts_ptr)
                if(facts_ptr != 0):
                    facts = _struct_from_ptr(VectorType, facts_ptr)
                    # print("facts", t_id, facts.head, facts.data)
                    if(curr_ind < len(facts)):
                        ptr = facts[curr_ind]
                        curr_ind +=1
                        if(ptr != 0): 
                            out.append(_struct_from_ptr(_fact_type, ptr))
                    else:
                        curr_ind = 0
                        curr_t_id_ind +=1
                else:
                    curr_ind = 0
                    curr_t_id_ind += 1
            else:
                curr_ind = 0
                curr_t_id_ind += 1
        return out
    return impl

        

@generated_jit(cache=True)
def fact_iter_next(it):
    # it_type = it.instance_type
    fact_type = it._fact_type
    def impl(it):
        ptr = fact_iter_next_raw_ptr(it)
        return _struct_from_ptr(fact_type, ptr)
    return impl


# @overload_method(MemoryTypeTemplate, "all_facts_of_type")
# def mem_all_facts_of_type(self, typ):
#     def impl(self, typ):
#         return all_facts_of_type(self,typ)

#     return impl
