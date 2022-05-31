
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


# from cre.subscriber import BaseSubscriberType
from cre.structref import define_structref, define_boxing, CastFriendlyStructref
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

memset_fields = {
    # Context Data object 
    "context_data" : CREContextDataType,

    # Vector<*Vector<*BaseFact>> i.e. 2D vector that holds pointers to facts
    "facts" : VectorType,

    # Vector<*Vector<i8>> i.e. 2D vector that holds retracted f_ids
    "retracted_f_ids" : VectorType,

    "names_to_idrecs" : DictType(unicode_type,u8),

    # Vector of change idrecs 
    "change_queue" : VectorType
}

@structref.register
class MemSetTypeClass(CastFriendlyStructref):
    pass

MemSetType = MemSetTypeClass([(k,v) for k,v in memset_fields.items()])


@njit(cache=True)
def memset_ctor(context_data):
    st = new(MemSetType)
    st.context_data = context_data
    st.facts = new_vector(BASE_T_ID_STACK_SIZE)
    st.retracted_f_ids = new_vector(BASE_T_ID_STACK_SIZE)
    st.names_to_idrecs = Dict.empty(unicode_type,u8)
    st.change_queue = new_vector(BASE_CHANGE_QUEUE_SIZE)
    L = max(len(context_data.parent_t_ids)+1,1)
    expand_mem_set_types(st,L)
    return st

@njit(cache=True)
def expand_mem_set_types(ms, n):
    ''' Expands facts and retracted_f_ids by n.'''
    for i in range(n):
        v = new_vector(BASE_F_ID_STACK_SIZE)    
        v_ptr = _raw_ptr_from_struct_incref(v)
        ms.retracted_f_ids.add(v_ptr)

        v = new_vector(BASE_FACT_SET_SIZE)    
        v_ptr = _raw_ptr_from_struct_incref(v)
        ms.facts.add(v_ptr)
        

@njit(cache=True)
def memset_dtor(ms):
    '''Decref out data structures in ms that we explicitly incref'ed '''

    #Decref all declared facts and their container vectors 
    for i in range(ms.facts.head):
        facts_ptr = ms.facts.data[i]
        facts = _struct_from_ptr(VectorType, facts_ptr)
        for j in range(facts.head):
            fact_ptr = facts.data[j]
            _decref_ptr(fact_ptr)
        _decref_ptr(facts_ptr)

    #Decref the inner vectors of retracted_f_ids
    for i in range(ms.retracted_f_ids.head):
        ptr = ms.retracted_f_ids.data[i]
        _decref_ptr(ptr)

#### MemSet Definition ####

class MemSet(structref.StructRefProxy):
    ''' '''
    def __new__(cls, context=None):
        context = cre_context(context)
        context_data = context.context_data
        self = memset_ctor(context_data)
        self.context_data = context_data
        self.context = context
        return self
    
    def declare(self,fact,name=None):
        return memset_declare(self,fact,name)

    def retract(self,identifier):
        return memset_retract(self,identifier)

    def get_facts(self,typ=None, no_subtypes=False):
        self.context._ensure_retro_registers()
        if(isinstance(typ,str)):
            typ = self.context.name_to_type[typ]

        return get_facts(self, typ, no_subtypes)

    def iter_facts(self,typ):
        self.context._ensure_retro_registers()
        if(isinstance(typ,str)):
            typ = self.context.name_to_type[typ]
        return iter_facts(self, typ)

    def get_fact(self,idrec):
        return get_fact(self, idrec)

    def modify(self, fact, attr, val):
        return memset_modify(self,fact, attr, val)

    def _repr_helper(self,rfunc,ind="    ",sep="\n",pad='\n'):
        strs = []
        for fact in self.get_facts():
            strs.append(str(fact))
        nl = "\n"
        return f'''MemSet(facts=({pad}{ind}{f"{sep}{ind}".join(strs)}{pad})'''

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
            memset_dtor(self)
        except Exception as e:
            # If the process is ending then global variables can be sporatically None
            #   thus skip any TypeError of this sort.
            if(isinstance(e, ImportError)): return
            if(isinstance(e, TypeError) and "not callable" in str(e)): return
            
            print("An error occured when trying to clean a cre.MemSet object:\n",e)

define_boxing(MemSetTypeClass, MemSet)


@overload(MemSet)
def overload_MemSet(context_data=None, mem_data=None):
    if(context_data is None):
        raise ValueError("MemSet() must be provided context_data when instantiated in jitted context.")

    def impl(context_data=None, mem_data=None):
        if(context_data is None):
            # Compile-time error should prevent  
            raise ValueError()            
        else:
            return memset_ctor(context_data)

    return impl


@njit(cache=True)
def facts_for_t_id(ms,t_id):
    L = len(ms.facts)
    if(t_id >= L):
        expand_mem_set_types(ms, (t_id+1)-L)
    # print("Declare", t_id, mem_data.facts[t_id])
    return _struct_from_ptr(VectorType, ms.facts[t_id])

@njit(cache=True)
def fact_at_f_id(typ, t_id_facts,f_id):
    ptr = t_id_facts.data[f_id]
    if(ptr != 0):
        return _struct_from_ptr(typ, ptr)
    else:
        return None

@njit(cache=True)
def retracted_f_ids_for_t_id(ms,t_id):
    return _struct_from_ptr(VectorType, ms.retracted_f_ids[t_id])

#### Helper Functions ####

@njit(cache=True)
def make_f_id_empty(ms, t_id, f_id):
    '''Removes fact at t_id, f_id freeing up its f_id. f_id is added 
        to the set of rectracted f_ids so it can be recycled.'''
    facts = facts_for_t_id(ms,t_id)
    fact_ptr = facts[f_id]
    if(fact_ptr != 0):
        retracted_f_ids_for_t_id(ms,t_id).add(f_id)
        _decref_ptr(fact_ptr)
    
    facts[f_id] = 0


@njit(cache=True)
def next_empty_f_id(ms,facts,t_id):
    '''Gets the next dead f_id from retracting facts otherwise returns 
        a fresh one pointing to the end of the meminfo list'''
    f_id_vec = retracted_f_ids_for_t_id(ms,t_id)
    if(f_id_vec.head <= 0):
        return len(facts) # fresh f_id.

    return f_id_vec.pop() # recycled f_id.


@njit(u2(MemSetType, BaseFact),cache=True)
def resolve_t_id(ms, fact):
    '''Gets a t_id from a fact, and ensure that the MemSet can hold
        facts with that t_id. Also ensures that ms.context_data's
        inheritance structures are up to date, so that we can
        do get_facts(Type) and get all subtypes of Type.    
    '''
    cd = ms.context_data

    # Get the fact's t_id
    t_id, _, _ = decode_idrec(fact.idrec)

    # Ensure that inheritance structures are up to date
    if(t_id >= len(cd.child_t_ids) or len(cd.child_t_ids[t_id])==0):
        inh_t_ids = get_inheritance_t_ids(fact)
        parent_t_id = -1
        for _t_id in inh_t_ids:
            ensure_inheritance(cd, _t_id, parent_t_id)
            parent_t_id = i8(_t_id)

        cd.has_unhandled_retro_register = True

    # Ensure that the data structures in MemSet are long enough to index t_id.
    L = len(ms.facts)
    if(t_id >= L):
        expand_mem_set_types(ms, 1+L-t_id)

    return t_id

@njit(cache=True)
def name_to_idrec(ms,name):
    names_to_idrecs = ms.names_to_idrecs
    if(name not in names_to_idrecs):
        raise KeyError("Fact not found.")
    return names_to_idrecs[name]



@njit(u8(MemSetType,BaseFact),cache=True)
def declare_fact(ms, fact):
    '''Declares a fact to a MemSet'''
    
    # Acquire a reference to the fact and get it's t_id
    fact_ptr = i8(_raw_ptr_from_struct_incref(fact)) #.4ms / 10000
    t_id = resolve_t_id(ms, fact)  #.1ms / 10000
    
    # Get the facts Vector for facts with t_id  
    facts = facts_for_t_id(ms, t_id) #negligible

    # Get the next empty f_id. Retracted f_ids are recycled for cache locality.
    f_id = next_empty_f_id(ms, facts, t_id) # .5ms / 10000

    # Assign a new idrec to fact
    idrec = encode_idrec(t_id,f_id,0) #negligable
    fact.idrec = idrec #negligable

    # Make sure facts is big enough 
    while(f_id >= len(facts.data)): # ??ms / 10000
        facts.expand()

    # Put the fact into facts at f_id. Release any old facts.
    if(facts.data[f_id] != 0): _decref_ptr(facts.data[f_id])
    facts.set_item_safe(f_id, fact_ptr)
    
    ms.change_queue.add(idrec)

    return idrec

@njit(cache=True)
def declare_name(ms,name,idrec):
    ms.names_to_idrecs[name] = idrec

@njit(cache=True)
def declare_fact_name(ms,fact,name):
    idrec = declare_fact(ms,fact)        
    declare_name(ms,name,idrec)
    return idrec

##### retract #####

@njit(cache=True)
def retract_by_idrec(ms,idrec):
    t_id, f_id, _ = decode_idrec(idrec) #negligible
    make_f_id_empty(ms,i8(t_id), i8(f_id)) #3.6ms
    ms.change_queue.add(encode_idrec(t_id, f_id, u1(0xFF)))
    # signal_subscribers_change(mem, idrec) #.8ms

@njit(cache=True)
def retract_by_name(ms,name):
    idrec = name_to_idrec(ms,name)
    retract_by_idrec(ms,idrec)
    del ms.names_to_idrecs[name]


##### modify #####

@njit(cache=True)
def modify_by_idrec(ms,idrec,attr,val):
    raise NotImplemented()
    #lower_setattr(fact,literally(attr),val)
    #TODO signal_subscribers w/ idrec w/ attr_ind

@njit(cache=True)
def modify(ms,fact,attr,val):
    return ms.modify(fact,attr,val)


#### get_facts ####

@generated_jit(cache=True)
@overload_method(MemSetTypeClass, "get_fact")
def mem_get_fact(self, identifier, typ=None):
    if(isinstance(typ,types.Omitted) or typ is None):
        return_typ = BaseFact    
    else:
        return_typ = typ.instance_type
    if(isinstance(identifier, types.Integer)):
        def impl(self, identifier, typ=None):
            t_id, f_id, _ =  decode_idrec(identifier)
            facts = facts_for_t_id(self, t_id) #negligible
            fact_ptr = facts.data[f_id]
            return _struct_from_ptr(return_typ, fact_ptr)

    elif(isinstance(identifier, unicode_type)):
        def impl(self, identifier, typ=None):
            idrec = self.names_to_idrecs[identifier]
            t_id, f_id, _ =  decode_idrec(idrec)
            facts = facts_for_t_id(self, t_id) #negligible
            fact_ptr = facts.data[f_id]
            return _struct_from_ptr(return_typ, fact_ptr)

    return impl

#### overloads for declare, rectract, modify ####

@generated_jit(cache=True)
@overload_method(MemSetTypeClass, "declare")
def memset_declare(self, fact, name=None):    
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
@overload_method(MemSetTypeClass, "retract")
def memset_retract(self, identifier):
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
@overload_method(MemSetTypeClass, "modify")
def memset_modify(self, fact, attr, val):
    if(not isinstance(fact,types.StructRef)): 
        raise TypingError(f"Modify requires a fact instance, got instance of'{type(fact)}'.")

    SentryLiteralArgs(['attr']).for_function(memset_modify).bind(self, fact, attr, val) 
    a_id = u1(list(fact.field_dict.keys()).index(attr._literal_value))

    def impl(self, fact, attr, val):
        fact_lower_setattr(fact, attr, val)
        #TODO signal_subscribers w/ idrec w/ attr_ind
        # signal_subscribers_change(mem, fact.idrec)
        t_id, f_id, _ = decode_idrec(fact.idrec)
        self.change_queue.add(encode_idrec(t_id, f_id, a_id))
    # if(isinstance(attr, types.Literal)):
    # def impl(self, fact, attr, val):
    #     modify_by_fact(self, fact, attr, val)
    #     # fact_lower_setattr(fact, attr, val)

    return impl









fact_iterator_field_dict = {
    "memset" : MemSetType,
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
from cre.memset import FactIterator, FactIteratorType, fact_iterator_field_dict, fact_iter_next_raw_ptr, GenericFactIteratorType
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
    def __new__(cls, ms, fact_type):
        return iter_facts(ms, fact_type)
        
    def __next__(self):
        return fact_iter_next(self)#self.fact_iterator_next(self)

    def __iter__(self):
        return self


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

@njit(GenericFactIteratorType(MemSetType,i8[::1]),cache=True)
def generic_fact_iterator_ctor(ms, t_ids):
    st = new(GenericFactIteratorType)
    st.memset = ms
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
@overload_method(MemSetTypeClass,'iter_facts')
def iter_facts(ms, fact_type, no_subtypes=False):
    assert isinstance(fact_type, types.TypeRef)

    fact_type = fact_type.instance_type
    fact_num = fact_type._fact_num

    hash_code = unique_hash([fact_type])
    if(not source_in_cache('FactIterator', hash_code)):
        source = gen_fact_iter_source(fact_type)
        source_to_cache('FactIterator', hash_code, source)
    it_type = import_from_cached('FactIterator', hash_code, ['f_iter_type'])['f_iter_type']
    def impl(ms, fact_type, no_subtypes=False):
        cd = ms.context_data
        if(no_subtypes):
            t_ids = np.array((cd.fact_num_to_t_id[fact_num],),dtype=np.int64)
        else:
            t_ids = cd.child_t_ids[cd.fact_num_to_t_id[fact_num]]
        _it = generic_fact_iterator_ctor(ms,t_ids)
        return _cast_structref(it_type, _it)
    return impl
    # context = mem.context



@njit(i8(GenericFactIteratorType),cache=True)
def fact_iter_next_raw_ptr(it):
    while(True):
        if(it.curr_t_id_ind >= len(it.t_ids)): raise StopIteration()
        t_id = it.t_ids[it.curr_t_id_ind]
        facts_ptr = it.memset.facts[i8(t_id)]
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


# @generated_jit(cache=True, nopython=True)
# @overload_method(MemoryTypeTemplate,'get_t_id')
# def get_t_id(mem, fact_type, no_subtypes=False):
#     def impl(mem, fact_type, no_subtypes=False):
#         return mem.context_data.fact_num_to_t_id[fact_num]
#     return impl

# @generated_jit(cache=True)
# @overload_method(MemoryTypeTemplate,'get_subtype_t_ids')
# def get_subtype_t_ids(mem, fact_type, no_subtypes=False):
#     def impl(mem, fact_type, no_subtypes=False):
#         return mem.context_data.child_t_ids[cd.fact_num_to_t_id[fact_num]]
#     return impl


@generated_jit(cache=True)
@overload_method(MemSetTypeClass,'get_facts')
def get_facts(ms, fact_type=None, no_subtypes=False):
    if(isinstance(fact_type, types.TypeRef)):
        _fact_type = fact_type.instance_type
    elif(isinstance(fact_type, types.NoneType)):
        _fact_type = BaseFact
    else:
        raise ValueError("fact_type of get_facts() must be a numba type or None.")
    
    get_all = fact_type is None or _fact_type is BaseFact

    fact_t_id = _fact_type.t_id
    def impl(ms, fact_type=None, no_subtypes=False):
        cd = ms.context_data
        if(get_all):
            t_ids = np.arange(len(ms.facts), dtype=np.int64)
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

            if(t_id < len(ms.facts)):                
                facts_ptr = ms.facts[i8(t_id)]
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
