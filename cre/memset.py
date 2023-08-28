
from numba import types, njit, guvectorize,vectorize,prange, generated_jit, literally, literal_unroll
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba import types
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, UniTuple, Tuple, Array, optional
from numba.cpython.unicode import  _set_code_point
from numba.experimental import structref
from numba.experimental.structref import new
from numba.extending import overload_method, intrinsic, overload, lower_cast, lower_builtin, SentryLiteralArgs
from numba.core.errors import RequireLiteralValue
from cre.core import TYPE_ALIASES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from numba.core import types, cgutils
from numba.core.errors import TypingError
from numba.experimental.function_type import _get_wrapper_address
from llvmlite import binding as ll

from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache
import numpy as np
import timeit
import itertools
import types as pytypes
import sys
import cloudpickle
import __main__

from cre.context import CREContextDataType, CREContext, ensure_inheritance, cre_context


# from cre.subscriber import BaseSubscriberType
from cre.structref import define_structref, define_boxing, CastFriendlyStructref
from cre.fact import Fact, BaseFact, cast_fact, get_inheritance_t_ids, resolve_deref_data_ptr
from cre.tuple_fact import TF, TupleFact
from cre.fact_intrinsics import fact_lower_setattr
from cre.utils import (cast, CastFriendlyMixin, lower_setattr, _meminfo_from_struct, decode_idrec, encode_idrec, _call_dtor,
  _ptr_from_struct_incref,  _decref_ptr, _decref_structref, _raw_ptr_from_struct_incref,  _obj_cast_codegen, _incref_structref, 
 _store, _load_ptr, DEREF_TYPE_ATTR, DEREF_TYPE_LIST, _ptr_to_data_ptr, _list_base_from_ptr, new_w_del, _store_safe)
from cre.vector import new_vector, VectorType
from cre.caching import import_from_cached, source_in_cache, source_to_cache
from cre.obj import copy_cre_obj, cre_obj_clear_refs, _iter_mbr_infos, OBJECT_MBR_ID

BASE_T_ID_STACK_SIZE = 16
BASE_F_ID_STACK_SIZE = 64
BASE_FACT_SET_SIZE = 64
BASE_CHANGE_QUEUE_SIZE = 2048
   
#### mem Data Definition ####

memset_fields = {
    # Vector<*Vector<*BaseFact>> i.e. 2D vector that holds pointers to facts
    "facts" : VectorType,

    # Context Data object 
    "context_data" : CREContextDataType,

    # Vector<*Vector<i8>> i.e. 2D vector that holds retracted f_ids
    "retracted_f_ids" : VectorType,

    "names_to_idrecs" : DictType(unicode_type,u8),

    # Vector of change idrecs 
    "change_queue" : VectorType,

    # Placeholder types to ensure that facts and fact vecs are refcounted
    "all_facts" : ListType(BaseFact),

    "all_vecs" : ListType(VectorType),
}

@structref.register
class MemSetTypeClass(CastFriendlyStructref):
    def __str__(self):
        return "cre.MemSetType"
    def __repr__(self):
        return self.__str__()

MemSetType = MemSetTypeClass([(k,v) for k,v in memset_fields.items()])






#### MemSet Definition ####

class MemSet(structref.StructRefProxy):
    ''' '''
    def __new__(cls, context=None, auto_clear_refs=False):
        context = cre_context(context)
        self = memset_ctor(context.context_data, auto_clear_refs)
        self._context = context
        self.indexers = {}
        return self

    # def __init__(self, context=None, auto_clear_refs=False):



    @property
    def context(self):
        if(not hasattr(self,'_context')):
            cd = get_context_data(self)
            self._context = cre_context(cd.name)
        return self._context

    @property
    def change_queue(self):
        return get_change_queue(self)
    
    def declare(self,fact,name=None):
        return memset_declare(self,fact,name)

    def retract(self,identifier):
        return memset_retract(self,identifier)

    def _get_indexer(self, attr):
        indexer = self.indexers.get(attr)
        if(indexer is None):
            indexer = self.indexers[attr] = Indexer(attr)
        return indexer

    def get_facts(self,typ=None, no_subtypes=False,**kwargs):
        self.context._ensure_retro_registers()

        # Implement .get_facts(attr=val)
        if(len(kwargs) > 0):
            attr, val = list(kwargs.items())[0]
            indexer = self._get_indexer(attr)
            return indexer.get_facts(self, val)

        # Implement .get_facts(type)
        if(isinstance(typ,str)):
            typ = self.context.name_to_type[typ]
        return get_facts(self, typ, no_subtypes)

    def __iter__(self):
        return iter(self.get_facts())

    def iter_facts(self,typ):
        self.context._ensure_retro_registers()
        if(isinstance(typ,str)):
            typ = self.context.name_to_type[typ]
        return iter_facts(self, typ)

    def get_fact(self,*args, **kwargs):
        # Implement .get_fact(attr=val)
        if(len(kwargs) > 0):
            attr, val = list(kwargs.items())[0]
            indexer = self._get_indexer(attr)
            return indexer.get_fact(self, val)

        # Implement .get_fact(idrec)
        identifier = args[0]
        if(isinstance(identifier, str)):
            return memset_get_fact_by_name(self, identifier)
        else:
            return memset_get_fact_by_idrec(self, identifier)

    def get_ptr(self):
        return get_ptr(self)

    def modify(self, fact, attr, val):
        return memset_modify(self,fact, attr, val)

    def _repr_helper(self,rfunc,ind="    ",sep="\n",pad='\n'):
        strs = []
        for fact in self.get_facts():
            strs.append(rfunc(fact))
        nl = "\n"
        return f'''MemSet(facts=({pad}{ind}{f"{sep}{ind}".join(strs)}{pad})'''

    def __str__(self,**kwargs):
        # from cre.utils import PrintElapse
        # with PrintElapse("__STR__"):
        return self._repr_helper(str,**kwargs)


    def __repr__(self,**kwargs):
        # from cre.utils import PrintElapse
        # with PrintElapse("__REPR__"):
        return self._repr_helper(repr,**kwargs)

    def __copy__(self):
        cpy = memset_copy(self)
        cpy._context = self._context
        cpy.indexers = {}
        return cpy

    copy = __copy__

    def clear_refs(self):
        memset_clear_refs(self)

    # def __hash__(self):
    #     hash_bytes = memset_long_hash(self)

    def long_hash(self, nchars=30):
        return memset_long_hash_str(self, nchars)
    

    def __del__(self):        
        # NOTE: This definitely has bugs
        try:
            pass
            # memset_dtor(self)
        except Exception as e:
            # If the process is ending then global variables can be sporatically None
            #   thus skip any TypeError of this sort.
            if(isinstance(e, ImportError)): return
            if(isinstance(e, TypeError) and "not callable" in str(e)): return
            
            print("An error occured when trying to clean a cre.MemSet object:\n",e)

    def as_dict(self, key_attr='idrec'):
        d = {}
        if(key_attr):
            for fact in self.get_facts():
                fd = fact.as_dict(key_attr)
                d[fd[key_attr]] = fd
        else:
            for i, fact in enumerate(self.get_facts()):
                d[i] = fact.as_dict(key_attr)
        return d


define_boxing(MemSetTypeClass, MemSet)

@njit(types.void(MemSetType),cache=True)
def memset_clear_refs(ms):
    '''Decref out data structures in ms that we explicitly incref'ed '''
    # print("MEMSET DEL!!!", )
    # ms = _struct_from_ptr(MemSetType, _raw_ptr_from_struct(_ms)-48)
    # _incref_structref(ms)
    # print(ms.facts)./f
    facts = ms.facts#_load_ptr(VectorType, _raw_ptr_from_struct(_ms))
    #Decref all declared facts and their container vectors 
    for i in range(facts.head):
        facts_ptr = facts.data[i]
        if(facts_ptr == 0): continue
        facts_i = cast(facts_ptr, VectorType)
        for j in range(facts_i.head):
            fact_ptr = facts_i.data[j]
            if(fact_ptr == 0): continue
            # print("ptr", j, fact_ptr)
            fact = cast(fact_ptr, BaseFact)
            cre_obj_clear_refs(fact)
            # _iter_mbr_infos(fact)

            # _decref_structref(fact)
            # _incref_structref(fact)
            # _call_dtor(fact)
            # _decref_ptr(fact_ptr)
        # _decref_ptr(facts_ptr)

    #Decref the inner vectors of retracted_f_ids
    # for i in range(ms.retracted_f_ids.head):
    #     ptr = ms.retracted_f_ids.data[i]
    #     _decref_ptr(ptr)
    # print("END")

# HARD CODE WARNING: Pointer misalignment if layout changes inside nrt.c.
SIZEOF_NRT_MEMINFO = 48
@njit(types.void(i8),cache=True)
def memset_data_clear_refs(ms_data_ptr):
    ms = cast(ms_data_ptr-SIZEOF_NRT_MEMINFO, MemSetType)
    memset_clear_refs(ms)


memset_data_clear_refs_addr = _get_wrapper_address(memset_data_clear_refs, types.void(i8))
ll.add_symbol("CRE_MemsetData_ClearRefs", memset_data_clear_refs_addr)

@njit(cache=True)
def expand_mem_set_types(ms, n):
    ''' Expands facts and retracted_f_ids by n.'''
    for i in range(n):
        v = new_vector(BASE_F_ID_STACK_SIZE)    
        ms.all_vecs.append(v)

        v_ptr = cast(v, i8)
        ms.retracted_f_ids.add(v_ptr)

        v = new_vector(BASE_FACT_SET_SIZE)    
        ms.all_vecs.append(v)
        
        v_ptr = cast(v, i8)
        ms.facts.add(v_ptr)

@njit(MemSetType(CREContextDataType,types.boolean), cache=True)
def memset_ctor(context_data, auto_clear_refs=False):
    if(auto_clear_refs):
        st = new_w_del(MemSetType, "CRE_MemsetData_ClearRefs")
    else:
        st = new(MemSetType)
    st.context_data = context_data
    st.facts = new_vector(BASE_T_ID_STACK_SIZE)
    st.retracted_f_ids = new_vector(BASE_T_ID_STACK_SIZE)
    st.names_to_idrecs = Dict.empty(unicode_type,u8)
    st.change_queue = new_vector(BASE_CHANGE_QUEUE_SIZE)
    st.all_facts = List.empty_list(BaseFact)
    st.all_vecs = List.empty_list(VectorType)
    L = max(len(context_data.parent_t_ids)+1,1)
    expand_mem_set_types(st,L)
    return st

@overload(MemSet)
def overload_MemSet(context_data=None, auto_clear_refs=False):
    if(context_data is None):
        raise ValueError("MemSet() must be provided context_data when instantiated in jitted context.")

    def impl(context_data=None, auto_clear_refs=False):
        if(context_data is None):
            # Compile-time error should prevent  
            raise ValueError()            
        else:
            return memset_ctor(context_data, auto_clear_refs)

    return impl

@njit(cache=True)
def get_context_data(self):
    return self.context_data

@njit(cache=True)
def get_change_queue(self):
    return self.change_queue

@njit(cache=True)
def get_ptr(self):
    return cast(self, i8)

@njit(cache=True)
def facts_for_t_id(ms,t_id):
    L = len(ms.facts)
    if(t_id >= L):
        expand_mem_set_types(ms, (t_id+1)-L)
    # print("Declare", t_id, mem_data.facts[t_id])
    return cast(ms.facts[u8(t_id)], VectorType)

@njit(cache=True)
def fact_at_f_id(typ, t_id_facts,f_id):
    ptr = t_id_facts.data[f_id]
    if(ptr != 0):
        return cast(ptr, typ)
    else:
        return None

@njit(cache=True)
def retracted_f_ids_for_t_id(ms,t_id):
    return cast(ms.retracted_f_ids[u8(t_id)], VectorType)

#### Helper Functions ####

@njit(cache=True)
def make_f_id_empty(ms, t_id, f_id):
    '''Removes fact at t_id, f_id freeing up its f_id. f_id is added 
        to the set of rectracted f_ids so it can be recycled.'''
    facts = facts_for_t_id(ms,t_id)
    fact_ptr = facts[f_id]
    if(fact_ptr != 0):
        retracted_f_ids_for_t_id(ms,t_id).add(f_id)
        # _decref_ptr(fact_ptr)
    
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
    if(t_id >= len(cd.child_t_ids) or len(cd.child_t_ids[u8(t_id)])==0):
        inh_t_ids = get_inheritance_t_ids(fact)
        parent_t_id = -1
        for _t_id in inh_t_ids:
            cd.unhandled_retro_registers.append(u2(_t_id))
            ensure_inheritance(cd, _t_id, parent_t_id)
            parent_t_id = i8(_t_id)

        

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
    # fact_ptr = i8(_raw_ptr_from_struct_incref(fact)) #.4ms / 10000
    ms.all_facts.append(fact)
    fact_ptr = cast(fact, i8) #.4ms / 10000
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
    # if(facts.data[f_id] != 0): _decref_ptr(facts.data[f_id])
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

    # fact = memset_get_fact(ms, idrec)
    # fact_lower_setattr(fact,'idrec',encode_idrec(t_id,0,u1(-1)))

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


#### get_fact ####

@njit(BaseFact(MemSetType,u8), cache=True)
def memset_get_fact_by_idrec(self, idrec):
    t_id, f_id, _ =  decode_idrec(idrec)
    # if(t_id != typ_t_id): raise ValueError(err_msg)
    facts = facts_for_t_id(self, t_id) #negligible
    fact_ptr = facts.data[f_id]
    return cast(fact_ptr, BaseFact)

@njit(BaseFact(MemSetType,unicode_type), cache=True)
def memset_get_fact_by_name(self, name):
    idrec = self.names_to_idrecs[name]
    t_id, f_id, _ =  decode_idrec(idrec)
    # if(t_id != typ_t_id): raise ValueError(err_msg)
    facts = facts_for_t_id(self, t_id) #negligible
    fact_ptr = facts.data[f_id]
    return cast(fact_ptr, BaseFact)

@overload_method(MemSetTypeClass, "get_fact")
def memset_get_fact(self, identifier, typ=None):
    if(isinstance(typ,types.Omitted) or typ is None):
        return_typ = BaseFact    
    else:
        return_typ = typ.instance_type
    if(isinstance(identifier, types.Integer)):
        def impl(self, identifier, typ=None):
            return cast(memset_get_fact_by_idrec(self, identifier), return_typ)

    elif(isinstance(identifier, unicode_type)):
        def impl(self, identifier, typ=None):
            return cast(memset_get_fact_by_name(self, identifier),return_typ)
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



@generated_jit(nopython=True, cache=True)
def memset_modify_w_deref_infos(self, fact, deref_infos, val):
    if(not isinstance(fact,types.StructRef)): 
        raise TypingError(f"Modify requires a fact instance, got instance of'{type(fact)}'.")
    # print("<<", deref_infos)
    val_type = val
    def impl(self, fact, deref_infos, val):
        if(len(deref_infos) == 0):
            raise ValueError("Cannot setattr with empty deref_infos.")

        final_deref_info = deref_infos[-1]
        a_id = u1(final_deref_info.a_id)

        field_data_ptr = resolve_deref_data_ptr(fact, deref_infos)
        t_id, f_id, _ = decode_idrec(fact.idrec)

        field_ptr = field_data_ptr#_load_ptr(val_type, field_data_ptr)
    
        # incref new value
        _incref_structref(val)

        # decref old value (must be last in case new value is old value)
        _decref_ptr(field_ptr)
        # print("D")

        if(val is None):
            # Attr case
            if(final_deref_info.type == DEREF_TYPE_ATTR):
                # print("IS NONE")
                _store(i8, field_data_ptr, 0)

            # List case
            else:
                # TODO: some issue with this 
                raise ValueError("List cannot be set to None")
                new_lst = List.empty_list(val_type) 
                _store(i8, field_data_ptr, _ptr_from_struct_incref(new_lst))
        else:
            # print("IS normal", val)
            _store(val_type, field_data_ptr, val)

        # print("Z")

        self.change_queue.add(encode_idrec(t_id, f_id, a_id))

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
from cre.utils import cast
fact_type = cloudpickle.loads({cloudpickle.dumps(fact_type)})
f_iter_type = FactIteratorType([(k,v) for k,v in {{**fact_iterator_field_dict ,"fact_type": types.TypeRef(fact_type)}}.items()])
f_iter_type._fact_type = fact_type


# @njit(fact_type(GenericFactIteratorType),cache=True)
# def fact_iterator_next(it):
#     return cast(fact_iter_next_raw_ptr(it), fact_type)
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
            t_ids = np.array((cd.fact_num_to_t_id[fact_num],),dtype=np.uint16)
        else:
            t_ids = cd.child_t_ids[cd.fact_num_to_t_id[fact_num]]
        _it = generic_fact_iterator_ctor(ms,t_ids)
        return cast(_it, it_type)
    return impl
    # context = mem.context



@njit(i8(GenericFactIteratorType),cache=True)
def fact_iter_next_raw_ptr(it):
    while(True):
        if(it.curr_t_id_ind >= len(it.t_ids)): raise StopIteration()
        t_id = it.t_ids[it.curr_t_id_ind]
        facts_ptr = it.memset.facts[i8(t_id)]
        if(facts_ptr != 0):
            facts = cast(facts_ptr, VectorType)
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
            t_ids = np.arange(len(ms.facts), dtype=np.uint16)
        else:
            # t_id = fact_type.t_id#cd.fact_num_to_t_id[fact_num]
            if(no_subtypes):
                t_ids = np.array((fact_t_id,),dtype=np.uint16)
            else:                
                t_ids = cd.child_t_ids[fact_t_id]
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
                    facts = cast(facts_ptr, VectorType)
                    # print("facts", t_id, facts.head, facts.data)
                    if(curr_ind < len(facts)):
                        ptr = facts[curr_ind]
                        curr_ind +=1
                        if(ptr != 0): 
                            out.append(cast(ptr, _fact_type))
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
        return cast(ptr, fact_type)
    return impl


# ----------------------------------------------------------------------
# : Indexer - Allows for .get_facts(attr=val)

base_indexer_fields = {
    # Context Data object 
    "head" : i8,
    "mapping" : types.Any,
    "attr" : types.Any,
    "types_t_ids" : types.Any,
}

@structref.register
class IndexerTypeClass(CastFriendlyStructref):
    pass

class Indexer(structref.StructRefProxy):
    def __new__(cls, attr):
        base_types = get_base_types_with_attr(attr)
        attr_type = base_types[0][1]
        ctor, indexer_type = indexer_ctor_impl(base_types, attr)
        self = ctor()

        indexer_get_facts.compile(ListType(BaseFact)(indexer_type, MemSetType, attr_type))
        indexer_get_fact.compile(BaseFact(indexer_type, MemSetType, attr_type))

        self.get_facts_ep = indexer_get_facts.overloads[(indexer_type, MemSetType, attr_type)].entry_point
        self.get_fact_ep = indexer_get_fact.overloads[(indexer_type, MemSetType, attr_type)].entry_point
        return self

    def get_facts(self, memset, val):
        return self.get_facts_ep(self, memset, val)


    def get_fact(self, memset, val):
        return self.get_fact_ep(self, memset, val)




define_boxing(IndexerTypeClass, Indexer)

def get_base_types_with_attr(attr):
    context = cre_context()
    context.children_of
    base_types = {}

    for fact_type, parents in context.parents_of.items():
        if(isinstance(fact_type, Fact)):
            base_type = fact_type
            if(not hasattr(base_type,'spec')): continue
            while(hasattr(base_type,'parent_type') and
                  hasattr(base_type.parent_type, 'spec') and
                  attr in base_type.parent_type.spec):
                base_type = base_type.parent_type
            if(attr in base_type.spec and base_type not in base_types):
                val_type = base_type.spec[attr]['type']
                # child_t_ids = context.context_data.child_t_ids[base_type.t_id]
                # child_t_ids = types.literal(tuple([t_id for t_id in child_t_ids]))
                base_types[base_type] = val_type

    out = Tuple(tuple([Tuple((bt, vt)) for bt,vt in base_types.items()]))
    return out

vec2_type = Tuple((VectorType,VectorType))

# TODO: Should probably see if there is a way to make more generic 
#  version of Indexer
# def new_indexer(attr):
#     base_types = get_base_types_with_attr(attr)
#     return indexer_ctor_impl(base_types, attr)()

_indexer_ctor_impls = {}
def indexer_ctor_impl(base_types, attr):
    # SentryLiteralArgs(['attr']).for_function(indexer_ctor).bind(base_types, attr)
    # _base_types = base_types.instance_type
    tup = (base_types, attr)
    if(tup not in _indexer_ctor_impls):
        if(len(base_types) == 0):
            context = cre_context()
            raise ValueError(f"No fact_types with attribute {attr!r} defined in context {context.name!r}.")
        val_type = None
        for _,_val_type in base_types:    
            if(val_type is not None and _val_type != val_type):
                raise ValueError("Cannot use indexer on attribute that differs in type across two fact types.")
            val_type = _val_type

        indexer_fields = {
            **base_indexer_fields, 
            "mapping" : DictType(val_type, vec2_type),
            "inv_mapping" : DictType(u8, val_type),
            "attr" : types.literal(attr),
            "base_types" : types.TypeRef(base_types),
        }
        # print(base_types)
        indexer_type = IndexerTypeClass([(k,v) for k,v in indexer_fields.items()])
        # print(indexer_type)        
        indexer_type._base_types = base_types

        long_hash = unique_hash(tup)

        if(not source_in_cache('MemSetIndexer', long_hash)):
            types_bytes = cloudpickle.dumps((indexer_type, val_type))
            source = f'''
import cloudpickle
from numba import u8, types, njit 
from numba.typed import Dict
from numba.experimental.structref import new
from cre.vector import VectorType

vec2_type = types.Tuple((VectorType,VectorType))
indexer_type, val_type = cloudpickle.loads({types_bytes})
@njit(indexer_type(), cache=True)
def ctor():
    st = new(indexer_type)
    st.head = 0
    st.mapping = Dict.empty(val_type, vec2_type)
    st.inv_mapping = Dict.empty(u8, val_type)
    return st
'''
            source_to_cache('MemSetIndexer', long_hash, source)
        ctor = import_from_cached('MemSetIndexer', long_hash, ['ctor'])['ctor']
        # print(ctor)
        _indexer_ctor_impls[tup] = (ctor, indexer_type)
    return _indexer_ctor_impls[tup]



from cre.fact_intrinsics import fact_lower_getattr
@njit(cache=True)
def indexer_update_declare(self, change_event, fact): 
    val = fact_lower_getattr(fact, self.attr)

    if(val not in self.mapping):
        self.mapping[val] = (new_vector(1), new_vector(1))
    idrec_vec, hole_vec = self.mapping[val]

    # Don't re-add on retract-declare pair
    if(not change_event.was_retracted):
        if(len(hole_vec) > 0):
            idrec_vec[hole_vec.pop()] = change_event.idrec
        else:
            idrec_vec.add(change_event.idrec)

    self.inv_mapping[change_event.idrec] = val

@njit(cache=True)
def indexer_update_retract(self, change_event): 
    val = self.inv_mapping[change_event.idrec]
    idrec_vec, hole_vec = self.mapping[val]
    index = np.nonzero(idrec_vec.data==change_event.idrec)[0][0]
    hole_vec.add(index)
    idrec_vec[index] = 0

    del self.inv_mapping[change_event.idrec]


from cre.change_event import accumulate_change_events
@generated_jit(cache=True, nopython=True)
def indexer_update(self, memset):
    context_data = cre_context().context_data
    base_types_t_ids = []
    for bt,vt in self._base_types:
        base_types_t_ids.append((bt,tuple(context_data.child_t_ids[bt.t_id])))
    base_types_t_ids = tuple(base_types_t_ids)
    def impl(self, memset):
        cq = memset.change_queue
        if(self.head == cq.head): return

        for change_event in accumulate_change_events(cq, self.head, cq.head):
            for tup in literal_unroll(base_types_t_ids):
                base_type, t_ids = tup
                if(change_event.t_id in t_ids):
                    if(change_event.was_declared):
                        facts = facts_for_t_id(memset, change_event.t_id)
                        fact = cast(facts[change_event.f_id], base_type)
                        indexer_update_declare(self, change_event, fact)

                    elif(change_event.was_retracted):
                        indexer_update_retract(self, change_event)
                        
        self.head = cq.head
    return impl

@njit(cache=True)
def indexer_get_facts(self, memset, val):
    if(self.head != memset.change_queue.head):
        indexer_update(self, memset)
    facts = List.empty_list(BaseFact)
    if(val in self.mapping):
        idrecs, _ = self.mapping[val]
        for i in range(idrecs.head):
            idrec = u8(idrecs.data[i])
            if(idrec != 0):
                facts.append(memset_get_fact_by_idrec(memset,idrec))
    return facts

@njit(cache=True)
def indexer_get_fact(self, memset, val):
    if(self.head != memset.change_queue.head):
        indexer_update(self, memset)
    if(val in self.mapping):
        idrecs, _ = self.mapping[val]
        for i in range(idrecs.head):
            idrec = u8(idrecs.data[i])
            if(idrec != 0):
                return memset_get_fact_by_idrec(memset,idrec)
    raise KeyError("No fact found.")





# ----------------------------------------------------------------------
# : memset_copy - Allows for .get_facts(attr=val)


@njit(cache=True)
def memset_copy(self):
    new = memset_ctor(self.context_data, False)    

    new_facts = List.empty_list(BaseFact)
    ptr_map = Dict.empty(i8,i8)
    for i in range(self.facts.head):
        facts_ptr = self.facts.data[i]
        if(facts_ptr != 0):
            facts = cast(facts_ptr, VectorType)
            for j in range(facts.head):
                fact_ptr = facts.data[j]
                if(fact_ptr != 0):
                    fact = cast(fact_ptr, BaseFact)
                    new_fact = copy_cre_obj(fact)
                    ptr_map[fact_ptr] = cast(new_fact,i8)
                    new.declare(new_fact)
                    new_facts.append(new_fact)

    # Copy any fact members as weak pointers
    for new_fact in new_facts:
        for t_id, m_id, data_ptr in _iter_mbr_infos(new_fact):
            if(m_id == OBJECT_MBR_ID):
                old_fact_ptr = _load_ptr(i8, data_ptr)
                if(old_fact_ptr != 0 and old_fact_ptr in ptr_map):
                    new_fact = cast(ptr_map[old_fact_ptr],BaseFact)
                    _store_safe(BaseFact, data_ptr, new_fact)

    # TODO should copy any lists of facts
    
    return new


from cre.hashing import accum_item_hash
from numba.cpython.hashing import _Py_hash_t, _PyHASH_XXPRIME_5
from numba import u8
from cre.type_conv import byte_string_to_base64

@njit(cache=True, locals={"i": u8, "j": u8, "c": u8, "nc": u8, "mx": u8,"rest": u8})
def memset_long_hash_bytes(ms, byte_width=24):
    facts = get_facts(ms, None)
    fact_hashes = np.empty(len(facts), dtype=np.uint64)
    for i, fact in enumerate(facts):
        fact_hashes[i] = hash(fact)

    hash_bytes = np.empty(byte_width, dtype=np.uint8)
    buffer0 = np.empty(1, dtype=np.uint64)
    buffer = np.empty(8, dtype=np.uint8)
    

    # Fill hash_bytes with starting stuff 
    acc = accum_item_hash(u8(_PyHASH_XXPRIME_5), len(facts))
    for c in range(0,byte_width,8):        
        buffer0[0] = u8(accum_item_hash(acc, c))
        mx = min(byte_width, c+8)
        hash_bytes[c:mx] = buffer0.view(np.uint8)[:mx-c]

    inds = np.argsort(fact_hashes)
    c = 0
    for j, i in enumerate(inds):
        fact = facts[i]
        fact_hash = fact_hashes[i]

        nc = c+8
        mx = min(len(hash_bytes), nc)
        rest = nc-mx
        l = 8-rest

        # lower bit
        buffer[:l] = hash_bytes[c:mx]

        # upper bit that might have been cut off
        if(rest):
            buffer[l:8] = hash_bytes[:rest]

        acc = buffer.view(np.uint64)[0]
        acc = accum_item_hash(acc, j)
        acc = accum_item_hash(acc, fact_hash)

        buffer0[:] = acc
        buffer = buffer0.view(np.uint8)
        hash_bytes[c:mx] = buffer[:l]

        if(rest):
            hash_bytes[:rest] = buffer[l:8]
        
        c = (c +8) % byte_width
    return hash_bytes

@njit(cache=True)
def memset_long_hash_str(ms, n_char=30):
    byte_width = u8(np.ceil((n_char * 6) / 8.0))

    hash_bytes = memset_long_hash_bytes(ms, byte_width)
    s = byte_string_to_base64(hash_bytes, n_char)

    return s



