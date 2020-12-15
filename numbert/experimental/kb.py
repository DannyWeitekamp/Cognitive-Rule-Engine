from numba import types, njit, guvectorize,vectorize,prange
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba import types
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple, Tuple, Array, optional
from numba.cpython.unicode import  _set_code_point
from numba.experimental import structref
from numba.extending import overload_method, intrinsic
from numbert.utils import cache_safe_exec
from numbert.core import TYPE_ALIASES, REGISTERED_TYPES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from numba.core import types, cgutils
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


from numbert.experimental.structref import define_structref
from numbert.experimental.fact import BaseFact,BaseFactType
from numbert.caching import import_from_cached, source_in_cache, source_to_cache

   
i8_arr = i8[:]
u1_arr = u1[:]
str_to_bool_dict = DictType(unicode_type,u1_arr)
two_str = UniTuple(unicode_type,2)
two_str_set = DictType(two_str,u1)

meminfo_type = types.MemInfoPointer(types.voidptr)
meminfo_list = ListType(meminfo_type)
i8_list = ListType(i8)


kb_data_fields = [
    ("fact_meminfos" , ListType(meminfo_list)),
    ("empty_f_id_stacks" , ListType(i8_list)),
    ("empty_f_id_heads" , ListType(i8)),
    ("names_to_idrecs" , DictType(unicode_type,u8)),

    ("enum_data" , DictType(unicode_type,i8_arr)),
    ("enum_consistency" , DictType(two_str,u1)),
    ("consistency_listeners" , DictType(i8, two_str_set)),
    ("consistency_listener_counter" , Array(i8, 0, "C")),
    ("unnamed_counter" , Array(i8, 0, "C")),    
]
# for x,t in kb_data_fields:

print(kb_data_fields)

# if(not source_in_cache("KnowledgeBaseData",'KnowledgeBaseData')):
#     source = gen_struct_code("KnowledgeBaseData",kb_data_fields)
#     source_to_cache("KnowledgeBaseData",'KnowledgeBaseData',source)
    
# KnowledgeBaseData, KnowledgeBaseDataTypeTemplate = import_from_cached("KnowledgeBaseData",
#     "KnowledgeBaseData",["KnowledgeBaseData","KnowledgeBaseDataTypeTemplate"]).values()
# print(KnowledgeBaseData, KnowledgeBaseDataTypeTemplate)

# KnowledgeBaseDataType = KnowledgeBaseDataTypeTemplate(fields=kb_data_fields)
KnowledgeBaseData, KnowledgeBaseDataType = define_structref("KnowledgeBaseData",kb_data_fields)

@njit(Tuple([u2,u8,u1])(u8),cache=True)
def decode_idrec(idrec):
    t_id = idrec >> 48
    f_id = (idrec >> 8) & 0x000FFFFF
    a_id = idrec & 0xF
    return (t_id, f_id, a_id)


@njit(u8(u2,u8,u1),cache=True)
def encode_idrec(t_id, f_id, a_id):
    return (t_id << 48) | (f_id << 8) | a_id

@intrinsic
def _struct_from_meminfo(typingctx, struct_type, meminfo):
    inst_type = struct_type.instance_type

    def codegen(context, builder, sig, args):
        _, meminfo = args

        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo
        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        return st._getvalue()

    sig = inst_type(struct_type, types.MemInfoPointer(types.voidptr))
    return sig, codegen


@intrinsic
def _meminfo_from_struct(typingctx, val):
    # struct_type = type(val)
    # print("struct_type", struct_type)
    
    # inst_type = struct_type.instance_type
    # print("inst_type", inst_type)

    def codegen(context, builder, sig, args):
        [td] = sig.args
        [d] = args

        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder, value=d)
        meminfo = dstruct.meminfo
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)
        # Returns the plain MemInfo
        return meminfo
        # struct_ref = cgutils.create_struct_proxy(struct_type)(
        #     context, builder, value=val)

        # return struct_ref.meminfo

    sig = meminfo_type(val,)
    return sig, codegen



# @njit(cache=True)
# def init_store_data(
#         NB_Type # Template attr
#     ):
#     data = Dict.empty(unicode_type,NB_Type)
#     return data

# @njit(cache=True)
# def declare(store_data, kb_data, name, obj):
#     store_data[name] = obj
#     signal_inconsistent(kb_data.consistency_listeners,name,"*")

# @njit(cache=True)
# def declare_unnamed(store_data, kb_data, obj):
#     name = "%" + str(kb_data.unnamed_counter.item())
#     kb_data.unnamed_counter += 1
#     store_data[name] = obj
#     signal_inconsistent(kb_data.consistency_listeners,name,"*")


# @njit(cache=True)
# def modify_attr(store_data, kb_data, name, attr, value):
#     if(name in store_data):
#         #This probably requires mutable types
#         raise NotImplemented()
#         # data[name].attr = value
#     else:
#         raise ValueError()

#     signal_inconsistent_attr(kb_data.consistency_listeners,name,attr)

# @njit(cache=True)
# def retract(store_data, kb_data, name):
#     del store_data[name]
#     signal_inconsistent(kb_data.consistency_listeners,name,"")  






# def gen_knowledge_store_aot_funcs(cc,typ,NB_Type):
#     '''Wraps jitted functions into an ahead of time compiled module
#        called from generated source code for each fact type
#     '''
#     store_data_type = DictType(unicode_type, NB_Type)

#     # @cc.export('init_store_data',store_data_type())
#     # @njit(nogil=True, fastmath=True, cache=True)
#     # def _init_store_data():
#     #   return init_store_data(NB_Type)

#     cc.export('declare',(store_data_type, KnowledgeBaseDataType, unicode_type, NB_Type))(declare)
#     cc.export('declare_unnamed',(store_data_type, KnowledgeBaseDataType, NB_Type))(declare_unnamed)
#     # cc.export('modify_attr',?(??))(modify_attr)
#     cc.export('retract',(store_data_type, KnowledgeBaseDataType, unicode_type))(retract)
   




        
# class KnowledgeStore(_BaseContextful):
#     ''' Stores KnowledgeBase data for a particular type of fact'''
#     def __init__(self, typ, kb, context=None):
#         super().__init__(context)

        
#         self.kb = kb
#         self.kb_data = kb.kb_data
#         # self.enum_data, self.enum_consistency, self.consistency_listeners, \
#         # self.consistency_listener_counter, self.unnamed_counter = self.kb_data
#         # print(self.kb_data)

#         # spec = self.context.registered_specs[typ]
#         # print(self.context.jitstructs)
#         struct = self.context.jitstructs[typ]
#         out = import_from_cached(typ,struct.hash,[
#             'init_store_data', 'declare', 'declare_unnamed', 
#             'retract'
#             ],typ).values()
#         self._init_store_data, self._declare, self._declare_unnamed, self._retract = out 

#         self.store_data = self._init_store_data()


#     def declare(self,*args):
#         if(len(args) == 2):
#             # print(self.kb_data)
#             self._declare(self.store_data,self.kb_data,args[0],args[1])
#         else:
#             self._declare_unnamed(self.store_data,self.kb_data,args[0])

#     def modify(self,name,obj):
#         raise NotImplemented()

#     def retract(self,name):
#         self._retract(name)


@njit(cache=True)
def init_kb_data(context_data):
    fact_meminfos = List.empty_list(meminfo_list)
    empty_f_id_stacks = List.empty_list(i8_list)
    empty_f_id_heads = List.empty_list(i8)
    L = max(len(context_data.attr_inds_by_type),1)
    for i in range(L):
        fact_meminfos.append(List.empty_list(meminfo_type))
        empty_f_id_stacks.append(List.empty_list(i8))
        empty_f_id_heads.append(0)

    names_to_idrecs = Dict.empty(unicode_type,u8)

    enum_data = Dict.empty(unicode_type,i8_arr)

    consistency_listeners = Dict.empty(i8, two_str_set)

    enum_consistency = Dict.empty(two_str,u1)
    consistency_listener_counter = np.zeros(1,dtype=np.int64) 
    consistency_listeners[0] = enum_consistency
    consistency_listener_counter += 1
    
    unnamed_counter = np.zeros(1,dtype=np.int64)

    return KnowledgeBaseData(fact_meminfos, empty_f_id_stacks, empty_f_id_heads, names_to_idrecs,
                            enum_data, enum_consistency, consistency_listeners,
                             consistency_listener_counter, unnamed_counter
                             )


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

class KnowledgeBase(structref.StructRefProxy):
    ''' '''
    # class KnowledgeBaseData(structref.StructRefProxy):
    def __new__(cls, context=None):
        context_data = KnowledgeBaseContext.get_context(context).context_data
        kb_data = init_kb_data(context_data)
        self = structref.StructRefProxy.__new__(cls, kb_data, context_data)
        # _BaseContextful.__init__(self,context) #Maybe want this afterall
        self.kb_data = kb_data
        self.context_data = context_data
        return self
    
    def _get_fact_type(self,x):
        x_t = type(x)
        assert hasattr(x_t, 'name'), "Can only declare namedtuples built w/ numbert.define_fact()"
        return x_t.name

    def declare(self,name,x):
        return declare(self,name,x)

    def retract(self,name):
        return retract(self,name)
    #     typ = self._get_fact_type(x)
    #     # if(typ not in self.stores):
    #     #     self.stores[typ] = KnowledgeStore(typ,self)
    #     # self.stores[typ].declare(name,x)

    def modify():

        raise NotImplemented()

    # def retract(self,name):

    #     raise NotImplemented()


@structref.register
class KnowledgeBaseTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        # This method is called by the type constructor for additional
        # preprocessing on the fields.
        # Here, we don't want the struct to take Literal types.
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

structref.define_proxy(KnowledgeBase, KnowledgeBaseTypeTemplate, ["kb_data", "context_data"])
KnowledgeBaseType = KnowledgeBaseTypeTemplate(fields=[
    ("kb_data",KnowledgeBaseDataType),
    ("context_data" , KnowledgeBaseContextDataType)])


def _get_fact_type(x):
    x_t = type(x)
    assert hasattr(x_t, 'name'), "Can only declare namedtuples built w/ numbert.define_fact()"
    return x_t.name

@njit
def make_f_id_empty(kb_data, t_id, f_id):
    '''Adds adds tracking info for an empty f_id for when a fact is retracted'''
    es_s = kb_data.empty_f_id_stacks[t_id]
    es_h = kb_data.empty_f_id_heads[t_id]
    if(es_h < len(es_s)):
        es_s[es_h] = f_id
    else:
        es_s.append(f_id)
    kb_data.empty_f_id_heads[t_id] += 1


@njit
def next_empty_f_id(kb_data,t_id):
    '''Gets the next dead f_id from retracting facts otherwise returns 
        a fresh one pointing to the end of the meminfo list'''
    es_s = kb_data.empty_f_id_stacks[t_id]
    es_h = kb_data.empty_f_id_heads[t_id]
    if(es_h <= 0):
        return len(kb_data.fact_meminfos[t_id])# a fresh f_id

    kb_data.empty_f_id_heads[t_id] = es_h = es_h - 1
    return es_s[es_h] # a recycled f_id

    

@overload_method(KnowledgeBaseTypeTemplate, "declare")
def kb_declare(self, name, fact):
    t_id = 0
    def impl(self, name, fact):
        f_id = next_empty_f_id(self.kb_data,t_id)
        meminfo = _meminfo_from_struct(fact)
        meminfos = self.kb_data.fact_meminfos[t_id]
        if(f_id < len(meminfos)):
            meminfos[f_id] = meminfo
        else:
            meminfos.append(meminfo)
        idrec = encode_idrec(t_id,f_id,0)
        self.kb_data.names_to_idrecs[name] = idrec
        b_fact = _struct_from_meminfo(BaseFactType,meminfo)
        b_fact.idrec = idrec


    return impl

@njit(cache=True)
def declare(kb,name,fact):
    return kb.declare(name,fact)


# @njit
@overload_method(KnowledgeBaseTypeTemplate, "retract")
def kb_retract(self, name):
    t_id = 0
    def impl(self, name):
        names_to_idrecs = self.kb_data.names_to_idrecs
        if(name not in names_to_idrecs):
        #     # pass
            raise KeyError("Fact not found.")
            # return
        t_id, f_id, a_id = decode_idrec(names_to_idrecs[name])
        make_f_id_empty(self.kb_data,i8(t_id), i8(f_id))
        # self.kb_data.fact_meminfos[t_id] = meminfo_type(0)
        del names_to_idrecs[name]

    return impl

@overload_method(KnowledgeBaseTypeTemplate, "all_facts_of_type")
def kb_all_facts_of_type(self, name):
    t_id = 0
    def impl(self, name):
        pass

    return impl

@njit(cache=True)
def retract(kb,name):
    return kb.retract(name)


# @overload_method(TypeRef, 'empty')
# def typeddict_empty(cls, key_type, value_type):
#     if cls.instance_type is not DictType:
#         return

#     def impl(cls, key_type, value_type):
#         return dictobject.new_dict(key_type, value_type)

#     return impl






#######Pseudo Code for KB#######

'''
#Note all fact types are going to need id_rec
#Context will need typ name->id map



class KB
   init()
      self.fact_meminfos = List(List(meminfos)), by t_id f_id -> 
      self.empty_f_ids_queue = List(List(i8)) t_id, f_id -> 
      self.names_to_idrecs = Dict()

      self.consistency_listeners 
      self.consistency_counter
      self.enum


   
'''
         


