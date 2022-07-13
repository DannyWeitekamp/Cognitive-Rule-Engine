import numpy as np
from numba import njit, generated_jit, types, literal_unroll, u8, i8, f8, u1, u2
from numba.types import unicode_type,  intp, Tuple,  Tuple, DictType, ListType
from numba.typed import Dict, List
from numba.experimental.structref import new
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.cre_object import CREObjType
from cre.fact import define_fact, UntypedFact, call_untyped_fact, BaseFact
from cre.fact_intrinsics import fact_lower_getattr, resolve_fact_getattr_type
from cre.context import cre_context
from cre.tuple_fact import TupleFact, TF
from cre.default_ops import Add, Subtract, Divide
from cre.var import Var, GenericVarType
from cre.op import GenericOpType
from cre.utils import PrintElapse,encode_idrec, _func_from_address, _cast_structref, _obj_cast_codegen, _func_from_address, _incref_structref, _struct_from_ptr, decode_idrec, _ptr_to_data_ptr, _load_ptr, _struct_tuple_from_pointer_arr, _incref_ptr
from cre.structref import define_structref
from numba.experimental import structref
from cre.structref import CastFriendlyStructref, define_boxing
from numba.extending import overload_method, overload, lower_cast, SentryLiteralArgs
from numba.experimental.function_type import _get_wrapper_address
import cloudpickle
from cre.gval import get_gval_type, new_gval, gval as gval_type
from cre.vector import VectorType, new_vector
from cre.transform.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor, ChangeEventType
from itertools import chain
import cre.dynamic_exec

vectorizer_fields = {
    "head_to_slot_ind" : DictType(CREObjType, i8),
    "one_hot_map" : DictType(Tuple((i8,u8)), i8),
    "one_hot_nominals" : types.boolean,
    "val_types" : types.Any
}

i8_u8_pair = Tuple((i8,u8))

@structref.register
class VectorizerTypeClass(CastFriendlyStructref):
    pass

GenericVectorizerType = VectorizerTypeClass([(k,v) for k,v in vectorizer_fields.items()])


def get_vectorizer_type(val_types):
    fields = {**vectorizer_fields, 'val_types' : Tuple(val_types)} 
    struct_type = VectorizerTypeClass([(k,v) for k,v in fields.items()])
    struct_type._val_types = val_types
    return struct_type

class Vectorizer(structref.StructRefProxy):
    def __new__(cls, val_types, one_hot_nominals=False):
        val_types = tuple(val_types)
        vectorizer_type = get_vectorizer_type(val_types)
        self = vectorizer_ctor(vectorizer_type, one_hot_nominals)
        self._val_types = val_types
        return self

    def transform(self, mem):
        return vectorizer_apply(self, mem)

    def __call__(self, mem):
        return self.transform(mem)

    def get_inv_map(self):
        return get_inv_map(self)

define_boxing(VectorizerTypeClass, Vectorizer)

@generated_jit(cache=True, nopython=True)    
def vectorizer_ctor(struct_type, one_hot_nominals):
    def impl(struct_type, one_hot_nominals):    
        st = new(struct_type)
        st.head_to_slot_ind = Dict.empty(CREObjType,i8)
        st.one_hot_map = Dict.empty(i8_u8_pair,i8)
        st.one_hot_nominals = one_hot_nominals
        return st
    return impl

@generated_jit(cache=True)
@overload_method(VectorizerTypeClass, "apply")
def vectorizer_apply(self, mem):
    context = cre_context()
    val_types = self._val_types
    # gval_types = tuple([get_gval_type(t) for t in val_types])
    def impl(self, mem):
        # Ensure all heads are in head_to_slot_ind
        for i, fact in enumerate(mem.get_facts(gval_type)):
            if(not(fact.head in self.head_to_slot_ind)):
                self.head_to_slot_ind[fact.head] = len(self.head_to_slot_ind)

        if(not self.one_hot_nominals):
            # Build the numpy arrays 
            flt_vals = np.empty((len(self.head_to_slot_ind),),dtype=np.float64)
            nom_vals = np.zeros((len(self.head_to_slot_ind),),dtype=np.uint64)
            for i, fact in enumerate(mem.get_facts(gval_type)):
                slot = self.head_to_slot_ind[fact.head]
                
                flt_vals[slot] = fact.flt
                nom_vals[slot] = fact.nom
            return flt_vals, nom_vals
        else:
            # Make sure one_hot_map is up to date.
            for i, fact in enumerate(mem.get_facts(gval_type)):
                tup = (self.head_to_slot_ind[fact.head], fact.nom)
                if(tup not in self.one_hot_map):
                    self.one_hot_map[tup] = len(self.one_hot_map)

            # Build the numpy arrays w/ nom_vals as one_hot encoded
            flt_vals = np.empty((len(self.head_to_slot_ind),),dtype=np.float64)
            nom_vals = np.zeros((len(self.one_hot_map),),dtype=np.uint64)
            for i, fact in enumerate(mem.get_facts(gval_type)):
                slot = self.head_to_slot_ind[fact.head]
                # print(fact.head,'->', slot)
                flt_vals[slot] = fact.flt
                one_hot_slot = self.one_hot_map[(slot, fact.nom)]
                nom_vals[one_hot_slot] = 1
            return flt_vals, nom_vals

    return impl


@njit(cache=True)
def get_inv_map(self):
    d = Dict.empty(i8,CREObjType)
    for head, ind, in self.head_to_slot_ind.items():
        d[ind] = head
    return d
