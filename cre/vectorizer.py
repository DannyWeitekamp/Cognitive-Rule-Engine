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
from cre.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor
from cre.memory import Memory, MemoryType
from cre.structref import CastFriendlyStructref, define_boxing
from numba.extending import overload_method, overload, lower_cast, SentryLiteralArgs
from numba.experimental.function_type import _get_wrapper_address
import cloudpickle
from cre.gval import get_gval_type, new_gval, gval as gval_type
from cre.vector import VectorType, new_vector
from cre.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor, ChangeEventType
from itertools import chain

vectorizer_fields = {
    "pattern_to_slot_ind" : DictType(CREObjType, i8),
    "val_types" : types.Any
}

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
    def __new__(cls, val_types):
        vectorizer_type = get_vectorizer_type(val_types)
        self = vectorizer_ctor(vectorizer_type)
        self._val_types = val_types
        return self

    def apply(self,mem):
        return vectorizer_apply(self, mem)


define_boxing(VectorizerTypeClass, Vectorizer)


@generated_jit(cache=True)    
def vectorizer_ctor(struct_type):
    def impl(struct_type):    
        st = new(struct_type)
        st.pattern_to_slot_ind = Dict.empty(CREObjType,i8)
        return st
    return impl



# @njit(Tuple(f8[:,::1],i8[:,::1])(VectorizerType, MemoryType),cache=True)
@generated_jit(cache=True)
def vectorizer_apply(self, mem):
    context = cre_context()
    val_types = self._val_types
    # print(val_types)
    gval_types = tuple([get_gval_type(t) for t in val_types])
    # print(gval_types)
    # print([context.get_t_id(_type=t) for t in gval_types])
    def impl(self, mem):
        for gval_type in literal_unroll(gval_types):
            # print("HI")
            for i, fact in enumerate(mem.get_facts(gval_type)):
                if(fact.head not in self.pattern_to_slot_ind):
                    self.pattern_to_slot_ind[fact.head] = len(self.pattern_to_slot_ind)
                print(hash(fact.head), fact.head==fact.head, fact.head in self.pattern_to_slot_ind)
                # print()
                # print(i, fact.head)
    return impl


