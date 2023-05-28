import numpy as np
from numba import njit, generated_jit, types, literal_unroll, u8, i8, f8, u1, u2
from numba.types import unicode_type,  intp, Tuple,  Tuple, DictType, ListType
from numba.typed import Dict, List
from numba.experimental.structref import new
from cre.obj import CREObjType
from cre.context import cre_context
from cre.tuple_fact import TupleFact, TF
from cre.var import Var, VarType
# from cre.op import GenericOpType
from cre.utils import PrintElapse,encode_idrec, _func_from_address, _obj_cast_codegen, _func_from_address, _incref_structref, decode_idrec, _ptr_to_data_ptr, _load_ptr, _struct_tuple_from_pointer_arr, _incref_ptr
from cre.structref import define_structref
from numba.experimental import structref
from cre.structref import CastFriendlyStructref, define_boxing
from numba.extending import overload_method, overload, lower_cast, SentryLiteralArgs
from numba.experimental.function_type import _get_wrapper_address
import cloudpickle
from cre.gval import gval as gval_type
from cre.vector import VectorType, new_vector
from cre.transform.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor, ChangeEventType
from itertools import chain
import cre.dynamic_exec

vectorizer_fields = {
    "slot_map" : DictType(CREObjType, i8),
    "one_hot_map" : DictType(Tuple((i8,u8)), i8),

    "inv_slot_map" : DictType(i8, CREObjType),
    "inv_one_hot_map" : DictType(i8, Tuple((i8,u8))),

    "one_hot_nominals" : types.boolean,
    "encode_missing" : types.boolean,
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
    def __new__(cls, val_types, one_hot_nominals=False, encode_missing=False):
        val_types = tuple(val_types)
        vectorizer_type = get_vectorizer_type(val_types)
        self = vectorizer_ctor(vectorizer_type, one_hot_nominals, encode_missing)
        self._val_types = val_types
        return self

    def transform(self, mem):
        return vectorizer_apply(self, mem)

    def __call__(self, mem):
        return self.transform(mem)

    def unvectorize(self, slot, nom=0):
        return unvectorize(self, slot, u8(nom))


define_boxing(VectorizerTypeClass, Vectorizer)

@generated_jit(cache=True, nopython=True)    
def vectorizer_ctor(struct_type, one_hot_nominals, encode_missing):
    def impl(struct_type, one_hot_nominals, encode_missing):    
        st = new(struct_type)
        st.slot_map = Dict.empty(CREObjType,i8)
        st.one_hot_map = Dict.empty(i8_u8_pair,i8)

        st.inv_slot_map = Dict.empty(i8, CREObjType)
        st.inv_one_hot_map = Dict.empty(i8, i8_u8_pair)

        st.one_hot_nominals = one_hot_nominals
        st.encode_missing = encode_missing
        return st
    return impl

@generated_jit(cache=True, nopython=True)
@overload_method(VectorizerTypeClass, "apply")
def vectorizer_apply(self, mem):
    context = cre_context()
    val_types = self._val_types
    # fact_types = tuple([get_fact_type(t) for t in val_types])
    def impl(self, mem):

        slot_map  = self.slot_map
        inv_slot_map  = self.inv_slot_map
        one_hot_map  = self.one_hot_map
        inv_one_hot_map  = self.inv_one_hot_map
        # Ensure all heads are in slot_map
        for i, gval in enumerate(mem.get_facts(gval_type)):

            if(not gval.head in slot_map):
                slot = len(slot_map)
                slot_map[gval.head] = slot
                inv_slot_map[slot] = gval.head
                if(self.encode_missing and self.one_hot_nominals):
                    one_hot_slot = len(one_hot_map)
                    one_hot_map[(slot, u8(0))] = one_hot_slot
                    inv_one_hot_map[one_hot_slot] = (slot, u8(0))


        if(not self.one_hot_nominals):
            # Build the numpy arrays 
            flt_vals = np.empty((len(slot_map),),dtype=np.float64)
            nom_vals = np.zeros((len(slot_map),),dtype=np.uint64)
            for i, gval in enumerate(mem.get_facts(gval_type)):
                slot = slot_map[gval.head]
                
                flt_vals[slot] = gval.flt
                nom_vals[slot] = gval.nom
            return flt_vals, nom_vals
        else:
            # Make sure one_hot_map is up to date.
            for i, gval in enumerate(mem.get_facts(gval_type)):

                tup = (slot_map[gval.head], gval.nom)
                if(tup not in one_hot_map):
                    one_hot_slot = len(one_hot_map)
                    one_hot_map[tup] = one_hot_slot
                    inv_one_hot_map[one_hot_slot] = tup

            # Build the numpy arrays w/ nom_vals as one_hot encoded
            flt_vals = np.empty((len(slot_map),),dtype=np.float64)
            nom_vals = np.zeros((len(one_hot_map),),dtype=np.uint64)

            if(self.encode_missing):
                for head in slot_map:
                    slot = slot_map[head]
                    one_hot_slot = one_hot_map[(slot, u8(0))]
                    nom_vals[one_hot_slot] = 1


            for i, gval in enumerate(mem.get_facts(gval_type)):
                slot = slot_map[gval.head]
                # print(slot, "<-", gval.head, gval.head.hash_val)
                flt_vals[slot] = gval.flt
                one_hot_slot = one_hot_map[(slot, gval.nom)]
                # print(one_hot_slot, gval)#gval.head, "==", gval.nom)
                nom_vals[one_hot_slot] = 1
                if(self.encode_missing):
                    one_hot_slot = one_hot_map[(slot, u8(0))]
                    nom_vals[one_hot_slot] = 0

            return flt_vals, nom_vals

    return impl


@njit(Tuple((CREObjType, u8))(GenericVectorizerType, i8,u8), cache=True)
def unvectorize(self, slot, nom=0):
    if(self.one_hot_nominals):
        slot, nom = self.inv_one_hot_map[slot]
    head = self.inv_slot_map[slot]
    return head, nom


# Note: Unecessary since inversion is now included with update, but may have other uses
@njit(cache=True)
def build_inv_map(self):
    # print("...............")
    # print(self.slot_map)
    # print("...............")
    d = Dict.empty(i8,CREObjType)
    if(self.one_hot_nominals):
        heads = List.empty_list(CREObjType)
        for i, head in enumerate(self.slot_map):
            heads.append(head)

        for (head_ind, nom), ind, in self.one_hot_map.items():
            d[ind] = TF(heads[head_ind], nom)
    else:
        for head, ind, in self.slot_map.items():
            d[ind] = head
    return d
