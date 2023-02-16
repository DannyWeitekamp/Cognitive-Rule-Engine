import numpy as np
import numba
from numba import njit, i8,i4,u1, f8, f4, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType
from cre.structref import define_structref, define_structref_template
from numba.experimental.structref import new, define_attributes
from numba.extending import lower_cast, overload_method
from cre.memset import MemSet,MemSetType
from cre.utils import _obj_cast_codegen
from cre.vector import VectorType
from cre.change_event import ChangeEventType, accumulate_change_events


########### IncrProcessor ##########

from numba import u1,u2,u8, types
from numba.types import Tuple, Set
from cre.fact import BaseFact
from cre.utils import decode_idrec, encode_idrec

incr_processor_fields = {
    "in_memset" : MemSetType,
    "change_queue_head" : i8,
}

IncrProcessor, IncrProcessorType, IncrProcessorTypeTemplate = define_structref("IncrProcessor", incr_processor_fields, define_constructor=True, return_type_class=True) 
IncrProcessorTypeTemplate.__str__ = lambda x : "cre.IncrProcessor"    


# print(IncrProcessor)
# print(IncrProcessorType)
# raise ValueError()

@njit(IncrProcessorType(IncrProcessorType, MemSetType),cache=True)
def init_incr_processor(st, ms):
    st.in_memset = ms
    st.change_queue_head = 0
    return st

# @njit(IncrProcessorType(IncrProcessorType, MemSetType),cache=True)
# def incr_processor_ctor(st, mem):
#     st = new(IncrProcessorType)
#     st.in_mem = mem
#     st.change_queue_head = 0
#     return st






@generated_jit(cache=True)
@overload_method(IncrProcessorTypeTemplate,'get_changes')
def incr_pr_accumulate_change_events(incr_pr, end=-1, exhaust_changes=True):
    def impl(incr_pr, end=-1, exhaust_changes=True):
        cq = incr_pr.in_memset.change_queue
        start = incr_pr.change_queue_head
        if(end == -1): end = cq.head
        change_events = accumulate_change_events(cq, start, end)
        if(exhaust_changes): incr_pr.change_queue_head = end
        return change_events
    return impl



def get_changes(incr_pr, end=-1, exhaust_changes=True):
    return incr_pr_accumulate_change_events(incr_pr, end, exhaust_changes)

IncrProcessor.get_changes = get_changes
