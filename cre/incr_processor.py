import numpy as np
import numba
from numba import njit, i8,i4,u1, f8, f4, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType
from cre.structref import define_structref, define_structref_template
from numba.experimental.structref import new, define_attributes
from numba.extending import lower_cast, overload_method
from cre.memory import Memory,MemoryType
from cre.utils import _cast_structref, _obj_cast_codegen
from cre.vector import VectorType


########### IncrProcessor ##########

from numba import u1,u2,u8, types
from numba.types import Tuple, Set
from cre.fact import BaseFact
from cre.utils import decode_idrec, encode_idrec

incr_processor_fields = {
    "in_mem" : MemoryType,
    "change_queue_head" : i8,
}

IncrProcessor, IncrProcessorType, IncrProcessorTypeTemplate = define_structref("IncrProcessor", incr_processor_fields, define_constructor=True, return_type_class=True) 
IncrProcessorTypeTemplate.__str__ = lambda x : "cre.IncrProcessor"    


# print(IncrProcessor)
# print(IncrProcessorType)
# raise ValueError()

@njit(IncrProcessorType(IncrProcessorType, MemoryType),cache=True)
def init_incr_processor(st, mem):
    st.in_mem = mem
    st.change_queue_head = 0
    return st

# @njit(IncrProcessorType(IncrProcessorType, MemoryType),cache=True)
# def incr_processor_ctor(st, mem):
#     st = new(IncrProcessorType)
#     st.in_mem = mem
#     st.change_queue_head = 0
#     return st

RETRACT = u1(0xFF)# u1(0)
DECLARE = u1(0)

change_event_fields = {
    # The idrec for the declared/modified/retracted object
    "idrec" : u8,
    # The t_id for the declared/modified/retracted object
    "t_id" : u2,
    # The f_id for the declared/modified/retracted object
    "f_id" : u8,
    
    # Whether or not a Fact with this idrec was ever retracted 
    #  indicating that we might need to clean something up
    "was_retracted" : u1,

    # Whether or not a Fact with this idrec was declared. If
    #  both a retract and a declare occured then 'was_retracted'
    #  will be true, but 'was_declared' will only be true if it
    #  occured after the retraction.
    "was_declared" : u1,

    # Whether or not a Fact with this idrec was modified. Will be False 
    #  if a declare or retract occured.
    "was_modified" : u1,

    # If a modification occured the a_ids associated with the change
    "a_ids" : types.optional(ListType(u1)),

}

ChangeEvent, ChangeEventType = define_structref("ChangeEvent", change_event_fields, define_constructor=False) 
ChangeEvent.__str__ = lambda x : f"ChangeEvent(t_id={x.t_id}, f_id={x.f_id}, ret={x.was_retracted}, dec={x.was_declared}, mod={x.was_modified}{f', a_ids={x.a_ids}' if x.a_ids else ''})"

@njit(cache=True)
def change_event_ctor(idrec):
    st = new(ChangeEventType)
    st.idrec = idrec
    st.t_id, st.f_id, _ = decode_idrec(idrec)
    st.a_ids = None
    st.was_retracted = False
    st.was_declared = False
    st.was_modified = False
    return st    

# change_event_ctor(0)
u1_opBaseFact_Tuple_type = Tuple((u8,u1[::1],types.optional(BaseFact)))
@njit(ListType(ChangeEventType)(VectorType,i8,i8), cache=True, locals={"a_id" : u1})
def accumulate_change_events(cq, start, end=-1):
    ''' Takes in a change queue (i.e. Vector<i8>) and a start and end position
        and returns an accumulated set of ChangeEvent objects associated with
        the changes between start and end. If end is not set then it defaults
        to the head of the the change_queue.
    '''
    # cq = incr_pr.in_mem.mem_data.change_queue
    if(end == -1): end = cq.head
    ce_dict = Dict.empty(u8,ChangeEventType)
    for i in range(start, end):
        t_id, f_id, a_id = decode_idrec(cq[i])
        # print(t_id, f_id, a_id)
        idrec = encode_idrec(t_id, f_id, u1(0))
        if(idrec not in ce_dict):
            ce_dict[idrec] = change_event_ctor(idrec)
        ce = ce_dict[idrec]

        if(a_id == RETRACT):
            ce.was_modified = False
            ce.was_declared = False
            ce.was_retracted = True
            ce.a_ids = None
        elif(a_id == DECLARE):
            ce.was_modified = False
            ce.was_declared = True
            ce.a_ids = None
        else:
            # Modify case
            if(not ce.was_declared):
                ce.was_modified = True
                if(ce.a_ids is None):
                    ce.a_ids = List.empty_list(u1)
                if(a_id not in ce.a_ids):
                    ce.a_ids.append(a_id)
    change_events = List.empty_list(ChangeEventType)
    for ce in ce_dict.values():
        change_events.append(ce)
    return change_events


@generated_jit(cache=True)
@overload_method(IncrProcessorTypeTemplate,'get_changes')
def incr_pr_accumulate_change_events(incr_pr, end=-1, exhaust_changes=True):
    def impl(incr_pr, end=-1, exhaust_changes=True):
        cq = incr_pr.in_mem.mem_data.change_queue
        start = incr_pr.change_queue_head
        if(end == -1): end = cq.head
        change_events = accumulate_change_events(cq, start, end)
        if(exhaust_changes): incr_pr.change_queue_head = end
        return change_events
    return impl



def get_changes(incr_pr, end=-1, exhaust_changes=True):
    return incr_pr_accumulate_change_events(incr_pr, end, exhaust_changes)

IncrProcessor.get_changes = get_changes
