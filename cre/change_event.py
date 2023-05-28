from numba import njit, i8,i4,u1, u2, u8, f8, f4, generated_jit, types
from numba.types import ListType, DictType, Tuple
from numba.typed import Dict, List
from cre.vector import VectorType
from cre.structref import define_structref
from cre.fact import BaseFact
from cre.utils import decode_idrec, encode_idrec
from numba.experimental.structref import new

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
    if(start == end):
        return List.empty_list(ChangeEventType)

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
