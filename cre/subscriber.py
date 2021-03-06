import numpy as np
from numba import types, njit, i8, u8, i4, u1, literally, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type, void
from numba.experimental.structref import new
from numba.extending import overload_method, intrinsic
from cre.structref import define_structref, define_structref_template
# from cre.memset import MemoryType, Memory
from cre.fact import define_fact
from cre.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref
from cre.vector import new_vector, VectorType
from copy import copy

meminfo_type = types.MemInfoPointer(types.voidptr)


base_subscriber_fields = [
    #Meminfo for the Memory to which this subscribes
    ("mem_meminfo", types.optional(meminfo_type)),
    #upstream BaseSubscribers' meminfos that need to be updated before this can be.
    ("upstream", ListType(meminfo_type)), 
    #The subscribers immediately downstream of this one.
    ("children", ListType(meminfo_type)), 

    ("change_head", i8),
    ("grow_head", i8),
    ("change_queue", VectorType),#ListType(u8)),
    ("grow_queue", VectorType),#ListType(u8)),

    # #Indicies or idrecs of things that have changed in the subscriber's parent (i.e. last 
    # #  upstream). The parent is responsible for filling this.
    # ("change_queue", VectorType),#ListType(u8)),
    # #Same as change_queue but for when the something has been added upstream
    # ("grow_queue", VectorType),#ListType(u8)),
    #An update function that updates state of the subscriber and pushes changes to all children.
    ("update_func", types.FunctionType(void(meminfo_type))), 
    # #The t_id corresponding to the type to which this subscriber subscribes
    # ("t_id", i8)
    #
]

BASE_SUBSCRIBER_QUEUE_SIZE = 8

BaseSubscriber, BaseSubscriberType = define_structref("BaseSubscriber", base_subscriber_fields)

@njit(cache=True)
def init_base_subscriber(bs):
    bs.mem_meminfo = None#_meminfo_from_struct(mem)
    bs.upstream = List.empty_list(meminfo_type)
    bs.children = List.empty_list(meminfo_type)

    bs.change_head = 0
    bs.grow_head = 0
    bs.change_queue = new_vector(BASE_SUBSCRIBER_QUEUE_SIZE)#List.empty_list(u8)
    bs.grow_queue = new_vector(BASE_SUBSCRIBER_QUEUE_SIZE)#List.empty_list(u8)


@njit(cache=True)
def link_downstream(parent, child):
    child_meminfo = _meminfo_from_struct(child)
    parent_meminfo = _meminfo_from_struct(parent)

    child.mem_meminfo = parent.mem_meminfo

    parent.children.append(child_meminfo)
    upstream = parent.upstream.copy()
    upstream.append(parent_meminfo)
    child.upstream = upstream


@njit(cache=True)
def link_downstream_of_mem(mem, child):
    child_meminfo = _meminfo_from_struct(child)
    mem_meminfo = _meminfo_from_struct(mem)

    child.mem_meminfo = mem_meminfo

    parent.children.append(child_meminfo)
    upstream = parent.upstream.copy()
    upstream.append(parent_meminfo)
    child.upstream = upstream

