import operator
import numpy as np
import numba
from numba.core.dispatcher import Dispatcher
from numba import types, njit, i8, u8, i4, u1, i8, literally, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template
from cre.memory import MemoryType, Memory, facts_for_t_id, fact_at_f_id
# from cre.fact import define_fact, BaseFactType, cast_fact, DeferredFactRefType, Fact
from cre.utils import (_struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_pointer,  lower_setattr, lower_getattr,
                       _pointer_from_struct, _decref_pointer, _incref_pointer, _incref_structref, _pointer_from_struct_incref,
                       _dict_from_ptr, _list_from_ptr)
from cre.utils import assign_to_alias_in_parent_frame
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.fact import Fact, gen_fact_import_str, get_offsets_from_member_types
from cre.var import Var
from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
 get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from cre.make_source import make_source, gen_def_func, gen_assign, resolve_template, gen_def_class
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.experimental.function_type import _get_wrapper_address
from cre.op import GenericOpType

from operator import itemgetter
from copy import copy
from os import getenv
from cre.utils import deref_type, OFFSET_TYPE_ATTR, OFFSET_TYPE_LIST, listtype_sizeof_item
import inspect, dill, pickle
from textwrap import dedent, indent

import time
class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.2f} ms')


SC_Record_field_dict = {
    # 'op' : GenericOpType,
    # 'depth' : i8,

    # A tensor where the values are the uniq_id
    'hist' : u8[::1],
    'hist_shape' : u8[::1],

    # 'last_update' : i8,
    #op has arg_types

    # Pointer to dictionary that maps values to a unique
    # id 'uid' i.e. is type DictType(op.sig.return_type, i8)
    'vals_to_uid_ptr' : i8
    
}
SC_Record_fields = [(k,v) for k,v in SC_Record_field_dict.items()]
SC_Record, SC_RecordType = \
    define_structref("SC_Record", SC_Record_fields)


record_list_type = ListType(SC_RecordType)
dict_str_to_record_list_type = DictType(unicode_type, record_list_type)
str_int_tuple = Tuple((unicode_type,i8))

SetChainingPlanner_field_dict = {
    'ops': ListType(GenericOpType),
    # List of dictionaries that map:
    #  Tuple(type_str[str],depth[int]) -> ListType[Record])
    'forward_records' : ListType(DictType(unicode_type, ListType(SC_RecordType))),

    # List of dictionaries that map:
    #  Tuple(type_str[str],depth[int]) -> ListType[Record])
    'backward_records' : ListType(DictType(unicode_type, ListType(SC_RecordType))),

    # Maps type_str[str] -> *Dict(val[any] -> depth[u1])
    'vals_to_depth_ptr_dict' : DictType(unicode_type, i8),
    # Maps (type_str[str],depth[int]) -> *Iterator[any]
    'flat_vals_ptr_dict' : DictType(Tuple((unicode_type,i8)), i8),

}
SetChainingPlanner_fields = [(k,v) for k,v in SetChainingPlanner_field_dict.items()]

@structref.register
class SetChainingPlannerTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class SetChainingPlanner(structref.StructRefProxy):
    def __new__(cls):
        self = sc_planner_ctor()
        self.curr_infer_depth = 0
        self.max_depth = 0
        return self

define_boxing(SetChainingPlannerTypeTemplate,SetChainingPlanner)
SetChainingPlannerType = SetChainingPlannerTypeTemplate(SetChainingPlanner_fields)


@njit(cache=True)
def sc_planner_ctor():
    st = new(SetChainingPlannerType)
    st.forward_records = List.empty_list(dict_str_to_record_list_type)
    st.backward_records = List.empty_list(dict_str_to_record_list_type)
    st.vals_to_depth_ptr_dict = Dict.empty(unicode_type, i8)
    st.flat_vals_ptr_dict = Dict.empty(str_int_tuple, i8)
    return st


### Planning Planning Planning ###

# When an op is applied it needs to insert the result into 
#  a dictionary so that only unique values go into then next step.
#  When an op broadcast applied a new Record is created and inserted into 
#  'forward_records'. 
#  After all ops (that will be tried) at a depth have been tried
#  each 'vals_to_uid' is joined into 'vals_to_depth' for the next depth
#  which is 


#  then it could map val[any] -> depth[u1].
#  Then a lookup to 'val' would yield rec.history and then
#  rec.history[ind]




def how_search(self, goal, ops=None,
             search_depth=1, max_solutions=10,
             min_stop_depth=-1):
    if(min_stop_depth == -1): min_stop_depth = search_depth

    # NOTE: Make sure that things exist here
    for depth in range(1, search_depth+1):
        if(depth < self.curr_infer_depth): continue
        if(_query_goal(self) is None and 
            self.curr_infer_depth > min_stop_depth):
            self.forward_chain_one(ops)

@njit(cache=True)
def _query_goal(self, type_name, typ, goal):
    vtd_ptr_d = self.vals_to_depth_ptr_dict
    if(type_name in vtd_ptr_d):
        vals_to_depth = _struct_from_pointer(typ,vtd_ptr_d)
        if(goal in vals_to_depth):
            return vals_to_depth[goal]
        else:
            return None
    else:
        return None

@njit(cache=True)
def insert_record(self, rec, ret_type_name, depth):
    '''Inserts a Record 'rec' into the list of records in  
        'self' for type 'ret_type_name' and depth 'depth' '''
    # Make sure records dictionaries up to this depth
    print("INSER",depth)
    while(depth >= len(self.forward_records)):
        self.forward_records.append(
            Dict.empty(unicode_type, record_list_type)
        )
    recs_at_depth = self.forward_records[depth]
    recs = recs_at_depth.get(ret_type_name,
            List.empty_list(SC_RecordType))
    recs.append(rec)
    recs_at_depth[ret_type_name] = recs

@njit(cache=True)
def join_records_of_type(self, depth, typ_name,
         typ, dict_typ):
    ''' Goes through every record at 'depth' and joins them
        so that 'val_to_depth' and 'flat_vals' are defined '''
    print("START", depth)
    recs = self.forward_records[depth][typ_name]
    prev_tup = (typ_name, depth-1)
    print("MID")
    if(typ_name in self.vals_to_depth_ptr_dict):
        vals_to_depth_ptr = self.vals_to_depth_ptr_dict[typ_name]
        vals_to_depth = _dict_from_ptr(dict_typ, vals_to_depth_ptr)
    else:
        vals_to_depth = Dict.empty(typ, i8)
    
    for rec in recs:
        vals_to_uid = _dict_from_ptr(dict_typ, rec.vals_to_uid_ptr)
        for val in vals_to_uid:
            if(val not in vals_to_depth):
                vals_to_depth[val] = depth

    flat_vals = List.empty_list(typ,len(vals_to_depth))
    for val in vals_to_depth:
        flat_vals.append(val) 
    
    if(typ_name in self.vals_to_depth_ptr_dict):
        _decref_pointer(self.vals_to_depth_ptr_dict[typ_name])
    self.vals_to_depth_ptr_dict[typ_name] = \
        _pointer_from_struct_incref(vals_to_depth)

    tup = (typ_name, depth)
    if(tup in self.flat_vals_ptr_dict):
        _decref_pointer(self.flat_vals_ptr_dict[tup])
    self.flat_vals_ptr_dict[tup] = \
         _pointer_from_struct_incref(flat_vals)


def join_records(self,depth,ops):
    typs = set([op.signature.return_type for op in ops])
    for typ in typs:
        typ_name = str(typ)
        val_to_depth = join_records_of_type(self,
            depth, typ_name, typ, DictType(typ,i8))

# Don't njit 
# Note: could probably not run all ops
def forward_chain_one(self, ops=None):
    '''Applies 'ops' on all current inferred values'''
    depth = self.curr_infer_depth
    if(ops is None): ops = self.ops
    for op in ops:
        rec = apply_multi(op, self, depth)
        print()
        if(rec is not None):
            insert_record(self, rec, op.return_type_name, depth+1)

    join_records(self, depth+1, ops)
    self.curr_infer_depth = depth+1

#### Source Generation -- apply_multi() ####

def _gen_retrieve_itr(tn,typ_name,ind='    '):
    return indent(f'''tup{tn} = ('{typ_name}',depth)
if(tup{tn} in planner.flat_vals_ptr_dict):
    iter_ptr{tn} = planner.flat_vals_ptr_dict[tup{tn}]
    iter{tn} = _dict_from_ptr(l_typ{tn}, iter_ptr{tn})
else:
    return None
''',prefix=ind)


def gen_apply_multi_source(op, ind='    '):
    has_check = hasattr(op,'check')
    sig = op.signature
    args = sig.args
    typs = {}
    for typ in sig.args:
        if(typ not in typs):
            typs[typ] = len(typs)

    src = \
f'''from numba import njit, i8
from numba.typed import Dict
from numba.types import ListType
import numpy as np
import dill
from cre.utils import _dict_from_ptr, _list_from_ptr, _pointer_from_struct_incref
from cre.sc_planner import SC_Record
''' 
    imp_targets = ['call'] + (['check'] if has_check else [])
    src += f'''{gen_import_str(type(op).__name__,
                 op.hash_code, imp_targets)}\n\n'''

    src += "".join([f'typ{i}'+", " for i in range(len(typs))]) + \
             f'= dill.loads({dill.dumps(tuple(typs.keys()))})\n'
    src += f'ret_typ = dill.loads({dill.dumps(sig.return_type)})\n\n'

    src += ", ".join([f'l_typ{i}' for i in range(len(typs))]) + \
            " = " + ", ".join([f'ListType(typ{i})' for i in range(len(typs))]) + '\n'

    a_cnt = list(range(len(args)))
    start_kwargs = ", ".join([f'start{i}=0' for i in a_cnt])
    src += f'''@njit(cache=True)
def apply_multi(planner, depth, {start_kwargs}):
'''


    for tn, typ_name in enumerate(typs.keys()):
        src += _gen_retrieve_itr(tn, typ_name)

    it_inds = [typs[args[i]] for i in range(len(args))]


    ls = ", ".join([f"l{i}" for i in a_cnt])
    l_defs = ", ".join([f"len(iter{it_inds[i]})-start{i}" for i in a_cnt])
    src += indent(f'''
{ls} = {l_defs}
hist_shape = ({ls})
hist = np.zeros(hist_shape, dtype=np.uint64)
vals_to_uid = Dict.empty(ret_typ, i8)

uid=1
''',prefix=ind)
    c_ind = copy(ind)
    for i, arg in enumerate(args):
        src += f'{c_ind}for i{i} in range(start{i},len(iter{it_inds[i]})):\n'
        c_ind += ind

    _is = ",".join([f"i{i}" for i in a_cnt])
    _as = ",".join([f"a{i}" for i in a_cnt])
    params = ",".join([f"iter{it_inds[i]}[i{i}]" for i in a_cnt])

    src += indent(f'''{_as} = {params}
{f'if(not check({_as})): continue' if has_check else ""}
v = call({_as})
v_uid = vals_to_uid.get(v,None)
if(v_uid is None):
    v_uid = vals_to_uid[v] = uid; uid+=1;
hist[{_is}] = v_uid

# if(v in vals_to_uid):
#     hist[{_is}] = vals_to_uid[v]
# else:
#     hist[{_is}] = uid
#     vals_to_uid[v] = uid; uid+=1''',prefix=c_ind)

    src += indent(f'''
vals_to_uid_ptr = _pointer_from_struct_incref(vals_to_uid)
return SC_Record(hist.flatten(),
                 np.array(hist_shape,dtype=np.uint64),
                 vals_to_uid_ptr
                )
    ''',prefix=ind)
    return src

###  apply_multi ###

def apply_multi(op, planner, depth):
    '''Applies 'op' at 'depth' and returns the SC_Record'''

    # If it doesn't already exist generate and inject '_apply_multi' into 'op'
    if(not hasattr(op,'_apply_multi')):
        hash_code = unique_hash(['_apply_multi',op.hash_code])  
        print(get_cache_path('apply_multi',hash_code))
        if(not source_in_cache('apply_multi',hash_code)):
            src = gen_apply_multi_source(op)
            source_to_cache('apply_multi',hash_code,src)
        l = import_from_cached('apply_multi',hash_code,['apply_multi'])
        setattr(op,'_apply_multi', l['apply_multi'])
        # print("<<<", type(am))

    am = getattr(op,'_apply_multi')
    return am(planner,depth)









### WILL BE GENERATED

# def apply_multi(planner, depth, start0=0, start1=0, start2=0):
#     tup0 = ('typ0',depth)
#     if(tup0 in planner.flat_vals_ptr_dict):
#         iter_ptr0 = planner.flat_vals_ptr_dict[tup0]
#         iter0 = _struct_from_pointer(LiteralType0, iter_ptr0)
#     else:
#         return None

#     tup1 = ('typ1',depth)
#     if(tup1 in planner.flat_vals_ptr_dict):
#         iter_ptr1 = planner.flat_vals_ptr_dict[tup1]
#         iter1 = _struct_from_pointer(LiteralType1, iter_ptr1)
#     else:
#         return None

#     l0,l1,l2 = len(iter0)-start0, len(iter1)-start1, len(iter2)-start2
#     hist_shape = (l0,l1,l0)
#     hist = np.zeros(hist_shape, dtype=np.uint64)
#     vals_to_uid = Dict(LiteralTypeOut, i8).empty()

#     uid = 1
#     for i0 in range(start0,len(iter0)):
#         for i1 in range(start1,len(iter1)):
#             for i2 in range(start2,len(iter0)):
#                 err, v = safe_apply(iter0[i0], iter1[i1], iter0[i2])
#                 if(err): continue
#                 if(v in vals_to_uid):
#                     hist[i0,i1,i2] = uid[v]
#                 else:
#                     hist[i0,i1,i2] = uid
#                     vals_to_uid[v] = uid; uid+=1


                    






















