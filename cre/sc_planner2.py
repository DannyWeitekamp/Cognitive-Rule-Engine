import operator
import numpy as np
import numba
from numba.core.dispatcher import Dispatcher
from numba import types, njit, f8, i8, u8, i4, u1, u4, i8, literally, generated_jit, void
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.context import kb_context
from cre.structref import define_structref, define_structref_template
from cre.kb import KnowledgeBaseType, KnowledgeBase, facts_for_t_id, fact_at_f_id
from cre.var import GenericVarType
# from cre.fact import define_fact, BaseFactType, cast_fact, DeferredFactRefType, Fact
from cre.utils import (_struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_pointer,  lower_setattr, lower_getattr,
                       _pointer_from_struct, _decref_pointer, _incref_pointer, _incref_structref, _pointer_from_struct_incref,
                       _dict_from_ptr, _list_from_ptr, _load_pointer, _arr_from_data_ptr)
from cre.utils import assign_to_alias_in_parent_frame
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.fact import Fact, gen_fact_import_str, get_offsets_from_member_types
from cre.var import Var, VarTypeTemplate
from cre.op import GenericOpType, OpTypeTemplate
from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
 get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from cre.make_source import make_source, gen_def_func, gen_assign, resolve_template, gen_def_class
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.experimental.function_type import _get_wrapper_address


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
    'data' : u4[::1],
    'stride' : i8[:,::1],
    'is_op' : u1,
    'op' : GenericOpType,
    'var' : GenericVarType,
    'depth' : i8,
    'nargs' : i8,
}
SC_Record_fields = [(k,v) for k,v in SC_Record_field_dict.items()]
SC_Record, SC_RecordType = \
    define_structref("SC_Record", SC_Record_fields, define_constructor=False)

@njit(cache=True)
def _sc_record_ctor_helper(data,stride, depth, nargs):
    st = new(SC_RecordType)
    st.data = data
    st.stride = stride
    st.depth = depth
    st.nargs = nargs
    return st

# @generated_jit(cache=True)
# def sc_record_ctor(data,stride, op_or_var, depth, nargs):
#     print(data,stride, op_or_var, depth, nargs)
#     if(isinstance(op_or_var, OpTypeTemplate)):
#         def impl(data, stride, op_or_var, depth, nargs):
#             st = _sc_record_ctor_helper(data, stride, op_or_var, depth, nargs)
#             st.is_op = True
#             st.op = op_or_var
#             return st
#     elif(isinstance(op_or_var, VarTypeTemplate)):
#         def impl(data, stride, op_or_var, depth, nargs):
#             st = _sc_record_ctor_helper(data, stride, op_or_var, depth, nargs)
#             st.is_op = False
#             st.var = op_or_var
#             return st
#     else:
#         print('fail')
#     return impl
    

@overload(SC_Record, prefer_literal=False)
def overload_SC_Record(data, stride, op_or_var, depth, nargs):
    if(isinstance(op_or_var, OpTypeTemplate)):
        def impl(data, stride, op_or_var, depth, nargs):
            st = _sc_record_ctor_helper(data, stride, depth, nargs)
            st.is_op = True
            st.op = op_or_var
            return st
    elif(isinstance(op_or_var, VarTypeTemplate)):
        def impl(data, stride, op_or_var, depth, nargs):
            st = _sc_record_ctor_helper(data, stride, depth, nargs)
            st.is_op = False
            st.var = op_or_var
            return st
    else:
        print('fail')
    return impl

SC_Record_Entry_field_dict = {
    'rec' : SC_RecordType,
    'next_entry_ptr' : i8,
    'args' : u4[::1],
}
SC_Record_Entry_fields = [(k,v) for k,v in SC_Record_Entry_field_dict.items()]
SC_Record_Entry, SC_Record_EntryType = \
    define_structref("SC_Record_Entry", SC_Record_Entry_fields)

@njit(cache=True)
def rec_entry_from_ptr(d_ptr, L=-1):
    ptrs = _arr_from_data_ptr(d_ptr, (2,),dtype=np.int64)
    rec_ptr, next_entry_ptr = ptrs[0], ptrs[1]
    rec = _struct_from_pointer(SC_RecordType, rec_ptr)
    args = _arr_from_data_ptr(d_ptr+16,(rec.nargs,),dtype=np.uint32)

    return SC_Record_Entry(rec,next_entry_ptr,args)

@njit(cache=True)
def next_rec_entry(re):
    if(re.next_entry_ptr != 0):
        return rec_entry_from_ptr(re.next_entry_ptr)
    else:
        return None


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

    # Maps type_str[str] -> *(Dict: val[any] -> *SC_Record_Entry)
    'val_map_ptr_dict' : DictType(unicode_type, i8),
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
    st.val_map_ptr_dict = Dict.empty(unicode_type, i8)
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
    recs = self.forward_records[depth][typ_name]

    val_map = _dict_from_ptr(dict_typ, self.val_map_ptr_dict[typ_name])
    flat_vals = List.empty_list(typ,len(val_map))

    for val in val_map:
        flat_vals.append(val) 

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
        if(rec is not None):
            insert_record(self, rec, op.return_type_name, depth+1)

    join_records(self, depth+1, ops)
    self.curr_infer_depth = depth+1

#### Source Generation -- apply_multi() ####

def _gen_retrieve_itr(tn,typ_name,ind='    '):
    return indent(f'''tup{tn} = ('{typ_name}',depth)
if(tup{tn} in planner.flat_vals_ptr_dict):
    iter_ptr{tn} = planner.flat_vals_ptr_dict[tup{tn}]
    iter{tn} = _list_from_ptr(l_typ{tn}, iter_ptr{tn})
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
f'''from numba import njit, i8, u4, u8
from numba.typed import Dict
from numba.types import ListType, DictType
import numpy as np
import dill
from cre.utils import _dict_from_ptr, _list_from_ptr, _pointer_from_struct_incref, _get_array_data_ptr
from cre.sc_planner2 import SC_Record, SC_Record_Entry, SC_Record_EntryType
''' 
    imp_targets = ['call'] + (['check'] if has_check else [])
    src += f'''{gen_import_str(type(op).__name__,
                 op.hash_code, imp_targets)}\n\n'''

    src += "".join([f'typ{i}'+", " for i in range(len(typs))]) + \
             f'= dill.loads({dill.dumps(tuple(typs.keys()))})\n'

    src += f'''ret_typ = dill.loads({dill.dumps(sig.return_type)})
ret_d_typ = DictType(ret_typ,i8)
'''

    src += ", ".join([f'l_typ{i}' for i in range(len(typs))]) + \
            " = " + ", ".join([f'ListType(typ{i})' for i in range(len(typs))]) + '\n'

    a_cnt = list(range(len(args)))
    # start_kwargs = ", ".join([f'start{i}=0' for i in a_cnt])
    src += f'''N_ARGS = {len(args)}
ENTRY_WIDTH = 4+N_ARGS
@njit(cache=True)
def apply_multi(op_inst, planner, depth):
'''

    for tn, typ_name in enumerate(typs.keys()):
        src += _gen_retrieve_itr(tn, typ_name)

    it_inds = [typs[args[i]] for i in range(len(args))]


    ls = ", ".join([f"l{i}" for i in a_cnt])
    l_defs = '\n'.join([f'l{i} = stride[{i}][1]-stride[{i}][0]' for i in a_cnt])#", ".join([f"len(iter{it_inds[i]})-start{i}" for i in a_cnt])
    stride_defaults = ",".join([f'[0,len(iter{it_inds[i]})]' for i in a_cnt])
    src += indent(f'''
stride = np.array([{stride_defaults}],dtype=np.int64)
val_map =  _dict_from_ptr(ret_d_typ,
{ind}planner.val_map_ptr_dict['{str(sig.return_type)}'])
{l_defs}
data_len = {"*".join([f'l{i}' for i in a_cnt])}*ENTRY_WIDTH
data = np.empty(data_len,dtype=np.uint32)
d_ptr = _get_array_data_ptr(data)
rec = SC_Record(data, stride, op_inst, depth, N_ARGS)
rec_ptr = _pointer_from_struct_incref(rec)

d_offset=0
''',prefix=ind)
    c_ind = copy(ind)
    for i, arg in enumerate(args):
        src += f'{c_ind}for i{i} in range(stride[{i}][0],stride[{i}][1]):\n'
        c_ind += ind

    _is = ",".join([f"i{i}" for i in a_cnt])
    _as = ",".join([f"a{i}" for i in a_cnt])
    params = ",".join([f"iter{it_inds[i]}[i{i}]" for i in a_cnt])
    arg_assigns = "\n".join([f"data[d_offset+{4+i}] = i{i}" for i in a_cnt])

    src += indent(f'''{_as} = {params}
{f'if(not check({_as})): continue' if has_check else ""}
v = call({_as})

prev_entry = val_map.get(v,0)

data[d_offset +0] = u4(rec_ptr) # get low bits
data[d_offset +1] = u4(rec_ptr>>32) # get high bits
data[d_offset +2] = u4(prev_entry) # get low bits
data[d_offset +3] = u4(prev_entry>>32)# get high bits

#Put arg inds at the end
{arg_assigns}

val_map[v] = d_ptr + d_offset*4
d_offset += ENTRY_WIDTH
        
''',prefix=c_ind)

    src += f'''{ind}return rec'''
    return src

###  apply_multi ###
u4_slice = u4[:]
@njit(cache=True)
def _assert_prepared(self, typ, typ_name, depth):
    while(depth >= len(self.forward_records)):
        self.forward_records.append(
            Dict.empty(unicode_type, record_list_type)
        )
    if(typ_name not in self.val_map_ptr_dict):
        val_map = Dict.empty(typ, i8)
        val_map_ptr = _pointer_from_struct_incref(val_map)
        self.val_map_ptr_dict[typ_name] = val_map_ptr
    
    


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

    typ = op.signature.return_type
    typ_name = str(typ)
    _assert_prepared(planner, typ, typ_name, depth)

    am = getattr(op,'_apply_multi')
    return am(op,planner,depth)

### Explanation Tree Entry ###

ExplanationTreeEntry_field_dict = {
    "is_op" : u1,
    "op" : GenericOpType,
    "var" : GenericVarType,
    "child_arg_ptrs" : i8[::1]
}
ExplanationTreeEntry_fields = [(k,v) for k,v in ExplanationTreeEntry_field_dict.items()]
ExplanationTreeEntry, ExplanationTreeEntryType = \
    define_structref("ExplanationTreeEntry", ExplanationTreeEntry_fields, define_constructor=False)

@generated_jit(cache=True, nopython=True)
def expl_tree_entry_ctor(op_or_var, child_arg_ptrs=None):
    if(isinstance(op_or_var, OpTypeTemplate)):
        def impl(op_or_var, child_arg_ptrs):
            st = new(ExplanationTreeEntryType)
            st.is_op = True
            st.op = op_or_var
            st.child_arg_ptrs = child_arg_ptrs
            return st
    elif(isinstance(op_or_var, VarTypeTemplate)):
        def impl(op_or_var, child_arg_ptrs=None):
            st = new(ExplanationTreeEntryType)
            st.is_op = False
            st.var = op_or_var
            return st
    return impl




### Explanation Tree ###

ExplanationTree_field_dict = {
    'children' : ListType(ExplanationTreeEntryType) #List[(op<GenericOpType> ,*ExplanationTree[::1])]
}
ExplanationTree_fields = [(k,v) for k,v in ExplanationTree_field_dict.items()]
ExplanationTree, ExplanationTreeType = \
    define_structref("ExplanationTree", ExplanationTree_fields, define_constructor=False)


@njit(cache=True)
def expl_tree_ctor():
    st = new(ExplanationTreeType)
    st.children = List.empty_list(ExplanationTreeEntryType)
    return st

i8_et_dict = DictType(i8,ExplanationTreeType)

@njit(void(
        SC_Record_EntryType,
        DictType(unicode_type,i8_et_dict),
        ExplanationTreeType),
     cache=True)
def _fill_arg_inds_from_rec_entries(re, new_arg_inds, expl_tree):
    '''Goes through linked list of record entries and
        adds the argument indicies into new_arg_inds
        with a new instance of an ExplanationTree if needed.
        Add the new ExplanationTrees to the children of 'expl_tree'
    '''
    while(re is not None):
        if(re.rec.is_op):
            op = re.rec.op 
            child_arg_ptrs = np.empty(len(re.args), dtype=np.int64)
            for i, (arg_type_name, arg_ind) in enumerate(zip(op.arg_type_names, re.args)):
                #Make sure a set of indicies has been instantied for 'arg_type_name'
                if(arg_type_name not in new_arg_inds):
                    new_arg_inds[arg_type_name] = Dict.empty(i8,ExplanationTreeType)
                uai = new_arg_inds[arg_type_name]

                # Fill in a new ExplanationTree instance if needed
                if(arg_ind not in uai):
                    uai[arg_ind] = expl_tree_ctor()

                # Throw new tree instance into the children of 'expl_tree'
                child_arg_ptrs[i] = _pointer_from_struct_incref(uai[arg_ind])
                entry = expl_tree_entry_ctor(op,child_arg_ptrs)
                expl_tree.children.append(entry)
        else:
            entry = expl_tree_entry_ctor(re.rec.var)
            expl_tree.children.append(entry)
        re = next_rec_entry(re)   


@generated_jit(cache=True, nopython=True) 
def retrace_arg_inds(planner, typ,  goals, new_arg_inds=None):
    '''Find applications of operations that resulted in each 
        goal in goals. Add the indicies of the args as they
        occur in flat_vals.
    '''
    # The actual type inside the type ref 'typ'
    _typ = typ.instance_type

    val_map_d_typ = DictType(_typ, i8) 
    _goals_d_typ = DictType(_typ, ExplanationTreeType) 
    typ_name = str(_typ)
    def impl(planner, typ, goals, new_arg_inds=None):
        if(new_arg_inds is None):
            new_arg_inds = Dict.empty(unicode_type, i8_et_dict)
        val_map =  _dict_from_ptr(val_map_d_typ,
            planner.val_map_ptr_dict[typ_name])

        _goals = _dict_from_ptr(_goals_d_typ, goals[typ_name])

        for goal, expl_tree in _goals.items():
            # 're' is the head of a linked list of rec_entries
            re = rec_entry_from_ptr(val_map[goal])
            print(goal, re)
            _fill_arg_inds_from_rec_entries(re,
                new_arg_inds, expl_tree)
            
        return new_arg_inds
    return impl    


@generated_jit(cache=True,nopython=True) 
def fill_subgoals_from_arg_inds(planner, arg_inds, typ, depth, new_subgoals):
    '''For'new_subgoals' with the actual values pointed to by
         the arg_inds of 'typ' '''
    _typ = typ.instance_type
    typ_name = str(_typ)
    lst_typ = ListType(_typ)
    def impl(planner, arg_inds, typ, depth, new_subgoals):
        _new_subgoals =  Dict.empty(typ, ExplanationTreeType)
        _arg_inds = arg_inds[typ_name]
        vals = _list_from_ptr(lst_typ, planner.flat_vals_ptr_dict[(typ_name,depth)])
        for ind, expl_tree in _arg_inds.items():
            _new_subgoals[vals[ind]] = expl_tree
        print("new_subgoals",List(_new_subgoals.keys()))
        # Inject the new subgoals for 'typ' into 'new_subgoals'
        new_subgoals[typ_name] = _pointer_from_struct_incref(_new_subgoals)
    return impl


def retrace_goals_back_one(planner, goals):
    new_arg_inds = None
    for typ_name in goals:
        # fix later 
        typ = f8 if typ_name == 'float64' else unicode_type
        new_arg_inds = retrace_arg_inds(planner, typ, goals, new_arg_inds)

    if(len(new_arg_inds) == 0): return None

    new_subgoals = _init_subgoals()
    for typ_name in new_arg_inds:
        # fix later 
        typ = f8 if typ_name == 'float64' else unicode_type
        new_subgoals = fill_subgoals_from_arg_inds(
                planner, new_arg_inds, typ,
                planner.curr_infer_depth, new_subgoals)


    return new_subgoals

@generated_jit(cache=True)
def _init_root_goals(g_typ, goal, root):
    g_typ_name = str(g_typ.instance_type)
    # print('g_typ_name', g_typ_name)
    def impl(g_typ, goal, root):
        goals = Dict.empty(unicode_type, i8)

        _goals = Dict.empty(g_typ, ExplanationTreeType)
        _goals[goal] = root

        goals[g_typ_name] = _pointer_from_struct_incref(_goals)
        return goals
    return impl

@njit(cache=True)
def _init_subgoals():
    return Dict.empty(unicode_type,i8)

def build_explanation_tree(planner, g_typ, goal):
    root = expl_tree_ctor()
    goals = _init_root_goals(g_typ, goal, root)

    new_subgoals = retrace_goals_back_one(planner, goals)    
    while(new_subgoals is not None):
        new_subgoals = retrace_goals_back_one(planner, new_subgoals)   

    return root

    # for typ_str in goals_d:
    #     goals = goals_d[typ_str]
    #     expl_trees = expl_tree_d[typ_str]
    #     if(typ_str == 'float64'):
    #         new_arg_inds = retrace_arg_inds(planner, typ, goals)
    #         for typ, inds in arg_inds_d.items():
    #             # select_from_collection(kb.u_vs[typ],arg_inds[typ])
    #             select_from_collection(?,arg_inds[typ])
    #         print(goals_d)
    # print()

    # print(goals)
    # goals, expl_trees = retrace_back_one(planner, d_g_typ, g_typ_str, goals, expl_trees)
    # while(True):

#THINKING
'''
The output is an explanation tree the input is all 
of the record entries (linked list w/ op + args) for
everything. per-cycle:

input: collection of subgoals (each typed)
output: collection of new subgoals (each typed)  

So what if each call to retrace_back_one filled into a
dictionary of subgoals typ_name -> *Set[typ] and there
is a just an auto-generated function that takes in a 
*List[typ1] and injects various *Set[typ2] into that 
working set of new subgoals.


A new explanation tree instance needs to be added every time
a completely new value is added to the various *Set[typ2] collections
the explanation tree could be the Value, so: *Set[typ2] is really
a *Dict(typ2,ExplTreeType).

There is really no point in flattening the running set of next values
into a list or something, the typ_name -> (*Dict[typ2] -> ExplTree) dictionary
can just be passed along

So then we have 

retrace_back_one()
-inputs:
     planner : PlannerType
     goals : typ_name -> (*Dict[typ] -> ExplTree) 
     new_args : typ_name -> (*Dict[i8] -> ExplTree) or None
     record_extractors : typ_name -> 
        *FunctType(RecEntryType(*val_map, val[typ])) 

-outputs: 
     new_args : ... same instance

extract_subgoals()
-inputs: 
    planner : PlannerType
    new_args : typ_name -> (*Dict[i8] -> ExplTree) or None
    slice_funcs : typ_name -> 
        *FunctionType( (collection : *List[typ], args : , new_goals : ...,  ) )
-outputs: 
    new_subgoals : typ_name -> (*Dict[typ] -> ExplTree) or None



Ok a thought type specific execution is only necessary for:
    1: getting the head of the record in the val_map
    2: extracting the subgoal values using the args of the flat lists

We need to do 1 for the root goal and 1 then 2 thereafter




'''

### If I pass in the goals as a Dict(unicode_type, *List[any])
# and the expl_trees as just the expl_tree_d, then 

### Thinking Thinking
'''
Explanation Tree:
    -Pretty much needs to be organized top down
    -Each node needs to be unique to a particular value
        -other wise it is just as good to enumerate all solutions
    -To sample from it, take a random walk on uniform edges
    -To iterate over it randomly... ?
        -Do I need to enumerate all possibilities and then shuffle?
        -Or can I keep taking the random walk?
    -What does an iterator look like?
        -Needs to be a head stack array of inds that is incremented
            then wrapped and the parent incremented until complete. 
    -Can it be anti-unified/intersected?
        -Seems like yes, with a shared traversal that prunes anything
            not in common.

#Note: probably need to add a depth check inside the inner bits
# so that record_entries are not overwritten on deeper inferences
#  needs to also keep track of min_exploration_depth?
#  -yes because there are cases where a zero inference could prevent
#    a single inference 
Notes on incrementally updating the planner:
    -Should be possible, although will probably need to either 
        keep around a copy of each inference level or reconstruct a copy
        up to where the record entries show a deeper depth.
    -Depth zero will have one such val_map
    -When things are injected in from a kb they go into depth zero
    -There needs to be back connections to the kb elements 
    -Could probably use t_id, f_id, a_id the integer
'''

### More Thinking ### 
'''
So we need a standard way of writing things into the starting buffer 
of the planner it seems like having a bunch of Var instances be 
essentially the ops for those depth 0 values. This way we already have
a standard way of encoding deref instructions.
So then we need some kind of flag on the recEntry like is_op
it seems like this warrants a new structref TreeEntry to replace 
the tuple that is there now. We probably should not recycle the Record 
instance because it has a ref to 'data' which is big should get able to 
be freed even if the Explanation tree sticks around for a long time.

So we have TreeEntry:
-is_op : u1
-op : GenericOpType
-var : GenericVarType

'''





















