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
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template
from cre.memory import MemoryType, Memory, facts_for_t_id, fact_at_f_id
from cre.var import GenericVarType
# from cre.fact import define_fact, BaseFactType, cast_fact, DeferredFactRefType, Fact
from cre.utils import (ptr_t, _struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_ptr,  lower_setattr, lower_getattr,
                       _raw_ptr_from_struct, _decref_ptr, _incref_ptr, _incref_structref, _ptr_from_struct_incref,
                       _dict_from_ptr, _list_from_ptr, _load_ptr, _arr_from_data_ptr, _get_array_raw_data_ptr, _get_array_raw_data_ptr_incref)
from cre.utils import assign_to_alias_in_parent_frame, _raw_ptr_from_struct_incref
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.fact import Fact, gen_fact_import_str, get_offsets_from_member_types
from cre.var import Var, VarTypeTemplate
from cre.op import GenericOpType, OpTypeTemplate
# from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
#  get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from cre.make_source import make_source, gen_def_func, gen_assign, resolve_template, gen_def_class
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.experimental.function_type import _get_wrapper_address


from operator import itemgetter
from copy import copy
from os import getenv
from cre.utils import deref_type, listtype_sizeof_item
import inspect, dill, pickle
from textwrap import dedent, indent
import time


'''
This file implements the SetChainingPlanner which solves planning problems
akin to the "24 game" (given 4 numbers find a way to manipulate them to get 24)
The set chaining planner solves these problems for a given a set of values 
of essentially any type (number, string, object, etc...), a set of operations, 
and a goal value of any type, the planner will find compositions of starting
values and operations that achieve the goal state. This planner searches for
the goal value in a forward chaining manner by exploring every combination of
values and compositions of operations up to a fixed depth. For instance,
(1+3) = 4 could be discovered at depth 1, and (1+2)+(3-2) = 4 could be discovered
at a depth of 2. The planner is "set chaining" because it uses a hashmap to 
ensure that at each depth only unique values are used to compute the values for
the next depth. This cuts down on the combinatoric explosion of searching for the goal.

This file implements:
-SetChainingPlanner
    The constructor for a set chaining planner instance. The set of initial values can
    be declared to this structure. ??? can be used to solve the search problem, 
    and outputs the root node of an ExplanationTree containing the solutions found up 
    to a fixed depth. Every forward pass of the planner is stored in memory. After the goal
    is produced at a particular depth an operation composition is reconstructed by 
    filtering backwards from the goal state to the initial declared values.

-SC_Record
    An internal structure to the SetChainingPlanner representing the application
    of an operation (a cre.Op) over a particular range of the unique values calculated for a
    particular depth. An SC_Record is a lightweight form of bookeeping that is used
    to reconstruct solutions (i.e. operation compositions + starting values) in a 
    backwards fashion from the depth where the goal value was found. For every application
    of a cre.Op the that produces value 'v', the 'data' of an SC_Record 'rec' fills in a record entry:
        [*rec, *prev_entry, arg_ind0, arg_ind1, ...] where '*rec' is a pointer to the SC_Record
    '*prev_entry' is the previous record entry associated with value 'v' (or zero if there wasn't one yet),
    and arg0,arg1,arg2,... are indicies for the arguments to this application of operation. To limit
    the memory/cache footprint of tracking each entry---there can be millions at deeper depths---these 
    all entry information is encoded in a contigous preallocated array.  
     
-ExplanationTree
    An explanation tree is a compact datastructure for holding all solutions to a SetChainingPlanner's 
    search. An ExplanationTree can hold a large number of solutions to a search (essentially compositions of 
    cre.Op instances) in a manner that is considerably more compact than a flat list of (possibly many thousands 
    of) solutions. An explanation tree is not a proper tree. Each node consists of multiple entries, each of which 
    represents the application of a cre.Op or terminal variable and has 'n' child ExplanationTrees, for each 
    of its 'n' arguments. Thus, unlike a normal tree which has a fixed set of children per node, a ExplanationTree 
    instance contains a set of sets of child ExplanationTrees. A fixed choice of entries starting from the root 
    ExplanationTree down to terminal entries represents a single cre.Op composition.

-ExplanationTreeEntry
    Represents a non-terminal cre.Op ('is_op'==True) in a composition or a terminal cre.Var ('is_op'==False).
    In the non-terminal case holds a reference to a cre.Op instance and child ExplanationTrees for each argument. 
    In the terminal case keeps a reference to a cre.Var.
'''

class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.2f} ms')


SC_Record_field_dict = {
    # Holds encoded record entries. Essentially [*rec, *prev_entry, arg_ind0, arg_ind1, ...]
    #  *rec and *prev_entry, are each distributed across two entries in this u4 array. 
    'data' : u4[::1],

    # List of ranges (start,end) over which the op has been applied for each argument.
    'stride' : i8[:,::1],

    # Whether the record is non-terminal---from applying a cre.Op and not from a declared value.
    'is_op' : u1,

    # If non-terminal then holds the cre.Op
    'op' : GenericOpType,

    # The number of input arguments for self.op.
    'nargs' : i8,

    # If terminal then holds a cre.Var as a placeholder for the declared value. 
    'var' : GenericVarType,

    # The depth at which the op was applied/inserted, or '0' for declared values
    'depth' : i8,
}

SC_Record_fields = [(k,v) for k,v in SC_Record_field_dict.items()]
SC_Record, SC_RecordType = \
    define_structref("SC_Record", SC_Record_fields, define_constructor=False)

# @njit(cache=True)
# def _sc_record_ctor_helper(depth, nargs):
#     st = new(SC_RecordType)
#     st.data = data
#     st.stride = stride
#     st.depth = depth
#     st.nargs = nargs
#     return st

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
def overload_SC_Record(op_or_var, depth=0, nargs=0, stride=None):
    '''Implements the constructor for an SC_Record'''
    if(isinstance(op_or_var, OpTypeTemplate)):
        def impl(op_or_var, depth=0, nargs=0, stride=None):
            st = new(SC_RecordType)
            st.is_op = True
            st.op = op_or_var
            st.depth = depth
            st.nargs = nargs
            
            if(stride is not None): st.stride = stride

            ENTRY_WIDTH = 4 + nargs
            n_items = 1
            for s_i in stride:
                s_len = s_i[1]-s_i[0]
                n_items *= s_len
            data_len = n_items*ENTRY_WIDTH
            st.data = np.empty(data_len,dtype=np.uint32)
            
            return st
    elif(isinstance(op_or_var, VarTypeTemplate)):
        def impl(op_or_var, depth=0, nargs=0, stride=None):
            st = new(SC_RecordType)
            st.is_op = False
            st.var = _cast_structref(GenericVarType, op_or_var) 
            st.depth = depth
            st.nargs = 0

            st.data = np.empty((2,),dtype=np.uint32)
            self_ptr = _raw_ptr_from_struct(st)
            st.data[0] = u4(self_ptr) # get low bits
            st.data[1] = u4(self_ptr>>32) # get high bits
            
            
            return st
    else:
        print('fail')
    return impl

# SC_Record_Entry_field_dict = {
#     'rec' : SC_RecordType,
#     'next_entry_ptr' : i8,
#     'args' : u4[::1],
# }
# SC_Record_Entry_fields = [(k,v) for k,v in SC_Record_Entry_field_dict.items()]
# SC_Record_Entry, SC_Record_EntryType = \
#     define_structref("SC_Record_Entry", SC_Record_Entry_fields)

# @njit(cache=True)
# def rec_entry_from_ptr(d_ptr):
#     ptrs = _arr_from_data_ptr(d_ptr, (2,),dtype=np.int64)
#     rec_ptr, next_entry_ptr = ptrs[0], ptrs[1]
#     rec = _struct_from_ptr(SC_RecordType, rec_ptr)
#     args = _arr_from_data_ptr(d_ptr+16,(rec.nargs,),dtype=np.uint32)

#     return SC_Record_Entry(rec,next_entry_ptr,args)

@njit(cache=True)
def extract_rec_entry(d_ptr):
    '''Decodes the record entry at a pointer to its location in the underlaying data of 
        an SC_Record.data array and outputs the SC_Record instance, next data pointer, 
        and argument indicies.'''
    ptrs = _arr_from_data_ptr(d_ptr, (2,),dtype=np.int64)
    rec_ptr, next_entry_ptr = ptrs[0], ptrs[1]
    # print(rec_ptr, next_entry_ptr)
    # if(next_entry_ptr <= 4): raise ValueError()
    rec = _struct_from_ptr(SC_RecordType, rec_ptr)
    args = _arr_from_data_ptr(d_ptr+16,(rec.nargs,),dtype=np.uint32)
    return rec, next_entry_ptr, args


# @njit(cache=True)
# def next_rec_entry(re):
#     if(re.next_entry_ptr != 0):
#         return rec_entry_from_ptr(re.next_entry_ptr)
#     else:
#         return None


record_list_type = ListType(SC_RecordType)
dict_str_to_record_list_type = DictType(unicode_type, record_list_type)
str_int_tuple = Tuple((unicode_type,i8))

SetChainingPlanner_field_dict = {
    'ops': ListType(GenericOpType),
    # List of dictionaries that map:
    #  Tuple(type_str[str],depth[int]) -> ListType[Record])
    'declare_records' : DictType(unicode_type, ListType(SC_RecordType)),

    'forward_records' : ListType(DictType(unicode_type, ListType(SC_RecordType))),

    # List of dictionaries that map:
    #  Tuple(type_str[str],depth[int]) -> ListType[Record])
    'backward_records' : ListType(DictType(unicode_type, ListType(SC_RecordType))),

    # Maps type_str[str] -> *(Dict: val[any] -> Tuple(depth[i8], *SC_Record_Entry) )
    'val_map_ptr_dict' : DictType(unicode_type, ptr_t),

    # Maps type_str[str] -> *(Dict: *Var -> val[any])
    'inv_val_map_ptr_dict' : DictType(unicode_type, ptr_t), 

    # Maps (type_str[str],depth[int]) -> *Iterator[any]
    'flat_vals_ptr_dict' : DictType(Tuple((unicode_type,i8)), ptr_t),

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

    def declare(self,val):
        return planner_declare(self, val)

    # def forward_chain_one(self,val):


define_boxing(SetChainingPlannerTypeTemplate,SetChainingPlanner)
SetChainingPlannerType = SetChainingPlannerTypeTemplate(SetChainingPlanner_fields)


@njit(cache=True)
def sc_planner_ctor():
    st = new(SetChainingPlannerType)
    st.declare_records = Dict.empty(unicode_type, record_list_type)
    st.forward_records = List.empty_list(dict_str_to_record_list_type)
    st.backward_records = List.empty_list(dict_str_to_record_list_type)
    st.val_map_ptr_dict = Dict.empty(unicode_type, ptr_t)
    st.inv_val_map_ptr_dict = Dict.empty(unicode_type, ptr_t)
    st.flat_vals_ptr_dict = Dict.empty(str_int_tuple, ptr_t)
    return st


@njit(cache=True)
def ensure_ptr_dicts(planner,typ,typ_name,lt,vt,ivt):
    tup = (typ_name,0)
    if(tup not in planner.flat_vals_ptr_dict):
        flat_vals = List.empty_list(typ)
        planner.flat_vals_ptr_dict[tup] = _ptr_from_struct_incref(flat_vals)
    if(typ_name not in planner.val_map_ptr_dict):
        val_map = Dict.empty(typ, i8_2x_tuple)    
        planner.val_map_ptr_dict[typ_name] = _ptr_from_struct_incref(val_map)
    if(typ_name not in planner.inv_val_map_ptr_dict):
        inv_val_map = Dict.empty(i8, typ)    
        planner.inv_val_map_ptr_dict[typ_name] = _ptr_from_struct_incref(inv_val_map)

    if(typ_name not in planner.declare_records):
        planner.declare_records[typ_name] = List.empty_list(SC_RecordType)        


    flat_vals = _list_from_ptr(lt, planner.flat_vals_ptr_dict[tup])    
    val_map = _dict_from_ptr(vt, planner.val_map_ptr_dict[typ_name])    
    inv_val_map = _dict_from_ptr(ivt, planner.inv_val_map_ptr_dict[typ_name])    
    declare_records = planner.declare_records[typ_name]
    return flat_vals, val_map, inv_val_map, declare_records

    



@generated_jit(cache=True)
def planner_declare(planner, val):
    '''Declares a value into the 0th depth of planner'''
    val_typ = val
    val_typ_name = str(val_typ)

    l_typ = ListType(val_typ)
    vm_typ = DictType(val_typ, i8_2x_tuple)
    ivm_typ = DictType(i8, val_typ)

    def impl(planner, val):
        # pass
        flat_vals, val_map, inv_val_map, declare_records = \
            ensure_ptr_dicts(planner, val_typ,
                val_typ_name, l_typ, vm_typ, ivm_typ)
        v = Var(val_typ)
        rec = SC_Record(v)
        var_ptr = _raw_ptr_from_struct(v)
        declare_records.append(rec)
        rec_entry_ptr = _get_array_raw_data_ptr(rec.data)
        # rec.data[0] = _raw_ptr_from_struct(rec)
        # rec_entry = np.empty((1,),dtype=np.int64)
        # rec_entry[0] = 
        
        flat_vals.append(val)
        val_map[val] = (0, rec_entry_ptr)
        inv_val_map[var_ptr] = val
    return impl

# @generated_jit(cache=True)
def gen_declare_attr_impl(attr_typ):
    context = cre_context()
    pass

    # def impl(planner, val)





def planner_declare_fact(planner, val, attrs_as_types=None):
    print(attrs_as_types)
    def impl(planner, val, attrs_as_types=None):
        pass
    return impl



        

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



from cre.core import standardize_type
def search_for_explanations(self, goal, ops=None,
             search_depth=1, max_solutions=10,
             min_stop_depth=-1,context=None):
    
    with PrintElapse("Start"):
        context = cre_context(context)
        if(min_stop_depth == -1): min_stop_depth = search_depth
        g_typ = standardize_type(type(goal), context)
    # g_typ_str = str(g_typ)
    # print(g_typ,g_typ_str)

    # NOTE: Make sure that things exist here
    with PrintElapse("Chain"):
        for depth in range(1, search_depth+1):
            if(depth < self.curr_infer_depth): continue
            found_at_depth = query_goal(self, g_typ, goal)
            print("found_at_depth:", found_at_depth, self.curr_infer_depth)
            
            if(found_at_depth is None or 
                self.curr_infer_depth < min_stop_depth):
                forward_chain_one(self,ops)

    with PrintElapse("Build Expl Tree"):
        expl_tree = build_explanation_tree(self, g_typ, goal)
    return expl_tree



@generated_jit(cache=True, nopython=True)
def query_goal(self, typ, goal):
    _typ = typ.instance_type
    dict_typ = DictType(_typ,i8_2x_tuple)
    typ_name = str(_typ)
    def impl(self, typ, goal):
        # vtd_ptr_d = self.val_map
        if(typ_name in self.val_map_ptr_dict):
            val_map = _dict_from_ptr(dict_typ, self.val_map_ptr_dict[typ_name])
            if(goal in val_map):
                depth, _ = val_map[goal]
                return depth
            else:
                return None
        else:
            return None
    return impl

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


i8_2x_tuple = Tuple((i8,i8))
@generated_jit(cache=True, nopython=True)
def join_records_of_type(self, depth, typ):
    _typ = typ.instance_type
    dict_typ = DictType(_typ,i8_2x_tuple)
    typ_name = str(_typ)
    def impl(self, depth, typ):
        ''' Goes through every record at 'depth' and joins them
            so that 'val_to_depth' and 'flat_vals' are defined '''
        # recs = self.forward_records[depth][typ_name]
        val_map = _dict_from_ptr(dict_typ, self.val_map_ptr_dict[typ_name])
        flat_vals = List.empty_list(typ, len(val_map))
        for val in val_map:
            flat_vals.append(val) 

        tup = (typ_name, depth)
        #If (typ_name, depth) was already there then decref the
        # entry in flat_vals_ptr_dict so it gets freed
        # if(tup in self.flat_vals_ptr_dict):
        #     _decref_ptr(self.flat_vals_ptr_dict[tup])
        self.flat_vals_ptr_dict[tup] = \
             _ptr_from_struct_incref(flat_vals)
    return impl


def join_records(self, depth, ops):
    typs = set([op.signature.return_type for op in ops])
    for typ in typs:
        val_to_depth = join_records_of_type(self, depth, typ)

# Don't njit 
# Note: could probably not run all ops
def forward_chain_one(self, ops=None):
    '''Applies 'ops' on all current inferred values'''
    nxt_depth = self.curr_infer_depth+1
    if(ops is None): ops = self.ops
    for op in ops:
        rec = apply_multi(op, self, self.curr_infer_depth)
        if(rec is not None):
            insert_record(self, rec, op.return_type_name, nxt_depth)
    join_records(self, nxt_depth, ops)
    self.curr_infer_depth = nxt_depth

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
    '''Generates source code for an apply_multi() implementation for a cre.Op'''
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
from numba.types import ListType, DictType, Tuple
import numpy as np
import dill
from cre.utils import _dict_from_ptr, _list_from_ptr, _raw_ptr_from_struct, _get_array_raw_data_ptr
from cre.sc_planner2 import SC_Record
''' 
    imp_targets = ['call'] + (['check'] if has_check else [])
    src += f'''{gen_import_str(type(op).__name__,
                 op.long_hash, imp_targets)}\n\n'''

    src += "".join([f'typ{i}'+", " for i in range(len(typs))]) + \
             f'= dill.loads({dill.dumps(tuple(typs.keys()))})\n'

    src += f'''ret_typ = dill.loads({dill.dumps(sig.return_type)})
ret_d_typ = DictType(ret_typ,Tuple((i8,i8)))
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
    # l_defs = '\n'.join([f'l{i} = stride[{i}][1]-stride[{i}][0]' for i in a_cnt])#", ".join([f"len(iter{it_inds[i]})-start{i}" for i in a_cnt])
    stride_defaults = ",".join([f'[0,len(iter{it_inds[i]})]' for i in a_cnt])
    src += indent(f'''
nxt_depth = depth + 1
stride = np.array([{stride_defaults}],dtype=np.int64)
val_map =  _dict_from_ptr(ret_d_typ,
{ind}planner.val_map_ptr_dict['{str(sig.return_type)}'])

rec = SC_Record(op_inst, nxt_depth, N_ARGS, stride)
data = rec.data
d_ptr = _get_array_raw_data_ptr(data)
rec_ptr = _raw_ptr_from_struct(rec)

d_offset=0
val_map_defaults = (-1,0)
''',prefix=ind)
    c_ind = copy(ind)
    for i, arg in enumerate(args):
        src += f'{c_ind}for i{i} in range(stride[{i}][0],stride[{i}][1]):\n'
        c_ind += ind
        if(i in op.right_commutes):
            ignore_conds = 'or'.join([f"i{i} > i{j}" for j in op.right_commutes[i]])
            src += f'{c_ind}if({ignore_conds}): continue\n'
        

    _is = ",".join([f"i{i}" for i in a_cnt])
    _as = ",".join([f"a{i}" for i in a_cnt])
    params = ",".join([f"iter{it_inds[i]}[i{i}]" for i in a_cnt])
    arg_assigns = "\n".join([f"data[d_offset+{4+i}] = i{i}" for i in a_cnt])

    src += indent(f'''{_as} = {params}
{f'if(not check({_as})): continue' if has_check else ""}
v = call({_as})


prev_depth, prev_entry = val_map.get(v, val_map_defaults)
if(prev_depth != -1 and prev_depth < nxt_depth): continue

data[d_offset +0] = u4(rec_ptr) # get low bits
data[d_offset +1] = u4(rec_ptr>>32) # get high bits
data[d_offset +2] = u4(prev_entry) # get low bits
data[d_offset +3] = u4(prev_entry>>32)# get high bits

#Put arg inds at the end
{arg_assigns}

val_map[v] = (nxt_depth, d_ptr + d_offset*4)
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
        val_map = Dict.empty(typ, i8_2x_tuple)
        val_map_ptr = _ptr_from_struct_incref(val_map)
        self.val_map_ptr_dict[typ_name] = val_map_ptr
    
    


def apply_multi(op, planner, depth):
    '''Applies 'op' at 'depth' and returns the SC_Record'''

    # If it doesn't already exist generate and inject '_apply_multi' into 'op'
    if(not hasattr(op,'_apply_multi')):
        hash_code = unique_hash(['_apply_multi',op.long_hash])  
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
    # Whether the entry is non-terminal and represents the application of a cre.Op. 
    "is_op" : u1,

    # If non-terminal the cre.Op applied
    "op" : GenericOpType,

    # If terminal the cre.Var instance
    "var" : GenericVarType,

    # If terminal the cre.Var instance
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
            st.var = _cast_structref(GenericVarType, op_or_var) 
            return st
    return impl




### Explanation Tree ###

ExplanationTree_field_dict = {
    'entries' : ListType(ExplanationTreeEntryType),
    'inv_val_map_ptr_dict' : DictType(unicode_type, ptr_t)
}
ExplanationTree_fields = [(k,v) for k,v in ExplanationTree_field_dict.items()]
# ExplanationTree, ExplanationTreeType = \
#     define_structref("ExplanationTree", ExplanationTree_fields, define_constructor=False)

@structref.register
class ExplanationTreeTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class ExplanationTree(structref.StructRefProxy):
    def __new__(cls):
        self = expl_tree_ctor()
        return self

    def __iter__(self):
        return ExplanationTreeIter(self)

@generated_jit(cache=True)
def read_inv_val_map(expl_tree, var_ptr, typ):
    _typ = typ.instance_type
    typ_name = str(_typ)
    ivm_typ = DictType(i8, _typ)
    def impl(expl_tree, var_ptr, typ):
        inv_val_map_ptr_dict = expl_tree.inv_val_map_ptr_dict
        inv_val_map = _dict_from_ptr(ivm_typ, inv_val_map_ptr_dict[typ_name])
        return inv_val_map[var_ptr]
    return impl


class ExplanationTreeIter():
    def __init__(self, expl_tree):
        self.expl_tree = expl_tree
        self.gen = gen_op_comps_from_expl_tree(expl_tree)

    def __iter__(self):
        return self
    def __next__(self):
        op_comp = next(self.gen)
        vals = []
        for var_ptr, val_typ in zip(op_comp.base_vars, op_comp.signature.args):
            vals.append(read_inv_val_map(self.expl_tree, var_ptr, val_typ))
        
        return op_comp, vals




define_boxing(ExplanationTreeTypeTemplate,ExplanationTree)
ExplanationTreeType = ExplanationTreeTypeTemplate(ExplanationTree_fields)

@njit(cache=True)
def expl_tree_ctor(entries=None, planner=None):
    st = new(ExplanationTreeType)
    if(entries is None):
        st.entries = List.empty_list(ExplanationTreeEntryType)
    else:
        st.entries = entries
    if(planner is not None):
        st.inv_val_map_ptr_dict = Dict.empty(unicode_type, ptr_t)
        for typ_name, ptr in planner.inv_val_map_ptr_dict.items():
            _incref_ptr(ptr) #Note might not need
            st.inv_val_map_ptr_dict[typ_name] = ptr
    return st

# lst_tree_entries = ListType(ExplanationTreeEntryType)
# @njit(cache=True)
# def consolidate_tree(tree):

#     op_ptr_children = Dict.empty(i8, lst_tree_entries)
#     for child in tree.children:
#         if(child.is_op):
#             ptr = _raw_ptr_from_struct(child.op)
#             if(ptr in op_ptr_children):
#                 lst = op_ptr_children[ptr]
#             else:
#                 lst = List.empty_list(ExplanationTreeEntryType)                
#                 op_ptr_children[ptr] = lst
#             lst.append(child)






i8_et_dict = DictType(i8,ExplanationTreeType)

@njit(void(
        i8,
        DictType(unicode_type,i8_et_dict),
        ExplanationTreeType),
     cache=True)
def _fill_arg_inds_from_rec_entries(re_ptr, new_arg_inds, expl_tree):
    '''Goes through linked list of record entries and
        adds the argument indicies into new_arg_inds
        with a new instance of an ExplanationTree if needed.
        Add the new ExplanationTrees to the children of 'expl_tree'
    '''

    while(re_ptr != 0):
        # print("OKAY")
        re_rec, re_next_re_ptr, re_args = extract_rec_entry(re_ptr)
        # print("DONE")
        if(re_rec.is_op):
            op = re_rec.op 
            child_arg_ptrs = np.empty(len(re_args), dtype=np.int64)
            for i, (arg_type_name, arg_ind) in enumerate(zip(op.arg_type_names, re_args)):
                #Make sure a set of indicies has been instantied for 'arg_type_name'
                if(arg_type_name not in new_arg_inds):
                    new_arg_inds[arg_type_name] = Dict.empty(i8,ExplanationTreeType)
                uai = new_arg_inds[arg_type_name]

                # Fill in a new ExplanationTree instance if needed
                if(arg_ind not in uai):
                    uai[arg_ind] = expl_tree_ctor()

                child_arg_ptrs[i] = _raw_ptr_from_struct_incref(uai[arg_ind])

            # Throw new tree entry instance into the children of 'expl_tree'
            entry = expl_tree_entry_ctor(op, child_arg_ptrs)
            expl_tree.entries.append(entry)
            # re = next_rec_entry(re)  
            re_ptr = re_next_re_ptr
            # re_rec, re_next_re, re_args = extract_rec_entry(re_next_re)
        else:
            entry = expl_tree_entry_ctor(re_rec.var)
            expl_tree.entries.append(entry)
            # re = None
            re_ptr = 0


@generated_jit(cache=True, nopython=True) 
def retrace_arg_inds(planner, typ,  goals, new_arg_inds=None):
    '''Find applications of operations that resulted in each 
        goal in goals. Add the indicies of the args as they
        occur in flat_vals.
    '''
    # The actual type inside the type ref 'typ'
    _typ = typ.instance_type

    val_map_d_typ = DictType(_typ, i8_2x_tuple) 
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
            _, entry_ptr = val_map[goal]
            # re = rec_entry_from_ptr(entry_ptr)
            _fill_arg_inds_from_rec_entries(entry_ptr,
                new_arg_inds, expl_tree)
        return new_arg_inds
    return impl    


@generated_jit(cache=True,nopython=True) 
def fill_subgoals_from_arg_inds(planner, arg_inds, typ, depth, new_subgoals):
    '''Fill 'new_subgoals' with the actual values pointed to by
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
        # Inject the new subgoals for 'typ' into 'new_subgoals'
        new_subgoals[typ_name] = _ptr_from_struct_incref(_new_subgoals)
        return new_subgoals
    return impl


def retrace_goals_back_one(planner, goals):
    # print("RETRACE")
    new_arg_inds = None
    for typ_name in goals:
        # fix later 
        typ = f8 if typ_name == 'float64' else unicode_type
        new_arg_inds = retrace_arg_inds(planner, typ, goals, new_arg_inds)
    if(len(new_arg_inds) == 0):
        return None
    new_subgoals = _init_subgoals()
    for typ_name in new_arg_inds:
        # fix later 
        typ = f8 if typ_name == 'float64' else unicode_type
        new_subgoals = fill_subgoals_from_arg_inds(
                planner, new_arg_inds, typ,
                planner.curr_infer_depth, new_subgoals)

    # print(new_subgoals)
    return new_subgoals

@generated_jit(cache=True)
def _init_root_goals(g_typ, goal, root):
    g_typ_name = str(g_typ.instance_type)
    # print('g_typ_name', g_typ_name)
    def impl(g_typ, goal, root):
        goals = Dict.empty(unicode_type, ptr_t)

        _goals = Dict.empty(g_typ, ExplanationTreeType)
        _goals[goal] = root

        goals[g_typ_name] = _ptr_from_struct_incref(_goals)
        return goals
    return impl

@njit(cache=True)
def _init_subgoals():
    return Dict.empty(unicode_type, ptr_t)

def build_explanation_tree(planner, g_typ, goal):
    root = expl_tree_ctor(None,planner)
    goals = _init_root_goals(g_typ, goal, root)
    new_subgoals = retrace_goals_back_one(planner, goals)    
    retrace_depth = planner.curr_infer_depth-1
    while(new_subgoals is not None):
        if(retrace_depth < 0):
            raise RecursionError("Retrace exceeded current inference depth.")
        new_subgoals = retrace_goals_back_one(planner, new_subgoals)   
        retrace_depth -= 1
    
    return root


# IterData_field_dict = {
#     "inds" : i8[::1],
#     # "head_ind" : i8,
#     "expl_tree" : ExplanationTreeType
#     #Holds dictionaries of dictionaries where 
#     # "iter_data" : DictType(i8, )
# }
# IterData_fields = [(k,v) for k,v in IterData_field_dict.items()]
# IterData, IterDataType = \
#     define_structref("IterData", IterData_fields, define_constructor=True)



# from cre.akd import AKDType, new_akd
# ExplanationTreeIter_field_dict = {
#     "root" : ExplanationTreeType,
#     # "iter_data_map" : AKDType(i8,IterData),
#     "heads" : i8[::1]
# }
# ExplanationTreeIter_fields = [(k,v) for k,v in ExplanationTreeIter_field_dict.items()]
# ExplanationTreeIter, ExplanationTreeIterType = \
#     define_structref("ExplanationTreeIter", ExplanationTreeIter_fields, define_constructor=False)


# def expl_tree_iter_ctor(expl_tree,random=False):
#     st = new(ExplanationTreeType)
#     st.root = expl_tree
#     st.iter_data_map = new_akd(i8, IterData)
#     st.heads = np.empty((0,),dtype=np.int64)

#     n_options = len(expl_tree.children)
#     st.iter_data_map[st.heads] = IterData(np.arange(n_options), expl_tree)



# def expl_tree_iter_random_next(it):
#     pass


@njit(i8(ExplanationTreeType,),cache=True)
def expl_tree_num_entries(tree):
    return len(tree.entries)

@njit(ExplanationTreeEntryType(ExplanationTreeType,i8),cache=True)
def expl_tree_ith_entry(tree, i):
    return tree.entries[i]

@njit(i8(ExplanationTreeEntryType),cache=True)
def expl_tree_entry_num_args(tree_entry):
    return len(tree_entry.child_arg_ptrs)

@njit(u1(ExplanationTreeEntryType),cache=True)
def expl_tree_entry_is_op(tree_entry):
    return tree_entry.is_op

@njit(GenericOpType(ExplanationTreeEntryType),cache=True)
def expl_tree_entry_get_op(tree_entry):
    return tree_entry.op

@njit(GenericVarType(ExplanationTreeEntryType),cache=True)
def expl_tree_entry_get_var(tree_entry):
    return tree_entry.var

@njit(ExplanationTreeType(ExplanationTreeEntryType,i8),cache=True)
def expl_tree_entry_jth_arg(tree_entry, j):
    return _struct_from_ptr(ExplanationTreeType,
        tree_entry.child_arg_ptrs[j])


def product_of_generators(generators):
    '''Takes a list of generators and applies the equivalent of
        itertools.product() on them. Has significantly less memory
        overhead in cases when you only need a subset of the full product.
    '''
    iters = []
    out = []
    
    while(True):
        # Create any missing iterators from generators
        while(len(iters) < len(generators)):
            it = generators[len(iters)]()
            iters.append(it)
        
        iter_did_end = False
        while(len(out) < len(iters)):
            # Try to fill in any missing part of out
            try:
                nxt = next(iters[len(out)])
                out.append(nxt)
            # If any of the iterators failed pop up an iterator
            except StopIteration as e:
                # Stop yielding when 0th iter reaches end
                if(len(iters) == 1):
                    return
                out = out[:-1]
                iters = iters[:-1]
                iter_did_end = True
        
        # If any of the iterators reached their end then 
        #  we'll need to generate one or more new ones.
        if(iter_did_end): continue

        yield out
        out = out[:-1]


class ExplTreeGen():
    '''Helper object that is essentially a lambda that applies
        gen_op_comps_from_expl_tree() on a particular ExplanationTree
    '''
    def __init__(self,child_tree):
        self.child_tree = child_tree
    def __call__(self):
        return gen_op_comps_from_expl_tree(self.child_tree)
        
from cre.op import OpComp
def gen_op_comps_from_expl_tree(tree):
    '''A generator of OpComps from an ExplanationTree'''
    for i in range(expl_tree_num_entries(tree)):
        tree_entry = expl_tree_ith_entry(tree, i)
        
        if(expl_tree_entry_is_op(tree_entry)):
            op = expl_tree_entry_get_op(tree_entry)
            # print(op)
            # op = op.recover_singleton_inst()
            # print(op)
            child_generators = []
            for j in range(expl_tree_entry_num_args(tree_entry)):
                child_tree = expl_tree_entry_jth_arg(tree_entry,j)
                child_gen = ExplTreeGen(child_tree)
                child_generators.append(child_gen)

            for args in product_of_generators(child_generators):                
                op_comp = OpComp(op, *args)
                yield op_comp
        else:
            v = expl_tree_entry_get_var(tree_entry)
            yield v


    # while()
    # head_data = st.iter_data_map[st.heads]
    
    # inds = head_data.inds
    # istructions = List.empty_list(GenericOpType)
    # if(len(inds) > 0):
    #     options = head_data.expl_tree.children
    #     ind = inds[0]
    #     head_data.inds = inds[1:]

    #     L = len(st.heads)
    #     new_heads = np.empty((L+1,), dtype=np.int64)
    #     new_heads[:L] = st.heads
    #     new_heads[L] = ind

    #     st.heads = new_heads
    #     st.iter_data_map[st.heads] = IterData()


         
    # else:

### THinking Thinking 
'''
When the fact is declared we need to rip off things and cast them
The casting of them will cause the expr to be ugly
-idea: Some types i.e. float, int, unicode_type can be casted between
    so if the type is wrong then it simply injects the casting when 
    flattening. check will invariably need to add explicit casts in

-alternatively we simply add str() or float() but then repr doesn't work
 because str(op_comp/var) is literally a string. 

-could also add a cast(typ, val) option that generates/recovers an op
  that makes a new cast

-TThoughts: 
Need to try to cast on declare, or try casting all reasonable 
    types on forward chain. The latter case would have some elegance
    but would be less performant. Frankly it may be necessary. 

-Casting could overloaded and built into the ops themselves either:
    -Explicitly with more than one signature, or a valid arg casting dict

THINGS:
1) Things need to be forced to be declared as the proper type
    -For the time being, just 



'''

###

###TODO TODO
'''
[x] Recover values from Vars
[x] Decalaration
[ ] Auto declaration:
   -How should that work?
   -Probably should just hook up to wm
   -Have flag on attribute that is like: "visible" 

[x]. Commuting: steal from numbert 
4. Mute Exception: steal from numbert 

5. Rename:
-mem -> WM 
-fact_type -> base_type

'''


##THINKING 
'''
Head depth, head stack (can be iter datas) 
'''

#THINKING 
# Do I want an iterator, or some other kind of structure
# Do I want to consolidate the tree into op comps plus 
#  bindings? 
# What kinds of things should I do with the expl tree?
#   -I should be able to know its size 
#   -I should be able to iterate over it
#   -I should be able to sample from it
#     -But you really need to enumerate all the possibilities
#       in order to sample without replacement
#     -Or you can randomize the iterator traversal order 


    # for typ_str in goals_d:
    #     goals = goals_d[typ_str]
    #     expl_trees = expl_tree_d[typ_str]
    #     if(typ_str == 'float64'):
    #         new_arg_inds = retrace_arg_inds(planner, typ, goals)
    #         for typ, inds in arg_inds_d.items():
    #             # select_from_collection(mem.u_vs[typ],arg_inds[typ])
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
    -When things are injected in from a mem they go into depth zero
    -There needs to be back connections to the mem elements 
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





















