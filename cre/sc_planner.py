import operator
import numpy as np
import numba
from numba.core.dispatcher import Dispatcher
from numba import types, njit, f8, i8, u8, i4, u4, u2, u1, literally, generated_jit, void, literal_unroll, objmode
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from cre.caching import gen_import_str, unique_hash_v, import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template
from cre.var import VarType, var_ctor_generic
from cre.utils import (cast, ptr_t,  decode_idrec, lower_getattr, lower_setattr, lower_getattr,
                        _decref_ptr, _incref_ptr, _incref_structref, _ptr_from_struct_incref,
                       _dict_from_ptr, _list_from_ptr, _load_ptr, _arr_from_data_ptr, _get_array_raw_data_ptr, _get_array_raw_data_ptr_incref)
from cre.utils import assign_to_alias_in_parent_frame, _raw_ptr_from_struct_incref, _func_from_address, _list_base_from_ptr, _listtype_sizeof_item
from cre.vector import VectorType
from cre.fact import Fact, gen_fact_import_str, get_offsets_from_member_types
from cre.fact_intrinsics import fact_lower_getattr
from cre.var import Var, VarTypeClass, var_append_deref, var_extend
# from CREFunc import CREFuncType, CREFuncTypeClass
from cre.func import CREFuncType, CREFuncTypeClass, CREFunc, cre_func_call_self, set_base_arg_val_impl, get_return_val_impl, CFSTATUS_ERROR, CFSTATUS_TRUTHY
from cre.make_source import make_source, gen_def_func, gen_assign, resolve_template, gen_def_class
from cre.obj import CREObjType
from cre.core import T_ID_FUNC, T_ID_VAR, T_ID_INT, T_ID_FLOAT, T_ID_STR
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.experimental.function_type import _get_wrapper_address

from cre.func import cre_func_deep_copy_generic, set_var_arg, set_func_arg, reinitialize


from operator import itemgetter
from copy import copy
from os import getenv
from cre.utils import deref_info_type, listtype_sizeof_item, _tuple_from_data_ptrs
import inspect, cloudpickle
from textwrap import dedent, indent
import time
# Ensure that dynamic hash/eq set
import cre.dynamic_exec
import warnings
import itertools
from cre.default_funcs import Identity


'''
This file implements the SetChainingPlanner which solves planning problems
akin to the "24 game" (given 4 numbers find a way to manipulate them to get 24)
The set chaining planner solves these problems for a given a set of values 
of essentially any type (number, string, object, etc...), a set of functions, 
and a goal value of any type, the planner will find compositions of starting
values and functions that achieve the goal state. This planner searches for
the goal value in a forward chaining manner by exploring every combination of
values and compositions of functions up to a fixed depth. For instance,
(1+3) = 4 could be discovered at depth 1, and (1+2)+(3-2) = 4 could be discovered
at a depth of 2. The planner is "set chaining" because it uses a hashmap to 
ensure that at each depth only unique values are used to compute the values for
the next depth. This can significantly cut down on the combinatorics explosion 
of searching for the goal.

This file implements:
-SetChainingPlanner
    The constructor for a set chaining planner instance. The set of initial values can 
    be declared to this structure. The method self.search_for_explanations() can be used to solve 
    the search problem, and outputs the root node of an ExplanationTree containing the solutions found 
    up to a fixed depth. Every forward pass of the planner is stored in memory. After the goal
    is produced at a particular depth a function composition is reconstructed by 
    filtering backwards from the goal state to the initial declared values.

-SC_Record
    An internal structure to the SetChainingPlanner representing the application
    of a function (a CREFunc) over a particular range of the unique values calculated for a
    particular depth. An SC_Record is a lightweight form of bookeeping that is used
    to reconstruct solutions (i.e. function compositions + starting values) in a 
    backwards fashion from the depth where the goal value was found. For every application
    of a CREFunc the that produces value 'v', the 'data' of an SC_Record 'rec' fills in a record entry:
        [*rec, *prev_entry, arg_ind0, arg_ind1, ...] where '*rec' is a pointer to the SC_Record
    '*prev_entry' is the previous record entry associated with value 'v' or zero if there wasn't one yet,
    and arg_ind0,arg_ind1,... are indicies for the arguments to this application of the function. Note that,
    we keep around *prev_entry in order to essentially build a linked list of entries associated with each 
    unique value. To limit the memory/cache footprint of tracking each entry (there can be millions at deeper 
    depths) all entry information is encoded in a contigous preallocated array. Consequently *rec, *prev_entry,
    are weak (i.e. not refcounted pointers). 
     
-ExplanationTree
    An explanation tree is a compact datastructure for holding all solutions to a SetChainingPlanner's 
    search. An ExplanationTree can hold a large number of solutions to a search (essentially compositions of 
    CREFunc instances) in a manner that is considerably more compact than a flat list of (possibly many thousands 
    of) solutions. An explanation tree is not a proper tree, it is more akin to an option-tree. Each node consists of multiple 
    entries, each of which represents the application of a CREFunc or terminal variable and has 'n' child ExplanationTrees, 
    for each  of its 'n' arguments. Thus, unlike a normal tree which has a fixed set of children per node, a ExplanationTree 
    instance contains a set of sets of child ExplanationTrees. A fixed choice of entries starting from the root 
    ExplanationTree down to terminal entries represents a single CREFunc composition.

-ExplanationTreeEntry
    Represents a non-terminal CREFunc ('is_func'==True) in a composition or a terminal cre.Var ('is_func'==False).
    In the non-terminal case holds a reference to a CREFunc instance and child ExplanationTrees for each argument. 
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

    # List of ranges (start,end) over which the func has been applied for each argument.
    'stride' : i8[:,::1],

    # Whether the record is non-terminal---from applying a CREFunc and not from a declared value.
    'is_func' : u1,

    # Whether the record is a constant
    'is_const' : u1,
    # 'const_t_id' : u2,
    # 'const_f_id' : u4,

    # If non-terminal then holds the CREFunc
    'func' : CREFuncType,

    # The number of input arguments for self.func.
    'n_args' : i8,

    # If terminal then holds a cre.Var as a placeholder for the declared value. 
    'var' : VarType,

    # The depth at which the func was applied/inserted, or '0' for declared values
    'depth' : i8,
}

SC_Record_fields = [(k,v) for k,v in SC_Record_field_dict.items()]
SC_Record, SC_RecordType = \
    define_structref("SC_Record", SC_Record_fields, define_constructor=False)
    
@overload(SC_Record, prefer_literal=False)
def overload_SC_Record(func_or_var, depth=0, n_args=0, stride=None, prev_entry_ptr=0, is_const=False):
    '''Implements the constructor for an SC_Record'''

    if(isinstance(func_or_var, CREFuncTypeClass)):
        def impl(func_or_var, depth=0, n_args=0, stride=None, prev_entry_ptr=0, is_const=False):
            st = new(SC_RecordType)
            st.is_func = True
            st.is_const = False
            st.func = func_or_var
            st.depth = depth
            st.n_args = n_args

            ENTRY_WIDTH = 4 + n_args
            n_items = 1
            if(stride is not None): 
                st.stride = stride
                for s_i in stride:
                    s_len = s_i[1]-s_i[0]
                    n_items *= s_len

            data_len = n_items*ENTRY_WIDTH
            st.data = np.empty(data_len,dtype=np.uint32)
            
            return st
    elif(isinstance(func_or_var, VarTypeClass)):
        def impl(func_or_var, depth=0, n_args=0, stride=None, prev_entry_ptr=0, is_const=False):
            st = new(SC_RecordType)
            st.is_func = False
            st.is_const = is_const
            st.var = cast(func_or_var, VarType) 
            st.depth = depth
            st.n_args = 0

            st.data = np.empty((4,),dtype=np.uint32)
            self_ptr = cast(st, i8)
            st.data[0] = u4(self_ptr) # get low bits
            st.data[1] = u4(self_ptr>>32) # get high bits
            st.data[2] = u4(prev_entry_ptr) # get low bits
            st.data[3] = u4(prev_entry_ptr>>32) # get high bits
            
            return st
    else:
        print('fail')
    return impl



record_list_type = ListType(SC_RecordType)
dict_t_id_to_record_list_type = DictType(u2, record_list_type)
t_id_int_tuple = Tuple((u2,i8))

SetChainingPlanner_field_dict = {
    'funcs': ListType(CREFuncType),
    # List of dictionaries that map:
    #  Tuple(type_t_id,depth[int]) -> ListType[Record])
    'declare_records' : DictType(u2, ListType(SC_RecordType)),

    'forward_records' : ListType(DictType(u2, ListType(SC_RecordType))),

    # List of dictionaries that map:
    #  Tuple(type_t_id,depth[int]) -> ListType[Record])
    'backward_records' : ListType(DictType(u2, ListType(SC_RecordType))),

    # Maps type_t_id -> *(Dict: val[any] -> Tuple(depth[i8], *SC_Record_Entry) )
    'val_map_ptr_dict' : DictType(u2, ptr_t),

    # Maps type_t_id -> *(Dict: *Var -> val[any])
    'inv_val_map_ptr_dict' : DictType(u2, ptr_t), 

    # Maps (type_t_id,depth[int]) -> *Iterator[any]
    'flat_vals_ptr_dict' : DictType(Tuple((u2,i8)), ptr_t),

    # A list of tuples with (fact_type, attr, conversions) for the planner's known fact_types
    'semantic_visible_attrs' : types.Any,

    # A mapping of strings like "('fact_name', 'attr', 'type')" to CREFunc instances used for conversions
    'conversion_funcs' :  DictType(unicode_type, CREFuncType)
}

@structref.register
class SetChainingPlannerTypeClass(types.StructRef):
    def __str__(self):
        return f"cre.SetChainingPlannerType"


GenericSetChainingPlannerType = SetChainingPlannerTypeClass([(k,v) for k,v in SetChainingPlanner_field_dict.items()])

def get_semantic_visible_attrs(fact_types):
    ''' Takes in a set of fact types and returns all (fact, attribute, ((conv_type, conv_func)...) tuples
        for attributes that are both "semnatic" and "visible". 
    '''
    if(not isinstance(fact_types,(list,tuple))): fact_types = [fact_types]
    context = cre_context()

    sem_vis_fact_attrs = {}
    for ft in fact_types:
        ft = ft.instance_type if (isinstance(ft, (types.TypeRef,))) else ft
        parents = context.parents_of.get(ft._fact_name,[])
        for attr, attr_spec in ft.filter_spec(["visible","semantic"]).items():
            is_new = True
            for p in parents:
                if((p,attr) in sem_vis_fact_attrs):
                    is_new = False
                    break

            # TODO: Can return this functionality when CREFuncs have specialized numba_types
            conversions = None            
            if('conversions' in attr_spec):
                # conversions = tuple([(k,v) for k,v in attr_spec['conversions'].items()])
                conversions = tuple([k for k,v in attr_spec['conversions'].items()])
            
            if(is_new): sem_vis_fact_attrs[(ft,attr)] =  conversions
            # if(is_new): sem_vis_fact_attrs[(ft,attr,)] = True

    return tuple([(fact_type, types.literal(attr), conversions) for fact_type, attr in sem_vis_fact_attrs.keys()])

from numba.core.typing.typeof import typeof
def get_sc_planner_type(fact_types):
    sv_attrs = get_semantic_visible_attrs(fact_types)
    if(len(fact_types) > 0 and len(sv_attrs) == 0):
        warnings.warn("SC_planner Warning: provided fact_types have no 'semantic' and 'visible' attributes.")
    field_dict = {**SetChainingPlanner_field_dict, 'semantic_visible_attrs': typeof(sv_attrs)}
    planner_type = SetChainingPlannerTypeClass([(k,v) for k,v in field_dict.items()])    
    planner_type._field_dict = field_dict
    return planner_type

def register_conversions(planner, typ):
    if(not hasattr(typ, '_field_dict')): return
    svas = typ._field_dict['semantic_visible_attrs']
    for fact_type, attr,_ in svas:
        attr_spec = fact_type.instance_type.clean_spec[attr.literal_value]

        if('conversions' in attr_spec):
            for conv_type, conv_func in attr_spec['conversions'].items(): 
                tup = (str(fact_type.instance_type), attr.literal_value, repr(conv_type))
                add_conversion_func(planner,str(tup),conv_func)
                # planner.conversion_funcs[str(tup)] = conv_func

class SetChainingPlanner(structref.StructRefProxy):
    def __new__(cls, fact_types=None):
        typ = GenericSetChainingPlannerType
        if(fact_types is not None):
            typ = get_sc_planner_type(fact_types)
        self = sc_planner_ctor(typ)
        register_conversions(self, typ)
        self.curr_infer_depth = 0
        return self

    def declare(self, val, var=None, is_const=False):
        return planner_declare(self, val, var, is_const)

    def search_for_explanations(self, goal, **kwargs):
        return search_for_explanations(self, goal, **kwargs)

    @property
    def conversion_funcs(self):
        return get_conversion_funcs(self)

    @property
    def num_forward_inferences(self):
        return count_forward_inferences(self)

define_boxing(SetChainingPlannerTypeClass, SetChainingPlanner)

@njit(cache=True)
def get_conversion_funcs(self):
    return self.conversion_funcs

@njit(cache=True)
def add_conversion_func(self, conv_str, func):
    self.conversion_funcs[conv_str] = func

@njit(cache=True)
def sc_planner_ctor(sc_planner_type):
    st = new(sc_planner_type)
    st.declare_records = Dict.empty(u2, record_list_type)
    st.forward_records = List.empty_list(dict_t_id_to_record_list_type)
    st.backward_records = List.empty_list(dict_t_id_to_record_list_type)
    st.val_map_ptr_dict = Dict.empty(u2, ptr_t)
    st.inv_val_map_ptr_dict = Dict.empty(u2, ptr_t)
    st.flat_vals_ptr_dict = Dict.empty(t_id_int_tuple, ptr_t)
    st.conversion_funcs = Dict.empty(unicode_type, CREFuncType)
    return st

#------------------------------------------------------------------
# : Planner.declare()

def ensure_ptr_dicts(planner, typ):
    pass

@overload(ensure_ptr_dicts)
def ensure_ptr_dicts_overload(planner, typ):
    _typ = typ.instance_type
    lt = ListType(_typ)
    vt = DictType(_typ, i8_3x_tuple)
    ivt = DictType(i8, _typ)
    typ_name = str(_typ)
    typ_t_id = u2(cre_context().get_t_id(_typ))
    def impl(planner, typ):
        # print(typ_name, typ_t_id)
        tup = (u2(typ_t_id),0)
        if(tup not in planner.flat_vals_ptr_dict):
            flat_vals = List.empty_list(typ)
            planner.flat_vals_ptr_dict[tup] = _ptr_from_struct_incref(flat_vals)
        if(u2(typ_t_id) not in planner.val_map_ptr_dict):
            val_map = Dict.empty(typ, i8_3x_tuple)    
            planner.val_map_ptr_dict[u2(typ_t_id)] = _ptr_from_struct_incref(val_map)
        if(u2(typ_t_id) not in planner.inv_val_map_ptr_dict):
            inv_val_map = Dict.empty(i8, typ)    
            planner.inv_val_map_ptr_dict[u2(typ_t_id)] = _ptr_from_struct_incref(inv_val_map)

        if(u2(typ_t_id) not in planner.declare_records):
            planner.declare_records[u2(typ_t_id)] = List.empty_list(SC_RecordType)        


        flat_vals = _list_from_ptr(lt, planner.flat_vals_ptr_dict[tup])    
        val_map = _dict_from_ptr(vt, planner.val_map_ptr_dict[u2(typ_t_id)])    
        inv_val_map = _dict_from_ptr(ivt, planner.inv_val_map_ptr_dict[u2(typ_t_id)])    
        declare_records = planner.declare_records[u2(typ_t_id)]
        return flat_vals, val_map, inv_val_map, declare_records
    return impl

def planner_declare_val(planner, val, func_or_var):
    pass

@overload(planner_declare_val)
def planner_declare_val_overload(planner, val, func_or_var, is_const=False):
    '''Declares a primative value or fact (but not its visible attributes)
        into the 0th depth of a planner instance.'''
    val_typ = val
    def impl(planner, val, func_or_var, is_const=False):
        # Ensure that various data structures exist for this val_type
        flat_vals, val_map, inv_val_map, declare_records = \
            ensure_ptr_dicts(planner, val_typ)

        _, _, prev_entry_ptr = val_map.get(val,(0,0,0))
        is_prev = prev_entry_ptr != 0

        # Make a new Record for this declaration
        rec = SC_Record(func_or_var,
             prev_entry_ptr=prev_entry_ptr, 
             is_const=is_const)
        var_ptr = cast(func_or_var, i8)
        declare_records.append(rec)
        
        # If the is new add it to 'flat_vals'
        if(not is_prev): flat_vals.append(val)

        # Add a pointer to the rec's rec_entry into val_map
        rec_entry_ptr = _get_array_raw_data_ptr(rec.data)
        val_map[val] = (0, rec_entry_ptr, rec_entry_ptr)

        # And associate the var_ptr with val in inv_val_map
        inv_val_map[var_ptr] = val

        # TODO: Find something faster than this
        # k = 0
        for i, _val in enumerate(flat_vals):
            if(_val == val):
                return i
        return -1
    return impl 

def _planner_declare_conversion(planner, val, func_or_var, source_ind=None):
    pass

@overload(_planner_declare_conversion)
def planner_declare_conversion_overload(planner, val, func_or_var, source_ind=None):
    '''Declares a primative value or fact (but not its visible attributes)
        into the 0th depth of a planner instance.'''
    val_typ = val
    def impl(planner, val, func_or_var, source_ind=None):
        # Ensure that various data structures exist for this val_type
        flat_vals, val_map, inv_val_map, declare_records = \
            ensure_ptr_dicts(planner, val_typ)

        _, prev_entry_ptr,_ = val_map.get(val,(0,0,0))
        is_prev = prev_entry_ptr != 0            

        if(source_ind is not None):
            # If has 'source_ind' then conversion is part of declaration, 
            #  in which case only one record is needed per unique input val.
            if(is_prev):
                return len(flat_vals)-1
            stride = np.empty((1,2),dtype=np.int64)
            stride[0][0] = source_ind
            stride[0][1] = source_ind+1
        else:
            stride = None

        # Make a new Record for this declaration
        rec = SC_Record(func_or_var, n_args=1, stride=stride, prev_entry_ptr=prev_entry_ptr)
        rec_ptr = cast(rec, i8)
        data = rec.data
        data[0] = u4(rec_ptr) # get low bits
        data[1] = u4(rec_ptr>>32) # get high bits
        data[2] = u4(prev_entry_ptr) # get low bits
        data[3] = u4(prev_entry_ptr>>32)# get high bits


        if(source_ind is not None):
            data[4] = source_ind

        f_ptr = cast(func_or_var, i8)
        declare_records.append(rec)
        
        # If the is new add it to 'flat_vals'
        if(not is_prev): flat_vals.append(val)

        # Add a pointer to the rec's rec_entry into val_map
        rec_entry_ptr = _get_array_raw_data_ptr(rec.data)

        val_map[val] = (0, rec_entry_ptr, rec_entry_ptr)

        # And associate the var_ptr with val in inv_val_map
        inv_val_map[f_ptr] = val
        # print("Conv_ptr", var_ptr, val, rec_entry_ptr)
        return len(flat_vals)-1
    return impl 

@njit(cache=True)
def planner_declare_conversion(planner, val, func_or_var, source_ind=None):
    return _planner_declare_conversion(planner, val, func_or_var, source_ind)



def planner_declare_conversions(planner, val, fact_type, attr, source_ind):
    pass

@overload(planner_declare_conversions)
def planner_declare_conversions_overload(planner, val, fact_type, attr, source_ind):
    fact_type = fact_type.instance_type
    attr_spec = fact_type.clean_spec[attr.literal_value]
    if('conversions' in attr_spec):
        conversions = attr_spec['conversions']
        conversion_strs = []
        for conv_type in conversions:
            conversion_strs.append(str((str(fact_type), attr.literal_value, repr(conv_type))))
        conversion_strs = tuple(conversion_strs)

        cf_type = CREFuncTypeClass(conv_type, (val,))

        get_ret = get_return_val_impl(conv_type)
        set_base = set_base_arg_val_impl(val)
        def impl(planner, val, fact_type, attr, source_ind):
            for conv_str in literal_unroll(conversion_strs):
                if(conv_str in planner.conversion_funcs):
                    _conv_func = planner.conversion_funcs[conv_str]    
                else:
                    raise ValueError("Planner not instantiated with fact_type, that defines a conversion.")
                
                conv_func = cast(_conv_func, cf_type)
                set_base(conv_func, 0, val)
                status = cre_func_call_self(conv_func)
                if(status > CFSTATUS_TRUTHY):
                    return 
                conv_val = get_ret(conv_func)
                planner_declare_conversion(planner, conv_val, conv_func, source_ind)                    
                    
    else:
        def impl(planner, val, fact_type, attr, source_ind):
            pass
    return impl


# Note/TODO: If using GenericSetChainingPlannerType this won't recompile on changes to 'conversions' 
# @generated_jit(cache=True,nopython=True)
def _planner_declare(planner, val, var=None):
    pass

@overload(_planner_declare)
@overload_method(SetChainingPlannerTypeClass, "declare")
def planner_declare_overload(planner, val, var=None, is_const=False):
    '''Declares a primative value or fact (and its visible attributes)
        into the 0th depth of a planner instance.'''

    val_typ = val

    if(isinstance(val_typ, Fact)):
        semantic_visible_attrs = get_semantic_visible_attrs(val_typ)
        if(len(semantic_visible_attrs) > 0):
            # Case 1: Declare the fact and its visible-semantic attributes
            def impl(planner, val, var=None, is_const=False):
                _var = Var(val_typ) if var is None else var

                # Declare the fact
                planner_declare_val(planner, val, _var)
                for tup in literal_unroll(semantic_visible_attrs):
                    fact_type, attr,_ = tup
                    attr_val = fact_lower_getattr(val, attr)
                    attr_var = var_append_deref(_var, attr)

                    # Declare the attribute value
                    ind = planner_declare_val(planner, attr_val, attr_var)

                    # Declare any conversion on the value
                    planner_declare_conversions(planner, attr_val, fact_type, attr, ind)
        else:
            # Case 2: Should declare just the fact
            def impl(planner, val, var=None, is_const=False):
                _var = Var(val_typ) if var is None else var
                planner_declare_val(planner, val, _var)
    else:
        # Case 3: Declare a plain primative (e.g. a float, int, str ect.)
        def impl(planner, val, var=None, is_const=False):
            # Note: Make a var even for constants so we can 
            #  recover it from the the inv_val_map
            _var = Var(val_typ) if var is None else var
            planner_declare_val(planner, val, _var, is_const)
        
    return impl

@njit(cache=True)
def planner_declare(planner, val, var=None, is_const=False):
    return _planner_declare(planner, val, var, is_const)

#------------------------------------------------------------------
# : Explanation Search Main Loop
# @generated_jit(cache=True)
def _recover_arg_ind(planner, arg):
    pass

@overload(_recover_arg_ind)
def recover_arg_ind_overload(planner, arg):
    # arg_type_name = str(arg)
    arg_t_id = cre_context().get_t_id(arg)
    arg_type = arg
    def impl(planner, arg):
        flat_vals, val_map, inv_val_map, declare_records = \
            ensure_ptr_dicts(planner, arg_type)
        # print(flat_vals)
        return arg_t_id, flat_vals.index(arg)
    return impl

@njit(cache=True)
def recover_arg_ind(planner, arg):
    return _recover_arg_ind(planner, arg)


def should_commute_skip(arg_inds, func):
    for k, ind in enumerate(arg_inds):
        comm_args = func.right_commutes.get(k,[])
        for j in comm_args:
            if(arg_inds[k] < arg_inds[j]):
                return True
    return False


#TODO njit it
def commute_sensitive_arg_ind_product(func, arg_inds_by_t_id):
    arg_ind_sets = [arg_inds_by_t_id[t_id] for t_id in func.arg_t_ids]
    lengths = np.array([len(x) for x in arg_ind_sets],dtype=np.int64)
    inds = np.zeros(len(arg_ind_sets),dtype=np.int64)

    done = False
    out = []
    max_k = k = len(lengths)-1
    while(not done):
        arg_inds = np.array([arg_ind_sets[_k][inds[_k]] for _k in range(len(lengths))])

        # No redundant Indicies
        if(np.sum(inds.reshape(1,-1) == inds.reshape(-1,1)) <= len(inds) and
           not should_commute_skip(arg_inds, func)):
            out.append(arg_inds)

        inds[k] += 1
        while(inds[k] >= lengths[k]):
            inds[k] = 0
            k -= 1
            if(k < 0): done = True; break;
            inds[k] += 1
        k = max_k
    return out



from numba.core.runtime.nrt import rtsys
from cre.core import standardize_type
def search_for_explanations(self, goal, funcs=None, policy=None,
             search_depth=1, min_stop_depth=None, min_solution_depth=None, 
             fewer_solutions=True, ignore_inner_zeros=False,
             context=None, **kwargs):
    '''For SetChainingPlanner 'self' produce an explanation tree
        holding all CREFunc compositions of 'funcs' up to depth 'search_depth'
        that produce 'goal' from the planner's declared starting values.
    '''
    # If a policy is given set the min_stop_depth to be its length
    if(min_stop_depth is None):
        if(policy is None):
            min_stop_depth = -1
        else:
            min_stop_depth = len(policy)

    if(min_solution_depth is None):
        if(policy is None):
            min_solution_depth = -1
        else:
            min_solution_depth = len(policy)

    # Declare any constant funcs
    if(funcs is not None):
        for func in funcs:
            if(func.n_args == 0):
                planner_declare_conversion(self, func(), func, None)


    context = cre_context(context)
    g_typ = standardize_type(type(goal), context)

    found_at_depth = query_goal(self, g_typ, goal, min_solution_depth)
    depth = 0

    while((found_at_depth is None or 
          self.curr_infer_depth < min_stop_depth)
          and (depth < search_depth)):

        depth += 1
        if(depth <= self.curr_infer_depth): continue
                
        # Apply the input funcs in the forward direction once.
        if(policy is None):
            if(funcs is None): raise ValueError("Must provide funcs or policy.")
            # with PrintElapse("forward"):
            # print([o._type for o in funcs])
            forward_chain_one(self, funcs)

        # Policy Case
        else:
            # Convert args in policy to arg_inds
            depth_policy = []
            # print(">>>", policy)
            for t in policy[depth-1]:
                func, arg_set = (t[0], t[1]) if(isinstance(t, tuple)) else (t, None)
                # print(depth, "::", op, arg_set)

                # If the policy didn't provide any args then try all permutations
                if(arg_set is None or len(arg_set) == 0 or
                    # TODO: For now also try all combinations if the number of args is too small
                    #  but in the future can do all vs this.
                    len(arg_set) < func.n_args
                    ):
                    depth_policy.append(func)
                    continue


                arg_inds_by_t_id = {}
                for arg in arg_set:
                    try:
                        type_t_id, arg_ind = recover_arg_ind(self, arg)
                    except ValueError as e:
                        continue

                    arr = arg_inds_by_t_id.get(u2(type_t_id), [])
                    arr.append(arg_ind)
                    arg_inds_by_t_id[u2(type_t_id)] = arr
                
                # try:
                    # Make Cartesian Product of arg_inds
                arg_inds = commute_sensitive_arg_ind_product(func, arg_inds_by_t_id)
                # arg_inds = list(itertools.product(*[arg_inds_by_t_id[str(typ)] for typ in func.signature.args]))
                arg_inds = np.array(arg_inds,dtype=np.int64)
                # print("arg_inds", arg_inds)
                depth_policy.append((func, arg_inds))

                # If policy args fail just skip
                # except KeyError:
                #     continue
                    # print("DEPTH ERRR")
                    # depth_policy.append(func)
            # print(">>>", depth_policy)

                

            # Apply policy
            forward_chain_one(self, depth_policy)
        
        # with PrintElapse("query_goal"):
        if(depth >= min_stop_depth):
            found_at_depth = query_goal(self, g_typ, goal, min_solution_depth)
        
        # if(depth >= search_depth): break
    # print(found_at_depth)
    if(found_at_depth is None):
        return None
    else:
        # with PrintElapse("build_explanation_tree"):
        return build_explanation_tree(self, g_typ, goal, fewer_solutions, ignore_inner_zeros)



@generated_jit(cache=True, nopython=True)
def query_goal(self, typ, goal, min_solution_depth):
    _typ = typ.instance_type
    dict_typ = DictType(_typ,i8_3x_tuple)
    # typ_name = str(_typ)
    t_id = cre_context().get_t_id(_typ)
    def impl(self, typ, goal, min_solution_depth):
        # vtd_ptr_d = self.val_map
        if(u2(t_id) in self.val_map_ptr_dict):
            val_map = _dict_from_ptr(dict_typ, self.val_map_ptr_dict[u2(t_id)])
            if(goal in val_map):

                _, low_entry_ptr, entry_ptr = val_map[goal]
                
                # Can tell from the last rec entry whether the solution
                #  exists 
                rec, entry_ptr, _ = extract_rec_entry(entry_ptr)
                depth = rec.depth
                # while(depth < min_solution_depth):
                #     if(entry_ptr == 0):
                #         return None
                #     rec, entry_ptr, _ = extract_rec_entry(entry_ptr)
                #     depth = rec.depth

                if(depth < min_solution_depth):
                    return None

                return depth
            else:
                return None
        else:
            return None
    return impl

#------------------------------------------------------------------
# : Forward Inference + Record Insertion

#------------------------------
# : Record Insertion

@njit(cache=True)
def insert_record(self, rec, ret_t_id, depth):
    '''Inserts a Record 'rec' into the list of records in  
        'self' for type 'ret_type_name' and depth 'depth' '''
    # Make sure records dictionaries up to this depth
    while(depth >= len(self.forward_records)):
        self.forward_records.append(
            Dict.empty(u2, record_list_type)
        )

    recs_at_depth = self.forward_records[depth]
    recs = recs_at_depth.get(u2(ret_t_id),
            List.empty_list(SC_RecordType))
    recs.append(rec)
    recs_at_depth[u2(ret_t_id)] = recs


def _join_records_of_type(self, depth, typ):
    pass

i8_3x_tuple = Tuple((i8,i8,i8))
@overload(_join_records_of_type)
def join_records_of_type_overload(self, depth, typ):
    ''' Goes through every record at 'depth' and joins them
            so that 'val_to_depth' and 'flat_vals' are defined '''
    _typ = typ.instance_type
    dict_typ = DictType(_typ,i8_3x_tuple)
    # typ_name = str(_typ)
    t_id = cre_context().get_t_id(_typ)
    def impl(self, depth, typ):
        val_map = _dict_from_ptr(dict_typ, self.val_map_ptr_dict[u2(t_id)])
        flat_vals = List.empty_list(typ, len(val_map))
        for val in val_map:
            flat_vals.append(val)

        tup = (u2(t_id), depth)
        self.flat_vals_ptr_dict[tup] = _ptr_from_struct_incref(flat_vals)
        # print("::", depth , "::")
        # print(flat_vals)
    return impl

@njit(cache=True)
def join_records_of_type(self, depth, typ):
    return _join_records_of_type(self, depth, typ)


def join_records(self, depth, funcs):
    typs = set([func.return_type for func in funcs])
    for typ in typs:
        val_to_depth = join_records_of_type(self, depth, typ)


#------------------------------
# : Forward Chaining

def forward_chain_one(self, depth_policy=None):
    '''Applies 'funcs' on all current inferred values'''
    nxt_depth = self.curr_infer_depth+1
    if(depth_policy is None): depth_policy = self.func

    funcs = []
    for func in depth_policy:

        # If policy provides arg_inds then apply each set 
        #  of arg_inds one at a time.
        if(isinstance(func, tuple)):
            func, arg_inds = func
            for inds in arg_inds:
                sig = func.signature
                # v = call_op_for_inds(self, func, sig.return_type, sig.args, self.curr_infer_depth, inds)
                # print(v)
                rec = apply_one(func, self, sig.return_type, sig.args,
                    inds, self.curr_infer_depth)

                # Insert records
                if(rec is not None):
                    insert_record(self, rec, func.return_t_id, nxt_depth)

        # If no arg_inds are provided then apply all permutations 
        #  of arguments to the func.
        elif(func.n_args > 0):
            rec = apply_multi(func, self,
                self.curr_infer_depth)

            # Insert records
            if(rec is not None):
                insert_record(self, rec, func.return_t_id, nxt_depth)
        else:
            # Constant CREFunc case ignore at forward step
            continue

        
        funcs.append(func)


    join_records(self, nxt_depth, funcs)
    self.curr_infer_depth = nxt_depth

#---------------------------------
# : apply_one()

# stub_function
def _call_op_for_inds(planner, func, return_type, arg_types, depth, inds):
    pass

@overload(_call_op_for_inds)
def call_op_for_inds_overload(planner, func, return_type, arg_types, depth, inds):
    return_type = return_type.instance_type
    arg_types = tuple([a.instance_type for a in arg_types])
    
    lst_types = tuple([ListType(arg_type) for arg_type in arg_types])
    call_type = types.FunctionType(return_type(*arg_types))
    check_type = types.FunctionType(types.boolean(*arg_types))

    context = cre_context()
    type_t_ids = tuple([u2(context.get_t_id(_type=t)) for t in arg_types])
    # arg_types = types.TypeRef(Tuple(arg_types))

    def impl(planner, func, return_type, arg_types, depth, inds):
        arg_ptrs = np.empty(len(arg_types),dtype=np.int64)
        i = 0
        for lst_typ in literal_unroll(lst_types):
            iter_base = _list_base_from_ptr(planner.flat_vals_ptr_dict[(u2(type_t_ids[i]),depth)])
            size = _listtype_sizeof_item(lst_typ)
            arg_ptrs[i] = iter_base + inds[i] * size
            i += 1

        args = _tuple_from_data_ptrs(arg_types, arg_ptrs)

        val = func(*args)
        return val

    return impl

# stub function
def _apply_one(func, planner, return_type, arg_types, inds, curr_infer_depth):
    pass

@overload(_apply_one)
def apply_one_overload(func, planner, return_type, arg_types, inds, curr_infer_depth):
    return_type = return_type.instance_type
    ret_d_typ = DictType(return_type,Tuple((i8,i8,i8)))  
    context = cre_context()
    ret_type_t_id = u2(context.get_t_id(_type=return_type))
    def impl(func, planner, return_type, arg_types, inds, curr_infer_depth):
        v = _call_op_for_inds(planner, func, return_type, arg_types, curr_infer_depth, inds)
        if(v is None):
            return

        stride = np.empty((len(inds),2),dtype=np.int64)
        nxt_depth = curr_infer_depth+1
        stride[:, 0] = inds
        stride[:, 1] = inds+1
        n_args = len(func.head_ranges)
        rec = SC_Record(func, nxt_depth, n_args, stride)
        data = rec.data

        val_map =  _dict_from_ptr(ret_d_typ,
            planner.val_map_ptr_dict[u2(ret_type_t_id)])

        d_ptr = _get_array_raw_data_ptr(data)
        rec_ptr = cast(rec, i8)

        # NOTES: Should be entries_start, entries_end 
        low_depth, prev_low_entry, prev_entry = val_map.get(v, (-1,0,0))
        # if(nxt_depth > min_stop_depth and
        #    prev_depth != -1 and
        #    prev_depth < nxt_depth):
        #     # print("SKIP", op, inds, min_stop_depth)
        #     return

        data[0] = u4(rec_ptr) # get low bits
        data[1] = u4(rec_ptr>>32) # get high bits
        data[2] = u4(prev_entry) # get low bits
        data[3] = u4(prev_entry>>32)# get high bits

        #Put arg inds at the end
        for i,ind in enumerate(inds):
            data[4+i] = u4(ind)


        entry_ptr = d_ptr
        if(low_depth == -1 or low_depth == nxt_depth):
            low_depth = nxt_depth
            prev_low_entry = entry_ptr

        val_map[v] = (low_depth, prev_low_entry, entry_ptr) 
        return rec


    return impl

@njit(cache=True)
def apply_one(func, planner, return_type, arg_types, inds, curr_infer_depth):
    return _apply_one(func, planner, return_type, arg_types, inds, curr_infer_depth)


#--------------------------------
# : Counting search size
@njit(cache=True)
def count_forward_inferences(self):
    n = 0
    for recs_at_depth in self.forward_records:
        for recs in recs_at_depth.values():
            for rec in recs:
                if(rec.is_func):
                    n += np.prod(rec.stride[:,1] - rec.stride[:,0])
    return n


#---------------------------------
# : Source Generation -- apply_multi()

def _gen_retrieve_itr(tn,typ,ind='    '):
    t_id = cre_context().get_t_id(typ)
    return indent(f'''tup{tn} = (u2({t_id!r}),depth)
if(tup{tn} in planner.flat_vals_ptr_dict):
    iter_ptr{tn} = planner.flat_vals_ptr_dict[tup{tn}]
    iter{tn} = _list_from_ptr(l_typ{tn}, iter_ptr{tn})
else:
    return None
''',prefix=ind)


def gen_apply_multi_source(func, generic, ind='    '):
    '''Generates source code for an apply_multi() implementation for a CREFunc'''
    has_check = hasattr(func,'check')
    sig = func.signature
    args = sig.args
    typs = {}
    for typ in sig.args:
        if(typ not in typs):
            typs[typ] = len(typs)

    src = \
f'''from numba import njit, i8, u8, u4, u2
from numba.typed import Dict
from numba.types import ListType, DictType, Tuple
import numpy as np
import cloudpickle
from cre.utils import cast, _dict_from_ptr, _list_from_ptr, _get_array_raw_data_ptr
from cre.func import CFSTATUS_TRUTHY, get_return_val_impl, set_base_arg_val_impl, cre_func_resolve_call_self
from cre.sc_planner import SC_Record

''' 
    if(not generic):
        imp_targets = ['call_heads'] + (['check'] if has_check else [])
        src += f'''{gen_import_str(func.func_name,
                     func.long_hash, imp_targets)}\n\n'''

    src += "".join([f'typ{i}'+", " for i in range(len(typs))]) + \
             f'= cloudpickle.loads({cloudpickle.dumps(tuple(typs.keys()))})\n'


    src += f'''ret_typ = cloudpickle.loads({cloudpickle.dumps(sig.return_type)})
ret_d_typ = DictType(ret_typ,Tuple((i8,i8,i8)))
'''

    src += ", ".join([f'l_typ{i}' for i in range(len(typs))]) + \
            " = " + ", ".join([f'ListType(typ{i})' for i in range(len(typs))]) + '\n'

    a_cnt = list(range(len(args)))

    if(generic):
        for i in range(len(typs)):
            src += f'set_base{i} = set_base_arg_val_impl(typ{i})\n'
        src += f'get_ret = get_return_val_impl(ret_typ)\n'
    # start_kwargs = ", ".join([f'start{i}=0' for i in a_cnt])
    src += f'''N_ARGS = {len(args)}

ENTRY_WIDTH = 4+N_ARGS
@njit(cache=True)
def apply_multi(func, planner, depth):
'''

    for tn, typ in enumerate(typs.keys()):
        src += _gen_retrieve_itr(tn, typ)

    it_inds = [typs[args[i]] for i in range(len(args))]


    ls = ", ".join([f"l{i}" for i in a_cnt])
    # l_defs = '\n'.join([f'l{i} = stride[{i}][1]-stride[{i}][0]' for i in a_cnt])#", ".join([f"len(iter{it_inds[i]})-start{i}" for i in a_cnt])
    stride_defaults = ",".join([f'[0,len(iter{it_inds[i]})]' for i in a_cnt])
    src += indent(f'''
nxt_depth = depth + 1
stride = np.array([{stride_defaults}],dtype=np.int64)
val_map =  _dict_from_ptr(ret_d_typ, planner.val_map_ptr_dict[u2({func.return_t_id!r})])

rec = SC_Record(func, nxt_depth, N_ARGS, stride)
data = rec.data
d_ptr = _get_array_raw_data_ptr(data)
rec_ptr = cast(rec, i8)

d_offset=0

''',prefix=ind)
    c_ind = copy(ind)
    for i, arg in enumerate(args):
        src += f'{c_ind}for i{i} in range(stride[{i}][0],stride[{i}][1]):\n'
        c_ind += ind
        if(i in func.right_commutes):
            ignore_conds = ' or '.join([f"i{i} > i{j}" for j in func.right_commutes[i]])
            src += f'{c_ind}if({ignore_conds}): continue\n'
        

    _is = ",".join([f"i{i}" for i in a_cnt])
    _as = ",".join([f"a{i}" for i in a_cnt])
    params = ",".join([f"iter{it_inds[i]}[i{i}]" for i in a_cnt])
    arg_assigns = "\n".join([f"data[d_offset+{4+i}] = i{i}" for i in a_cnt])

    src += indent(f'{_as} = {params}\n',prefix=c_ind)
# {f'if(not check({_as})): continue' if has_check else ""}

    if(generic):
        for i in a_cnt:
            src += indent(f'set_base{it_inds[i]}(func,{i},a{i})\n',prefix=c_ind)
        src += indent(f'''
status = cre_func_resolve_call_self(func)
if(status > CFSTATUS_TRUTHY):
    continue
v = get_ret(func)\n''', prefix=c_ind)
    else:
        src += indent(f'v = call_heads({_as})\n',prefix=c_ind)

    src += indent(f'''
low_depth, prev_low_entry, prev_entry = val_map.get(v, (-1,0,0))

data[d_offset +0] = u4(rec_ptr) # get low bits
data[d_offset +1] = u4(rec_ptr>>32) # get high bits
data[d_offset +2] = u4(prev_entry) # get low bits
data[d_offset +3] = u4(prev_entry>>32)# get high bits

#Put arg inds at the end
{arg_assigns}

entry_ptr = d_ptr + d_offset*4
if(low_depth == -1 or low_depth == nxt_depth):
    low_depth = nxt_depth
    prev_low_entry = entry_ptr

val_map[v] = (low_depth, prev_low_entry, entry_ptr)
d_offset += ENTRY_WIDTH
        
''',prefix=c_ind)

    src += f'''{ind}return rec'''
    return src

#---------------------------------
# : apply_multi()

def __assert_prepared(self, typ, depth):
    pass

u4_slice = u4[:]
@overload(__assert_prepared)
def _assert_prepared_overload(self, typ, depth):
    t_id = cre_context().get_t_id(typ.instance_type)
    def impl(self, typ, depth):
        while(depth >= len(self.forward_records)):
            self.forward_records.append(
                Dict.empty(u2, record_list_type)
            )
        if(u2(t_id) not in self.val_map_ptr_dict):
            val_map = Dict.empty(typ, i8_3x_tuple)
            val_map_ptr = _ptr_from_struct_incref(val_map)
            self.val_map_ptr_dict[u2(t_id)] = val_map_ptr
    return impl

@njit(cache=True)
def _assert_prepared(self, typ, depth):
    return __assert_prepared(self, typ, depth)
    

def apply_multi(func, planner, depth, min_stop_depth=-1):
    '''Applies 'func' at 'depth' and returns the SC_Record'''

    # If it doesn't already exist generate and inject '_apply_multi' into 'func'
    # print("apply_multi", type(func))
    if(not hasattr(func,'_apply_multi')):
        # Run a generic implementation if we cannot safely inline call_heads
        #  i.e. if can raise an error,  is composed, or otherwise untyped
        generic = (not getattr(func._type,'no_raise',False) or func.long_hash is None or 
                   func.is_composed or func.func_name == "GenericCREFunc")

        if(generic):
            hash_code = unique_hash_v([func.return_type, func.arg_types, func.right_commutes])
        else:
            hash_code = unique_hash_v([func.long_hash])
        # print(get_cache_path('apply_multi',hash_code))
        if(not source_in_cache('apply_multi',hash_code)):
            src = gen_apply_multi_source(func, generic)
            source_to_cache('apply_multi',hash_code,src)
        l = import_from_cached('apply_multi',hash_code,['apply_multi'])
        setattr(func,'_apply_multi', l['apply_multi'])
        # print("<<<", type(am))

    typ = func.return_type
    # typ_name = str(typ)
    _assert_prepared(planner, typ, depth)

    am = getattr(func,'_apply_multi')
    return am(func, planner, depth)

#------------------------------------------------------------------
# : Retracing + Explanation Tree Construction 

#-----------------------------
# : Explanation Tree Entry

ExplanationTreeEntry_field_dict = {
    # Whether the entry is non-terminal and represents the application of a CREFunc. 
    "is_func" : u1,

    # Whether the entry is a constant,
    "is_const" : u1,

    # If non-terminal the CREFunc applied
    "func" : CREFuncType,

    # If terminal the cre.Var instance
    "var" : VarType,

    # A List of refcounted pointers to child explanation trees for each argument
    "child_arg_ptrs" : ListType(ptr_t)#i8[::1]
}
ExplanationTreeEntry_fields = [(k,v) for k,v in ExplanationTreeEntry_field_dict.items()]
ExplanationTreeEntry, ExplanationTreeEntryType = \
    define_structref("ExplanationTreeEntry", ExplanationTreeEntry_fields, define_constructor=False)

# def expl_tree_entry_ctor(func_or_var, child_arg_ptrs=None):
#     pass

# @overload(expl_tree_entry_ctor)
# def expl_tree_entry_ctor_overload(func_or_var, child_arg_ptrs=None):
#     if(isinstance(func_or_var, CREFuncTypeClass)):
#         def impl(func_or_var, child_arg_ptrs=None):
#             st = new(ExplanationTreeEntryType)
#             st.is_func = True
#             st.func = func_or_var
#             st.child_arg_ptrs = child_arg_ptrs
#             return st
#     elif(isinstance(func_or_var, VarTypeClass)):
#         def impl(func_or_var, child_arg_ptrs=None):
#             st = new(ExplanationTreeEntryType)
#             st.is_func = False
#             st.var = cast(func_or_var, VarType) 
#             return st
#     print("T::", func_or_var)
    # return impl #ExplanationTreeType(func_or_var, types.Optional(ListType(ptr_t)))

@njit(cache=True)
def func_expl_tree_entry_ctor(func, child_arg_ptrs):
    st = new(ExplanationTreeEntryType)
    st.is_func = True
    st.is_const = False
    st.func = func
    st.child_arg_ptrs = child_arg_ptrs
    return st

@njit(cache=True)
def var_expl_tree_entry_ctor(var, is_const=False):
    st = new(ExplanationTreeEntryType)
    st.is_func = False
    st.is_const = is_const
    st.var = var
    return st



#-----------------------------
# : Explanation Tree

ExplanationTree_field_dict = {
    'entries' : ListType(ExplanationTreeEntryType),
    'inv_val_map_ptr_dict' : DictType(u2, ptr_t),
}

@structref.register
class ExplanationTreeTypeClass(types.StructRef):
    def __str__(self):
        return "ExplanationTreeType"
    

ExplanationTreeType = ExplanationTreeTypeClass([(k,v) for k,v in ExplanationTree_field_dict.items()])

class ExplanationTree(structref.StructRefProxy):
    def __new__(cls):
        self = expl_tree_ctor()
        return self

    def __iter__(self):
        return ExplanationTreeIter(self)

read_inv_val_map_impls = {} 
# @generated_jit(cache=True)
def get_read_inv_val_map_impl(t_id):
    impl = read_inv_val_map_impls.get(t_id, None)
    if(impl is None):
        typ = cre_context().get_type(t_id=t_id)
        ivm_typ = DictType(i8, typ)
        @njit(typ(ExplanationTreeType,i8),cache=True)
        def read_inv_val_map(expl_tree, var_ptr):
            inv_val_map_ptr_dict = expl_tree.inv_val_map_ptr_dict
            inv_val_map = _dict_from_ptr(ivm_typ, inv_val_map_ptr_dict[u2(t_id)])
            out = inv_val_map[var_ptr]
            return out#inv_val_map[var_ptr]

        impl_ep = read_inv_val_map.overloads[(ExplanationTreeType,i8)].entry_point
        impl = read_inv_val_map_impls[t_id] = impl_ep
    return impl

define_boxing(ExplanationTreeTypeClass,ExplanationTree)


@njit(cache=True)
def expl_tree_ctor(entries=None, planner=None):
    st = new(ExplanationTreeType)
    if(entries is None):
        st.entries = List.empty_list(ExplanationTreeEntryType)
    else:
        st.entries = entries
    if(planner is not None):
        st.inv_val_map_ptr_dict = Dict.empty(u2, ptr_t)
        for t_id, ptr in planner.inv_val_map_ptr_dict.items():
            # _incref_ptr(ptr) #Note might not need
            st.inv_val_map_ptr_dict[u2(t_id)] = ptr
    return st




#----------------------------
# : build_explanation_tree()

@njit(cache=True)
def extract_rec_entry(d_ptr):
    '''Decodes the record entry at a pointer to its location in the underlaying data of 
        an SC_Record.data array and outputs the SC_Record instance, next data pointer, 
        and argument indicies.'''
    ptrs = _arr_from_data_ptr(d_ptr, (2,),dtype=np.int64)
    rec_ptr, next_entry_ptr = ptrs[0], ptrs[1]
    if(rec_ptr == 0):
        raise ValueError("Null Record Pointer. Check that func or conversion didn't silently fail.")
    # print("R", rec_ptr, next_entry_ptr)
    # print(rec_ptr, next_entry_ptr)
    # if(next_entry_ptr <= 4): raise ValueError()
    rec = cast(rec_ptr, SC_RecordType)
    # print("S", rec_ptr, d_ptr, rec.n_args)
    args = _arr_from_data_ptr(d_ptr+16,(rec.n_args,),dtype=np.uint32)
    # print("T")
    return rec, next_entry_ptr, args


i8_et_dict = DictType(i8,ExplanationTreeType)

@njit(void(
        i8,
        DictType(u2,i8_et_dict),
        ExplanationTreeType,
        i8),
     cache=True)
def _fill_arg_inds_from_rec_entries(re_ptr, new_arg_inds, expl_tree, retrace_depth):
    '''Goes through linked list of record entries and
        adds the argument indicies into new_arg_inds
        with a new instance of an ExplanationTree if needed.
        Add the new ExplanationTrees to the children of 'expl_tree'
    '''
    while(re_ptr != 0):
        re_rec, re_next_re_ptr, re_args = extract_rec_entry(re_ptr)
        # print("REARGS", re_args, re_next_re_ptr)

        # Skip any records that 
        if(re_rec.depth > retrace_depth and re_rec.depth != 0):
            re_ptr = re_next_re_ptr
            continue
        # print(re_rec.depth, retrace_depth)
        if(re_rec.is_func):
            func = re_rec.func 
            # print(func, re_ptr, re_next_re_ptr, re_args)
            child_arg_ptrs = List.empty_list(ptr_t, len(re_args))#np.empty(len(re_args), dtype=np.int64)
            for i, (hr, arg_ind) in enumerate(zip(func.head_ranges, re_args)):
                arg_t_id = u2(func.head_infos[hr.start].base_t_id)
                #Make sure a set of indicies has been instantied for 'arg_t_id'
                if(u2(arg_t_id) not in new_arg_inds):
                    new_arg_inds[u2(arg_t_id)] = Dict.empty(i8,ExplanationTreeType)
                uai = new_arg_inds[u2(arg_t_id)]

                # Fill in a new ExplanationTree instance if needed
                if(arg_ind not in uai):
                    uai[arg_ind] = expl_tree_ctor()

                child_arg_ptrs.append(_ptr_from_struct_incref(uai[arg_ind]))

            # Throw new tree entry instance into the children of 'expl_tree'
            entry = func_expl_tree_entry_ctor(func, child_arg_ptrs)
            expl_tree.entries.append(entry)
            # re = next_rec_entry(re)  
            # re_ptr = re_next_re_ptr
            # re_rec, re_next_re, re_args = extract_rec_entry(re_next_re)
        else:
            entry = var_expl_tree_entry_ctor(re_rec.var, is_const=re_rec.is_const)
            expl_tree.entries.append(entry)
            # re = None
        re_ptr = re_next_re_ptr


def _retrace_arg_inds(planner, typ,  goal_expltree_maps, retrace_depth,
                 new_arg_inds=None, lowest_only=False):
    pass

@overload(_retrace_arg_inds) 
def retrace_arg_inds_overload(planner, typ,  goal_expltree_maps, retrace_depth,
                 new_arg_inds=None, lowest_only=False):
    '''Find applications of operations that resulted in each 
        goal in goals. Add the indicies of the args as they
        occur in flat_vals.
    '''
    # The actual type inside the type ref 'typ'
    _typ = typ.instance_type

    val_map_d_typ = DictType(_typ, i8_3x_tuple) 
    _goal_map_d_typ = DictType(_typ, ExplanationTreeType) 
    # typ_name = str(_typ)
    typ_t_id = cre_context().get_t_id(_typ)
    def impl(planner, typ, goal_expltree_maps, retrace_depth,
                new_arg_inds=None, lowest_only=False):
        if(new_arg_inds is None):
            new_arg_inds = Dict.empty(u2, i8_et_dict)
        val_map =  _dict_from_ptr(val_map_d_typ,
            planner.val_map_ptr_dict[u2(typ_t_id)])
        
        _goal_map = _dict_from_ptr(_goal_map_d_typ, goal_expltree_maps[u2(typ_t_id)])
        for goal, expl_tree in _goal_map.items():
            low_depth, prev_low_entry, prev_entry = val_map[goal]
            entry_ptr = prev_low_entry if lowest_only else prev_entry

            _fill_arg_inds_from_rec_entries(entry_ptr,
                new_arg_inds, expl_tree, retrace_depth)
            
        return new_arg_inds
    return impl

@njit(cache=True)
def retrace_arg_inds(planner, typ,  goal_expltree_maps, retrace_depth,
                 new_arg_inds=None, lowest_only=False):
    return _retrace_arg_inds(planner, typ,  goal_expltree_maps, retrace_depth,
                 new_arg_inds, lowest_only)


def _fill_subgoals_from_arg_inds(planner, arg_inds, typ, depth, new_subgoals):
    pass

@overload(_fill_subgoals_from_arg_inds)
def fill_subgoals_from_arg_inds_overload(planner, arg_inds, typ, depth, new_subgoals):
    '''Fill 'new_subgoals' with the actual values pointed to by
         the arg_inds of 'typ' '''
    _typ = typ.instance_type
    # typ_name = str(_typ)
    lst_typ = ListType(_typ)
    typ_t_id = cre_context().get_t_id(_typ)
    def impl(planner, arg_inds, typ, depth, new_subgoals):
        _new_subgoals =  Dict.empty(typ, ExplanationTreeType)
        _arg_inds = arg_inds[u2(typ_t_id)]
        
        # In the event that we haven't produced flat vals for this 
        #  type then do it now.
        if((u2(typ_t_id), depth) not in planner.flat_vals_ptr_dict):
            join_records_of_type(planner, depth, _typ)
        
        vals = _list_from_ptr(lst_typ, planner.flat_vals_ptr_dict[(u2(typ_t_id), depth)])
        for ind, expl_tree in _arg_inds.items():
            sub_goal = vals[ind]

            _new_subgoals[sub_goal] = expl_tree
        
        # Inject the new subgoals for 'typ' into 'new_subgoals'
        new_subgoals[u2(typ_t_id)] = _ptr_from_struct_incref(_new_subgoals)
        # print(">> new: ", typ_name, new_subgoals[typ_name])
        return new_subgoals
    return impl

@njit(cache=True)
def fill_subgoals_from_arg_inds(planner, arg_inds, typ, depth, new_subgoals):
    return _fill_subgoals_from_arg_inds(planner, arg_inds, typ, depth, new_subgoals)


def retrace_goals_back_one(planner, goal_expltree_maps, retrace_depth,
             lowest_only=False, ignore_zero=False):
    # print("\nRETRACE:", goal_expltree_maps, planner.curr_infer_depth)
    context = cre_context()

    new_arg_inds = None
    for t_id in goal_expltree_maps:
        
        typ = context.get_type(t_id=t_id)#f8 if typ_name == 'float64' else unicode_type
        # print("! gem", typ_name, typ)
        new_arg_inds = retrace_arg_inds(planner, typ, goal_expltree_maps,
         retrace_depth, new_arg_inds, lowest_only)

    # print(retrace_depth, "new_arg_inds", {k : list(v) for k,v in new_arg_inds.items()})
    if(len(new_arg_inds) == 0):
        return None
    new_subgoals = _init_subgoal_expltree_maps()

    for t_id in new_arg_inds:
        # typ = context.name_to_type[typ_name]#f8 if typ_name == 'float64' else unicode_type
        typ = context.get_type(t_id=t_id)#f8 if typ_name == 'float64' else unicode_type
        # print("! nai", typ_name, typ)
        new_subgoals = fill_subgoals_from_arg_inds(
                planner, new_arg_inds, typ,
                planner.curr_infer_depth, new_subgoals)

    return new_subgoals

def __init_root_goal_expltree_maps(planner, g_typ, goal):
    pass

@overload(__init_root_goal_expltree_maps)
def _init_root_goal_expltree_maps_overload(planner, g_typ, goal):
    g_typ_t_id = cre_context().get_t_id(g_typ.instance_type)
    def impl(planner, g_typ, goal):
        root = expl_tree_ctor(None,planner)
        goal_expltree_maps = Dict.empty(u2, ptr_t)

        _goal_map = Dict.empty(g_typ, ExplanationTreeType)
        _goal_map[goal] = root

        goal_expltree_maps[u2(g_typ_t_id)] = _ptr_from_struct_incref(_goal_map)
        return root, goal_expltree_maps
    return impl

@njit(cache=True)
def _init_root_goal_expltree_maps(planner, g_typ, goal):
    return __init_root_goal_expltree_maps(planner, g_typ, goal)

@njit(cache=True)
def _init_subgoal_expltree_maps():
    return Dict.empty(u2, ptr_t)

def build_explanation_tree(planner, g_typ, goal, fewer_solutions=True, ignore_inner_zeros=False):

    if(ignore_inner_zeros):
        raise NotImplemented()
    # print("GOAL: ", goal)
    # t_id = cre_context().get_t_id(g_typ)
    root, goal_expltree_maps = _init_root_goal_expltree_maps(planner, g_typ, goal)
    # print("GOAL: ", goal_expltree_maps)
    retrace_depth = planner.curr_infer_depth

    ignore_zero = ignore_inner_zeros and retrace_depth > 0
    subgoal_expltree_maps = retrace_goals_back_one(planner, goal_expltree_maps,
        retrace_depth, ignore_zero=ignore_zero)
    # print("SUB GOAL: ", subgoal_expltree_maps)
    while(subgoal_expltree_maps is not None):
        retrace_depth -= 1

        # NOTE: Conversions can cause depth zero to be expanded a second time
        if(retrace_depth < 0): retrace_depth = 0

        ignore_zero = ignore_inner_zeros and retrace_depth > 0

        subgoal_expltree_maps = retrace_goals_back_one(planner, subgoal_expltree_maps,
             retrace_depth, lowest_only=fewer_solutions, ignore_zero=ignore_zero)   
        # print(subgoal_expltree_maps)
        
    
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


# @njit(i8(ExplanationTreeType,),cache=True)
# def expl_tree_num_entries(tree):
#     return len(tree.entries)

# @njit(ExplanationTreeEntryType(ExplanationTreeType,i8),cache=True)
# def expl_tree_ith_entry(tree, i):
#     return tree.entries[i]

# @njit(i8(ExplanationTreeEntryType),cache=True)
# def expl_tree_entry_num_args(tree_entry):
#     return len(tree_entry.child_arg_ptrs)

# @njit(u1(ExplanationTreeEntryType),cache=True)
# def expl_tree_entry_is_func(tree_entry):
#     return tree_entry.is_func

# @njit(CREFuncType(ExplanationTreeEntryType),cache=True)
# def expl_tree_entry_get_op(tree_entry):
#     return tree_entry.op

# @njit(VarType(ExplanationTreeEntryType),cache=True)
# def expl_tree_entry_get_var(tree_entry):
#     return tree_entry.var

# @njit(ExplanationTreeType(ExplanationTreeEntryType,i8),cache=True)
# def expl_tree_entry_jth_arg(tree_entry, j):
#     return cast(tree_entry.child_arg_ptrs[j], ExplanationTreeType)


# def product_of_generators(generators):
#     '''Takes a list of generators and applies the equivalent of
#         itertools.product() on them. Has significantly less memory
#         overhead in cases when you only need a subset of the full product.
#     '''
#     iters = []
#     out = []
    
#     while(True):
#         # Create any missing iterators from generators
#         while(len(iters) < len(generators)):
#             it = generators[len(iters)]()
#             iters.append(it)
        
#         iter_did_end = False
#         while(len(out) < len(iters)):
#             # Try to fill in any missing part of out
#             try:
#                 nxt = next(iters[len(out)])
#                 out.append(nxt)
#             # If any of the iterators failed pop up an iterator
#             except StopIteration as e:
#                 # Stop yielding when 0th iter reaches end
#                 if(len(iters) == 1):
#                     return
#                 out = out[:-1]
#                 iters = iters[:-1]
#                 iter_did_end = True
        
#         # If any of the iterators reached their end then 
#         #  we'll need to generate one or more new ones.
#         if(iter_did_end): continue

#         yield out
#         out = out[:-1]


# class ExplTreeGen():
#     '''Helper object that is essentially a lambda that applies
#         gen_op_comps_from_expl_tree() on a particular ExplanationTree
#     '''
#     def __init__(self,child_tree):
#         self.child_tree = child_tree
#     def __call__(self):
#         return gen_op_comps_from_expl_tree(self.child_tree)
        
# # from CREFunc import OpComp
# def gen_op_comps_from_expl_tree(tree):
#     '''A generator of OpComps from an ExplanationTree'''
#     for i in range(expl_tree_num_entries(tree)):
#         tree_entry = expl_tree_ith_entry(tree, i)
        
#         if(expl_tree_entry_is_func(tree_entry)):
#             func = expl_tree_entry_get_op(tree_entry)
#             # print(op)
#             # op = func.recover_singleton_inst()
#             # print(op)
#             child_generators = []
#             for j in range(expl_tree_entry_num_args(tree_entry)):
#                 child_tree = expl_tree_entry_jth_arg(tree_entry,j)
#                 child_gen = ExplTreeGen(child_tree)
#                 child_generators.append(child_gen)

#             for args in product_of_generators(child_generators): 
#                 # print("<<", op, [(str(x.base_type), str(x.head_type)) if(isinstance(x,Var)) else x for x in args])
                
#                 func_comp = func(*args)
#                 # print("::", func_comp, args, [x._meminfo.refcount if isinstance(x,Var) else "" for x in args])
#                 yield func_comp
#         else:
#             v = expl_tree_entry_get_var(tree_entry)
#             yield v

#-----------------------------
# : Explanation Tree Iterator

# skip "g" to reserve for goals, i->o because those are often used for iterators 
default_param_names = ("a","b","c","d","e","f",  "h",  "p","q",
                       "r","s","t","u","v","w","x","y","z")
@njit(CREFuncType(CREFuncType), cache=True)
def default_reparam(cf):
    if(cf.n_args > len(default_param_names)):
        raise ValueError("Too many arguments to reparametrize.")

    cf = cre_func_deep_copy_generic(cf)
    for i in range(cf.n_args):
        hi = cf.head_infos[cf.head_ranges[i].start]
        v = cast(hi.var_ptr, VarType)
        new_base = var_ctor_generic(v.base_t_id, default_param_names[i])
        set_var_arg(cf, i, new_base)
    reinitialize(cf)
    return cf

# reparam_func_type = types.FunctionType(CREFuncType(CREFuncType))
# default_reparam_addr = _get_wrapper_address(default_reparam, CREFuncType(CREFuncType))


expl_tree_iterator_field_dict = {
    # A reference to the ExplanationTree being iterated over
    "tree" : ExplanationTreeType,
    
    # The current entry index, exactly one instance of ExplanationTreeIterator
    #  will be incremented on each iteration. Several may overflow back to 0. 
    "entry_ind" : i8,

    # The number of entries in tree.entries
    "n_entries" : i8,
    
    # Iterators for child arguments
    "arg_iters" : ListType(CREObjType), # Note: Ought to be defered type,

    # The last function generated by the iterator or NULL ptr, used 
    #  to cache functions that remain unchanged between iterations
    "cached_func" : CREFuncType,

    # Address of a function that reparametrizes the output of the iterator
    #  to give it more conventional base var names (i.e. zDb.B*qBH.B -> a.B*b.B)
    # "reparam_func_addr" : i8 

    # A singleton reference to the Identity builtin CREFunc used in
    #  cases where the composition is just a Var to make it callable.
    # "identity_func" : CREFuncType
}

ExplanationTreeIterator, ExplanationTreeIteratorType = define_structref("ExplanationTreeIterator", expl_tree_iterator_field_dict)


@njit(ExplanationTreeIteratorType(ExplanationTreeType), cache=True)
def new_expl_tree_iterator(tree):
    st = new(ExplanationTreeIteratorType)
    st.tree = tree
    st.entry_ind = 0
    st.n_entries = len(tree.entries)
    st.arg_iters = List.empty_list(CREObjType)
    st.cached_func = cast(0,CREFuncType)
    # st.reparam_func_addr = reparam_func_addr
    # if(identity_func is not None):
    #     st.identity_func = identity_func
    # st.args = List.empty_list(CREObjType)
    return st
    # st.arg_inds = np.zeros(len(tree.child_arg_ptrs),dtype=np.uint64)


from cre.func import set_const_arg_impl
f_inv_val_t = DictType(i8, f8)
i_inv_val_t = DictType(i8, i8)
s_inv_val_t = DictType(i8, unicode_type)
set_const_arg_f = set_const_arg_impl(f8)
set_const_arg_i = set_const_arg_impl(i8)
set_const_arg_s = set_const_arg_impl(unicode_type)

# get_read_inv_val_map_impl()

@njit
def set_preset_const_arg(cf, j, var, ivm_dict):
    t_id = var.head_t_id
    var_ptr = cast(var, i8)
    inv_val_map_ptr = ivm_dict[u2(t_id)]
    
    if(t_id == T_ID_FLOAT):
        f_inv_val_map = _dict_from_ptr(f_inv_val_t, inv_val_map_ptr)
        f_val = f_inv_val_map[var_ptr]
        set_const_arg_f(cf, j, f_val)
    elif(t_id == T_ID_INT):
        i_inv_val_map = _dict_from_ptr(i_inv_val_t, inv_val_map_ptr)
        i_val = i_inv_val_map[var_ptr]
        set_const_arg_i(cf, j, i_val)
    elif(t_id == T_ID_STR):
        s_inv_val_map = _dict_from_ptr(s_inv_val_t, inv_val_map_ptr)
        s_val = s_inv_val_map[var_ptr]
        set_const_arg_s(cf, j, s_val)
    else:
        raise ValueError(f"Constant with T_ID {t_id} not supported.")




obj_u1_tup_t = Tuple((CREObjType, u1))

@njit(Tuple((types.optional(VarType),types.Optional(CREFuncType)))(ExplanationTreeIteratorType), cache=True)
def expl_tree_iter_next(t_iter):
    if(t_iter.entry_ind == -1):
        return None, None

    ivm_dict = t_iter.tree.inv_val_map_ptr_dict
    # Whether to increment the t_iter. Remains true until an
    #  incrementation suceeds without overflow.
    keep_incrementing = True

    stack = List()
    i = 0 
    args = List.empty_list(obj_u1_tup_t)
    while(True):
        
        entry = t_iter.tree.entries[t_iter.entry_ind]
        # ------
        #  : Depth first traversal 

        # Func Case
        if(entry.is_func):

            if(keep_incrementing):
                # Don't use cached func until hits an unchanged iter
                t_iter.cached_func = cast(0,CREFuncType)

            # If no more incrementation use the cached func if not NULL
            if(not keep_incrementing and cast(t_iter.cached_func,i8) != 0):
                # print(f"Skip ({t_iter.entry_ind}/{t_iter.n_entries})", entry.func.origin_data.name)
                f_obj = cast(t_iter.cached_func, CREObjType)
                t_iter, i, args = stack.pop(-1)
                args.append((f_obj, entry.is_const))
            else:
                if(i < len(entry.child_arg_ptrs)):
                    # print(f"Iter ({t_iter.entry_ind}/{t_iter.n_entries})", f"i:{i}/{len(entry.child_arg_ptrs)}", entry.func)

                    # Push this frame to stack
                    stack.append((t_iter, i+1, args))

                    # Recurse into argument's frame
                    if(i < len(t_iter.arg_iters)):
                        t_iter = cast(t_iter.arg_iters[i], ExplanationTreeIteratorType)
                    else:
                        # Make fresh t_iter if it is missing
                        tree = cast(entry.child_arg_ptrs[i], ExplanationTreeType)
                        _t_iter = new_expl_tree_iterator(tree)
                        t_iter.arg_iters.append(cast(_t_iter,CREObjType))
                        t_iter = _t_iter

                    args = List.empty_list(obj_u1_tup_t)
                    i = 0
                else:
                    # print(f"Term ({t_iter.entry_ind}/{t_iter.n_entries})", entry.func)
                    # ------
                    #  : Reached the last entry in breadth 
                    #     i.e. all args have been resolved so need to pop back w/ CREFunc

                    # Make the cre_func from the collected args
                    cf = cre_func_deep_copy_generic(entry.func)
                    for j, (arg, is_const) in enumerate(args):
                        t_id, _, _ = decode_idrec(arg.idrec)
                        if(t_id == T_ID_VAR):
                            var = cast(arg, VarType)
                            if(is_const):
                                # print()
                                set_preset_const_arg(cf, j, var, ivm_dict)
                                print("IS CONST")
                            else:
                                # print(f"  {j}:", cast(arg, VarType))
                                set_var_arg(cf, j, var)
                        else:
                            # print(f"  {j}:", cast(arg, CREFuncType))
                            set_func_arg(cf, j, cast(arg, CREFuncType))
                    reinitialize(cf)

                    # if(t_iter.reparam_func_addr != 0):
                    #     reparam_func = _func_from_address(reparam_func_type, t_iter.reparam_func_addr)
                    #     cf = reparam_func(cf)


                    # Increment any t_iters in this frame if appropriate 
                    if(keep_incrementing):
                        t_iter.entry_ind += 1
                        if(t_iter.entry_ind >= t_iter.n_entries):
                            if(len(stack) == 0):
                                # Indicate that the iterator is exhausted
                                t_iter.entry_ind = -1
                            else:
                                t_iter.entry_ind = 0
                        else:
                            keep_incrementing = False

                        t_iter.arg_iters = List.empty_list(CREObjType)
                        t_iter.cached_func = cast(0,CREFuncType)
                    else:
                        t_iter.cached_func = cf

                    # Return when hit initial frame.
                    if(len(stack) == 0):
                        return None, cf

                    # Pop back frame from stack.
                    t_iter, i, args = stack.pop(-1)
                    args.append((cast(cf, CREObjType), entry.is_const))
        # Var and Const Case
        else:
            # print(f"Var:({t_iter.entry_ind}/{t_iter.n_entries})",  str(entry.var))
            
            # Increment if appropriate
            if(keep_incrementing):
                t_iter.entry_ind += 1
                if(t_iter.entry_ind >= t_iter.n_entries):
                    t_iter.entry_ind = 0
                else:
                    keep_incrementing = False

            if(len(stack) == 0):
                # Indicate that the iterator is exhausted
                t_iter.entry_ind = -1
                return entry.var, None

            # Pop back frame from stack.
            t_iter, i, args = stack.pop(-1)
            args.append((cast(entry.var, CREObjType), types.boolean(entry.is_const)))
    
    # Should never reach here
    return None, None

class ExplanationTreeIter():
    def __init__(self, expl_tree, reparam_func=None):
        self.expl_tree = expl_tree
        self.iter = new_expl_tree_iterator(expl_tree)

        if(reparam_func is None):
            reparam_func = default_reparam

        # Get entry point to avoid numba type checking pass
        self.reparam_func = reparam_func
        self.reparam_func_ep = self.reparam_func.overloads[(CREFuncType,)].entry_point

    def __iter__(self):
        return self

    def __next__(self):
        var, func = expl_tree_iter_next(self.iter)
        if(func is None):
            if(var is not None):
                # If iteration returns a Var then convert it to
                #  an Identity CREFunc so that it is callable
                func = Identity(var)
            else:
                raise StopIteration()

        # Get the match values from their associated base var ptrs
        vals = []
        for base_ptr, arg_t_id in zip(func.base_var_ptrs, func.base_t_ids):
            val = get_read_inv_val_map_impl(arg_t_id)(self.expl_tree, base_ptr)
            vals.append(val)

        func = self.reparam_func_ep(func)

        return func, vals


#-----------------------------
# : Explanation Tree Esimate Length and Sample 

# TODO
# @njit(cache=True)
# def expl_tree_estimate_length(expl_tree):
#     pass






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

-Casting could overloaded and built into the funcs themselves either:
    -Explicitly with more than one signature, or a valid arg casting dict

THINGS:
1) Things need to be forced to be declared as the proper type
    -For the time being, just 



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
essentially the funcs for those depth 0 values. This way we already have
a standard way of encoding deref instructions.
So then we need some kind of flag on the recEntry like is_func
it seems like this warrants a new structref TreeEntry to replace 
the tuple that is there now. We probably should not recycle the Record 
instance because it has a ref to 'data' which is big should get able to 
be freed even if the Explanation tree sticks around for a long time.

So we have TreeEntry:
-is_func : u1
-func : CREFuncType
-var : VarType

'''





















