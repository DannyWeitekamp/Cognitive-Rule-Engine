import operator
import numpy as np
import numba
from numba.core.dispatcher import Dispatcher
from numba import types, njit, i8, u8, i4, u1, u2, u4, literally, generated_jit, boolean, literal_unroll
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple, UniTuple, StructRef
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils, StructRefProxy
from numba.extending import lower_cast, NativeValue, box, unbox, overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload
from numba.core.typing.templates import AttributeTemplate
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template, StructRefType
from cre.utils import (_func_from_address, _struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_ptr,  
                       _raw_ptr_from_struct, _raw_ptr_from_struct_incref, _decref_ptr, _incref_ptr, _incref_structref, _ptr_from_struct_incref, ptr_t, _load_ptr)
from cre.utils import encode_idrec, assign_to_alias_in_parent_frame, as_typed_list, lower_setattr, _store
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.fact import Fact, gen_fact_import_str, get_offsets_from_member_types
from cre.var import Var, var_memcopy, GenericVarType, VarTypeClass
from cre.cre_object import CREObjType, cre_obj_field_dict, CREObjTypeClass, CREObjProxy, member_info_type, set_chr_mbrs, cre_obj_get_item_t_id_ptr, cre_obj_set_item
from cre.core import T_ID_OP, register_global_default
from cre.make_source import make_source, gen_def_func, gen_assign, gen_if, gen_not, resolve_template, gen_def_class
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models, register_default
from numba.experimental.function_type import _get_wrapper_address


from operator import itemgetter
from copy import copy
from os import getenv
import inspect, cloudpickle, pickle
from textwrap import dedent, indent
from collections.abc import Iterable
import warnings

from llvmlite import binding as ll
from llvmlite import ir
import cre_cfuncs

from numba.extending import intrinsic, overload_method, overload

_head_range_type = np.dtype([('start', np.uint8), ('length', np.uint8)])
head_range_type = numba.from_dtype(_head_range_type)

_arg_infos_type = np.dtype([('type', np.uint8), ('ptr', np.int64)])
arg_infos_type = numba.from_dtype(_arg_infos_type)

_op_and_arg_ind = np.dtype([('op_ptr', np.int64), ('arg_ind', np.int64)])
op_and_arg_ind = numba.from_dtype(_op_and_arg_ind)

_head_info_type = np.dtype([
    ('cf_ptr', np.int64),
    ('type', np.uint32),
    ('arg_ind', np.uint32),
    ('var_ptr', np.int64),
    ('arg_data_ptr', np.int64),
    ('head_data_ptr', np.int64)])
head_info_type = numba.from_dtype(_head_info_type)

ARGINFO_CONST = u4(0)
ARGINFO_VAR = u4(1)
ARGINFO_OP = u4(2)
ARGINFO_OP_UNEXPANDED = u4(3)

# -----------------------------------------------------------------
# : CREFunc_method decorator

def warn_cant_compile(func_name, cre_f_name, e):
    s = f'''
########## WARNING CAN'T COMPILE {cre_f_name}.{func_name}() ########### 
{indent(str(e),'  ')} 
########################################################
Numba was unable to compile {func_name}() for CREFunc {cre_f_name}. 
Using objmode (i.e. native python) instead. To ignore warning set nopython=False.\n\
'''
    warnings.warn(s, NumbaPerformanceWarning)


def CREFunc_assign_method_addr(cf_type, fn_name, addr):
    name = f"{cf_type.symbol_prefix}_{fn_name}"
    print("88888", name)
    ll.add_symbol(name, addr)

def CREFunc_method(cf_type, fn_name, sig, on_error='error'):
    def wrapper(func):
        try:
            dispatcher = njit(sig, cache=True)(func)
        except NumbaError as e:
            if(on_error == "error"): raise e
            if(on_error == "warn"):
                warn_cant_compile(fn_name, cf_type.name, e)
            return None
        addr = _get_wrapper_address(dispatcher, sig)
        CREFunc_assign_method_addr(cf_type, fn_name, addr)
        dispatcher.cre_method_addr = addr
        return dispatcher
    return wrapper

# ----------------------------------------------------------------------
# : CREFunc jitted side

# ()

cre_func_fields_dict = {
    **cre_obj_field_dict, 

    # The number of arguments taken by this cf
    "n_args" : i8,

    # The args to the root op 
    "root_arg_infos" : arg_infos_type[::1],

    # List of other ops 
    "children" : ListType(StructRefType),

    # References to base vars like Var(BOOP,"A")
    "base_vars" : ListType(GenericVarType),


    "base_to_head_infos" : ListType(head_info_type[::1]),

    # For each base var the underlying arg_ptrs i.e. to a0,...,an
    #  that it is meant to fill in
    "bases_to_arg_ptrs" : ListType(i8[::1]),

    

    # Data ptr of the return value    
    "return_data_ptr" : i8,

    # Data ptrs of args
    # i.e. points to a0,a1,...
    "arg_data_ptrs" : i8[::1],

    # Data ptrs of head_args i.e. args after following deref chains 
    # i.e. points to h0,h1,...
    "arg_head_data_ptrs" : i8[::1],

    # Gives the op ptr and root_arg_ind for each arg_ind
    "arg_op_and_arg_inds" : op_and_arg_ind[::1],
    
    # "call_addr" : i8,
    "call_self_addr" : i8,
    "call_heads_addr" : i8,
    # "call_head_ptrs_addr" : i8,
    # "match_addr" : i8,
    # "match_heads_addr" : i8,
    # "match_head_ptrs_addr" : i8,
    # "check_addr" : i8,

    # True if the op has beed initialized
    "is_initialized" : types.boolean,

    # True if this op is a ptr op
    "is_ptr_op" : types.boolean,

    # True if dereferencing the head args succeeded  
    "heads_derefs_succeeded" : types.boolean,

    # True if check in this and all children suceeded
    "exec_passed_checks" : types.boolean,

    "has_base_to_head_infos" : types.boolean,
    
    "return_t_id" : u2,

    # Placeholders to keep 64-bit aligned
    "padding0": u1,
    # "padding1": u1,

    # Literals and Types Specialized by
    "name" : types.literal(f"GenericCFType"),
    "return_type" : types.Any,
    "arg_types" : types.Any

    # Other members like a0,a1,... h0,h1,... etc. filled in on specialize
}

@structref.register
class CREFuncTypeClass(CREObjTypeClass):
    t_id = T_ID_OP
    def preprocess_fields(self, fields):
        self.t_id = T_ID_OP
        self._field_dict = fields if isinstance(fields,dict) else {k:v for k,v in fields}
        self.return_type = self._field_dict['return_type']
        self.arg_types = self._field_dict['arg_types']
        self.func_name = "GenericCREFunc"
        return fields

    def __str__(self):
        if(self.return_type is not types.Any):
            return f"CREFuncType({self.func_name!r}, {self.arg_types}->{self.return_type})"
        else:
            return f"{self.func_name}"

    def __repr__(self):
        if(self.return_type is not types.Any):
            return f"CREFuncType(name={self.func_name!r}, arg_types={self.arg_types}, return_type={self.return_type})"
        else:
            return f"{self.func_name}"

    @property
    def symbol_prefix(self):
        if(not hasattr(self, "_symbol_prefix")):
            shortened_hash = self.long_hash[:18]
            self._symbol_preix = f"CREFunc_{self.func_name}_{shortened_hash}"
        return self._symbol_preix

    def __getstate__(self):
        d = self.__dict__.copy()
        if('call' in d): del d['call']
        if('check' in d): del d['check']
        return d
    #     print("<< get_state")
    #     return {"name" : self.name, "signature" : self.signature}

    # def __str__(self):
    #     return f"{self.name}({self.arg_types}->{self.return_type})"


GenericCREFuncType = CREFuncTypeClass(cre_func_fields_dict)

def get_cre_func_type(name, return_type, arg_types):
    print(return_type, arg_types)
    arg_fields = {}
    for i,t in enumerate(arg_types):
        arg_fields[f'a{i}'] = CREObjType
        arg_fields[f'h{i}'] = t

    field_dict = {**cre_func_fields_dict,
        'return_val' : return_type,
        **arg_fields,
        **{f'ref{i}' : types.optional(CREObjType) for i in range(len(arg_types))},
        'chr_mbrs_infos' : UniTuple(member_info_type,1+len(arg_types)*2),
        'name' : types.literal(name),
        'return_type' : types.TypeRef(return_type),
        'arg_types' : types.TypeRef(types.Tuple(arg_types)),
        # 'head_chr_mbrs_infos' : UniTuple(member_info_type,len(arg_types)),
    }
    cf_type = CREFuncTypeClass(field_dict)
    cf_type.func_name = name
    cf_type.return_type = return_type
    cf_type.arg_types = tuple(arg_types)
    return cf_type

# ------------------------------
# : CREFunc initialization

# @generated_jit(nopython=True)
# def _cre_func_assign_arg(cf, i, val_or_var):
#     if(isinstance(val_or_var, VarTypeClass)):
#         def impl(cf, i, val_or_var):
#             cf.head_var_cf_ptrs[i] = _raw_ptr_from_struct(val_or_var)
#     else:
#         attr = f'a{i.literal_value}'
#         def impl(cf, i, val_or_var):
#             cf.head_var_cf_ptrs[i] = 0
#             lower_setattr(cf, attr, val_or_var)
#     return impl

@intrinsic
def _get_global_fn_addr(typingctx, literal_name):    
    name = literal_name.literal_value
    def codegen(context, builder, sig, args):
        mod = builder.module
        # Actual type doesn't matter
        fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.voidptr_t])
        fn = cgutils.get_or_insert_function(mod, fnty, name)
        addr = builder.ptrtoint(fn, cgutils.intp_t)
        return addr
        
    sig = i8(literal_name)
    return sig, codegen

@generated_jit(cache=True, nopython=True)
def cre_func_assign_method_table(cf):
    prefix = cf.symbol_prefix
    print('prefix', cf.symbol_prefix)
    method_names = (
        f"{prefix}_call_heads",
        f"{prefix}_call_head_ptrs",
        f"{prefix}_call",
        f"{prefix}_match_heads",
        f"{prefix}_match_head_ptrs",
        f"{prefix}_match",
        f"{prefix}_check",
        f"{prefix}_call_self",
    )
    def impl(cf):
        # cf.call_heads_addr = _get_global_fn_addr(method_names[0])
        # cf.call_head_ptrs_addr = _get_global_fn_addr(method_names[1])
        # cf.call_addr = _get_global_fn_addr(method_names[2])
        # cf.match_heads_addr = _get_global_fn_addr(method_names[3])
        # cf.match_head_ptrs_addr = _get_global_fn_addr(method_names[4])
        # cf.match_addr = _get_global_fn_addr(method_names[5])
        # # cf.check_addr = _get_global_fn_addr(method_names[6])
        cf.call_self_addr = _get_global_fn_addr(method_names[7])
    return impl


head_info_arr = head_info_type[::1]

@generated_jit(cache=True, nopython=True)
def cre_func_new(cf_type):
    fd = cf_type.instance_type._field_dict
    n_args = len(fd['arg_types'].instance_type)
    chr_mbr_attrs = ["return_val"]
    for i in range(n_args):
        chr_mbr_attrs.append(f'a{i}')
        chr_mbr_attrs.append(f'h{i}')
        chr_mbr_attrs.append(f'ref{i}')
    chr_mbr_attrs = tuple(chr_mbr_attrs)
    print(chr_mbr_attrs)
    
    def impl(cf_type):
        cf = new(cf_type)
        cf.n_args = n_args
        cf.root_arg_infos = np.zeros(n_args, dtype=arg_infos_type)
        cf.is_initialized = False
        set_chr_mbrs(cf, chr_mbr_attrs)
        # Point the return_data_ptr to 'return_val'
        _,_,return_data_ptr = cre_obj_get_item_t_id_ptr(cf, 0)
        cf.return_data_ptr = return_data_ptr

        self_ptr = _raw_ptr_from_struct(cf)
        cf.base_to_head_infos = List.empty_list(head_info_arr)
        for i in range(n_args):
            head_infos = np.zeros(1,dtype=head_info_type)
            head_infos[0].cf_ptr = self_ptr
            head_infos[0].arg_ind = u4(i)
            head_infos[0].type = u4(ARGINFO_VAR)
            _,_,arg_data_ptr = cre_obj_get_item_t_id_ptr(cf, 1+(i<<1))
            _,_,head_data_ptr = cre_obj_get_item_t_id_ptr(cf, 1+(i<<1)+1)
            head_infos[0].arg_data_ptr = arg_data_ptr
            head_infos[0].head_data_ptr = head_data_ptr

            cf.base_to_head_infos.append(head_infos)

        # cf.arg_chr_mbrs_infos = _get_chr_mbrs_infos_from_attrs(cf,chr_mbr_attrs)
        # cf.head_chr_mbrs_infos = _get_chr_mbrs_infos_from_attrs(cf,head_attrs)
        # cf.head_var_cf_ptrs = np.zeros(n_args, dtype=np.int64)
        # cf.is_constant = np.zeros(n_args, dtype=np.bool8)
        cre_func_assign_method_table(cf)
        return cf
    return impl


#--------------------------------------------------------------------
# Construction Functions


@njit(cache=True)
def set_const_arg(self, i, val):
    self.is_initialized = False

    head_infos = self.base_to_head_infos[i]
    for j in range(len(head_infos)):
        cf = _struct_from_ptr(GenericCREFuncType, head_infos[j].cf_ptr)
        arg_ind = head_infos[j].arg_ind

        head_infos[j].type = ARGINFO_CONST
        # set 'a{i}' to zero
        cre_obj_set_item(cf, i8(1+(arg_ind<<1)), 0)
        # set 'h{i}' to val
        cre_obj_set_item(cf, i8(1+(arg_ind<<1)+1), val)
        # set 'ref{i}' to None
        cre_obj_set_item(cf, i8(1+cf.n_args*2 + arg_ind), None)

        cf.root_arg_infos[arg_ind].type = ARGINFO_CONST
        cf.root_arg_infos[arg_ind].ptr = 0

    
    # Set 
    # cre_obj_set_item(cf, 1+(i<<1)+1, val)
    

@njit(types.void(GenericCREFuncType,i8,GenericVarType), cache=True)
def set_var_arg(self, i, val):
    self.is_initialized = False

    head_infos = self.base_to_head_infos[i]
    var_ptr = _raw_ptr_from_struct(val)
    for j in range(len(head_infos)):
        cf = _struct_from_ptr(GenericCREFuncType, head_infos[j].cf_ptr)
        arg_ind = head_infos[j].arg_ind
        head_infos[j].var_ptr = var_ptr
        head_infos[j].type = ARGINFO_VAR

        # set 'ai' to zero, set 'hi' to val
        cre_obj_set_item(cf, i8(1+(arg_ind<<1)), var_ptr)
        # set 'ref{i}' to None
        print("A", cf.num_chr_mbrs)
        cre_obj_set_item(cf, i8(1+cf.n_args*2 + arg_ind), val)
        print("B")

        cf.root_arg_infos[arg_ind].type = ARGINFO_VAR
        cf.root_arg_infos[arg_ind].ptr = _raw_ptr_from_struct(val)

@njit(types.void(GenericCREFuncType,i8,GenericCREFuncType), cache=True)
def set_root_op_arg(self, i, val):
    self.is_initialized = False

    head_infos = self.base_to_head_infos[i]
    cf_ptr = _raw_ptr_from_struct(val)
    for j in range(len(head_infos)):
        cf = _struct_from_ptr(GenericCREFuncType, head_infos[j].cf_ptr)
        arg_ind = head_infos[j].arg_ind

        head_infos[j].cf_ptr = _raw_ptr_from_struct(val)
        head_infos[j].type = ARGINFO_OP_UNEXPANDED

        cre_obj_set_item(cf, i8(1+cf.n_args*2 + arg_ind), val)
        # set 'ai' to zero, set 'hi' to val
        # cre_obj_set_item(cf, i8(1+(head_infos[j].arg_ind<<1)), var_ptr)

        cf.root_arg_infos[arg_ind].type = ARGINFO_OP
        cf.root_arg_infos[arg_ind].ptr = _raw_ptr_from_struct(val)


# -set_root_const_arg(i, val) : Primatives
#     -set_root_var_arg(i, val) : Vars
#     -set_root_func_arg(i, val) : CRE_Funcs
    
#     # For modifying with normal call conventions
#     -set_const_arg(i, val) : Primatives
#     -set_var_arg(i, val) : Vars
#     -set_func_arg(i, val) : CRE_Funcs
    
#     # Update children, base_vars, head_vars
#     -reinitialize()

i8_arr = i8[::1]
i8_lst = ListType(i8)
@njit(cache=True)
def reinitialize(self):
    if(self.is_initialized): return

    # self.children = List.empty_list(StructRefType)
    # self.base_vars = List.empty_list(GenericVarType)
    # self.bases_to_arg_ptrs = List.empty_list(i8_arr)
    # self.head_vars = List.empty_list(GenericVarType)

    print(">>", self.root_arg_infos)
    # Count the number of head vars in the new composition
    n_head_vars = 0
    for info in self.root_arg_infos:
        if(info.type == ARGINFO_VAR):
            n_head_vars += 1
        elif(info.type == ARGINFO_OP):
            n_head_vars +=len(_struct_from_ptr(GenericCREFuncType, info.ptr).head_vars) 

    # self.arg_data_ptrs = np.empty(n_head_vars, dtype=np.int64)
    self.arg_head_data_ptrs = np.empty(n_head_vars, dtype=np.int64)
    # self.arg_op_and_arg_inds = np.empty(n_head_vars, dtype=op_and_arg_ind)

    c = 0
    base_var_map = Dict.empty(i8,i8_lst)
    self_ptr = _raw_ptr_from_struct(self)
    for i, info in enumerate(self.root_arg_infos):
        if(info.type == ARGINFO_VAR):
            var = _struct_from_ptr(GenericVarType, info.ptr) 
            if(var.base_ptr not in base_var_map):
                base_var_map[var.base_ptr] = List.empty_list(i8)

            
            self.head_vars.append(var)
            # self.arg_data_ptrs[c] = 0 #??

            _,_,arg_data_ptr = cre_obj_get_item_t_id_ptr(self, 1+(i<<1))
            _,_,head_data_ptr = cre_obj_get_item_t_id_ptr(self, 1+(i<<1)+1)
            base_arg_ptrs = base_var_map[var.base_ptr]
            base_arg_ptrs.append(arg_data_ptr)
            print("data_ptr", 1+(i<<1)+1, head_data_ptr)
            self.arg_head_data_ptrs[c] = head_data_ptr #??
            # self.arg_op_and_arg_inds[c].op_ptr = self_ptr
            # self.arg_op_and_arg_inds[c].arg_ind = i #??
            c += 1

        elif(info.type == ARGINFO_OP):
            cf = _struct_from_ptr(GenericCREFuncType, info.ptr)
            # reinitialize(cf)
            for bv in cf.base_vars:
                base_var_map[bv.base_ptr] = List.empty_list(i8)
            for hv in cf.head_vars:
                self.head_vars.append(hv)

            L = len(cf.arg_head_data_ptrs)
            # self.arg_data_ptrs[c:c+L] = cf.arg_data_ptrs[:L]
            self.arg_head_data_ptrs[c:c+L] = cf.arg_head_data_ptrs[:L] #??
            # self.arg_op_and_arg_inds[c:c+L] = cf.arg_op_and_arg_inds[:L]



    for b_ptr,base_arg_ptrs in base_var_map.items():
        self.base_vars.append(_struct_from_ptr(GenericVarType, b_ptr))
        _base_arg_ptrs = np.empty(len(base_arg_ptrs),dtype=np.int64)
        for i,v in enumerate(base_arg_ptrs):
            _base_arg_ptrs[i] = v
        print(_base_arg_ptrs)
        self.bases_to_arg_ptrs.append(_base_arg_ptrs)

    print("children", self.children)
    self.is_initialized = True
        



#--------------------------------------------------------------------
# Execution Functions

# -set_head_arg_val(i, val)

# -set_base_arg_val(i, val) 
#   Assigns argument i for all subtrees to val
    
# -update_heads()
#   -use a0,...,an to set h0,....,hn, heads_derefs_succeeded 

# -update_children()
#   -in a DFS of children call exec()
  
# -call_heads()
#   Run call() and check() on h0,...hn 
#   set exec_passed_checks, return_val

@generated_jit(cache=True, nopython=True)
def set_head_arg_val(self, i, val):
    val_type = val
    def impl(self, i, val):
        data_ptr = self.arg_head_data_ptrs[i]
        _store(val_type, data_ptr, val)
    return impl
    

@generated_jit(cache=True, nopython=True)
def set_base_arg_val(self, i, val):
    val_type = val
    if(isinstance(val_type,StructRef)):
        def impl(self, i, val):
            arg_ptrs = self.bases_to_arg_ptrs[i]
            for data_ptr in arg_ptrs:
                _store(CREObjType, data_ptr, val)
    else:
        def impl(self, i, val):
            arg_ptrs = self.bases_to_arg_ptrs[i]
            for data_ptr in arg_ptrs:
                _store(val_type, data_ptr+8, val)

    return impl


# @njit(cache=True)
# def update_heads(self, i, val):
#     for 

@njit(cache=True)
def update_children(self, i, val):
    pass


call_self_f_type = types.FunctionType(types.void(GenericCREFuncType))

@njit(cache=True)
def cre_func_exec_heads(self):
    print("call_self_addr", self.call_self_addr)
    call_self_func = _func_from_address(call_self_f_type,
                     self.call_self_addr)
    call_self_func(self)

    

@njit(cache=True)
def cre_func_exec(self):
    for child in self.children:
        cre_func_exec_heads(_cast_structref(GenericCREFuncType, child))    
    cre_func_exec_heads(self)


@njit(cache=True)
def get_return_val(self, return_type):
    return _load_ptr(return_type, self.return_data_ptr)
# @generated_jit(cache=True, nopython=True)
# def cre_func_init(cf, *args):
#     print(cf.__dict__.keys())
#     if(isinstance(args,tuple)): args = args[0]
#     arg_types = tuple([*cf._field_dict['arg_types'].instance_type])
#     n_args = len(arg_types)
#     rng = tuple(range(n_args))
#     is_constant = tuple([not isinstance(x, VarTypeClass) for x in args])

#     if(len(args) > 0):
#         if(n_args != len(args)): 
#             raise ValueError(f"{cf}, takes {n_args} but got {len(args)}")
#         def impl(cf, *args):
#             for i in literal_unroll(rng):
#                 _cre_func_assign_arg(cf, i, args[i])
#                 cf.is_constant[i] = is_constant[i]
#             cf.has_initialized = False
#             return cf
#     else:
#         def impl(cf, *args):
#             for i in literal_unroll(rng):
#                 _cre_func_assign_arg(cf, i, Var(arg_types[i], f'a{str(i)}'))
#             cf.has_initialized = False
#             return cf
#     return impl


#------------------------------------------------------------------
# : define_CREFunc()

def _standardize_commutes(members):
    '''Standardize 'commutes' and build 'right_commutes' from 'commutes' '''

    # If commutes wasn't given then just assign default empty values (i.e. nothing commutes)
    commutes = members['commutes'] = members.get('commutes',[])
    right_commutes = members['right_commutes'] = members.get('right_commutes',{})
    if('commutes' not in members): return

    # If commutes is True then all args of the same type commute
    arg_types = members['arg_types']
    if(isinstance(commutes,bool)):
        if(commutes == True):
            commutes, d = [], {}
            for i, typ in enumerate(arg_types): 
                d[typ] = d.get(typ,[]) + [i] 
            for typ, inds in d.items():
                commutes += [inds]
        else:
            commutes = []
    else:
        assert(isinstance(commutes,Iterable))

    # Fill in right commutes 
    right_commutes = {}
    for i in range(len(commutes)):
        commuting_set = commutes[i]
        for j in range(len(commuting_set)-1,0,-1):
            right_commutes[commuting_set[j]] = np.array(commuting_set[0:j],dtype=np.int64)
            for k in commuting_set[0:j]:
                typ1, typ2 = arg_types[k], arg_types[commuting_set[j]]
                assert typ1==typ2, \
        f"cre.Op {members['name']!r} has invalid 'commutes' member {commutes}. Signature arguments {k} and {commuting_set[j]} have different types {typ1} and {typ2}."
    right_commutes = right_commutes

    members['commutes'] = commutes
    members['right_commutes'] = right_commutes

def _standardize_nopython(members):
    ''' 'nopython_call' and 'nopython_check' default to value of 'no_python' or None '''
    default = members.get('nopython',None)
    members['nopython_call'] = members.get('nopython', default)
    members['nopython_check'] = members.get('nopython', default)

    return members['nopython_call'], members['nopython_check']


def _standardize_string_templates(members):
    if('expression' not in members):
        n_args = len(members['arg_types'])
        members['expression'] = f"{members['name']}({', '.join([f'{{{i}}}' for i in range(n_args)])})"  
    
    if('shorthand' not in members):
        members['shorthand'] = members['expression']

def _standardize_method(members, method_name):
    ''' Extract the py_funcs associated with 'call' or 'check' and pickle them.'''
    if(method_name in members):
        # Extract and store thes python_func
        func = members.get(method_name)
        py_func = func.py_func if isinstance(func, Dispatcher) else func
        # members[f'{method_name}_pyfunc'] = py_func

        # Speed up by caching cloudpickle.dumps(py_func) inside py_func object
        if(hasattr(py_func,'_cre_cloudpickle_bytes')):
            members[f'{method_name}_bytes'] = py_func._cre_cloudpickle_bytes       
        else:
            # If pickle bytes weren't cached then pickle it 
            cp_bytes = cloudpickle.dumps(py_func)
            members[f'{method_name}_bytes'] = cp_bytes
            py_func._cre_cloudpickle_bytes = cp_bytes

        return members[f'{method_name}_bytes']
    else:
        return None
    
def define_CREFunc(name, members):
    ''' Defines a new CREFuncType '''

    # Ensure that 'call' is present and that 'call' and 'check' are functions
    has_check = 'check' in members
    assert 'call' in members, "CREFunc must have call() defined"
    assert hasattr(members['call'], '__call__'), "call() must be a function"
    assert not (has_check and not hasattr(members['check'], '__call__')), "check() must be a function"

    # If no signature is given then create an untyped instance that can be specialized later
    if('signature' not in members):
        return UntypedCREFunc(name,members)

    # inp_members = {**members}

    # Unpack signature
    signature = members['signature']
    return_type = members['return_type'] = signature.return_type
    arg_types = members['arg_types'] = signature.args
    
    # Standardize commutes and nopython
    _standardize_commutes(members)
    nopy_call, nopy_check = _standardize_nopython(members)

    # Get pickle bytes for 'call' and 'check'
    call_bytes =_standardize_method(members, 'call')
    check_bytes = _standardize_method(members, 'check')

    # Regenerate the source for this type if wasn't cached or if 'cache=False' 
    gen_src_args = [name, return_type, arg_types, nopy_call, nopy_check, call_bytes, check_bytes]

    long_hash = unique_hash(gen_src_args)
    print(name, long_hash)
    if(not source_in_cache(name, long_hash)): #or members.get('cache',True) == False):
        source = gen_cre_func_source(*gen_src_args, long_hash)
        source_to_cache(name, long_hash, source)

    # Update members with jitted 'call' and 'check'
    to_import = ['cf_type', 'call'] + (['check'] if(has_check) else [])
    l = import_from_cached(name, long_hash, to_import)

    typ = l['cf_type']
    typ.name = f'{name}_{long_hash}'
    typ.func_name = name
    print("??", typ.name)
    typ.call = l['call']
    if(has_check): typ.check = l['check']
    
    for k,v in members.items():
        setattr(typ,k,v)

    return typ

# -----------------------------------------------------------------------
# : CREFunc Proxy Class

class CREFunc(StructRefProxy):

    def __new__(self,*args, **kwargs):
        ''' A decorator function that builds a new Op'''
        if(len(args) > 1): raise ValueError("CREFunc() takes at most one position argument 'signature'.")
        
        def wrapper(call_func):
            # Make new cre_func_type
            members = kwargs
            members["call"] = call_func
            cf_type = define_CREFunc(call_func.__name__, members)

            print(">>", cf_type)

            # Since decorator replaces, ensure original function is pickle accessible. 
            # cf_type.call_pyfunc = call_func
            # call_func.__qualname__  = call_func.__qualname__ + ".call_pyfunc"

            # print(cf_type.call_pyfunc)
            # print(cloudpickle.dumps(cf_type))

            cre_func = cre_func_new(cf_type)
            cre_func.return_type = cf_type.return_type
            cre_func.cf_type = cf_type
            _vars = []                
            for i, typ in enumerate(cf_type.arg_types):
                v = Var(typ,f'a{i}'); _vars.append(v);
                set_var_arg(cre_func, i, v)
            reinitialize(cre_func)

            #Prevent _vars from freeing itself until after reinitialize
            _vars = None 
            # cre_func.numba_type = cf_type

            return cre_func

        if(len(args) == 1):
            if(isinstance(args[0],(str, numba.core.typing.templates.Signature))):
                kwargs['signature'] = args[0]
            elif(hasattr(args[0],'__call__')):
                return wrapper(args[0])
            else:
                raise ValueError(f"Unrecognized type {type(args[0])} for 'signature'")        

        return wrapper

    def __call__(self,*args):
        print("CALL", args)
        all_const = True
        for arg in args:
            if(isinstance(arg, Var) or isinstance(arg, CREFunc)):
                all_const = False

        if(all_const):
            for i, arg in enumerate(args):
                set_base_arg_val(self, i, arg)
            cre_func_exec(self)
            return get_return_val(self,self.return_type)
        else:
            new_cf = cre_func_new(self.cf_type)
            new_cf.return_type = self.return_type
            new_cf.cf_type = self.cf_type
            for i, arg in enumerate(args):
                if(isinstance(arg, Var)):
                    set_root_var_arg(new_cf, i, arg)
                elif(isinstance(arg, CREFunc)):
                    set_root_op_arg(new_cf, i, arg)
                else:
                    set_root_const_arg(new_cf,i,arg)
            reinitialize(new_cf)
            return new_cf


    
    def set_root_var_arg(self, i, val):
        set_root_var_arg(self, i, val)

    def set_root_op_arg(self, i, val):
        set_root_op_arg(self, i, val)

    def set_root_const_arg(self, i, val):
        set_root_const_arg(self, i, val)



define_boxing(CREFuncTypeClass, CREFunc)


# -------------------------------------------------------------
# : CREFunc Source Generation

def gen_cre_func_source(name, return_type, arg_types,
            nopy_call, nopy_check, call_bytes, check_bytes, long_hash):
    '''Generate source code for the relevant functions of a user defined CREFunc.'''
    arg_names = ', '.join([f'a{i}' for i in range(len(arg_types))])

    on_error_map = {True : 'error', False : 'none', None : 'warn'}
    on_error_call = on_error_map[nopy_call]
    on_error_check = on_error_map[nopy_check]

    nl = "\n"
    source = \
f'''import numpy as np
from numba import njit, void, i8, boolean, objmode
from numba.extending import lower_cast
from numba.experimental.function_type import _get_wrapper_address
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.utils import _func_from_address, _load_ptr, _obj_cast_codegen, _store
from cre.cre_func import CREFunc_method, CREFunc_assign_method_addr, get_cre_func_type, GenericCREFuncType
import cloudpickle


return_type = cloudpickle.loads({cloudpickle.dumps(return_type)})
arg_types = cloudpickle.loads({cloudpickle.dumps(arg_types)})
cf_type = get_cre_func_type({name!r},return_type, arg_types)
cf_type.long_hash = {long_hash!r}

@lower_cast(cf_type, GenericCREFuncType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)

call_sig = return_type(*arg_types)
call_pyfunc = cloudpickle.loads({call_bytes})
{"".join([f'h{i}_type, ' for i in range(len(arg_types))])} = call_sig.args

call = CREFunc_method(cf_type, 'call', call_sig, on_error={on_error_call!r})(call_pyfunc)
if(call is None):
    @CREFunc_method(cf_type, 'call', call_sig)
    def call({arg_names}):
        with objmode(_return=return_type):
            _return = call_pyfunc({arg_names})
        return _return

call_heads = call
CREFunc_assign_method_addr(cf_type, 'call_heads', call.cre_method_addr)

@CREFunc_method(cf_type, 'call_head_ptrs', return_type(i8[::1],))
def call_head_ptrs(ptrs):
{indent(nl.join([f'i{i} = _load_ptr(h{i}_type,ptrs[{i}])' for i in range(len(arg_types))]),prefix='    ')}
    return call_heads({",".join([f'i{i}' for i in range(len(arg_types))])})

@CREFunc_method(cf_type, 'call_self', void(cf_type))
def call_self(self):
    return_val = call({", ".join([f'self.h{i}' for i in range(len(arg_types))])})
    _store(return_type, self.return_data_ptr, return_val)
'''
    if(check_bytes is not None):
        source += f'''
check_pyfunc = cloudpickle.loads({check_bytes})
check_sig = boolean(*arg_types)

check = CREFunc_method(cf_type, 'check', check_sig, on_error={on_error_check!r})(check_pyfunc)
if(check is None):
    @CREFunc_method(cf_type, 'check', check_sig)
    def check({arg_names}):
        with objmode(_return=boolean):
            _return = check_pyfunc({arg_names})
        return _return
'''
    else:
        source += f'''
CREFunc_assign_method_addr(cf_type, 'check', -1)
'''
    source += f'''
@CREFunc_method(cf_type, 'match', boolean(*arg_types))
def match({arg_names}):
    {f"if(not check({arg_names})): return 0" if(check_bytes is not None) else ""}
    return 1 if(call({arg_names})) else 0

match_heads = match
CREFunc_assign_method_addr(cf_type, 'match_heads', match.cre_method_addr)

@CREFunc_method(cf_type, 'match_head_ptrs', boolean(i8[::1],))
def match_head_ptrs(ptrs):
{indent(nl.join([f'i{i} = _load_ptr(h{i}_type,ptrs[{i}])' for i in range(len(arg_types))]),prefix='    ')}
    return match_heads({",".join([f'i{i}' for i in range(len(arg_types))])})


'''
    return source




###### PLAN ####
'''
A few things need to be settled
0) Data Structures
 children
 base_vars
 head_vars

 // 0=const, 1=var 2=op ptr
 arg_infos : (i8,u1)[::1]

 // Points to args
 arg_data_ptrs : i8[::1],

 // Points to head_args i.e. args after following deref chains 
 arg_head_data_ptrs : i8[::1],

 // Gives the op and root_arg_ind for an arg_ind
 arg_func_root_arg_inds : (i8,i8)[::1],
 return_data_ptr : i8,
 heads_derefs_succeeded : u1,
 exec_passed_checks : u1,

 //Generated 
 return_val : Any
 a0...an, h0,...hn : Any

1) Construction Conventions
    # For modifying the root CRE_Func 
    -set_root_const_arg(i, val) : Primatives
    -set_root_var_arg(i, val) : Vars
    -set_root_func_arg(i, val) : CRE_Funcs
    
    # For modifying with normal call conventions
    -set_const_arg(i, val) : Primatives
    -set_var_arg(i, val) : Vars
    -set_func_arg(i, val) : CRE_Funcs
    
    # Update children, base_vars, head_vars
    -reinitialize()

    ### What happens when: ###
    
    << x = Add(a,b.B)
    set_var_arg(0, a) # root_arg_ptrs[0] 
    set_var_arg(1, b.B) # root_arg_ptrs[1] 
    restructure() 
    # Updates children, base_vars, head_vars,
      arg_data_ptrs, arg_head_data_ptrs, arg_func_root_arg_inds

    Add(x,1)
    set_op_arg(0, a) # root_arg_ptrs[0] 
    set_const_arg(1, b.B) # root_arg_ptrs[1] 
    restructure() 

2) Call Conventions
    -set_head_arg_val(i, val)

    -set_base_arg_val(i, val) 
      Assigns argument i for all subtrees to val
        
    -update_heads()
      -use a0,...,an to set h0,....,hn, heads_derefs_succeeded 

    -update_children()
      -in a DFS of children call exec()
      
    -call_heads()
      Run call() and check() on h0,...hn 
      set exec_passed_checks, return_val



    
    



'''
