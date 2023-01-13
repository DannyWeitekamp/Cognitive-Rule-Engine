import operator
import numpy as np
import numba
from numba.core.dispatcher import Dispatcher
from numba import types, njit, i8, u8, i4, u1, u2, u4, literally, generated_jit, boolean, literal_unroll
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple, UniTuple, StructRef
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils, StructRefProxy
from numba.extending import lower_cast, NativeValue, box, unbox, overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, SentryLiteralArgs
from numba.core.typing.templates import AttributeTemplate
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template, StructRefType
from cre.utils import (_nullify_attr, new_w_del, _memcpy, _func_from_address, _struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_ptr,  
                       _raw_ptr_from_struct, _raw_ptr_from_struct_incref, _decref_ptr, _incref_ptr, _incref_structref, _decref_structref, _ptr_from_struct_incref, ptr_t, _load_ptr,
                       _obj_cast_codegen)
from cre.utils import PrintElapse, encode_idrec, assign_to_alias_in_parent_frame, as_typed_list, lower_setattr, _store, _store_safe
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.fact import Fact, gen_fact_import_str, get_offsets_from_member_types
from cre.var import Var, var_memcopy, GenericVarType, VarTypeClass
from cre.cre_object import CREObjType, cre_obj_field_dict, CREObjTypeClass, CREObjProxy, member_info_type, set_chr_mbrs, cre_obj_get_item_t_id_ptr, cre_obj_set_item, PRIMITIVE_MBR_ID
from cre.core import T_ID_OP, T_ID_STR, register_global_default
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

# _head_range_type = np.dtype([('start', np.uint8), ('length', np.uint8)])
# head_range_type = numba.from_dtype(_head_range_type)

# _arg_infos_type = np.dtype([('type', np.uint8), ('ptr', np.int64)])
_arg_infos_type = np.dtype([('type', np.int64), ('ptr', np.int64)])
arg_infos_type = numba.from_dtype(_arg_infos_type)

# _op_and_arg_ind = np.dtype([('op_ptr', np.int64), ('arg_ind', np.int64)])
# op_and_arg_ind = numba.from_dtype(_op_and_arg_ind)

_instr_type = np.dtype([('op_ptr', np.int64), ('return_data_ptr', np.int64), ('size',np.uint32), ('is_ref',np.uint32)])
# _instr_type = np.dtype([('op_ptr', np.int64), ('return_data_ptr', np.int64), ('size',np.int64), ('is_ref',np.int64)])
instr_type = numba.from_dtype(_instr_type)

_head_info_type = np.dtype([
    ('cf_ptr', np.int64),
    ('type', np.uint16),
    ('n_more', np.uint16),
    ('arg_ind', np.uint32),
    ('var_ptr', np.int64),
    ('arg_data_ptr', np.int64),
    ('head_data_ptr', np.int64)])
head_info_type = numba.from_dtype(_head_info_type)

ARGINFO_CONST = u2(0)
ARGINFO_VAR = u2(1)
ARGINFO_OP = u2(2)
ARGINFO_OP_UNEXPANDED = u2(3)

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
        if(not hasattr(cf_type,'dispatchers')):
            cf_type.dispatchers = {}    
        cf_type.dispatchers[fn_name] = dispatcher
        return dispatcher
    return wrapper

# ----------------------------------------------------------------------
# : CREFunc jitted side

# ()

NameData, NameDataType = define_structref("NameData", {
    "name" : unicode_type,
    "expr_template" : unicode_type,
    "shorthand_template" : unicode_type,
})

cre_func_fields_dict = {
    **cre_obj_field_dict, 

    # The object
    "name_data" : NameDataType,
    # "expr_template" : unicode_type,
    # "shorthand_template" : unicode_type,
    # "bob" : unicode_type,

    # The number of arguments taken by this cf.
    "n_args" : i8,

    # The number of arguments taken by the original call() of this cf.
    "root_n_args" : i8,

    # The args to the root op 
    "root_arg_infos" : arg_infos_type[::1],

    # List of other ops 
    # "children" : ListType(StructRefType),

    # References to base vars like Var(BOOP,"A")
    # "base_vars" : ListType(GenericVarType),
    "base_to_head_inds" : u4[::1],
    "head_infos" : head_info_type[::1],
    # "base_to_head_infos" : ListType(head_info_type[::1]),

    "prereq_instrs" : instr_type[::1],

    # For each base var the underlying arg_ptrs i.e. to a0,...,an
    #  that it is meant to fill in
    # "bases_to_arg_ptrs" : ListType(i8[::1]),

    

    # Data ptr of the return value    
    "return_data_ptr" : i8,

    # Data ptrs of args
    # i.e. points to a0,a1,...
    # "arg_data_ptrs" : i8[::1],

    # Data ptrs of head_args i.e. args after following deref chains 
    # i.e. points to h0,h1,...
    # "arg_head_data_ptrs" : i8[::1],

    # Gives the op ptr and root_arg_ind for each arg_ind
    # "arg_op_and_arg_inds" : op_and_arg_ind[::1],
    
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
    # "name" : types.literal(f"GenericCFType"),
    # "return_type" : types.Any,
    # "arg_types" : types.Any

    # Other members like a0,a1,... h0,h1,... etc. filled in on specialize
}

@structref.register
class CREFuncTypeClass(types.Callable, CREObjTypeClass):
    t_id = T_ID_OP

    def __new__(cls, return_type=None, arg_types=None, is_composed=False, name="GenericCREFunc", long_hash=None):
        self = super().__new__(cls)

        if(name == "GenericCREFunc"):
            field_dict = {**cre_func_fields_dict}
        else:
            arg_fields = {}
            for i,t in enumerate(arg_types):
                arg_fields[f'a{i}'] = CREObjType
                arg_fields[f'h{i}'] = t

            N_MINFOS = 1+len(arg_types)*3

            field_dict = {**cre_func_fields_dict,
                'return_val' : return_type,
                **arg_fields,
                **{f'ref{i}' : CREObjType for i in range(len(arg_types))},
                'chr_mbrs_infos' : UniTuple(member_info_type,N_MINFOS),
                # 'name' : types.literal(name),
                # 'return_type' : types.TypeRef(return_type),
                # 'arg_types' : types.TypeRef(types.Tuple(arg_types)),
                # 'head_chr_mbrs_infos' : UniTuple(member_info_type,len(arg_types)),
            }
        self.func_name = name
        self.return_type = return_type
        self.arg_types = arg_types
        self.is_composed = is_composed
        self.t_id = T_ID_OP
        self.long_hash = long_hash
        self._field_dict = field_dict
        return self

    def __init__(self,*args,**kwargs):
        types.StructRef.__init__(self,[(k,v) for k,v in self._field_dict.items()])
        self.name = repr(self)


    # Impl these 3 funcs to subclass types.Callable so can @overload_method('__call__')
    def get_call_type(self, context, args, kws):
        return self.return_type(args)

    def get_call_signatures(self):
        return [self.return_type(self.arg_types)]

    def get_impl_key(self, sig):
        return (type(self), '__call__')

    

    def __str__(self):
        if(self.return_type is not None):
            arg_s = str(self.arg_types) if self.arg_types is not None else "(...)"
            return f"{self.func_name}{arg_s}->{self.return_type}{':c' if self.is_composed else ''}"
        else:
            return f"{self.func_name}"

    def __repr__(self):
        if(self.return_type is not None):
            hsh = f"_{self.long_hash}" if self.long_hash is not None else ""
            arg_s = str(self.arg_types) if self.arg_types is not None else "(...)"
            return f"CREFunc[{self.func_name}{hsh}{arg_s}->{self.return_type}{':c' if self.is_composed else ''}]"
            # return f"CREFunc(name={self.func_name!r}, arg_types={self.arg_types}, return_type={self.return_type}, is_composed={self.is_composed})"
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


GenericCREFuncType = CREFuncTypeClass()

@lower_cast(CREFuncTypeClass, GenericCREFuncType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)
# print(GenericCREFuncType)

# def get_cre_func_type(name, return_type, arg_types, is_composed=False):
#     # print(return_type, arg_types)
    
#     cf_type = CREFuncTypeClass(field_dict, name, return_type, arg_types, is_composed)
#     cf_type.func_name = name
#     cf_type.return_type = return_type
#     cf_type.arg_types = tuple(arg_types)
#     return cf_type

# -----------------------------------------------------------------------
# : CREFunc Proxy Class
@njit(i8(GenericCREFuncType), cache=True)
def cf_get_n_args(self):
    return self.n_args

class CREFunc(StructRefProxy):

    def __new__(self,*args, **kwargs):
        ''' A decorator function that builds a new Op'''
        if(len(args) > 1): raise ValueError("CREFunc() takes at most one position argument 'signature'.")
        
        def wrapper(call_func):
            # Make new cre_func_type
            members = kwargs
            members["call"] = call_func
            # print(">>", members)

            name = call_func.__name__
            arg_names = inspect.getfullargspec(call_func)[0]

            cf_type = define_CREFunc(call_func.__name__, members)
            n_args = len(members['arg_types'])
            expr_template = f"{name}({', '.join([f'{{{i}}}' for i in range(n_args)])})"
            shorthand_template = members.get('shorthand',expr_template)
            # print("shorthand_template:", shorthand_template)

            with PrintElapse("new"):
                cre_func = cre_func_new(
                    cf_type,
                    n_args,
                    name,
                    expr_template,
                    shorthand_template                
                )

            cre_func.return_type = members['return_type']
            cre_func.arg_types = members['arg_types']
            cre_func.cf_type = cf_type
            _vars = []
            get_str_return_val_impl(cre_func.return_type)
            for i, (arg_alias, typ) in enumerate(zip(arg_names,cf_type.arg_types)):
                v = Var(typ,arg_alias); _vars.append(v);
                set_var_arg(cre_func, i, v)
                set_base_arg_val_impl(typ)
                set_const_arg_impl(typ)
            reinitialize(cre_func)

            cre_func._type = cf_type
            # print(cre_func.__dict__.keys())

            # This prevent _vars from freeing until after reinitialize
            _vars = None 
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
        if(len(args) != self.n_args):
            raise ValueError(f"Got {len(args)} args for ?? with {self.n_args} positional arguments.")
        # print("CALL", args)
        all_const = True
        for arg in args:
            # print("----", arg)
            if(isinstance(arg, Var) or isinstance(arg, CREFunc)):
                all_const = False

        if(all_const):
            # print("ALL CONST")
            # Assign each argument to it's slot h{i} in the op's memory space
            for i, arg in enumerate(args):
                impl = set_base_arg_val_impl(arg)
                impl(self, i, arg)

            cre_func_call_self(self)
            args = None
            ret_impl = get_str_return_val_impl(self.return_type)
            return ret_impl(self)
        else:
            # print("START COMPOSE OP", args)
            new_cf = cre_func_copy(self)
            # print(">>", new_cf)
            new_cf.return_type = self.return_type
            new_cf.arg_types = self.arg_types
            new_cf.cf_type = self.cf_type
            for i, arg in enumerate(args):
                if(isinstance(arg, Var)):
                    set_var_arg(new_cf, i, arg)
                elif(isinstance(arg, CREFunc)):
                    # print("--BEF", arg._meminfo.refcount)
                    set_op_arg(new_cf, i, arg)
                    # print("--AFT", arg._meminfo.refcount)
                else:
                    # print("B")
                    impl = set_const_arg_impl(arg)
                    impl(new_cf, i, arg)
                    # print("AF")
            reinitialize(new_cf)

            new_cf._type = CREFuncTypeClass(self.return_type,None,True)
            args = None
            return new_cf

    def __str__(self):
        return cre_func_str(self, True)

    def __repr__(self):
        return cre_func_str(self, True)


    
    def set_var_arg(self, i, val):
        set_var_arg(self, i, val)

    def set_op_arg(self, i, val):
        set_op_arg(self, i, val)

    def set_const_arg(self, i, val):
        set_const_arg(self, i, val)

    n_args = property(cf_get_n_args)



define_boxing(CREFuncTypeClass, CREFunc)

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
    # print('prefix', cf.symbol_prefix)
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
        cf.call_heads_addr = _get_global_fn_addr(method_names[0])
        # cf.call_head_ptrs_addr = _get_global_fn_addr(method_names[1])
        # cf.call_addr = _get_global_fn_addr(method_names[2])
        # cf.match_heads_addr = _get_global_fn_addr(method_names[3])
        # cf.match_head_ptrs_addr = _get_global_fn_addr(method_names[4])
        # cf.match_addr = _get_global_fn_addr(method_names[5])
        # # cf.check_addr = _get_global_fn_addr(method_names[6])
        cf.call_self_addr = _get_global_fn_addr(method_names[7])
    return impl

#################Garbage###################3
SIZEOF_NRT_MEMINFO = 48
@njit(types.void(i8),cache=True)
def cf_del_inject(data_ptr):
    print("BEFORE REMOVE!!!")
    cf = _struct_from_ptr(GenericCREFuncType, data_ptr-SIZEOF_NRT_MEMINFO)

    


cf_del_inject_addr = _get_wrapper_address(cf_del_inject, types.void(i8))
ll.add_symbol("CRE_cf_del_inject", cf_del_inject_addr)

##############################################


@njit(cache=True)
def rebuild_buffer(self, base_to_head_inds=None, head_infos=None, prereq_instrs=None):
    n_args = self.n_args if(base_to_head_inds is None) else len(base_to_head_inds)
    n_heads = self.n_args if(head_infos is None) else len(head_infos)
    n_prereqs = 0 if prereq_instrs is None else len(prereq_instrs)

    # Layout: base_to_head_inds | head_infos | prereq_instrs
    L_ = L = n_args + 10*n_heads
    L += 6 * n_prereqs

    buff = np.zeros(L, dtype=np.uint32)

    if(base_to_head_inds is not None):
        # print("A", n_args, len(base_to_head_inds[:].view(np.uint32)))
        buff[:n_args] = base_to_head_inds[:].view(np.uint32)
    if(head_infos is not None):
        # print("B", L_-n_args, len(head_infos[:].view(np.uint32)))
        buff[n_args: L_] = head_infos[:].view(np.uint32)
    if(prereq_instrs is not None):
        # print("C", L-L_, len(prereq_instrs[:].view(np.uint32)), "n_prereqs", n_prereqs)
        buff[L_: L] = prereq_instrs[:].view(np.uint32)

    # print("BEF")
    self.base_to_head_inds = buff[:n_args]
    self.head_infos = buff[n_args: L_].view(_head_info_type)
    self.prereq_instrs = buff[L_:L].view(_instr_type)
    # print("END")


head_info_arr_type = head_info_type[::1]

@generated_jit(cache=True, nopython=True)
def cre_func_new(cf_type, n_args, name, expr_template, shorthand_template):
    SentryLiteralArgs(['n_args']).for_function(cre_func_new).bind(cf_type, n_args, name, expr_template, shorthand_template)

    fd = cf_type.instance_type._field_dict
    n_args = n_args.literal_value
    chr_mbr_attrs = ["return_val"]
    for i in range(n_args):
        chr_mbr_attrs.append(f'a{i}')
        chr_mbr_attrs.append(f'h{i}')
    for i in range(n_args):
        chr_mbr_attrs.append(f'ref{i}')
    chr_mbr_attrs = tuple(chr_mbr_attrs)
    
    def impl(cf_type, n_args, name, expr_template, shorthand_template):
        cf = new(cf_type)
        # cf = new_w_del(cf_type, 'CRE_cf_del_inject')
        cf.n_args = n_args
        cf.root_n_args = n_args
        cf.root_arg_infos = np.zeros(n_args, dtype=arg_infos_type)
        cf.is_initialized = False
        set_chr_mbrs(cf, chr_mbr_attrs)
        # Point the return_data_ptr to 'return_val'
        _,_,return_data_ptr = cre_obj_get_item_t_id_ptr(cf, 0)
        cf.return_data_ptr = return_data_ptr

        self_ptr = _raw_ptr_from_struct(cf)

        # Init: base_to_head_inds | head_infos | prereq_instrs
        rebuild_buffer(cf)
        # buff = np.zeros(n_args + 10*n_arg, dtype=np.uint32)


        # cf.base_to_head_inds = buff[:n_args]
        # cf.head_infos = buff[n_args: n_args + 10*n_arg].astype(head_info_type)
        # cf.prereq_instrs = buff[n_args + 10*n_arg:].astype(instr_type)

        # cf.base_to_head_infos = List.empty_list(head_info_arr_type)
        # cf.prereq_instrs = np.zeros(0, dtype=instr_type)
        for i in range(n_args):
            head_infos = np.zeros(1,dtype=head_info_type)
            head_infos[0].cf_ptr = self_ptr
            head_infos[0].arg_ind = u4(i)
            head_infos[0].type = ARGINFO_VAR
            head_infos[0].n_more = 0
            # head_infos[0].has_deref = 0
            _,_,arg_data_ptr = cre_obj_get_item_t_id_ptr(cf, 1+(i<<1))
            _,_,head_data_ptr = cre_obj_get_item_t_id_ptr(cf, 1+(i<<1)+1)
            head_infos[0].arg_data_ptr = arg_data_ptr
            head_infos[0].head_data_ptr = head_data_ptr

            cf.base_to_head_inds[i] = i
            cf.head_infos[i] = head_infos[0]

            # cf.base_to_head_infos.append(head_infos)

        cf.name_data = NameData(name, expr_template, shorthand_template)
        # cf.expr_template = expr_template
        # cf.shorthand_template = shorthand_template

        # cf.arg_chr_mbrs_infos = _get_chr_mbrs_infos_from_attrs(cf,chr_mbr_attrs)
        # cf.head_chr_mbrs_infos = _get_chr_mbrs_infos_from_attrs(cf,head_attrs)
        # cf.head_var_cf_ptrs = np.zeros(n_args, dtype=np.int64)
        # cf.is_constant = np.zeros(n_args, dtype=np.bool8)
        
        cre_func_assign_method_table(cf)
        # print("PTR", _raw_ptr_from_struct(cf), _raw_ptr_from_struct(cf)%8)
        casted = _cast_structref(GenericCREFuncType,cf)
        # print("PTR", _raw_ptr_from_struct(casted), _raw_ptr_from_struct(casted)%8)
        return casted 
    return impl

from cre.cre_object import copy_cre_obj
@njit(GenericCREFuncType(GenericCREFuncType), cache=True)
def cre_func_copy(cf):
    # Make a copy of the CreFunc via a memcpy
    cpy = _cast_structref(GenericCREFuncType, copy_cre_obj(cf))

    # Find the the byte offset between the op and its copy
    cf_ptr = _raw_ptr_from_struct(cf)
    cpy_ptr = _raw_ptr_from_struct(cpy)
    cpy_delta = cpy_ptr-cf_ptr

    # Make a copy of base_to_head_infos
    # base_to_head_infos = List.empty_list(head_info_arr_type)
    old_base_to_head_inds = cf.base_to_head_inds.copy()
    old_head_infos = cf.head_infos.copy()
    old_prereq_instrs = cf.prereq_instrs.copy()

    # Nullify these attributes since we don't want the pointers from the 
    #  original to get decref'ed on assignment
    # _nullify_attr(cpy, 'base_to_head_infos')
    _nullify_attr(cpy, 'base_to_head_inds')
    _nullify_attr(cpy, 'head_infos')
    _nullify_attr(cpy, 'root_arg_infos')
    _nullify_attr(cpy, 'name_data')
    _nullify_attr(cpy, 'prereq_instrs')

    rebuild_buffer(cpy, old_base_to_head_inds, old_head_infos, old_prereq_instrs)
    # cpy.old_base_to_head_inds[:] = old_base_to_head_inds[:]
    # cpy.head_infos[:] = old_head_infos[:]

    # head_infos = cf.head_infos.copy()
    # for 

    for i, head_info in enumerate(cf.head_infos):
        cpy.head_infos[i] = head_info
        if(head_info.cf_ptr == cf_ptr):
            cpy.head_infos[i].cf_ptr = cpy_ptr
            cpy.head_infos[i].arg_data_ptr = head_info.arg_data_ptr+cpy_delta
            cpy.head_infos[i].head_data_ptr = head_info.head_data_ptr+cpy_delta




    # for head_infos in cf.base_to_head_infos:
    #     hi_arr = np.empty(len(head_infos),head_info_type)
    #     for i in range(len(head_infos)):
    #         hi_arr[i] = head_infos[i]
    #         if(head_infos[i].cf_ptr == cf_ptr):
    #             hi_arr[i].cf_ptr = cpy_ptr
    #             hi_arr[i].arg_data_ptr = head_infos[i].arg_data_ptr+cpy_delta
    #             hi_arr[i].head_data_ptr = head_infos[i].head_data_ptr+cpy_delta
    #     base_to_head_infos.append(hi_arr)

    cpy.return_data_ptr = cf.return_data_ptr+cpy_delta

    
    # _nullify_attr(cpy, 'expr_template')
    # _nullify_attr(cpy, 'shorthand_template')
    
    # NOTE: Some quirky optimization is probably causing root_arg_infos 
    #  to be allocated on the stack because some cf copies seem to be sharing
    #  root_arg_infos instances even though they are independantly instantiated
    #  making a copy() on instantiation seems to fix it.
    cpy.root_arg_infos = np.zeros(cf.root_n_args, dtype=arg_infos_type).copy()
    # cpy.base_to_head_infos = base_to_head_infos
    # cpy.base_to_head_inds = base_to_head_inds
    cpy.name_data = cf.name_data
    # cpy.prereq_instrs = cf.prereq_instrs
    cpy.is_initialized = False
    # cpy.expr_template = cf.expr_template
    # cpy.shorthand_template = cf.shorthand_template

    # print("::", f"{cf_ptr}:{_raw_ptr_from_struct(cf.root_arg_infos)}","|", f"{cpy_ptr}:{_raw_ptr_from_struct(cpy.root_arg_infos)}")
    # print("&&>",cf_ptr,cf.return_data_ptr,cf.return_data_ptr-cf_ptr)
    # print("&&<",cpy_ptr,cpy.return_data_ptr,cpy.return_data_ptr-cpy_ptr)

    # Return as a GenericCREFuncType to avoid overloading 
    return _cast_structref(GenericCREFuncType,cpy)

#--------------------------------------------------------------------
# Construction Functions


from numba.core.typing.typeof import typeof
set_const_arg_overloads = {}
def set_const_arg_impl(_val):
    nb_val_type = typeof(_val) if not isinstance(_val, types.Type) else _val
    if(nb_val_type not in set_const_arg_overloads):
        
        sig = types.void(GenericCREFuncType, i8, nb_val_type)
        # print("impl",nb_val_type)
        @njit(sig,cache=True)
        def _set_const_arg(self, i, val):
            self.is_initialized = False

            head_infos = self.head_infos
            start = self.base_to_head_inds[i]
            end = start + head_infos[start].n_more
            # head_infos = self.base_to_head_infos[i]
            for j in range(start,end+1):
                cf = _struct_from_ptr(GenericCREFuncType, head_infos[j].cf_ptr)
                arg_ind = head_infos[j].arg_ind

                head_infos[j].type = ARGINFO_CONST
                # set 'h{i}' to val
                cre_obj_set_item(cf, i8(1+(arg_ind<<1)+1), val)
                # set 'ref{i}' to None
                cre_obj_set_item(cf, i8(1+cf.root_n_args*2 + arg_ind), None)

                cf.root_arg_infos[arg_ind].type = ARGINFO_CONST
                # cf.root_arg_infos[arg_ind].has_deref = False
                cf.root_arg_infos[arg_ind].ptr = 0

        set_const_arg_overloads[nb_val_type] = _set_const_arg
    return set_const_arg_overloads[nb_val_type]

# @njit(locals={'self':GenericCREFuncType}, cache=True)
# def set_const_arg(self, i, val):
#     # print("START")
#     self.is_initialized = False

#     head_infos = self.base_to_head_infos[i]
#     for j in range(len(head_infos)):
#         cf = _struct_from_ptr(GenericCREFuncType, head_infos[j].cf_ptr)
#         arg_ind = head_infos[j].arg_ind

#         head_infos[j].type = ARGINFO_CONST
#         # set 'a{i}' to zero
#         # cre_obj_set_item(cf, i8(1+(arg_ind<<1)), 0)
#         # set 'h{i}' to val
#         cre_obj_set_item(cf, i8(1+(arg_ind<<1)+1), val)
#         # set 'ref{i}' to None
#         cre_obj_set_item(cf, i8(1+cf.root_n_args*2 + arg_ind), None)

#         cf.root_arg_infos[arg_ind].type = ARGINFO_CONST
#         cf.root_arg_infos[arg_ind].ptr = 0

    
    # Set 
    # cre_obj_set_item(cf, 1+(i<<1)+1, val)
    

@njit(types.void(GenericCREFuncType,i8,GenericVarType), cache=True)
def set_var_arg(self, i, var):
    # print("I:",i)
    self.is_initialized = False
    # print(i, self.base_to_head_infos[i])
    # head_infos = self.base_to_head_infos[i]
    # has_deref = u2(1) if len(var.deref_infos) > 0 else u2(0)
    head_infos = self.head_infos
    start = self.base_to_head_inds[i]
    end = start + head_infos[start].n_more

    var_ptr = _raw_ptr_from_struct(var)
    for j in range(start,end+1):
        cf = _struct_from_ptr(GenericCREFuncType, head_infos[j].cf_ptr)
        arg_ind = head_infos[j].arg_ind
        # print(i,j, "<<", var_ptr)
        head_infos[j].var_ptr = var_ptr
        head_infos[j].type = ARGINFO_VAR
        # head_infos[j].has_deref = has_deref

        # set 'a{i}' to zero
        # cre_obj_set_item(cf, i8(1+(arg_ind<<1)), 0)
        # set 'a{i}' to zero, set 'h{i}' to var
        # cre_obj_set_item(cf, i8(1+(arg_ind<<1)), var_ptr)
        # set 'ref{i}' to var
        cre_obj_set_item(cf, i8(1+cf.root_n_args*2 + arg_ind), var)

        cf.root_arg_infos[arg_ind].type = ARGINFO_VAR
        # cf.root_arg_infos[arg_ind].has_deref = has_deref
        cf.root_arg_infos[arg_ind].ptr = _raw_ptr_from_struct(var)

@njit(types.void(GenericCREFuncType,i8,GenericCREFuncType), cache=True, debug=False)
def set_op_arg(self, i, val):
    self.is_initialized = False
    # _incref_structref(val)
    # head_infos = self.base_to_head_infos[i]
    head_infos = self.head_infos
    start = self.base_to_head_inds[i]
    end = start + head_infos[start].n_more

    val_ptr = _raw_ptr_from_struct(val)
    for j in range(start,end+1):
        cf = _struct_from_ptr(GenericCREFuncType, head_infos[j].cf_ptr)
        arg_ind = head_infos[j].arg_ind

        head_infos[j].cf_ptr = val_ptr
        head_infos[j].type = ARGINFO_OP_UNEXPANDED
        cre_obj_set_item(cf, i8(1+cf.root_n_args*2 + arg_ind), val)
        # _incref_structref(val)

        # set 'ai' to zero, set 'hi' to val
        # cre_obj_set_item(cf, i8(1+(head_infos[j].arg_ind<<1)), var_ptr)

        cf.root_arg_infos[arg_ind].type = ARGINFO_OP
        # cf.root_arg_infos[arg_ind].has_deref = 0
        cf.root_arg_infos[arg_ind].ptr = val_ptr


# -set_const_arg(i, val) : Primatives
#     -set_var_arg(i, val) : Vars
#     -set_root_func_arg(i, val) : CRE_Funcs
    
#     # For modifying with normal call conventions
#     -set_const_arg(i, val) : Primatives
#     -set_var_arg(i, val) : Vars
#     -set_func_arg(i, val) : CRE_Funcs
    
#     # Update children, base_vars, head_vars
#     -reinitialize()

# @njit(cache=True)
# def _get_base_head_info()

cf_ind_tup_t = Tuple((GenericCREFuncType,i8))
@njit(instr_type[::1](GenericCREFuncType),cache=True)
def build_instr_set(self):
    # NOTE: Since we can't cache recursive functions use a stack to 
    #  do a DF traversal of ops and build the instr_set which defines
    #  the execution order of this op's child ops.

    instrs = List.empty_list(instr_type)
    stack = List.empty_list(cf_ind_tup_t)
    cf = self
    i = 0 
    n_iter = 0
    keep_looping = True
    while(keep_looping):
        
        # any_children = False
        # for i, arg_info in enumerate(cf.root_arg_infos):
        arg_info = cf.root_arg_infos[i]
        # print(": R", _raw_ptr_from_struct(cf), _raw_ptr_from_struct(cf.root_arg_infos), cf.root_arg_infos)
        assert arg_info.ptr != _raw_ptr_from_struct(cf), "CREFunc has self reference"
        # print("RAW!", _raw_ptr_from_struct(cf), i, arg_info.type)
        if(arg_info.type == ARGINFO_OP):
            _, _, head_data_ptr = cre_obj_get_item_t_id_ptr(cf,(1+(i<<1)+1))
            instr = np.zeros(1,dtype=instr_type)[0]
            instr.op_ptr = arg_info.ptr
            instr.return_data_ptr = head_data_ptr

            # Set the size to be the difference between the data_ptrs for 
            #  the return value and the a0 member 
            t_id, m_id, ret_data_ptr = cre_obj_get_item_t_id_ptr(cf,0)
            _, _, first_data_ptr = cre_obj_get_item_t_id_ptr(cf,1)
            instr.size = u4(first_data_ptr-ret_data_ptr)

            if(t_id == T_ID_STR):
                instr.is_ref = u4(1)
            elif(m_id != PRIMITIVE_MBR_ID):
                instr.is_ref = u4(2)

            # print("SIZE-ISREF", instr.size, instr.is_ref, arg_info.ptr)

            instrs.append(instr)
            stack.append((cf, i+1))
            cf = _struct_from_ptr(GenericCREFuncType, arg_info.ptr)
            i = 0 
        else:
            i += 1


        while(i >= len(cf.root_arg_infos)):
            if(len(stack) == 0):
                keep_looping = False
                break
            cf, i = stack.pop(-1)

        n_iter += 1
        if(n_iter >= 128):
            raise RuntimeError("CREFunc max construction depth exceeded.")

    prereq_instrs = np.zeros(len(instrs), dtype=instr_type)
    j = 0
    for i in range(len(instrs)-1,-1,-1): #enumerate(reversed(instrs)):
        if(instrs[i].size > 100): raise RuntimeError("Bad instr")

        prereq_instrs[j] = instrs[i]
        j += 1
    return prereq_instrs
    # self.prereq_instrs = prereq_instrs

i8_arr = i8[::1]
head_info_lst = ListType(head_info_type)
@njit(types.void(GenericCREFuncType),cache=True)
def reinitialize(self):
    if(self.is_initialized): return

    # print("REINIT")
    base_var_map = Dict.empty(i8, head_info_lst)
    for start in self.base_to_head_inds:
        # start = self.base_to_head_inds[i]
        end = start + self.head_infos[start].n_more

        # print("--L", start,end)
        for j in range(start,end+1):
            head_info = self.head_infos[j]
            # print(":", start, j, head_info.type, head_info.var_ptr)
            if(head_info.type == ARGINFO_VAR):
                var = _struct_from_ptr(GenericVarType, head_info.var_ptr)
                base_ptr = var.base_ptr
                if(base_ptr not in base_var_map):
                    base_var_map[base_ptr] = List.empty_list(head_info_type)
                base_var_map[base_ptr].append(head_info)
                # print(base_ptr, base_var_map[base_ptr])

            elif(head_info.type == ARGINFO_OP_UNEXPANDED):
                # print("HAS DEP")
                cf = _struct_from_ptr(GenericCREFuncType, head_info.cf_ptr)
                # print(cf.base_to_head_inds)
                for start_k in cf.base_to_head_inds:
                    # print("--k", k, len(cf.base_to_head_inds))
                    # start_k = cf.base_to_head_inds[k]
                    end_k = start_k + cf.head_infos[start_k].n_more

                    # print("--D", start_k,end_k,len(cf.head_infos))
                    for n  in range(start_k, end_k+1):
                        # print("N", n)
                        head_info_n = cf.head_infos[n]
                        # print("-N", n, head_info_n.type, head_info_n.var_ptr)
                    # for n, head_info_n in enumerate(head_infos_k):
                        var = _struct_from_ptr(GenericVarType, head_info_n.var_ptr)
                        # print("--",var.alias)
                        base_ptr = var.base_ptr
                        if(base_ptr not in base_var_map):
                            base_var_map[base_ptr] = List.empty_list(head_info_type)
                        # print("PRE APPEND ---")
                        base_var_map[base_ptr].append(head_info_n)
                        # print("---")


                        # print("head_infos",head_infos)
                # print("HAS DEP", cf.n_args)
    # print("MID REINIT")
    # print([len(x) for x in base_var_map.values()])
    n_bases = len(base_var_map)
    base_to_head_inds = np.zeros(n_bases, dtype=np.uint32)

    n_heads = 0
    for i, base_head_infos in enumerate(base_var_map.values()):
        base_to_head_inds[i] = n_heads
        n_heads += len(base_head_infos)

    # print("base_to_head_inds", base_to_head_inds)

    head_infos = np.zeros(n_heads,dtype=head_info_type)
    c = 0
    # base_to_head_infos = List.empty_list(head_info_arr_type)
    for i, (base_ptr, base_head_infos) in enumerate(base_var_map.items()):
        for j in range(len(base_head_infos)):
            head_infos[c] = base_head_infos[j]
            # print("n_more", i, c,":",(len(base_head_infos)-1)-j, "/", len(head_infos))
            head_infos[c].n_more = (len(base_head_infos)-1)-j
            c += 1

    # self.base_to_head_infos = base_to_head_infos
    self.n_args = n_bases

    prereq_instrs = build_instr_set(self)
    # print("<<", prereq_instrs)
    rebuild_buffer(self, base_to_head_inds, head_infos, prereq_instrs)
    # print("END REINIT")


    # print("prereq_instrs", prereq_instrs)





        # print(head_infos)


    # self.children = List.empty_list(StructRefType)
    # self.base_vars = List.empty_list(GenericVarType)
    # self.bases_to_arg_ptrs = List.empty_list(i8_arr)
    # self.head_vars = List.empty_list(GenericVarType)

    # print(">>", self.root_arg_infos)
    # Count the number of head vars in the new composition
    # n_head_vars = 0
    # for info in self.root_arg_infos:
    #     if(info.type == ARGINFO_VAR):
    #         n_head_vars += 1
    #     elif(info.type == ARGINFO_OP):
    #         n_head_vars +=len(_struct_from_ptr(GenericCREFuncType, info.ptr).head_vars) 

    # self.arg_data_ptrs = np.empty(n_head_vars, dtype=np.int64)
    # self.arg_head_data_ptrs = np.empty(n_head_vars, dtype=np.int64)
    # # self.arg_op_and_arg_inds = np.empty(n_head_vars, dtype=op_and_arg_ind)

    # c = 0
    # base_var_map = Dict.empty(i8,i8_lst)
    # self_ptr = _raw_ptr_from_struct(self)
    # for i, info in enumerate(self.root_arg_infos):
    #     if(info.type == ARGINFO_VAR):
    #         var = _struct_from_ptr(GenericVarType, info.ptr) 
    #         if(var.base_ptr not in base_var_map):
    #             base_var_map[var.base_ptr] = List.empty_list(i8)

            
    #         self.head_vars.append(var)
    #         # self.arg_data_ptrs[c] = 0 #??

    #         _,_,arg_data_ptr = cre_obj_get_item_t_id_ptr(self, 1+(i<<1))
    #         _,_,head_data_ptr = cre_obj_get_item_t_id_ptr(self, 1+(i<<1)+1)
    #         base_arg_ptrs = base_var_map[var.base_ptr]
    #         base_arg_ptrs.append(arg_data_ptr)
    #         print("data_ptr", 1+(i<<1)+1, head_data_ptr)
    #         self.arg_head_data_ptrs[c] = head_data_ptr #??
    #         # self.arg_op_and_arg_inds[c].op_ptr = self_ptr
    #         # self.arg_op_and_arg_inds[c].arg_ind = i #??
    #         c += 1

    #     elif(info.type == ARGINFO_OP):
    #         cf = _struct_from_ptr(GenericCREFuncType, info.ptr)
    #         # reinitialize(cf)
    #         for bv in cf.base_vars:
    #             base_var_map[bv.base_ptr] = List.empty_list(i8)
    #         for hv in cf.head_vars:
    #             self.head_vars.append(hv)

    #         L = len(cf.arg_head_data_ptrs)
    #         # self.arg_data_ptrs[c:c+L] = cf.arg_data_ptrs[:L]
    #         self.arg_head_data_ptrs[c:c+L] = cf.arg_head_data_ptrs[:L] #??
    #         # self.arg_op_and_arg_inds[c:c+L] = cf.arg_op_and_arg_inds[:L]



    # for b_ptr,base_arg_ptrs in base_var_map.items():
    #     self.base_vars.append(_struct_from_ptr(GenericVarType, b_ptr))
    #     _base_arg_ptrs = np.empty(len(base_arg_ptrs),dtype=np.int64)
    #     for i,v in enumerate(base_arg_ptrs):
    #         _base_arg_ptrs[i] = v
    #     print(_base_arg_ptrs)
    #     self.bases_to_arg_ptrs.append(_base_arg_ptrs)

    # print("children", self.children)
    # self.is_initialized = True
        



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

# @generated_jit(cache=True, nopython=True)
# def set_head_arg_val(self, i, val):
#     val_type = val
#     def impl(self, i, val):
#         data_ptr = self.arg_head_data_ptrs[i]
#         _store_safe(val_type, data_ptr, val)
#     return impl

# @njit(types.void(GenericCREFuncType,i8,unicode_type), cache=True)
# def set_base_arg_str_val(self, i, val):    
#     head_infos = self.base_to_head_infos[i]
#     for head_info in head_infos:
#         # _incref_structref(val)
#         _store_safe(unicode_type, head_info.head_data_ptr, val)
#         # print("HI", head_info, val)
        
from numba.core.typing.typeof import typeof
set_base_arg_val_overloads = {}
def set_base_arg_val_impl(_val):
    # Get the type of _val or if a typeref was given use that
    nb_val_type = typeof(_val) if not isinstance(_val, types.Type) else _val

    # If is a Fact or other object type upcast to CREObjType
    nb_val_type = CREObjType if isinstance(nb_val_type, CREObjTypeClass) else nb_val_type

    # Compile the implementation if it doesn't exist
    if(nb_val_type not in set_base_arg_val_overloads):        
        sig = types.void(GenericCREFuncType, u8, nb_val_type)
        if(nb_val_type is CREObjType):
            # Fact case
            @njit(sig,cache=True)
            def _set_base_arg_val(self, i, val):
                start = self.base_to_head_inds[i]
                end = start + self.head_infos[start].n_more
                for i in range(start,end+1):
                    _store_safe(nb_val_type, self.head_infos[i].arg_data_ptr, val)
        else:
            # Primitive case
            @njit(sig,cache=True)
            def _set_base_arg_val(self, i, val):
                start = self.base_to_head_inds[i]
                end = start + self.head_infos[start].n_more
                for i in range(start,end+1):
                    _store_safe(nb_val_type, self.head_infos[i].head_data_ptr, val)
        set_base_arg_val_overloads[nb_val_type] = _set_base_arg_val
    return set_base_arg_val_overloads[nb_val_type]

# @njit(types.void(GenericCREFuncType), cache=True)
# def do_somethin_self(self):
#     # _incref_structref(self)
#     print("n_args", self.n_args)

# @njit(types.void(unicode_type), cache=True)
# def do_somethin_arg(arg):
#     print("ARG", arg)

# @njit(types.void(i8), cache=True)
# def do_somethin_ind(arg):
#     print("IND", arg)

# @njit(types.void(GenericCREFuncType, i8, unicode_type), cache=True)
# def do_somethin_all(self, i, val):
#     print("somethin")
#     head_infos = self.base_to_head_infos[i]
#     for head_info in head_infos:
#         _store_safe(unicode_type, head_info.head_data_ptr, val)

# @njit(cache=True)
# def set_base_arg_val(self, i, val):
#     print("set_base_arg_val")
#     head_infos = self.base_to_head_infos[i]
#     for head_info in head_infos:
#         _store_safe(unicode_type, head_info.head_data_ptr, val)

# @generated_jit(cache=True, nopython=True)
# def set_base_arg_val(self, i, val):
#     val_type = val
#     # print(">", self, val_type)
#     if(isinstance(val_type,StructRef)):
#         def impl(self, i, val):
#             # print("IS OBJ")
#             head_infos = self.base_to_head_infos[i]
#             for head_info in head_infos:
#                 _store_safe(CREObjType, head_info.arg_data_ptr, val)
#     else:
#         def impl(self, i, val):
#             print("NOT OBJ")
#             head_infos = self.base_to_head_infos[i]
#             for head_info in head_infos:
#                 _store_safe(val_type, head_info.head_data_ptr, val)


    # return impl


# @njit(cache=True)
# def update_heads(self, i, val):
#     for 

# @njit(cache=True)
# def update_children(self, i, val):
#     pass





# @njit(types.void(GenericCREFuncType), cache=True)
# def blargle(self):
#     print("self ret ptr" , self.return_data_ptr)
#     for instr in self.prereq_instrs:
#         cf = _struct_from_ptr(GenericCREFuncType, instr.op_ptr)
#         _func_from_address(call_self_f_type, cf.call_self_addr)(cf)
#         # print("<<",_load_ptr(i8, cf.return_data_ptr))
#         # print(instr.return_data_ptr, "<-", cf.return_data_ptr)
#         # print("<<",_load_ptr(i8, instr.return_data_ptr))
#         if(instr.is_ref==1):
#             new_obj = _load_ptr(unicode_type, cf.return_data_ptr)   
#             _incref_structref(new_obj)
#         elif(instr.is_ref==2):
#             new_obj_ptr = _load_ptr(i8, cf.return_data_ptr)   
#             _incref_ptr(new_obj_ptr)
        
#         # print("MEMCPY", instr.return_data_ptr, "<-", cf.return_data_ptr)
#         _memcpy(instr.return_data_ptr, cf.return_data_ptr, instr.size)
#         # print("<<",_load_ptr(i8, instr.return_data_ptr))   
#     # for child in self.children:
#     #     cre_func_exec_heads(_cast_structref(GenericCREFuncType, child))
#     # print("BEF")
#     _func_from_address(call_self_f_type, self.call_self_addr)(self)

call_self_f_type = types.FunctionType(types.void(GenericCREFuncType))
@njit(types.void(GenericCREFuncType), locals={"i":u4}, cache=True, inline='always')
def cre_func_call_self(self):
    # print("prereq_instrs" , self.prereq_instrs)
    for i in range(len(self.prereq_instrs)):
    # for instr in self.prereq_instrs:
        instr = self.prereq_instrs[i]
        # print("A")
        cf = _struct_from_ptr(GenericCREFuncType, instr.op_ptr)
        # print("B")
        _func_from_address(call_self_f_type, cf.call_self_addr)(cf)
        # print("C")
        # print("<<",_load_ptr(i8, cf.return_data_ptr))
        # print(instr.return_data_ptr, "<-", cf.return_data_ptr)
        # print("<<",_load_ptr(i8, instr.return_data_ptr))
        if(instr.is_ref==1):
            new_obj = _load_ptr(unicode_type, cf.return_data_ptr)   
            _incref_structref(new_obj)
        elif(instr.is_ref==2):
            new_obj_ptr = _load_ptr(i8, cf.return_data_ptr)   
            _incref_ptr(new_obj_ptr)
        
        # print("MEMCPY", instr.return_data_ptr, "<-", cf.return_data_ptr)
        _memcpy(instr.return_data_ptr, cf.return_data_ptr, instr.size)
        # print("AFT")
        # print("<<",_load_ptr(i8, instr.return_data_ptr))   
    # for child in self.children:
    #     cre_func_exec_heads(_cast_structref(GenericCREFuncType, child))
    # print("BEF")
    _func_from_address(call_self_f_type, self.call_self_addr)(self)

    # for instr in self.prereq_instrs:
    #     if(instr.is_ref==1):
    #         new_obj = _load_ptr(unicode_type, cf.return_data_ptr)   
    #         _decref_structref(new_obj)
    #     elif(instr.is_ref==2):
    #         new_obj_ptr = _load_ptr(i8, cf.return_data_ptr)   
    #         _decref_ptr(new_obj_ptr)
    #     if(instr.is_ref):

    # print("AFT")
    # print("<<",_load_ptr(i8, self.return_data_ptr))

from numba.core.typing.typeof import typeof
get_str_return_val_overloads = {}
def get_str_return_val_impl(_val):
    nb_val_type = typeof(_val) if not isinstance(_val, types.Type) else _val
    if(nb_val_type not in get_str_return_val_overloads):
        @njit(nb_val_type(GenericCREFuncType),cache=True, inline='always')
        def _get_str_return_val(self):
            return _load_ptr(nb_val_type, self.return_data_ptr)
        get_str_return_val_overloads[nb_val_type] = _get_str_return_val
    return get_str_return_val_overloads[nb_val_type]


# @njit(unicode_type(GenericCREFuncType),cache=True)
# def get_str_return_val(self):
#     # print("< return_val", self, return_type)
#     # def impl(self, return_type):
#         # print(self.return_val)
#         # print("??", self.return_data_ptr, ":", _load_ptr(return_type, self.return_data_ptr))
#     return _load_ptr(unicode_type, self.return_data_ptr)
    # return impl



# @generated_jit(cache=True)
# def get_return_val(self, return_type):
#     # print("< return_val", self, return_type)
#     def impl(self, return_type):
#         # print(self.return_val)
#         # print("??", self.return_data_ptr, ":", _load_ptr(return_type, self.return_data_ptr))
#         return _load_ptr(return_type, self.return_data_ptr)
#     return impl

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


#-----------------------------------------------------------------
# : CREFunc Overload __call__
@overload_method(CREFuncTypeClass, '__call__')#, strict=False)
def overload_call(cf, *args):
    # print(cf.__dict__.keys())
    # print("overload_call:", cf, args)
    # print(cf.name)
    cf_type = cf
    if(not hasattr(cf_type, 'return_type')):
        raise ValueError("Cannot call CREFunc without return_type")
    return_type = cf_type.return_type
    if(cf_type.is_composed):
        if(len(args) > 0 and isinstance(args[0],types.BaseTuple)):
            # print("isinstance tup")
            args = tuple(x for x in args[0])

        # print("args", args)
        set_base_impls = tuple(set_base_arg_val_impl(a) for a in args)
        # print(set_base_impls)
        set_base = set_base_impls[0]
        ret_impl = get_str_return_val_impl(cf_type.return_type)
        range_inds = tuple(u8(i) for i in range(len(args)))
        def impl(cf,*args):
            for i in literal_unroll(range_inds):
                set_base(cf, i, args[i])
            cre_func_call_self(cf)
            return ret_impl(cf)

    elif(hasattr(cf_type,'dispatchers') and 'call' in cf_type.dispatchers):
        call = cf_type.dispatchers['call']
        # print(call)
        def impl(cf, *args):
            return call(*args)
            # return args[0] + args[1]
            # return i8(a + b)
    else:
        fn_type = types.FunctionType(cf_type.signature)
        def impl(cf, *args):
            f = _func_from_address(fn_type, cf.call_heads_addr)
            return f(*args)

            
    # print(impl)
    return impl


#-----------------------------------------------------------------
# : cre_func_str()

# cf_ind_lst_tup_t = Tuple((GenericCREFuncType,i8,ListType(unicode_type)))
@njit(unicode_type(GenericCREFuncType, types.boolean), cache=True)
def cre_func_str(self, use_shorthand):
    stack = List()
    cf = self
    i = 0 
    arg_strs = List.empty_list(unicode_type)
    s = ""
    keep_looping = True
    while(keep_looping):
        arg_info = cf.root_arg_infos[i]
        # print("arg_info", i, arg_info)
        if(arg_info.type == ARGINFO_OP):
            stack.append((cf, i+1,arg_strs))
            # print("op ptr", arg_info.ptr)
            cf = _struct_from_ptr(GenericCREFuncType, arg_info.ptr)
            arg_strs = List.empty_list(unicode_type)
            i = 0 
        else:
            if(arg_info.type == ARGINFO_VAR):
                # print("var ptr", arg_info.ptr)
                var = _struct_from_ptr(GenericVarType, arg_info.ptr)
                arg_strs.append(str(var)) 
            i += 1

        while(i >= len(cf.root_arg_infos)):
            nd = cf.name_data
            tmp = nd.shorthand_template if use_shorthand else nd.expr_template
            s = tmp.format(arg_strs)
            if(len(stack) == 0):
                keep_looping = False
                break
            if(use_shorthand): s = f"({s})"

            cf, i, arg_strs = stack.pop(-1)
            arg_strs.append(s)
    return s

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
    # print(name, long_hash)
    if(not source_in_cache(name, long_hash)): #or members.get('cache',True) == False):
        source = gen_cre_func_source(*gen_src_args, long_hash)
        source_to_cache(name, long_hash, source)

    # Update members with jitted 'call' and 'check'
    to_import = ['cf_type', 'call'] + (['check'] if(has_check) else [])
    l = import_from_cached(name, long_hash, to_import)

    typ = l['cf_type']
    # typ.name = f'{name}_{long_hash}'
    typ.func_name = name
    # print("??", typ.name)
    typ.call = l['call']
    if(has_check): typ.check = l['check']
    
    for k,v in members.items():
        setattr(typ,k,v)

    return typ









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
from cre.utils import _struct_from_ptr, _func_from_address, _load_ptr, _obj_cast_codegen, _store_safe, _cast_structref, _attr_is_null, _nullify_attr, _decref_structref
from cre.cre_func import CREFunc_method, CREFunc_assign_method_addr, CREFuncTypeClass, GenericCREFuncType, ARGINFO_VAR, GenericVarType
from cre.memset import resolve_deref_data_ptr
from cre.fact import BaseFact
import cloudpickle


return_type = cloudpickle.loads({cloudpickle.dumps(return_type)})
arg_types = cloudpickle.loads({cloudpickle.dumps(arg_types)})
cf_type = CREFuncTypeClass(return_type, arg_types,is_composed=False, name={name!r}, long_hash={long_hash!r})

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

@CREFunc_method(cf_type, 'call_self', void(GenericCREFuncType))
def call_self(_self):
    self = _cast_structref(cf_type, _self)
    {"".join([f""" 
    #if(arg_infos[{i}].has_deref):
    if(not _attr_is_null(self,'a{i}')):
        #var = _struct_from_ptr(GenericVarType, self.root_arg_infos[{i}].ptr)
        var = _cast_structref(GenericVarType, self.ref{i})
        data_ptr = resolve_deref_data_ptr(_cast_structref(BaseFact, self.a{i}), var.deref_infos)
        _nullify_attr(self,'a{i}')
        self.h{i} = _load_ptr(h{i}_type, data_ptr)
""" for i in range(len(arg_types))])
    }
    return_val = call({", ".join([f'self.h{i}' for i in range(len(arg_types))])})
    # print("WW", _load_ptr(i8,self.return_data_ptr))
    _store_safe(return_type, self.return_data_ptr, return_val)
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
    return {f'1' if isinstance(return_type, StructRef) else f'1 if(call({arg_names})) else 0'}

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
    -set_const_arg(i, val) : Primatives
    -set_var_arg(i, val) : Vars
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
