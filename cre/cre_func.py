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
from cre.cre_object import CREObjType, cre_obj_field_dict, CREObjTypeClass, CREObjProxy, member_info_type, set_chr_mbrs, cre_obj_get_item_t_id_ptr, cre_obj_set_item, cre_obj_get_item, PRIMITIVE_MBR_ID
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



def all_args_are_const(arg_types):
    '''Helper function for checking if a set of arg_types are all primative types'''
    all_const = True
    for at in arg_types:
        if(isinstance(at, VarTypeClass) or isinstance(at, CREObjTypeClass)):
            all_const = False
            break
    return all_const

# ----------------------------------------------------------------------
# : Record Array Struct definitions

# Struct that indicates a span of 'head_infos' that are all 
#  associated with the same variable
_head_range_type = np.dtype([
    ('start', np.uint16), # Start of span
    ('end', np.uint16) # Start + length
])
head_range_type = numba.from_dtype(_head_range_type)


ARGINFO_CONST = u1(0)
ARGINFO_VAR = u1(1)
ARGINFO_OP = u1(2)
ARGINFO_OP_UNEXPANDED = u1(3)

# Struct for each argument to a CREFunc which might be assigned 
#  to a primative constant, Var, or other CREFunc.
_arg_infos_type = np.dtype([
    # Enum for CONST, VAR, OP, OP_UNEXPANDED.
    ('type', np.int32), 
    # Whether is Var w/ dereference (e.g. V.nxt.nxt).
    ('has_deref', np.int32), 
    # Ptr to Var or CREFUnc.
    ('ptr', np.int64) 
]) 
arg_infos_type = numba.from_dtype(_arg_infos_type)

# Struct w/ instructions for executing a child CREFunc that needs to be 
#  run before the main root CREFunc in a composition of CREFuncs
_instr_type = np.dtype([
    # The ptr to the CREFunc
    ('cf_ptr', np.int64), 
    # Data ptr to copy the CREFunc's return value into.
    ('return_data_ptr', np.int64), 
    # The byte-width of the return value  
    ('size',np.uint32), 
    # 1 for unicode_type 2 for other. Needed because unicode_strings 
    #  don't have meminfos as first member.
    ('ref_kind',np.uint32)
])
instr_type = numba.from_dtype(_instr_type)


# Struct for head_infos 
_head_info_type = np.dtype([
    ('cf_ptr', np.int64),
    ('type', np.uint8),
    ('has_deref', np.uint8),
    ('n_more', np.uint16),
    ('arg_ind', np.uint16),
    ('t_id', np.uint16),
    ('var_ptr', np.int64),
    ('arg_data_ptr', np.int64),
    ('head_data_ptr', np.int64)])
head_info_type = numba.from_dtype(_head_info_type)



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
    '''A decorator used in CREFunc codegens for trying to compile member
         methods like 'call_heads' and registering their ptrs as globals.
    '''
    def wrapper(func):
        # Try to compile the CREFunc
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
    "repr_const_addrs" : i8[::1]#ListType(types.FunctionType(unicode_type(CREObjType, i8)))
    #i8[::1]
})

cre_func_fields_dict = {
    **cre_obj_field_dict, 

    # Keeps track of name and various ways of printing the CREFunc
    "name_data" : NameDataType,

    # The number of arguments taken by this cf.
    "n_args" : i8,

    # The number of arguments taken by the original call() of this cf.
    "root_n_args" : i8,

    # The args to the root op 
    "root_arg_infos" : arg_infos_type[::1],

    # Maps base arguments to particular head_infos
    "head_ranges" : head_range_type[::1],

    # Keeps references to each head, their types, the CREFunc they live in 
    #  among other things
    "head_infos" : head_info_type[::1],

    # A set of instructions for calling CREFuncs that need to be called as
    #  prerequisites to this one, as part of a function composition.
    "prereq_instrs" : instr_type[::1],

    # Data ptr of the return value    
    "return_data_ptr" : i8,
    
    # "call_addr" : i8,
    "call_self_addr" : i8,
    "call_heads_addr" : i8,
    "call_head_ptrs_addr" : i8,
    # "match_addr" : i8,
    "match_heads_addr" : i8,
    "match_head_ptrs_addr" : i8,
    # "check_addr" : i8,

    # True if the op has beed initialized
    "is_initialized" : types.boolean,

    # True if this op is a ptr op
    "is_ptr_op" : types.boolean,

    # True if dereferencing the head args succeeded  
    "heads_derefs_succeeded" : types.boolean,

    # True if check in this and all children suceeded
    "exec_passed_checks" : types.boolean,
    
    "return_t_id" : u2,

    # Placeholders to keep 64-bit aligned, might not be necessary
    "padding0": u1,
    "padding1": u1,
    
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

        if(return_type is None or arg_types is None):
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
        if(all_args_are_const(args)):
            return self.return_type(*args)
        else:
            ty = CREFuncTypeClass(return_type=self.return_type,name="FuncComp")
            return ty(*args)

    def get_call_signatures(self):
        return [self.return_type(*self.arg_types),
                GenericCREFuncType(*([GenericCREFuncType]*len(self.arg_types))),
                GenericCREFuncType(*([GenericVarType]*len(self.arg_types)))
                ]

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
        # Need to delete any functions members to avoid loading undefined functions
        #  when the type is unpickled
        d = self.__dict__.copy()
        if('call' in d): del d['call']
        if('check' in d): del d['check']
        if('ctor' in d): del d['ctor']
        return d


GenericCREFuncType = CREFuncTypeClass()

@lower_cast(CREFuncTypeClass, GenericCREFuncType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)

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
            # print(">>", members)

            name = call_func.__name__
            arg_names = inspect.getfullargspec(call_func)[0]

            cf_type = define_CREFunc(call_func.__name__, members)
            n_args = len(members['arg_types'])

            # Make sure that constant members can be repr'ed
            # for arg_type in cf_type.arg_types:
            #     ensure_repr_const(arg_type)

            expr_template = f"{name}({', '.join([f'{{{i}}}' for i in range(n_args)])})"
            shorthand_template = members.get('shorthand',expr_template)
            # print("shorthand_template:", shorthand_template)

            # with PrintElapse("new"):
            cre_func = cf_type.ctor(
                # n_args,
                name,
                expr_template,
                shorthand_template                
            )

            cre_func._return_type = members['return_type']
            cre_func._arg_types = members['arg_types']
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

    @property
    def return_type(self):
        self._ensure_has_types()        
        return self._return_type

    @property
    def arg_types(self):
        self._ensure_has_types()        
        return self._arg_types

    @property
    def signature(self):
        self._ensure_has_types()        
        return self._return_type(*self._arg_types)

    def _ensure_has_types(self):
        if(not hasattr(self, '_return_type')):
            if(hasattr(self, '_type')):
                self._return_type = self._type.return_type
                if(self._type.arg_types is None):
                    self.recover_reinit_arg_types()
                else:
                    self._arg_types = self._type.arg_types

    def __call__(self,*args):
        if(len(args) != self.n_args):
            raise ValueError(f"Got {len(args)} args for ?? with {self.n_args} positional arguments.")
        # print("CALL", args)
        # print("_type", self._type)
        self._ensure_has_types()
        all_const = True
        for arg in args:
            # print("----", arg)
            if(isinstance(arg, Var) or isinstance(arg, CREFunc)):
                all_const = False

        if(all_const):
            
            # print("ALL CONST")
            # Assign each argument to it's slot h{i} in the op's memory space
            for i, arg in enumerate(args):
                if(isinstance(arg,CREObjProxy)):
                    impl = set_base_arg_val_impl(CREObjType)
                else:
                    impl = set_base_arg_val_impl(self._arg_types[i])
                impl(self, i, arg)

            cre_func_call_self(self)
            args = None
            ret_impl = get_str_return_val_impl(self._return_type)
            return ret_impl(self)
        else:
            # print("START COMPOSE OP", args)
            new_cf = cre_func_copy_generic(self)
            # print(">>", new_cf)
            new_cf._return_type = self._return_type
            new_cf._arg_types = self._arg_types
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
            self.recover_reinit_arg_types()


            new_cf._type = CREFuncTypeClass(self._return_type, self._arg_types, True)
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
        impl = set_const_arg_impl(arg)
        impl(new_cf, i, arg)

    @property
    @njit(i8(GenericCREFuncType), cache=True)
    def n_args(self):
        return self.n_args

    @property
    @njit(head_range_type[::1](GenericCREFuncType), cache=True)
    def head_ranges(self):
        return self.head_ranges

    @property
    @njit(head_info_type[::1](GenericCREFuncType), cache=True)
    def head_infos(self):
        return self.head_infos

    def recover_reinit_arg_types(self):
        context = cre_context()
        arg_types = []
        head_ranges = self.head_ranges
        head_infos = self.head_infos
        for hrng in head_ranges:
            start, end = hrng['start'], hrng['end']
            for j in range(start, end):
                hi = head_infos[j]
                t_id = hi['t_id']
                arg_types.append(context.get_type(t_id=t_id))
        self._arg_types = arg_types

    @property    
    def base_var_ptrs(self):
        return get_base_var_ptrs(self)  

    @property    
    def head_var_ptrs(self):
        return get_head_var_ptrs(self)    

define_boxing(CREFuncTypeClass, CREFunc)

@njit(i8[::1](GenericCREFuncType), cache=True)
def get_base_var_ptrs(self):
    base_var_ptrs = np.empty(self.n_args,dtype=np.int64)
    for i, hrng in enumerate(self.head_ranges):
        hi = self.head_infos[hrng.start]
        v = _struct_from_ptr(GenericVarType, hi.var_ptr)
        base_var_ptrs[i] = v.base_ptr
    return base_var_ptrs

@overload_attribute(CREFuncTypeClass, "base_var_ptrs")
def overload_base_var_ptrs(self):
    return get_base_var_ptrs

@njit(i8[::1](GenericCREFuncType), cache=True)
def get_head_var_ptrs(self):
    head_var_ptrs = np.empty(self.n_args,dtype=np.int64)
    for i, hrng in enumerate(self.head_ranges):
        for j in range(hrng.start,hrng.end):
            hi = self.head_infos[j]
            head_var_ptrs[i] = hi.var_ptr
    return head_var_ptrs

@overload_attribute(CREFuncTypeClass, "head_var_ptrs")
def overload_head_var_ptrs(self):
    return get_head_var_ptrs



# ------------------------------
# : CREFunc initialization

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
        cf.call_head_ptrs_addr = _get_global_fn_addr(method_names[1])
        # cf.call_addr = _get_global_fn_addr(method_names[2])
        cf.match_heads_addr = _get_global_fn_addr(method_names[3])
        cf.match_head_ptrs_addr = _get_global_fn_addr(method_names[4])
        # cf.match_addr = _get_global_fn_addr(method_names[5])
        # cf.check_addr = _get_global_fn_addr(method_names[6])
        cf.call_self_addr = _get_global_fn_addr(method_names[7])
    return impl

#################Garbage###################3
SIZEOF_NRT_MEMINFO = 48
@njit(types.void(i8),cache=True)
def cf_del_inject(data_ptr):
    cf = _struct_from_ptr(GenericCREFuncType, data_ptr-SIZEOF_NRT_MEMINFO)

cf_del_inject_addr = _get_wrapper_address(cf_del_inject, types.void(i8))
ll.add_symbol("CRE_cf_del_inject", cf_del_inject_addr)

##############################################


@njit(#types.void(
    # GenericCREFuncType, 
    # types
    # types.optional(u4[::1]),
    # types.optional(head_info_type[::1]),
    # types.optional(instr_type[::1])),
     cache=True)
def rebuild_buffer(self, head_ranges=None, head_infos=None, prereq_instrs=None):
    ''' 'head_ranges', 'head_infos', and 'prereq_instrs' are held in a single
         buffer to ensure data contiguity. This function rebuilds that buffer.
    '''

    n_args = self.n_args if(head_ranges is None) else len(head_ranges)
    n_heads = self.n_args if(head_infos is None) else len(head_infos)
    n_prereqs = 0 if prereq_instrs is None else len(prereq_instrs)

    # Layout: head_ranges | head_infos | prereq_instrs
    L_ = L = n_args + 10*n_heads
    L += 6 * n_prereqs

    buff = np.zeros(L, dtype=np.uint32)

    if(head_ranges is not None):
        # print("A", n_args, len(head_ranges[:].view(np.uint32)))
        buff[:n_args] = head_ranges[:].view(np.uint32)
    if(head_infos is not None):
        # print("B", L_-n_args, len(head_infos[:].view(np.uint32)))
        buff[n_args: L_] = head_infos[:].view(np.uint32)
    if(prereq_instrs is not None):
        # print("C", L-L_, len(prereq_instrs[:].view(np.uint32)), "n_prereqs", n_prereqs)
        buff[L_: L] = prereq_instrs[:].view(np.uint32)

    # print("BEF")
    self.head_ranges = buff[:n_args].view(_head_range_type)
    self.head_infos = buff[n_args: L_].view(_head_info_type)
    self.prereq_instrs = buff[L_:L].view(_instr_type)
    # print("END")


cached_repr_const_symbols = {}
def ensure_repr_const(typ):
    # print("A")
    # with PrintElapse("ensure"):
    if(isinstance(typ, Fact)):
        typ = None
    if(typ not in cached_repr_const_symbols):
        sig = unicode_type(CREObjType, i8)
        if(typ is None):
            # TODO: For now str isn't overloaded for fact types so ignore this
            @njit(sig, cache=True)
            def repr_const(cf, i):
                # Conditional prevents return value from being None
                if(i > 0):
                    raise NotImplementedError("str not implemented for fact types")  
                return ""

        #TODO SHOULD PROPERLY IMPLEMENT REPR For int, float, unicode_type
        elif(typ == unicode_type):
            @njit(sig, cache=True)
            def repr_const(cf, i):
                v = cre_obj_get_item(cf, typ, i8(1+(i<<1)+1))
                return f"'{str(v)}'"
        else:
            @njit(sig, cache=True)
            def repr_const(cf, i):
                v = cre_obj_get_item(cf, typ, i8(1+(i<<1)+1))
                return str(v)
                
        addr = _get_wrapper_address(repr_const, sig)
        name = f"CREFunc_repr_const_{repr(typ)}"
        ll.add_symbol(name, addr)
        cached_repr_const_symbols[typ] = name
    return cached_repr_const_symbols[typ]

head_info_arr_type = head_info_type[::1]

@generated_jit(cache=True, nopython=True)
def cre_func_new(_cf_type, name, expr_template, shorthand_template):
    # SentryLiteralArgs(['cf_type','n_args']).for_function(cre_func_new).bind(cf_type, n_args, name, expr_template, shorthand_template)
    cf_type = _cf_type.instance_type
    fd = cf_type._field_dict
    arg_types = cf_type.arg_types
    n_args = len(arg_types)
    # n_args = n_args.literal_value
    chr_mbr_attrs = ["return_val"]
    for i in range(n_args):
        chr_mbr_attrs.append(f'a{i}')
        chr_mbr_attrs.append(f'h{i}')
    for i in range(n_args):
        chr_mbr_attrs.append(f'ref{i}')
    chr_mbr_attrs = tuple(chr_mbr_attrs)
    

    rc_fn_typ = types.FunctionType(unicode_type(CREObjType, i8))
    repr_const_symbols = tuple(ensure_repr_const(at) for  at in arg_types)
    unroll_inds = tuple(range(len(arg_types)))


    prefix = cf_type.symbol_prefix
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

    # print("repr_const_symbols", repr_const_symbols)
    # repr_const_impl_ptrs = tuple(_get_wrapper_address(fn, rc_fn_typ)
                                 # for fn in repr_const_impls)
    # print("repr_const_impl_ptrs", repr_const_impl_ptrs)
    # @njit(GenericCREFuncType(unicode_type,unicode_type,unicode_type), cache=True)
    def impl(_cf_type, name, expr_template, shorthand_template):
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

        # Init: head_ranges | head_infos | prereq_instrs
        rebuild_buffer(_cast_structref(GenericCREFuncType,cf))

        head_infos = cf.head_infos
        for i in range(n_args):
            head_infos[i].cf_ptr = self_ptr
            head_infos[i].arg_ind = u4(i)
            head_infos[i].type = ARGINFO_VAR
            head_infos[i].has_deref = 0
            head_infos[i].n_more = 0
            _,_,arg_data_ptr = cre_obj_get_item_t_id_ptr(cf, 1+(i<<1))
            t_id,_,head_data_ptr = cre_obj_get_item_t_id_ptr(cf, 1+(i<<1)+1)
            head_infos[i].t_id = t_id
            head_infos[i].arg_data_ptr = arg_data_ptr
            head_infos[i].head_data_ptr = head_data_ptr
            cf.head_ranges[i].start = i
            cf.head_ranges[i].end = i+1

        # cre_func_assign_method_table(cf)
        cf.call_heads_addr = _get_global_fn_addr(method_names[0])
        # cf.call_head_ptrs_addr = _get_global_fn_addr(method_names[1])
        # cf.call_addr = _get_global_fn_addr(method_names[2])
        # cf.match_heads_addr = _get_global_fn_addr(method_names[3])
        # cf.match_head_ptrs_addr = _get_global_fn_addr(method_names[4])
        # cf.match_addr = _get_global_fn_addr(method_names[5])
        # # cf.check_addr = _get_global_fn_addr(method_names[6])
        cf.call_self_addr = _get_global_fn_addr(method_names[7])

        repr_const_addrs = np.zeros(n_args, dtype=np.int64)
        # for i in range(n_args):
        for i in literal_unroll(unroll_inds):
            repr_const_addrs[i] = _get_global_fn_addr(repr_const_symbols[i])
            # fn = _func_from_address(rc_fn_typ, repr_const_addr)
            # repr_consts.append(fn)

        cf.name_data = NameData(name, expr_template, shorthand_template, repr_const_addrs)        
        
        casted = _cast_structref(GenericCREFuncType,cf)
        return casted 
    return impl

from cre.cre_object import copy_cre_obj
@njit(cache=True)
def cre_func_copy(cf):
    # Make a copy of the CreFunc via a memcpy
    cpy = copy_cre_obj(cf)

    # Find the the byte offset between the op and its copy
    cf_ptr = _raw_ptr_from_struct(cf)
    cpy_ptr = _raw_ptr_from_struct(cpy)
    cpy_delta = cpy_ptr-cf_ptr

    # Make a copy of base_to_head_infos
    # base_to_head_infos = List.empty_list(head_info_arr_type)
    old_head_ranges = cf.head_ranges.copy()
    old_head_infos = cf.head_infos.copy()
    old_prereq_instrs = cf.prereq_instrs.copy()

    # Nullify these attributes since we don't want the pointers from the 
    #  original to get decref'ed on assignment
    # _nullify_attr(cpy, 'base_to_head_infos')
    _nullify_attr(cpy, 'head_ranges')
    _nullify_attr(cpy, 'head_infos')
    _nullify_attr(cpy, 'root_arg_infos')
    _nullify_attr(cpy, 'name_data')
    _nullify_attr(cpy, 'prereq_instrs')

    cpy_generic = _cast_structref(GenericCREFuncType,cpy)
    rebuild_buffer(cpy_generic, old_head_ranges, old_head_infos, old_prereq_instrs)

    # Make the arg_data_ptr and head_data_ptr point to the copy
    for i, head_info in enumerate(cf.head_infos):
        cpy.head_infos[i] = head_info
        if(head_info.cf_ptr == cf_ptr):
            cpy.head_infos[i].cf_ptr = cpy_ptr
            cpy.head_infos[i].arg_data_ptr = head_info.arg_data_ptr+cpy_delta
            cpy.head_infos[i].head_data_ptr = head_info.head_data_ptr+cpy_delta


    cpy.return_data_ptr = cf.return_data_ptr+cpy_delta

    # NOTE: Some quirky optimization is probably causing root_arg_infos 
    #  to be allocated on the stack because some cf copies seem to be sharing
    #  root_arg_infos instances even though they are independantly instantiated
    #  making a copy() on instantiation seems to fix it.
    cpy.root_arg_infos = np.zeros(cf.root_n_args, dtype=arg_infos_type).copy()
    cpy.name_data = cf.name_data
    cpy.is_initialized = False

    # print("::", f"{cf_ptr}:{_raw_ptr_from_struct(cf.root_arg_infos)}","|", f"{cpy_ptr}:{_raw_ptr_from_struct(cpy.root_arg_infos)}")
    # print("&&>",cf_ptr,cf.return_data_ptr,cf.return_data_ptr-cf_ptr)
    # print("&&<",cpy_ptr,cpy.return_data_ptr,cpy.return_data_ptr-cpy_ptr)

    return cpy

@njit(GenericCREFuncType(GenericCREFuncType), cache=True)
def cre_func_copy_generic(cf):
    return cre_func_copy(_cast_structref(GenericCREFuncType,cf))

#--------------------------------------------------------------------
# Construction Functions


from numba.core.typing.typeof import typeof
set_const_arg_overloads = {}
def set_const_arg_impl(_val):
    nb_val_type = typeof(_val) if not isinstance(_val, types.Type) else _val
    if(nb_val_type not in set_const_arg_overloads):
        
        sig = types.void(GenericCREFuncType, i8, nb_val_type)
        @njit(sig,cache=True)
        def _set_const_arg(self, i, val):
            self.is_initialized = False

            head_infos = self.head_infos
            start = self.head_ranges[i].start
            end = self.head_ranges[i].end
            for j in range(start,end):
                cf = _struct_from_ptr(GenericCREFuncType, head_infos[j].cf_ptr)
                arg_ind = head_infos[j].arg_ind

                head_infos[j].type = ARGINFO_CONST
                head_infos[j].has_deref = 0
                # set 'h{i}' to val
                cre_obj_set_item(cf, i8(1+(arg_ind<<1)+1), val)
                # set 'ref{i}' to None
                cre_obj_set_item(cf, i8(1+cf.root_n_args*2 + arg_ind), None)

                cf.root_arg_infos[arg_ind].type = ARGINFO_CONST
                cf.root_arg_infos[arg_ind].has_deref = 0
                cf.root_arg_infos[arg_ind].ptr = 0

        set_const_arg_overloads[nb_val_type] = _set_const_arg
    return set_const_arg_overloads[nb_val_type]


@generated_jit(cache=True)
def set_const_arg(self, i, val):
    impl = set_const_arg_impl(val)
    return impl

@njit(types.void(GenericCREFuncType,i8,GenericVarType), cache=True)
def set_var_arg(self, i, var):
    self.is_initialized = False
    head_infos = self.head_infos
    start = self.head_ranges[i].start
    end = self.head_ranges[i].end

    var_ptr = _raw_ptr_from_struct(var)
    hd = u1(len(var.deref_infos) > 0)
    for j in range(start,end):
        cf = _struct_from_ptr(GenericCREFuncType, head_infos[j].cf_ptr)
        arg_ind = head_infos[j].arg_ind
        head_infos[j].var_ptr = var_ptr
        head_infos[j].has_deref = u1(hd)
        head_infos[j].type = ARGINFO_VAR

        # set 'a{i}' to zero
        # cre_obj_set_item(cf, i8(1+(arg_ind<<1)), 0)
        # set 'a{i}' to zero, set 'h{i}' to var
        # cre_obj_set_item(cf, i8(1+(arg_ind<<1)), var_ptr)
        # set 'ref{i}' to var
        cre_obj_set_item(cf, i8(1+cf.root_n_args*2 + arg_ind), var)

        cf.root_arg_infos[arg_ind].type = ARGINFO_VAR
        cf.root_arg_infos[arg_ind].has_deref = hd
        cf.root_arg_infos[arg_ind].ptr = _raw_ptr_from_struct(var)

@njit(types.void(GenericCREFuncType,i8,GenericCREFuncType), cache=True, debug=False)
def set_op_arg(self, i, op):
    self.is_initialized = False
    # _incref_structref(val)
    # head_infos = self.base_to_head_infos[i]
    head_infos = self.head_infos
    start = self.head_ranges[i].start
    end = self.head_ranges[i].end

    op_ptr = _raw_ptr_from_struct(op)
    for j in range(start,end):
        cf = _struct_from_ptr(GenericCREFuncType, head_infos[j].cf_ptr)
        arg_ind = head_infos[j].arg_ind

        head_infos[j].cf_ptr = op_ptr
        head_infos[j].has_deref = 0
        head_infos[j].type = ARGINFO_OP_UNEXPANDED
        cre_obj_set_item(cf, i8(1+cf.root_n_args*2 + arg_ind), op)

        # set 'ai' to zero, set 'hi' to val
        # cre_obj_set_item(cf, i8(1+(head_infos[j].arg_ind<<1)), var_ptr)

        cf.root_arg_infos[arg_ind].type = ARGINFO_OP
        cf.root_arg_infos[arg_ind].has_deref = 0
        cf.root_arg_infos[arg_ind].ptr = op_ptr




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
            instr.cf_ptr = arg_info.ptr
            instr.return_data_ptr = head_data_ptr

            # Set the size to be the difference between the data_ptrs for 
            #  the return value and the a0 member 
            t_id, m_id, ret_data_ptr = cre_obj_get_item_t_id_ptr(cf,0)
            _, _, first_data_ptr = cre_obj_get_item_t_id_ptr(cf,1)
            instr.size = u4(first_data_ptr-ret_data_ptr)

            if(t_id == T_ID_STR):
                instr.ref_kind = u4(1)
            elif(m_id != PRIMITIVE_MBR_ID):
                instr.ref_kind = u4(2)

            # print("SIZE-ISREF", instr.size, instr.ref_kind, arg_info.ptr)

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
    for hrng in self.head_ranges:
        # print("--L", start,end)
        for j in range(hrng.start,hrng.end):
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
                for hrng_k in cf.head_ranges:
                    # print("--D", start_k,end_k,len(cf.head_infos))
                    for n  in range(hrng_k.start, hrng_k.end):
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
    n_bases = len(base_var_map)
    head_ranges = np.zeros(n_bases, dtype=head_range_type)

    n_heads = 0
    for i, base_head_infos in enumerate(base_var_map.values()):
        head_ranges[i].start = n_heads
        head_ranges[i].end = n_heads+len(base_head_infos)
        n_heads += len(base_head_infos)

    head_infos = np.zeros(n_heads,dtype=head_info_type)
    c = 0
    for i, (base_ptr, base_head_infos) in enumerate(base_var_map.items()):
        for j in range(len(base_head_infos)):
            head_infos[c] = base_head_infos[j]
            # print("n_more", i, c,":",(len(base_head_infos)-1)-j, "/", len(head_infos))
            head_infos[c].n_more = (len(base_head_infos)-1)-j
            c += 1

    self.n_args = n_bases

    prereq_instrs = build_instr_set(self)
    rebuild_buffer(self, head_ranges, head_infos, prereq_instrs)
  


#--------------------------------------------------------------------
# Execution Functions
        
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
                start = self.head_ranges[i].start
                end = self.head_ranges[i].end
                for j in range(start,end):
                    if(self.head_infos[j].has_deref):
                        _store_safe(nb_val_type, self.head_infos[j].arg_data_ptr, val)
                    else:
                        _store_safe(nb_val_type, self.head_infos[j].head_data_ptr, val)
        else:
            # Primitive case
            @njit(sig,cache=True)
            def _set_base_arg_val(self, i, val):
                start = self.head_ranges[i].start
                end = self.head_ranges[i].end
                for j in range(start,end):
                    _store_safe(nb_val_type, self.head_infos[j].head_data_ptr, val)
        set_base_arg_val_overloads[nb_val_type] = _set_base_arg_val
    return set_base_arg_val_overloads[nb_val_type]


call_self_f_type = types.FunctionType(types.void(GenericCREFuncType))
@njit(types.void(GenericCREFuncType), locals={"i":u4}, cache=True, inline='always')
def cre_func_call_self(self):
    for i in range(len(self.prereq_instrs)):
        instr = self.prereq_instrs[i]
        cf = _struct_from_ptr(GenericCREFuncType, instr.cf_ptr)
        _func_from_address(call_self_f_type, cf.call_self_addr)(cf)

        if(instr.ref_kind==1):
            new_obj = _load_ptr(unicode_type, cf.return_data_ptr)   
            _incref_structref(new_obj)
        elif(instr.ref_kind==2):
            new_obj_ptr = _load_ptr(i8, cf.return_data_ptr)   
            _incref_ptr(new_obj_ptr)
        
        _memcpy(instr.return_data_ptr, cf.return_data_ptr, instr.size)
    
    _func_from_address(call_self_f_type, self.call_self_addr)(self)
    
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


# def cleanup_base_args(self):
#     for head_info in self.head_infos:
#         if(head_info.has_deref):
#             _,_,arg_data_ptr = cre_obj_get_item_t_id_ptr(cf, 1+(i<<1))
#             a = _load_ptr(CREObjType, arg_data_ptr)
#             _decref_structref(a) 
#             _store_safe(arg_data_ptr)


def overload_compose_codegen(cf_type, arg_types):
    ind_kinds = []
    for i, at in enumerate(arg_types):
        if(isinstance(at, VarTypeClass)):
            ind_kinds.append((i,ARGINFO_VAR))
        elif(isinstance(at,CREFuncTypeClass)):
            ind_kinds.append((i,ARGINFO_OP))
        else:
            ind_kinds.append((i,ARGINFO_CONST))

    most_precise_cf_type = CREFuncTypeClass(return_type=cf_type.return_type, name="FuncComp")
    ind_kinds = tuple(ind_kinds)
    def impl(cf, *args):
        _cf = cre_func_copy(cf)
        for tup in literal_unroll(ind_kinds):
            i, kind = tup
            arg = args[i]
            if(kind==ARGINFO_VAR):
                set_var_arg(_cf, i, _cast_structref(GenericVarType,arg))
            elif(kind==ARGINFO_OP):
                # NOTE: Copying the OP here causes issue, but might be necessary
                # f = cre_func_copy(_cast_structref(GenericCREFuncType,arg))
                set_op_arg(_cf, i, _cast_structref(GenericCREFuncType,arg))
            elif(kind==ARGINFO_CONST):
                set_const_arg(_cf, i, arg)
            
            reinitialize(_cf)
        return _cast_structref(most_precise_cf_type, _cf)
    return impl


def overload_call_codegen(cf_type, arg_types):
    if(cf_type.is_composed):
        if(len(arg_types) > 0 and isinstance(arg_types[0],types.BaseTuple)):
            arg_types = tuple(x for x in arg_types[0])

        set_base_impls = tuple(set_base_arg_val_impl(a) for a in arg_types)
        set_base = set_base_impls[0]
        ret_impl = get_str_return_val_impl(cf_type.return_type)
        range_inds = tuple(u8(i) for i in range(len(arg_types)))
        def impl(cf,*args):
            for i in literal_unroll(range_inds):
                set_base(cf, i, args[i])
            cre_func_call_self(cf)
            return ret_impl(cf)

    elif(hasattr(cf_type,'dispatchers') and 'call' in cf_type.dispatchers):
        call = cf_type.dispatchers['call']
        def impl(cf, *args):
            return call(*args)
    else:
        fn_type = types.FunctionType(cf_type.signature)
        def impl(cf, *args):
            f = _func_from_address(fn_type, cf.call_heads_addr)
            return f(*args)
    # return cf_type.signature, impl
    return impl


#-----------------------------------------------------------------
# : CREFunc Overload __call__
@overload_method(CREFuncTypeClass, '__call__')#, strict=False)
def overload_call(cf, *args):
    cf_type = cf
    if(not hasattr(cf_type, 'return_type')):
        raise ValueError("Cannot call CREFunc without return_type")

    if(len(args) > 0 and isinstance(args[0], types.BaseTuple)):
        args = tuple(*args)

    all_const = all_args_are_const(args)
    if(all_const):
        return overload_call_codegen(cf, args)
    else:
        return overload_compose_codegen(cf, args)
            
    return impl


#-----------------------------------------------------------------
# : cre_func_str()

rc_fn_typ = types.FunctionType(unicode_type(CREObjType, i8))

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
        if(arg_info.type == ARGINFO_OP):
            stack.append((cf, i+1,arg_strs))
            cf = _struct_from_ptr(GenericCREFuncType, arg_info.ptr)
            arg_strs = List.empty_list(unicode_type)
            i = 0 
        else:
            if(arg_info.type == ARGINFO_VAR):
                var = _struct_from_ptr(GenericVarType, arg_info.ptr)
                arg_strs.append(str(var)) 
            elif(arg_info.type == ARGINFO_CONST):
                addr = cf.name_data.repr_const_addrs[i]
                fn = _func_from_address(rc_fn_typ, addr)
                arg_strs.append(fn(cf,i))
                # s = cf.name_data.repr_consts[i](cf,i)
                
                # arg_strs.append("?")
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

    # print(gen_src_args)
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
from numba.types import unicode_type
from numba.extending import lower_cast
from numba.experimental.function_type import _get_wrapper_address
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.utils import PrintElapse, _struct_from_ptr, _func_from_address, _load_ptr, _obj_cast_codegen, _store_safe, _cast_structref, _attr_is_null, _nullify_attr, _decref_structref
from cre.cre_func import ensure_repr_const, cre_func_new, CREFunc_method, CREFunc_assign_method_addr, CREFuncTypeClass, GenericCREFuncType, ARGINFO_VAR, GenericVarType
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
    if(self.root_arg_infos[{i}].has_deref):
        var = _cast_structref(GenericVarType, self.ref{i})
        a = _cast_structref(BaseFact, self.a{i})
        data_ptr = resolve_deref_data_ptr(a, var.deref_infos)
        self.h{i} = _load_ptr(h{i}_type, data_ptr)
""" for i in range(len(arg_types))])
    }
    return_val = call_heads({", ".join([f'self.h{i}' for i in range(len(arg_types))])})
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

    source +='''

# Make sure that constant members can be repr'ed
for at in arg_types:
    ensure_repr_const(at)

# ctor = gen_cre_func_ctor(cf_type)

@njit(GenericCREFuncType(unicode_type, unicode_type, unicode_type), cache=True)
def ctor(name, expr_template, shorthand_template):
    return cre_func_new(cf_type, name, expr_template, shorthand_template)
cf_type.ctor = ctor
'''

    return source
