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
from cre.caching import gen_import_str, unique_hash_v, import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template, StructRefType
from cre.utils import (cast, _sizeof_type, _nullify_attr, new_w_del, _memcpy, _func_from_address, decode_idrec, lower_getattr,  
                       _decref_ptr, _incref_ptr, _incref_structref, _decref_structref, _ptr_from_struct_incref, ptr_t, _load_ptr,
                       _obj_cast_codegen)
from cre.utils import PrintElapse, encode_idrec, assign_to_alias_in_parent_frame, as_typed_list, lower_setattr, _store, _store_safe, _tuple_getitem
from cre.vector import VectorType
from cre.fact import Fact, gen_fact_import_str, get_offsets_from_member_types
from cre.var import Var, var_memcopy, VarType, VarTypeClass
from cre.obj import CREObjType, cre_obj_field_dict, CREObjTypeClass, CREObjProxy, member_info_type, set_chr_mbrs, cre_obj_get_item_t_id_ptr, cre_obj_set_item, cre_obj_get_item, PRIMITIVE_MBR_ID
from cre.core import T_ID_OP, T_ID_STR, register_global_default
from cre.make_source import make_source, gen_def_func, gen_assign, gen_if, gen_not, resolve_template, gen_def_class
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models, register_default
from numba.experimental.function_type import _get_wrapper_address

# Import to ensure that .format is defined
import cre.type_conv


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

def dispatcher_sig_from_arg_types(disp, arg_types):
    cres = disp.overloads.get(arg_types,None)
    if(cres is not None):
        sig = cres.signature
    else:
        # Note: get_call_template is an internal method of numba.Dispatcher 
        (template,*rest) = disp.get_call_template(arg_types,{})
        #Note: this can be finicky might need to find the best one
        sig = [x for x in template.cases if x.args==arg_types][0]
    return sig

# ----------------------------------------------------------------------
# : Record Array Struct definitions

# Struct that indicates a span of 'head_infos' that are all 
#  associated with the same variable
_head_range_type = np.dtype([
    ('start', np.uint16), # Start of span
    ('end', np.uint16) # Start + length
])
head_range_type = numba.from_dtype(_head_range_type)

# ------------------------------
# ARGINFO type enums
ARGINFO_CONST = u1(1)
ARGINFO_VAR = u1(2)
ARGINFO_OP = u1(3)
ARGINFO_OP_UNEXPANDED = u1(4)

# ------------------------------
# STATUS enums
CFSTATUS_FALSEY = u1(0)
CFSTATUS_TRUTHY = u1(1)
CFSTATUS_NULL_DEREF = u1(2)
CFSTATUS_ERROR = u1(3)

# ------------------------------
# REFKIND enums
REFKIND_PRIMATIVE = u4(1)
REFKIND_UNICODE = u4(2)
REFKIND_STRUCTREF = u4(3)

# Struct for each argument to a CREFunc which might be assigned 
#  to a primative constant, Var, or other CREFunc.
_arg_infos_type = np.dtype([
    # Enum for CONST, VAR, OP, OP_UNEXPANDED.
    ('type', np.uint16), 
    # t_id of the argument
    ('t_id', np.uint16), 
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
    #  don't have meminfo ptrs as first member.
    ('ref_kind',np.uint32)
])
instr_type = numba.from_dtype(_instr_type)


# Struct for head_infos 
_head_info_type = np.dtype([
    # The pointer to the CREFunc for which this is a head argument.
    ('cf_ptr', np.int64),
    # Enum for CONST, VAR, OP, OP_UNEXPANDED.
    ('type', np.uint8),
    # If the Var associated with this has a dereference chain.
    ('has_deref', np.uint8),
    # The index of the root argument this head corresponds to.
    ('arg_ind', np.uint16),
    # t_id for the base value's type.
    ('base_t_id', np.uint16),
    # t_id for the head value's type.
    ('head_t_id', np.uint16),
    # The pointer to the Var for this head. Its ref is held in "ref{i}" member.
    ('var_ptr', np.int64),
    # Data pointer for the a{i} member for the base.
    ('arg_data_ptr', np.int64),
    # Data pointer for the h{i} member for the resolved head.
    ('head_data_ptr', np.int64),
    # 1 for unicode_type 2 for other. Needed because unicode_strings don't 
    #  have meminfo ptrs as first member so need different inc/decref implmentation.
    ('ref_kind', np.uint32),
    # The byte-width of the head value.
    ('head_size', np.uint32)])
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

def CREFunc_method(cf_type, sig, _fn_name=None, on_error='error',**kwargs):
    '''A decorator used in CREFunc codegens for trying to compile member
         methods like 'call_heads' and registering their ptrs as globals.
    '''
    def wrapper(func):        
        if(isinstance(func,Dispatcher)):
            dispatcher = func
            fn_name = func.py_func.__name__ if _fn_name is None else _fn_name 
        else:
            fn_name = func.__name__ if _fn_name is None else _fn_name 
            # Try to compile the CREFunc
            try:
                dispatcher = njit(sig, cache=True, **kwargs)(func)
            except NumbaError as e:
                if(on_error == "error"): raise e
                if(on_error == "warn"):
                    warn_cant_compile(fn_name, cf_type.func_name, e)
                return None
        addr = _get_wrapper_address(dispatcher, sig)
        CREFunc_assign_method_addr(cf_type, fn_name, addr)
        # dispatcher.cre_method_addr = addr
        if(not hasattr(cf_type,'dispatchers')):
            cf_type.dispatchers = {}    
        cf_type.dispatchers[fn_name] = dispatcher
        return dispatcher
    return wrapper

# ----------------------------------------------------------------------
# : CREFunc jitted side

NameData, NameDataType = define_structref("NameData", {
    "name" : unicode_type,
    "expr_template" : unicode_type,
    "shorthand_template" : unicode_type,
    "repr_const_addrs" : i8[::1],
})

cre_func_fields_dict = {
    **cre_obj_field_dict, 

    # Keeps track of name and various ways of printing the CREFunc
    "name_data" : NameDataType,

    # The number of arguments taken by this cf.
    "n_args" : i8,

    # The number of arguments taken by the original call() of this cf.
    "n_root_args" : i8,

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
        
    # The address for the root CREFunc's 'call_head' implementation.
    #  Essentially the CREFunc's main call function.     
    "call_heads_addr" : i8,

    # The address for the root CREFunc's 'call_self' implementation.
    #  Calls call_heads on any values stored in 'h{i}' (i.e. 'head') slots.
    "call_self_addr" : i8,
    
    # The address for the root CREFunc's 'resolve_heads_addr' implementation.
    #  Dereferences any objects in 'a{i}' (i.e. 'arg') slots and writes them
    #  do each corresponding 'h{i}' (i.e. 'head') slot.
    "resolve_heads_addr" : i8,

    # True if the op has beed initialized
    "is_initialized" : types.boolean,

    # True if this op is a ptr op
    "is_ptr_op" : types.boolean,

    # NOTE: Not Used 
    # True if dereferencing the head args succeeded  
    # "heads_derefs_succeeded" : types.boolean,

    # The composition depth
    "depth" : types.uint16,

    # NOTE: Not Used 
    # True if check in this and all children suceeded
    # "exec_passed_checks" : types.boolean,
    
    "return_t_id" : u2,

    "has_any_derefs": u1,

    "is_composed" : u1,

    # Other members like a0,a1,... h0,h1,... etc. filled in on specialize
}

@structref.register
class CREFuncTypeClass(types.Callable, CREObjTypeClass):
    '''The numba type class for CREFuncs. '''
    t_id = T_ID_OP
    type_cache = {}

    def __new__(cls, return_type=None, arg_types=None, is_composed=False, name=None, long_hash=None):
        if(name is None): name = "CREFuncType"
        if(isinstance(arg_types,list)): arg_types = tuple(arg_types)

        # Check if this type is already cached 
        unq_tup = (return_type, arg_types, is_composed, name, long_hash)
        self = cls.type_cache.get(unq_tup,None)
        if(self is not None):
            return self

        self = super().__new__(cls)

        # If full signature isn't specified then use unspecialized struture 
        if(return_type is None or arg_types is None):
            field_dict = {**cre_func_fields_dict}

        # Otherwise define type-specific slots for 
        #    a{i}: Slot for base CREObject like arguments that will be  
        #            dereferenced into heads (i.e. obj.nxt.nxt.val)
        #    h{i}: Slot for type type-specific head arguments
        #    ref{i} : Slot for Var or Op instances that are part of composition
        #    return_val: Slot for return value.
        #    chr_mbrs_infos: Characteristic Member Infos -- see CREObject
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
            }

        self.func_name = name
        self.return_type = return_type
        self.arg_types = arg_types
        self.is_composed = is_composed
        self.t_id = T_ID_OP
        self.long_hash = long_hash
        self._field_dict = field_dict
        cls.type_cache[unq_tup] = self

        # Apply typical content of __init__() here to avoid reinitialization   
        types.StructRef.__init__(self,[(k,v) for k,v in field_dict.items()])
        self.name = repr(self)
        return self

    def __init__(self,*args,**kwargs):
        pass

    # -------------------------------------------
    # Impl these 3 funcs to subclass types.Callable so can @overload_method('__call__')
    def get_call_type(self, context, args, kws):
        if(all_args_are_const(args)):
            return self.return_type(*args)
        else:
            ty = CREFuncTypeClass(return_type=self.return_type,name="FuncComp")
            return ty(*args)

    def get_call_signatures(self):
        return [self.return_type(*self.arg_types),
                CREFuncType(*([CREFuncType]*len(self.arg_types))),
                CREFuncType(*([VarType]*len(self.arg_types)))
                ]

    def get_impl_key(self, sig):
        return (type(self), '__call__')
    # -------------------------------------------

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
        else:
            return f"{self.func_name}"

    @property
    def symbol_prefix(self):
        if(not hasattr(self, "_symbol_prefix")):
            if(self.func_name == "CREFuncType"):
                self._symbol_preix = "CREFuncType"
            else:
                shortened_hash = self.long_hash[:18]
                self._symbol_preix = f"CREFunc_{self.func_name}_{shortened_hash}"
        return self._symbol_preix

    def __getstate__(self):
        # Need to delete any functions members to avoid loading undefined functions
        #  when the type is unpickled. Otherwise "FunctionType not have '_key'" error.
        d = self.__dict__.copy()
        if('call_heads' in d): del d['call_heads']
        if('check' in d): del d['check']
        if('ctor' in d): del d['ctor']
        if('_code' in d): del d['_code']
        if('dispatchers' in d): del d['dispatchers']
        return d


CREFuncType = CREFuncTypeClass()
register_global_default("Op", CREFuncType)

@lower_cast(CREFuncTypeClass, CREFuncType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)

# -----------------------------------------------------------------------
# : CREFunc Proxy Class

def new_cre_func(name, members):
    ''' Returns a singleton instance for a CREFunc definition.'''
    call_func = members['call']

    cf_type = define_CREFunc(name, members)
    if(not hasattr(members, 'arg_types')): 
        members['arg_types'] = members['signature'].args
    if(not hasattr(members, 'return_type')): 
        members['return_type'] = members['signature'].return_type
    n_args = len(members['arg_types'])

    expr_template = f"{name}({', '.join([f'{{{i}}}' for i in range(n_args)])})"
    shorthand_template = members.get('shorthand',expr_template)

    cre_func = cf_type.ctor(
        name,
        expr_template,
        shorthand_template                
    )

    cre_func._return_type = members['return_type']
    cre_func._arg_types = members['arg_types']
    _vars = []
    get_return_val_impl(cre_func.return_type)
    for i, (arg_alias, typ) in enumerate(zip(cf_type.root_arg_names, cf_type.arg_types)):
        v = Var(typ,arg_alias); _vars.append(v);
        set_var_arg(cre_func, i, v)
        set_base_arg_val_impl(typ)
        set_const_arg_impl(typ)
    reinitialize(cre_func)

    cre_func._type = cf_type

    # This prevent _vars from freeing until after reinitialize
    _vars = None 
    return cre_func

def _standardize_py_args(self, py_args):
    '''Helper function that: 
        1) Determines if all arguments are constants.
        2) Replaces type instances with Var instances.
        3) Resolves the head types of the arguments.
    '''
    head_types = [] 
    py_args = list(py_args)
    all_const = True
    for i, x in enumerate(py_args):
        # If given a type make into a Var
        if(isinstance(x, types.Type)):
            x = py_args[i] = Var(x, self.root_arg_names[i])
        if(isinstance(x, Var)):
            head_types.append(x.head_type)
            all_const = False
        elif(isinstance(x, CREFunc)):
            head_types.append(x.return_type)
            all_const = False
        elif(isinstance(x, UntypedCREFunc)):
            raise ValueError("Cannot compose UntypedCREFunc.")
        else:
            head_types.append(resolve_return_type(x))
    head_types = tuple(head_types)
    return py_args, all_const, head_types


class CREFunc(StructRefProxy):

    def __new__(self,*args, **kwargs):
        ''' A decorator function that builds a new Op'''
        if(len(args) > 1): raise ValueError("CREFunc() takes at most one position argument 'signature'.")
        
        def wrapper(call_func):
            # Make new cre_func_type
            members = kwargs
            members['call'] = call_func
            members['root_arg_names'] = inspect.getfullargspec(call_func)[0]
            name = call_func.__name__

            # If no signature is given then create an untyped instance that can be specialized later
            if('signature' not in members):
                return UntypedCREFunc(name,members)

            # Otherwise define a CREFunc with 'name' and 'members' and return
            #  a new singleton instance of it.
            return new_cre_func(name, members)            

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
            raise ValueError(f"Got {len(args)} args for {str(self)} with {self.n_args} positional arguments.")
        self._ensure_has_types()
        arg, all_const, head_type = _standardize_py_args(self, args)
        
        # Call Case, will return a value
        if(all_const):            
            # Assign each argument to it's slot h{i} in the op's memory space
            for i, arg in enumerate(args):
                if(isinstance(arg, CREObjProxy)):
                    impl = set_base_arg_val_impl(CREObjType)
                else:
                    impl = set_base_arg_val_impl(self._arg_types[i])
                impl(self, i, arg)

            
            status = cre_func_resolve_call_self(self)

            if(status == CFSTATUS_NULL_DEREF):
                raise Exception(f"{str(self)} failed to execute because of a dereferencing error.")
            elif(status == CFSTATUS_ERROR):
                raise Exception(f"{str(self)} failed to execute because of an internal error.")

            args = None
            get_ret = get_return_val_impl(self._return_type)
            return get_ret(self)

        # Compose Case, will return a new CREFunc
        else:
            new_cf = cf_deep_copy_ep(self)
            # new_cf = self
            new_cf._return_type = self.return_type
            for i, arg in enumerate(args):
                # Optmization: Assign arguments via entry_points
                #  to skip numba's internal type checking.  
                if(isinstance(arg, Var)):
                    set_var_arg_ep(new_cf, i, arg)
                elif(isinstance(arg, CREFunc)):
                    set_op_arg_ep(new_cf, i, arg)
                else:
                    impl = set_const_arg_impl(arg, use_ep=True)
                    impl(new_cf, i, arg)

            reinitialize(new_cf)

            # NOTE: Slowest step because of cre_context()
            new_cf.recover_arg_types()

            name, long_hash = None, None
            if(hasattr(self,'_type')):
                func_name = self._type.func_name
                long_hash = self._type.long_hash
            new_cf._type = CREFuncTypeClass(self._return_type, self._arg_types,
                                is_composed=new_cf.is_composed,
                                name=func_name, long_hash=long_hash)
            args = None
            return new_cf

    def __str__(self):
        return cre_func_str(self, True)

    def __repr__(self):
        return cre_func_str(self, True)

    @property
    def return_type(self):
        self._ensure_has_types()        
        return self._return_type

    @property
    @njit(cache=True)
    def return_t_id(self):
        return self.return_t_id

    @property
    def arg_types(self):
        self._ensure_has_types()        
        return self._arg_types

    @property
    def root_arg_names(self):
        return self._type.root_arg_names

    @property
    def signature(self):
        self._ensure_has_types()        
        return self._return_type(*self._arg_types)

    @property
    def func_name(self):
        return self._type.func_name

    @property
    def long_hash(self):
        return self._type.long_hash

    @property
    def right_commutes(self):
        return getattr(self._type,'right_commutes',{})

    @property
    def commutes(self):
        return self._type.commutes

    def _ensure_has_types(self):
        '''Ensures that the CREFunc has a _retur_type and _arg_types 
            which it might be missing if instantied in jitted code.
        '''
        if(getattr(self, '_return_type',None) is None):
            return_type = getattr(self, '_type', CREFuncType).return_type
            if(return_type is not None):
                self._return_type = return_type
                if(self._type.arg_types is None):
                    self.recover_arg_types()
                else:
                    self._arg_types = self._type.arg_types
            else:
                self.recover_return_type()
                self.recover_arg_types()

    def set_var_arg(self, i, val):
        set_var_arg(self, i, val)

    def set_op_arg(self, i, val):
        set_op_arg(self, i, val)

    def set_const_arg(self, i, val):
        impl = set_const_arg_impl(arg)
        impl(new_cf, i, arg)

    @property
    @njit(i8(CREFuncType), cache=True)
    def n_args(self):
        return self.n_args

    @property
    @njit(i8(CREFuncType), cache=True)
    def n_funcs(self):
        return len(self.prereq_instrs)+1

    @property
    @njit(head_range_type[::1](CREFuncType), cache=True)
    def head_ranges(self):
        return self.head_ranges

    @property
    @njit(head_info_type[::1](CREFuncType), cache=True)
    def head_infos(self):
        return self.head_infos

    @property
    @njit(u1(CREFuncType), cache=True)
    def is_composed(self):
        return self.is_composed

    @property
    @njit(i8(CREFuncType), cache=True)
    def depth(self):
        return i8(self.depth)


    def recover_return_type(self):
        self._return_type = cre_context().get_type(t_id=self.return_t_id, ensure_retro=False)

    def recover_arg_types(self):
        context = cre_context()
        arg_types = []
        for t_id in get_base_t_ids_ep(self):
            arg_types.append(context.t_id_to_type[t_id])

        self._arg_types = arg_types

    @property    
    def base_var_ptrs(self):
        return get_base_var_ptrs(self)  

    @property    
    def base_t_ids(self):
        return get_base_t_ids_ep(self)  

    @property    
    def head_var_ptrs(self):
        return get_head_var_ptrs(self)    

    def __lt__(self, other): 
        from cre.default_funcs import LessThan
        return LessThan(self, other)
    def __le__(self, other): 
        from cre.default_funcs import LessThanEq
        return LessThanEq(self, other)
    def __gt__(self, other): 
        from cre.default_funcs import GreaterThan
        return GreaterThan(self, other)
    def __ge__(self, other):
        from cre.default_funcs import GreaterThanEq
        return GreaterThanEq(self, other)
    def __eq__(self, other): 
        from cre.default_funcs import Equals
        return Equals(self, other)
    def __ne__(self, other): 
        from cre.default_funcs import Equals
        return ~Equals(self, other)

    def __add__(self, other):
        from cre.default_funcs import Add
        return Add(self, other)

    def __radd__(self, other):
        from cre.default_funcs import Add
        return Add(other, self)

    def __sub__(self, other):
        from cre.default_funcs import Subtract
        return Subtract(self, other)

    def __rsub__(self, other):
        from cre.default_funcs import Subtract
        return Subtract(other, self)

    def __mul__(self, other):
        from cre.default_funcs import Multiply
        return Multiply(self, other)

    def __rmul__(self, other):
        from cre.default_funcs import Multiply
        return Multiply(other, self)

    def __truediv__(self, other):
        from cre.default_funcs import Divide
        return Divide(self, other)

    def __rtruediv__(self, other):
        from cre.default_funcs import Divide
        return Divide(other, self)

    def __floordiv__(self, other):
        from cre.default_funcs import FloorDivide
        return FloorDivide(self, other)

    def __rfloordiv__(self, other):
        from cre.default_funcs import FloorDivide
        return FloorDivide(other, self)

    def __pow__(self, other):
        from cre.default_funcs import Power
        return Power(self, other)

    def __rpow__(self, other):
        from cre.default_funcs import Power
        return Power(other, self)

    def __mod__(self, other):
        from cre.default_funcs import Modulus
        return Modulus(other, self)

    def __and__(self,other):
        from cre.conditions import cre_func_to_cond, conditions_and
        self = cre_func_to_cond(self)
        if(isinstance(other, CREFunc)): other = cre_func_to_cond(other)
        # print("<<", type(self), type(other))
        return conditions_and(self, other)

    def __or__(self,other):
        from cre.conditions import cre_func_to_cond, conditions_or
        self = cre_func_to_cond(self)
        if(isinstance(other, CREFunc)): other = cre_func_to_cond(other)
        return conditions_or(self, other)

    def __invert__(self):
        from cre.conditions import literal_ctor, literal_to_cond, literal_not
        return literal_to_cond(literal_not(literal_ctor(self)))


define_boxing(CREFuncTypeClass, CREFunc)

@njit(i8[::1](CREFuncType), cache=True)
def get_base_var_ptrs(self):
    base_var_ptrs = np.empty(self.n_args,dtype=np.int64)
    for i, hrng in enumerate(self.head_ranges):
        hi = self.head_infos[hrng.start]
        v = cast(hi.var_ptr, VarType)
        base_var_ptrs[i] = v.base_ptr
    return base_var_ptrs

@overload_attribute(CREFuncTypeClass, "base_var_ptrs")
def overload_base_var_ptrs(self):
    return get_base_var_ptrs.py_func

@njit(u2[::1](CREFuncType), cache=True)
def get_base_t_ids(self):
    t_ids = np.empty(self.n_args,dtype=np.uint16)
    for i, hrng in enumerate(self.head_ranges):
        hi = self.head_infos[hrng.start]
        t_ids[i] = hi.base_t_id
    return t_ids

get_base_t_ids_ep = get_base_t_ids.overloads[(CREFuncType,)].entry_point

@overload_attribute(CREFuncTypeClass, "base_t_ids")
def overload_base_t_ids(self):
    return get_base_t_ids.py_func

@njit(i8[::1](CREFuncType), cache=True)
def get_head_var_ptrs(self):
    head_var_ptrs = np.empty(self.n_args,dtype=np.int64)
    for i, hrng in enumerate(self.head_ranges):
        for j in range(hrng.start,hrng.end):
            hi = self.head_infos[j]
            head_var_ptrs[i] = hi.var_ptr
    return head_var_ptrs

@overload_attribute(CREFuncTypeClass, "head_var_ptrs")
def overload_head_var_ptrs(self):
    return get_head_var_ptrs.py_func

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

#################Garbage###################3
SIZEOF_NRT_MEMINFO = 48
@njit(types.void(i8),cache=True)
def cf_del_inject(data_ptr):
    cf = cast(data_ptr-SIZEOF_NRT_MEMINFO, CREFuncType)

cf_del_inject_addr = _get_wrapper_address(cf_del_inject, types.void(i8))
ll.add_symbol("CRE_cf_del_inject", cf_del_inject_addr)

##############################################


head_info_size = head_info_type.dtype.itemsize
instr_type_size = instr_type.dtype.itemsize

@njit(#types.void(
    # CREFuncType, 
    # types.optional(head_range_type[::1]),
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
    L_ = L = n_args + (head_info_size>>2)*n_heads
    L += (instr_type_size>>2) * n_prereqs

    buff = np.zeros(L, dtype=np.uint32)
    if(head_ranges is not None):
        buff[:n_args] = head_ranges[:].view(np.uint32)
    if(head_infos is not None):
        buff[n_args: L_] = head_infos[:].view(np.uint32)
    if(prereq_instrs is not None):
        buff[L_: L] = prereq_instrs[:].view(np.uint32)

    self.head_ranges = buff[:n_args].view(_head_range_type)
    self.head_infos = buff[n_args: L_].view(_head_info_type)
    self.prereq_instrs = buff[L_:L].view(_instr_type)


cached_repr_const_symbols = {}
def ensure_repr_const(typ):
    ''' Implements various jitted repr() functions for various kinds
        of constants that could be part of a CREFunc composition and
        writes their implementations to global variables. These are
        used to str() or repr() CREFuncs.
    '''
    if(isinstance(typ, (Fact,CREObjTypeClass))):
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

@generated_jit(nopython=True)
def cre_func_ctor(_cf_type, name, expr_template, shorthand_template, is_ptr_op):
    cf_type = _cf_type.instance_type
    fd = cf_type._field_dict
    arg_types = cf_type.arg_types
    n_args = len(arg_types)
    chr_mbr_attrs = ["return_val"]
    for i in range(n_args):
        # NOTE: order of a{i} then h{i} matters.
        chr_mbr_attrs.append(f'a{i}')
        chr_mbr_attrs.append(f'h{i}')
    for i in range(n_args):
        chr_mbr_attrs.append(f'ref{i}')
    chr_mbr_attrs = tuple(chr_mbr_attrs)
    
    repr_const_symbols = tuple(ensure_repr_const(at) for  at in arg_types)    
    unroll_inds = tuple(range(len(arg_types)))

    prefix = cf_type.symbol_prefix
    method_names = (
        f"{prefix}_call_heads",
        f"{prefix}_call_self",
        f"{prefix}_resolve_heads",
    )
    # return_t_id = cre_context().get_t_id(cf_type.return_type)

    def impl(_cf_type, name, expr_template, shorthand_template, is_ptr_op):
        cf = new(cf_type)
        cf.idrec = encode_idrec(T_ID_OP,0,0xFF)
        cf.n_args = n_args
        cf.n_root_args = n_args
        cf.root_arg_infos = np.zeros(n_args, dtype=arg_infos_type)
        cf.is_initialized = False
        cf.is_ptr_op = is_ptr_op
        set_chr_mbrs(cf, chr_mbr_attrs)

        # Point the return_data_ptr to 'return_val'
        ret_t_id,_,return_data_ptr = cre_obj_get_item_t_id_ptr(cf, 0)

        if(ret_t_id == T_ID_STR):
            #Note: Bad things happen if unicode string slots left NULL
            cre_obj_set_item(cf,0,"")

        cf.return_data_ptr = return_data_ptr
        cf.return_t_id = ret_t_id

        self_ptr = cast(cf, i8)

        # Init: head_ranges | head_infos | prereq_instrs
        rebuild_buffer(cast(cf,CREFuncType))

        # Fill head_infos for a newly instantiated CREFunc
        head_infos = cf.head_infos
        for i in range(n_args):
            hi = head_infos[i]
            hi.cf_ptr = self_ptr
            hi.arg_ind = u4(i)
            hi.type = ARGINFO_VAR
            hi.has_deref = 0
            _,_,arg_data_ptr = cre_obj_get_item_t_id_ptr(cf, 1+(i<<1))
            t_id, m_id, head_data_ptr = cre_obj_get_item_t_id_ptr(cf, 1+(i<<1)+1)
            hi.base_t_id = t_id
            hi.head_t_id = t_id
            hi.arg_data_ptr = arg_data_ptr
            hi.head_data_ptr = head_data_ptr

            _,_,elm_aft_ptr = cre_obj_get_item_t_id_ptr(cf, 1+(i<<1)+2)
            hi.head_size = u4(elm_aft_ptr-head_data_ptr)

            if(t_id == T_ID_STR):
                hi.ref_kind = REFKIND_UNICODE
                #Note: Bad things happen if unicode string slots left NULL
                cre_obj_set_item(cf,1+(i<<1)+1,"")
            elif(m_id != PRIMITIVE_MBR_ID):
                hi.ref_kind = REFKIND_STRUCTREF
            else:
                hi.ref_kind = REFKIND_PRIMATIVE

            cf.head_ranges[i].start = i
            cf.head_ranges[i].end = i+1

        # Load addresses of methods from globals 
        if(not is_ptr_op):
            cf.call_heads_addr = _get_global_fn_addr(method_names[0])
            cf.call_self_addr = _get_global_fn_addr(method_names[1])
            cf.resolve_heads_addr = _get_global_fn_addr(method_names[2])        

        repr_const_addrs = np.zeros(n_args, dtype=np.int64)
        for i in literal_unroll(unroll_inds):
            repr_const_addrs[i] = _get_global_fn_addr(repr_const_symbols[i])

        cf.name_data = NameData(name, expr_template, shorthand_template, repr_const_addrs)        
        cf.is_composed = False
        cf.depth = 1
        
        casted = cast(cf, CREFuncType)
        return casted 
    return impl

# --------------------------------------------------------------
# : Copy

from cre.obj import copy_cre_obj
@njit(cache=False)
def cre_func_copy(cf):
    # Make a copy of the CreFunc via a memcpy (plus incref obj members)
    cpy = copy_cre_obj(cf)

    # Find the the byte offset between the op and its copy
    cf_ptr = cast(cf, i8)
    cpy_ptr = cast(cpy, i8)
    cpy_delta = cpy_ptr-cf_ptr

    # Make a copy of base_to_head_infos
    # base_to_head_infos = List.empty_list(head_info_arr_type)
    old_head_ranges = cf.head_ranges.copy()
    old_head_infos = cf.head_infos.copy()
    old_prereq_instrs = cf.prereq_instrs.copy()

    # Nullify these attributes since we don't want the pointers from the 
    #  original to get decref'ed on assignment
    _nullify_attr(cpy, 'head_ranges')
    _nullify_attr(cpy, 'head_infos')
    _nullify_attr(cpy, 'root_arg_infos')
    _nullify_attr(cpy, 'name_data')
    _nullify_attr(cpy, 'prereq_instrs')

    # Rebuild head_ranges, head_infos, prereq_instrs from common buffer
    cpy_generic = cast(cpy, CREFuncType)
    rebuild_buffer(cpy_generic, old_head_ranges, old_head_infos, old_prereq_instrs)

    # Make the arg_data_ptr and head_data_ptr point to the copy
    for i, head_info in enumerate(cf.head_infos):
        cpy.head_infos[i] = head_info
        if(head_info.cf_ptr == cf_ptr):
            cpy.head_infos[i].cf_ptr = cpy_ptr
            cpy.head_infos[i].arg_data_ptr = head_info.arg_data_ptr+cpy_delta
            cpy.head_infos[i].head_data_ptr = head_info.head_data_ptr+cpy_delta

    cpy.return_data_ptr = cf.return_data_ptr+cpy_delta
    cpy.root_arg_infos = cf.root_arg_infos.copy()
    cpy.name_data = cf.name_data
    cpy.is_initialized = False

    # print("::", f"{cf_ptr}:{cast(cf.root_arg_infos, i8)}","|", f"{cpy_ptr}:{cast(cpy.root_arg_infos, i8)}")
    # print("&&>",cf_ptr,cf.return_data_ptr,cf.return_data_ptr-cf_ptr)
    # print("&&<",cpy_ptr,cpy.return_data_ptr,cpy.return_data_ptr-cpy_ptr)

    return cpy

@njit(CREFuncType(CREFuncType), cache=True)
def cre_func_copy_generic(cf):
    return cre_func_copy(cast(cf, CREFuncType))


@njit(CREFuncType(CREFuncType),cache=True)
def cre_func_deep_copy_generic(cf):
    '''Makes a deep copy of a CREFunc Object.'''

    if(len(cf.prereq_instrs)==0):
        return cre_func_copy_generic(cf)

    # Initialize Data Structures
    stack = List()
    i = 0 
    op_copies = List.empty_list(CREFuncType)
    remap = Dict.empty(i8,i8)

    keep_looping = True
    while(keep_looping):
        arg_info = cf.root_arg_infos[i]
        if(arg_info.type == ARGINFO_OP):
            # Push Stack i.e. Recurse Arg
            stack.append((cf, i+1, op_copies))
            cf = cast(arg_info.ptr, CREFuncType)
            op_copies = List.empty_list(CREFuncType)
            i = 0 
        else:
            # Skip Arg
            i += 1

        # When past end arg of deepest op, copy it. Then pop prev frame from stack.
        while(i >= len(cf.root_arg_infos)):
            cpy = cre_func_copy_generic(cf)

            # Set the "ref{i}" and root_arg_info.ptr to copies of op children 
            j = 0 
            for k, inf in enumerate(cpy.root_arg_infos):
                if(inf.type == ARGINFO_OP):
                    sub_op = op_copies[j]
                    inf.ptr = cast(sub_op, i8)
                    cre_obj_set_item(cpy, i8(1+cpy.n_root_args*2 + k), sub_op)
                j += 1

            remap[cast(cf, i8)] = cast(cpy, i8)
            remap[cf.return_data_ptr] = cpy.return_data_ptr

            # Remap pointers in head_infos 
            for hi, hi_cpy in zip(cf.head_infos, cpy.head_infos):
                hi_cpy.cf_ptr = remap.get(hi.cf_ptr, hi_cpy.cf_ptr)
                hi_cpy.arg_data_ptr = remap.get(hi.arg_data_ptr, hi_cpy.arg_data_ptr)
                hi_cpy.head_data_ptr = remap.get(hi.head_data_ptr, hi_cpy.head_data_ptr)
                remap[hi.arg_data_ptr] = hi_cpy.arg_data_ptr
                remap[hi.head_data_ptr] = hi_cpy.head_data_ptr

            # Remap pointers in prereq_instrs
            for instr, instr_cpy in zip(cf.prereq_instrs, cpy.prereq_instrs):
                instr_cpy.cf_ptr = remap.get(instr.cf_ptr, instr_cpy.cf_ptr)
                instr_cpy.return_data_ptr = remap.get(instr.return_data_ptr, instr_cpy.return_data_ptr)

            # End Case: Stack exhausted
            if(len(stack) == 0):
                keep_looping = False
                break

            # Pop off stack i.e. equivalent to returning recursive call
            cf, i, op_copies = stack.pop(-1)
            op_copies.append(cpy)
    return cpy

cf_deep_copy_ep = cre_func_deep_copy_generic.overloads[(CREFuncType,)].entry_point

#--------------------------------------------------------------------
# Construction Functions


from numba.core.typing.typeof import typeof
set_const_arg_overloads = {}
def set_const_arg_impl(_val, use_ep=False):
    nb_val_type = typeof(_val) if not isinstance(_val, types.Type) else _val
    if(nb_val_type not in set_const_arg_overloads):
        
        sig = types.void(CREFuncType, i8, nb_val_type)
        @njit(sig,cache=True)
        def _set_const_arg(self, i, val):
            self.is_initialized = False

            head_infos = self.head_infos
            start = self.head_ranges[i].start
            end = self.head_ranges[i].end
            for j in range(start,end):
                cf = cast(head_infos[j].cf_ptr, CREFuncType)
                arg_ind = head_infos[j].arg_ind

                head_infos[j].type = ARGINFO_CONST
                head_infos[j].has_deref = 0
                # set 'h{i}' to val
                cre_obj_set_item(cf, i8(1+(arg_ind<<1)+1), val)
                # set 'ref{i}' to None
                cre_obj_set_item(cf, i8(1+cf.n_root_args*2 + arg_ind), None)

                cf.root_arg_infos[arg_ind].type = ARGINFO_CONST
                cf.root_arg_infos[arg_ind].has_deref = 0
                cf.root_arg_infos[arg_ind].ptr = head_infos[j].head_data_ptr
                cf.root_arg_infos[arg_ind].t_id = head_infos[j].head_t_id

        entry_point = next(iter(_set_const_arg.overloads.values())).entry_point
        set_const_arg_overloads[nb_val_type] = (_set_const_arg, entry_point)
    if(use_ep):
        return set_const_arg_overloads[nb_val_type][1]
    else:
        return set_const_arg_overloads[nb_val_type][0]


@generated_jit(cache=True)
def set_const_arg(self, i, val):
    impl = set_const_arg_impl(val)
    return impl

@njit(types.void(CREFuncType,i8,VarType), cache=True)
def set_var_arg(self, i, var):
    self.is_initialized = False
    head_infos = self.head_infos
    start = self.head_ranges[i].start
    end = self.head_ranges[i].end

    var_ptr = cast(var, i8)
    hd = u1(len(var.deref_infos) > 0)
    for j in range(start,end):
        cf = cast(head_infos[j].cf_ptr, CREFuncType)
        arg_ind = head_infos[j].arg_ind
        head_infos[j].var_ptr = var_ptr
        head_infos[j].has_deref = u1(hd)
        head_infos[j].type = ARGINFO_VAR

        # set 'ref{i}' to var
        cre_obj_set_item(cf, i8(1+cf.n_root_args*2 + arg_ind), var)

        cf.root_arg_infos[arg_ind].type = ARGINFO_VAR
        cf.root_arg_infos[arg_ind].has_deref = hd
        cf.root_arg_infos[arg_ind].ptr = cast(var, i8)
        cf.root_arg_infos[arg_ind].t_id = head_infos[j].head_t_id

set_var_arg_ep = set_var_arg.overloads[(CREFuncType,i8,VarType)].entry_point


@njit(types.void(CREFuncType,i8,CREFuncType), cache=True, debug=False)
def set_op_arg(self, i, op):
    self.is_initialized = False
    head_infos = self.head_infos
    start = self.head_ranges[i].start
    end = self.head_ranges[i].end

    op_ptr = cast(op, i8)
    for j in range(start,end):
        cf = cast(head_infos[j].cf_ptr, CREFuncType)

        arg_ind = head_infos[j].arg_ind

        head_infos[j].cf_ptr = op_ptr
        head_infos[j].has_deref = 0
        head_infos[j].type = ARGINFO_OP_UNEXPANDED

        # Set ref{i} to op
        cre_obj_set_item(cf, i8(1+cf.n_root_args*2 + arg_ind), op)

        cf.root_arg_infos[arg_ind].type = ARGINFO_OP
        cf.root_arg_infos[arg_ind].has_deref = 0
        cf.root_arg_infos[arg_ind].ptr = op_ptr
        cf.root_arg_infos[arg_ind].t_id = head_infos[j].head_t_id

set_op_arg_ep = set_op_arg.overloads[(CREFuncType,i8,CREFuncType)].entry_point


cf_ind_tup_t = Tuple((CREFuncType,i8))
@njit(instr_type[::1](CREFuncType),cache=True)
def build_instr_set(self):
    ''' Rebuilds  prereq_intrs for a CREFunc '''
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
        arg_info = cf.root_arg_infos[i]
        assert arg_info.ptr != cast(cf, i8), "CREFunc has self reference"
        if(arg_info.type == ARGINFO_OP):
            t_id, m_id, head_data_ptr = cre_obj_get_item_t_id_ptr(cf,(1+(i<<1)+1))
            instr = np.zeros(1,dtype=instr_type)[0]
            instr.cf_ptr = arg_info.ptr
            instr.return_data_ptr = head_data_ptr

            if(t_id == T_ID_STR):
                instr.size = _sizeof_type(unicode_type) #Hard-coded ... see commented out part below
                instr.ref_kind = REFKIND_UNICODE
            elif(m_id != PRIMITIVE_MBR_ID):
                instr.size = _sizeof_type(CREFuncType)
                instr.ref_kind = REFKIND_STRUCTREF
            else:
                instr.size = _sizeof_type(i8)
                instr.ref_kind = REFKIND_PRIMATIVE

            # Set the size to be the difference between the data_ptrs for 
            #  the return value and the a0 member 
            # _, _, ret_data_ptr = cre_obj_get_item_t_id_ptr(cf,0)
            # _, _, first_data_ptr = cre_obj_get_item_t_id_ptr(cf,1)
            # instr.size = u4(first_data_ptr-ret_data_ptr)

            instrs.append(instr)
            stack.append((cf, i+1))
            cf = cast(arg_info.ptr, CREFuncType)
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

i8_arr = i8[::1]
head_info_lst = ListType(head_info_type)
@njit(types.void(CREFuncType),cache=True)
def reinitialize(self):
    ''' Reinitializes a CREFunc after a composition has been made with 
        a sequence of calls to set_[op,var,const]_arg(). This function 
        ensures that head_ranges, head_infos, and prereq_instrs are all
        properly set to reflect the composition. 
    '''
    if(self.is_initialized): return

    max_arg_depth = 0

    base_var_map = Dict.empty(i8, head_info_lst)
    # Go through the current HEAD_INFOS looking for entries with 
    #  kind ARGINFO_VAR or ARGINFO_OP_UNEXPANDED 
    for hrng in self.head_ranges:
        for j in range(hrng.start,hrng.end):
            head_info = self.head_infos[j]

            # For ARGINFO_VAR kinds insert base_ptr into the base_var_map
            if(head_info.type == ARGINFO_VAR):
                var = cast(head_info.var_ptr, VarType)
                base_ptr = var.base_ptr
                if(base_ptr not in base_var_map):
                    base_var_map[base_ptr] = List.empty_list(head_info_type)
                base_var_map[base_ptr].append(head_info)

            # For ARGINFO_OP_UNEXPANDED kind insert the base_ptrs of all of the
            #  CRE_Funcs's base vars into base_var_map
            elif(head_info.type == ARGINFO_OP_UNEXPANDED):
                cf = cast(head_info.cf_ptr, CREFuncType)
                for hrng_k in cf.head_ranges:
                    for n  in range(hrng_k.start, hrng_k.end):
                        head_info_n = cf.head_infos[n]
                        var = cast(head_info_n.var_ptr, VarType)
                        base_ptr = var.base_ptr
                        if(base_ptr not in base_var_map):
                            base_var_map[base_ptr] = List.empty_list(head_info_type)
                        base_var_map[base_ptr].append(head_info_n)
                max_arg_depth = max(cf.depth, max_arg_depth)

    # Make new head_ranges according to base_var_map
    n_bases = len(base_var_map)
    head_ranges = np.zeros(n_bases, dtype=head_range_type)
    n_heads = 0
    for i, base_head_infos in enumerate(base_var_map.values()):
        head_ranges[i].start = n_heads
        head_ranges[i].end = n_heads+len(base_head_infos)
        n_heads += len(base_head_infos)

    # Make new head_infos according to base_var_map
    head_infos = np.zeros(n_heads,dtype=head_info_type)
    c = 0
    for i, (base_ptr, base_head_infos) in enumerate(base_var_map.items()):
        base_var = cast(base_ptr, VarType)
        for j in range(len(base_head_infos)):
            head_infos[c] = base_head_infos[j]
            head_infos[c].base_t_id = base_var.base_t_id
            c += 1

    self.n_args = n_bases

    prereq_instrs = build_instr_set(self)
    rebuild_buffer(self, head_ranges, head_infos, prereq_instrs)

    # Record if the CREFunc was modified enough from it's base definition
    #  that it should be considered 'is_composed'. Also check 'has_any_derefs'.
    any_hd = False
    is_composed = False
    for inf in self.root_arg_infos:
        any_hd = any_hd | inf.has_deref
        if(any_hd or inf.type == ARGINFO_OP or inf.type == ARGINFO_CONST):
            is_composed = True
    self.has_any_derefs = any_hd
    self.is_composed = is_composed
    self.depth = self.depth + max_arg_depth
  

#--------------------------------------------------------------------
# Execution Functions

#--------------------------------
# Set Arguments
        
from numba.core.typing.typeof import typeof
set_base_arg_val_overloads = {}
def set_base_arg_val_impl(_val):
    # Get the type of _val or if a typeref was given use that
    nb_val_type = typeof(_val) if not isinstance(_val, types.Type) else _val

    # If is a Fact or other object type upcast to CREObjType
    nb_val_type = CREObjType if isinstance(nb_val_type, CREObjTypeClass) else nb_val_type
    # Compile the implementation if it doesn't exist
    if(nb_val_type not in set_base_arg_val_overloads):        
        sig = types.void(CREFuncType, u8, nb_val_type)
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

set_obj_base_val = set_base_arg_val_impl

@njit(types.void(CREFuncType, u4, i8[::1]), cache=True, locals={"i" : u4,"k":u4, "j": u4})
def set_heads_from_data_ptrs(cf, i, data_ptrs):
    for k, j in enumerate(range(cf.head_ranges[i].start, cf.head_ranges[i].end)):
        hi = cf.head_infos[j]
        if(hi.ref_kind==REFKIND_UNICODE): 
            # _decref_structref(_load_ptr(unicode_type, hi.head_data_ptr))
            _incref_structref(_load_ptr(unicode_type, data_ptrs[k]))
        elif(hi.ref_kind==REFKIND_STRUCTREF):
            # _decref_ptr(_load_ptr(i8, hi.head_data_ptr))
            _incref_ptr(_load_ptr(i8, data_ptrs[k]))

        _memcpy(hi.head_data_ptr, data_ptrs[k], hi.head_size)

#--------------------------------
# Call

call_self_f_type = types.FunctionType(u1(CREFuncType))

@njit(u1(CREFuncType), locals={"i":u8}, cache=True)
def cre_func_call_self(self):
    ''' Calls the CREFunc including any dependant CREFuncs in its composition
        Each CREFunc's a{i} and h{i} characteristic members are used as
        intermediate memory spaces for the results of these calculations.
    '''
    for i in range(len(self.prereq_instrs)):
        instr = self.prereq_instrs[i]
        cf = cast(instr.cf_ptr, CREFuncType)

        status = _func_from_address(call_self_f_type, cf.call_self_addr)(cf)
        if(status > 1): return status

        if(instr.ref_kind==REFKIND_UNICODE):
            new_obj = _load_ptr(unicode_type, cf.return_data_ptr)   
            _incref_structref(new_obj)
        elif(instr.ref_kind==REFKIND_STRUCTREF):
            new_obj_ptr = _load_ptr(i8, cf.return_data_ptr)   
            _incref_ptr(new_obj_ptr)

        _memcpy(instr.return_data_ptr, cf.return_data_ptr, instr.size)
    status = _func_from_address(call_self_f_type, self.call_self_addr)(self)
    return status

CREFunc_method(CREFuncType, u1(CREFuncType), "call_self")(cre_func_call_self)


@njit(u1(CREFuncType), locals={"i":u8}, cache=True)
def cre_func_resolve_call_self(self):
    ''' As 'cre_func_call_self' but also dereferences in the composition's
        head variables. For instance like my_cf(a.nxt.B, b.nxt.A).
    '''
    for i in range(len(self.prereq_instrs)):
        
        instr = self.prereq_instrs[i]
        cf = cast(instr.cf_ptr, CREFuncType)
        if(cf.has_any_derefs):
            status = _func_from_address(call_self_f_type, cf.resolve_heads_addr)(cf)
            if(status): return status
        status = _func_from_address(call_self_f_type, cf.call_self_addr)(cf)
        if(status > 1): return status

        if(instr.ref_kind==REFKIND_UNICODE):
            new_obj = _load_ptr(unicode_type, cf.return_data_ptr)   
            _incref_structref(new_obj)
        elif(instr.ref_kind==REFKIND_STRUCTREF):
            new_obj_ptr = _load_ptr(i8, cf.return_data_ptr)   
            _incref_ptr(new_obj_ptr)
    
        _memcpy(instr.return_data_ptr, cf.return_data_ptr, instr.size)

    if(self.has_any_derefs):
        status = _func_from_address(call_self_f_type, self.resolve_heads_addr)(self)
        if(status): return status

    status = _func_from_address(call_self_f_type, self.call_self_addr)(self)
    if(status > 1): return status
    return status

CREFunc_method(CREFuncType, u1(CREFuncType), "resolve_call_self")(cre_func_resolve_call_self)


# TODO: Make this
# def identity_call_self():
#     if(self.has_any_derefs):
#         status = _func_from_address(call_self_f_type, self.resolve_heads_addr)(self)
#         if(status): return status

#     if(instr.ref_kind==REFKIND_UNICODE):
#         new_obj = _load_ptr(unicode_type, cf.return_data_ptr)   
#         _incref_structref(new_obj)
#     elif(instr.ref_kind==REFKIND_STRUCTREF):
#         new_obj_ptr = _load_ptr(i8, cf.return_data_ptr)   
#         _incref_ptr(new_obj_ptr)

#     _memcpy(cf.return_data_ptr, cf.head_infos[0].head_data_ptr return_data_ptr, instr.size)
#     _store_safe(return_type, self.return_data_ptr, return_val)



cs_gbl_name = 'CREFuncType_call_self'
res_cs_gbl_name = 'CREFuncType_resolve_call_self'

@njit
def get_best_call_self(self, ignore_derefs=False):
    '''Returns the fastest implementation of call_self for a CREFunc.
        In the best case, if the CREFunc doesn't have other CREFuncs in it's 
        composition and no head Vars with dereference instructions then the 
        fastest implementation is it's type specific implementation.
    '''
    if(not ignore_derefs and self.has_any_derefs):
        # Use cre_func_resolve_call_self
        cs_addr = _get_global_fn_addr(res_cs_gbl_name)
    elif(len(self.prereq_instrs) > 0):
        # Use cre_func_call_self
        cs_addr = _get_global_fn_addr(cs_gbl_name)
    else:
        # Use base type-specific implementation
        cs_addr = self.call_self_addr
    return _func_from_address(call_self_f_type, cs_addr)

#--------------------------------
# Get Return Value
    
from numba.core.typing.typeof import typeof
get_return_val_overloads = {}
# i8_x6 = UniTuple(i8,6)
def get_return_val_impl(val_type):
    ''' Implementation for extracting return value of 'val_type' from CREFunc '''
    nb_val_type = typeof(val_type) if not isinstance(val_type, types.Type) else val_type
    if(nb_val_type not in get_return_val_overloads):
        @njit(nb_val_type(CREFuncType),cache=True, inline='always')
        def _get_return_val(self):
            val = _load_ptr(nb_val_type, self.return_data_ptr)
            # _incref_structref(val)
            # _store(i8_x6, self.return_data_ptr, (0,0,0,0,0,0))
            # print(_load_ptr(nb_val_type, self.return_data_ptr))
            return val
        get_return_val_overloads[nb_val_type] = _get_return_val
    return get_return_val_overloads[nb_val_type]


#--------------------------------------------------------------------
@generated_jit
def _set_var_helper(kind, cf, i, arg):
    '''Helper function for applying set_[var,op,const]_arg depending on 'kind' '''
    # SentryLiteralArgs(['kind']).for_function(_set_var_helper).bind(kind, cf, i, arg)
    kind = kind.literal_value
    if(kind==ARGINFO_VAR):
        def impl(kind, cf, i, arg):    
            set_var_arg(cf, i, cast(arg, VarType))  
    elif(kind==ARGINFO_OP):
        def impl(kind, cf, i, arg):    
            set_op_arg(cf, i, cast(arg, CREFuncType))
    elif(kind==ARGINFO_CONST):
        def impl(kind, cf, i, arg):    
            set_const_arg(cf, i, arg)
    return impl
    
# Overload call/compose
from numba.types import unliteral
def overload_compose_codegen(cf_type, arg_types):
    ''' Codegen for composing a CREFunc from jitted code'''
    kinds = []
    for i, at in enumerate(arg_types):
        if(isinstance(at, VarTypeClass)):
            kinds.append(int(ARGINFO_VAR))
        elif(isinstance(at,CREFuncTypeClass)):
            kinds.append(int(ARGINFO_OP))
        else:
            kinds.append(int(ARGINFO_CONST))

    most_precise_cf_type = CREFuncTypeClass(return_type=cf_type.return_type, name="FuncComp")
    inds = tuple([*range(len(arg_types))])
    kinds = tuple(kinds)
    def impl(cf, *args):
        _cf = cast(cre_func_deep_copy_generic(cf), most_precise_cf_type)#cre_func_copy(cf)
        for i in literal_unroll(inds):
            _set_var_helper(kinds[i],_cf,i,args[i])
        reinitialize(_cf)
        return cast(_cf, most_precise_cf_type)
    return impl


def overload_call_codegen(cf_type, arg_types):
    ''' Codegen for calling a CREFunc from jitted code'''
    if(cf_type.is_composed):
        if(len(arg_types) > 0 and isinstance(arg_types[0],types.BaseTuple)):
            arg_types = tuple(x for x in arg_types[0])

        set_base_impls = tuple(set_base_arg_val_impl(a) for a in arg_types)
        set_base = set_base_impls[0]
        ret_impl = get_return_val_impl(cf_type.return_type)
        range_inds = tuple(u8(i) for i in range(len(arg_types)))
        def impl(cf,*args):
            for i in literal_unroll(range_inds):
                set_base(cf, i, args[i])
            status = cre_func_call_self(cf)
            return ret_impl(cf)

    elif(hasattr(cf_type,'dispatchers') and 'call' in cf_type.dispatchers):
        call = cf_type.dispatchers['call']
        def impl(cf, *args):
            return call(*args)
    else:
        fn_type = types.FunctionType(cf_type.return_type(*cf_type.arg_types))
        def impl(cf, *args):
            f = _func_from_address(fn_type, cf.call_heads_addr)
            return f(*args)
    return impl


#-----------------------------------------------------------------
# : CREFunc Overload __call__

@overload_method(CREFuncTypeClass, '__call__')
def overload_call(cf, *args):
    ''' Allows for CREFuncs to be called and composed from jitted code.'''
    cf_type = cf
    if(len(args) > 0 and isinstance(args[0], types.BaseTuple)):
        args = tuple(*args)

    all_const = all_args_are_const(args)
    if(all_const):
        if(not getattr(cf_type, 'return_type',None)):
            raise ValueError("Cannot call CREFunc without return_type")

        return overload_call_codegen(cf, args)
    else:
        if(not getattr(cf_type, 'return_type',None)):
            raise ValueError("Cannot compose CREFunc without return_type")

        return overload_compose_codegen(cf, args)
            
    return impl


#-----------------------------------------------------------------
# : cre_func_str()

rc_fn_typ = types.FunctionType(unicode_type(CREObjType, i8))


@njit(unicode_type(CREFuncType, types.boolean), cache=True)
def cre_func_str(self, use_shorthand):
    ''' Generates a string representation for a CREFunc.'''
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
            cf = cast(arg_info.ptr, CREFuncType)
            arg_strs = List.empty_list(unicode_type)
            i = 0 
        else:
            if(arg_info.type == ARGINFO_VAR):
                var = cast(arg_info.ptr, VarType)
                arg_strs.append(str(var)) 
            elif(arg_info.type == ARGINFO_CONST):
                addr = cf.name_data.repr_const_addrs[i]
                fn = _func_from_address(rc_fn_typ, addr)
                arg_strs.append(fn(cf,i))
            else:
                raise ValueError("Bad arginfo type.")
            i += 1
        while(i >= len(cf.root_arg_infos)):
            nd = cf.name_data
            tmp = nd.shorthand_template if use_shorthand else nd.expr_template
            s = tmp.format(arg_strs)
            if(len(stack) == 0):
                keep_looping = False
                break
            if(use_shorthand):
                parent_nd = stack[-1][0].name_data
                parent_tmp = parent_nd.shorthand_template if use_shorthand else parent_nd.expr_template
                if(tmp[-1] != ")" and parent_tmp[-1] != ")"):
                    s = f"({s})"

            cf, i, arg_strs = stack.pop(-1)
            arg_strs.append(s)
    return s

@overload(str)
def overload_cre_func_str(self):
    if(self is not CREFuncType): return
    def impl(self):
        return cre_func_str(self, True)
    return impl

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
        f"CREFunc {members['func_name']!r} has invalid 'commutes' member {commutes}. Signature arguments {k} and {commuting_set[j]} have different types {typ1} and {typ2}."
    right_commutes = right_commutes

    members['commutes'] = commutes
    members['right_commutes'] = right_commutes

def _standardize_nopython(members):
    ''' 'nopython_call' and 'nopython_check' default to value of 'no_python' or None '''
    default = members.get('nopython',None)
    members['nopython_call'] = members.get('nopython', default)
    members['nopython_check'] = members.get('nopython', default)

    return members['nopython_call'], members['nopython_check']

def _py_func_unq(py_func):
    codebytes = py_func.__code__.co_code
    return codebytes
    # TODO: ALSO NEED __closure__?
    # print("cb:", codebytes)
    # if py_func.__closure__ is not None:
    #     cvars = tuple([x.cell_contents for x in self._py_func.__closure__])
    #     # Note: cloudpickle serializes a function differently depending
    #     #       on how the process is launched; e.g. multiprocessing.Process
    #     cvarbytes = dumps(cvars)
    # else:
    #     cvarbytes = b''


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

        return members[f'{method_name}_bytes'], _py_func_unq(py_func)
    else:
        return None, None
    
def define_CREFunc(name, members):
    ''' Defines a new CREFuncType '''

    # Ensure that 'call' is present and that 'call' and 'check' are functions
    has_check = 'check' in members
    assert 'call' in members, "CREFunc must have call() defined"
    assert hasattr(members['call'], '__call__'), "call() must be a function"
    assert not (has_check and not hasattr(members['check'], '__call__')), "check() must be a function"

    members['func_name'] = name
    # inp_members = {**members}

    no_raise = members.get('no_raise',False)
    ptr_args = members.get('ptr_args',False)

    # Unpack signature
    signature = members['signature']
    return_type = members['return_type'] = signature.return_type
    if(not ptr_args):
        arg_types = members['arg_types'] = signature.args
    else:
        arg_types = members['arg_types'] = tuple([CREObjType]*len(signature.args))

    members['signature'] = return_type(*arg_types)

    # Standardize commutes and nopython
    _standardize_commutes(members)
    nopy_call, nopy_check = _standardize_nopython(members)

    # Get pickle bytes for 'call' and 'check'
    call_bytes, call_unq =_standardize_method(members, 'call')
    check_bytes, check_unq = _standardize_method(members, 'check')

    # Regenerate the source for this type if wasn't cached or if 'cache=False' 
    unq_args = [name, return_type, arg_types, call_bytes, check_bytes, no_raise, ptr_args]
    gen_src_args = [name, return_type, arg_types, nopy_call, nopy_check, call_bytes, check_bytes, no_raise, ptr_args]
    long_hash = unique_hash_v(unq_args)

    if(not source_in_cache(name, long_hash)): #or members.get('cache',True) == False):
        source = gen_cre_func_source(*gen_src_args, long_hash)
        source_to_cache(name, long_hash, source)

    # Update members with jitted 'call' and 'check'
    to_import = ['cf_type', 'call_heads'] + (['check'] if(has_check) else [])
    l = import_from_cached(name, long_hash, to_import)

    typ = l['cf_type']
    typ.call_heads = l['call_heads']
    if(has_check): typ.check = l['check']
    
    for k,v in members.items():
        setattr(typ,k,v)

    return typ


# -------------------------------------------------------------
# : CREFunc Source Generation

def gen_cre_func_source(name, return_type, arg_types, nopy_call, nopy_check,
             call_bytes, check_bytes, no_raise, ptr_args,
             long_hash ):
    '''Generate source code for the relevant functions of a user defined CREFunc.'''
    arg_names = ', '.join([f'a{i}' for i in range(len(arg_types))])

    on_error_map = {True : 'error', False : 'none', None : 'warn'}
    on_error_call = on_error_map[nopy_call]
    on_error_check = on_error_map[nopy_check]

    nl = "\n"
    source = \
f'''import numpy as np
from numba import njit, void, i8, u1, boolean, objmode
from numba.types import unicode_type
from numba.extending import lower_cast
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.utils import _incref_structref, cast, PrintElapse, _load_ptr, _obj_cast_codegen, _store_safe, _struct_get_attr_ptr
from cre.func import (ensure_repr_const, cre_func_ctor, CREFunc_method, CREFunc_assign_method_addr,
    CREFuncTypeClass, CREFuncType, VarType, CFSTATUS_TRUTHY, CFSTATUS_FALSEY, CFSTATUS_NULL_DEREF, CFSTATUS_ERROR)
from cre.memset import resolve_deref_data_ptr
from cre.fact import BaseFact
import cloudpickle


return_type = cloudpickle.loads({cloudpickle.dumps(return_type)})
arg_types = cloudpickle.loads({cloudpickle.dumps(arg_types)})
cf_type = CREFuncTypeClass(return_type, arg_types,is_composed=False, name={name!r}, long_hash={long_hash!r})


@lower_cast(cf_type, CREFuncType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)

{"call_sig = return_type(*arg_types)" if not ptr_args else
 "call_sig = return_type(*([i8]*len(arg_types)))"}
call_pyfunc = cloudpickle.loads({call_bytes})
{"".join([f'h{i}_type, ' for i in range(len(arg_types))])} = arg_types

call_heads = CREFunc_method(cf_type, call_sig, 'call_heads', on_error={on_error_call!r})(call_pyfunc)
if(call_heads is None):
    @CREFunc_method(cf_type, call_sig, 'call_heads')
    def call_heads({arg_names}):
        with objmode(_return=return_type):
            _return = call_pyfunc({arg_names})
        return _return

@CREFunc_method(cf_type, u1(CREFuncType))
def resolve_heads(_self):
    self = cast(_self, cf_type)
    has_null_deref = False
    {"".join([f"""
    if(self.root_arg_infos[{i}].has_deref):
        var = cast(self.ref{i}, VarType)
        a = cast(self.a{i}, BaseFact)
        data_ptr = resolve_deref_data_ptr(a, var.deref_infos)
        if(data_ptr != 0):
            self.h{i} = _load_ptr(h{i}_type, data_ptr)
        else:
            has_null_deref = True
""" for i in range(len(arg_types))])
    }
    if(has_null_deref):
        return CFSTATUS_NULL_DEREF
    return u1(0)

@CREFunc_method(cf_type, u1(CREFuncType))
def call_self(_self):
    self = cast(_self, cf_type)
'''
    if(no_raise):
        source +=f'''
    return_val = call_heads({", ".join([
            (f'_load_ptr(i8,_struct_get_attr_ptr(self,"h{i}"))' if(ptr_args) else f'self.h{i}') 
            for i in range(len(arg_types))])
        })
    _store_safe(return_type, self.return_data_ptr, return_val)
    return {"u1(return_val > 0)"
            if(isinstance(return_type, types.Number)) else (
             "CFSTATUS_TRUTHY if(return_val) else CFSTATUS_FALSEY"
             if(not isinstance(return_type, types.StructRef)) else
             "CFSTATUS_TRUTHY"
            )}
'''
    else:    
        source +=f'''
    try:
        return_val = call_heads({", ".join([
            (f'cast(self.h{i}, i8)' if(ptr_args) else f'self.h{i}')
            for i in range(len(arg_types))])
        })
        _store_safe(return_type, self.return_data_ptr, return_val)
        return {"u1(return_val > 0)"
                if(isinstance(return_type, types.Number)) else (
                 "CFSTATUS_TRUTHY if(return_val) else CFSTATUS_FALSEY"
                 if(not isinstance(return_type, types.StructRef)) else
                 "CFSTATUS_TRUTHY"
                )}
    except Exception:
        return CFSTATUS_ERROR    
'''
    if(check_bytes is not None):
        source += f'''
# check_pyfunc = cloudpickle.loads({check_bytes})
# check_sig = boolean(*arg_types)

# check = CREFunc_method(cf_type, check_sig, 'check', on_error={on_error_check!r})(check_pyfunc)
# if(check is None):
#     @CREFunc_method(cf_type, 'check', check_sig)
#     def check({arg_names}):
#         with objmode(_return=boolean):
#             _return = check_pyfunc({arg_names})
#         return _return
'''
    else:
        source += f'''
# CREFunc_assign_method_addr(cf_type, 'check', -1)
'''
    source += f'''
# @CREFunc_method(cf_type, boolean(*arg_types))
# def match({arg_names}):
#     {f"if(not check({arg_names})): return 0" if(check_bytes is not None) else ""}
#     return {f'1' if isinstance(return_type, StructRef) else f'1 if(call({arg_names})) else 0'}

# match_heads = match
# CREFunc_assign_method_addr(cf_type, 'match_heads', match.cre_method_addr)

# @CREFunc_method(cf_type, boolean(i8[::1],))
# def match_head_ptrs(ptrs):
# {indent(nl.join([f'#i{i} = _load_ptr(h{i}_type,ptrs[{i}])' for i in range(len(arg_types))]),prefix='    ')}
#     return match_heads({",".join([f'i{i}' for i in range(len(arg_types))])})


'''

    source +='''

# Make sure that constant members can be repr'ed
for at in arg_types:
    ensure_repr_const(at)

# ctor = gen_cre_func_ctor(cf_type)

@njit(CREFuncType(unicode_type, unicode_type, unicode_type), cache=True)
def ctor(name, expr_template, shorthand_template):
    return cre_func_ctor(cf_type, name, expr_template, shorthand_template, False)
cf_type.ctor = ctor
'''

    return source

# ---------------------------------------------------------------------------
# : Unique String (independant of base variables)

@njit(unicode_type(CREFuncType), cache=True)
def cre_func_unique_string(self):
    '''Outputs a tuple that uniquely identifies an instance
         of a literal independant of the base Vars of its underlying CREFunc.
    '''
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
            cf = cast(arg_info.ptr, CREFuncType)
            arg_strs = List.empty_list(unicode_type)
            i = 0 
        else:
            if(arg_info.type == ARGINFO_VAR):
                var = cast(arg_info.ptr, VarType)
                vs = ""
                for j, d in enumerate(var.deref_infos):
                    delim = "," if j != len(var.deref_infos)-1 else ""
                    vs += f'({str(i8(d.type))},{str(i8(d.t_id))},{str(i8(d.a_id))},{str(i8(d.offset))}){delim}'
                arg_strs.append(vs) 
            elif(arg_info.type == ARGINFO_CONST):
                addr = cf.name_data.repr_const_addrs[i]
                fn = _func_from_address(rc_fn_typ, addr)
                arg_strs.append(fn(cf,i))
            i += 1
        while(i >= len(cf.root_arg_infos)):
            _a = ','.join([f'{{{str(i)}}}' for i in range(len(cf.root_arg_infos))])
            tmp = f"{str(cf.call_self_addr)}({_a})"
            s = tmp.format(arg_strs)
            if(len(stack) == 0):
                keep_looping = False
                break

            cf, i, arg_strs = stack.pop(-1)
            arg_strs.append(s)
    return s

# ----------------------------------------------------------------------------
# : UntypeCREFunc

def resolve_return_type(x):
    # Helper function for determining how to type CREFunc arguments
    if(isinstance(x,Var)):
        return x.head_type
    elif(isinstance(x,CREFunc)):
        return x.return_type
    elif(isinstance(x, UntypedCREFunc)):
        raise ValueError(f"Cannot resolve return type of UntypedCREFunc {x}")
    elif(isinstance(x, (int))):
        return types.int64
    elif(isinstance(x, (float))):
        return types.float64
    elif(isinstance(x, (str))):
        return types.unicode_type

class UntypedCREFunc():
    '''An CREFunc that has not been given a signature yet'''
    def __init__(self, name, members):
        self.name = name
        self.members = members
        self.members['root_arg_names'] = inspect.getfullargspec(self.members['call'])[0]
        self.root_arg_names = self.members['root_arg_names']
        self.members['call'] = njit(cache=True)(self.members['call'])
        self._specialize_cache = {}

    def __repr__(self):
        return f'UntypedCREFunc(name={self.name}, members={self.members})'

    def __str__(self):
        return f'{self.name}({", ".join(self.root_arg_names)})'


    def __call__(self,*args):
        if(len(self.root_arg_names) != len(args)):
            raise ValueError(f"Got {len(args)} args for UntypedCREFunc {self.name!r} with {len(self.root_arg_names)} positional arguments.")

        args, all_const, head_types = _standardize_py_args(self, args)

        # Cache singleton CREFunc definition for args of 'head_types'.
        if(head_types not in self._specialize_cache):
            call = self.members['call']
            if(not self.members.get("ptr_args", False)):
                sig = dispatcher_sig_from_arg_types(call, tuple(head_types))
            else:
                sig = dispatcher_sig_from_arg_types(call, tuple([i8]*len(head_types)))

            members = {**self.members, 'signature': sig}#
            cf = new_cre_func(self.name, members)
            self._specialize_cache[head_types] = cf

        # Pass 'args' to the singleton instance to either call or compose it.
        cf = self._specialize_cache[head_types]
        return cf(*args)

# ----------------------------------------------------------------------------
# : PtrCREFunc

# def define_PtrCREFunc(name, members):
#     ''' Defines a new CREFuncType '''

#     # Ensure that 'call' is present and that 'call' and 'check' are functions
#     assert 'call' in members, "PtrCREFunc must have call() defined"
#     assert hasattr(members['call'], '__call__'), "call() must be a function"

#     # Get pickle bytes for 'call_head_ptrs'
#     call_bytes, call_unq = _standardize_method(members, 'call')

#     # Regenerate the source for this type if wasn't cached or if 'cache=False' 
#     gen_src_args = [name, members['return_type'], call_bytes]
#     unq_args = [name, members['return_type'], call_unq]

#     long_hash = unique_hash(unq_args)

#     # print(name, long_hash)
#     if(not source_in_cache(name, long_hash)): #or members.get('cache',True) == False):
#         source = gen_ptr_cre_func_source(*gen_src_args, members['arg_types'], long_hash)
#         source_to_cache(name, long_hash, source)

#     # Update members with jitted 'call_head_ptrs'
#     to_import = ['cf_type', 'call']
#     l = import_from_cached(name, long_hash, to_import)

#     typ = l['cf_type']
#     typ.func_name = name
#     typ.call = l['call']
    
#     for k,v in members.items():
#         setattr(typ,k,v)

#     return typ

# def new_ptr_cre_func(name, members):
#     call = members['call']
#     arg_types = members['arg_types'] = tuple([i8]*len(members['arg_names']))

#     # Determine the signature 
#     if('signature' not in members):
#         if('return_type' in members):
#             members['signature'] = members['return_type'](arg_types)
#         else:
#             sig = dispatcher_sig_from_arg_types(call, arg_types)
#             members['signature'] = sig
#             members['return_type'] = sig.return_type

#     cf_type = define_PtrCREFunc(name, members)

#     expr_template = f"{name}({', '.join([f'{{{i}}}' for i in range(len(arg_types))])})"
#     shorthand_template = members.get('shorthand',expr_template)

#     cre_func = cf_type.ctor(
#         name,
#         expr_template,
#         shorthand_template                
#     )

#     cre_func._return_type = members['return_type']
#     cre_func._arg_types = arg_types
#     # cre_func.cf_type = cf_type
#     return cre_func

# class PtrCREFunc(CREFunc):

#     def __new__(self,*args, **kwargs):
#         ''' A decorator function that builds a new Op'''
#         # if(len(args) > 1): raise ValueError("PtrCREFunc() takes at most one position argument 'signature'.")
        
#         def wrapper(call_func):
#             # Make new cre_func_type
#             members = kwargs
#             members["call"] = njit(cache=True)(call_func)
#             name = call_func.__name__
#             members["arg_names"] = inspect.getfullargspec(call_func)[0]
#             # If no signature is given then create an untyped instance that can be specialized later
#             # if('signature' not in members):
#             #     return UntypedCREFunc(name,members)
#             # print(">>", members)
#             return new_ptr_cre_func(name, members)
            

#         if(len(args) == 1):
#             if(isinstance(args[0],(str, numba.core.typing.templates.Signature))):
#                 kwargs['signature'] = args[0]
#             elif(hasattr(args[0],'__call__')):
#                 return wrapper(args[0])
#             else:
#                 raise ValueError(f"Unrecognized type {type(args[0])} for 'signature'")        

#         return wrapper

#     def __call__(self,*py_args):
#         assert len(py_args) == self.nargs, f"{str(self)} takes {self.nargs}, but got {len(py_args)}"
#         if(any([not isinstance(x,Var) for x in py_args])):
#             raise ValueError("PtrOps only take Vars as input")
#             # If all of the arguments are constants then just call the Op
#             # return self.call(*py_args)
#         else:
#             # head_var_ptrs = np.empty((self.nargs,),dtype=np.int64)
#             # for i, v in enumerate(py_args):
#             #     head_var_ptrs[i] = v.get_ptr()
#             return self.__class__.make_singleton_inst(py_args)



# def gen_ptr_cre_func_source(name, return_type, call_bytes, arg_types, long_hash):
#     source = \
# f'''import numpy as np
# from numba import njit, void, i8, boolean, objmode
# from numba.types import unicode_type
# from numba.extending import lower_cast
# from numba.core.errors import NumbaError, NumbaPerformanceWarning
# from cre.utils import PrintElapse, _struct_from_ptr, _load_ptr, _obj_cast_codegen, _store_safe, 
# from cre.func import ensure_repr_const, cre_func_ctor, CREFunc_method, CREFunc_assign_method_addr, CREFuncTypeClass, CREFuncType, VarType
# import cloudpickle

# return_type = cloudpickle.loads({cloudpickle.dumps(return_type)})
# arg_types = cloudpickle.loads({cloudpickle.dumps(arg_types)})

# cf_type = CREFuncTypeClass(return_type, arg_types, is_composed=False, name={name!r}, long_hash={long_hash!r})

# call_pyfunc = cloudpickle.loads({call_bytes})
# call_sig = return_type(*arg_types)
# call = CREFunc_method(cf_type, call_sig, 'call', on_error='error')(call_pyfunc)

# call_heads = call

# @CREFunc_method(cf_type, return_type(i8[::1]))
# def call_head_ptrs(ptrs):
#     return call_heads({",".join([f'ptrs[{i}]' for i in range(len(arg_types))])})

# @CREFunc_method(cf_type, boolean(i8[::1]))
# def match_head_ptrs(ptrs):
#     return {f'1' if isinstance(return_type, StructRef) else
#             f'1 if(call_head_ptrs(ptrs)) else 0'}

# #Placeholder repr
# ensure_repr_const(i8)

# @njit(CREFuncType(unicode_type, unicode_type, unicode_type), cache=True)
# def ctor(name, expr_template, shorthand_template):
#     return cre_func_ctor(cf_type, name, expr_template, shorthand_template, True)
# cf_type.ctor = ctor
# '''
#     return source
