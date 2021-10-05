import operator
import numpy as np
import numba
from numba.core.dispatcher import Dispatcher
from numba import types, njit, i8, u8, i4, u1, i8, literally, generated_jit, boolean
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import NativeValue, box, unbox, overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template
from cre.memory import MemoryType, Memory, facts_for_t_id, fact_at_f_id
# from cre.fact import define_fact, BaseFactType, cast_fact, DeferredFactRefType, Fact
from cre.utils import (_struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_pointer,  lower_setattr, lower_getattr,
                       _pointer_from_struct, _decref_pointer, _incref_pointer, _incref_structref, _pointer_from_struct_incref)
from cre.utils import assign_to_alias_in_parent_frame, as_typed_list, iter_typed_list
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.fact import Fact, gen_fact_import_str, get_offsets_from_member_types
from cre.var import Var
from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
 get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from cre.make_source import make_source, gen_def_func, gen_assign, resolve_template, gen_def_class
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models, register_default
from numba.experimental.function_type import _get_wrapper_address


from operator import itemgetter
from copy import copy
from os import getenv
from cre.utils import deref_type, OFFSET_TYPE_ATTR, OFFSET_TYPE_LIST, listtype_sizeof_item
import inspect, cloudpickle, pickle
from textwrap import dedent, indent
# from itertools import combinations
from collections.abc import Iterable
import warnings


import time
class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.2f} ms')

def warn_cant_compile(func_name, op_name, e):
    s = f'''
########## WARNING CAN'T COMPILE {op_name}.{func_name}() ########### 
{indent(str(e),'  ')} 
########################################################
Numba was unable to compile {func_name}() for op {op_name}. 
Using objmode (i.e. native python) instead. To ignore warning set nopython=False.\n\
'''
    warnings.warn(s, NumbaPerformanceWarning)


def gen_op_source(cls, call_needs_jitting, check_needs_jitting):
    '''Generate source code for the relevant functions of a user defined Op.
       Note: The jitted call() and check() functions are pickled, the bytes dumped  
        into the source, and then reconstituted at runtime. This strategy seems to work
        even if globals are used in the function. 
    '''
    arg_types = cls.signature.args
    return_type = cls.signature.return_type
    # arg_imports = "\n".join([gen_fact_import_str(x) for x in arg_types if isinstance(x,Fact)])
    field_dict = {**op_fields_dict,**{f"arg{i}" : t for i,t in enumerate(arg_types)}}
    arg_names = ', '.join([f'a{i}' for i in range(len(arg_types))])
    # offsets = get_offsets_from_member_types(field_dict)
    # arg_offsets = offsets[len(op_fields_dict):]
    nl = "\n"
    source = \
f'''from numba import njit, void, i8, boolean, objmode
from numba.experimental.function_type import _get_wrapper_address
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.utils import _func_from_address, _load_pointer
from cre.op import op_fields_dict, OpTypeTemplate, warn_cant_compile
import cloudpickle
nopython_call = {cls.nopython_call} 
nopython_check = {cls.nopython_check} 

call_sig = cloudpickle.loads({cloudpickle.dumps(cls.call_sig)})
call_pyfunc = cloudpickle.loads({cls.call_bytes})

try:
    call = njit(call_sig,cache=True)(call_pyfunc)
except NumbaError as e:
    nopython_call=False
    warn_cant_compile('call',{cls.__name__!r}, e)

if(nopython_call==False):
    return_type = call_sig.return_type
    if(not nopython_call):
        @njit(cache=True)
        def call({arg_names}):
            with objmode(_return=return_type):
                _return = call_pyfunc({arg_names})
            return _return

call_addr = _get_wrapper_address(call, call_sig)
'''
    if(hasattr(cls,'check')):
        source += f'''
check_sig = cloudpickle.loads({cloudpickle.dumps(cls.check_sig)})
check_pyfunc = cloudpickle.loads({cls.check_bytes})

try:
    check = njit(check_sig,cache=True)(check_pyfunc)
except NumbaError as e:
    nopython_check=False
    warn_cant_compile('check',{cls.__name__!r}, e)

if(nopython_check==False):
    @njit(cache=True)
    def check({arg_names}):
        with objmode(_return=boolean):
            _return = check_pyfunc({arg_names})
        return _return

check_addr = _get_wrapper_address(check, check_sig)
'''
# arg_offsets = {str(arg_offsets)}
    source += f'''
@njit(boolean(*call_sig.args), cache=True)
def match({arg_names}):
    {f"if(not check({arg_names})): return 0" if(hasattr(cls,'check')) else ""}
    return 1 if(call({arg_names})) else 0

match_head = match

{",".join([f'h{i}_type' for i in range(len(arg_types))])} = call_sig.args

@njit(boolean(i8[::1],) ,cache=True)
def match_head_ptrs(ptrs):
{indent(nl.join([f'i{i} = _load_pointer(h{i}_type,ptrs[{i}])' for i in range(len(arg_types))]),prefix='    ')}
    return match_head({",".join([f'i{i}' for i in range(len(arg_types))])})



field_dict = {{**op_fields_dict,**{{f"arg{{i}}" : t for i,t in enumerate(call_sig.args)}}}}
{cls.__name__+'Type'} = OpTypeTemplate([(k,v) for k,v in field_dict.items()]) 
'''
    return source

def gen_call_intrinsic_source(return_type,n_args):
    source = f''' 
@intrinsic
def call_intrinsic(typingctx, {[f"a{i}" for i in range(n_args)]}):
    sig = 
    return 
'''
    return source

from cre.var import GenericVarType

_head_range_type = np.dtype([('start', np.uint8), ('length', np.uint8)])
head_range_type = numba.from_dtype(_head_range_type)

op_fields_dict = {
    "name" : unicode_type,
    "expr_template" : unicode_type,
    "shorthand_template" : unicode_type,

    # Mapping variable base_ptrs to aliases
    "base_var_map" : DictType(i8, unicode_type),
    # Inverse of 'base_var_map'
    "inv_base_var_map" : DictType(unicode_type, i8),
    # Pointers of op's head vars (i.e. x.nxt and x.nxt.nxt are both heads of x)
    "head_var_ptrs" : i8[::1],
    "head_ranges" : head_range_type[::1],

    "return_type_name" : unicode_type,
    "arg_type_names" : ListType(unicode_type),
    "call_addr" : i8,
    "call_multi_addr" : i8,
    "check_addr" : i8,
    "match_head_ptrs_addr" : i8,
    "is_ptr_op" : types.boolean,
    
    # "arg_types" : types.Any,
    # "out_type" : types.Any,
    # "is_const" : i8[::1]
}

class OpTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


@register_default(OpTypeTemplate)
class OpModel(models.StructModel):
    """Model for cre.Op. A reference to the structref payload, and the Op class.
    """
    def __init__(self, dmm, fe_typ):
        dtype = fe_typ.get_data_type()
        members = [
            ("meminfo", types.MemInfoPointer(dtype)),
            ("py_class", types.pyobject),
        ]
        super().__init__(dmm, fe_typ, members)


# @structref.register

default_manager.register(OpTypeTemplate, OpModel)
define_attributes(OpTypeTemplate)


def op_define_boxing(struct_type, obj_class):
    """
        Like structref.define_boxing but adds in __class__ to the datamodel
    """
    if struct_type is types.StructRef:
        raise ValueError(f"cannot register {types.StructRef}")

    obj_ctor = obj_class._numba_box_

    @box(struct_type)
    def box_op(typ, val, c):
        ''' NRT -> Python '''
        struct_ref = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        meminfo = struct_ref.meminfo
        py_class = struct_ref.py_class

        mip_type = types.MemInfoPointer(types.voidptr)
        boxed_meminfo = c.box(mip_type, meminfo)
        boxed_py_class = c.box(types.pyobject, py_class)

        ctor_pyfunc = c.pyapi.unserialize(c.pyapi.serialize_object(obj_ctor))
        ty_pyobj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))

        res = c.pyapi.call_function_objargs(
            ctor_pyfunc, [ty_pyobj, boxed_meminfo, boxed_py_class],
        )
        c.pyapi.decref(ctor_pyfunc)
        c.pyapi.decref(ty_pyobj)
        c.pyapi.decref(boxed_meminfo)
        c.pyapi.decref(boxed_py_class)
        return res

    @unbox(struct_type)
    def unbox_op(typ, obj, c):
        ''' Python -> NRT '''
        mi_obj = c.pyapi.object_getattr_string(obj, "_meminfo")
        op_class_obj = c.pyapi.object_getattr_string(obj, "__class__")

        mip_type = types.MemInfoPointer(types.voidptr)
        mi = c.unbox(mip_type, mi_obj).value

        op_class = c.unbox(types.pyobject, op_class_obj).value

        struct_ref = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        struct_ref.meminfo = mi
        struct_ref.py_class = op_class

        out = struct_ref._getvalue()

        c.pyapi.decref(mi_obj)
        # c.pyapi.decref(op_class_obj) <- Don't decref this
        return NativeValue(out)


GenericOpType = OpTypeTemplate([(k,v) for k,v in op_fields_dict.items()])


class UntypedOp():
    '''An Op that has not been given a signature yet'''
    def __init__(self,name,members):
        self.name = name
        self.members = members
        self.members['call'] = njit(cache=True)(self.members['call'])
        self._specialize_cache = {}
    @property
    def arg_names(self):
        if(not hasattr(self,"_arg_names")):
            f = self.members['call']
            py_func = f.py_func if isinstance(f, Dispatcher) else f
            self._arg_names = inspect.getfullargspec(f)[0]
        return self._arg_names

    def __repr__(self):
        return f'UntypedOp(name={self.name}, members={self.members})'

    def __str__(self):
        return f'{self.name}({", ".join(self.arg_names)})'


    def __call__(self,*py_args):

        # with PrintElapse("  get types"):
        # Fill types and check for all_const/not_var 
        arg_types = [] 
        all_const = True
        any_not_var = False
        for x in py_args:
            arg_types.append(resolve_return_type(x))
            if(isinstance(x,(OpMeta,OpComp, Op)) or
               (isinstance(x, Var) and len(x.deref_offsets) > 0)):
                all_const = False
                any_not_var = True
            elif(isinstance(x,(Var))):
                all_const = False
            else:
                any_not_var = True

        call = self.members['call']
        if(all_const):
            return call(*py_args)


        cres = call.overloads.get(tuple(arg_types))
        if(cres is not None):
            sig = cres.signature
        else:
            # Note: get_call_template is an internal method of numba.Dispatcher 
            (template,*rest) = call.get_call_template(arg_types,{})
            sig = template.cases[0]
        # self.members['signature'] = sig
        members = {**self.members, 'signature': sig}

        if(any_not_var):
            op = new_op(self.name, members)
            return OpComp(op, *py_args).flatten()
        else:
            var_ptrs = np.array([v.get_ptr_incref() for v in py_args],dtype=np.int64)
            # Retreive from Cache/Make typed version 
            if(sig not in self._specialize_cache):
                op_cls = new_op(self.name, members,return_class=True)
                self._specialize_cache[sig] = op_cls
            op_cls = self._specialize_cache[sig]
            return op_cls.make_singleton_inst(head_var_ptrs=var_ptrs)


def resolve_return_type(x):
    if(isinstance(x,Var)):
        return x.head_type
    elif(isinstance(x,Op)):
        return x.signature.return_type
    elif(isinstance(x, OpComp)):
        return x.op.signature.return_type
    elif(isinstance(x, UntypedOp)):
        raise ValueError(f"Cannot resolve return type of UntypedOp {x}")
    elif(isinstance(x, (float,int))):
        return types.float64
    elif(isinstance(x, (str))):
        return types.unicode_type

        



@njit(cache=True)
def var_to_ptr(x):
    return _pointer_from_struct_incref(x)

def new_var_ptrs_from_types(types, names):
    '''Generate vars with 'types' and 'names'
        and output a ptr_array of their memory addresses.'''
    #TODO: Op should have deconstructor so these are decrefed.
    ptrs = np.empty(len(types), dtype=np.int64)
    for i, t in enumerate(types):
        ptrs[i] = var_to_ptr(Var(t,names[i]))
    return ptrs

@njit(cache=True)
def gen_placeholder_aliases(var_ptrs):
    _vars = Dict.empty(i8, GenericVarType)
    temp_inv_base_var_map = Dict.empty(unicode_type, i8)
    
    n_auto_gen = 0
    for var_ptr in var_ptrs:
        head_var = _struct_from_pointer(GenericVarType, var_ptr)
        base_ptr = head_var.base_ptr
        if(base_ptr not in _vars):
            base_var = _struct_from_pointer(GenericVarType, base_ptr)
            _vars[base_ptr] = base_var
            
    for var in _vars.values():
        if(var.alias != ""):
            if(var.alias not in temp_inv_base_var_map):
                temp_inv_base_var_map[var.alias] = var_ptr
            else:
                raise NameError("Duplicate Var alias.")
        else:
            n_auto_gen += 1
        
    generated_aliases = List.empty_list(unicode_type)
    ascii_code = 97# i.e. 'a'
    bad_aliases = ['i','j','k','o','I','J','K','O']
    for i in range(n_auto_gen):
        alias = chr(ascii_code)
        while(alias in temp_inv_base_var_map or alias in bad_aliases):
            ascii_code += 1
            alias = chr(ascii_code)
        generated_aliases.append(alias)
        ascii_code += 1

    return generated_aliases


@njit(cache=True)
def make_head_ranges(n_args,head_var_ptrs):
    head_ranges = np.empty((n_args,),
                            dtype=head_range_type)
    start, length = 0,0
    prev_base_ptr = 0
    k = 0 
    for ptr in head_var_ptrs:
        head_var = _struct_from_pointer(GenericVarType, ptr)
        base_ptr = head_var.base_ptr
        if(prev_base_ptr == 0 or base_ptr == prev_base_ptr):
            length += 1
        else:
            head_ranges[k].start = start 
            head_ranges[k].length = length
            k += 1
            start += length; length = 1
            
        prev_base_ptr = base_ptr
    head_ranges[k][0] = start 
    head_ranges[k][1] = length
    return head_ranges


@njit(cache=True)
def op_ctor(name, return_type_name, arg_type_names, head_var_ptrs, 
            expr_template="MOOSE", shorthand_template="ROOF",
            call_addr=0, call_multi_addr=0, check_addr=0, match_head_ptrs_addr=0, is_ptr_op=False):
    '''The constructor for an Op instance that can be passed to the numba runtime'''
    st = new(GenericOpType)
    st.name = name
    st.return_type_name = return_type_name

    st.arg_type_names = arg_type_names
    st.base_var_map = Dict.empty(i8, unicode_type)
    st.inv_base_var_map = Dict.empty(unicode_type, i8)

    generated_aliases = gen_placeholder_aliases(head_var_ptrs)
    j = 0
    for head_ptr in head_var_ptrs:
        head_var = _struct_from_pointer(GenericVarType, head_ptr)
        base_ptr = head_var.base_ptr
        if(base_ptr not in st.base_var_map):
            base_var = _struct_from_pointer(GenericVarType, base_ptr)
            alias = base_var.alias
            if(alias == ""):
                alias = generated_aliases[j]; j+=1;
            
            st.base_var_map[base_ptr] = alias
            st.inv_base_var_map[alias] = base_ptr

    st.head_var_ptrs = head_var_ptrs
    st.head_ranges = make_head_ranges(len(st.base_var_map), head_var_ptrs)

    st.expr_template = expr_template
    st.shorthand_template = shorthand_template
            
    st.call_addr = call_addr
    st.call_multi_addr = call_multi_addr
    st.check_addr = check_addr
    st.match_head_ptrs_addr = match_head_ptrs_addr
    st.is_ptr_op = is_ptr_op

    return st



def new_op(name, members, var_ptrs=None, return_class=False):
    '''Creates a new custom Op with 'name' and 'members' '''
    has_check = 'check' in members
    assert 'call' in members, "Op must have call() defined"
    assert hasattr(members['call'], '__call__'), "call() must be a function"
    # assert 'signature' in members, "Op must have signature"
    assert not (has_check and not hasattr(members['check'], '__call__')), "check() must be a function"

    if('signature' not in members):
        return UntypedOp(name,members)
    
    # with PrintElapse("\thandling"):
    # See if call/check etc. are raw python functions and need to be wrapped in @jit.
    call_needs_jitting = not isinstance(members['call'], Dispatcher)
    check_needs_jitting = has_check and (not isinstance(members['check'],Dispatcher))
    
    # 'cls' is the class of the user defined Op (i.e. Add(a,b)).
    cls = type.__new__(OpMeta,name,(Op,),members) 
    cls.__module__ = members["call"].__module__
    # cls.__call__ = call_op
    cls._handle_commutes()
    cls._handle_nopython()

    # with PrintElapse("\tprocess_method"):
    cls.process_method("call", cls.signature)
    
    if(has_check):
        cls.process_method("check", u1(*cls.signature.args))

    cls._handle_defaults()  
    # cls._handle_default_templates()
    
    # If either call or check needs to be jitted then generate source code
    #  to do the jitting and put it in the cache. Or retrieve if already
    #  defined.  This way jit(cache=True) can reliably re-retreive.
    # with PrintElapse("\tget_src"):
    # if(call_needs_jitting or check_needs_jitting):
    name = cls.__name__

    #Triggers getter
    long_hash = cls.long_hash 
    if(not source_in_cache(name, long_hash) or getattr(cls,'cache',True) == False):
        source = gen_op_source(cls, call_needs_jitting, check_needs_jitting)
        source_to_cache(name,long_hash,source)

    # to_import = ['call','call_addr'] if call_needs_jitting else []
    to_import = ['match_head_ptrs']
    if(call_needs_jitting): to_import += ['call','call_addr']
    if(check_needs_jitting): to_import += ['check', 'check_addr']
    l = import_from_cached(name, long_hash, to_import)
    for key, value in l.items():
        setattr(cls, key, value)

    # with PrintElapse("\tgwap"):
    # Make static so that self isn't the first argument for call/check.
    cls.call = staticmethod(cls.call)
    if(has_check): cls.check = staticmethod(cls.check)

    # Get store the addresses for call/check
    if(not call_needs_jitting): 
        cls.call_sig = cls.signature
        cls.call_addr = _get_wrapper_address(cls.call,cls.call_sig)

    if(not check_needs_jitting and has_check):
        # cls.check_sig = u1(*cls.signature.args)
        cls.check_addr = _get_wrapper_address(cls.check,cls.check_sig)

    # print(cls.match_head_ptrs.overloads.keys())
    cls.match_head_ptrs_addr = _get_wrapper_address(cls.match_head_ptrs,boolean(i8[::1],))

    # Standardize shorthand definitions
    if(hasattr(cls,'shorthand') and 
        not isinstance(cls.shorthand,dict)):
        cls.shorthand = {'*' : cls.shorthand}

    # with PrintElapse("\tMakeInst"):
    if(cls.__name__ != "__GenerateOp__" and not return_class):
        op_inst = cls.make_singleton_inst(head_var_ptrs=var_ptrs)
        return op_inst

    return cls



# def call_op(self,*py_args):
#     if(all([not isinstance(x,(Var,OpMeta,OpComp, Op)) for x in py_args])):
#         # If all of the arguments are constants then just call the Op
#         return self.call(*py_args)
#     else:
#         # Otherwise build an OpComp and flatten it into a new Op
#         op_comp = OpComp(self,*py_args)
#         op = op_comp.flatten()
        
#         return op

        
class OpMeta(type):
    ''' A the metaclass for op. Useful for singleton generation.
    '''
    # def __repr__(cls):
    #     return cls.__name__ + f'({",".join(["?"] * len(cls.signature.args))})'

    # This is similar to defining __init_subclass__ in Op inside except we can
    #    return whatever we want. In this case we return a singleton instance.
    def __new__(meta_cls, *args):
        name, members = args[0], args[2]
        if(name == "Op"):
            ret = super().__new__(meta_cls,*args)
            print(ret)
            return ret
        # print("<<<<", [str(x.__name__) for x in args[1]])
        # members = 
        return new_op(name, members)


    def __call__(self,*args, **kwargs):
        ''' A decorator function that builds a new Op'''
        if(len(args) > 1): raise ValueError("Op() takes at most one position argument 'signature'.")
        
        def wrapper(call_func):
            assert hasattr(call_func,"__call__")
            name = call_func.__name__
            members = kwargs
            members["call"] = call_func
            return new_op(name, members)

        if(len(args) == 1):
            if(isinstance(args[0],(str, numba.core.typing.templates.Signature))):
                kwargs['signature'] = args[0]
            elif(hasattr(args[0],'__call__')):
                return wrapper(args[0])
            else:
                raise ValueError(f"Unrecognized type {type(args[0])} for 'signature'")        

        return wrapper

    def _get_simple_sig_str(cls):
        sig = getattr(cls,'signature',None)
        if(sig is None): return "None"
        return f'{str(sig.return_type)}({",".join([str(x) for x in sig.args])})'

    def __str__(cls):
        return f"cre.{cls.__name__}(signature={cls._get_simple_sig_str()})"

    def __repr__(cls):
        return f"cre.OpMeta(name={cls.__name__!r}, signature={cls._get_simple_sig_str()})"
        

    def _handle_nopython(cls):
        if(not hasattr(cls,'nopython_call')):
            cls.nopython_call = getattr(cls,'nopython',None)
        if(not hasattr(cls,'nopython_check')):
            cls.nopython_check = getattr(cls,'nopython',None)

    def _handle_commutes(cls):
        if(not hasattr(cls,'commutes')):
            cls.commutes = []
            cls.right_commutes = {}
            return

        arg_types = cls.signature.args
        if(isinstance(cls.commutes,bool)):
            if(cls.commutes == True):
                cls.commutes, d = [], {}
                for i, typ in enumerate(arg_types): 
                    d[typ] = d.get(typ,[]) + [i] 
                for typ, inds in d.items():
                    cls.commutes += [inds]#list(combinations(inds,2))
            else:
                cls.commutes = []
        else:
            assert(isinstance(cls.commutes,Iterable))


        right_commutes = {}#Dict.empty(i8,i8[::1])
        for i in range(len(cls.commutes)):
            commuting_set =  cls.commutes[i]
            # print(commuting_set)
            for j in range(len(commuting_set)-1,0,-1):
                right_commutes[commuting_set[j]] = np.array(commuting_set[0:j],dtype=np.int64)
                for k in commuting_set[0:j]:
                    typ1, typ2 = arg_types[k], arg_types[commuting_set[j]]
                    assert typ1==typ2, \
            f"cre.Op {cls.__name__!r} has invalid 'commutes' member {cls.commutes}. Signature arguments {k} and {commuting_set[j]} have different types {typ1} and {typ2}."
        cls.right_commutes = right_commutes
        

    def _handle_defaults(cls):
        # By default use the variable names that the user defined in the call() fn.
        cls.default_arg_names = inspect.getfullargspec(cls.call_pyfunc)[0]
        cls.default_var_ptrs = new_var_ptrs_from_types(cls.call_sig.args, cls.default_arg_names)

        cls.arg_type_names = as_typed_list(unicode_type,
                            [str(x) for x in cls.signature.args])

        if(not hasattr(cls,'_expr_template')):
            cls._expr_template = f"{cls.__name__}({','.join([f'{{{i}}}' for i in range(len(cls.arg_type_names))])})"  
        
        cls._shorthand_template = cls.shorthand if(hasattr(cls,'shorthand')) else cls._expr_template

        
        


    def process_method(cls,name,sig):
        if(not hasattr(cls, name+"_pyfunc")):
            func = getattr(cls,name)
            if(isinstance(func, Dispatcher)):
                py_func = func.py_func
            else:
                py_func = func

            setattr(cls, name+"_pyfunc", py_func)
            setattr(cls, name+"_sig", sig)
            
            # Speed up by caching cloudpickle.dumps(py_func) inside py_func object
            if(hasattr(py_func,'_cre_cloudpickle_bytes')):
                setattr(cls, name+'_bytes', py_func._cre_cloudpickle_bytes)
            else:
                # print("<<", py_func)
                # print("<<", py_func.__dict__)
                # print(py_func.__loader__)
                cloudpickle_bytes = cloudpickle.dumps(py_func)
                # print(cloudpickle_bytes)
                setattr(cls, name+'_bytes', cloudpickle_bytes)
                py_func._cre_cloudpickle_bytes = cloudpickle_bytes


    @property
    def long_hash(cls):
        if(not hasattr(cls,'_long_hash')):
            long_hash = unique_hash([cls.signature, cls.call_bytes,
                    cls.check_bytes if hasattr(cls,'check') else None])

            cls._long_hash = long_hash
        return cls._long_hash



class Op(structref.StructRefProxy,metaclass=OpMeta):
    ''' Base class for an functional operation. Custom operations can be
        created by subclassing Op and defining a signature, call(), and 
        optional check() method. 
    '''
    # @classmethod
    

    @classmethod
    def _numba_box_(cls, ty, mi, py_cls=None):
        # print(ty, mi, py_cls)
        # instance = structref.StructRefProxy._numba_box_(cls, ty, mi)
        instance = super(structref.StructRefProxy,cls).__new__(cls)
        instance._type = ty
        instance._meminfo = mi
        if(isinstance(py_cls,OpMeta)):# is not None):
            instance.__class__ = py_cls# is not None):



        # else:
        #     raise ValueError()

        # print("CLASS:", py_cls)
        # print("<< ", type(instance))
        return instance

    def get_var(self,i):
        return get_var(self, i)

    def as_op_comp(self):
        if(hasattr(self,'op_comp')):
            return self.op_comp            
        return OpComp(self,*[self.get_var(i) for i in range(len(self.signature.args))])

    
    #     pass
    # def __call__(self,*py_args):
        
    #     if(all([not isinstance(x,(Var,OpMeta,OpComp, Op)) for x in py_args])):
    #         # If all of the arguments are constants then just call the Op
    #         return self.call(*py_args)
    #     else:
    #         # Otherwise build an OpComp and flatten it into a new Op
    #         op_comp = OpComp(self,*py_args)
    #         op = op_comp.flatten()
    #         op.op_comp = op_comp
    #         set_expr_template(op, op.gen_expr())
    #         set_shorthand_template(op, op.gen_expr(use_shorthand=True))
    #         return op

    def __call__(self,*py_args):
        if(all([not isinstance(x,(Var,OpMeta,OpComp, Op)) for x in py_args])):
            # If all of the arguments are constants then just call the Op
            return self.call(*py_args)
        else:
            # Otherwise build an OpComp and flatten it into a new Op
            op_comp = OpComp(self,*py_args)
            op = op_comp.flatten()
            
            return op

    def __str__(self):
        return op_str(self)
        # name = get_name(self)
        # arg_names = op_get_arg_names(self)
        # return self.shorthand_template % (arg_names)
        # if(name == "__GenerateOp__"):
        #     print("THIS!!!!!", self.expr_template)
        #     return self.expr_template
        # else:
        #     return op_str(self)

    def __repr__(self):
        return op_repr(self)
        # name = get_name(self)
        # arg_names = op_get_arg_names(self)
        # return self.expr_template % (arg_names)
        # if(name == "__GenerateOp__"):
        #     return self.expr_template
        # else:
        #     return op_repr(self)

    @classmethod
    def make_singleton_inst(cls,head_var_ptrs=None):
        '''Creates a singleton instance of the user defined subclass of Op.
            These instances unbox into the NRT as GenericOpType.
        '''        
        if(head_var_ptrs is None): head_var_ptrs = cls.default_var_ptrs
        # print("---------------")
        # print(cls.__name__,
        #     str(cls.signature.return_type),
        #     cls.arg_type_names,
        #     var_ptrs,
        #     cls._expr_template,
        #     cls._shorthand_template,
        #     cls.call_addr,
        #     cls.__dict__.get('check_addr',0),)
        # Make the op instance
        op_inst = op_ctor(
            cls.__name__,
            str(cls.signature.return_type),
            cls.arg_type_names,
            head_var_ptrs,
            str(cls._expr_template),
            str(cls._shorthand_template),
            call_addr=cls.call_addr,
            check_addr=cls.__dict__.get('check_addr',0),
            match_head_ptrs_addr=cls.match_head_ptrs_addr,
            )
        
        op_inst.__class__ = cls
        # set_expr_template(op_inst, cls._expr_template)
        # set_shorthand_template(op_inst, cls._shorthand_template)
        

        # In the NRT the instance will be a GenericOpType, but on
        #   the python side we need it to be its own custom class
        

        # template_placeholders = [f'{{{i}}}' for i in range(len(arg_names))]

        #Make and store repr and shorthand expressions
        # expr_template = op_inst.gen_expr(
        #     arg_names=template_placeholders,
        #     use_shorthand=False,
        # )
        # set_expr_template(op_inst, expr_template)
        # if(hasattr(op_inst,"shorthand")):
        #     set_shorthand_template(op_inst, op_inst.gen_expr(
        #         arg_names=template_placeholders,
        #         use_shorthand=True,
        #     ))
        # else:
        #     set_shorthand_template(op_inst, expr_template)

        return op_inst

    # def recover_singleton_inst(self,context=None):
    #     '''Sometimes numba needs to emit an op instance 
    #      that isn't the singleton instance. This recovers
    #      the singleton instance from the cre_context'''
    #     context = cre_context(context)
    #     return context.op_instances[self.name]



    @property
    def name(self):
        return get_name(self)

    @property
    def expr_template(self):
        return get_expr_template(self)

    @property
    def shorthand_template(self):
        return get_shorthand_template(self)

    @property
    def base_var_map(self):
        return get_base_var_map(self)

    @property
    def head_var_ptrs(self):
        return get_head_var_ptrs(self)

    @property
    def head_ranges(self):
        return get_head_ranges(self)

    @property
    def return_type_name(self):
        return get_return_type_name(self)


    @property
    def arg_names(self):
        if(not hasattr(self,'_arg_names')):
            self._arg_names = extract_arg_names(self)
        return self._arg_names

    @property
    def long_hash(self):
        return self.__class__.long_hash


    @property
    def unq(self):
        '''Returns a unique tuple for the the Op, useful for use
           in dictionary keys since the Op overloads __eq__ and thus
           cannot be a dictionary key '''
        if(not hasattr(self,'_unq')):
            self._unq = (self.name, self.long_hash)
        return self._unq


    def __hash__(self):
        return hash(self.long_hash)

    def __lt__(self, other): 
        from cre.default_ops import LessThan
        return LessThan(self, other)
    def __le__(self, other): 
        from cre.default_ops import LessThanEq
        return LessThanEq(self, other)
    def __gt__(self, other): 
        from cre.default_ops import GreaterThan
        return GreaterThan(self, other)
    def __ge__(self, other):
        from cre.default_ops import GreaterThanEq
        return GreaterThanEq(self, other)
    def __eq__(self, other): 
        from cre.default_ops import Equals
        return Equals(self, other)
    def __ne__(self, other): 
        from cre.default_ops import Equals
        return ~Equals(self, other)

    def __add__(self, other):
        from cre.default_ops import Add
        return Add(self, other)

    def __radd__(self, other):
        from cre.default_ops import Add
        return Add(other, self)

    def __sub__(self, other):
        from cre.default_ops import Subtract
        return Subtract(self, other)

    def __rsub__(self, other):
        from cre.default_ops import Subtract
        return Subtract(other, self)

    def __mul__(self, other):
        from cre.default_ops import Multiply
        return Multiply(self, other)

    def __rmul__(self, other):
        from cre.default_ops import Multiply
        return Multiply(other, self)

    def __truediv__(self, other):
        from cre.default_ops import Divide
        return Divide(self, other)

    def __rtruediv__(self, other):
        from cre.default_ops import Divide
        return Divide(other, self)

    def __floordiv__(self, other):
        from cre.default_ops import FloorDivide
        return FloorDivide(self, other)

    def __rfloordiv__(self, other):
        from cre.default_ops import FloorDivide
        return FloorDivide(other, self)

    def __pow__(self, other):
        from cre.default_ops import Power
        return Power(self, other)

    def __rpow__(self, other):
        from cre.default_ops import Power
        return Power(other, self)

    def __and__(self,other):
        from cre.conditions import op_to_cond, conditions_and
        self = op_to_cond(self)
        if(isinstance(other, Op)): other = op_to_cond(other)
        # print("<<", type(self), type(other))
        return conditions_and(self, other)

    def __or__(self,other):
        from cre.conditions import op_to_cond, conditions_or
        self = op_to_cond(self)
        if(isinstance(other, Op)): other = op_to_cond(other)
        return conditions_or(self, other)

    def __invert__(self):
        from cre.conditions import literal_ctor, literal_to_cond, literal_not
        return literal_to_cond(literal_not(literal_ctor(self)))

    
             

    def gen_expr(self, lang='python',
         arg_names=None, use_shorthand=False, **kwargs):
        '''Generates a one line expression for this Op (which might have been built
            from a composition of ops) in terms of user defined Ops.
            E.g. Multiply(Add(x,y),2).gen_expr() -> "Multiply(Add(x,y),2)"
                 or if shorthands are defined -> "((x+y)*2)" '''
        if(arg_names is None):
            arg_names = get_arg_seq(self).split(",") 
        if(hasattr(self,'op_comp')):
            return self.op_comp.gen_expr(
                lang=lang,
                arg_names=arg_names,
                use_shorthand=use_shorthand,
                **kwargs)
        else:
            if( use_shorthand and 
                hasattr(self,'shorthand')):
                template = resolve_template(lang,self.shorthand,'shorthand')
                return template.format(*arg_names)
            else:
                return f"{self.name}({','.join(arg_names)})"

    # Make Source:
    @make_source('*')
    def mk_src(self, lang='python', ind='   ', **kwargs):
        body = ''
        call = self.make_source(lang,'call')
        body += indent(self.make_source(lang,'call'),prefix=ind)
        pc = self.make_source(lang,'parent_class')
        return gen_def_class(lang, get_name(self), body, parent_classes=pc, inst_self=True)

    # Make Source: parent_class
    @make_source('python','parent_class')
    def mk_src_py_parent_class(self, **kwargs):
        return "cre.Op"
    @make_source('js','parent_class')
    def mk_src_js_parent_class(self, **kwargs):
        return "Callable"
    @make_source('*','parent_class')
    def mk_src_any_parent_class(self, **kwargs):
        return ""

    # Make Source: call
    @make_source("python", 'call')
    def mk_src_py_call(self, **kwargs):
        return dedent(inspect.getsource(self.call.py_func))

    @make_source("*", 'call')
    def mk_src_any_call(self, lang='*', ind='    ', **kwargs):
        try:
            temp = resolve_template(lang, self.call_body, 'call_body')
            call_body = f'{temp.format(*self.arg_names,ind=ind)}\n'
        except Exception as e:
            temp = resolve_template(lang, self.shorthand, 'shorthand')
            call_body = f'{ind}return {temp.format(*self.arg_names)}\n'

        return gen_def_func(lang,'call',", ".join(self.arg_names), call_body)




@njit(cache=True)
def extract_arg_names(op):
    out = List.empty_list(unicode_type)
    for v_ptr in op.base_var_map:
        v =_struct_from_pointer(GenericVarType, v_ptr)
        out.append(v.alias)
    return out






### Various jitted getters ###

@njit(cache=True)
def get_name(self):
    return self.name

@njit(cache=True)
def get_expr_template(self):
    return self.expr_template

@njit(cache=True)
def get_shorthand_template(self):
    return self.shorthand_template

@njit(cache=True)
def get_base_var_map(self):
    return self.base_var_map

@njit(cache=True)
def get_head_var_ptrs(self):
    return self.head_var_ptrs

@njit(cache=True)
def get_head_ranges(self):
    return self.head_ranges

@njit(cache=True)
def get_var(self,i):
    v_ptr = List(self.base_var_map.keys())[i]
    return _struct_from_pointer(GenericVarType,v_ptr)


@njit(cache=True)
def get_return_type_name(self):
    return self.return_type_name

# @njit(cache=True)
# def op_str(self):
#     return self.name + "(" + get_arg_seq(self) + ")"

@njit(cache=True)
def op_str(self):
    s = self.shorthand_template
    arg_names = op_get_arg_names(self)
    for i,arg_name in enumerate(arg_names):
        s =  s.replace(f'{{{i}}}',arg_name)
    return s

# @njit(cache=True)
# def op_repr(self):
#     return self.name + "(" + get_arg_seq(self,type_annotations=True) + ")"

@njit(cache=True)
def op_repr(self):
    s = self.expr_template
    arg_names = op_get_arg_names(self)
    for i,arg_name in enumerate(arg_names):
        s =  s.replace(f'{{{i}}}',f'[{arg_name}:{self.arg_type_names[i]}]')

    #Add spaces after commas
    s = s.replace(", ", ",")
    s = s.replace(",", ", ")
    return s

@njit(cache=True)
def get_arg_seq(self,type_annotations=False):
    s = ""
    for i,alias in enumerate(self.inv_base_var_map):
        s += alias
        if(type_annotations): 
            v = _struct_from_pointer(GenericVarType, self.inv_base_var_map[alias])
            s += ":" + v.base_type_name

        if(i < len(self.inv_base_var_map)-1): 
            s += "," + (" " if type_annotations else "")
    return s

@njit(cache=True)
def op_get_arg_names(self):
    arg_names = List.empty_list(unicode_type)
    for alias in self.inv_base_var_map:
        arg_names.append(alias)
    return arg_names

#### Various Setters ####
@njit(cache=True)
def set_expr_template(self, expr):
    self.expr_template = expr

@njit(cache=True)
def set_shorthand_template(self, expr):
    self.shorthand_template = expr




op_define_boxing(OpTypeTemplate,Op)   

def g_nm(x,names):
    '''Helper function that helps with putting instances of Var
        as dictionary keys by replacing them with their NRT pointers.
        This must be done because __eq__ is not a comparator on Var.'''
    if(isinstance(x, Var)):
        return names.get(x.base_ptr,None)
    else:
        return names.get(x,None)

class DerefInstr():
    '''A placeholder instruction for dereferencing'''
    def __init__(self, var):
        self.var = var
        self.deref_attrs = var.deref_attrs

    @property
    def template(self):
        return f'{{}}.{".".join(list(iter_typed_list(self.deref_attrs)))}'

    def _get_hashable(self):
        if(not hasattr(self,"_hashable")):
            self._hashable = (self.var.base_ptr, tuple(iter_typed_list(self.deref_attrs)))
        return self._hashable

    def __eq__(self, other):
        return self._get_hashable() == other._get_hashable()

    def __hash__(self):
        return hash(self._get_hashable())


# class InstrProxy():
#     '''A proxy class to wrap an Op or OpComp that safely implements 
#         __eq__ and __hash__'''
#     def __init__(self,op_comp):
#         if(isinstance(op_comp, Op)):
#             op_comp = op.as_op_comp()
#         self.op_comp = op_comp
#         self._hash = hash(self.op_comp.used_ops)

#     def __str__(self):
#         return str(self.op_comp)

#     def __repr__(self):
#         return f"InstrProxy({self.op_comp})"

#     def __eq__(self):

# @njit(cache=True)
# def var_from_ptr(ptr):
#     return _struct_from_pointer(GenericVarType,ptr)

# def extract_head_ptr(x):
#     if(isinstance(x,DerefInstr)):
#         return x.var.get_ptr_incref()
#     return x


# def gen_ranges(head_vars):
#     start,n = 0,0
#     prev_arg_ind = 0
#     for (head_inst, arg_ind, head_typ) in head_vars.values():
#         if(arg_ind == tup):
#             n += 1
#         else:
#             start += n
#             n = 0
#             yield (start, n)

            



class OpComp():
    '''A helper class representing a composition of operations
        that can be flattened into a new Op definition.'''

    def _repr_arg_helper(self,x):
        if isinstance(x, Var):
            return f'{{{self.base_vars[x.base_ptr][1]}}}'
        elif(isinstance(x, OpComp)):
            arg_names = [f'{{{self.base_vars[v_p][1]}}}' for v_p in x.base_vars]
            # print("arg_names", arg_names)
            return x.gen_expr('python',arg_names=arg_names)
        elif(isinstance(x,DerefInstr)):
            deref_attrs = list(iter_typed_list(x.deref_attrs))
            return f'{{{self.base_vars[x.var.base_ptr][1]}}}.{".".join(deref_attrs)}'
        else:
            return repr(x) 


    def __init__(self,op,*py_args):
        '''Constructs the the core pieces of an OpComp from a set of arguments.
            These pieces are the outermost 'op', and a set of 'constants',
            'vars', and 'instructions' (i.e. other op_comps)'''
        base_vars = {} #base_var_ptr -> (var_inst, arg_index, base_type)
        head_vars = {} #head_var_ptr or DerefInstr -> (var_inst, arg_index, head_type)

        # head_var_ptrs = []
        constants = {}
        instructions = {}
        args = []
        arg_types = []

        
        

        for i, x in enumerate(py_args):
            if(isinstance(x, Op)): x = x.as_op_comp()
            if(isinstance(x, OpComp)):
                for v_ptr,(v,_,t) in x.base_vars.items():
                    if(v_ptr not in base_vars):
                        base_vars[v_ptr] = (v,len(arg_types),t)
                        arg_types.append(t)

                for c,t in x.constants.items():
                    constants[c] = t
                for instr, sig in x.instructions.items():
                    instructions[instr] = sig
                for vptr_or_instr, (head_var,_,typ) in x.head_vars.items():
                    if(vptr_or_instr not in head_vars):
                        arg_ind = base_vars[head_var.base_ptr][1]
                        head_vars[vptr_or_instr] = (head_var, arg_ind, typ)
                        # head_var_ptrs.append(extract_head_ptr(vptr_or_instr))
            else:
                if(isinstance(x,Var)):
                    b_ptr = x.base_ptr
                    if(b_ptr not in base_vars):
                        t = x.base_type
                        base_vars[b_ptr] = (x,len(arg_types),t)
                        arg_types.append(t)
                    
                    if(x.base_type != x.head_type):
                        d_instr = DerefInstr(x)
                        if(d_instr not in head_vars): 
                            arg_ind = base_vars[b_ptr][1]
                            head_vars[d_instr] = (x, arg_ind, x.head_type)

                            # head_var_ptrs.append(x.get_ptr())
                        instructions[d_instr] = x.head_type(x.base_type,)
                        x = d_instr
                    else:
                        if(b_ptr not in head_vars):
                            arg_ind = base_vars[b_ptr][1]
                            head_vars[b_ptr] =  (x, arg_ind ,x.head_type)
                            # head_var_ptrs.append(x.get_ptr())
                else:
                    constants[x] = op.signature.args[i]
            args.append(x)

        head_vars = {k:v for i,(k,v) in enumerate(sorted(head_vars.items(), key=lambda x: x[1][1]))}
        # print("<<",_vars)
        self.op = op
        self.base_vars = base_vars
        self.head_vars = head_vars
        self.args = args
        self.arg_types = arg_types
        # self.head_types = head_types
        # print([x[0].get_ptr() for x in head_vars.values()])
        self.head_var_ptrs = np.array([x[0].get_ptr_incref() for x in head_vars.values()],dtype=np.int64)
        self.constants = constants
        self.signature = op.signature.return_type(*arg_types)
        

        self.expr_template = f"{op.name}({', '.join([self._repr_arg_helper(x) for x in self.args])})"  
        self.name = self.expr_template

        instructions[self] = op.signature

        
        self.instructions = instructions

    def flatten(self):
        ''' Flattens the OpComp into a single Op. Generates the source
             for the new Op as needed.'''
        if(not hasattr(self,'_generate_op')):
            # print([type(x) for x in self.used_ops])
            long_hash = unique_hash([self.expr_template,*[op.unq for _,op in self.used_ops.values()]])
            if(not source_in_cache('__GenerateOp__', long_hash)):
                source = self.gen_flattened_op_src(long_hash)
                source_to_cache('__GenerateOp__', long_hash, source)
            l = import_from_cached('__GenerateOp__', long_hash, ['__GenerateOp__'])
            op_cls = self._generate_op_cls = l['__GenerateOp__']
            # print("<<",type(op_cls))
            print(get_cache_path('__GenerateOp__',long_hash))

            # var_ptrs = np.empty(len(self.base_vars),dtype=np.int64)
            # for i,v_p in enumerate(self.base_vars):
            #     var_ptrs[i] = v_p#v.get_ptr()

            op = self._generate_op = op_cls.make_singleton_inst(self.head_var_ptrs)
            op.op_comp = self
            place_holder_arg_names = [f'{{{i}}}' for i in range(len(self.base_vars))]
            set_expr_template(op, op.gen_expr(arg_names=place_holder_arg_names))
            set_shorthand_template(op, op.gen_expr(arg_names=place_holder_arg_names, use_shorthand=True))
            
        return self._generate_op

    @property
    def used_ops(self):
        ''' Returns a dictionary keyed by ops used in the OpComp expression.
            values are unique integers 0,1,2 etc.'''
        if(not hasattr(self,'_used_ops')):
            _used_ops = {}
            # Since __equal__() is overloaded it's best to use .unq as unqiue keys
            # used_unqs = set() 
            oc = 0
            for i,instr in enumerate(self.instructions):
                if(isinstance(instr, OpComp) and instr.op.unq not in _used_ops):
                    _used_ops[instr.op.unq] = (oc,instr.op)
                    oc += 1
            # print("**", _used_ops)
            self._used_ops = _used_ops
        return self._used_ops

    #### Code Generation Methods ####
    def gen_op_imports(self,lang='python'):
        ''' Generates import expressions for any 'used_ops' in the OpComp'''
        op_imports = ""
        for i,(_,op) in enumerate(self.used_ops.values()):
            to_import = {"call_sig" : f'call_sig{i}',
                "call" : f'call{i}'}
            if(hasattr(op,"check")):
                to_import.update({"check" : f'check{i}', "check_sig" : f'check_sig{i}'})

            op_imports += gen_import_str(op.name, op.long_hash, to_import) + "\n"
        return op_imports

        
    def _gen_constant_defs(self,lang,names={},ind='    '):
        constant_defs = ''
        for i,(c,t) in enumerate(self.constants.items()):
            names[c] = f'c{i}'
            constant_defs += f"{ind}{gen_assign(lang, f'c{i}', f'{c!r}')}\n"
        return constant_defs

    def _gen_arg_seq(self, lang, arg_names=None, names={}):
        if(arg_names is None):
            arg_names = [f'a{i}' for i in range(len(self.base_vars))]
        for i,(v_p,(v,_,t)) in enumerate(self.base_vars.items()):
            names[v_p] = arg_names[i]
            # print(i, v_p, arg_names[i], len(self.base_vars))
        return arg_names

        
    def _call_check_prereqs(self,lang, arg_names=None,
                            op_fnames=None,
                            op_call_fnames=None,
                            op_check_fnames=None,
                            skip_consts=False,
                            args_are_heads=False,
                            **kwargs):
        ''' Helper function that fills 'names' which is a dictionary of 
            various objects to their equivalent string expressions, in 
            addition to 'arg_names' which is autofilled as needed, 
            and optionally 'const_defs'. '''
        names = {}
        
        if(not args_are_heads):
            arg_names = self._gen_arg_seq(lang, arg_names, names)
            for i,instr in enumerate(self.instructions):
                names[instr] = f'i{i}'
        else:
            # arg_names = self._gen_arg_seq(lang, arg_names, names)
            arg_names = [f'a{i}' for i in range(len(self.head_vars))]
            # print(">>",self.head_types)
            for i, var_ptr_or_instr in enumerate(self.head_vars):
                # if(isinstance(var_or_instr, Var)):
                    # names[var_ptr_or_instr] = arg_names[i]
                names[var_ptr_or_instr] = arg_names[i]
                # else:
                    # names[var_or_instr] = arg_names[i]

            for i,instr in enumerate(self.instructions):
                if(not isinstance(instr, DerefInstr)):
                    names[instr] = f'i{i}'

            #     print("<<", type(instr))
            #     if(isinstance(instr, OpComp)):
            #         names[instr] = f'a{i}'
            #         # arg_names.append(f'a{i}')
            #     else:
            #         names[instr] = f'i{i}'


            # assert len(arg_names) == len(self.head_types)

        if(op_call_fnames is None):
            op_call_fnames = {op : f'{op.name}.call' for _,op in self.used_ops.values()}
        for op, n in op_call_fnames.items(): names[(op.unq,'call')] = n
            
        if(op_check_fnames is None):
            op_check_fnames = {op : f'{op.name}.check' for _,op in self.used_ops.values()}
        for op, n in op_check_fnames.items(): names[(op.unq,'check')] = n
            
        if(op_fnames is None):
            op_fnames = {op : op.name for _,op in self.used_ops.values()}
        for op, n in op_fnames.items(): names[op.unq] = n
            
        if(skip_consts):
            return names, arg_names
        else:
            const_defs = self._gen_constant_defs(lang,names)
            return names, arg_names, const_defs

    def gen_expr(self, lang='python', **kwargs):
        '''Generates a oneline expression for the OpComp'''
        
        names, arg_names = self._call_check_prereqs(lang, skip_consts=True, **kwargs)
        # print("gen_expr", arg_names)
        # raise ValueError()
        for i,instr in enumerate(self.instructions):
            instr_reprs = []
            if(isinstance(instr, DerefInstr)):
                names[instr] = instr.template.format(g_nm(instr.var,names))
            else:
                for x in instr.args:
                    if(g_nm(x,names) is not None):
                        instr_reprs.append(g_nm(x,names))
                    elif(isinstance(x,(Op,OpComp))):
                        instr_reprs.append(x.gen_expr(lang,**kwargs))
                    else:
                        instr_reprs.append(repr(x))

            # instr_names = ",".join(instr_reprs)
                instr_kwargs = {**kwargs,'arg_names':instr_reprs}
                names[instr] = instr.op.gen_expr(lang=lang,**instr_kwargs) #f'{names[]}({instr_names})'
        # print(names)
        return names[list(self.instructions.keys())[-1]]

    
    def gen_call(self, lang='python', fname="call", ind='    ', **kwargs):
        '''Generates source for the equivalent call function for the OpComp'''
        names, arg_names, const_defs = self._call_check_prereqs(lang, **kwargs)
        body = const_defs
        for i,instr in enumerate(self.instructions):
            if(isinstance(instr, DerefInstr)):
                deref_str = f'{g_nm(instr.var,names)}.{".".join(instr.deref_attrs)}'
                body += f"{ind}{gen_assign(lang, f'i{i}', deref_str)}\n"
            else:
                inp_seq = ", ".join([g_nm(x,names) for x in instr.args])
                call_fname = names[(instr.op.unq,'call')]
                body += f"{ind}{gen_assign(lang, f'i{i}', f'{call_fname}({inp_seq})')}\n"

        tail = f'{ind}return i{len(self.instructions)-1}\n'
        return gen_def_func(lang, fname, ", ".join(arg_names), body, tail)

    def gen_head_wrapper(self, fname, wrapped_fname,  lang='python',  ind='    ', **kwargs):
        # kwargs['args_are_heads'] = True
        names, arg_names, const_defs = self._call_check_prereqs(lang, **kwargs)
        body = ""
        # print("^^", self.instructions)
        # inps = {}
        for i,instr in enumerate(self.instructions):
            # print(type(instr))
            if(isinstance(instr, DerefInstr)):
                deref_str = f'{g_nm(instr.var,names)}.{".".join(instr.deref_attrs)}'
                body += f"{ind}{gen_assign(lang, f'i{i}', deref_str)}\n"
                # inps.append(g_nm(instr,names))
            # else:
            #     print("instr.args", instr.args, [g_nm(x,names) for x in instr.args])
            #     for x in instr.args:

            #         if(isinstance(x, (Var,DerefInstr))): inps[g_nm(x,names)] =1
                        
                # inps += [g_nm(x,names) ]

        inp_seq = ", ".join([g_nm(v_ptr_or_dref,names) for v_ptr_or_dref in self.head_vars])
        # inp_seq = ",".join(inps.keys())
        tail = f"{ind}return {wrapped_fname}({inp_seq})\n"
        return gen_def_func(lang, fname, ", ".join(arg_names), body, tail)        


    def gen_ptr_wrapper(self, fname, wrapped_fname, head_type_names=None, lang='python',  ind='    ', **kwargs):
        kwargs['args_are_heads'] = True
        names, arg_names, const_defs = self._call_check_prereqs(lang, **kwargs)
        # print(arg_names)
        n_heads = len(self.head_vars)
        if(head_type_names is None): head_type_names = [f"h{i}_type" for i in range(n_heads)] 
        body = ""
        for i in range(n_heads):
            # print(i, head_type_names)
            load_str = f'_load_pointer({head_type_names[i]},ptrs[{i}])'
            body += f"{ind}{gen_assign(lang,f'i{i}',load_str)}\n"
        inps = ",".join([f'i{i}' for i in range(len(arg_names))])
        tail = f"{ind}return {wrapped_fname}({inps})\n"
        return gen_def_func(lang, fname, 'ptrs', body, tail)


    def gen_match(self, lang='python', fname="match", ind='    ', **kwargs):
        '''Generates source for the equivalent check function for the OpComp'''
        names, arg_names, const_defs = self._call_check_prereqs(lang, **kwargs)
        args_are_heads = kwargs.get('args_are_heads',False)
        # print(names)
        body = const_defs
        final_k = "True"

        for i,instr in enumerate(self.instructions):            
            if(isinstance(instr, DerefInstr)):
                if(args_are_heads):
                    continue
                deref_str = f'{g_nm(instr.var,names)}.{".".join(instr.deref_attrs)}'
                body += f"{ind}{gen_assign(lang, f'i{i}', deref_str)}\n"
            else:
                # print([(g_nm(x,names), x) for x in instr.args])
                inp_seq = ", ".join([g_nm(x,names) for x in instr.args])
                if(hasattr(instr.op,'check')):#hasattr(instr, 'check')):
                    check_fname = names[(instr.op.unq,'check')]
                    # print("CHECK", (instr.op.unq,'check'), check_fname)
                    body += f"{ind}{gen_assign(lang, f'k{i}', f'{check_fname}({inp_seq}')})\n"
                    body += f"{ind}if(not k{i}): return 0\n"

                call_fname = names[(instr.op.unq,'call')]
                body += f"{ind}{gen_assign(lang, f'i{i}', f'{call_fname}({inp_seq})')}\n"
            # call_fname = names[(instr.op.unq,'call')]
            # check_body += f"{ind}{gen_assign(lang, f'i{i}', f'{call_fname}({inp_seq})')}\n"
        tail = f"{ind}return True if i{len(self.instructions)-1} else False\n"
        return gen_def_func(lang, fname, ", ".join(arg_names), body, tail)


    def gen_check(self, lang='python', fname="check", ind='    ', **kwargs):
        '''Generates source for the equivalent check function for the OpComp'''
        names, arg_names, const_defs = self._call_check_prereqs(lang, **kwargs)
        # print(names)
        body = const_defs
        final_k = "True"
        for i,instr in enumerate(self.instructions):
            # print(type(self))
            # print(self.used_ops)
            # j,_ = self.used_ops[instr]
            if(isinstance(instr, DerefInstr)):
                deref_str = f'{g_nm(instr.var,names)}.{".".join(instr.deref_attrs)}'
                body += f"{ind}{gen_assign(lang, f'i{i}', deref_str)}\n"
            else:
                inp_seq = ", ".join([g_nm(x,names) for x in instr.args])
                if(hasattr(instr.op,'check')):#hasattr(instr, 'check')):
                    check_fname = names[(instr.op.unq,'check')]
                    # print("CHECK", (instr.op.unq,'check'), check_fname)
                    body += f"{ind}{gen_assign(lang, f'k{i}', f'{check_fname}({inp_seq}')})\n"
                    body += f"{ind}if(not k{i}): return 0\n"
                    final_k = f'k{i}'

                if(i < len(self.instructions)-1):
                    call_fname = names[(instr.op.unq,'call')]
                    body += f"{ind}{gen_assign(lang, f'i{i}', f'{call_fname}({inp_seq})')}\n"
        tail = f"{ind}return {final_k}\n" 
        return gen_def_func(lang, fname, ", ".join(arg_names), body, tail)

    
    def gen_flattened_op_src(self, long_hash, ind='    '):
        '''Generates python source for the equivalant flattened Op'''
        op_imports = self.gen_op_imports('python')
        
        op_call_fnames = {op : f'call{j}' for j,op in self.used_ops.values()}
        call_src = self.gen_call('python', op_call_fnames=op_call_fnames)

        has_check = any([hasattr(op, 'check') for _,op in self.used_ops.values()])
        if(has_check):
            op_check_fnames = {op : f'check{j}' for j,op in self.used_ops.values()}
            check_src = self.gen_check('python', op_call_fnames=op_call_fnames, op_check_fnames=op_check_fnames)
            match_heads_src = self.gen_match('python', fname="match_heads",
             args_are_heads=True, op_call_fnames=op_call_fnames, op_check_fnames=op_check_fnames)
        else:
            match_heads_src = self.gen_match('python', fname="match_heads",
             args_are_heads=True, op_call_fnames=op_call_fnames)
        match_head_ptrs_src = self.gen_ptr_wrapper("match_head_ptrs", "match_heads")
        match_src = self.gen_head_wrapper("match", "match_heads")


        return_type = list(self.instructions.values())[-1].return_type
        call_sig = return_type(*self.arg_types)
        check_sig = u1(*self.arg_types)
        head_types = [x[2] for x in self.head_vars.values()]
        # print(call_sig)
# 
# boolean(*head_arg_types)
        source = f'''
from numba import i8, boolean, njit
from numba.types import unicode_type
from numba.extending import intrinsic
from numba.core import cgutils
from cre.op import Op
from cre.utils import _load_pointer
import cloudpickle
{op_imports}

call_sig = cloudpickle.loads({cloudpickle.dumps(call_sig)})
head_types = cloudpickle.loads({cloudpickle.dumps(head_types)})

{"".join([f'h{i}_type,' for i in range(len(head_types))])} = head_types

@njit(call_sig,cache=True)
{call_src}

@njit(boolean(*head_types), cache=True)
{match_heads_src}

@njit(boolean(*call_sig.args),cache=True)
{match_src}

@njit(boolean(i8[::1],),cache=True)
{match_head_ptrs_src}
'''
        if(has_check): source +=f'''
@njit(boolean(*call_sig.args),cache=True)
{check_src}
'''
        source += f'''

class __GenerateOp__(Op):
    signature = call_sig
    call = call
    {"check = check" if(has_check) else ""}
    _long_hash = {long_hash!r}
'''
        return source

    
    def __hash__(self):
        return hash(self.expr_template)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.expr_template

    def __call__(self,*args):
        flattened_inst = self.flatten()
        # print("FALT")
        return flattened_inst(*args)


