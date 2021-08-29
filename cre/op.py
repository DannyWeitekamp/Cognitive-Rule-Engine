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
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template
from cre.memory import MemoryType, Memory, facts_for_t_id, fact_at_f_id
# from cre.fact import define_fact, BaseFactType, cast_fact, DeferredFactRefType, Fact
from cre.utils import (_struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_pointer,  lower_setattr, lower_getattr,
                       _pointer_from_struct, _decref_pointer, _incref_pointer, _incref_structref, _pointer_from_struct_incref)
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


from operator import itemgetter
from copy import copy
from os import getenv
from cre.utils import deref_type, OFFSET_TYPE_ATTR, OFFSET_TYPE_LIST, listtype_sizeof_item
import inspect, dill, pickle
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

    source = \
f'''from numba import njit, void, u1, objmode
from numba.experimental.function_type import _get_wrapper_address
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.utils import _func_from_address
from cre.op import op_fields_dict, OpTypeTemplate, warn_cant_compile
import dill
nopython_call = {cls.nopython_call} 
nopython_check = {cls.nopython_check} 
'''
    if(call_needs_jitting):

        source +=f'''
call_sig = dill.loads({dill.dumps(cls.call_sig)})
call_pyfunc = dill.loads({cls.call_bytes})
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
    if(check_needs_jitting):
        source += f'''
check_sig = dill.loads({dill.dumps(cls.check_sig)})
check_pyfunc = dill.loads({cls.check_bytes})
try:
    check = njit(check_sig,cache=True)(check_pyfunc)
except NumbaError as e:
    nopython_check=False
    warn_cant_compile('check',{cls.__name__!r}, e)

if(nopython_check==False):
    @njit(cache=True)
    def check({arg_names}):
        with objmode(_return=u1):
            _return = check_pyfunc({arg_names})
        return _return

check_addr = _get_wrapper_address(check, check_sig)
'''
# arg_offsets = {str(arg_offsets)}
    source += f'''
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

op_fields_dict = {
    "name" : unicode_type,
    "repr_expr" : unicode_type,
    "shorthand_expr" : unicode_type,
    # Mapping variable ptrs to aliases
    "var_map" : DictType(i8, unicode_type),
    "inv_var_map" : DictType(unicode_type, i8),
    "return_type_name" : unicode_type,
    "arg_type_names" : ListType(unicode_type),
    "call_addr" : i8,
    "call_multi_addr" : i8,
    "check_addr" : i8,
    
    # "arg_types" : types.Any,
    # "out_type" : types.Any,
    # "is_const" : i8[::1]
}

@structref.register
class OpTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

GenericOpType = OpTypeTemplate([(k,v) for k,v in op_fields_dict.items()])


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
    temp_inv_var_map = Dict.empty(unicode_type, i8)
    
    n_auto_gen = 0
    for var_ptr in var_ptrs:
        if(var_ptr not in _vars):
            var = _struct_from_pointer(GenericVarType, var_ptr)
            _vars[var_ptr] = var
            
    for var in _vars.values():
        if(var.alias != ""):
            if(var.alias not in temp_inv_var_map):
                temp_inv_var_map[var.alias] = var_ptr
            else:
                raise NameError("Duplicate Var alias.")
        else:
            n_auto_gen += 1
        
    generated_aliases = List.empty_list(unicode_type)
    ascii_code = 97# i.e. 'a'
    bad_aliases = ['i','j','k','o','I','J','K','O']
    for i in range(n_auto_gen):
        alias = chr(ascii_code)
        while(alias in temp_inv_var_map or alias in bad_aliases):
            ascii_code += 1
            alias = chr(ascii_code)
        generated_aliases.append(alias)
        ascii_code += 1

    return generated_aliases

@njit(cache=True)
def op_ctor(name, return_type_name, arg_type_names, var_ptrs, call_addr=0, call_multi_addr=0, check_addr=0):
    '''The constructor for an Op instance that can be passed to the numba runtime'''
    st = new(GenericOpType)
    st.name = name
    st.return_type_name = return_type_name

    st.arg_type_names = arg_type_names
    st.var_map = Dict.empty(i8, unicode_type)
    st.inv_var_map = Dict.empty(unicode_type, i8)

    generated_aliases = gen_placeholder_aliases(var_ptrs)
    j = 0
    for var_ptr in var_ptrs:
        var = _struct_from_pointer(GenericVarType, var_ptr)
        alias = var.alias
        if(alias == ""):
            alias = generated_aliases[j]; j+=1;
        
        st.var_map[var_ptr] = alias
        st.inv_var_map[alias] = var_ptr
            
    st.call_addr = call_addr
    st.call_multi_addr = call_multi_addr
    st.check_addr = check_addr
    return st


def new_op(name, members):
    '''Creates a new custom Op with 'name' and 'members' '''
    has_check = 'check' in members
    assert 'call' in members, "Op must have call() defined"
    assert hasattr(members['call'], '__call__'), "call() must be a function"
    assert 'signature' in members, "Op must have signature"
    assert not (has_check and not hasattr(members['check'], '__call__')), "check() must be a function"
    
    # See if call/check etc. are raw python functions and need to be wrapped in @jit.
    call_needs_jitting = not isinstance(members['call'], Dispatcher)
    check_needs_jitting = has_check and (not isinstance(members['check'],Dispatcher))
    
    # 'cls' is the class of the user defined Op (i.e. Add(a,b)).
    cls = type.__new__(OpMeta,name,(Op,),members) 
    cls.__call__ = call_op
    cls._handle_commutes()
    cls._handle_nopython()

    if(call_needs_jitting): cls.process_method("call", cls.signature)
    
    if(has_check):
        if(check_needs_jitting): cls.process_method("check", u1(*cls.signature.args))        
    
    # If either call or check needs to be jitted then generate source code
    #  to do the jitting and put it in the cache. Or retrieve if already
    #  defined.  This way jit(cache=True) can reliably re-retreive.
    if(call_needs_jitting or check_needs_jitting):
        name = cls.__name__
        hash_code = unique_hash([cls.signature, cls.call_bytes,
            cls.check_bytes if has_check else None])

        cls.hash_code = hash_code
        if(not source_in_cache(name, hash_code) or getattr(cls,'cache',True) == False):
            source = gen_op_source(cls, call_needs_jitting, check_needs_jitting)
            source_to_cache(name,hash_code,source)

        print(get_cache_path(name,hash_code))

        to_import = ['call','call_addr'] if call_needs_jitting else []
        if(check_needs_jitting): to_import += ['check', 'check_addr']
        l = import_from_cached(name, hash_code, to_import)
        for key, value in l.items():
            setattr(cls, key, value)

    # Make static so that self isn't the first argument for call/check.
    cls.call = staticmethod(cls.call)
    if(has_check): cls.check = staticmethod(cls.check)

    # Get store the addresses for call/check
    if(not call_needs_jitting): 
        cls.call_sig = cls.signature
        cls.call_addr = _get_wrapper_address(cls.call,cls.call_sig)

    if(not check_needs_jitting and has_check):
        cls.check_sig = u1(*cls.signature.args)
        cls.check_addr = _get_wrapper_address(cls.check,cls.check_sig)

    # Standardize shorthand definitions
    if(hasattr(cls,'shorthand') and 
        not isinstance(cls.shorthand,dict)):
        cls.shorthand = {'*' : cls.shorthand}

    
    if(cls.__name__ != "__GenerateOp__"):
        context = cre_context()
        op_inst = cls.make_singleton_inst()
        context._register_op_inst(op_inst)
        return op_inst

    return cls

def call_op(self,*py_args):
    if(all([not isinstance(x,(Var,OpMeta,OpComp, Op)) for x in py_args])):
        # If all of the arguments are constants then just call the Op
        return self.call(*py_args)
    else:
        # Otherwise build an OpComp and flatten it into a new Op
        op_comp = OpComp(self,*py_args)
        op = op_comp.flatten()
        op.op_comp = op_comp
        set_repr_expr(op, op.gen_expr())
        set_shorthand_expr(op, op.gen_expr(use_shorthand=True))
        return op

        
class OpMeta(type):
    ''' A the metaclass for op. Useful for singleton generation.
    '''
    def __repr__(cls):
        return cls.__name__ + f'({",".join(["?"] * len(cls.signature.args))})'

    # This is similar to defining __init_subclass__ in Op inside except we can
    #    return whatever we want. In this case we return a singleton instance.
    def __new__(meta_cls, *args):
        name, members = args[0], args[2]
        if(name == "Op"):
            return super().__new__(meta_cls,*args)
        # print("<<<<", [str(x.__name__) for x in args[1]])
        # members = 
        return new_op(name, members)


    def __call__(self,*args, **kwargs):
        ''' A decorator function that builds a new Op'''
        if(len(args) > 1): raise ValueError("Op() takes at most one position argument 'signature'.")
        if(isinstance(args[0],(str, numba.core.typing.templates.Signature))):
            kwargs['signature'] = args[0]
        else:
            raise ValueError(f"Unrecognized type {type(args[0])} for 'signature'")

        def wrapper(call_func):
            assert hasattr(call_func,"__call__")
            name = call_func.__name__
            members = kwargs
            members["call"] = call_func
            return new_op(name, members)

        return wrapper

        

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

        print(cls.commutes)

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



class Op(structref.StructRefProxy,metaclass=OpMeta):
    ''' Base class for an functional operation. Custom operations can be
        created by subclassing Op and defining a signature, call(), and 
        optional check() method. 
    '''
    @classmethod
    def process_method(cls,name,sig):
        py_func = getattr(cls,name)
        setattr(cls, name+"_pyfunc", py_func)
        setattr(cls, name+"_sig", sig)
        setattr(cls, name+'_bytes', dill.dumps(py_func))#dedent(inspect.getsource(py_func)))




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
    #         set_repr_expr(op, op.gen_expr())
    #         set_shorthand_expr(op, op.gen_expr(use_shorthand=True))
    #         return op

    def __str__(self):
        name = get_name(self)
        if(name == "__GenerateOp__"):
            return self.repr_expr
        else:
            return op_str(self)

    @classmethod
    def make_singleton_inst(cls,var_ptrs=None):
        '''Creates a singleton instance of the user defined subclass of Op.
            These instances unbox into the NRT as GenericOpType.
        '''
        # Use the variable names that the user defined in the call() fn.
        arg_names = inspect.getfullargspec(cls.call.py_func)[0]
        if(var_ptrs is None):
            var_ptrs = new_var_ptrs_from_types(cls.call_sig.args, arg_names)

        # Make the op instance
        arg_type_names = List([str(x) for x in cls.signature.args])
        sig = cls.signature
        op_inst = op_ctor(
            cls.__name__,
            str(sig.return_type),
            arg_type_names,
            var_ptrs,
            call_addr=cls.call_addr,
            check_addr=cls.__dict__.get('check_addr',0),
            )

        op_inst.arg_names = arg_names
        # In the NRT the instance will be a GenericOpType, but on
        #   the python side we need it to be its own custom class
        op_inst.__class__ = cls

        #Make and store repr and shorthand expressions
        repr_expr = op_inst.gen_expr(
            arg_names=arg_names,
            use_shorthand=False,
        )
        set_repr_expr(op_inst, repr_expr)
        if(hasattr(op_inst,"shorthand")):
            set_shorthand_expr(op_inst, op_inst.gen_expr(
                arg_names=arg_names,
                use_shorthand=True,
            ))
        else:
            set_shorthand_expr(op_inst, repr_expr)

        return op_inst

    def recover_singleton_inst(self,context=None):
        '''Sometimes numba needs to emit an op instance 
         that isn't the singleton instance. This recovers
         the singleton instance from the cre_context'''
        context = cre_context(context)
        return context.op_instances[self.name]



    @property
    def name(self):
        return get_name(self)

    @property
    def repr_expr(self):
        return get_repr_expr(self)

    @property
    def shorthand_expr(self):
        return get_shorthand_expr(self)

    @property
    def var_map(self):
        return get_var_map(self)

    @property
    def return_type_name(self):
        return get_return_type_name(self)



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



### Various jitted getters ###

@njit(cache=True)
def get_name(self):
    return self.name

@njit(cache=True)
def get_repr_expr(self):
    return self.repr_expr

@njit(cache=True)
def get_shorthand_expr(self):
    return self.shorthand_expr

@njit(cache=True)
def get_var_map(self):
    return self.var_map

@njit(cache=True)
def get_return_type_name(self):
    return self.return_type_name

@njit(cache=True)
def op_str(self):
    return self.name + "(" + get_arg_seq(self) + ")"

@njit(cache=True)
def get_arg_seq(self):
    s = ""
    for i,alias in enumerate(self.inv_var_map):
        s += alias
        if(i < len(self.inv_var_map)-1): s += ","
    return s

#### Various Setters ####
@njit(cache=True)
def set_repr_expr(self, expr):
    self.repr_expr = expr

@njit(cache=True)
def set_shorthand_expr(self, expr):
    self.shorthand_expr = expr




define_boxing(OpTypeTemplate,Op)   

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
        return f'{{}}.{".".join(list(self.deref_attrs))}'

    def _get_hashable(self):
        if(not hasattr(self,"_hashable")):
            self._hashable = (self.var.base_ptr, tuple(*self.deref_attrs))
        return self._hashable

    def __eq__(self, other):
        return self._get_hashable() == other._get_hashable()

    def __hash__(self):
        return hash(self._get_hashable())





class OpComp():
    '''A helper class representing a composition of operations
        that can be flattened into a new Op definition.'''

    def _repr_arg_helper(self,x):
        if isinstance(x, Var):
            return f'a{self.vars[x.base_ptr][1]}'
        elif(isinstance(x, OpComp)):
            arg_names = [f'a{self.vars[v_p][1]}' for v_p in x.vars]
            # print("arg_names", arg_names)
            return x.gen_expr('python',arg_names=arg_names)
        elif(isinstance(x,DerefInstr)):
            return f'a{self.vars[x.var.base_ptr][1]}.{".".join(x.deref_attrs)}'
        else:
            return repr(x) 


    def __init__(self,op,*py_args):
        '''Constructs the the core pieces of an OpComp from a set of arguments.
            These pieces are the outermost 'op', and a set of 'constants',
            'vars', and 'instructions' (i.e. other op_comps)'''
        _vars = {} #var_ptr -> (var_inst, index, type)
        constants = {}
        instructions = {}
        args = []
        arg_types = []
        # v_ptr_to_ind = {} 

        for i, x in enumerate(py_args):
            if(isinstance(x, (OpMeta,Op))): x = x.op_comp
            if(isinstance(x, OpComp)):
                for v_ptr,(v,_,t) in x.vars.items():
                    if(v_ptr not in _vars):
                        # v_ptr = v.get_ptr()
                        _vars[v_ptr] = (v,len(arg_types),t)
                        # v_ptr_to_ind[] = len(arg_types)
                        arg_types.append(t)

                for c,t in x.constants.items():
                    constants[c] = t
                for instr, sig in x.instructions.items():
                    instructions[instr] = sig
            else:
                if(isinstance(x,Var)):
                    v_ptr = x.base_ptr
                    if(v_ptr not in _vars):
                        t = x.base_type
                        _vars[v_ptr] = (x,len(arg_types),t)
                        arg_types.append(t)
                    if(x.base_type != x.head_type):
                        d_instr = DerefInstr(x)
                        instructions[d_instr] = x.head_type(x.base_type,)
                        x = d_instr
                        

                        
                else:
                    constants[x] = op.signature.args[i]
            args.append(x)

        self.op = op
        self.vars = _vars
        self.args = args
        self.arg_types = arg_types
        self.constants = constants
        self.signature = op.signature.return_type(*arg_types)
        # self.v_ptr_to_ind = v_ptr_to_ind
        

        self.repr_expr = f"{op.name}({', '.join([self._repr_arg_helper(x) for x in self.args])})"  
        self.name = self.repr_expr

        instructions[self] = op.signature

        
        self.instructions = instructions

    def flatten(self):
        ''' Flattens the OpComp into a single Op. Generates the source
             for the new Op as needed.'''
        if(not hasattr(self,'_generate_op')):
            hash_code = unique_hash([self.repr_expr,*[(x.name,x.hash_code) for x in self.used_ops]])
            if(not source_in_cache('__GenerateOp__', hash_code)):
                
                source = self.gen_flattened_op_src(hash_code)
                source_to_cache('__GenerateOp__', hash_code, source)
            l = import_from_cached('__GenerateOp__', hash_code, ['__GenerateOp__'])
            op_cls = self._generate_op_cls = l['__GenerateOp__']

            var_ptrs = np.empty(len(self.vars),dtype=np.int64)
            for i,v_p in enumerate(self.vars):
                var_ptrs[i] = v_p#v.get_ptr()

            op = self._generate_op = op_cls.make_singleton_inst(var_ptrs)
            
        return self._generate_op

    @property
    def used_ops(self):
        ''' Returns a dictionary keyed by ops used in the OpComp expression.
            values are unique integers 0,1,2 etc.'''
        if(not hasattr(self,'_used_ops')):
            used_ops = {}
            oc = 0
            for i,instr in enumerate(self.instructions):
                if(isinstance(instr, OpComp) and instr.op not in used_ops):
                    used_ops[instr.op] = (oc,instr.op.signature)
                    oc += 1
            self._used_ops = used_ops
        return self._used_ops

    #### Code Generation Methods ####
    def gen_op_imports(self,lang='python'):
        ''' Generates import expressions for any 'used_ops' in the OpComp'''
        op_imports = ""
        for i,(op,(_,sig)) in enumerate(self.used_ops.items()):
            to_import = {"call_sig" : f'call_sig{i}',
                "call" : f'call{i}'}
            if(hasattr(op,"check")):
                to_import.update({"check" : f'check{i}', "check_sig" : f'check_sig{i}'})

            op_imports += gen_import_str(op.name, op.hash_code, to_import) + "\n"
        return op_imports

        
    def _gen_constant_defs(self,lang,names={},ind='    '):
        constant_defs = ''
        for i,(c,t) in enumerate(self.constants.items()):
            names[c] = f'c{i}'
            constant_defs += f"{ind}{gen_assign(lang, f'c{i}', f'{c!r}')}\n"
        return constant_defs

    def _gen_arg_seq(self, lang, arg_names=None, names={}):
        if(arg_names is None):
            arg_names = [f'a{i}' for i in range(len(self.vars))]
        for i,(v_p,(v,_,t)) in enumerate(self.vars.items()):
            names[v_p] = arg_names[i]
            # print(i, v_p, arg_names[i], len(self.vars))
        return arg_names

        
    def _call_check_prereqs(self,lang, arg_names=None,
                            op_fnames=None,
                            op_call_fnames=None,
                            op_check_fnames=None,
                            skip_consts=False,
                            **kwargs):
        ''' Helper function that fills 'names' which is a dictionary of 
            various objects to their equivalent string expressions, in 
            addition to 'arg_names' which is autofilled as needed, 
            and optionally 'const_defs'. '''
        names = {}
        arg_names = self._gen_arg_seq(lang, arg_names, names)
        for i,instr in enumerate(self.instructions):
            names[instr] = f'i{i}'

        if(op_call_fnames is None):
            op_call_fnames = {op : f'{op.name}.call' for op in self.used_ops}
        for op, n in op_call_fnames.items(): names[(op,'call')] = n
            
        if(op_check_fnames is None):
            op_check_fnames = {op : f'{op.name}.check' for op in self.used_ops}
        for op, n in op_check_fnames.items(): names[(op,'check')] = n
            
        if(op_fnames is None):
            op_fnames = {op : op.name for op in self.used_ops}
        for op, n in op_fnames.items(): names[op] = n
            
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
        call_body = const_defs
        for i,instr in enumerate(self.instructions):
            if(isinstance(instr, DerefInstr)):
                deref_str = f'{g_nm(instr.var,names)}.{".".join(instr.deref_attrs)}'
                call_body += f"{ind}{gen_assign(lang, f'i{i}', deref_str)}\n"
            else:
                inp_seq = ", ".join([g_nm(x,names) for x in instr.args])
                call_fname = names[(instr.op,'call')]
                call_body += f"{ind}{gen_assign(lang, f'i{i}', f'{call_fname}({inp_seq})')}\n"

        tail = f'{ind}return i{len(self.instructions)-1}\n'
        return gen_def_func(lang, fname, ", ".join(arg_names), call_body, tail)


    def gen_check(self, lang='python', fname="check", ind='    ', **kwargs):
        '''Generates source for the equivalent check function for the OpComp'''
        names, arg_names, const_defs = self._call_check_prereqs(lang, **kwargs)
        check_body = const_defs
        final_k = "True"
        for i,instr in enumerate(self.instructions):
            j,_ = self.used_ops[instr.op]
            inp_seq = ", ".join([g_nm(x,names) for x in instr.args])
            if(hasattr(instr.op, 'check')):
                check_fname = names[(instr.op,'check')]
                check_body += f"{ind}{gen_assign(lang, f'k{i}', f'{check_fname}({inp_seq}')})\n"
                check_body += f"{ind}if(not k{i}): return 0\n"
                final_k = f'k{i}'
            if(i < len(self.instructions)-1):
                call_fname = names[(instr.op,'call')]
                check_body += f"{ind}{gen_assign(lang, f'i{i}', f'{call_fname}({inp_seq})')}\n"
        tail = f"{ind}return {final_k}\n" 
        return gen_def_func(lang, fname, ", ".join(arg_names), check_body, tail)

    
    def gen_flattened_op_src(self, hash_code, ind='    '):
        '''Generates python source for the equivalant flattened Op'''
        op_imports = self.gen_op_imports('python')
        
        op_call_fnames = {op : f'call{j}' for op,(j,_) in self.used_ops.items()}
        call_src = self.gen_call('python', op_call_fnames=op_call_fnames)

        has_check = any([hasattr(op, 'check') for op in self.used_ops])
        if(has_check):
            op_check_fnames = {op : f'check{j}' for op,(j,_) in self.used_ops.items()}
            check_src = self.gen_check('python', op_call_fnames=op_call_fnames, op_check_fnames=op_check_fnames)
        return_type = list(self.instructions.values())[-1].return_type
        call_sig = return_type(*self.arg_types)
        check_sig = u1(*self.arg_types)
        # print(call_sig)

        source = f'''
from numba import float64, int64, njit
from numba.types import unicode_type
from numba.extending import intrinsic
from numba.core import cgutils
from cre.op import Op
import dill
{op_imports}

call_sig = dill.loads({dill.dumps(call_sig)})
check_sig = dill.loads({dill.dumps(check_sig)})

@njit(call_sig,cache=True)
{call_src}
'''
        if(has_check): source +=f'''
@njit(check_sig,cache=True)
{check_src}
'''
        source += f'''

class __GenerateOp__(Op):
    signature = call_sig
    call = call
    {"check = check" if(has_check) else ""}
    hash_code = {hash_code!r}
'''
        return source

    
    def __hash__(self):
        return hash(self.repr_expr)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.repr_expr

    def __call__(self,*args):
        flattened_inst = self.flatten()
        # print("FALT")
        return flattened_inst(*args)




