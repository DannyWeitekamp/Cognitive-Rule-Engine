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
from cre.context import kb_context
from cre.structref import define_structref, define_structref_template
from cre.kb import KnowledgeBaseType, KnowledgeBase, facts_for_t_id, fact_at_f_id
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
from cre.make_source import make_source, gen_def_func, gen_assign, resolve_template
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.experimental.function_type import _get_wrapper_address


from operator import itemgetter
from copy import copy
from os import getenv
from cre.utils import deref_type, OFFSET_TYPE_ATTR, OFFSET_TYPE_LIST, listtype_sizeof_item
import inspect, dill, pickle
from textwrap import dedent

import time
class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.2f} ms')


def gen_op_source(cls, call_needs_jitting, check_needs_jitting):
    '''Generate source code for the relevant functions of a user defined Op.
       Note: The jitted call() and check() functions are pickled, the bytes dumped  
        into the source, and then reconstituted at runtime. This strategy seems to work
        even if globals are used in the function. 
    '''
    arg_types = cls.signature.args
    # arg_imports = "\n".join([gen_fact_import_str(x) for x in arg_types if isinstance(x,Fact)])
    field_dict = {**op_fields_dict,**{f"arg{i}" : t for i,t in enumerate(arg_types)}}

    # offsets = get_offsets_from_member_types(field_dict)
    # arg_offsets = offsets[len(op_fields_dict):]

    source = \
f'''from numba import njit, void, u1
from numba.experimental.function_type import _get_wrapper_address
from cre.utils import _func_from_address
from cre.op import op_fields_dict, OpTypeTemplate
import dill'''
    if(call_needs_jitting):

        source +=f'''
call_sig = dill.loads({dill.dumps(cls.call_sig)})
call = njit(call_sig,cache=True)(dill.loads({cls.call_bytes}))
call_addr = _get_wrapper_address(call, call_sig)
'''
    if(check_needs_jitting):
        source += f'''
check_sig = dill.loads({dill.dumps(cls.check_sig)})
check = njit(check_sig,cache=True)(dill.loads({cls.check_bytes}))
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
def op_ctor(name, return_type_name, var_ptrs, call_addr=0, call_multi_addr=0, check_addr=0):
    st = new(GenericOpType)
    st.name = name
    st.return_type_name = return_type_name

    st.arg_type_names = List.empty_list(unicode_type)
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

        
class OpMeta(type):
    ''' A the metaclass for op. Useful for singleton generation.
    '''
    def __repr__(cls):
        return cls.__name__ + f'({",".join(["?"] * len(cls.signature.args))})'

    # This is similar to defining __init_subclass__ in Op inside except we can
    #    return whatever we want. In this case we return a singleton instance.
    def __new__(meta_cls, *args):
        if(args[0] == "Op"):
            return super().__new__(meta_cls,*args)
        members = args[2]

        has_check = 'check' in members
        assert 'call' in members, "Op must have call() defined"
        assert hasattr(members['call'], '__call__'), "call() must be a function"
        assert 'signature' in members, "Op must have signature"
        assert not (has_check and not hasattr(members['check'], '__call__')), "check() must be a function"
        
        # See if call/check etc. are raw python functions and need to be wrapped in @jit.
        call_needs_jitting = not isinstance(members['call'], Dispatcher)
        check_needs_jitting = has_check and (not isinstance(members['check'],Dispatcher))
        
        # 'cls' is the class of the user defined Op (i.e. Add(a,b)).
        cls = super().__new__(meta_cls,*args) 
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

        # Standardize short_hand definitions
        if(hasattr(cls,'short_hand') and 
            not isinstance(cls.short_hand,dict)):
            cls.short_hand = {'*' : cls.short_hand}

        if(cls.__name__ != "__GenerateOp__"):
            return cls.make_singleton_inst()

        return cls


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

    def __call__(self,*py_args):
        
        if(all([not isinstance(x,(Var,OpMeta,OpComp, Op)) for x in py_args])):
            # If all of the arguments are constants then just call the Op
            return self.call(*py_args)
        else:
            # Otherwise build an OpComp and flatten it into a new Op
            op_comp = OpComp(self,*py_args)
            op = op_comp.flatten()
            op.op_comp = op_comp
            op._expr = op.gen_expr()
            return op

    def __str__(self):
        name = get_name(self)
        if(name == "__GenerateOp__"):
            return self._expr
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
        sig = cls.signature
        op_inst = op_ctor(
            cls.__name__,
            str(sig.return_type),
            var_ptrs,
            call_addr=cls.call_addr,
            check_addr=cls.__dict__.get('check_addr',0),
            )
        op_inst._expr = op_inst.gen_expr(
            arg_names=arg_names,
            use_shorthand=True,
        )
        # In the NRT the instance will be a GenericOpType, but on
        #   the python side we need it to be its own custom class
        op_inst.__class__ = cls
        return op_inst

    @property
    def name(self):
        return get_name(self)

    @property
    def var_map(self):
        return get_var_map(self)

    def gen_expr(self, lang='python',
         arg_names=None, use_shorthand=False, **kwargs):
        '''Generates a one line expression for this Op (which might have been built
            from a composition of ops) in terms of user defined Ops.
            E.g. Multiply(Add(x,y),2).gen_expr() -> "Multiply(Add(x,y),2)"
                 or if short_hands are defined -> "((x+y)*2)" '''
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
                hasattr(self,'short_hand')):
                template = resolve_template(lang,self.short_hand,'short_hand')
                return template.format(*arg_names)
            else:
                return f"{self.name}({','.join(arg_names)})"

    @make_source('*')
    def mk_src_py(self,lang='python'):
        print("mk_src_py",lang)
        




@njit(cache=True)
def get_name(self):
    return self.name


@njit(cache=True)
def get_var_map(self):
    return self.var_map

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


define_boxing(OpTypeTemplate,Op)   

class OpComp():
    '''A helper class representing a composition of operations
        that can be flattened into a new Op definition.'''

    def _repr_arg_helper(self,x):
        if isinstance(x, Var):
            return f'a{self.v_ptr_to_ind[x.get_ptr()]}'
        elif(isinstance(x, OpComp)):
            return x.gen_expr('python')
        else:
            return repr(x) 


    def __init__(self,op,*py_args):
        '''Constructs the the core pieces of an OpComp from a set of arguments.
            These pieces are the outermost 'op', and a set of 'constants',
            'vars', and 'instructions' (i.e. other op_comps)'''
        _vars = {}
        constants = {}
        instructions = {}
        args = []
        arg_types = []
        v_ptr_to_ind = {}

        for i, x in enumerate(py_args):
            if(isinstance(x, (OpMeta,Op))): x = x.op_comp
            if(isinstance(x, OpComp)):
                for v,t in x.vars.items():
                    if(v not in _vars):
                        _vars[v] = t
                        v_ptr_to_ind[v.get_ptr()] = len(arg_types)
                        arg_types.append(t)

                for c,t in x.constants.items():
                    constants[c] = t
                for op_comp in x.instructions:
                    instructions[op_comp] = op_comp.op.signature
            else:
                if(isinstance(x,Var)):
                    if(x not in _vars):
                        t = op.signature.args[i]
                        _vars[x] = op.signature.args[i]
                        v_ptr_to_ind[x.get_ptr()] = len(arg_types)
                        arg_types.append(t)
                else:
                    constants[x] = op.signature.args[i]
            args.append(x)

        self.op = op
        self.vars = _vars
        self.args = args
        self.arg_types = arg_types
        self.constants = constants
        self.v_ptr_to_ind = v_ptr_to_ind

        self._expr = f"{op.name}({', '.join([self._repr_arg_helper(x) for x in self.args])})"  
        self.name = self._expr

        instructions[self] = op.signature

        
        self.instructions = instructions

    def flatten(self):
        ''' Flattens the OpComp into a single Op. Generates the source
             for the new Op as needed.'''
        if(not hasattr(self,'_generate_op')):
            hash_code = unique_hash([self._expr,*[(x.name,x.hash_code) for x in self.used_ops]])
            if(not source_in_cache('__GenerateOp__', hash_code)):
                
                source = self.gen_flattened_op_src(hash_code)
                source_to_cache('__GenerateOp__', hash_code, source)
            l = import_from_cached('__GenerateOp__', hash_code, ['__GenerateOp__'])
            op_cls = self._generate_op_cls = l['__GenerateOp__']

            var_ptrs = np.empty(len(self.vars),dtype=np.int64)
            for i,v in enumerate(self.vars):
                var_ptrs[i] = v.get_ptr()

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
                if(instr.op not in used_ops):
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

    def _gen_arg_seq(self,lang, arg_names=None, names={}):
        if(arg_names is None):
            arg_names = [f'a{i}' for i in range(len(self.vars))]
        for i,(v,t) in enumerate(self.vars.items()):
            names[v] = arg_names[i]
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
        for i,instr in enumerate(self.instructions):
            instr_reprs = []
            for x in instr.args:
                if(x in names):
                    instr_reprs.append(names[x])
                elif(isinstance(x,(Op,OpComp))):
                    instr_reprs.append(x.gen_expr(lang,**kwargs))
                else:
                    instr_reprs.append(repr(x))

            # instr_names = ",".join(instr_reprs)
            instr_kwargs = {**kwargs,'arg_names':instr_reprs}
            names[instr] = instr.op.gen_expr(lang=lang,**instr_kwargs) #f'{names[]}({instr_names})'
        return names[list(self.instructions.keys())[-1]]

    
    def gen_call(self, lang='python', fname="call", ind='    ', **kwargs):
        '''Generates source for the equivalent call function for the OpComp'''
        names, arg_names, const_defs = self._call_check_prereqs(lang, **kwargs)
        call_body = const_defs
        for i,instr in enumerate(self.instructions):
            inp_seq = ", ".join([names[x] for x in instr.args])
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
            inp_seq = ", ".join([names[x] for x in instr.args])
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
        return hash(self._expr)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self._expr

    def __call__(self,*args):
        flattened_inst = self.flatten()
        # print("FALT")
        return flattened_inst(*args)




##### PLANNING PLANNING PLANNING ####
# Op:
#  members:
#    -name
#    -apply_ptr: 
#    -apply_multi_ptr: 
#    -condition_ptr: 
#    -arg_types: 
#    -out_type: 
#  -commutes... might want to just keep at python object level
#  should have option translations = {
#   "javascript" : {
#       "apply_body": '''return a + b'''
#       "condition_body": '''return a != 0'''
#   }
# }
#  
# OpComp: 
#  built in python with __call__
#  when __call__ is run on all on non-vars its apply/condition is cached and compiled
#   -happens in overload in numba context 
#   -happens at runtime in python context
#  Structref should subclass Operator
#  The structref proxy should keep around a linearization 
#  Should have a list of vars like a Condition
#  The linearization is a list of operation objects 
#    all that is necessary is that their modules can be identified
#    and imported to that an intrinsic can be compiled and then njit wrapped
#    then the various function ptrs can be retrieved at runtime
#  More broadly the Operator's actual class can be retrieved from a context store
#   by using it's name.
#  Storing constants might be tricky, it seems like they should be
#   stored in the OpComp structref, and retreived in the intrinsic
#   from something similar to attr_offsets for Vars. This would avoid
#   overspecialization if an OpComp needs to be rebuilt with new constants
#  
#  We need to be able to do things like:
#    -Add(Add(x,y),z)
#    -Add(Add(x,y),x)


