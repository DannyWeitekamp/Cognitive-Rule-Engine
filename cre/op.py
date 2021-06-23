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

def new_var_ptrs_from_types(types):
    ptrs = np.empty(len(types), dtype=np.int64)
    for i, t in enumerate(types):
        ptrs[i] = var_to_ptr(Var(t))
    return ptrs

@njit(cache=True)
def fill_empty_aliases(var_ptrs):
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

    j = 0
    for var in _vars.values():
        if(var.alias == ""):
            var.alias = generated_aliases[j]; j+=1;
        



@njit(cache=True)
def op_ctor(name, return_type_name, var_ptrs, call_addr=0, call_multi_addr=0, check_addr=0):
    st = new(GenericOpType)
    st.name = name
    st.return_type_name = return_type_name

    st.arg_type_names = List.empty_list(unicode_type)
    st.var_map = Dict.empty(i8, unicode_type)
    st.inv_var_map = Dict.empty(unicode_type, i8)

    fill_empty_aliases(var_ptrs)

    nxt_alias_code = 97 # i.e. 'a'
    nxt_alias_chr = chr(nxt_alias_code)
    for var_ptr in var_ptrs:
        var = _struct_from_pointer(GenericVarType, var_ptr)
        st.var_map[var_ptr] = var.alias
        st.inv_var_map[var.alias] = var_ptr
        
        # st.arg_type_names.append(var.type_name)
        # if(var.alias != ''):
        #     alias = var.alias
        #     while(alias in st.inv_var_map):
        #         alias = "_" + alias
        # else:
        #     while(nxt_alias_chr in st.inv_var_map):
        #         nxt_alias_code += 1; nxt_alias_chr = chr(nxt_alias_code)
        #     alias = nxt_alias_chr
        #     nxt_alias_code += 1; nxt_alias_chr = chr(nxt_alias_code)
        # st.var_map[var_ptr] = alias
        # if(alias not in st.inv_var_map):
        #     st.inv_var_map[alias] = var_ptr
        # else:
        #     raise Exception("Unresolved reuse of Var alias.")
    
    st.call_addr = call_addr
    st.call_multi_addr = call_multi_addr
    st.check_addr = check_addr
    return st


@njit(cache=True)
def op_str(st):
    s = st.name + "("
    for i,alias in enumerate(st.inv_var_map):
        s += alias
        if(i < len(st.inv_var_map)-1): s += ","
    return s + ")"


# def dynam_gen_intrinsic(sig,codegen):
#     l = {}
#     g = {'codegen':codegen, 'sig': sig, 'intrinsic': intrinsic}
#     exec(f'''
# @intrinsic
# def intr_call(typingctx, {",".join([f'a{i}' for i in range(len(sig.args))])}):
#     print(sig, codegen)
#     return sig, codegen
# ''',g,l)
#     return l['intr_call']

# @njit(cache=True)
# def op_ctor(name):
#     st = new(GenericOpType)
#     st.name = name
#     return st

        
class OpMeta(type):
    def __repr__(cls):
        return cls.__name__ + f'({",".join(["?"] * len(cls.signature.args))})'

    # This is similar to defining __init_subclass__ in Op inside except we can
    #    return whatever we want
    def __new__(cls, *args):
        if(args[0] == "Op"):
            return super().__new__(cls,*args)
        members = args[2]
        # print(members)

        has_check = 'check' in members
        assert 'call' in members, "Op must have call() defined"
        assert hasattr(members['call'], '__call__'), "call() must be a function"
        assert 'signature' in members, "Op must have signature"
        assert not (has_check and not hasattr(members['check'], '__call__')), "check() must be a function"
        
        # see if call/check etc. are raw python functions and need to be wrapped in @jit 
        call_needs_jitting = not isinstance(members['call'], Dispatcher)
        check_needs_jitting = has_check and (not isinstance(members['check'],Dispatcher))
        
        # cls is the class of the user defined Op (i.e. Add(a,b))
        cls = super().__new__(cls,*args) 
        if(call_needs_jitting): cls.process_method("call", cls.signature)
        
        if(has_check):
            if(check_needs_jitting): cls.process_method("check", u1(*cls.signature.args))        
                
        if(call_needs_jitting or check_needs_jitting):
            name = cls.__name__

            # print(cls.call_bytes)
            with PrintElapse("gen_hash"):
                hash_code = unique_hash([cls.signature, cls.call_bytes,
                    cls.check_bytes if has_check else None])

            cls.hash_code = hash_code
            # print(hash_code)
            # print(get_cache_path(name, hash_code))
            if(not source_in_cache(name, hash_code) or getattr(cls,'cache',True) == False):
                with PrintElapse("gen_source"):
                    source = gen_op_source(cls, call_needs_jitting, check_needs_jitting)
                with PrintElapse("source_to_cache"):
                    source_to_cache(name,hash_code,source)

            with PrintElapse("import_cached"):
                to_import = ['call','call_addr'] if call_needs_jitting else []
                if(check_needs_jitting): to_import += ['check', 'check_addr']
                l = import_from_cached(name, hash_code, to_import)
                # time5 = time.time_ns()/float(1e6)
                for key, value in l.items():
                    setattr(cls, key, value)

        if(not call_needs_jitting): 
            cls.call_sig = cls.signature
            cls.call_addr = _get_wrapper_address(cls.call,cls.call_sig)

        if(not check_needs_jitting and has_check):
            cls.check_sig = u1(*cls.signature.args)
            cls.check_addr = _get_wrapper_address(cls.check,cls.check_sig)

        # print("L", l, to_import)
        with PrintElapse("new_var_ptrs"):
            new_var_ptrs = new_var_ptrs_from_types(cls.call_sig.args)
        with PrintElapse("build_instance"):
            sig = cls.signature
            op_inst = op_ctor(
                cls.__name__,
                str(sig.return_type),
                new_var_ptrs,#new_var_ptrs_from_types(cls.call_sig.args),
                # List([str(x) for x in sig.args]),
                call_addr=cls.call_addr,
                check_addr=cls.__dict__.get('check_addr',0),
                )
        op_inst.__class__ = cls
        # print("---------------",op_inst)
        # print(op_inst.call_sig)

        return cls



class Op(structref.StructRefProxy,metaclass=OpMeta):
    @classmethod
    def process_method(cls,name,sig):
        py_func = getattr(cls,name)
        setattr(cls, name+"_pyfunc", py_func)
        setattr(cls, name+"_sig", sig)
        # jitted_func = njit(sig,cache=True)(py_func)
        setattr(cls, name+'_bytes', dill.dumps(py_func))#dedent(inspect.getsource(py_func)))
        # setattr(cls, name, jitted_func)
        # setattr(cls, name+"_fndesc", jitted_func.overloads[sig.args].fndesc)
        # setattr(cls, name+"_addr", _get_wrapper_address(jitted_func, sig))

    # def __init_subclass__(cls, **kwargs):
    #     ''' Define a rule, it must have when/conds and then defined'''
    #     super().__init_subclass__(**kwargs)
        # if(not hasattr(cls,"call")):
        

        # if(not check_needs_jitting):
        #     cls.call_fndesc = cls.

    def __new__(cls,*py_args,return_instance=False):
        if(return_instance): return super().__new__(Op,cls)
        if(all([not isinstance(x,(Var,OpMeta,OpComp)) for x in py_args])):
            return cls.call(*py_args)
        else:
            op_comp = OpComp(cls,*py_args)
            op = op_comp.flatten()
            op.op_comp = op_comp
            return op
            # return op_comp
        

    def __call__(self,*args):
        print("CALLed")
        return self.call(*args)

    def __str__(self):
        return op_str(self)

    # @classmethod
    # def gen_sig_codegen(cls, py_args, fn_choice='call'):
    #     arg_types = cls.signature.args
    #     print(py_args)
    #     new_arg_types = [arg_types[i] for i,x in enumerate(py_args) if(isinstance(x, Var))]
    #     if(fn_choice =='call'):
    #         sig = cls.signature.return_type(*new_arg_types)
    #     else:
    #         sig = u1(*new_arg_types)

    #     def codegen(context, builder, _sig, _args):
    #         args = []; i = 0;
    #         for typ, py_arg in zip(arg_types, py_args):
    #             if(isinstance(py_arg, Op)):
    #                 args.append(_args[i]); i += 1;
    #             elif(isinstance(py_arg, Var)):
    #                 args.append(_args[i]); i += 1;
    #             else:
    #                 args.append(context.get_constant_generic(builder, typ, py_arg))

    #         fndesc = getattr(cls,fn_choice+"_fndesc")
    #         fn_sig = getattr(cls,fn_choice+"_sig")
    #         ret = context.call_internal(builder, cls.call_fndesc, fn_sig, args)
    #         return ret
    #     return sig, codegen

    # def __repr__(self):
        return 


define_boxing(OpTypeTemplate,Op)   

class OpComp():
    # @classmethod
    # def _new_backdoor(cls):
    #     super().__
    def __init__(self,op,*py_args):
        _vars = {}
        constants = {}
        instructions = {}
        args = []

        for i, x in enumerate(py_args):
            if(isinstance(x,OpMeta)): x = x.op_comp
            if(isinstance(x,OpComp)):
                for v,t in x.vars.items():
                    _vars[v] = t
                for c,t in x.constants.items():
                    constants[c] = t
                for op_comp in x.instructions:
                    instructions[op_comp] = op_comp.op.signature
            else:
                if(isinstance(x,Var)):
                    _vars[x] = op.signature.args[i]
                else:
                    constants[x] = op.signature.args[i]
            args.append(x)
        print("VARS", _vars)
        self.name = f"{op.__name__}({', '.join([repr(x) for x in py_args])})"  
        instructions[self] = op.signature

        self.op = op
        self.vars = _vars
        self.args = args
        self.constants = constants
        self.instructions = instructions
        print("DONE:", self.name)
        
        # print("-----START------")
        # print(list(self.vars.keys()))
        # print(list(self.constants.keys()))
        # print(list(self.instructions.keys()))
        # print("-----END------")
    @property
    def used_ops(self):
        if(not hasattr(self,'_used_ops')):
            used_ops = {}
            oc = 0
            for i,instr in enumerate(self.instructions):
                if(instr.op not in used_ops):
                    used_ops[instr.op] = (oc,instr.op.signature)
                    oc += 1
            self._used_ops = used_ops
        return self._used_ops
    
    def gen_source(self, hash_code, ind='    '):
        names = {} 
        # call_body = ''
        shared_body = ''
        for i,(c,t) in enumerate(self.constants.items()):
            names[c] = f'c{i}'
            # call_body +=  ind+f'c{i} = context.get_constant_generic(builder, {t}, {repr(c)})\n'
            shared_body +=  ind+f"c{i} = {c!r}\n"

        var_names = []
        for i,(v,t) in enumerate(self.vars.items()):
            names[v] = f'v{i}'
            var_names.append(f'v{i}')
        var_names = ", ".join(var_names)

        for i,instr in enumerate(self.instructions):
            names[instr] = f'i{i}'

        ops = self.used_ops
            
        op_imports = ""
        op_has_check = []
        for i,(op,(_,sig)) in enumerate(ops.items()):
            to_import = {"call_sig" : f'call_sig{i}',
                "call" : f'call{i}'}
            if(hasattr(op,"check")):
                op_has_check.append(True)
                to_import.update({"check" : f'check{i}', "check_sig" : f'check_sig{i}'})
            else:
                op_has_check.append(False)

            op_imports += gen_import_str(op.__name__, op.hash_code, to_import) + "\n"

                # "check_fndesc" : f'check_fndesc{i}', "check_sig" : f'check_sig{i}'}) + "\n"
        has_check = any(op_has_check)

        call_body = shared_body
        for i,instr in enumerate(self.instructions):
            j,_ = ops[instr.op]
            instr_names = "".join([names[x]+", " for x in instr.args])
            # call_body += f'''{ind}i{i} = context.call_internal(builder, call_fndesc{j}, call_sig{j}, ({instr_names}))\n'''
            call_body += f'''{ind}i{i} = call{j}({instr_names})\n'''

        # check_body = ""
        if(has_check):
            check_body = shared_body
            final_k = ""
            for i,instr in enumerate(self.instructions):
                j,_ = ops[instr.op]
                instr_names = "".join([names[x]+", " for x in instr.args])

                # check_body += f'{ind}c{i} = {repr(c)!r}\n'
                if(op_has_check[j]):
                    check_body += f'''{ind}k{i} = check{j}({instr_names})\n'''
                    check_body += f'''{ind}if(not k{i}): return 0\n'''
                    final_k = f'k{i}'
                # check_body += f'''{ind}k{i} = context.call_internal(builder, check_fndesc{j}, call_sig{j}, ({instr_names}))\n'''
                # check_body += f'''{ind}with cgutils.if_zero(builder, k{i}): return k{i}\n'''
                if(i < len(self.instructions)-1):
                    check_body += f'''{ind}i{i} = call{j}({instr_names})\n'''
                    # check_body += f'''{ind}i{i} = context.call_internal(builder, call_fndesc{j}, call_sig{j}, ({instr_names}))\n'''

        return_type = list(self.instructions.values())[-1].return_type
        call_sig = return_type(*self.vars.values())
        check_sig = u1(*self.vars.values())
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
def call({var_names}):
{call_body}
{ind}return i{len(self.instructions)-1}


'''
        if(has_check): source +=f'''
@njit(check_sig,cache=True)
def check({var_names}):
{check_body}
{ind}return {final_k}
'''
        source += f'''

class GenerateOp(Op):
    signature = call_sig
    call = call
    {"check = check" if(has_check) else ""}
    hash_code = {hash_code!r}
'''
        return source

    def flatten(self):
        if(not hasattr(self,'_generate_op')):
            hash_code = unique_hash([self.name,*[(x.__name__,x.hash_code) for x in self.used_ops]])
            # print("HAH", hash_code)
            if(not source_in_cache('GenerateOp', hash_code)):
                
                source = self.gen_source(hash_code)
                source_to_cache('GenerateOp', hash_code, source)
            l = import_from_cached('GenerateOp', hash_code, ['GenerateOp'])
            self._generate_op = l['GenerateOp']
        return self._generate_op

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

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


