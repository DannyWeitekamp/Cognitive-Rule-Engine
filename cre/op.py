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
                       _pointer_from_struct, _decref_pointer, _incref_pointer, _incref_structref)
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

def gen_op_source(cls, call_needs_jitting, check_needs_jitting):
    '''Generate source code for the relevant functions of a user defined Op.
       Note: The jitted call() and check() functions are pickled, the bytes dumped  
        into the source, and then reconstituted at runtime. This strategy seems to work
        even if globals are used in the function. 
    '''
    arg_types = cls.signature.args
    print(cls.signature.return_type)
    # arg_imports = "\n".join([gen_fact_import_str(x) for x in arg_types if isinstance(x,Fact)])
    field_dict = {**op_fields_dict,**{f"arg{i}" : t for i,t in enumerate(arg_types)}}

    offsets = get_offsets_from_member_types(field_dict)
    arg_offsets = offsets[len(op_fields_dict):]

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
# call.enable_caching()
print(call)
call_fndesc = call.overloads[call_sig.args].fndesc
call_addr = _get_wrapper_address(call, call_sig)
'''
    if(check_needs_jitting):
        source += f'''
check_sig = dill.loads({dill.dumps(cls.check_sig)})
check = njit(check_sig,cache=True)(dill.loads({cls.check_bytes}))
# check.enable_caching()
check_fndesc = check.overloads[check_sig.args].fndesc
check_addr = _get_wrapper_address(check, check_sig)
'''

    source += f'''
arg_offsets = {str(arg_offsets)}
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
    "call_ptr" : i8,
    "call_multi_ptr" : i8,
    "check_ptr" : i8,
    
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
def op_ctor(name, return_type_name, arg_type_names, call_ptr=0, call_multi_ptr=0, check_ptr=0):
    st = new(GenericOpType)
    st.name = name
    st.return_type_name = return_type_name
    st.arg_type_names = arg_type_names
    st.call_ptr = call_ptr
    st.call_multi_ptr = call_multi_ptr
    st.check_ptr = check_ptr
    return st

def dynam_gen_intrinsic(sig,codegen):
    l = {}
    g = {'codegen':codegen, 'sig': sig, 'intrinsic': intrinsic}
    exec(f'''
@intrinsic
def intr_call(typingctx, {",".join([f'a{i}' for i in range(len(sig.args))])}):
    print(sig, codegen)
    return sig, codegen
''',g,l)
    return l['intr_call']
        
class OpMeta(type):
    def __repr__(cls):
        return cls.__name__ + f'({",".join(["?"] * len(cls.signature.args))})'


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

    def __init_subclass__(cls, **kwargs):
        ''' Define a rule, it must have when/conds and then defined'''
        super().__init_subclass__(**kwargs)
        # if(not hasattr(cls,"call")):
        assert hasattr(cls,'call'), "Op must have call() defined"
        assert hasattr(cls.call, '__call__'), "call() must be a function"
        call_needs_jitting = not isinstance(cls.call,Dispatcher)
        check_needs_jitting = False 

        has_check = hasattr(cls,"check")
        if(has_check):
            check_needs_jitting = not isinstance(cls.check,Dispatcher)
            assert hasattr(cls.check, '__call__'), "check() must be a function"

        time1 = time.time_ns()/float(1e6)
        if(call_needs_jitting): cls.process_method("call", cls.signature)
        
        if(has_check):
            check_sig = u1(*cls.signature.args)
            if(check_needs_jitting): cls.process_method("check", check_sig)        

        time2 = time.time_ns()/float(1e6)
        print(f'2: {time2-time1} ms')

        if(check_needs_jitting or check_needs_jitting):
            name = cls.__name__
            hash_code = unique_hash([cls.signature, cls.call_bytes, cls.check_bytes])
            cls.hash_code = hash_code
            # print(hash_code)
            # print(get_cache_path(name, hash_code))
            if(not source_in_cache(name, hash_code) or getattr(cls,'cache',True) == False):
                source = gen_op_source(cls, check_needs_jitting, check_needs_jitting)
                source_to_cache(name,hash_code,source)

            time3 = time.time_ns()/float(1e6)
            print(f'3: {time3-time2} ms')
            to_import = ['call','call_addr','call_fndesc'] if check_needs_jitting else []
            if(check_needs_jitting): to_import += ['check', 'check_addr','check_fndesc']
            l = import_from_cached(name, hash_code, to_import)
            time4 = time.time_ns()/float(1e6)
            print(f'4: {time4-time3} ms')
            for key, value in l.items():
                setattr(cls, key, value)

        # if(not check_needs_jitting):
        #     cls.call_fndesc = cls.

    def __new__(cls,*py_args):
        print("py_args",py_args)
        if(all([not isinstance(x,(Var,Op,OpComp)) for x in py_args])):
            return cls.call(*py_args)
        else:
            op_comp = OpComp(cls,*py_args)
            op = op_comp.flatten()
            op.op_comp = op_comp
            return op
        # _vars = set()
        # var_map = {}
        # all_const = True
        # # any_const = True
        # # var_map = Dict.empty(i8, unicode_type)
        # unrolled_vars = []
        # for x in py_args:
        #     if(isinstance(x,Var)):
        #         unrolled_vars.append(x)
        #         var_map[x] = x.alias
        #         all_const = False
        #     elif(isinstance(x,Op)):
        #         unrolled_vars += [x for x in x.var_map.values()]


        # print(_vars)

        # is_const = [not isinstance(x,Var) ]
        # assert len(py_args) == len(cls.signature.args)
        # if(all_const):
        #     return cls.call(*py_args)
        # else:
        #     return        
# #         elif(any(is_const)):   
# #             # If any constants are passed then redefine a jitted
# #             #   function where the constants are injected into the llvm. 
# #             sig, codegen = cls.gen_sig_codegen(py_args)
# #             print(sig)
# #             py_args = None
# #             call_intr = dynam_gen_intrinsic(sig, codegen)
# #             l = {}
# #             cache_safe_exec(f'''
# # @njit(sig,cache=False)
# # def call(a):
# #     return call_intr(a)
# #                 ''', l, {"njit": njit, "sig":sig, "call_intr": call_intr}
# #             )
# #             call = l['call']
            
# #             call_addr = 0#_get_wrapper_address(call, sig)
# #             # self.call = call
#         else:
#             # Otherwise 
#             sig, codegen = cls.signature, None
#             call_addr = cls.call_addr

#         st = op_ctor(cls.__name__,
#                     str(sig.return_type),
#                     List([str(x) for x in sig.args]),
#                     call_addr)
#         st.call = call
#         return st

    def __call__(self,*args):
        return self.call(*args)


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

        self.name = f"{op.__name__}({', '.join([repr(x) for x in py_args])})"  
        instructions[self] = op.signature

        self.op = op
        self.vars = _vars
        self.args = args
        print("ARGS", args,)
        self.constants = constants
        self.instructions = instructions
        
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
        shared_body = ''
        for i,(c,t) in enumerate(self.constants.items()):
            names[c] = f'c{i}'
            shared_body +=  ind+f'c{i} = context.get_constant_generic(builder, {t}, {repr(c)})\n'

        var_names = []
        for i,(v,t) in enumerate(self.vars.items()):
            names[v] = f'v{i}'
            var_names.append(f'v{i}')
        var_names = ", ".join(var_names)

        for i,instr in enumerate(self.instructions):
            names[instr] = f'i{i}'

        ops = self.used_ops
            
        op_imports = ""
        for i,(op,(_,sig)) in enumerate(ops.items()):
            op_imports += gen_import_str(op.__name__, op.hash_code,
                {"call_fndesc" : f'call_fndesc{i}', "call_sig" : f'call_sig{i}',
                "check_fndesc" : f'check_fndesc{i}', "check_sig" : f'check_sig{i}'}) + "\n"

        call_body = shared_body
        for i,instr in enumerate(self.instructions):
            j,_ = ops[instr.op]
            instr_names = "".join([names[x]+", " for x in instr.args])
            call_body += f'''{ind}i{i} = context.call_internal(builder, call_fndesc{j}, call_sig{j}, ({instr_names}))\n'''

        check_body = shared_body
        for i,instr in enumerate(self.instructions):
            j,_ = ops[instr.op]
            instr_names = "".join([names[x]+", " for x in instr.args])
            check_body += f'''{ind}k{i} = context.call_internal(builder, check_fndesc{j}, call_sig{j}, ({instr_names}))\n'''
            check_body += f'''{ind}with cgutils.if_zero(builder, k{i}): return k{i}\n'''
            if(i < len(self.instructions)-1):
                check_body += f'''{ind}i{i} = context.call_internal(builder, call_fndesc{j}, call_sig{j}, ({instr_names}))\n'''

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

def call_codegen(context, builder, sig, args):
    [{var_names}] = args
{call_body}
{ind}return i{len(self.instructions)-1}

def check_codegen(context, builder, sig, args):
    [{var_names}] = args
{check_body}
{ind}return k{len(self.instructions)-1}

@intrinsic
def call_intr(ctx, {var_names}):
    return call_sig, call_codegen

@intrinsic
def check_intr(ctx, {var_names}):
    return check_sig, check_codegen

@njit(call_sig,cache=True)
def call({var_names}):
    return call_intr({var_names})

# @njit(check_sig,cache=True)
# def check({var_names}):
#     return check_intr({var_names})

call_fndesc = call.overloads[call_sig.args].fndesc
check_fndesc = 0#call.overloads[check_sig.args].fndesc

class GenerateOp(Op):
    signature = call_sig
    call = call
    # check = check
    hash_code = {hash_code!r}
'''
        return source

    def flatten(self):
        # name = cls.__name__
        hash_code = unique_hash([self.name,*[(x.__name__,x.hash_code) for x in self.used_ops]])
        # print(hash_code)
        # print(get_cache_path(name, hash_code))
        if(not source_in_cache('GenerateOp', hash_code)):
            source = self.gen_source(hash_code)
            source_to_cache('GenerateOp', hash_code, source)

        l = import_from_cached('GenerateOp', hash_code, ['GenerateOp'])
        return l['GenerateOp']

        





    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name



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
