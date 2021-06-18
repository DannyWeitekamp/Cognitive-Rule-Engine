import operator
import numpy as np
import numba
from numba import types, njit, i8, u8, i4, u1, i8, literally, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type, void, Tuple
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from cre.context import kb_context
from cre.structref import define_structref, define_structref_template
from cre.kb import KnowledgeBaseType, KnowledgeBase, facts_for_t_id, fact_at_f_id
from cre.fact import define_fact, BaseFactType, cast_fact, DeferredFactRefType, Fact
from cre.utils import (_struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_pointer,  lower_setattr, lower_getattr,
                       _pointer_from_struct, _decref_pointer, _incref_pointer, _incref_structref, get_offsets_from_member_types)
from cre.utils import assign_to_alias_in_parent_frame
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.fact import Fact, gen_fact_import_str
from cre.var import Var
from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
 get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models


from operator import itemgetter
from copy import copy
from os import getenv
from cre.utils import deref_type, OFFSET_TYPE_ATTR, OFFSET_TYPE_LIST, listtype_sizeof_item
import inspect, dill, pickle
from textwrap import dedent
# import inspect

def gen_op_source(cls):
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
import pickle

call_sig = pickle.loads({pickle.dumps(cls.signature)})
call = pickle.loads({pickle.dumps(njit(cls.signature)(cls.call_pyfunc))})
call.enable_caching()
print(call)
call_addr = _get_wrapper_address(call, call_sig)
'''
    if(hasattr(cls,"check_pyfunc")):
        source += f'''
check_sig = pickle.loads({pickle.dumps(cls.signature)})
check = pickle.loads({pickle.dumps(njit(u1(*arg_types))(cls.check_pyfunc))})
check.enable_caching()
check_addr = _get_wrapper_address(check, check_sig)
'''

    source += f'''
arg_offsets = {str(arg_offsets)}
field_dict = {{**op_fields_dict,**{{f"arg{{i}}" : t for i,t in enumerate(call_sig.args)}}}}
{cls.__name__+'Type'} = OpTypeTemplate([(k,v) for k,v in field_dict.items()]) 
'''
    return source

op_fields_dict = {
    "name" : unicode_type,
    "apply_ptr" : i8,
    "apply_multi_ptr" : i8,
    "condition_ptr" : i8,
    "arg_types" : types.Any,
    "out_type" : types.Any,
    "is_const" : i8[::1]
}

@structref.register
class OpTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

GenericOpType = OpTypeTemplate([(k,v) for k,v in op_fields_dict.items()])

class Op(structref.StructRefProxy):
    def __init_subclass__(cls, **kwargs):
        ''' Define a rule, it must have when/conds and then defined'''
        super().__init_subclass__(**kwargs)
        # if(not hasattr(cls,"call")):
        assert hasattr(cls,'call'), "Op must have call() defined"
        assert hasattr(cls.call, '__call__'), "call() must be a function"

        has_check = hasattr(cls,"check")
        if(has_check):
            assert hasattr(cls.check, '__call__'), "check() must be a function"

        cls.call_pyfunc = cls.call
        cls.call_src = dedent(inspect.getsource(cls.call_pyfunc))
        if(has_check):
            cls.check_pyfunc = cls.check
            cls.check_src = dedent(inspect.getsource(cls.check_pyfunc))
        else:
            cls.check_src = None

        name = cls.__name__
        hash_code = unique_hash([cls.signature, cls.call_src, cls.check_src])
        # print(hash_code)
        if(not source_in_cache(name, hash_code) or getattr(cls,'cache',True) == False):
            source = gen_op_source(cls)
            source_to_cache(name,hash_code,source)


        to_import = ['call','call_addr']
        if(has_check): to_import += ['check', 'check_addr']
        l = import_from_cached(name, hash_code, to_import)

        cls.call = l['call']
        cls._call_addr = l['call_addr']
        if(has_check):
            cls.check = l['check']
            cls._check_addr = l['check_addr']

    def __new__(self,*args):
        is_const = [isinstance(x,Var) for x in args]
        if(all(is_const)):
            return self.call(*args)
            # self = rule_ctor(cls.__name__,
            #              conds if conds else cls.conds,
            #             cls._atfp_addr)#, self._apply_then_from_ptrs)
            # # self.apply_then_from_ptrs = apply_then_from_ptrs_type(cls._apply_then_from_ptrs)
            # if(conds is not None): self.conds = conds
            
        else:   
            return 1



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
