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
                       _pointer_from_struct, _decref_pointer, _incref_pointer, _incref_structref, _pointer_from_struct_incref, _struct_from_pointer)
from cre.utils import assign_to_alias_in_parent_frame, as_typed_list, iter_typed_list
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.fact import Fact, gen_fact_import_str, get_offsets_from_member_types
from cre.var import Var,str_var_ptr_derefs
from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
 get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from cre.make_source import make_source, gen_def_func, gen_assign, resolve_template, gen_def_class
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models, register_default
from numba.experimental.function_type import _get_wrapper_address


from operator import itemgetter
from copy import copy
from os import getenv
from cre.utils import deref_type, listtype_sizeof_item
import inspect, cloudpickle, pickle
from textwrap import dedent, indent
# from itertools import combinations
from collections.abc import Iterable
from cre.op import Op, OpMeta, UntypedOp, resolve_return_type, new_var_ptrs_from_types, op_ctor
import warnings


##### PTR OP #### 

class UntypedPtrOp():
    def __init__(self, name, members):
        self.name = name
        self.members = members
        # self.members['match_head_ptrs'] = njit(boolean(i8[::1]),cache=True)(self.members['match_head_ptrs'])
        self._specialize_cache = {}

    def __repr__(self):
        return f'cre.UntypedPtrOp(name={self.name}, members={self.members})'

    def __str__(self):
        return f'cre.{self.name}(signature=None)'

    def __call__(self,*py_args):
        assert all([isinstance(x, Var) for x in py_args]), "Cannot specialize PtrOp with constants."
        arg_types = [resolve_return_type(x) for x in py_args]

        members = {**self.members}
        members['signature'] = sig = boolean(*arg_types)
        var_ptrs = np.array([v.get_ptr_incref() for v in py_args],dtype=np.int64)

        return new_ptr_op(self.name, members,var_ptrs=var_ptrs)




def gen_ptr_op_source(cls):
    return f'''from numba import njit, i8, boolean
import cloudpickle

match_head_ptrs_pyfunc = cloudpickle.loads({cls.match_head_ptrs_bytes})
match_head_ptrs = njit(boolean(i8[::1]),cache=True)(match_head_ptrs_pyfunc)

'''

# @njit(cache=True)
def new_ptr_op(name, members, var_ptrs=None):
    assert 'match_head_ptrs' in members, "PtrOp must have match_head_ptrs() defined"

    if('signature' not in members):
        return UntypedPtrOp(name,members)
    elif('nargs' in members and members['nargs'] != len(members['signature'].args)):
        raise ValueError(f"Explicit {name}.nargs={members['nargs']}, does not match signature {members['signature']} with {len(members['signature'].args)} arguments.")

    cls = type.__new__(PtrOpMeta,name,(PtrOp,),members) 
    cls.__module__ = members["match_head_ptrs"].__module__
    cls.process_method("match_head_ptrs", types.boolean(i8[::1]))

    # cls._handle_defaults()  

    # cls.arg_type_names = as_typed_list(unicode_type,[str(x) for x in cls.signature.args])
    cls._handle_defaults(var_ptrs)

    
    name = cls.__name__

    #Triggers getter
    long_hash = cls.long_hash 
    if(not source_in_cache(name, long_hash) or getattr(cls,'cache',True) == False):
        source = gen_ptr_op_source(cls)
        source_to_cache(name,long_hash,source)

    l = import_from_cached(name, long_hash, ['match_head_ptrs'])
    cls.match_head_ptrs = l['match_head_ptrs']
    cls.match_head_ptrs = staticmethod(cls.match_head_ptrs)
    cls.match_head_ptrs_addr = _get_wrapper_address(cls.match_head_ptrs, boolean(i8[::1]))
    
    # Standardize shorthand definitions
    if(hasattr(cls,'shorthand') and not isinstance(cls.shorthand,dict)):
        cls.shorthand = {'*' : cls.shorthand}

    # if(cls.__name__ != "__GenerateOp__"):
    #     op_inst = cls.make_singleton_inst()
    #     return op_inst

    return cls.make_singleton_inst(head_var_ptrs=var_ptrs)

class PtrOpMeta(OpMeta):
    def __new__(meta_cls, *args):

        name, members = args[0], args[2]
        if(name == "PtrOp"):
            ret = type.__new__(meta_cls,*args)
            return ret

        return new_ptr_op(name, members)


    def __call__(self,*args, **kwargs):
        ''' A decorator function that builds a new Op'''
        if(len(args) > 1): raise ValueError("PtrOp() takes at most one position argument 'signature'.")
        
        def wrapper(match_head_ptrs_func):
            assert hasattr(match_head_ptrs_func,"__call__")
            name = match_head_ptrs_func.__name__
            members = kwargs
            members["match_head_ptrs"] = match_head_ptrs_func
            print(name, members)
            return new_ptr_op(name, members)

        if(len(args) == 1):
            if(isinstance(args[0],(str, numba.core.typing.templates.Signature))):
                kwargs['signature'] = args[0]
            elif(hasattr(args[0],'__call__')):
                return wrapper(args[0])
            else:
                raise ValueError(f"Unrecognized type {type(args[0])} for 'signature'")        

        return wrapper

    def __repr__(cls):
        return f"cre.ptrop.PtrOpMeta(name={cls.__name__!r}, signature={cls._get_simple_sig_str()})"




    def _handle_defaults(cls, var_ptrs=None):
        # By default use the variable names that the user defined in the call() fn.
        cls.default_arg_names = [f"a{i}" for i in range(len(cls.signature.args))]
        cls.arg_type_names = as_typed_list(unicode_type,
                            [str(x) for x in cls.signature.args])

        if(var_ptrs is not None):
            cls.default_var_ptrs = var_ptrs 
        else: 
            cls.default_var_ptrs = new_var_ptrs_from_types(cls.signature.args, cls.default_arg_names)

        dereffed_arg_names = [f'{{{i}}}{str_var_ptr_derefs(ptr)}' for i,ptr in enumerate(cls.default_var_ptrs)]
                
        if(not hasattr(cls,'_expr_template')):
            # mk_str = lambda i, ptr : 
            cls._expr_template = f"{cls.__name__}({','.join(dereffed_arg_names)})"
        
        cls._shorthand_template = cls.shorthand.format(*dereffed_arg_names) if(hasattr(cls,'shorthand')) else cls._expr_template
        # print("<<", dereffed_arg_names, cls._shorthand_template)


    @property
    def long_hash(cls):
        if(not hasattr(cls,'_long_hash')):
            long_hash = unique_hash([cls.match_head_ptrs_bytes])

            cls._long_hash = long_hash
        return cls._long_hash


class PtrOp(Op,metaclass=PtrOpMeta):

    @classmethod
    def make_singleton_inst(cls,head_var_ptrs=None):
        if(head_var_ptrs is None): head_var_ptrs = cls.default_var_ptrs
        op_inst = op_ctor(
            cls.__name__,
            str(types.boolean),
            cls.arg_type_names,
            head_var_ptrs,
            str(cls._expr_template),
            str(cls._shorthand_template),
            match_head_ptrs_addr=cls.match_head_ptrs_addr,
            is_ptr_op=True
            )
        
        op_inst.__class__ = cls

        return op_inst

    def __call__(self,*py_args):
        assert len(py_args) == self.nargs, f"{str(self)} takes {self.nargs}, but got {len(py_args)}"
        if(any([not isinstance(x,Var) for x in py_args])):
            raise ValueError("PtrOps only take Vars as input")
            # If all of the arguments are constants then just call the Op
            # return self.call(*py_args)
        else:
            head_var_ptrs = np.empty((self.nargs,),dtype=np.int64)
            for i, v in enumerate(py_args):
                head_var_ptrs[i] = v.get_ptr()
            return self.__class__.make_singleton_inst(head_var_ptrs)


