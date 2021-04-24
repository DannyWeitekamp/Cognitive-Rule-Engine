import numpy as np
from numba import types, njit, guvectorize, vectorize, prange, generated_jit, literal_unroll
from numba.experimental import jitclass, structref
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple, Array
from numba.core.extending import (
    infer_getattr,
    lower_getattr_generic,
    lower_setattr_generic,
    overload_method,
    intrinsic,
    overload,
)
from numba.core.datamodel import default_manager, models
from numba.core.typing.templates import AttributeTemplate
from numba.core import types, cgutils
from numba.np.arrayobj import _getitem_array_single_int, make_array
from numba.experimental.structref import define_boxing, new
from cre.caching import _UniqueHashable
# from numba.core.extending import overload

from cre.core import TYPE_ALIASES, REGISTERED_TYPES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from cre.gensource import assert_gen_source
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path, cache_safe_exec
from cre.structref import gen_structref_code, define_structref, define_structref_template
from cre.context import kb_context
from cre.utils import _cast_structref, _struct_from_pointer
from cre.var import Var
from cre.fact import define_fact
from cre.condition_node import get_linked_conditions_instance
from cre.kb import KnowledgeBaseType
from cre.matching import get_pointer_matches_from_linked
from cre.condition_node import ConditionsType
import inspect, dill, pickle
from textwrap import dedent
# from numba


# Rule, RuleTemplateType = define_structref("Rule")


class RuleMeta(type, _UniqueHashable):
    pass
    # def __repr__(cls):
    #     return cls.template.format(*(['?']*len(cls.arg_types)),name=cls.__name__)

    # def get_hashable(cls):
    #     d = {k: v for k,v in vars(cls).items() if k in cls.hash_on}
    #     print("WHEE",d)
    #     return d


# @generated_jit(cache=True)
# def then_from_ptrs(f,arg_types,ptrs):
#     print(f.__dict__)
#     def impl(f,ptrs):
#         pass
        
#     return impl




class ConflictSetIter(object):
    def __init__(self, rule_match_pairs):
        self.rule_match_pairs = rule_match_pairs
        self.r_n = 0
        self.m_n = 0
    
    def __iter__(self):
        self.r_n = 0
        self.m_n = 0
        return self

    def __next__(self):
        if(self.r_n >= len(self.rule_match_pairs)):
            raise StopIteration()

        rule, matches = self.rule_match_pairs[self.r_n]
        if(self.m_n >= len(matches)):
            self.r_n +=1
            self.m_n = 0
        
        match = matches[self.m_n]
        return rule, match
            
from time import time_ns



                

            # print(rule.conds)


# def gen_then_source(sig,arg_names):
#     s = \
# f'''

# then = njit(rule_cls.then, cache=True)
# @njit(cache=True)
# def then(kb,{",".join(arg_names)}):
#     {"\n".join()}



# '''



@intrinsic
def _struct_tuple_from_pointer_arr(typingctx, struct_types, ptr_arr):
    ''' Takes a tuple of fact types and a ptr_array i.e. an i8[::1] and outputs 
        the facts pointed to, casted to the appropriate types '''
    if(isinstance(struct_types, UniTuple)):
        typs = tuple([struct_types.dtype.instance_type] * struct_types.count)
        out_type =  UniTuple(struct_types.dtype.instance_type,struct_types.count)
    else:
        raise NotImplemented("Need to write intrinsic for multi-type ")
    
    sig = out_type(struct_types,i8[::1])
    def codegen(context, builder, sig, args):
        _,ptrs = args

        vals = []
        ary = make_array(i8[::1])(context, builder, value=ptrs)
        for i, inst_type in enumerate(typs):
            i_val = context.get_constant(types.intp, i)

            # Same as _struct_from_pointer
            raw_ptr = _getitem_array_single_int(context,builder,i8,i8[::1],ary,i_val)
            meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)

            st = cgutils.create_struct_proxy(inst_type)(context, builder)
            st.meminfo = meminfo

            context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

            vals.append(st._getvalue())


        
        return context.make_tuple(builder,out_type,vals)

    return sig,codegen

def gen_rule_source(cls):
    '''Generate source code for the relevant functions of a user defined rule.
       Note: The jitted then() function is pickled, the bytes dumped into the 
        source, and then reconstituted at runtime. This strategy seems to work
        even if globals are used in the function. 
    '''
    arg_types = cls.sig.args
    arg_imports = "\n".join([f"from cre_cache.{x._fact_name}._{x._hash_code} import {x._fact_name + 'Type'}" for x in arg_types])
    source = \
f'''from numba import njit
from cre.rule import _struct_tuple_from_pointer_arr
import pickle
{arg_imports}
arg_types = ({" ".join([x._fact_name +"Type," for x in arg_types])})


then = pickle.loads({pickle.dumps(njit(cls.then_pyfunc))})
then.enable_caching()

@njit(cache=True)
def apply_then_from_ptrs(kb,ptrs):
    facts_tuple = _struct_tuple_from_pointer_arr(arg_types,ptrs)
    then(kb,*facts_tuple)
'''
    return source


class Rule(structref.StructRefProxy, metaclass=RuleMeta):
    def __new__(self,conds=None):
        
        self = rule_ctor(conds if conds else self.conds, self._apply_then_from_ptrs)
        if(conds is not None): self.conds = conds
        return self
    # def __init__(self, conds):
    #     '''An instance of a rule has its conds linked to a KnowledgeBase'''
    #     self.conds = conds

    def __init_subclass__(cls, **kwargs):
        ''' Define a rule, it must have when/conds and then defined'''
        super().__init_subclass__(**kwargs)
        if(not hasattr(cls,"conds")):
            assert hasattr(cls,'when'), "Rule must have when() or conds defined"
            if(hasattr(cls.when, '__call__')):
                cls.conds = cls.when()
        assert hasattr(cls,"then"), "Rule must have then() defined"

        cls.sig = cls.conds.signature

        # print(types.void(*cls.sig.args))
        # print(cls.sig.args)
        cls.then_pyfunc = cls.then
        # cls.then = njit(cls.then_pyfunc)
        # cls.then_src = dedent(inspect.getsource(cls.then_pyfunc))
        cls.then_src = dedent(inspect.getsource(cls.then_pyfunc))

        arg_types = cls.sig.args

        name = cls.__name__
        hash_code = unique_hash(list(arg_types)+[cls.then_src])
        # print(hash_code)
        if(not source_in_cache(name, hash_code) or getattr(cls,'cache_then',True) == False):

            source = gen_rule_source(cls)
            source_to_cache(name,hash_code,source)

        l = import_from_cached(name, hash_code, ['then','apply_then_from_ptrs'])
        # l,g = cache_safe_exec(source,gbls={'then':then, 'arg_types':arg_types})



        # raise ValueError()
        cls.then = l['then']
        cls._apply_then_from_ptrs = l['apply_then_from_ptrs']

        # then_from_ptrs(cls.then, np.zeros(5,dtype=np.int64))

            
        # print("SURPRISE!")
        # print(cls.conds)

    # @classmethod
    def apply_then_from_ptrs(cls, kb, ptrs):
        '''Applies then() on a matching set of facts given as an array of pointers'''
        return rule_apply_then_from_ptrs(cls, kb, ptrs)

    # @property
    # def conds(self):
    #     return rule_get_conds(self)


@njit(cache=True)
def rule_apply_then_from_ptrs(self, kb, ptrs):
    return self.apply_then_from_ptrs(kb,ptrs)

@njit(cache=True) 
def rule_get_conds(self):
    return self.conds


@structref.register
class RuleTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

rule_fields = [
    ("conds" , ConditionsType),
    ("apply_then_from_ptrs",types.FunctionType(types.void(KnowledgeBaseType,i8[::1]))),
    ]


define_boxing(RuleTypeTemplate,Rule)

RuleType = RuleTypeTemplate(rule_fields)

@njit(cache=True)
def rule_ctor(conds, apply_then_from_ptrs):
    st = new(RuleType)
    st.conds = conds
    st.apply_then_from_ptrs = apply_then_from_ptrs
    return st



rule_matches_tuple_type = types.Tuple((RuleType,i8[:,::1]))

conflict_set_iter_fields = [
    ("rule_match_pairs", types.ListType(rule_matches_tuple_type)),
    ("r_n", i8),
    ("m_n", i8),
]

ConflictSetIter, ConflictSetIterType = define_structref("ConflictSetIter", conflict_set_iter_fields)

@njit(cache=True)
def conflict_set_iter_ctor(rule_match_pairs):
    st = new(ConflictSetIterType)
    st.rule_match_pairs = rule_match_pairs
    st.r_n = 0
    st.m_n = 0
    return st

@njit(cache=True)
def cs_iter_empty(self):
    return self.r_n >= len(self.rule_match_pairs)

@njit(cache=True)
def cs_iter_next(self):
    if(self.r_n >= len(self.rule_match_pairs)):
        raise StopIteration()

    rule, matches = self.rule_match_pairs[self.r_n]
    if(self.m_n >= len(matches)):
        self.r_n +=1
        self.m_n = 0
    
    match = matches[self.m_n]
    return rule, match


class RuleEngine(object):
    def __init__(self,kb, rule_classes):
        self.kb = kb
        self.rule_classes = rule_classes
        self.rules = List.empty_list(RuleType)
        for rule_cls in rule_classes:
            conds =  get_linked_conditions_instance(rule_cls.conds, kb, copy=True) 
            self.rules.append(rule_cls(conds))
        # self.rules = List(self.rules)

    def start(self):
        rule_engine_start(self.kb,self.rules)
        
            

@njit(cache=True)
def rule_engine_start(kb, rules):
    stack = List.empty_list(ConflictSetIterType)

    while True:
        conflict_set = List.empty_list(rule_matches_tuple_type)
        for rule in rules:
            # t0 = time_ns()
            matches = get_pointer_matches_from_linked(rule.conds)
            # t1 = time_ns()
            # print("\tget_matches", (t1-t0)/1e6)
            if(len(matches) > 0):
                conflict_set.append((rule,matches))

        cs_iter = conflict_set_iter_ctor(conflict_set)

        if(not cs_iter_empty(cs_iter)):
            rule, match = cs_iter_next(cs_iter)
            # t0 = time_ns()
            rule.apply_then_from_ptrs(kb, match)
            # t1 = time_ns()
            # print("\tthen", (t1-t0)/1e6)
            if(kb.halt_flag):
                return

            if(kb.backtrack_flag):
                pass
            else:
                stack.append(cs_iter)
        
            




