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

from cre.core import TYPE_ALIASES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from cre.gensource import assert_gen_source
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path, cache_safe_exec
from cre.structref import gen_structref_code, define_structref, define_structref_template
from cre.context import cre_context
from cre.utils import _cast_structref, _struct_from_ptr, _func_from_address
from cre.var import Var
from cre.fact import define_fact, gen_fact_import_str
from cre.conditions import get_linked_conditions_instance
from cre.memset import MemoryType
from cre.matching import _get_ptr_matches, _struct_tuple_from_pointer_arr
from cre.conditions import ConditionsType
import inspect, dill, pickle
from textwrap import dedent

#### Rule ####

class RuleMeta(type, _UniqueHashable):
    pass


def gen_rule_source(cls):
    '''Generate source code for the relevant functions of a user defined rule.
       Note: The jitted then() function is pickled, the bytes dumped into the 
        source, and then reconstituted at runtime. This strategy seems to work
        even if globals are used in the function. 
       Note: apply_then_from_ptrs is reconstituted in rule_ctor() from it's address
         which is aquired via _get_wrapper_address on import.
    '''
    arg_types = cls.sig.args
    arg_imports = "\n".join([gen_fact_import_str(x) for x in arg_types])
    source = \
f'''from numba import njit
from numba.experimental.function_type import _get_wrapper_address
from cre.matching import _struct_tuple_from_pointer_arr
from cre.rule import rule_ctor, atfp_type
from cre.utils import _func_from_address
import pickle
{arg_imports}
arg_types = ({" ".join([x._fact_name +"Type," for x in arg_types])})

then = pickle.loads({pickle.dumps(njit(cls.then_pyfunc))})
then.enable_caching()

@njit(atfp_type.signature, cache=True)
def apply_then_from_ptrs(mem,ptrs):
    facts_tuple = _struct_tuple_from_pointer_arr(arg_types,ptrs)
    then(mem,*facts_tuple)

atfp_addr = _get_wrapper_address(apply_then_from_ptrs, atfp_type.signature)

'''
    return source


class Rule(structref.StructRefProxy, metaclass=RuleMeta):
    def __new__(cls,conds=None):
        # self = cls._ctor(conds if conds else cls.conds)
        self = rule_ctor(cls.__name__,
                         conds if conds else cls.conds,
                        cls._atfp_addr)#, self._apply_then_from_ptrs)
        # self.apply_then_from_ptrs = apply_then_from_ptrs_type(cls._apply_then_from_ptrs)
        if(conds is not None): self.conds = conds
        return self
    # def __init__(self, conds):
    #     '''An instance of a rule has its conds linked to a Memory'''
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

        l = import_from_cached(name, hash_code, ['then','apply_then_from_ptrs', 'atfp_addr'])
        # l,g = cache_safe_exec(source,gbls={'then':then, 'arg_types':arg_types})



        # raise ValueError()
        cls.then = l['then']
        cls._apply_then_from_ptrs = l['apply_then_from_ptrs']
        cls._atfp_addr = l['atfp_addr']
        # cls._ctor = l['ctor']

        # then_from_ptrs(cls.then, np.zeros(5,dtype=np.int64))

            
        # print("SURPRISE!")
        # print(cls.conds)

    # @classmethod
    def apply_then_from_ptrs(cls, mem, ptrs):
        '''Applies then() on a matching set of facts given as an array of pointers'''
        return rule_apply_then_from_ptrs(cls, mem, ptrs)

    def __str__(self):
        return str_rule(self)

    def __repr__(self):
        return str(self)
    # @property
    # def conds(self):
    #     return rule_get_conds(self)


@njit(cache=True)
def rule_apply_then_from_ptrs(self, mem, ptrs):
    return self.apply_then_from_ptrs(mem,ptrs)

@njit(cache=True) 
def rule_get_conds(self):
    return self.conds


@structref.register
class RuleTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

apply_then_from_ptrs_type = types.FunctionType(types.void(MemoryType,i8[::1]))

rule_fields = [
    ('name', unicode_type),
    ("conds" , ConditionsType),
    ("apply_then_from_ptrs", apply_then_from_ptrs_type),
    ]


define_boxing(RuleTypeTemplate,Rule)

RuleType = RuleTypeTemplate(rule_fields)
atfp_type = types.FunctionType(types.void(MemoryType,i8[::1]))

@njit(cache=False)
def rule_ctor(name, conds, atfp_addr):
    st = new(RuleType)
    st.name = name
    st.conds = conds
    st.apply_then_from_ptrs = _func_from_address(atfp_type, atfp_addr)
    return st

@njit(cache=True)
@overload(str)
def str_rule(self):
    return "Rule[" + self.name + "]"


#### ConflictSetItr ####

rule_matches_tuple_type = types.Tuple((RuleType,i8[:,::1]))

conflict_set_iter_fields = [
    ("rule_match_pairs", types.ListType(rule_matches_tuple_type)),
    ("r_n", i8),
    ("m_n", i8),
]

ConflictSetIter, ConflictSetIterType = define_structref("ConflictSetIter", conflict_set_iter_fields)

@njit(cache=True)
def cs_iter_ctor(rule_match_pairs):
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
    match = matches[self.m_n]
    self.m_n +=1 

    if(self.m_n >= len(matches)):
        self.r_n += 1
        self.m_n = 0 
    
    return rule, match

#### RuleEngine ####

# Cache these two to avoid recompile
@njit(cache=True)
def _new_rule_list():
    return List.empty_list(RuleType)

@njit(cache=True)
def _append_list(l,x):
    return l.append(x)

class RuleEngine(object):
    def __init__(self,mem, rule_classes):
        self.mem = mem
        self.rule_classes = rule_classes
        self.rules = _new_rule_list()
        for rule_cls in rule_classes:
            conds =  get_linked_conditions_instance(rule_cls.conds, mem, copy=True) 
            print(rule_cls)
            _append_list(self.rules, rule_cls(conds))
        # self.rules = List(self.rules)

    def start(self):
        rule_engine_start(self.mem,self.rules)
        
            
MAX_CYCLES = 100

@njit(cache=True)
def rule_engine_start(mem, rules):
    stack = List.empty_list(ConflictSetIterType)

    for cycles in range(MAX_CYCLES):
        conflict_set = List.empty_list(rule_matches_tuple_type)
        for rule in rules:
            # t0 = time_ns()
            matches = _get_ptr_matches(rule.conds)
            # t1 = time_ns()
            # print("\tget_matches", (t1-t0)/1e6)
            if(len(matches) > 0):
                conflict_set.append((rule,matches))

        print(conflict_set)
        cs_iter = cs_iter_ctor(conflict_set)

        while(not cs_iter_empty(cs_iter)):
            rule, match = cs_iter_next(cs_iter)
            # t0 = time_ns()
            # print(rule, match)
            rule.apply_then_from_ptrs(mem, match)
            # t1 = time_ns()
            # print("\tthen", (t1-t0)/1e6)
            if(mem.halt_flag):
                print("HALT")
                return

            # if(mem.backtrack_flag):
            #     pass
            # else:
            #     stack.append(cs_iter)
        # else:
        #     break

    if(cycles >= MAX_CYCLES-1): raise RuntimeError("Exceeded max cycles.")
        
            




