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
from cre.caching import _UniqueHashable
# from numba.core.extending import overload

from cre.core import TYPE_ALIASES, REGISTERED_TYPES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from cre.gensource import assert_gen_source
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.structref import gen_structref_code, define_structref, define_structref_template
from cre.context import kb_context
from cre.utils import _cast_structref, _struct_from_pointer
from cre.var import Var
from cre.fact import define_fact
from cre.condition_node import get_linked_conditions_instance
from cre.kb import KnowledgeBaseType
from cre.matching import get_pointer_matches_from_linked
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
            
            
class RuleEngine(object):
    def __init__(self,kb, rule_classes):
        self.kb = kb
        self.rule_classes = rule_classes
        self.rules = []
        for rule_cls in rule_classes:
            conds =  get_linked_conditions_instance(rule_cls.conds, kb, copy=True) 
            self.rules.append(rule_cls(conds))

    def start(self):
        stack = []

        while True:
            conflict_set = []
            for rule in self.rules:
                matches = get_pointer_matches_from_linked(rule.conds)
                if(len(matches) > 0):
                    conflict_set.append((rule,matches))

            cs_iter = ConflictSetIter(conflict_set)

            rule, match = next(cs_iter)
            
            rule.then_from_ptrs(self.kb, match)

            if(self.kb.halt_flag):
                return


                

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

    

class Rule(metaclass=RuleMeta):
    def __init__(self, conds):
        '''An instance of a rule has its conds linked to a KnowledgeBase'''
        self.conds = conds

    def __init_subclass__(cls, **kwargs):
        ''' Define a rule, it must have when/conds and then defined'''
        super().__init_subclass__(**kwargs)
        if(not hasattr(cls,"conds")):
            assert hasattr(cls,'when'), "Rule must have when() or conds defined"
            if(hasattr(cls.when, '__call__')):
                cls.conds = cls.when()
        assert hasattr(cls,"then"), "Rule must have then() defined"

        cls.sig = cls.conds.signature
        print(cls.sig.args)
        cls.then_pyfunc = cls.then
        then = cls.then = njit(cls.then_pyfunc, cache=True)
        arg_types = cls.sig.args

        @njit(cache=True)
        def _then_from_ptrs(kb,ptrs):
            facts_tuple = _struct_tuple_from_pointer_arr(arg_types,ptrs)
            # for i,arg_type in enumerate(literal_unroll(arg_types)):
            #     l.append(_struct_from_pointer(arg_type, ptrs[i]))
            then(kb,*facts_tuple)
            # then(*[_struct_from_pointer(arg_type, ptrs[i]) for i,arg_type in enumerate(literal_unroll(arg_types))])

        cls._then_from_ptrs = _then_from_ptrs

        # then_from_ptrs(cls.then, np.zeros(5,dtype=np.int64))

            
        print("SURPRISE!")
        print(cls.conds)

    @classmethod
    def then_from_ptrs(cls,kb,ptrs):
        '''Applies then() on a matching set of facts given as an array of pointers'''
        return cls._then_from_ptrs(kb,ptrs)








        
        

        



