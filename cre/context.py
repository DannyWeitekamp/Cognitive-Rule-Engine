from numba import types, njit, guvectorize,vectorize,prange
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple, Array
from numba.cpython.unicode import  _set_code_point
from cre.core import TYPE_ALIASES, DEFAULT_REGISTERED_TYPES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from cre.gensource import assert_gen_source
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache
# from cre.struct_gen import gen_struct_code
from cre.structref import define_structref
from numba.experimental.structref import new
from collections import namedtuple
import numpy as np
import timeit
import itertools
import types as pytypes
import sys
import __main__
import os
import contextvars




# numba_type_ids = {k:i  for i,k in enumerate(numba_type_map)}


Dict_Unicode_to_Enums = DictType(unicode_type,i8[:])
Dict_Unicode_to_i8 = DictType(unicode_type,i8)
Dict_Unicode_to_Flags = DictType(unicode_type,u1[:])


context_data_fields = [
    ("string_enums" , DictType(unicode_type,i8)),
    ("number_enums" , DictType(f8,i8)),
    ("string_backmap" , DictType(i8,unicode_type)),
    ("number_backmap" , DictType(i8,f8)),
    ("enum_counter" , Array(i8, 0, "C")),
    ("attr_inds_by_type" , DictType(unicode_type,Dict_Unicode_to_i8)),
    ("spec_flags" , DictType(unicode_type,Dict_Unicode_to_Flags)),
    ("fact_to_t_id" , DictType(unicode_type,i8)),
    ("fact_num_to_t_id" , i8[::1]),#DictType(i8,i8)),
    ("parent_t_ids", ListType(i8[::1])),
    ("child_t_ids", ListType(i8[::1]))
]

CREContextData, CREContextDataType = define_structref("CREContextData",context_data_fields, define_constructor=False)

# if(not source_in_cache("CREContextData",'CREContextData')):
#     source = gen_struct_code("CREContextData",context_data_fields)
#     source_to_cache("CREContextData",'CREContextData',source)
    
# CREContextData, CREContextDataTypeTemplate = import_from_cached("CREContextData",
#     "CREContextData",["CREContextData","CREContextDataTypeTemplate"]).values()

# CREContextDataType = CREContextDataTypeTemplate(fields=context_data_fields)
i8_arr_type = i8[::1]
@njit(cache=True)
def new_cre_context():
    st = new(CREContextDataType)
    st.string_enums = Dict.empty(unicode_type,i8)
    st.number_enums = Dict.empty(f8,i8)
    st.string_backmap = Dict.empty(i8,unicode_type)
    st.number_backmap = Dict.empty(i8,f8)
    st.enum_counter = np.array(0)
    st.attr_inds_by_type = Dict.empty(unicode_type,Dict_Unicode_to_i8)
    # nominal_maps = Dict.empty(unicode_type,u1[:])
    st.spec_flags = Dict.empty(unicode_type,Dict_Unicode_to_Flags)
    st.fact_to_t_id = Dict.empty(unicode_type,i8)
    st.fact_num_to_t_id = np.empty(2,dtype=np.int64)#Dict.empty(i8,i8)
    st.parent_t_ids = List.empty_list(i8_arr_type)
    st.child_t_ids = List.empty_list(i8_arr_type)
    return st 
    #CREContextData(string_enums, number_enums,
        # string_backmap, number_backmap,
        # enum_counter, attr_inds_by_type, spec_flags, fact_to_t_id,
        # fact_num_to_t_id)

# @overload(Conditions,strict=False)
# def context_data_ctor():


@njit(cache=True)
def grow_fact_num_to_t_id(cd, new_size):
    new_fact_num_to_t_id = np.empty(new_size,dtype=np.int64)
    new_fact_num_to_t_id[:len(cd.fact_num_to_t_id)] = cd.fact_num_to_t_id
    cd.fact_num_to_t_id = new_fact_num_to_t_id
    return new_fact_num_to_t_id

@njit(cache=True)
def assign_name_num_to_t_id(cd, t_id, name, fact_num, inh_fact_num=-1):
    # Assign the new t_id to the given name and fact_num
    cd.fact_to_t_id[name] = t_id 
    if(fact_num >= len(cd.fact_num_to_t_id)):
        cd.fact_num_to_t_id = grow_fact_num_to_t_id(cd, fact_num*2)
    cd.fact_num_to_t_id[fact_num] = t_id 

    # Ensure that parent_t_ids and child_t_ids are big enough
    for i in range(len(cd.parent_t_ids),t_id+1):
        cd.parent_t_ids.append(np.empty((0,),dtype=np.int64))

    for i in range(len(cd.child_t_ids),t_id+1):
        cd.child_t_ids.append(np.empty((0,),dtype=np.int64))

    # Use inh_fact_num to fill in the parents 
    if(inh_fact_num != -1):
        inh_t_id = cd.fact_num_to_t_id[inh_fact_num]
        old_arr = cd.parent_t_ids[inh_t_id]
        new_arr = np.empty((len(old_arr)+1,),dtype=np.int64)
        new_arr[:len(old_arr)] = old_arr
        new_arr[-1] = inh_t_id
        cd.parent_t_ids[t_id] = new_arr
    # else:
    #     cd.parent_t_ids.append(np.empty((0,),dtype=np.int64))

    # Use the updated parents to update child relations (facts count as their own child) 
    for p_t_id in cd.parent_t_ids[t_id]:
        old_arr = cd.child_t_ids[p_t_id]
        new_arr = np.empty((len(old_arr)+1,),dtype=np.int64)
        new_arr[:len(old_arr)] = old_arr
        new_arr[-1] = t_id
        cd.child_t_ids[p_t_id] = new_arr
    cd.child_t_ids[t_id] = np.array((t_id,),dtype=np.int64)
    return t_id

class CREContext(object):
    _contexts = {}

    @classmethod
    def get_default_context(cls):
        df_c = os.environ.get("cre_DEFAULT_CONTEXT")
        df_c = df_c if df_c else "cre_default"
        return cls.get_context(df_c)

    @classmethod
    def init(cls, name):
        # print("INIT" ,name)
        if(name not in cls._contexts):
            cls._contexts[name] = cls(name)

    @classmethod
    def get_context(cls, name=None):
        if(name is None):
            return cls.get_context(cre_context_ctxvar.get(cls.get_default_context()))
        if(isinstance(name,CREContext)): return name
        if(name not in cls._contexts): cls.init(name)
        return cls._contexts[name]

    @classmethod
    def set_default_context(cls, name):
        os.environ["cre_DEFAULT_CONTEXT"] = cls.get_context(name)

    def __init__(self,name):
        self.name = name
        self.type_registry = {**DEFAULT_REGISTERED_TYPES}
        self.op_instances = {}
        self.deferred_types = {}
        
        self.parents_of = {}
        self.children_of = {}
        
        self.context_data = cd = new_cre_context()
        self.string_enums = cd.string_enums
        self.number_enums = cd.number_enums
        self.string_backmap = cd.string_backmap
        self.number_backmap = cd.number_backmap
        self.enum_counter = cd.enum_counter
        self.attr_inds_by_type = cd.attr_inds_by_type
        self.spec_flags = cd.spec_flags
        self.fact_to_t_id = cd.fact_to_t_id
        self.fact_num_to_t_id = cd.fact_num_to_t_id
        # print("CONTEXT:", name)

        #Auto register TupleFact
        # from cre.tuple_fact import TupleFact
        # from cre.fact import BaseFact
        # from cre.core import T_ID_TUPLE_FACT
        # self._register_fact_type("BaseFact", BaseFact)
        # self._register_fact_type("TupleFact", TupleFact)
        # self.fact_num_to_t_id[TF_FACT_NUM] = T_ID_TUPLE_FACT


        
    def get_deferred_type(self,name):
        if(name not in self.deferred_types):
            from cre.fact import DeferredFactRefType
            self.deferred_types[name] = DeferredFactRefType(name)
        return self.deferred_types[name]



    def _register_fact_type(self, name, fact_type, inherit_from=None):
        #Add attributes to the ctor and type objects
        # fact_ctor.fact_type = fact_type
        # fact_type.fact_ctor = fact_ctor
        # fact_ctor.fact_name = name
        # fact_type.fact_name = name

        if("BaseFact" not in self.type_registry and name != "BaseFact" and name != "TupleFact"):
            from cre.tuple_fact import TupleFact
            from cre.fact import BaseFact
            self._register_fact_type("BaseFact", BaseFact)
            self._register_fact_type("TupleFact", TupleFact)
        
        # self.fact_ctors[name] = fact_ctor
        self.type_registry[name] = fact_type
        # print("REGISTER", name, self.name)
        #Map to t_ids
        t_id = len(self.type_registry)
        inh_fact_num = inherit_from._fact_num if inherit_from is not None else -1
        # assert inh_fact_num >= 0, "Invalid 
        # print(t_id, fact_type._fact_num, inh_fact_num)
        assign_name_num_to_t_id(self.context_data, t_id, name, fact_type._fact_num, inh_fact_num)
        assert t_id == len(self.type_registry), f'{t_id}, {len(self.type_registry)}'
        # fact_type._t_id = t_id

        # self.fact_to_t_id[fact_ctor] = t_id 
        # self.fact_to_t_id[fact_type] = t_id 
        from numba.core.typeconv.rules import TypeCastingRules, default_type_manager as tm
        # from numba.core.typeconv.rules import 
        from cre.fact import BaseFact
        # tcr = TypeCastingRules(tm)

        ft_str = str(fact_type)
        if_str = str(inherit_from)
        if(inherit_from): tm.set_safe_convert(fact_type, inherit_from)
        
        tm.set_safe_convert(fact_type, BaseFact)
        # tm.safe(fact_type, BaseFact)


        #Track inheritence structure
        i_name = inherit_from._fact_name if inherit_from else None
        self.parents_of[name] = self.parents_of.get(i_name,[]) + [inherit_from] if i_name else []
        if(i_name):
            for parent in self.parents_of[name]:
                p = parent._fact_name
                self.children_of[p] = self.children_of.get(p,[]) + [fact_type]
        self.children_of[name] = []

        # Index on both the type and it's name
        self.parents_of[fact_type] = self.parents_of[name]
        self.children_of[fact_type] = self.children_of[name]

        


    def _register_flag(self,flag):
        d = self.spec_flags[flag] = Dict.empty(unicode_type,u1[:])
        for name, fact_type in self.type_registry.items():
            spec = fact_type.spec
            d[name] = np.array([flag in x['flags'] for attr,x in spec.items()], dtype=np.uint8)


    def _assert_flags(self, name, spec):
        return #TODO: Need to rewrite this so it doesn't trigger typed container overloads
        for flag in itertools.chain(*[x['flags'] for atrr,x in spec.items()]):
            if flag not in self.spec_flags:
                self._register_flag(flag)
        for flag, d in self.spec_flags.items():
            d[name] = np.array([flag in x['flags'] for attr,x in spec.items()], dtype=np.uint8)

    def _update_attr_inds(self,name,spec):
        d = Dict.empty(unicode_type,i8)
        for i,attr in enumerate(spec.keys()):
            d[attr] = i
        self.attr_inds_by_type[name] = d

    def __str__(self):
        return f"CREContext({self.name})"

    def __enter__(self):
        self.token_prev_context = cre_context_ctxvar.set(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cre_context_ctxvar.reset(self.token_prev_context)
        self.token_prev_context = None
        if(exc_val): raise exc_val.with_traceback(exc_tb)

        return self


    # def define_fact(self, name, spec):
    #     spec = self._standardize_spec(spec)
    #     if(name in self.registered_specs):
    #         assert self.registered_specs[name] == spec, \
    #         "Specification redefinition not permitted. Attempted on %r" % name
    #     else:
    #         self.registered_specs[name] = spec
    #         self._assert_flags(name,spec)
    #         self._update_attr_inds(name,spec)
    #         jitstruct = self.jitstruct_from_spec(name,spec)
    #         self.jitstructs[name] = jitstruct

    #         REGISTERED_TYPES[name] = jitstruct.numba_type
    #         TYPE_ALIASES[name] = jitstruct.__name__
    #         JITSTRUCTS[name] = jitstruct
    #     return self.jitstructs[name]
    # def jitstruct_from_spec(self,name,spec,ind="   "):
        
    #     #For the purposes of autogenerating code we need a clean alphanumeric name 
    #     name = "".join(x for x in name if x.isalnum())

    #     #Unstandardize to use types only. Probably don't need tags for source gen.
    #     spec = {attr:x['type'] for attr,x in spec.items()}

    #     hash_code = unique_hash([name,spec])
    #     assert_gen_source(name, hash_code, spec=spec, custom_type=True)

    #     print("HEY!")
    #     out = import_from_cached(name,hash_code,[
    #         '{}_get_enumerized'.format(name),
    #         '{}_pack_from_numpy'.format(name),
    #         '{}'.format(name),
    #         'NB_{}'.format(name),
    #         '{}_enumerize_nb_objs'.format(name)
    #         ]).values()
    #     print("HEY")
    #     get_enumerized, pack_from_numpy, nt, nb_nt, enumerize_nb_objs = tuple(out)

    #     def py_get_enumerized(_self,assert_maps=True):
    #         return get_enumerized(_self,
    #                                string_enums=self.string_enums,
    #                                number_enums=self.number_enums,
    #                                string_backmap=self.string_backmap,
    #                                number_backmap=self.number_backmap,
    #                                enum_counter=self.enum_counter,
    #                                assert_maps=assert_maps)
    #     nt.get_enumerized = py_get_enumerized#pytypes.MethodType(_get_enumerized, self) 
    #     nt._get_enumerized = get_enumerized#pytypes.MethodType(_get_enumerized, self) 
    #     nt.pack_from_numpy = pack_from_numpy
    #     nt.enumerize_nb_objs = enumerize_nb_objs
    #     nt.numba_type = nb_nt
    #     nt.hash = hash_code
    #     nt.name = name

    #     return nt

    # def _standardize_spec(self,spec):
    #     out = {}
    #     # print("prestandardize")
    #     # print(spec)
    #     for attr,v in spec.items():
    #         if(isinstance(v,str)):
    #             typ, flags = v.lower(), []
    #         elif(isinstance(v,dict)):
    #             assert "type" in v, "Attribute specifications must have 'type' property, got %s." % v
    #             typ = v['type'].lower()
    #             flags = [x.lower() for x in v.get('flags',[])]
    #         else:
    #             raise ValueError("Spec attribute %r = %r is not valid type with type %s." % (attr,v,type(v)))

    #         #Strings are always nominal
    #         if(typ == 'string' and ('nominal' not in flags)): flags.append('nominal')

    #         out[attr] = {"type": typ, "flags" : flags}
    #     # print("poaststandardize")
    #     # print(out)
    #     return out



    
def cre_context(context=None):
    return CREContext.get_context(context)

def define_fact(name : str, spec : dict, context=None):
    return cre_context(context).define_fact(name,spec)

def define_facts(specs, #: list[dict[str,dict]],
                 context=None):
    for name, spec in specs.items():
        define_fact(name,spec,context=context)


class _BaseContextful(object):
    def __init__(self, context):

        #Context stuff
        self.context = CREContext.get_context(context)
        cd = self.context.context_data
        self.string_enums = cd.string_enums
        self.number_enums = cd.number_enums
        self.string_backmap = cd.string_backmap
        self.number_backmap = cd.number_backmap
        self.enum_counter = cd.enum_counter
        self.attr_inds_by_type = cd.attr_inds_by_type
        self.spec_flags = cd.spec_flags
        
        
cre_context_ctxvar = contextvars.ContextVar("cre_context",
        default=CREContext.get_default_context()) 
