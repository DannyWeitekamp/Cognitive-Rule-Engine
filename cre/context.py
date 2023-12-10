from numba import types, njit, generated_jit
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, Tuple
# from numba.cpython.unicode import  _set_code_point
from cre.core import (TYPE_ALIASES, DEFAULT_REGISTERED_TYPES, JITSTRUCTS,
                        py_type_map, numba_type_map, numpy_type_map,
                        unpickle_type_from_t_id, t_id_from_type_name,
                        add_to_type_registry, SHORT_NAMES, DEFAULT_TYPE_T_IDS)
from cre.caching import unique_hash
# from cre.struct_gen import gen_struct_code
from cre.structref import define_structref
from cre.utils import PrintElapse, _set_global, _get_global, cast, _incref_ptr
from numba.extending import overload_method
from numba.experimental.structref import new
import numpy as np
import os

t_id_a_id_tup_type = Tuple((u2,u1))
context_data_fields = {
    "name": unicode_type,
    "unhandled_retro_registers": ListType(u2),
    "fact_to_t_id": DictType(unicode_type, u2),
    "t_id_to_type_names": DictType(u2, unicode_type),
    "parent_t_ids": ListType(u2[::1]),
    "child_t_ids": ListType(u2[::1]),
    "attr_names" : DictType(t_id_a_id_tup_type, unicode_type)
}

CREContextData, CREContextDataType, CREContextDataTypeClass  = define_structref("CREContextData",context_data_fields, define_constructor=False, return_type_class=True)

# @njit(types.void(CREContextDataType), cache=True)
@njit(cache=True)
def set_cre_context_data(context_data):
    _set_global(i8, "_CRE_context_data", cast(context_data,i8))

# @njit(CREContextDataType(), cache=True)
@njit(cache=True)
def get_cre_context_data():
    context_data_ptr = _get_global(i8, "_CRE_context_data")
    _incref_ptr(context_data_ptr)
    return cast(context_data_ptr, CREContextDataType)

u2_arr_type = u2[::1]
# @njit(CREContextDataType(unicode_type), cache=True)
@njit(cache=True)
def new_cre_context(name):
    st = new(CREContextDataType)
    st.name = name
    st.unhandled_retro_registers = List.empty_list(u2)
    st.fact_to_t_id = Dict.empty(unicode_type, u2)
    st.t_id_to_type_names = Dict.empty(u2, unicode_type)
    st.parent_t_ids = List.empty_list(u2_arr_type)
    st.child_t_ids = List.empty_list(u2_arr_type)
    st.attr_names = Dict.empty(t_id_a_id_tup_type, unicode_type)
    return st 


# @njit(types.void(CREContextDataType, unicode_type, u2), cache=True)
@njit(cache=True)
def assign_name_to_t_id(cd, name, t_id):
    cd.fact_to_t_id[name] = u2(t_id)
    cd.t_id_to_type_names[u2(t_id)] = name

# @njit(types.void(CREContextDataType, u2, u1, unicode_type), cache=True)
@njit(cache=True)
def assign_a_id_attr(cd, t_id, a_id, name):
    cd.attr_names[(u2(t_id), u1(a_id))] = name 

@njit(cache=True)
def ensure_inheritance(cd, t_id, inh_t_id=-1):
    # Ensure that parent_t_ids and child_t_ids are big enough
    for i in range(len(cd.parent_t_ids),t_id+1):
        cd.parent_t_ids.append(np.zeros((0,),dtype=np.uint16))

    for i in range(len(cd.child_t_ids),t_id+1):
        cd.child_t_ids.append(np.zeros((0,),dtype=np.uint16))

    # Use inh_fact_num to fill in the parents 
    if(inh_t_id != -1):
        old_arr = cd.parent_t_ids[inh_t_id]
        new_arr = np.empty((len(old_arr)+1,),dtype=np.uint16)
        new_arr[:len(old_arr)] = old_arr
        new_arr[-1] = inh_t_id
        cd.parent_t_ids[t_id] = new_arr
    
    # Use the updated parents to update child relations (facts count as their own child) 
    for p_t_id in cd.parent_t_ids[t_id]:
        if(t_id not in cd.child_t_ids[p_t_id]):
            old_arr = cd.child_t_ids[p_t_id]
            new_arr = np.empty((len(old_arr)+1,),dtype=np.uint16)
            new_arr[:len(old_arr)] = old_arr
            new_arr[-1] = t_id
            cd.child_t_ids[p_t_id] = new_arr

    # Always treat as own child
    if(t_id not in cd.child_t_ids[t_id]):
        old_arr = cd.child_t_ids[t_id]
        new_arr = np.empty((len(old_arr)+1,),dtype=np.uint16)
        new_arr[:len(old_arr)] = old_arr
        new_arr[-1] = t_id
        cd.child_t_ids[t_id] = new_arr



@njit(cache=True)
def clear_unhandled_retro_registers(self):
    self.unhandled_retro_registers = List.empty_list(u2)

@njit(cache=True)
def get_unhandled_retro_registers(self):
    arr = np.empty((len(self.unhandled_retro_registers),), dtype=np.uint16)
    for i, t_id in enumerate(self.unhandled_retro_registers):
        arr[i] = t_id
    return arr

@njit(cache=True)
def has_unhandled_retro_registers(self):
    return len(self.unhandled_retro_registers) > 0

class CREContext(object):
    _contexts = {}

    @classmethod
    def get_default_context(cls):
        '''Returns the default context set in the cre_DEFAULT_CONTEXT 
            environment variable.'''
        df_c = os.environ.get("cre_DEFAULT_CONTEXT")
        df_c = df_c if df_c else "cre_default"
        return cls.get_context(df_c)

    @classmethod
    def init(cls, name):
        ''' Builds a new context with 'name'.'''
        if(name not in cls._contexts):
            self = cls(name)
            self.enter_count = 0
            cls._contexts[name] = self
        else:
            raise ValueError(f"Context redefinition attempted for name {name}.")

    @classmethod
    def get_context(cls, name=None):
        ''' Gets a context by name or the current scope's context if no name
            is given. Will instantiate a new context if no context with of 'name' exists.'''
        if(name is None):
            return cls.get_context(cre_context_ctxvar.get(cls.get_default_context()))
        if(isinstance(name,CREContext)): return name
        if(name not in cls._contexts): cls.init(name)
        return cls._contexts[name]

    @classmethod
    def set_default_context(cls, name):
        os.environ["cre_DEFAULT_CONTEXT"] = name

    def __init__(self,name):
        self.name = name
        self.name_to_type = {**DEFAULT_REGISTERED_TYPES}
        self.name_to_type.update({str(x): x for x in DEFAULT_REGISTERED_TYPES.values()})

        self.t_id_to_type = list(DEFAULT_REGISTERED_TYPES.values())
        self.type_to_t_id = {typ:i for i,typ in enumerate(self.t_id_to_type)}

        self.op_instances = {}
        self.deferred_types = {}
        
        self.parents_of = {}
        self.children_of = {}
        
        self.context_data = cd = new_cre_context(name)

        for s,t in DEFAULT_REGISTERED_TYPES.items():
            t_id = DEFAULT_TYPE_T_IDS[s]
            s = SHORT_NAMES.get(t, s)
            assign_name_to_t_id(self.context_data, s, t_id)

        
    def get_deferred_type(self,name):
        if(name not in self.deferred_types):
            from cre.fact import DeferredFactRefType
            self.deferred_types[name] = DeferredFactRefType(name)
        return self.deferred_types[name]


    def _ensure_retro_registers(self):
        from cre.core import DEFAULT_REGISTERED_TYPES

        cd = self.context_data
        if(has_unhandled_retro_registers(cd)):
            child_t_ids = cd.child_t_ids
            for t_id in get_unhandled_retro_registers(cd):
                if(len(child_t_ids[t_id]) == 0):
                # if(t_id >= len(self.t_id_to_type) or self.t_id_to_type[t_id] is None):
                    self._retroactive_register(t_id)
            clear_unhandled_retro_registers(cd)

    @property
    def registered_types(self):
        self._ensure_retro_registers()
        return self.t_id_to_type

    @property
    def unhandled_retro_registers(self):
        return get_unhandled_retro_registers(self.context_data)

    def _assert_written_to_type_registry(self, typ):
        '''Ensures that a type is written to the on disk type registery'''
        if(typ not in self.type_to_t_id):
            hash_code = getattr(typ,'_hash_code',unique_hash(typ.name))
            t_id = t_id_from_type_name(str(typ), hash_code)
            name = str(typ)
            if(t_id == -1):
                t_id = add_to_type_registry(name, hash_code, typ)
            self._assign_name_t_id(str(typ), typ, t_id)

    def _assign_name_t_id(self, name, typ, t_id):
        '''For this context associate a name and t_id with typ.''' 
        assign_name_to_t_id(self.context_data, name, t_id)

        # Fill in the python facing 'name_to_type'
        # print(name, "->", str(typ))
        # if(str(typ) == "TextField_vVBHx8mvD8"):
        #     raise ValueError()
        self.name_to_type[name] = typ
        self.name_to_type[str(typ)] = typ

        # Ensure 't_id_to_type' is long enough. In the case of a retroactive
        #  registration there can be holes so fill with None first then assign t_id
        for i in range(len(self.t_id_to_type),t_id+1):
            self.t_id_to_type.append(None)
        self.t_id_to_type[t_id] = typ
        self.type_to_t_id[typ] = t_id

    def _register_attr_names(self, typ, t_id):
        for a_id, (attr,_) in enumerate(typ._fields):
            assign_a_id_attr(self.context_data, u2(t_id), u1(a_id), attr)

    def _register_fact_type(self, name, fact_type, inherit_from=None):
        '''Registers a fact_type to this context. Keeps track of inheritance and casting information'''

        t_id = fact_type.t_id
        inh_t_id = inherit_from.t_id if inherit_from is not None else -1
        ensure_inheritance(self.context_data, t_id, inh_t_id)
        self._assign_name_t_id(name, fact_type, t_id)
        self._register_attr_names(fact_type, t_id)


        
        # Dispatcher args of type inherit_from also accept fact_type 
        from numba.core.typeconv.rules import TypeCastingRules, default_type_manager as tm
        if(inherit_from): 
            tm.set_safe_convert(fact_type, inherit_from)
            tm.set_safe_convert(types.ListType(fact_type), types.ListType(inherit_from))

        # Dispatcher args of type BaseFact also accept fact_type.
        from cre.fact import BaseFact
        tm.set_safe_convert(fact_type, BaseFact)
        tm.set_safe_convert(types.ListType(fact_type), types.ListType(BaseFact))
        
        # Track inheritence structure
        i_name = inherit_from._fact_name if inherit_from else None
        self.parents_of[name] = self.parents_of.get(i_name,[]) + [inherit_from] if i_name else []
        if(i_name):
            for parent in self.parents_of[name]:
                p = parent._fact_name
                self.children_of[p] = self.children_of.get(p,[]) + [fact_type]
                self.children_of[parent] = self.children_of[p]
        self.children_of[name] = []

        # Index inheritance on both the type and it's name
        self.parents_of[fact_type] = self.parents_of[name]
        self.children_of[fact_type] = self.children_of[name]

        
    def get_type(self, name:str = None, t_id:int = None, ensure_retro=True):
        '''Retrieves the type associated with a user defined fact given a
            name, fact_num, or t_id. '''
        if(ensure_retro): self._ensure_retro_registers()
        
        # If t_id defined then retrieve the type for t_id
        if(t_id is not None):
            typ = None
            if(t_id < len(self.t_id_to_type)):
                typ = self.t_id_to_type[t_id]
                if(typ is types.undefined): typ = None
            if(typ is None): typ = self._retroactive_register(t_id)
            if(typ is None): 
                print(f"No type with t_id={t_id} registered in cre_context {self.name}.")
                raise ValueError(f"No type with t_id={t_id} registered in cre_context {self.name}.")
            return typ
        
        # If got a name then check the registry for the name 
        if(name is not None):
            if(not isinstance(name, str)):
                raise ValueError(f"Type name should be str got {type(name)}")
            if(name in self.name_to_type):
                # print("get_type->", name, self.name_to_type[name], "in", self.name)
                return self.name_to_type[name]
            else:
                raise ValueError(f"No type {name} registered in cre_context {self.name}.")
                
        raise ValueError("Bad arguments for 'get_type'. Expecting one keyword argument name:str, or t_id:int")

    def _retroactive_register(self, t_id):
        '''Retroactively registers a type in cases where a fact_type is used but not defined (e.g. if cached dispatcher 
            returns an instance of it). Pulls the type definition from hard drive cache to register it retroactively.'''
        if(t_id == 0): 
            raise ValueError("Tried to register t_id=0 (i.e. undefined)")
        try:
            ft = ret_typ = _type = unpickle_type_from_t_id(t_id)
        except ImportError:
            return

        types = []
        while(ft is not None):
            types.append(ft)
            ft = getattr(ft,"parent_type",None)
        for ft in reversed(types):
            if(hasattr(ft,"_fact_name")):
                self._register_fact_type(ft._fact_name, ft, getattr(ft,'parent_type', None))
            else:
                self._assign_name_t_id(str(ft), ft, t_id)
        return ret_typ

    def get_t_id(self, _type=None, name:str=None, retro_register:bool=True):
        '''Retreives the t_id for a _type, or the name of the type'''
        if(name is not None):
            _type = self.get_type(name=name)

        if(_type is not None):
            if(hasattr(_type,'t_id')):
                t_id = _type.t_id
            else:
                self._assert_written_to_type_registry(_type)
                t_id = self.type_to_t_id[_type]


            return t_id

        raise ValueError("Bad arguments for 'get_t_id'. Expecting one keyword argument name:str, or _type:Type")

    
    def get_parent_t_ids(self, _type=None, name:str=None, t_id=None):
        if(t_id is None):
            t_id = self.get_t_id(_type=_type,name=name)

        if(t_id is None):
            raise ValueError("Bad arguments for 'get_parent_t_ids'. Expecting one keyword argument name:str, t_id:int, or _type:Type")

        p_t_ids = self.context_data.parent_t_ids
        if(t_id >= len(p_t_ids) or len(p_t_ids[t_id])):
            self._retroactive_register(t_id)
        p_t_ids = self.context_data.parent_t_ids

        return p_t_ids[t_id]
        

    def get_child_t_ids(self, _type=None, name:str=None, t_id=None, inclusive=True):
        # NOTE: Fix inclusive
        if(t_id is None):
            t_id = self.get_t_id(_type=_type,name=name)

        if(t_id is None):
            raise ValueError("Bad arguments for 'get_parent_t_ids'. Expecting one keyword argument name:str, t_id:int, or _type:Type")

        c_t_ids = self.context_data.child_t_ids
        if(t_id >= len(c_t_ids) or len(c_t_ids[t_id])):
            self._retroactive_register(t_id)
        c_t_ids = self.context_data.child_t_ids

        return c_t_ids[t_id]#[0 if inclusive else 1:]


    def standardize_type(self, typ, name='', attr=''):
        '''Takes in a string or type and returns the standardized type'''
        if(isinstance(typ, type)):
            typ = typ.__name__
        if(isinstance(typ,str)):
            typ_str = typ
            is_list = typ_str.lower().startswith("list")
            if(is_list): typ_str = typ_str.split("(")[1][:-1]

            is_deferred = False
            if(typ_str.lower() in TYPE_ALIASES): 
                typ = numba_type_map[TYPE_ALIASES[typ_str.lower()]]
            # elif(typ_str == name):
            #     typ = context.get_deferred_type(name)# DeferredFactRefType(name)
            elif(typ_str in self.name_to_type):
                typ = self.name_to_type[typ_str]
            else:
                typ = self.get_deferred_type(typ_str)
                is_deferred = True
                # raise TypeError(f"Attribute type {typ_str!r} not recognized in spec" + 
                #     f" for attribute definition {attr!r}." if attr else ".")

            if(is_list): typ = ListType(typ)

        if(hasattr(typ, "_fact_type")): typ = typ._fact_type
        return typ




    def __str__(self):
        return f"CREContext({self.name})"

    def __enter__(self):
        self.enter_count += 1
        self.token_prev_context = cre_context_ctxvar.set(self.name)
        set_cre_context_data(self.context_data)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.enter_count -= 1
        if(self.enter_count == 0):
            cre_context_ctxvar.reset(self.token_prev_context)
            set_cre_context_data(CREContext.get_context().context_data)
        if(exc_val): raise exc_val.with_traceback(exc_tb)

        return self
    
def cre_context(context=None):
    return CREContext.get_context(context)

set_cre_context_data(CREContext.get_default_context().context_data)
