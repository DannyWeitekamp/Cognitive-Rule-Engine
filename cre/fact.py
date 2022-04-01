from numba import types, njit, guvectorize, vectorize, prange, generated_jit
from numba.experimental import jitclass, structref
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import (DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple, Array, Tuple)
from numba.core.extending import (
    infer_getattr,
    lower_getattr_generic,
    lower_setattr_generic,
    overload_method,
    intrinsic,
    overload,
    box,
    unbox,
    NativeValue
)
from numba.core.datamodel import default_manager, models
from numba.core import types, cgutils
from numba.types import ListType
# from numba.core.extending import overload

from cre.core import TYPE_ALIASES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from cre.gensource import assert_gen_source
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.structref import gen_structref_code, define_structref
from cre.context import cre_context
from cre.utils import (_struct_from_ptr, _cast_structref, struct_get_attr_offset, _obj_cast_codegen,
                       _ptr_from_struct_codegen, _raw_ptr_from_struct, CastFriendlyMixin, _obj_cast_codegen,
                        PrintElapse, _struct_get_data_ptr)
from cre.cre_object import CREObjTypeTemplate, cre_obj_field_dict, CREObjModel, CREObjType, member_info_type

from numba.core.typeconv import Conversion
import operator
from numba.core.imputils import (lower_cast)
import cloudpickle
import numpy as np

GLOBAL_FACT_COUNT = -1
SPECIAL_ATTRIBUTES = ["inherit_from"]

class Fact(CREObjTypeTemplate):
    def __init__(self, fields):
        super().__init__(fields)

    def __str__(self):
        return self._fact_name if hasattr(self, '_fact_name') else "Fact"

    def get_attr_offset(self,attr):
        fd = self.field_dict
        return self._attr_offsets[list(fd.keys()).index(attr)]

    

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     if(hasattr(state,'spec')): del state['spec']
    #     if(hasattr(state,'fact_ctor')): del state['fact_ctor']
    #     print("SERIALIZE", self.name)
    #     print(state)
    #     return state

    # def __setstate__(self,state):
    #     print(state)
    #     self.__dict__.update(state)
    #     print("UNSERIALIZE", self.name)
    #     state = self.__dict__.copy()
        
    #     print(state)
    #     return state







###### Fact Specification Preprocessing #######
class DeferredFactRefType():
    '''A placeholder type for when referencing a fact type that
        is not defined yet. Note: Sort of mimics deferred_type but
         doesn't subclass because would use special numba pickling,
         which I haven't quite figured out.
    '''
    def __init__(self,typ):
        self._fact_name = typ._fact_name if isinstance(typ, types.StructRef) else typ 
        super(DeferredFactRefType,self).__init__()
    def __eq__(self,other):
        return isinstance(other,DeferredFactRefType) \
               and self._fact_name == other._fact_name
    def __str__(self):
        return f"DeferredFactRefType[{self._fact_name}]"

    def __repr__(self):
        return f"DeferredFactRefType({self._fact_name!r})"

    def define(self,x):
        self._define = x

    def get(self):
        if(not hasattr(self,"_define")): raise TypeError(f"Attempt to use {str(self)} without definition")
        return self._define

    def __hash__(self):
        return hash(self._fact_name)

    def __setstate__(self,state):
        # print("UNSERIALIZE",state)
        # self.dict.update(state)
        self._fact_name = state[0]
        # d = self.__dict__
        # del d['type']
        # return d

    def __getstate__(self):
        # print("SERIALIZE",self.__dict__.copy())
        return (self._fact_name,)#({'_fact_name' : self._fact_name})



def _standardize_type(typ, context, name='', attr=''):
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
        elif(typ_str in context.type_registry):
            typ = context.type_registry[typ_str]
        else:
            typ = context.get_deferred_type(typ_str)
            is_deferred = True
            # raise TypeError(f"Attribute type {typ_str!r} not recognized in spec" + 
            #     f" for attribute definition {attr!r}." if attr else ".")

        if(is_list): typ = ListType(typ)

    if(hasattr(typ, "_fact_type")): typ = typ._fact_type
    return typ


def _get_attr_type_flags(attr, v, context, name=''):
    '''A helper function for _standardize_spec'''

    # Extract typ_str + flags
    
    if(isinstance(v,dict)):
        assert "type" in v, "Attribute specifications must have 'type' property, got %s." % v
        typ = _standardize_type(v['type'], context, name, attr)
        flags = [x.lower() for x in v.get('flags',[])]
    else:
        typ, flags = _standardize_type(v, context, name, attr), []
    

    return typ, flags

def _merge_spec_inheritance(spec : dict, context):
    '''Expands a spec with attributes from its 'inherit_from' type'''
    if("inherit_from" not in spec): return spec, None
    inherit_from = spec["inherit_from"]

    if(isinstance(inherit_from, str)):
        temp = inherit_from
        inherit_from = context.type_registry[inherit_from]
        # print(context.type_registry)
        # print("RESOLVE", temp, type(temp), inherit_from,type(inherit_from),)
        # print()
    # print("INHERIT_FROMM",inherit_from)
    if(not isinstance(inherit_from,types.StructRef)):
        inherit_from = context.type_registry[inherit_from._fact_name]
        
        
    if(not hasattr(inherit_from, 'spec')):
        raise ValueError(f"Invalid inherit_from : {inherit_from}")

    inherit_spec = inherit_from.spec

    _intersect = set(inherit_spec.keys()).intersection(set(spec.keys()))
    for k in _intersect:
        if(spec[k]['type'] != inherit_spec[k]['type']): 
            raise TypeError(f"Attribute type {k}:{spec[k]['type']} does not" +
                            f"match inherited attribute {k}:{inherit_spec[k]['type']}")
    del spec['inherit_from']
    return {**inherit_spec, **spec}, inherit_from

def _standardize_spec(spec : dict, context, name=''):
    '''Takes in a spec and puts it in standard form'''

    out = {}
    for attr,v in spec.items():
        if(attr in SPECIAL_ATTRIBUTES): out[attr] = v; continue;

        typ, flags = _get_attr_type_flags(attr,v, context, name)

        #Strings are always nominal
        if(typ == unicode_type and ('nominal' not in flags)): flags.append('nominal')

        out[attr] = {"type": typ, "flags" : flags}

    return out





###### Fact Definition #######
class FactProxy:
    '''Essentially the same as numba.experimental.structref.StructRefProxy 0.51.2
        except that __new__ is not defined to statically define the constructor.
    '''
    __slots__ = ('_fact_type', '_meminfo')

    @classmethod
    def _numba_box_(cls, mi):
        """Called by boxing logic, the conversion of Numba internal
        representation into a PyObject.

        Parameters
        ----------
        ty :
            a Numba type instance.
        mi :
            a wrapped MemInfoPointer.

        Returns
        -------
        instance :
             a FactProxy instance.
        """
        instance = super().__new__(cls)
        # instance._type = BaseFactType
        instance._meminfo = mi
        return instance

    @property
    def _numba_type_(self):
        """Returns the Numba type instance for this structref instance.

        Subclasses should NOT override.
        """
        return self._fact_type

    def __eq__(self, other):
        from cre.dynamic_exec import fact_eq
        if(isinstance(other, FactProxy)):
            return fact_eq(self,other)
        return False

    # def __hash__(self):
    #     from cre.dynamic_exec import fact_hash
    #     return fact_hash(self)

    def get_ptr(self):
        return fact_to_ptr(self)

    def get_ptr_incref(self):
        return fact_to_ptr_incref(self)

    def _gen_val_var_possibilities(self, self_var):
        for attr, config in self._fact_type.spec.items():
            typ = config['type']
            val = getattr(self,attr)
            # with PrintElapse("getattr_var"):
            attr_var = getattr(self_var, attr)
            if(isinstance(val, List)):
                for i in range(len(val)):
                    item_var = attr_var[i]
                    item_val = val[i]
                    yield (item_val, item_var)
            else:
                yield (val, attr_var)
            # else:
                # Primitive case
                # one_lit_conds.append(attr_var==val)

    def as_conditions(self, fact_ptr_to_var_map, keep_null=True, add_implicit_neighbor_self_refs=True):
        from cre.default_ops import Equals
        from cre.conditions import op_to_cond 
        from cre.utils import as_typed_list

        self_ptr = self.get_ptr()
        assert self_ptr in fact_ptr_to_var_map, "'fact_ptr_to_var_map' must include self.get_ptr()."
        self_var = fact_ptr_to_var_map[self_ptr]
        one_lit_conds = []
        
        with PrintElapse("CONSTRUCTS"):       
            # for attr, config in self._fact_type.spec.items():
            for attr_val, attr_var in self._gen_val_var_possibilities(self_var):
                if(isinstance(attr_val, FactProxy)):
                    # Fact case
                    attr_val_fact_ptr = attr_val.get_ptr()
                    if(attr_val_fact_ptr in fact_ptr_to_var_map):
                        val_var = fact_ptr_to_var_map[attr_val_fact_ptr]
                        #   FIXME: use cre_obj.__eq__()
                        
                            # str(attr_var) == str(val_var)

                        if(add_implicit_neighbor_self_refs and str(attr_var) == str(val_var)):
                            # for case like x.next == x.next, try make conditions like x == x.next.prev
                            # with PrintElapse("LOOP"):
                            #     list(attr_val._gen_val_var_possibilities(attr_var))
                            for attr_val2, attr_var2 in attr_val._gen_val_var_possibilities(attr_var):
                                if(isinstance(attr_val2, FactProxy) and 
                                    attr_val2.get_ptr() == self_ptr):
                                    one_lit_conds.append(self_var==attr_var2)

                        else:
                            one_lit_conds.append(attr_var==fact_ptr_to_var_map[val_fact_ptr])
                        
                else:
                    # Primitive case
                    if(not keep_null and attr_val is None): continue
                    one_lit_conds.append(attr_var==attr_val)

        with PrintElapse("ANDS"):        
            conds = one_lit_conds[0]
            for c in one_lit_conds[1:]:
                conds = conds & c

        return conds

    def isa(self, typ):
        return isa(self,typ)



        
        



    # def __setattr__(self,attr,val):
    #     from cre.fact_intrinsics import fact_lower_setattr
    #     fact_lower_setattr(self,attr,val)


def gen_fact_import_str(t):
    return f"from cre_cache.{t._fact_name}._{t._hash_code} import {t._fact_name + 'Type'}"

def gen_inherit_import_str(t):
    return f"from cre_cache.{t._fact_name}._{t._hash_code} import inheritance_bytes as parent_inh_bytes"

def _gen_getter_jit(f_typ,typ,attr):
    if(isinstance(typ,(Fact,DeferredFactRefType))):
        return \
f'''@njit(cache=True)
def {f_typ}_get_{attr}_as_ptr(self):
    return get_fact_attr_ptr(self, '{attr}')

@njit(cache=True)
def {f_typ}_get_{attr}(self):
    return self.{attr}
    #_struct_from_ptr({typ._fact_name}Type, self.{attr})
'''
    else:
        return \
f'''@njit(cache=True)
def {f_typ}_get_{attr}(self):
    return self.{attr}
'''

def _gen_props(typ,attr):
    return f'''    {attr} = property({typ}_get_{attr},lambda s,v : lower_setattr(s,"{attr}",v))'''

# from .structref import _gen_getter, _gen_getter_jit

def get_type_default(t):
    if(isinstance(t,(str,types.UnicodeType))):
        return ""
    elif(isinstance(t,(float,types.Float))):
        return 0.0
    elif(isinstance(t,(int,types.Integer))):
        return 0
    else:
        return None



fact_types = (Fact, DeferredFactRefType)

#### Resolving Byte Offsets of Struct Members ####

def get_offsets_from_member_types(fields):
    # from cre.fact import fact_types, FactModel, BaseFactType
    if(isinstance(fields, dict)): fields = [(k,v) for k,v in fields.items()]
    #Replace fact references with BaseFactType
    # fact_types = (types.StructRef, DeferredFactRefType)
    fields = [(a,BaseFactType if isinstance(t,fact_types) else t) for a,t in fields]

    class TempTypeTemplate(types.StructRef):
        pass

    default_manager.register(TempTypeTemplate, CREObjModel)

    TempType = TempTypeTemplate(fields)

    return [struct_get_attr_offset(TempType,attr) for attr, _ in fields]

# def repr_type(typ):
#     '''Helper function for turning a type into code that reproduces it'''
#     if(isinstance(typ,fact_types)):
#         # To avoid various typing issues fact refs are just i8 (i.e. raw pointers)
#         return 'BaseFactType'
#     elif(isinstance(typ, ListType)):
#         # To avoid various typing issues lists of facts are stored with dtype BaseFactType
#         dt = typ.dtype
#         dtype_repr = f'BaseFactType' if(isinstance(dt,fact_types)) else repr(dt)
#         return f'ListType({dtype_repr})'
#     elif(isinstance(typ, UniTuple)):
#         typ.type 

#     else:
#         return repr(typ)

def repr_fact_attr(inst,fact_name,get_ptr=None):
    # if(isinstance(val,Fact)):
    ptr = get_ptr(inst)
    if(ptr != 0):
        return f'<{fact_name} at {hex(ptr)}>'
    else:
        return 'None'

def repr_list_attr(val,dtype_name=None):
    # if(isinstance(val,Fact)):
    # ptr = get_ptr(val)
    if(val is not None):
        if(dtype_name is not None):
            return f'List([{", ".join([f"<{dtype_name} at {hex(fact_to_ptr(x))}>" for x in val])}])'
        else:
            return f'List([{", ".join([repr(x) for x in val])}])'
    else:
        return 'None'



def gen_repr_attr_code(a,t,typ_name):
    '''Helper function for generating code for the repr/str of the fact'''
    if(isinstance(t,fact_types)):
        return f'{a}={{repr_fact_attr(self,"{t._fact_name}",{typ_name}_get_{a}_as_ptr)}}'
        # return f'{a}=<{t._fact_name} at {{hex({typ_name}_get_{a}_as_ptr(self))}}>'
    elif(isinstance(t,ListType)):
        # TODO : might want to print lists like reference where just the address is printed
        # if():
        s = ", " + f'"{t.dtype._fact_name}"' if isinstance(t.dtype,fact_types) else ""
        # print("FN!!", t.dtype._fact_name)
        return f'{a}={{repr_list_attr(self.{a}{s})}}'
            # s = f'f"<{t.dtype._fact_name} at {{hex(fact_to_ptr(x))}}>"'
            # return f'{a}={{"List([" + ", ".join([{s} for x in self.{a}]) + "])" if self.{a} is not None else "None"}}'
        # else:
            # return f'{a}=List([{{", ".join([repr(x) for x in self.{a}])}}])'
    else:
        return f'{a}={{repr(self.{a})}}' 

# def gen_assign_str(a,t):
#     # if(isinstance(t,fact_types)):
#         # s = f"_ptr_from_struct_incref({a}) if ({a} is not None) else 0"
#     # elif(isinstance(t,ListType) and isinstance(t.dtype,fact_types)):
#     #     # s = f"{a}_c = _cast_list(base_list_type,{a})\n    "
#     #     # s += f"st.{a} = {a}_c" 
#     #     # s = f"st.{a} = _cast_list(base_list_type,{a})" 
#     #     s = f"st.{a} = {a}"
#     # else:
#     s = f"{a}"

#     return f"st.{a} = " + s

@njit(u1[::1](u8),cache=True)
def uint_to_inheritance_bytes(n):
    buffer = np.empty((8,), dtype=np.uint8)
    i = 0
    while(n != 0):
        buffer[8-(i+1)] = n & 0xFF
        i += 1
        n = n >> 8
    return buffer[-i:]

from cre.cre_object import member_info_type
from cre.utils import _sizeof_type, _load_ptr



def _prep_field(attr, t, imports_set):
    
    # elif(isinstance(t,types.ListType) and isinstance(t.dtype,types.StructRef)):
    #     imports_set.add(f"{gen_fact_import_str(t.dtype)}")

    # upcast any facts to BaseFactType since references to undefined fact types not supported
    if(isinstance(t,fact_types)):
        if(isinstance(t,Fact)):
            imports_set.add(f"{gen_fact_import_str(t)}")
        return attr, BaseFactType
    elif(isinstance(t,ListType)):
        if(isinstance(t.dtype,fact_types)):
            _, dtype = _prep_field(attr, t.dtype, imports_set)
            # if(isinstance(dtype, types.Optional)): dtype = dtype.type

            return (attr, ListType(dtype))
        return attr, t
    else:
        return attr, t


def _prep_fields_populate_imports(fields, inherit_from=None):
    imports_set = set()
    if(inherit_from is not None):
        imports_set.add(f"{gen_inherit_import_str(inherit_from)}")
    fields = [_prep_field(attr,t,imports_set) for attr, t in fields]
        
    return fields, "\n".join(list(imports_set))

def gen_fact_src(typ, fields, fact_num, inherit_from=None, ind='    '):
    '''Generate the source code for a new fact '''
    # print(typ.spec)
    fields, fact_imports = _prep_fields_populate_imports(fields, inherit_from)

    _base_fact_field_dict = {**base_fact_field_dict}
    all_fields = [(k,v) for k,v in _base_fact_field_dict.items()] + fields
    # print(all_fields)
    # all_fields = [(k,v) for (k,v) in all_fields]

    # all_fields = base_fact_fields+fields
    properties = "\n".join([_gen_props(typ,attr) for attr,t in all_fields])
    getter_jits = "\n".join([_gen_getter_jit(typ,t,attr) for attr,t in all_fields])
    # field_list = ",".join(["'%s'"%attr for attr,t in fields])

    param_defaults_seq = ",".join([f"{attr}={get_type_default(t)!r}" for attr,t in fields])
    param_seq = ",".join([f"{attr}" for attr,t in fields])
    attr_tup = tuple([attr for attr,t in fields])

    # base_list = ",".join([f"'{k}'" for k,v in _base_fact_fields])
    # base_type_list = ",".join([str(v) for k,v in _base_fact_fields])

    # fact_types = (types.StructRef, DeferredFactRefType)
    
    # field_type_list = ",".join([repr_type(v) for k,v in fields])


    # assign_str = lambda a,t: f"st.{a} = " + (f"_ptr_from_struct_incref({a}) if ({a} is not None) else 0" \
    #                         if isinstance(t,fact_types) else f"{a}")
    init_fields = f'\n{ind}'.join([f"fact_lower_setattr(st,'{k}',{k})" for k,v in fields])

    str_temp = ", ".join([gen_repr_attr_code(k,v,typ) for k,v in fields])

    #TODO get rid of this
    attr_offsets = get_offsets_from_member_types(all_fields)

# The source code template for a user defined fact. Written to the
#  system cache so it can be its own module. Doing so helps njit(cache=True)
#  work when using user defined facts.
    code = \
f'''
import numpy as np
from numba.core import types
from numba import njit, literally, literal_unroll
from numba.core.types import *
from numba.core.types import unicode_type, ListType, UniTuple, Tuple
from numba.experimental import structref
from numba.experimental.structref import new#, define_boxing
from numba.core.extending import overload
from cre.fact_intrinsics import define_boxing, get_fact_attr_ptr, _register_fact_structref, fact_mutability_protected_setattr, fact_lower_setattr, _fact_get_chr_mbrs_infos
from cre.fact import repr_list_attr, repr_fact_attr,  FactProxy, Fact{", BaseFactType, base_list_type, fact_to_ptr, get_inheritance_bytes_len_ptr" if typ != "BaseFact" else ""}, uint_to_inheritance_bytes
from cre.utils import _raw_ptr_from_struct, ptr_t, _get_member_offset, _cast_structref, _load_ptr
import cloudpickle
from cre.cre_object import member_info_type, set_chr_mbrs
{fact_imports}

attr_offsets = np.array({attr_offsets!r},dtype=np.int16)
inheritance_bytes = tuple({"list(parent_inh_bytes) + " if inherit_from else ""}list(uint_to_inheritance_bytes({fact_num}))) 
num_inh_bytes = len(inheritance_bytes)

@_register_fact_structref
class {typ}TypeTemplate(Fact):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._fact_name = '{typ}'
        self._fact_num = {fact_num}
        self._attr_offsets = attr_offsets

    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

field_list = cloudpickle.loads({cloudpickle.dumps(all_fields)})
{typ}Type = {typ}TypeTemplate(field_list)
{typ}Type_w_mbr_infos = {typ}TypeTemplate(field_list+
[("chr_mbrs_infos", UniTuple(member_info_type,{len(fields)})),
 ("num_inh_bytes", u1),
 ("inh_bytes", UniTuple(u1, num_inh_bytes))])


@njit(cache=True)
def get_chr_mbrs_infos():
    st = new({typ}Type)
    return _fact_get_chr_mbrs_infos(st)

chr_mbrs_infos = get_chr_mbrs_infos()
{typ}TypeTemplate._chr_mbrs_infos = chr_mbrs_infos

{(f"""#locals={{'inheritance_bytes':Tuple((u1,num_inh_bytes))}}
@njit(u1(BaseFactType), cache=True)
def isa_{typ}(fact):
    l, p = get_inheritance_bytes_len_ptr(fact)
    if(l >= num_inh_bytes):
        for i,b in enumerate(literal_unroll(inheritance_bytes)):
            f_b = _load_ptr(u1, p+i)
            if(b != f_b):
                return False
        return True
    else:
        return False
""") if typ != "BaseFact" else (
f"""@njit(u1(BaseFactType), cache=True)
def isa_{typ}(fact):
    return True
"""
)
}
{typ}TypeTemplate._isa = isa_{typ}

@njit(cache=True)
def ctor({param_defaults_seq}):
    st = new({typ}Type_w_mbr_infos)
    fact_lower_setattr(st,'idrec',u8(-1))
    fact_lower_setattr(st,'hash_val',0)
    fact_lower_setattr(st,'fact_num',{fact_num})
    set_chr_mbrs(st, {attr_tup!r})
    fact_lower_setattr(st,'num_inh_bytes', num_inh_bytes)
    fact_lower_setattr(st,'inh_bytes', inheritance_bytes)
    {init_fields}
    return _cast_structref({typ}Type,st)


{getter_jits}

@njit(cache=True)
def lower_setattr(self,attr,val):
    fact_mutability_protected_setattr(self,literally(attr),val)

        
class {typ}(FactProxy):
    __numba_ctor = ctor
    _fact_type = {typ}Type
    _fact_name = '{typ}'
    _fact_num = {fact_num}
    _attr_offsets = attr_offsets
    _chr_mbrs_infos = chr_mbrs_infos
    _isa = isa_{typ}
    _code = {typ}Type._code

    def __new__(cls, *args,**kwargs):
        return ctor(*args,**kwargs)

    def __str__(self):
        return f'{typ}({str_temp})'

    def __repr__(self):
        return str(self)



{properties}

@overload({typ})
def _ctor({param_defaults_seq}):
    def impl({param_defaults_seq}):
        return ctor({param_seq})
    return impl

define_boxing({typ}TypeTemplate,{typ})
'''
    return code


# def resolve_fact_attr_type(typ, attr):
#     typ.spec[attr]





# The fact registry is used to give a unique number to each fact definition
#  it is just a text file with <Fact Name> <Hash Code> on each line
def lines_in_fact_registry():
    global GLOBAL_FACT_COUNT
    if(GLOBAL_FACT_COUNT == -1):
        try:
            with open(get_cache_path("fact_registry",suffix=''),'r') as f:
                GLOBAL_FACT_COUNT = len([1 for line in f])
        except FileNotFoundError:
            GLOBAL_FACT_COUNT = 0
    return GLOBAL_FACT_COUNT

def add_to_fact_registry(name,hash_code):
    global GLOBAL_FACT_COUNT
    if(GLOBAL_FACT_COUNT == -1): lines_in_fact_registry()
    with open(get_cache_path("fact_registry",suffix=''),'a') as f:
        f.write(f"{name} {hash_code} \n")
    GLOBAL_FACT_COUNT += 1



def _fact_from_fields(name, fields, context=None, inherit_from=None):
    context = cre_context(context)
    hash_code = unique_hash([name,fields])
    if(not source_in_cache(name,hash_code)):
        fact_num = lines_in_fact_registry()
        source = gen_fact_src(name,fields,fact_num, inherit_from)
        source_to_cache(name, hash_code, source)
        add_to_fact_registry(name, hash_code)

    # print(get_cache_path(name,hash_code))
        
    fact_ctor, fact_type = import_from_cached(name, hash_code,[name,name+"Type"]).values()
    fact_ctor._hash_code = hash_code
    fact_type._hash_code = hash_code
    # fact_type = fact_type_template(fields=fields)
    # print("fact_type",fact_type)

    return fact_ctor, fact_type

def _fact_from_spec(name, spec, context=None, inherit_from=None):
    # assert parent_fact_type
    fields = [(k,v['type']) for k, v in spec.items()]
    return _fact_from_fields(name,fields,context=context,
             inherit_from=inherit_from)

    

def define_fact(name : str, spec : dict, context=None):
    '''Defines a new fact.'''
    context = cre_context(context)

    spec = _standardize_spec(spec,context,name)
    spec, inherit_from = _merge_spec_inheritance(spec,context)

    if(name in context.type_registry):
        # print(str(context.type_registry[name].spec))
        # print(str(spec))
        assert str(context.type_registry[name].spec) == str(spec), \
        f"Redefinition of fact '{name}' in context '{context.name}' not permitted"

        # print(f"FACT REDEFINITION: '{name}' in context '{context.name}' ")
        return context.fact_ctors[name], context.type_registry[name]


    fact_ctor, fact_type = _fact_from_spec(name, spec, context=context, inherit_from=inherit_from)

    # Define the deferred type for this
    dt = context.get_deferred_type(name)
    dt.define(fact_type)

    # If a deffered type was used on specialization then define it
    # for attr,d in spec.items():
    #     dt = d.get('type',None)
    #     if(isinstance(dt,ListType)): dt = dt.dtype
        # if(isinstance(dt,DeferredFactRefType) and dt._fact_name == name):
        #     dt.define(fact_type)
    # for k,v in spec.items():
    #     if(isinstance(v['type'],DeferredFactRefType)): spec[k]['type'] = fact_type
    context._assert_flags(name,spec)
    # print("PASSING IN", inherit_from)
    context._register_fact_type(name,spec,fact_ctor,fact_type,inherit_from=inherit_from)

    # fact_ctor.name = fact_type.name = name
    fact_ctor.spec = fact_type.spec = spec

    return fact_ctor, fact_type


def define_facts(specs, #: list[dict[str,dict]],
                 context=None):
    '''Defines several facts at once.'''
    for name, spec in specs.items():
        define_fact(name,spec,context=context)

###### Base #####

base_fact_field_dict = {
    **cre_obj_field_dict,
    "fact_num": i8,
    # "member_info" : types.UniTuple(member_info_type,1),# sentry type will be as long as there are members
}

base_fact_fields  = [(k,v) for k,v in base_fact_field_dict.items()]

BaseFact, BaseFactType = _fact_from_fields("BaseFact", [])
base_list_type = ListType(BaseFactType)

# @lower_cast(Fact, CREObjType)
@lower_cast(Fact, BaseFactType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)


@njit(cache=True)
def fact_to_ptr(fact):
    return _raw_ptr_from_struct(fact)

@njit(cache=True)
def fact_to_basefact(fact):
    return _struct_from_ptr(BaseFactType,_raw_ptr_from_struct(fact)) 
    # return _cast_structref(BaseFactType, fact)

@njit(cache=True)
def fact_to_ptr_incref(fact):
    return _ptr_from_struct_incref(fact)

# def _fact_eq(a,b):
#     if(isinstance(a,Fact) and isinstance(b,Fact)):
#         def impl(a,b):
#             return _raw_ptr_from_struct(a) ==_raw_ptr_from_struct(b)
#         return impl

# fact_eq = generated_jit(cache=True)(_fact_eq)
# overload(operator.eq)(_fact_eq)


###### Fact Casting #######
@generated_jit
def cast_fact(typ, val):
    '''Casts a fact to a new type of fact if possible'''
    context = cre_context()    
    inst_type = typ.instance_type
    # print("CAST", val._fact_name, "to", inst_type._fact_name)

    #Check if the fact_type can be casted 
    if(inst_type._fact_name != "BaseFact" and val._fact_name != "BaseFact" and
       inst_type._fact_name not in context.children_of[val._fact_name] and 
       inst_type._fact_name not in context.parents_of[val._fact_name]
       
    ):
        error_message = f"Cannot cast fact of type '{val._fact_name}' to '{inst_type._fact_name}.'"
        #If it shouldn't be possible then throw an error
        def error(typ,val):
            raise TypeError(error_message)
        return error
    
    def impl(typ,val):
        return _cast_structref(inst_type,val)

    return impl


@njit(Tuple((u1,i8))(BaseFactType), cache=True)
def get_inheritance_bytes_len_ptr(st):
    ptr = _struct_get_data_ptr(st) + st.chr_mbrs_infos_offset + \
             (st.num_chr_mbrs * _sizeof_type(member_info_type))
    num_inh_bytes = _load_ptr(u1, ptr)
    return num_inh_bytes, ptr+1



@generated_jit(cache=True,nopython=True)
@overload_method(Fact, "isa")
def isa(self, typ):
    # print("<<", typ)
    _isa = typ.instance_type._isa

    def impl(self, typ):
        return _isa(self)
    return impl

#### Hashing ####

# @njit(cache=True)
# def fact_hash():


