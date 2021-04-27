from numba import types, njit, guvectorize, vectorize, prange, generated_jit
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
    overload
)
from numba.core.datamodel import default_manager, models
from numba.core.typing.templates import AttributeTemplate
from numba.core import types, cgutils
# from numba.core.extending import overload

from cre.core import TYPE_ALIASES, REGISTERED_TYPES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from cre.gensource import assert_gen_source
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.structref import gen_structref_code, define_structref
from cre.context import kb_context
from cre.utils import _cast_structref

import numpy as np

GLOBAL_FACT_COUNT = -1
SPECIAL_ATTRIBUTES = ["inherit_from"]

###### Fact Specification Preprocessing #######
class DefferedFactRefType():
    '''A placeholder type for when referencing a fact type that
        is not defined yet.'''
    def __init__(self,typ):
        self._fact_name = typ._fact_name if isinstance(typ,types.StructRef) else typ 
    def __equal__(self,other):
        return isinstance(other,DefferedRefType) \
               and self._fact_name == other._fact_name
    def __str__(self):
        return f"DefferedFactRefType[{self._fact_name}]"


def _get_type(typ, context, name='', attr=''):
    '''Takes in a string or type and returns the type'''
    if(isinstance(typ,str)):
        typ_str = typ
        if(typ_str.lower() in TYPE_ALIASES): 
            typ = numba_type_map[TYPE_ALIASES[typ_str.lower()]]
        elif(typ_str in context.fact_types):
            typ = context.fact_types[typ_str]
        elif(typ_str == name):
            typ = DefferedFactRefType(name)
        else:
            raise TypeError(f"Attribute type {typ_str!r} not recognized in spec" + 
                f" for attribute definition {attr!r}." if attr else ".")
    if(hasattr(typ, "_fact_type")): typ = typ._fact_type
    return typ


def _get_attr_type_flags(attr, v, context, name=''):
    '''A helper function for _standardize_spec'''

    # Extract typ_str + flags
    
    if(isinstance(v,dict)):
        assert "type" in v, "Attribute specifications must have 'type' property, got %s." % v
        typ = _get_type(v['type'], context, name, attr)
        flags = [x.lower() for x in v.get('flags',[])]
    else:
        typ, flags = _get_type(v, context, name, attr), []
    
    print(typ, flags)

    return typ, flags

def _merge_spec_inheritance(spec : dict, context):
    '''Expands a spec with attributes from its 'inherit_from' type'''
    if("inherit_from" not in spec): return spec, None
    inherit_from = spec["inherit_from"]

    if(isinstance(inherit_from, str)):
        temp = inherit_from
        inherit_from = context.fact_types[inherit_from]
        # print(context.fact_types)
        # print("RESOLVE", temp, type(temp), inherit_from,type(inherit_from),)
        # print()
    # print("INHERIT_FROMM",inherit_from)
    if(not isinstance(inherit_from,types.StructRef)):
        inherit_from = context.fact_types[inherit_from._fact_name]
        
        
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
        if(typ == 'unicode_type' and ('nominal' not in flags)): flags.append('nominal')

        out[attr] = {"type": typ, "flags" : flags}

    return out

###### Fact Definition #######
class FactProxy:
    '''Essentially the same as numba.experimental.structref.StructRefProxy 0.51.2
        except that __new__ is not defined to statically define the constructor.
    '''
    __slots__ = ('_type', '_meminfo')

    @classmethod
    def _numba_box_(cls, ty, mi):
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
        instance._type = ty
        instance._meminfo = mi
        return instance

    @property
    def _numba_type_(self):
        """Returns the Numba type instance for this structref instance.

        Subclasses should NOT override.
        """
        return self._type

def gen_fact_import_str(t):
    return f"from cre_cache.{t._fact_name}._{t._hash_code} import {t._fact_name + 'Type'}"

def _gen_getter_jit(f_typ,typ,attr):
    if(isinstance(typ,(types.StructRef,DefferedFactRefType))):
        return \
f'''@njit(cache=True)
def {f_typ}_get_{attr}_as_ptr(self):
    return self.{attr}

@njit(cache=True)
def {f_typ}_get_{attr}(self):
    return _struct_from_pointer({typ._fact_name}Type, self.{attr})
'''
    else:
        return \
f'''@njit(cache=True)
def {f_typ}_get_{attr}(self):
    return self.{attr}
'''

def _gen_getter(typ,attr):
    return f'''    @property
    def {attr}(self):
        return {typ}_get_{attr}(self)
    '''

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


def gen_fact_code(typ, fields, fact_num, ind='    '):
    '''Generate the source code for a new fact '''
    fact_imports = ""
    for attr,t in fields:
        if(isinstance(t,types.StructRef)):
            fact_imports += f"{gen_fact_import_str(t)}\n"


    all_fields = base_fact_fields+fields
    getters = "\n".join([_gen_getter(typ,attr) for attr,t in all_fields])
    getter_jits = "\n".join([_gen_getter_jit(typ,t,attr) for attr,t in all_fields])
    field_list = ",".join(["'%s'"%attr for attr,t in fields])
    param_list = ",".join([f"{attr}={get_type_default(t)!r}" for attr,t in fields])
    base_list = ",".join([f"'{k}'" for k,v in base_fact_fields])
    base_type_list = ",".join([str(v) for k,v in base_fact_fields])

    fact_types = (types.StructRef, DefferedFactRefType)

    # To avoid various typing issues fact refs are just i8 (i.e. raw pointers)
    typ_str = lambda x: f"int64" if isinstance(x,fact_types) else str(x)
    field_type_list = ",".join([typ_str(v)  for k,v in fields])


    assign_str = lambda a,t: f"st.{a} = " + (f"_pointer_from_struct_incref({a}) if ({a} is not None) else 0" \
                            if isinstance(t,fact_types) else f"{a}")
    init_fields = f'\n{ind}'.join([assign_str(k,v) for k,v in fields])

    temp_str_f = lambda a, t: f'{a}=<{t._fact_name} at {{hex({typ}_get_{a}_as_ptr(self))}}>' \
                            if(isinstance(t,fact_types))  else f'{a}={{repr(self.{a})}}' 
    str_temp = ", ".join([temp_str_f(k,v) for k,v in fields])



# The source code template for a user defined fact. Written to the
#  system cache so it can be its own module. Doing so helps njit(cache=True)
#  work when using user defined facts.

    code = \
f'''
import numpy as np
from numba.core import types
from numba import njit
from numba.core.types import *
from numba.core.types import unicode_type
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing
from numba.core.extending import overload
from cre.fact import _register_fact_structref, FactProxy
from cre.utils import struct_get_attr_offset, _pointer_from_struct_incref, _struct_from_pointer
{fact_imports}

attr_offsets = np.empty(({len(all_fields)},),dtype=np.int16)

@_register_fact_structref
class {typ}TypeTemplate(types.StructRef):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._fact_name = '{typ}'
        self._fact_num = {fact_num}
        self._attr_offsets = attr_offsets

    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

{typ}Type = {typ}TypeTemplate(list(zip([{base_list},{field_list}], [{base_type_list},{field_type_list}])))


#TODO: This is an expensive operation, ~5ms/iter, consider combining
for i,attr in enumerate([{base_list},{field_list}]):
    attr_offsets[i] = struct_get_attr_offset({typ}Type,attr)

@njit(cache=True)
def ctor({param_list}):
    st = new({typ}Type)
    st.idrec = -1
    st.fact_num = {fact_num}
    {init_fields}
    return st

{getter_jits}

class {typ}(FactProxy):
    __numba_ctor = ctor
    _fact_type = {typ}Type
    _fact_name = '{typ}'
    _fact_num = {fact_num}
    _attr_offsets = attr_offsets

    def __new__(cls, *args,**kwargs):
        return ctor(*args,**kwargs)

    def __str__(self):
        return f'{typ}({str_temp})'

{getters}

@overload({typ})
def _ctor(*args):
    def impl(*args):
        return ctor(*args)
    return impl

define_boxing({typ}TypeTemplate,{typ})
'''
    return code

    

from numba.experimental.structref import _Utils, imputils
def define_attributes(struct_typeclass):
    """
    Copied from numba.experimental.structref 0.51.2, but added protected mutability
    """
    @infer_getattr
    class StructAttribute(AttributeTemplate):
        key = struct_typeclass

        def generic_resolve(self, typ, attr):
            if attr in typ.field_dict:
                attrty = typ.field_dict[attr]
                return attrty

    @lower_getattr_generic(struct_typeclass)
    def struct_getattr_impl(context, builder, typ, val, attr):
        utils = _Utils(context, builder, typ)
        dataval = utils.get_data_struct(val)
        ret = getattr(dataval, attr)
        fieldtype = typ.field_dict[attr]
        return imputils.impl_ret_borrowed(context, builder, fieldtype, ret)

    @lower_setattr_generic(struct_typeclass)
    def struct_setattr_impl(context, builder, sig, args, attr):
        [inst_type, val_type] = sig.args
        [instance, val] = args
        utils = _Utils(context, builder, inst_type)
        dataval = utils.get_data_struct(instance)
        # cast val to the correct type
        field_type = inst_type.field_dict[attr]
        casted = context.cast(builder, val, val_type, field_type)

        pyapi = context.get_python_api(builder)

        # print("BBB", attr,[x[0] for x in base_fact_fields], attr in [x[0] for x in base_fact_fields])

        if(attr not in [x[0] for x in base_fact_fields]):
            idrec = getattr(dataval, "idrec")
            idrec_set = builder.icmp_signed('!=', idrec, idrec.type(-1))
            with builder.if_then(idrec_set):
                msg =("Facts objects are immutable once declared. Use kb.modify instead.",)
                context.call_conv.return_user_exc(builder, AttributeError, msg)
            
        # read old
        old_value = getattr(dataval, attr)
        # incref new value
        context.nrt.incref(builder, val_type, casted)
        # decref old value (must be last in case new value is old value)
        context.nrt.decref(builder, val_type, old_value)
        # write new
        setattr(dataval, attr, casted)


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


def _register_fact_structref(fact_type):
    if fact_type is types.StructRef:
        raise ValueError(f"cannot register {types.StructRef}")
    default_manager.register(fact_type, models.StructRefModel)
    define_attributes(fact_type)
    return fact_type

def _fact_from_fields(name, fields, context=None):
    context = kb_context(context)
    hash_code = unique_hash([name,fields])
    if(not source_in_cache(name,hash_code)):
        fact_num = lines_in_fact_registry()
        source = gen_fact_code(name,fields,fact_num)
        source_to_cache(name, hash_code, source)
        add_to_fact_registry(name, hash_code)
        
    fact_ctor, fact_type = import_from_cached(name, hash_code,[name,name+"Type"]).values()
    fact_ctor._hash_code = hash_code
    fact_type._hash_code = hash_code
    # fact_type = fact_type_template(fields=fields)
    # print("fact_type",fact_type)

    return fact_ctor, fact_type

def _fact_from_spec(name, spec, context=None):
    # assert parent_fact_type
    fields = [(k,v['type']) for k, v in spec.items()]
    return _fact_from_fields(name,fields,context=context)

    

def define_fact(name : str, spec : dict, context=None):
    '''Defines a new fact.'''
    context = kb_context(context)

    spec = _standardize_spec(spec,context,name)
    spec, inherit_from = _merge_spec_inheritance(spec,context)

    if(name in context.fact_types):
        print(str(context.fact_types[name].spec))
        print(str(spec))
        assert str(context.fact_types[name].spec) == str(spec), \
        f"Redefinition of fact '{name}' in context '{context.name}' not permitted"

        print(f"FACT REDEFINITION: '{name}' in context '{context.name}' ")
        return context.fact_ctors[name], context.fact_types[name]


    fact_ctor, fact_type = _fact_from_spec(name, spec, context=context)
    # for k,v in spec.items():
    #     if(isinstance(v['type'],DefferedFactRefType)): spec[k]['type'] = fact_type
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

base_fact_fields = [
    ("idrec", u8),
    ("fact_num", i8)
    # ("kb", kb)
]

BaseFact, BaseFactType = _fact_from_fields("BaseFact", [])

###### Fact Casting #######




@generated_jit
def cast_fact(typ, val):
    '''Casts a fact to a new type of fact if possible'''
    context = kb_context()    
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




    



