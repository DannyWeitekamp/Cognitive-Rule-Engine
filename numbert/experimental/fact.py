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
    intrinsic
)
from numba.core.datamodel import default_manager, models
from numba.core.typing.templates import AttributeTemplate
from numba.core import types, cgutils

from numbert.utils import cache_safe_exec
from numbert.core import TYPE_ALIASES, REGISTERED_TYPES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from numbert.gensource import assert_gen_source
from numbert.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache
from numbert.experimental.structref import gen_structref_code, define_structref
from numbert.experimental.context import kb_context

###### Base #####

base_fact_fields = [
    ("idrec", u8),
    # ("kb", kb)
]
default_base_values = [
    ("idrec", 0),
]

BaseFact, BaseFactType = define_structref("BaseFact", base_fact_fields)

SPECIAL_ATTRIBUTES = ["inherit_from"]

###### Fact Specification Preprocessing #######

def _get_entry_type_flags(attr, v):
    '''A helper function for _standardize_spec'''
    if(isinstance(v,str)):
        typ, flags = v.lower(), []
    elif(isinstance(v,dict)):
        assert "type" in v, "Attribute specifications must have 'type' property, got %s." % v
        typ = v['type'].lower()
        flags = [x.lower() for x in v.get('flags',[])]
    else:
        raise ValueError(f"Spec attribute '{attr}' = '{v}' is not valid type with type {type(v)}.")

    if(typ not in TYPE_ALIASES): 
        raise TypeError(f"Spec attribute type {typ} not recognized on attribute {attr}")

    typ = TYPE_ALIASES[typ]

    return typ, flags

def _merge_spec_inheritance(spec : dict, context):
    '''Expands a spec with attributes from its 'inherit_from' type'''
    if("inherit_from" not in spec): return spec, None
    inherit_from = spec["inherit_from"]

    if(isinstance(inherit_from, str)):
        inherit_from = context.fact_types[inherit_from]
        
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

def _standardize_spec(spec : dict):
    '''Takes in a spec and puts it in standard form'''
    out = {}
    for attr,v in spec.items():
        if(attr in SPECIAL_ATTRIBUTES): out[attr] = v; continue;

        typ, flags = _get_entry_type_flags(attr,v)

        #Strings are always nominal
        if(typ == 'unicode_type' and ('nominal' not in flags)): flags.append('nominal')

        out[attr] = {"type": typ, "flags" : flags}

    return out

###### Fact Definition #######



def _gen_fact_ctor(name,base_fields, fields, ):
    print(base_fields, fields)
    field_names = [x[0] for x in fields]
    base_field_names = [x[0] for x in base_fields]
    all_names = base_field_names + field_names

    params = ', '.join(field_names)
    indent = ' ' * 8
    init_fields_buf = []
    for k in field_names:
        init_fields_buf.append(f"st.{k} = {k}")
    init_fields = f'\n{indent}'.join(init_fields_buf)

    source = f"""
def ctor({params}):
    struct_type = struct_typeclass(list(zip({all_names}, [{params}])))
    def impl({params}):
        st = new(struct_type)
        {init_fields}
        return st
    return impl
"""
    return source

    

# from numba.experimental.structref import _Utils
# def define_attributes(fact_typeclass, fields):
#     """Define attributes on `fact_typeclass`.

#     Defines both setters and getters in jit-code.

#     This is called directly in `register()`.
#     """
#     @infer_getattr
#     class StructAttribute(AttributeTemplate):
#         key = fact_typeclass

#         def generic_resolve(self, typ, attr):
#             if attr in typ.field_dict:
#                 attrty = typ.field_dict[attr]
#                 return attrty

#     @lower_getattr_generic(fact_typeclass)
#     def struct_getattr_impl(context, builder, typ, val, attr):
#         utils = _Utils(context, builder, typ)
#         dataval = utils.get_data_struct(val)
#         ret = getattr(dataval, attr)
#         fieldtype = typ.field_dict[attr]
#         return imputils.impl_ret_borrowed(context, builder, fieldtype, ret)

#     @intrinsic
#     def _struct_set_attr_intrinsic(typingctx, structty, attrty, valty):

#         def codegen(context, builder, sig, args):
#             [inst_type, attr_type, val_type] = sig.args
#             [instance, attr, val] = args
#             utils = _Utils(context, builder, inst_type)
#             dataval = utils.get_data_struct(instance)
#             # cast val to the correct type
#             field_type = inst_type.field_dict[attr]
#             casted = context.cast(builder, val, val_type, field_type)
#             # read old
#             old_value = getattr(dataval, attr)
#             # incref new value
#             context.nrt.incref(builder, val_type, casted)
#             # decref old value (must be last in case new value is old value)
#             context.nrt.decref(builder, val_type, old_value)
#             # write new
#             setattr(dataval, attr, casted)
#         sig = void(structty, attrty, valty)
#         return sig, codegen
        

#     @lower_setattr(fact_typeclass, attr)
#     def _fact_set_attr():
#         return _struct_set_attr_intrinsic()





def _register_fact_structref(fact_type):
    if fact_type is types.StructRef:
        raise ValueError(f"cannot register {types.StructRef}")
    default_manager.register(fact_type, models.StructRefModel)
    define_attributes(fact_type)
    return fact_type


def _fact_from_spec(name, spec, context=None):
    # assert parent_fact_type
    fields = base_fact_fields + [(k,numba_type_map[v['type']]) for k, v in spec.items()]

    hash_code = unique_hash([name,fields])
    if(not source_in_cache(name,hash_code)):
        source = gen_structref_code(name,fields,
            # extra_imports="from numbert.experimental.fact import _register_fact_structref\n" +
            #               "from numba.experimental import structref",
            # register_decorator="@_register_fact_structref"
            )
        source_to_cache(name, hash_code, source)
        
    fact_ctor, fact_type_template = import_from_cached(name, hash_code,[name,name+"TypeTemplate"]).values()
    fact_type = fact_type_template(fields=fields)

    return fact_ctor, fact_type

def define_fact(name : str, spec : dict, context=None):
    '''Defines a new fact.'''
    context = kb_context(context)

    if(name in context.fact_types):
        assert context.fact_types[name].spec == spec, \
        f"Redefinition of fact '{name}' not permitted"

     
    spec = _standardize_spec(spec)
    spec, inherit_from = _merge_spec_inheritance(spec,context)

    fact_ctor, fact_type = _fact_from_spec(name, spec, context=context)

    context._assert_flags(name,spec)
    context._register_fact_type(name,spec,fact_ctor,fact_type,inherit_from=inherit_from)

    fact_ctor.name = fact_type.name = name
    fact_ctor.spec = fact_type.spec = spec

    return fact_ctor, fact_type


def define_facts(specs, #: list[dict[str,dict]],
                 context=None):
    '''Defines several facts at once.'''
    for name, spec in specs.items():
        define_fact(name,spec,context=context)


###### Fact Casting #######

@intrinsic
def _cast_structref(typingctx, cast_type_ref, inst_type):
    # inst_type = struct_type.instance_type
    cast_type = cast_type_ref.instance_type
    def codegen(context, builder, sig, args):
        # [td] = sig.args
        _,d = args

        ctor = cgutils.create_struct_proxy(inst_type)
        dstruct = ctor(context, builder, value=d)
        meminfo = dstruct.meminfo
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        st = cgutils.create_struct_proxy(cast_type)(context, builder)
        st.meminfo = meminfo
        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        # context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        return st._getvalue()
    sig = cast_type(cast_type_ref, inst_type)
    return sig, codegen


@generated_jit
def cast_fact(typ, valty):
    '''Casts a fact to a new type of fact if possible'''
    context = kb_context()
    print("CONTEXT", context)
    inst_type = typ.instance_type

    #Check if the fact_type can be casted 
    if(inst_type is not BaseFactType or valty is not BaseFactType or
       inst_type.name not in context.children_of[name] or 
       inst_type.name not in context.parents_of[name]):
        error_message = f"Cannot cast fact of type '{valty.name}' to '{inst_type.name}.'"
        #If it shouldn't be possible then throw an error
        def error(typ,val):
            raise TypeError(error_message)
        return error
    
    def impl(typ,val):
        return _cast_structref(inst_type,val)

    return impl



