from numba import types, njit, guvectorize, vectorize, prange, generated_jit
from numba.experimental import jitclass, structref
from numba import deferred_type, optional, objmode
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
from numba.core.typing import signature

# from numba.core.extending import overload

from cre.core import TYPE_ALIASES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map, register_global_default, lines_in_type_registry, add_to_type_registry, add_type_pickle
from cre.context import cre_context
from cre.caching import unique_hash_v, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.structref import gen_structref_code, define_structref
# from cre.context import cre_context
from cre.utils import (cast, struct_get_attr_offset, _obj_cast_codegen, 
                       _ptr_from_struct_codegen, CastFriendlyMixin, _obj_cast_codegen,
                       PrintElapse, _struct_get_data_ptr, _ptr_from_struct_incref,
                       deref_info_type, DEREF_TYPE_ATTR, DEREF_TYPE_LIST,
                       _ptr_to_data_ptr, _list_base_from_ptr)
from cre.obj import CREObjTypeClass, cre_obj_field_dict, CREObjModel, CREObjType, member_info_type, CREObjProxy

from numba.core.typeconv import Conversion
import operator
from numba.core.imputils import (lower_cast)
import cloudpickle
import numpy as np

from cre.new_fact.fact import FactTypeClass
from cre.fact import DeferredFactRefType


def gen_fact_import_str(t):
    return f"from cre_cache.{t._fact_name}._{t._hash_code} import {t._fact_name}"

def gen_inherit_import_str(t):
    return f"from cre_cache.{t._fact_name}._{t._hash_code} import {t._fact_name} as parent_type, inheritance_bytes as parent_inh_bytes"

def _gen_getter_jit(f_typ,typ,attr):
    if(isinstance(typ,(Fact,DeferredFactRefType))):
        return \
f'''@njit(cache=True)
def get_{attr}_as_ptr(self):
    return get_fact_attr_ptr(self, '{attr}')

@njit(cache=True)
def get_{attr}(self):
    return self.{attr}
'''
    else:
        return \
f'''@njit(cache=True)
def get_{attr}(self):
    return self.{attr}
'''

def _gen_setter_jit(f_typ, attr, a_id):
    return f'''@njit(types.void({f_typ},field_list[{a_id}][1]), cache=True)
def set_{attr}(self, val):
    fact_mutability_protected_setattr(self,'{attr}',val)
'''

# def _gen_props(attr):
#     return f'''    {attr} = property(get_{attr}, set_{attr})'''

# def _gen_props(attr, index, typ):
#     from cre.new_fact.fact import fact_getitem_impl, fact_setitem_impl
    

#     return f'''    {attr} = property(get_{attr}, set_{attr})'''

def get_type_default(t):
    if(isinstance(t,(bool,types.Boolean))):
        return False
    elif(isinstance(t,(str,types.UnicodeType))):
        return ""
    elif(isinstance(t,(float,types.Float))):
        return 0.0
    elif(isinstance(t,(int,types.Integer))):
        return 0
    else:
        return None



fact_types = (FactTypeClass, DeferredFactRefType)

#### Resolving Byte Offsets of Struct Members ####

def get_offsets_from_member_types(fields):
    # from cre.fact import fact_types, FactModel, BaseFact
    if(isinstance(fields, dict)): fields = [(k,v) for k,v in fields.items()]
    #Replace fact references with BaseFact
    # fact_types = (types.StructRef, DeferredFactRefType)
    fields = [(a,BaseFact if isinstance(t,fact_types) else t) for a,t in fields]

    class TempTypeTemplate(types.StructRef):
        pass

    default_manager.register(TempTypeTemplate, CREObjModel)

    TempType = TempTypeTemplate(fields)

    return [struct_get_attr_offset(TempType,attr) for attr, _ in fields]


def repr_fact_attr(inst):
    if(inst is None): return 'None'

    inst_type = type(inst)
    # cre_context().get
    # print(isinstance(inst, Fact), str(inst_type))
    if(hasattr(inst_type,"_fact_type") and
        hasattr(inst_type._fact_type, "_specialization_name")):
        return str(inst)

    ptr = inst.get_ptr()
    if(ptr != 0):
        return f"<{inst_type._fact_name} at {hex(ptr)}>"
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



def gen_repr_attr_code(a,t):
    '''Helper function for generating code for the repr/str of the fact'''
    if(isinstance(t,fact_types)):
        return f'{a}={{repr_fact_attr(self.{a})}}'
        # return f'{a}=<{t._fact_name} at {{hex({typ_name}_get_{a}_as_ptr(self))}}>'
    elif(isinstance(t,ListType)):
        # TODO : might want to print lists like reference where just the address is printed
        # if():
        s = ", " + f'"{t.dtype._fact_name}"' if isinstance(t.dtype,fact_types) else ""
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

from cre.obj import member_info_type
from cre.utils import _sizeof_type, _load_ptr



def _prep_field(attr, t, imports_set):
    from cre.tuple_fact import TupleFactClass
    # elif(isinstance(t,types.ListType) and isinstance(t.dtype,types.StructRef)):
    #     imports_set.add(f"{gen_fact_import_str(t.dtype)}")

    # upcast any facts to BaseFact since references to undefined fact types not supported
    if(isinstance(t,fact_types)):
        if(isinstance(t,Fact) and not isinstance(t,TupleFactClass)):
            imports_set.add(f"{gen_fact_import_str(t)}")
        return attr, BaseFact
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

def gen_fact_src(typ, fields, t_id, inherit_from=None, specialization_name=None, is_untyped=False, hash_code="", ind='    '):
    '''Generate the source code for a new fact '''
    fields, fact_imports = _prep_fields_populate_imports(fields, inherit_from)

    # _base_fact_field_dict = {**base_fact_field_dict}
    # all_fields = [(k,v) for k,v in _base_fact_field_dict.items()] + fields    # all_fields = [(k,v) for (k,v) in all_fields]
    all_fields = ""
    # all_fields = base_fact_fields+fields
    properties = "\n".join([_gen_props(attr) for attr,t in all_fields])
    getter_jits = ""#"\n".join([_gen_getter_jit(typ,t,attr) for attr,t in all_fields])
    setter_jits = ""#"\n".join([_gen_setter_jit(typ,attr,a_id) for a_id, (attr,t) in enumerate(all_fields)])
    # field_list = ",".join(["'%s'"%attr for attr,t in fields])

    param_defaults_seq = ",".join([f"{attr}={get_type_default(t)!r}" for attr,t in fields])
    param_seq = ",".join([f"{attr}" for attr,t in fields])
    attr_tup = tuple([attr for attr,t in fields])


    as_dict_setters = []
    for attr,t in fields:
        print(attr, t)
        if(isinstance(t, FactTypeClass)):
            s = f"'{attr}':getattr(get_{attr}(self), key_attr, None)"
        elif(isinstance(t, ListType)):
            if(isinstance(t.item_type, Fact)):
                s = f"'{attr}':[getattr(x, key_attr, None) for x in get_{attr}(self)]"
            else:
                s = f"'{attr}':list(get_{attr}(self))"
        else:
            s = f"'{attr}':get_{attr}(self)"

        as_dict_setters.append(s)
    as_dict_setters = ", ".join(as_dict_setters)

    # base_list = ",".join([f"'{k}'" for k,v in _base_fact_fields])
    # base_type_list = ",".join([str(v) for k,v in _base_fact_fields])

    # fact_types = (types.StructRef, DeferredFactRefType)
    
    # field_type_list = ",".join([repr_type(v) for k,v in fields])


    # assign_str = lambda a,t: f"st.{a} = " + (f"_ptr_from_struct_incref({a}) if ({a} is not None) else 0" \
    #                         if isinstance(t,fact_types) else f"{a}")
    init_fields = f'\n{ind}'.join([f"fact_lower_setattr(st,'{k}',{k})" for k,v in fields])

    str_temp = ", ".join([gen_repr_attr_code(k,v) for k,v in fields])

    #TODO get rid of this

    attr_offsets = ""#get_offsets_from_member_types(all_fields)

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
from numba.core.extending import overload, overload_method, lower_cast, type_callable
from numba.core.imputils import numba_typeref_ctor
#from cre.fact import repr_list_attr, repr_fact_attr, FactProxy, Fact, UntypedFact{", BaseFact, base_list_type, fact_to_ptr, get_inheritance_bytes_len_ptr" if typ != "BaseFact" else ""}, uint_to_inheritance_bytes


from cre.utils import cast, ptr_t, _get_member_offset,  _load_ptr, _obj_cast_codegen, encode_idrec
import cloudpickle
from cre.obj import member_info_type, set_chr_mbrs

from cre.new_fact.fact import FactProxy, FactTypeClass, new_fact, make_proxy_properties, define_attributes, fact_mutability_protected_setattr, fact_lower_setattr
#from cre.new_fact.fact_intrinsics import define_boxing, get_fact_attr_ptr, _register_fact_structref, fact_mutability_protected_setattr, fact_lower_setattr, _fact_get_chr_mbrs_infos

{fact_imports}


field_list = cloudpickle.loads({cloudpickle.dumps(fields)})
{typ} = fact_type = FactTypeClass('{typ}', field_list, {hash_code!r})
n_members = len(field_list)
define_attributes({typ})

@njit
def ctor({param_defaults_seq}):
    st = cast(new_fact(n_members), {typ})
    fact_lower_setattr(st,'idrec',encode_idrec({t_id},0,u1(-1)))
    fact_lower_setattr(st,'hash_val',0)
    {init_fields}
    return st
{typ}._ctor = (ctor,)

# Overload '{typ}' as its own constructor
@type_callable({typ})
def ssp_call(context):
    {"raise NotImplementedError('NotImplementedError: Cannot initialize UntypedFact in jitted context.')" if is_untyped else ""}
    # Note to self this requires *args, see https://github.com/numba/numba/issues/7973
    def typer({param_seq}):    
        {f'return {typ}.specialize({param_seq})' if is_untyped else f'return {typ}'}
    return typer

@overload(numba_typeref_ctor)
def overload_{typ}(self, {param_defaults_seq}):
    if(self.instance_type is not {typ}): return
    def impl(self, {param_defaults_seq}):
        return ctor({param_seq})
    return impl

{(f"""{typ}.parent_type = parent_type
pt = parent_type
while(pt is not None):
    @lower_cast({typ}, pt)
    def upcast(context, builder, fromty, toty, val):
        return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)                        
    pt = getattr(pt, 'parent_type', None)
""") if inherit_from is not None else ""
}


# Define Proxy
class {typ}Proxy(FactProxy):
    __numba_ctor = ctor
    _fact_type = {typ}
    _fact_name = '{typ}'
    t_id = {t_id}
    #_isa = isa_{typ}
    # _code = {typ}._code
    #_hash_code = '{hash_code}'

    def __new__(cls, *args,**kwargs):
        return ctor(*args,**kwargs)

    def __repr__(self):
        return f'{typ}({str_temp})'

    def __str__(self):
        return self.__repr__()

    def as_dict(self, key_attr='idrec'):
        return {{'type' : '{typ}', {as_dict_setters}}}

    def __getitem__(self, i):
        return self._getters[i](self, i)

    def __setitem__(self, i, val):
        self._setters[i](self, i, val)

make_proxy_properties({typ}Proxy, field_list)

{typ}.t_id = {t_id}
{typ}._proxy_class = {typ}Proxy


#define_boxing({typ}Class,{typ}Proxy)



{(f"""from cre.var import VarType, var_ctor
# @njit(VarType(unicode_type), cache=True)
# def as_var(alias):
#     return var_ctor({typ}, {t_id}, alias)
# {typ}._as_var = as_var
""") if typ != "BaseFact" else ""
}


'''
    return code

