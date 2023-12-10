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

from cre.new_fact.fact import FactTypeClass, FactProxy
from cre.new_fact.spec import standardize_spec, merge_spec_inheritance
from cre.new_fact.gen_src import gen_fact_src
# from cre.new_fact.fact_intrinsics import FactTypeClass

# def _fact_from_fields(name, fields, inherit_from=None, specialization_name=None, is_untyped=False, return_proxy=False, return_type_class=False):
#     # context = cre_context(context)
    

#     hash_code = unique_hash_v([name,fields])
#     if(not source_in_cache(name,hash_code)):
#         # Possible for other types to be defined while running the Fact source
#         #  so preregister the t_id then add the pickle later.
#         t_id = add_to_type_registry(name, hash_code)
#         source = gen_fact_src(name, fields, t_id, inherit_from, specialization_name, is_untyped, hash_code)
#         source_to_cache(name, hash_code, source)
        
#         fact_type = tuple(import_from_cached(name, hash_code, [name]).values())[0]
#         add_type_pickle(fact_type, t_id)
        
#     to_get = [name]
#     if(return_proxy): to_get.append(name+"Proxy")
#     if(return_type_class): to_get.append(name+"Class")
        
#     out = tuple(import_from_cached(name, hash_code, to_get).values())
#     for x in out: x._hash_code = hash_code
        
#     return tuple(out) if len(out) > 1 else out[0]

def _fact_from_spec(name, spec, inherit_from=None, return_proxy=False, return_type_class=False):

    fields = [(k,v['type']) for k, v in spec.items()] if spec else {}
    hash_code = unique_hash_v([name, fields])
    fact_type = FactTypeClass(name, spec, hash_code)    

    hash_code = unique_hash_v([name,fields])
    if(not source_in_cache(name,hash_code)):
        # Possible for other types to be defined while running the Fact source
        #  so preregister the t_id then add the pickle later.
        t_id = add_to_type_registry(name, hash_code)
        source = gen_fact_src(name, fields, t_id, inherit_from, None, False, hash_code)
        source_to_cache(name, hash_code, source)
        
        add_type_pickle(fact_type, t_id)
        
    to_get = [name]
    if(return_proxy): to_get.append(name+"Proxy")
    if(return_type_class): to_get.append(name+"Class")
        
    out = tuple(import_from_cached(name, hash_code, to_get).values())

    return tuple(out) if len(out) > 1 else out[0]

    # return tuple(out.values())
    # print(tuple([fact_type, *out.values()]))
    

    # for x in out: x._hash_code = hash_code
        
    # return tuple(out) if len(out) > 1 else out[0]

    # define_boxing({typ}Class,{typ}Proxy)

    # return _fact_from_fields(name, fields, inherit_from=inherit_from, 
    #         is_untyped=is_untyped, return_proxy=return_proxy,
    #         return_type_class=return_type_class)




def define_fact(name : str, spec : dict = None, context=None, return_proxy=False, return_type_class=False, allow_redef=False):
    '''Defines a new fact.'''

    from cre.context import cre_context
    context = cre_context(context)
    # print("DEFINE", name, context.name)
    if(spec is not None):
        spec = standardize_spec(spec,context,name)
        spec, inherit_from = merge_spec_inheritance(spec, context)        
    else:
        inherit_from = None


    if(name in context.name_to_type):
        assert _spec_eq(context.name_to_type[name].spec, spec), \
        f"Redefinition of fact '{name}' in context '{context.name}' not permitted"
        # print("SPECIALIZATION NAME:", specialization_name)
        fact_type = context.name_to_type[name]
    else:

        fact_type = _fact_from_spec(name, spec, inherit_from=inherit_from, 
            return_proxy=False, return_type_class=False)
        dt = context.get_deferred_type(name)
        dt.define(fact_type)
        # context._assert_flags(name, spec)
        context._register_fact_type(name, fact_type, inherit_from=inherit_from)
    _spec = spec if(spec is not None) else {}
    # _spec = _undeffer_spec(_spec)
    # print({_id : str(config['type']) for _id, config in _spec.items()})
    fact_type.spec = _spec
    fact_type._proxy_class.spec = _spec

    # Needs to be done because different definitions can share a 
    #  fact_type object
    if(hasattr(fact_type,'_clean_spec')): del fact_type._clean_spec

    out = [fact_type]
    if(return_proxy): out.append(fact_type._proxy_class)
    if(return_type_class): out.append(fact_type._fact_type_class)
    return tuple(out) if len(out) > 1 else out[0]
    # return fact_ctor, fact_type


if __name__ == "__main__":
    import faulthandler; faulthandler.enable()
    BOOP = define_fact("BOOP", {"A" :str, "B" : int})

    print(BOOP())

    b = BOOP("A",1)
    b[0] = "B"
    b[1] = 2

    print(b[0])
    print(b[1])
    

    b.A = "A"
    b.B = 1
    print(b.A, b.B)
    print(type(b), b)

    b.idrec = "100"

    b[0] = "C"
    print(b, b.idrec)


    print("------------------")
    ## Test assign jitted
    from cre.new_fact.fact import new_fact
    @njit(cache=False)
    def assign_jitted():
        b = BOOP("", 1)
        b.A = "A"

        print(b.A)
        print(b.A)
        print(b[0])
        print(b[0])
        return b#cast(new_fact(2), BOOP)

    assign_jitted()

    print("------------------")

    # b[0] = 1
    with PrintElapse("Loop_10000"):
        for i in range(10000):
            pass


    # from cre.new_fact.fact import fact_get_str, fact_getitem_impl
    # f = fact_getitem_impl(unicode_type)
    # with PrintElapse("Geti_10000_str"):
    #     for i in range(10000):
    #         f(b, 0)

    from cre.new_fact.fact import new_fact
    # @njit(cache=False)
    # def alloca_fact():
    #     return BOOP("", 1)#cast(new_fact(2), BOOP)

    # print(alloca_fact())
    print(BOOP())

    # with PrintElapse("alloca_fact"):
    #     for i in range(10000):
    #         alloca_fact()
            # b.A = "A"
            # b.B = 1

    with PrintElapse("BOOP_ctor_empty"):
        for i in range(10000):
            BOOP()

    with PrintElapse("BOOP_ctor"):
        for i in range(10000):
            BOOP("A", B=1)

    with PrintElapse("Geti_10000_str"):
        for i in range(10000):
            x = b[0]

    with PrintElapse("Geti_10000_int"):
        for i in range(10000):
            x = b[1]

    with PrintElapse("Get_10000_str"):
        for i in range(10000):
            x = b.A

    with PrintElapse("Get_10000_int"):
        for i in range(10000):
            x = b.B



