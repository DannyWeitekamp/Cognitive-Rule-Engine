import numpy as np
from numba import types, njit, i8, u8, i4, u1, literally, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type, void
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr
from numba.core.typing.templates import AttributeTemplate
from numbert.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from numbert.experimental.context import kb_context
from numbert.experimental.structref import define_structref, define_structref_template
from numbert.experimental.kb import KnowledgeBaseType, KnowledgeBase, facts_for_t_id, fact_at_f_id
from numbert.experimental.fact import define_fact, BaseFactType, cast_fact
from numbert.experimental.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, decode_idrec, lower_getattr, _struct_from_pointer
from numbert.experimental.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from numbert.experimental.vector import VectorType
from numbert.experimental.predicate_node import BasePredicateNode
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models

from copy import copy


BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

var_fields_dict = {
    'fact_type': types.Any,
    'deref_attrs': types.Any,
}

var_fields =  [(k,v) for k,v, in var_fields_dict.items()]

class VarTypeTemplate(types.StructRef):
    pass

# Manually register the type to avoid automatic getattr overloading 
default_manager.register(VarTypeTemplate, models.StructRefModel)

def var_str_from_type(inst_type):
    fn = inst_type.field_dict['fact_type'].instance_type._fact_name
    attr_str = inst_type.field_dict['deref_attrs'].literal_value
    return f'Var[{fn}]{attr_str}'

class Var(structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)

    def __getattr__(self, attr):
        if(attr == 'fact_type'):
            return self._numba_type_.field_dict['fact_type'].instance_type
        elif(attr == 'deref_attrs'):
            return self._numba_type_.field_dict['deref_attrs'].literal_value
        elif(True): #TODO
            return Var(self.fact_type,types.literal(self.deref_attrs+f'.{attr}'))

    def __str__(self):
        return var_str_from_type(self._numba_type_)


# Manually define the boxing to avoid constructor overloading
define_boxing(VarTypeTemplate,Var)

@overload(Var,strict=False,prefer_literal=False)
def ctor(typ,attr_chain_str=types.literal('')):
    if(not isinstance(attr_chain_str, types.Literal)):
        return 

    struct_type = VarTypeTemplate([('fact_type', typ), ('deref_attrs', attr_chain_str)])
    if(len(attr_chain_str.literal_value) > 0):
        def impl(typ,attr_chain_str):
            return new(struct_type)
    else:
        def impl(typ):
            return new(struct_type)
    return impl

def resolve_deref_type(inst_type, attr):
    old_str = inst_type.field_dict['deref_attrs'].literal_value
    fact_type = inst_type.field_dict['fact_type']
    new_str = old_str + f".{attr}"
    new_struct_type = VarTypeTemplate([('fact_type', fact_type), ('deref_attrs', types.literal(new_str))])    
    return new_struct_type

@overload(str)
def str_var(inst_type):
    if(not isinstance(inst_type, VarTypeTemplate)): return
    str_val = var_str_from_type(inst_type)
    def impl(typ, attr_chain_str):
        return str_val

    return impl

#### Get Attribute Overloading ####

@infer_getattr
class StructAttribute(AttributeTemplate):
    key = VarTypeTemplate
    def generic_resolve(self, typ, attr):
        if attr in typ.field_dict:
            attrty = typ.field_dict[attr]
            return attrty
        #TODO Should check that all subtype references are valid
        elif(attr in typ.field_dict['fact_type'].instance_type.field_dict):
            return resolve_deref_type(typ, attr)

@lower_getattr_generic(VarTypeTemplate)
def struct_getattr_impl(context, builder, typ, val, attr):
    if(attr in var_fields_dict):
        utils = _Utils(context, builder, typ)
        dataval = utils.get_data_struct(val)
        ret = getattr(dataval, attr)
        fieldtype = typ.field_dict[attr]
        return imputils.impl_ret_borrowed(context, builder, fieldtype, ret)
    else:
        new_struct_type = resolve_deref_type(typ,attr)

        ctor = cgutils.create_struct_proxy(typ)
        dstruct = ctor(context, builder, value=val)
        meminfo = dstruct.meminfo
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        st = cgutils.create_struct_proxy(new_struct_type)(context, builder)
        st.meminfo = meminfo
        
        return st._getvalue()


@njit
def foo():
    b = Var(BOOPType)
    print(b)
    b7 = b.A.B
    print(b7)
    b8 = b.B
    print(b8)    
foo()

b = Var(BOOPType)
b1 = b.A
print(b1)


term_fields_dict = {
    "pred_node" : BasePredicateNode,
}

term_fields =  [(k,v) for k,v, in term_fields_dict.items()]





# var_fields = [
#     ('var', ???)
#     ('attr', unicode_type)
#     ('a_id', i8)
# ]

# term_fields = [
#     ("negated", u1),
#     ("is_beta", u1),
#     ("left_var", unicode_type),
#     ("right_var", unicode_type),
#     ("_str", unicode_type),
#     ("predicate_node", BasePredicateNode)
# ]


# condition_fields = [
#     #A Vector<*var>
#     ("vars", VectorType)
#     #A Vector<*Vector<*Term>>
#     ("conjuncts", VectorType)
    
# ]


# def new_condition(vars, conjucts)







# NOT(ab+c) = NOT(ab)+c = (a'+b')c' = a'c'+b'c'
# AND((ab+c), (de+f)) = abde+abf+cde+cf
# OR((ab+c), (de+f)) = ab+c+de+f


#There needs to be var and Var types probably
# because we can Bind to something without conditioning on it
# so







#### PLANNING PLANNING PLANNING ###

'''
Condition_node has alpha bits and beta bits. It has a DNF structure


'''
