import operator
import numpy as np
from numba import types, njit, i8, u8, i4, u1, i8, literally, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type, void, Tuple
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from cre.context import kb_context
from cre.structref import define_structref, define_structref_template
from cre.kb import KnowledgeBaseType, KnowledgeBase, facts_for_t_id, fact_at_f_id
from cre.fact import define_fact, BaseFactType, cast_fact, DefferedFactRefType
from cre.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_pointer,  lower_setattr, lower_getattr, _pointer_from_struct
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
 get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models


from operator import itemgetter
from copy import copy



var_fields_dict = {
    'base_ptr' : i8,
    'alias' : unicode_type,
    'deref_attrs': ListType(unicode_type),
    'deref_offsets': ListType(i8),
    'fact_type_name': unicode_type,
    'fact_type': types.Any,
    'head_type': types.Any,
}

var_fields =  [(k,v) for k,v, in var_fields_dict.items()]

class VarTypeTemplate(types.StructRef):
    pass


# Manually register the type to avoid automatic getattr overloading 
default_manager.register(VarTypeTemplate, models.StructRefModel)

GenericVarType = VarTypeTemplate([(k,v) for k,v in var_fields_dict.items()])

class Var(structref.StructRefProxy):
    def __new__(cls, typ, alias=None):
        if(not isinstance(typ, types.StructRef)): typ = typ.fact_type
        fact_type_name = typ._fact_name
        typ = types.TypeRef(typ)
        struct_type = get_var_definition(typ,typ)
        st = var_ctor(struct_type, fact_type_name, alias)
        return st
        # return structref.StructRefProxy.__new__(cls, *args)

    def __getattr__(self, attr):
        # print("DEREF", attr)
        if(attr == 'fact_type'):
            fact_type = self._numba_type_.field_dict['fact_type']
            return fact_type.instance_type if(fact_type != types.Any) else None
        elif(attr == 'head_type'):
            head_type = self._numba_type_.field_dict['head_type']
            return head_type.instance_type if(head_type != types.Any) else None
        elif(attr == 'deref_attrs'):
            return var_get_deref_attrs(self)
        elif(attr == 'alias'):
            return var_get_alias(self)
        elif(attr == 'fact_type_name'):
            return var_get_fact_type_name(self)
        elif(True): 
            typ = self._numba_type_
            
            fact_type = typ.field_dict['fact_type'].instance_type 
            fact_type_name = fact_type._fact_name
            head_type = fact_type.spec[attr]['type']
            if(isinstance(head_type,DefferedFactRefType)):
                head_type = kb_context().fact_types[head_type._fact_name]
            # head_type = fact_type.field_dict[attr]

            fd = fact_type.field_dict
            offset = fact_type._attr_offsets[list(fd.keys()).index(attr)]
            struct_type = get_var_definition(types.TypeRef(fact_type), types.TypeRef(head_type))
            new = var_ctor(struct_type, fact_type_name, var_get_alias(self))
            copy_and_append(self, new,attr, offset)
            return new

    def __str__(self):
        return ".".join([f'Var[{self.fact_type_name},{self.alias!r}]']+list(self.deref_attrs))

    def _cmp_helper(self,op_str,other,negate):
        check_legal_cmp(self, op_str, other)
        opt_str = types.literal(types.unliteral(op_str))
        if(not isinstance(other,(VarTypeTemplate,Var))):
            return var_cmp_alpha(self,op_str,other, negate)
        else:
            return var_cmp_beta(self,op_str,other, negate)
    

    def __lt__(self,other): return self._cmp_helper("<",other,False)
    def __le__(self,other): return self._cmp_helper("<=",other,False)
    def __gt__(self,other): return self._cmp_helper(">",other,False)
    def __ge__(self,other): return self._cmp_helper(">=",other,False)
    def __eq__(self,other): return self._cmp_helper("==",other,False)
    def __ne__(self,other): return self._cmp_helper("==",other,True)

    def __and__(self, other):
        from cre.condition_node import conditions_and
        return conditions_and(self, other)

    def __or__(self, other):
        from cre.condition_node import conditions_or
        return conditions_or(self, other)
    





def var_cmp_alpha(left_var, op_str, right_var,negated):
    from cre.condition_node import pt_to_cond, gen_pterm_ctor_alpha, gen_pterm_ctor_beta
    # Treat None as 0 for comparing against a fact ref
    if(right_var is None and isinstance(left_var.head_type, types.StructRef)): right_var = 0
    right_var_type = types.unliteral(types.literal(right_var)) #if (isinstance(right_var, types.NoneType)) else types.int64
    ctor = gen_pterm_ctor_alpha(left_var._numba_type_, op_str, right_var_type)
    pt = ctor(left_var, op_str, right_var)
    lbv = cast_structref(GenericVarType,left_var)
    return pt_to_cond(pt, lbv, None, negated)
    

def var_cmp_beta(left_var, op_str, right_var, negated):
    from cre.condition_node import pt_to_cond, gen_pterm_ctor_alpha, gen_pterm_ctor_beta
    ctor = gen_pterm_ctor_beta(left_var._numba_type_, op_str, right_var._numba_type_)
    pt = ctor(left_var, op_str, right_var)
    lbv = cast_structref(GenericVarType,left_var)
    rbv = cast_structref(GenericVarType,right_var)
    return pt_to_cond(pt, lbv, rbv, negated)


def check_legal_cmp(var, op_str, other_var):
    if(isinstance(var,Var)): var = var._numba_type_
    if(isinstance(other_var,Var)): other_var = other_var._numba_type_
    if(op_str != "=="):
        head_type = var.field_dict['head_type'].instance_type
        other_head_type = None
        if(isinstance(other_var, (VarTypeTemplate,Var))):
            other_head_type = other_var.field_dict['head_type'].instance_type
        if(hasattr(head_type, '_fact_name') or hasattr(other_head_type, '_fact_name')):
            raise AttributeError("Inequality not valid comparitor for Fact types.")



@njit(cache=True)
def var_get_deref_attrs(self):
    return self.deref_attrs

@njit(cache=True)
def var_get_alias(self):
    return self.alias

@njit(cache=True)
def var_get_fact_type_name(self):
    return self.fact_type_name

# Manually define the boxing to avoid constructor overloading
define_boxing(VarTypeTemplate,Var)


var_type_cache = {}
def get_var_definition(fact_type, head_type):
    t = (fact_type, head_type)
    if(t not in var_type_cache):
        d = {**var_fields_dict,**{'fact_type':fact_type, 'head_type':head_type}}
        struct_type = VarTypeTemplate([(k,v) for k,v, in d.items()])
        var_type_cache[t] = struct_type
        return struct_type
    else:
        return var_type_cache[t]

@njit(cache=True)
def var_ctor(var_struct_type, fact_type_name, alias):
    st = new(var_struct_type)
    st.fact_type_name = fact_type_name
    st.base_ptr = _pointer_from_struct(st)
    st.alias =  "" if(alias is  None) else alias
    st.deref_attrs = List.empty_list(unicode_type)
    st.deref_offsets = List.empty_list(i8)
    return st


@overload(Var,strict=False)
def overload_Var(typ,alias=None):
    fact_type = typ.instance_type
    fact_type_name = fact_type._fact_name
    struct_type = get_var_definition(typ,typ)
    def impl(typ, alias=None):
        return var_ctor(struct_type,fact_type_name,alias)

    return impl

@njit(cache=True)
def repr_var(self):
    alias_part = ", '" + self.alias + "'" if len(self.alias) > 0 else ""
    s = "Var(" + self.fact_type_name + "Type" + alias_part + ")"
    for attr in self.deref_attrs:
        s += "." + attr
    return s


@overload(repr)
def overload_repr_var(self):
    if(not isinstance(self, VarTypeTemplate)): return
    return lambda self: repr_var(self)
    

@njit(cache=True)
def str_var(self):
    s = self.alias
    if (len(s) > 0):
        for attr in self.deref_attrs:
            s += "." + attr
        return s
    else:
        return repr(self)


@overload(str)
def overload_str_var(self):
    if(not isinstance(self, VarTypeTemplate)): return
    return lambda self: str_var(self)



#### Get Attribute Overloading ####

@infer_getattr
class StructAttribute(AttributeTemplate):
    key = VarTypeTemplate
    def generic_resolve(self, typ, attr):
        if attr in typ.field_dict:
            attrty = typ.field_dict[attr]
            return attrty
        head_type = typ.field_dict['head_type'].instance_type 
        #TODO Should check that all subtype references are valid
        if(not hasattr(head_type,'field_dict')):
            raise AttributeError(f"Cannot dereference attribute '{attr}' of {typ}.")

        fact_type = typ.field_dict['fact_type']
        if(attr in head_type.field_dict):
            new_head_type = types.TypeRef(head_type.field_dict[attr])
            field_dict = {
                **var_fields_dict,
                **{"fact_type" : fact_type,
                 "head_type" : new_head_type}
            }
            attrty = VarTypeTemplate([(k,v) for k,v, in field_dict.items()])
            return attrty
        else:
            raise AttributeError(f"Var[{fact_type}] has no attribute '{attr}'")

#### getattr and dereferencing ####

@njit(cache=True)
def copy_and_append(self,st,attr,offset):
    new_deref_attrs = List.empty_list(unicode_type)
    new_deref_offsets = List.empty_list(i8)
    for x in lower_getattr(self,"deref_attrs"):
        new_deref_attrs.append(x)
    for y in lower_getattr(self,"deref_offsets"):
        new_deref_offsets.append(y)
    new_deref_attrs.append(attr)
    new_deref_offsets.append(offset)
    lower_setattr(st,'base_ptr',lower_getattr(self,"base_ptr"))
    lower_setattr(st,'deref_attrs',new_deref_attrs)
    lower_setattr(st,'deref_offsets',new_deref_offsets)
    lower_setattr(st,'alias',self.alias)


@lower_getattr_generic(VarTypeTemplate)
def var_getattr_impl(context, builder, typ, val, attr):
    #If the attribute is one of the var struct fields then retrieve it.
    if(attr in var_fields_dict):
        # print("GETATTR", attr)
        utils = _Utils(context, builder, typ)
        dataval = utils.get_data_struct(val)
        ret = getattr(dataval, attr)
        fieldtype = typ.field_dict[attr]
        return imputils.impl_ret_borrowed(context, builder, fieldtype, ret)

    #Otherwise return a new instance with a new 'attr' and 'offset' append 
    else:
        fact_type = typ.field_dict['fact_type'].instance_type 
        fd = fact_type.field_dict
        offset = fact_type._attr_offsets[list(fd.keys()).index(attr)]
        ctor = cgutils.create_struct_proxy(typ)
        st = ctor(context, builder, value=val)._getvalue()

        def new_var_and_append(self):
            st = new(typ)
            copy_and_append(self,st,attr,offset)
            return st

        ret = context.compile_internal(builder, new_var_and_append, typ(typ,), (st,))
        context.nrt.incref(builder, typ, ret)
        return ret

@lower_setattr_generic(VarTypeTemplate)
def struct_setattr_impl(context, builder, sig, args, attr):
    [inst_type, val_type] = sig.args
    [instance, val] = args
    utils = _Utils(context, builder, inst_type)
    dataval = utils.get_data_struct(instance)
    # cast val to the correct type
    field_type = inst_type.field_dict[attr]
    casted = context.cast(builder, val, val_type, field_type)
    # read old
    old_value = getattr(dataval, attr)
    # incref new value
    context.nrt.incref(builder, val_type, casted)
    # decref old value (must be last in case new value is old value)
    context.nrt.decref(builder, val_type, old_value)
    # write new
    setattr(dataval, attr, casted)

#### dereferencing for py_funcs ####

@intrinsic
def _var_deref(typingctx, typ, attr_type):
    attr = attr_type.literal_value
    def codegen(context, builder, sig, args):
        impl = context.get_getattr(sig.args[0], attr)
        return impl(context, builder, sig.args[0], args[0], attr)

    sig = typ(typ,attr_type)
    return sig, codegen

@njit(cache=True)
def var_deref(self,attr):
    return _var_deref(self,literally(attr))




