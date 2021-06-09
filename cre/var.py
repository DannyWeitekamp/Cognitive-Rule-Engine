import operator
import numpy as np
import numba
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
from cre.fact import define_fact, BaseFactType, cast_fact, DeferredFactRefType, Fact
from cre.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_pointer,  lower_setattr, lower_getattr, _pointer_from_struct, _decref_pointer, _incref_pointer, _incref_structref
from cre.utils import assign_to_alias_in_parent_frame
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
 get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models


from operator import itemgetter
from copy import copy
from os import getenv
from cre.utils import deref_type, OFFSET_TYPE_ATTR, OFFSET_TYPE_LIST
# import inspect



var_fields_dict = {
    # If true then instead of testing for the existence of the Var
    #  we test that the Var does not exist.
    'is_not' : u1,

    # A pointer to the Var instance which is the NOT() of this Var
    'conj_ptr' : i8,

    # The pointer of the Var instance before any attribute selection
    #   e.g. if '''v = Var(Type); v_b = v.B;''' then v_b.base_ptr = &v
    'base_ptr' : i8,

    # The name of the Var 
    'alias' : unicode_type,

    # A list of attributes that have been dereferenced so far 
    #  e.g. v.B.B.deref_attrs = ['B','B']
    'deref_attrs': ListType(unicode_type),

    # The byte offsets for each attribute relative to the previous resolved
    #  fact. E.g.  if attr "B" is at offset 10 in the type assigned to "B"
    #  then v.B.B.deref_offsets = [10,10]
    'deref_offsets': deref_type[::1],

    # The name of the fact that the base Var is meant to match.
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
    def __new__(cls, typ, alias=None, skip_assign_alias=False):
        if(not isinstance(typ, types.StructRef)): typ = typ.fact_type
        fact_type_name = typ._fact_name
        print(repr(fact_type_name))
        typ_ref = types.TypeRef(typ)

        if(getenv("CRE_SPECIALIZE_VAR_TYPE",default=False)):
            struct_type = get_var_definition(typ_ref,typ_ref)
        else:
            struct_type = GenericVarType

        st = var_ctor(struct_type, fact_type_name, alias)
        st._fact_type = typ
        st._head_type = typ

        if(not skip_assign_alias):
            assign_to_alias_in_parent_frame(st,alias)


        # print("after")
        return st
        # return structref.StructRefProxy.__new__(cls, *args)

    def _handle_deref(self, attr_or_ind):
        '''Helper function that... '''
        fact_type = self.fact_type
        fact_type_name = fact_type._fact_name

        if(isinstance(attr_or_ind, str)):
            attr = attr_or_ind
            fd = fact_type.field_dict
            head_type = fact_type.spec[attr]['type']
            if(isinstance(head_type,DeferredFactRefType)):
                head_type = kb_context().fact_types[head_type._fact_name]
            # head_type = fact_type.field_dict[attr]
            offset = fact_type._attr_offsets[list(fd.keys()).index(attr)]
            deref_type = 'attr'
        else:
            assert isinstance(self.head_type, ListType), \
                f'__getitem__() not supported for Var with head_type {type(self.head_type)}'

            print(self.head_type.__dict__)
            head_type = self.head_type.item_type

            attr = str(attr_or_ind)
            offset = int(attr_or_ind)
            deref_type = 'list'

        if(getenv("CRE_SPECIALIZE_VAR_TYPE",default=False)):
            if(deref_type == 'attr'):
                struct_type = get_var_definition(types.TypeRef(fact_type), types.TypeRef(head_type))
            else:
                raise NotImplemented("Haven't implemented getitem() when CRE_SPECIALIZE_VAR_TYPE=true.")
        else:
            struct_type = GenericVarType
        #CHECK THAT PTRS ARE SAME HERE
        new = new_appended_var(struct_type, self, attr, offset, deref_type)
        new._fact_type = fact_type
        new._head_type = head_type
        # new = var_ctor(struct_type, str(fact_type_name), var_get_alias(self))
        # var_memcopy(self, new)
        # var_append_deref(new, attr, offset)
        return new



    def __getattr__(self, attr):
        # print("DEREF", attr)
        if(attr in ['_head_type', '_fact_type']): return None
        if(attr == 'fact_type'):
            fact_type_ref = self._numba_type_.field_dict['fact_type']
            if(fact_type_ref != types.Any):
                # If the Var is type specialize then grab its fact_type
                return fact_type_ref.instance_type 
            else:
                # Otherwise we need to resolve the type from the current context
                return self._fact_type
                # return kb_context().fact_types[self.fact_type_name]
        elif(attr == 'head_type'):
            head_type_ref = self._numba_type_.field_dict['head_type']
            if(head_type_ref != types.Any):
                return fact_type_ref.instance_type 
            else:
                return self._head_type
                # return head_type.instance_type if(head_type != types.Any) else None
        elif(attr == 'is_not'):
            return var_get_is_not(self)
        elif(attr == 'deref_attrs'):
            return var_get_deref_attrs(self)
        elif(attr == 'deref_offsets'):
            return var_get_deref_offsets(self)
        elif(attr == 'alias'):
            return var_get_alias(self)
        elif(attr == 'fact_type_name'):
            return var_get_fact_type_name(self)
        elif(attr == 'base_ptr'):
            return var_get_base_ptr(self)
        elif(True): 
            return self._handle_deref(attr)

    def __getitem__(self,ind):
        return self._handle_deref(ind)



    def __str__(self):
        prefix = "NOT" if(self.is_not) else "Var"
        base = f'{prefix}({self.fact_type_name},{self.alias!r})'
        deref_strs = [f"[{a}]" if t==OFFSET_TYPE_LIST else "." + a 
                for (t,_), a in zip(self.deref_offsets, self.deref_attrs)]
        s = base + "".join(deref_strs)
        # print(self.is_not)
         # s = f'NOT({s})'
        return s

    def _cmp_helper(self,op_str,other,negate):
        check_legal_cmp(self, op_str, other)
        opt_str = types.literal(types.unliteral(op_str))
        # print(other)
        
        if(not isinstance(other,(VarTypeTemplate,Var))):
            # print("other",other)
            if(isinstance(other,(bool,))): other = int(other)
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

    def __invert__(self):
        from cre.condition_node import _var_NOT
        return _var_NOT(self)
    





def var_cmp_alpha(left_var, op_str, right_var,negated):
    from cre.condition_node import pt_to_cond, gen_pterm_ctor_alpha, gen_pterm_ctor_beta
    # Treat None as 0 for comparing against a fact ref
    print("***", isinstance(left_var.head_type, Fact),isinstance(left_var.head_type, types.StructRef), left_var.head_type)
    if(right_var is None and isinstance(left_var.head_type, types.StructRef)): right_var = 0
    right_var_type = types.unliteral(types.literal(right_var)) #if (isinstance(right_var, types.NoneType)) else types.int64
    ctor = gen_pterm_ctor_alpha(left_var, op_str, right_var_type)
    pt = ctor(left_var, op_str, right_var)
    lbv = cast_structref(GenericVarType,left_var)
    return pt_to_cond(pt, lbv, None, negated)
    

def var_cmp_beta(left_var, op_str, right_var, negated):
    from cre.condition_node import pt_to_cond, gen_pterm_ctor_alpha, gen_pterm_ctor_beta
    ctor = gen_pterm_ctor_beta(left_var, op_str, right_var)
    pt = ctor(left_var, op_str, right_var)
    lbv = cast_structref(GenericVarType,left_var)
    rbv = cast_structref(GenericVarType,right_var)
    return pt_to_cond(pt, lbv, rbv, negated)


def check_legal_cmp(var, op_str, other_var):
    # if(isinstance(var,Var)): var = var
    # if(isinstance(other_var,Var)): other_var = other_var
    if(op_str != "=="):
        head_type = var.head_type
        other_head_type = None
        if(isinstance(other_var, (VarTypeTemplate,Var))):
            other_head_type = other_var.head_type

        # if(hasattr(head_type, '_fact_name') or hasattr(other_head_type, '_fact_name')):
        #     raise AttributeError("Inequality not valid comparitor for Fact types.")


@njit(cache=True)
def var_get_is_not(self):
    return self.is_not

@njit(cache=True)
def var_get_deref_attrs(self):
    return self.deref_attrs

@njit(cache=True)
def var_get_deref_offsets(self):
    return self.deref_offsets

@njit(cache=True)
def var_get_alias(self):
    return self.alias

@njit(cache=True)
def var_get_base_ptr(self):
    return self.base_ptr

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
def var_ctor(var_struct_type, fact_type_name="", alias=""):
    st = new(var_struct_type)
    st.is_not = u1(0)
    st.conj_ptr = 0
    st.fact_type_name = fact_type_name
    st.base_ptr = _pointer_from_struct(st)
    st.alias =  "" if(alias is  None) else alias
    st.deref_attrs = List.empty_list(unicode_type)
    st.deref_offsets = np.empty(0,dtype=deref_type)
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
def new_appended_var(struct_type, base_var, attr, offset,typ='attr'):
    _incref_structref(base_var)
    st = new(struct_type)
    var_memcopy(base_var,st)
    var_append_deref(st,attr,offset,typ)
    return st


@njit(cache=True)
def var_memcopy(self,st):
    new_deref_attrs = List.empty_list(unicode_type)
    # new_deref_offsets = np.empty(len(),dtype=deref_type)
    for x in lower_getattr(self,"deref_attrs"):
        new_deref_attrs.append(x)
    # old_deref_offsets
    # for i,y in enumerate(lower_getattr(self,"deref_offsets")):
    #     new_deref_offsets[i] = y

    lower_setattr(st,'is_not', lower_getattr(self,"is_not"))
    lower_setattr(st,'base_ptr',lower_getattr(self,"base_ptr"))
    lower_setattr(st,'alias',lower_getattr(self,"alias"))
    lower_setattr(st,'deref_attrs',new_deref_attrs)
    lower_setattr(st,'deref_offsets',lower_getattr(self,"deref_offsets").copy())
    fact_type_name = lower_getattr(self,"fact_type_name")
    lower_setattr(st,'fact_type_name',str(fact_type_name))
    
    

@njit(cache=True)
def var_append_deref(self,attr,offset,typ='attr'):
    lower_getattr(self,"deref_attrs").append(attr)
    old_deref_offsets = lower_getattr(self,"deref_offsets")
    L = len(old_deref_offsets)
    new_deref_offsets = np.empty(L+1,dtype=deref_type)
    new_deref_offsets[:L] = old_deref_offsets
    if(typ == 'attr'):
        new_deref_offsets[L].type = u1(OFFSET_TYPE_ATTR)
    elif(typ == 'list'):
        new_deref_offsets[L].type = u1(OFFSET_TYPE_LIST)

    new_deref_offsets[L].offset = i8(offset)

    lower_setattr(self,'deref_offsets', new_deref_offsets)


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
            return new_appended_var(typ, self, attr, offset,'attr')
            # st = new(typ)
            # var_memcopy(self,st)
            # var_append_deref(st,attr,offset)
            # return st

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




