import operator
import numpy as np
from numba import types, njit, i8, u8, i4, u1, i8, literally, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type, void, Tuple
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from numbert.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from numbert.experimental.context import kb_context
from numbert.experimental.structref import define_structref, define_structref_template
from numbert.experimental.kb import KnowledgeBaseType, KnowledgeBase, facts_for_t_id, fact_at_f_id
from numbert.experimental.fact import define_fact, BaseFactType, cast_fact
from numbert.experimental.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, decode_idrec, lower_getattr, _struct_from_pointer,  lower_setattr, lower_getattr
from numbert.experimental.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from numbert.experimental.vector import VectorType
from numbert.experimental.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
 get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models


from operator import itemgetter
from copy import copy


BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})

var_fields_dict = {
    'alias' : unicode_type,
    'deref_attrs': ListType(unicode_type),
    'deref_offsets': ListType(i8),
    'fact_type': types.Any,
    'head_type': types.Any,
}

var_fields =  [(k,v) for k,v, in var_fields_dict.items()]

class VarTypeTemplate(types.StructRef):
    pass

# Manually register the type to avoid automatic getattr overloading 
default_manager.register(VarTypeTemplate, models.StructRefModel)

class Var(structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)

    def __getattr__(self, attr):
        if(attr == 'fact_type'):
            return self._numba_type_.field_dict['fact_type'].instance_type
        elif(attr == 'deref_attrs'):
            # return self._numba_type_.field_dict['deref_attrs'].literal_value
            return var_get_deref_attrs(self)
        elif(True): #TODO
            return var_deref(self,attr)

    def __str__(self):
        return ".".join([f'Var[{self.fact_type._fact_name}]']+list(self.deref_attrs))

    def __lt__(self,other): return var_lt(self,other)
    def __le__(self,other): return var_le(self,other)
    def __gt__(self,other): return var_gt(self,other)
    def __ge__(self,other): return var_ge(self,other)
    def __eq__(self,other): return var_eq(self,other)
    def __ne__(self,other): return var_ne(self,other)
        

@njit(cache=True)
def var_get_deref_attrs(self):
    return self.deref_attrs
# def var_str_from_type(inst_type):
#     fn = inst_type.field_dict['fact_type'].instance_type._fact_name
#     # attr_str = inst_type.field_dict['deref_attrs'].literal_value
#     return f'Var[{fn}]{attr_str}'


# Manually define the boxing to avoid constructor overloading
define_boxing(VarTypeTemplate,Var)

@overload(Var,strict=False)
def var_ctor(typ,alias=None):
    d = {**var_fields_dict,**{'fact_type':typ, 'head_type':typ}}
    struct_type = VarTypeTemplate([(k,v) for k,v, in d.items()])

    def impl(typ, alias=None):
        # print(alias)
        # print("JEEEEFFEE",alias)q
        st = new(struct_type)
        st.alias =  "boop" if(alias is  None) else alias
        st.deref_attrs = List.empty_list(unicode_type)
        st.deref_offsets = List.empty_list(i8)
        # print("WHEEEEEEEE")
        # print(st.alias)
        # lower_setattr(st,'alias', 'jeffe')
        # lower_setattr(st,'deref_attrs',List.empty_list(unicode_type))
        # lower_setattr(st,'deref_offsets',List.empty_list(i8))
        return st

    return impl

# def resolve_deref_type(inst_type, attr):
#     old_str = inst_type.field_dict['deref_attrs'].literal_value
#     fact_type = inst_type.field_dict['fact_type']
#     new_str = old_str + f".{attr}"
#     new_struct_type = VarTypeTemplate([('fact_type', fact_type), ('deref_attrs', types.literal(new_str))])    
#     return new_struct_type

@overload(repr)
def repr_var(self):
    if(not isinstance(self, VarTypeTemplate)): return
    fact_name = self.field_dict['fact_type'].instance_type._fact_name
    def impl(self):
        # print("WHAAA",len(self.alias))
        alias_part = ", '" + self.alias + "'" if len(self.alias) > 0 else ""
        s = "Var(" + fact_name + "Type" + alias_part + ")"
        for attr in self.deref_attrs:
            s += "." + attr
        return s

    return impl

@overload(str)
def str_var(self):
    if(not isinstance(self, VarTypeTemplate)): return
    fact_name = self.field_dict['fact_type'].instance_type._fact_name
    def impl(self):
        # print("ALIAS", len(self.alias))
        s = self.alias
        if (len(s) > 0):
            for attr in self.deref_attrs:
                s += "." + attr
            return s
        else:
            return repr(self)
    return impl



#### Get Attribute Overloading ####

@infer_getattr
class StructAttribute(AttributeTemplate):
    key = VarTypeTemplate
    def generic_resolve(self, typ, attr):
        if attr in typ.field_dict:
            attrty = typ.field_dict[attr]
            # print("GETATTR_TY",attrty,attr)
            return attrty
        head_type = typ.field_dict['head_type'].instance_type 
        #TODO Should check that all subtype references are valid
        if(not hasattr(head_type,'field_dict')):
            raise AttributeError(f"Cannot dereference attribute '{attr}' of {typ}.")

        fact_type = typ.field_dict['fact_type']
        if(attr in head_type.field_dict):
            new_head_type = types.TypeRef(head_type.field_dict[attr])
            # print(head_type, new_head_type)
            field_dict = {
                **var_fields_dict,
                **{"fact_type" : fact_type,
                 "head_type" : new_head_type}
            }
            attrty = VarTypeTemplate([(k,v) for k,v, in field_dict.items()])
            # print("GETATTR_TY",attrty,attr)
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
    # print("SETATTR",attr,sig.args[-1])
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


@njit(cache=True)
def foo():
    b = Var(BOOPType)
    print("F",b)
    print(b.A)
    b7 = b.A
    print(b7.deref_attrs)
    print(b7)    
    print(b7.deref_attrs)
    print(b7)
    print(b.deref_offsets)
    print(b7.deref_offsets)
    # print(b7.B)
    # b8 = b.B
    # print(str(b8))    
    # print(b8)    
# foo()

# b = Var(BOOPType)
# b1 = b.A
# print(b1)

#### PTerm ####




pterm_fields_dict = {
    "str_val" : unicode_type,
    "pred_node" : BasePredicateNodeType,
    "negated" : u1,
    "is_alpha" : u1,
}

pterm_fields =  [(k,v) for k,v, in pterm_fields_dict.items()]

@structref.register
class PTermTypeTemplate(types.StructRef):
    pass

class PTerm(structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)
    def __str__(self):
        return pterm_get_str_val(self)

    @property
    def pred_node(self):
        return pterm_get_pred_node(self)



@njit(cach=True)
def pterm_get_str_val(self):
    return self.str_val

@njit(cach=True)
def pterm_get_pred_node(self):
    return self.pred_node

define_boxing(PTermTypeTemplate,PTerm)
# structref.define_proxy(PTerm, PTermTypeTemplate, list(pterm_fields_dict.keys()))
PTermType = PTermTypeTemplate(pterm_fields)


@overload(PTerm)
def pterm_ctor(left_var, op_str, right_var):
    if(not isinstance(op_str, types.Literal)): return 
    if(not isinstance(left_var, VarTypeTemplate)): return

    # print("PTERM CONSTRUCTOR", left_var, op_str, right_var)

    left_type = left_var.field_dict['head_type'].instance_type
    # left_attr_chain = left_var.field_dict['deref_attrs'].literal_value.split(".")[1:]
    op_str = op_str.literal_value
    if(not isinstance(right_var, VarTypeTemplate)):
        right_type = right_var.literal_type if (isinstance(right_var,types.Literal)) else right_var
        # ctor, _ = define_alpha_predicate_node(left_type, op_str, right_type)
        # print(ctor.__module__)

        def impl(left_var, op_str, right_var):
            st = new(PTermType)
            left_t_id = -1 #Not defined yet, needs Kb to resolve
            l_offsets = np.empty((len(left_var.deref_offsets),),dtype=np.int64)
            for i,x in enumerate(left_var.deref_offsets): l_offsets[i] = x
            # pred_node = ctor(left_t_id, l_offsets, right_var)
            pred_node = AlphaPredicateNode(left_type, l_offsets, op_str, right_var)
            st.pred_node = _cast_structref(BasePredicateNodeType, pred_node)
            st.str_val = str(left_var) + " " + op_str + " " + "?" #base_str + "?"#TODO str->float needs to work
            st.negated = False
            st.is_alpha = True
            return st

    else:
        right_type = right_var.field_dict['head_type'].instance_type
        # ctor, _ = define_beta_predicate_node(left_type, op_str, right_type)

        def impl(left_var, op_str, right_var):
            st = new(PTermType)
            left_t_id = -1 #Not defined yet, needs Kb to resolve
            right_t_id = -1 #Not defined yet, needs Kb to resolve
            l_offsets = np.empty((len(left_var.deref_offsets),),dtype=np.int64)
            r_offsets = np.empty((len(right_var.deref_offsets),),dtype=np.int64)
            for i,x in enumerate(left_var.deref_offsets): l_offsets[i] = x
            for i,x in enumerate(right_var.deref_offsets): r_offsets[i] = x

            # pred_node = ctor(left_t_id, l_offsets, right_t_id, r_offsets)
            pred_node = BetaPredicateNode(left_type, l_offsets, op_str, right_type, r_offsets)
            st.pred_node = _cast_structref(BasePredicateNodeType, pred_node)
            st.str_val = str(left_var) + " " + op_str + " " + str(right_var)
            st.negated = False
            st.is_alpha = False
            return st


    
    
    return impl



@overload(str)
def str_pterm(self):
    if(not isinstance(self, PTermTypeTemplate)): return
    def impl(self):
        return self.str_val
    return impl

@njit(cache=True)
def pterm_copy(self):
    st = new(PTermType)
    st.str_val = self.str_val
    st.pred_node = self.pred_node
    st.negated = self.negated
    st.is_alpha = self.is_alpha
    return st
    
@njit(cache=True)
def pterm_not(self):
    npt = pterm_copy(self)
    npt.negated = not npt.negated
    return npt

# @overload(operator.)



@njit(cache=True)
def bar():
    l = Var(BOOPType).B
    print(l)
    print(str(l))
    print(l.deref_attrs)
    print(l.deref_offsets)
    print("BREAK")
    # l.alias = "x"
    # print(str(l))
    r_l = 5
    r = Var(BOOPType, 'y').B
    # print(alias)
    
    print("R:", r)
    print("R:", str(r))
    print("R:", str(r.alias))
    pt = PTerm(l,"<",r_l)
    print(pt.str_val)
    # print(alias)
    # # print(pt)
    # pt2 = PTerm(l,"<",r)

    # print(pterm_not(pt2).str_val)
    # print(pt2.str_val)

    # print(pterm_not(pt2))
    # return pt2
bar()



# def lower_var_alpha_comparator(context, builder, sig, args, op_str):


# def lower_var_beta_comparator(context, builder, sig, args, op_str):

# @lower_builtin(operator.lt, VarTypeTemplate, types.Any)
# def var_a_lt(context, builder, sig, args):
#     return lower_var_alpha_comparator(context, builder, sig, args, "<")
    
# @lower_builtin(operator.lt, VarTypeTemplate, VarTypeTemplate)
# def var_b_lt(context, builder, sig, args):

@njit(cache=True)
def comparator_jitted(left_var, op_str, right_var, negated):
    pt = PTerm(left_var, op_str, right_var)
    dnf = new_dnf(1)
    ind = 0 if (pt.is_alpha) else 1
    dnf[0][ind].append(pt)
    _vars = List([left_var.alias])
    # if(not is_alpha): _vars.append(right_var.alias)
    pt.negated = negated
    # print(type(right_var))
    c = Conditions(_vars, dnf)
    return c


def comparator_helper(op_str, left_var, right_var,negate=False):
    if(isinstance(left_var,VarTypeTemplate)):
        if(isinstance(right_var,VarTypeTemplate)):
            def impl(left_var, right_var):
                c = comparator_jitted(left_var, op_str, right_var,negate)
                c.vars.append(right_var.alias)
                return c
        else:
            def impl(left_var, right_var):
                return comparator_jitted(left_var, op_str, right_var,negate)
        return impl


@generated_jit(cache=True)
@overload(operator.lt)
def var_lt(left_var, right_var):
    return comparator_helper("<", left_var, right_var)

@generated_jit(cache=True)
@overload(operator.le)
def var_le(left_var, right_var):
    return comparator_helper("<=", left_var, right_var)

@generated_jit(cache=True)
@overload(operator.gt)
def var_gt(left_var, right_var):
    return comparator_helper(">", left_var, right_var)

@generated_jit(cache=True)
@overload(operator.ge)
def var_ge(left_var, right_var):
    return comparator_helper(">=", left_var, right_var)

@generated_jit(cache=True)
@overload(operator.eq)
def var_eq(left_var, right_var):
    return comparator_helper("==", left_var, right_var)

@generated_jit(cache=True)
@overload(operator.ne)
def var_ne(left_var, right_var):
    return comparator_helper("==", left_var, right_var, negate=True)


@njit(cache=True)
def baz():
    l = Var(BOOPType).B
    print(l)
    r_l = 5
    r = Var(BOOPType, "y").B
    print(r)
    pt = l < r_l
    # print(pt.str_val)
    print(pt)
    # pt2 = PTerm(l,"<",r)
    pt2 = l < r

    print(pt2)

    pt3 = var_lt(l,r)
    print(pt3)
    return pt2
# baz()
# print("--------------------")
# baz.py_func()



pterm_list_type = ListType(PTermType)
pterm_list_x2_type = Tuple((pterm_list_type, pterm_list_type))
list_of_pterm_list_x2_type = ListType(Tuple((pterm_list_type, pterm_list_type)))

conditions_fields_dict = {
    'vars': ListType(unicode_type),
    'dnf': ListType(pterm_list_x2_type),
}

conditions_fields =  [(k,v) for k,v, in conditions_fields_dict.items()]

@structref.register
class ConditionsTypeTemplate(types.StructRef):
    pass

# Manually register the type to avoid automatic getattr overloading 
# default_manager.register(VarTypeTemplate, models.StructRefModel)

class Conditions(structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)
    def __str__(self):
        return conditions_str(self)
    def __and__(self, other):
        return conditions_and(self, other)
    def __or__(self, other):
        return conditions_or(self, other)
    def __not__(self):
        return conditions_not(self)
    def __invert__(self):
        return conditions_not(self)

define_boxing(ConditionsTypeTemplate,Conditions)

@njit(cache=True)
def new_dnf(n):
    dnf = List.empty_list(pterm_list_x2_type)
    for i in range(n):
        dnf.append( (List.empty_list(PTermType), List.empty_list(PTermType)) )
    return dnf


@overload(Conditions,strict=False)
def conditions_ctor(_vars, dnf=None):
    print("CONDITIONS CONSTRUCTOR", _vars, dnf)
    struct_type = ConditionsTypeTemplate(conditions_fields)
        
    if(isinstance(_vars,VarTypeTemplate)):
        def impl(_vars,dnf=None):
            st = new(struct_type)
            st.vars = List.empty_list(unicode_type)
            st.vars.append(str(_vars)) #TODO should actually make a thing
            st.dnf = dnf if(dnf) else new_dnf(1)
            return st
    else:
        def impl(_vars,dnf=None):
            st = new(struct_type)
            st.vars = List([str(x) for x in _vars])
            st.dnf = dnf if(dnf) else new_dnf(len(_vars))
            return st

    return impl

@overload(str)
def str_pterm(self):
    if(not isinstance(self, ConditionsTypeTemplate)): return
    def impl(self):
        s = ""
        for j, v in enumerate(self.vars):
            s += str(v)
            if(j < len(self.vars)-1): s += ", "
        s += '\n'
        for j, conjunct in enumerate(self.dnf):
            alphas, betas = conjunct
            for i, alpha_term in enumerate(alphas):
                s += "~" if alpha_term.negated else ""
                s += "(" + str(alpha_term) + ")" 
                if(i < len(alphas)-1 or len(betas)): s += " & "

            for i, beta_term in enumerate(betas):
                s += "!" if beta_term.negated else ""
                s += "(" + str(beta_term) + ")" 
                if(i < len(betas)-1): s += " & "

            if(j < len(self.dnf)-1): s += " |\n"
        return s
    return impl

@njit(cache=True)
def conditions_str(self):
    return str(self)


# NOT(ab+c) = NOT(ab)+c = (a'+b')c' = a'c'+b'c'
# AND((ab+c), (de+f)) = abde+abf+cde+cf
# OR((ab+c), (de+f)) = ab+c+de+f


@njit(cache=True)
def conditions_and(left,right):
    '''AND is distributive
    AND((ab+c), (de+f)) = abde+abf+cde+cf'''
    return Conditions(Var(BOOPType), dnf_and(left.dnf, right.dnf))

@njit(cache=True)
def dnf_and(l_dnf, r_dnf):
    dnf = new_dnf(len(l_dnf)*len(r_dnf))
    for i, l_conjuct in enumerate(l_dnf):
        for j, r_conjuct in enumerate(r_dnf):
            k = i*len(r_dnf) + j
            for x in l_conjuct[0]: dnf[k][0].append(x)
            for x in r_conjuct[0]: dnf[k][0].append(x)
            for x in l_conjuct[1]: dnf[k][1].append(x)
            for x in r_conjuct[1]: dnf[k][1].append(x)
    return dnf


@njit(cache=True)
def conditions_or(left,right):
    '''OR is additive like
    OR((ab+c), (de+f)) = ab+c+de+f'''
    return Conditions(Var(BOOPType), dnf_or(left.dnf, right.dnf))

@njit(cache=True)
def dnf_or(l_dnf, r_dnf):
    dnf = new_dnf(len(l_dnf)+len(r_dnf))
    for i, conjuct in enumerate(l_dnf):
        for x in conjuct[0]: dnf[i][0].append(x)
        for x in conjuct[1]: dnf[i][1].append(x)

    for i, conjuct in enumerate(r_dnf):
        k = len(l_dnf)+i
        for x in conjuct[0]: dnf[k][0].append(x)
        for x in conjuct[1]: dnf[k][1].append(x)

    return dnf

@njit(cache=True)
def dnf_not(c_dnf):
    dnfs = List.empty_list(list_of_pterm_list_x2_type)
    for i, conjunct in enumerate(c_dnf):
        dnf = new_dnf(len(conjunct[0])+len(conjunct[1]))
        for j, term in enumerate(conjunct[0]):
            dnf[j][0].append(pterm_not(term))
        for j, term in enumerate(conjunct[1]):
            k = len(conjunct[0]) + j
            dnf[k][1].append(pterm_not(term))
        dnfs.append(dnf)

    # print("PHAZZZZZZ")
    out_dnf = dnfs[0]
    for i in range(1,len(dnfs)):
        out_dnf = dnf_and(out_dnf,dnfs[i])
    return out_dnf


@njit(cache=True)
def conditions_not(c):
    '''NOT inverts the qualifiers and terms like
    NOT(ab+c) = NOT(ab)+c = (a'+b')c' = a'c'+b'c'''
    dnf = dnf_not(c.dnf)
    return Conditions(Var(BOOPType), dnf)




@generated_jit(cache=True)
@overload(operator.and_)
def var_and(l, r):
    return lambda l,r : conditions_and(l, r)

@generated_jit(cache=True)
@overload(operator.or_)
def var_and(l, r):
    return lambda l,r : conditions_or(l, r)

@generated_jit(cache=True)
@overload(operator.not_)
@overload(operator.invert)
def var_and(c):
    return lambda c : conditions_not(c)


@njit
def booz():
    # c = Conditions(Var(BOOPType))
    # c2 = Conditions(List([Var(BOOPType),Var(BOOPType)]))
    # print(c)
    # print(str(c))
    # print(c2)
    # print(str(c2))
    l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
    r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")
    print(l1.B < 1)
    print(l1.B > 7)
    print(l1.B < r1.B)
    print(l1.B < r2.B)

    c1 = l1 < 1
    c2 = l1 < l2

    c3 = (l1.B < 1) & (l1.B > 7) & (l1.B < r1.B) & (l1.B < r2.B) | \
         (l2.B < 1) & (l2.B > 7) & (l2.B < r1.B) & (l2.B < r2.B)
    print(c3)
    print(~c3)

    c4 = (l1 == 5) & (l1 == l2) & (l1 == 5) & (l1 != l2)
    print(c4)
    print(~c4)

    print(c3 & c4)
    print(c3 | c4)



    return

    dnf = new_dnf(2)
    l1, l2 = Var(BOOPType,"l1").B, Var(BOOPType,"l2").B
    r1, r2 = Var(BOOPType,"r1").B, Var(BOOPType,"r2").B
    dnf[0][0].append(l1 < 1)
    dnf[0][0].append(l1 > 7)
    dnf[0][1].append(l1 < r1)
    dnf[0][1].append(l1 < r2)

    dnf[1][0].append(l2 < 1)
    dnf[1][0].append(l2 > 7)
    dnf[1][1].append(l2 < r1)
    dnf[1][1].append(l2 < r2)

    c3 = Conditions(List([l1,l2,r1,r2]),dnf)
    print("C3")
    print(str(c3))
    print("NOT C3")
    print(str(conditions_not(c3)))
    
    dnf = new_dnf(2)
    dnf[0][0].append(l1 == 5)
    dnf[0][1].append(l1 == l2)
    dnf[1][0].append(l1 == 5)
    dnf[1][1].append(l1 != l2)
    c4 = Conditions(List([l1,l2,r1,r2]),dnf)
    print("C4")
    print(str(c4))
    print("NOT C4")
    print(str(conditions_not(c4)))


    dnf1 = new_dnf(1)
    dnf1[0][0].append(l1 < 1)
    dnf2 = new_dnf(1)
    dnf2[0][1].append(l1 < l2)

    print("___BOOP___")
    beep = conditions_and(Conditions(List([l1,l2]),dnf1),Conditions(List([l1,l2]),dnf2))    
    print("___BLAAAP___")
    print(str(beep))
    print("___BEEEEP___")

    c3a4 = conditions_and(c3,c4)
    print(str(c3a4))

    c3o4 = conditions_or(c3,c4)
    print(str(c3o4))

    print("----------")
    
    


    return c,c2

print("start")
booz()
print("end")


    # return c,c2


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









#There needs to be var and Var types probably
# because we can Bind to something without conditioning on it
# so







#### PLANNING PLANNING PLANNING ###

'''
Condition_node has alpha bits and beta bits. It has a DNF structure


'''
