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
        # print("DEREF", attr)
        if(attr == 'fact_type'):
            return self._numba_type_.field_dict['fact_type'].instance_type
        elif(attr == 'head_type'):
            return self._numba_type_.field_dict['head_type'].instance_type

        elif(attr == 'deref_attrs'):
            return var_get_deref_attrs(self)
        elif(attr == 'alias'):
            return var_get_alias(self)
        elif(True): 
            typ = self._numba_type_
            
            fact_type = typ.field_dict['fact_type'].instance_type 
            head_type = fact_type.field_dict[attr]

            fd = fact_type.field_dict
            offset = fact_type._attr_offsets[list(fd.keys()).index(attr)]
            struct_type = get_var_definition(types.TypeRef(fact_type), types.TypeRef(head_type))
            new = var_ctor(struct_type, var_get_alias(self))
            copy_and_append(self, new,attr, offset)
            return new

    def __str__(self):
        return ".".join([f'Var[{self.fact_type._fact_name}]']+list(self.deref_attrs))

    def _cmp_helper(self,op_str,other,negate):
        check_legal_cmp(self, op_str, other)
        if(not isinstance(other,(VarTypeTemplate,Var))):
            return var_cmp_alpha(self,types.literal(op_str),other, negate)
        else:
            return var_cmp_beta(self,types.literal(op_str),other, negate)
    

    def __lt__(self,other): return self._cmp_helper("<",other,False)
    def __le__(self,other): return self._cmp_helper("<=",other,False)
    def __gt__(self,other): return self._cmp_helper(">",other,False)
    def __ge__(self,other): return self._cmp_helper(">=",other,False)
    def __eq__(self,other): return self._cmp_helper("==",other,False)
    def __ne__(self,other): return self._cmp_helper("==",other,True)
        

def var_cmp_alpha(left_var, op_str, right_var,negated):
    right_var_type = types.unliteral(types.literal(right_var))
    ctor = gen_pterm_ctor_alpha(left_var._numba_type_, op_str, right_var_type)
    pt = ctor(left_var, op_str, right_var)
    return pt_to_cond(pt, left_var.alias, None, negated)
    

def var_cmp_beta(left_var, op_str, right_var, negated):
    ctor = gen_pterm_ctor_beta(left_var._numba_type_, op_str, right_var._numba_type_)
    pt = ctor(left_var, op_str, right_var)
    return pt_to_cond(pt, left_var.alias, right_var.alias, negated)


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
def var_ctor(var_struct_type, alias):
    st = new(var_struct_type)
    st.alias =  "boop" if(alias is  None) else alias
    st.deref_attrs = List.empty_list(unicode_type)
    st.deref_offsets = List.empty_list(i8)
    return st


@overload(Var,strict=False)
def overload_Var(typ,alias=None):
    struct_type = get_var_definition(typ,typ)
    def impl(typ, alias=None):
        return var_ctor(struct_type,alias)

    return impl


@overload(repr)
def repr_var(self):
    if(not isinstance(self, VarTypeTemplate)): return
    fact_name = self.field_dict['fact_type'].instance_type._fact_name
    def impl(self):
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
PTermType = PTermTypeTemplate(pterm_fields)


@njit(cache=True)
def alpha_pterm_ctor(pn, left_var, op_str, right_var):
    st = new(PTermType)
    st.pred_node = pn
    st.str_val = str(left_var) + " " + op_str + " " + "?" #base_str + "?"#TODO str->float needs to work
    st.negated = False
    st.is_alpha = True
    return st

@njit(cache=True)
def beta_pterm_ctor(pn, left_var, op_str, right_var):
    st = new(PTermType)
    st.pred_node = pn
    st.str_val = str(left_var) + " " + op_str + " " + str(right_var)
    st.negated = False
    st.is_alpha = False
    return st


pnode_dat_cache = {}
def get_alpha_pnode_ctor(left_var, op_str, right_var):
    t = ("alpha",str(left_var), op_str, right_var)
    if(t not in pnode_dat_cache):
        left_fact_type = left_var.field_dict['fact_type'].instance_type
        left_type = left_var.field_dict['head_type'].instance_type
        left_fact_type_name = left_fact_type._fact_name

        ctor, _ = define_alpha_predicate_node(left_type, op_str, right_var)
        pnode_dat_cache[t] = (ctor, left_fact_type_name)
    return pnode_dat_cache[t]


def get_beta_pnode_ctor(left_var, op_str, right_var):
    t = ("beta",str(left_var), op_str,str(right_var))
    if(t not in pnode_dat_cache):
        left_fact_type = left_var.field_dict['fact_type'].instance_type
        left_type = left_var.field_dict['head_type'].instance_type
        left_fact_type_name = left_fact_type._fact_name

        right_fact_type = right_var.field_dict['fact_type'].instance_type
        right_type = right_var.field_dict['head_type'].instance_type
        right_fact_type_name = right_fact_type._fact_name

        ctor, _ = define_beta_predicate_node(left_type, op_str, right_type)
        pnode_dat_cache[t] = (ctor, left_fact_type_name, right_fact_type_name)
    return pnode_dat_cache[t]

@njit(cache=True)
def cpy_derefs(var):
    offsets = np.empty((len(var.deref_offsets),),dtype=np.int64)
    for i,x in enumerate(var.deref_offsets): offsets[i] = x
    return offsets

@njit(cache=True)
def cast_pn_to_base(pn):
    return _cast_structref(BasePredicateNodeType,pn) 

def gen_pterm_ctor_alpha(left_var, op_str, right_var):
    ctor, left_fact_type_name = \
            get_alpha_pnode_ctor(left_var, op_str, right_var)
    def impl(left_var, op_str, right_var):
        l_offsets = cpy_derefs(left_var)
        apn = ctor(str(left_fact_type_name), l_offsets, right_var)
        pn = cast_pn_to_base(apn) 
        return alpha_pterm_ctor(pn, left_var, op_str, right_var)
    return impl

def gen_pterm_ctor_beta(left_var, op_str, right_var):
    ctor, left_fact_type_name, right_fact_type_name = \
            get_beta_pnode_ctor(left_var, op_str, right_var)
    
    def impl(left_var, op_str, right_var):
        l_offsets = cpy_derefs(left_var)
        r_offsets = cpy_derefs(right_var)
        apn = ctor(str(left_fact_type_name), l_offsets, str(left_fact_type_name), r_offsets)
        pn = cast_pn_to_base(apn) 
        return beta_pterm_ctor(pn, left_var, op_str, right_var)            
    return impl

@overload(PTerm)
def pterm_ctor(left_var, op_str, right_var):
    if(not isinstance(op_str, types.Literal)): return 
    if(not isinstance(left_var, VarTypeTemplate)): return

    op_str = op_str.literal_value
    if(not isinstance(right_var, VarTypeTemplate)):
        return gen_pterm_ctor_alpha(left_var, op_str, right_var)
    else:
        return gen_pterm_ctor_beta(left_var, op_str, right_var)


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


@njit(cache=True)
def pt_to_cond(pt, left_alias, right_alias, negated):
    dnf = new_dnf(1)
    ind = 0 if (pt.is_alpha) else 1
    dnf[0][ind].append(pt)
    _vars = List.empty_list(unicode_type)
    _vars.append(left_alias)
    if(right_alias is not None):
        _vars.append(right_alias)
    pt.negated = negated
    c = Conditions(_vars, dnf)
    return c


def comparator_helper(op_str, left_var, right_var,negated=False):
    if(isinstance(left_var,VarTypeTemplate)):
        check_legal_cmp(left_var, op_str, right_var)
        op_str = types.unliteral(op_str)
        if(not isinstance(right_var,VarTypeTemplate)):
            # print("POOP")
            right_var_type = types.unliteral(right_var)
            # ctor = gen_pterm_ctor_alpha(left_var, op_str, right_var_type)
            # print("POOP")

            def impl(left_var, right_var):
                pt = PTerm(left_var, op_str, right_var)
                return pt_to_cond(pt, left_var.alias, None, negated)
        else:
            # ctor = gen_pterm_ctor_beta(left_var, op_str, right_var)
            def impl(left_var, right_var):
                pt = PTerm(left_var, op_str, right_var)
                return pt_to_cond(pt, left_var.alias, right_var.alias, negated)

            # return var_cmp_beta


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
    return comparator_helper("==", left_var, right_var, True)



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

ConditionsType = ConditionsTypeTemplate(conditions_fields)

@overload(Conditions,strict=False)
def conditions_ctor(_vars, dnf=None):
    # print("CONDITIONS CONSTRUCTOR", _vars, dnf)
    if(isinstance(_vars,VarTypeTemplate)):
        def impl(_vars,dnf=None):
            st = new(ConditionsType)
            st.vars = List.empty_list(unicode_type)
            st.vars.append(str(_vars)) #TODO should actually make a thing
            st.dnf = dnf if(dnf) else new_dnf(1)
            return st
    else:
        def impl(_vars,dnf=None):
            st = new(ConditionsType)
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
    if(not isinstance(l,ConditionsTypeTemplate)): return
    if(not isinstance(r,ConditionsTypeTemplate)): return
    return lambda l,r : conditions_and(l, r)

@generated_jit(cache=True)
@overload(operator.or_)
def var_or(l, r):
    if(not isinstance(l,ConditionsTypeTemplate)): return
    if(not isinstance(r,ConditionsTypeTemplate)): return
    return lambda l,r : conditions_or(l, r)

@generated_jit(cache=True)
@overload(operator.not_)
@overload(operator.invert)
def var_not(c):
    if(not isinstance(c,ConditionsTypeTemplate)): return
    return lambda c : conditions_not(c)
