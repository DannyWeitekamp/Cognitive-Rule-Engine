import operator
import numpy as np
import numba
from numba import types, njit, i8, u8, i4, u1,u2,u4,  i8, literally, generated_jit, objmode
from numba.typed import List
from numba.core.types import ListType, unicode_type, void, Tuple
from numba.experimental import structref
from numba.experimental.structref import new, define_attributes, _Utils
from numba.extending import SentryLiteralArgs, lower_cast, overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from cre.context import cre_context
from cre.structref import define_structref, define_boxing, define_structref_template, CastFriendlyStructref
from cre.fact import define_fact, BaseFact, cast_fact, DeferredFactRefType, Fact, _standardize_type
from cre.utils import PrintElapse, ptr_t, _struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_ptr,  lower_setattr, lower_getattr, _raw_ptr_from_struct, _decref_ptr, _incref_ptr, _incref_structref, _ptr_from_struct_incref
from cre.utils import assign_to_alias_in_parent_frame, encode_idrec, _obj_cast_codegen
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.cre_object import cre_obj_field_dict,CREObjType, CREObjTypeClass, CREObjProxy, set_chr_mbrs
# from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
# get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models


from operator import itemgetter
from copy import copy
from os import getenv
from cre.utils import deref_info_type, DEREF_TYPE_ATTR, DEREF_TYPE_LIST, listtype_sizeof_item, _obj_cast_codegen
from cre.core import T_ID_VAR, register_global_default
# import inspect



var_fields_dict = {
    **cre_obj_field_dict,
    # The pointer of the Var instance before any attribute selection
    #   e.g. if '''v = Var(Type); v_b = v.B;''' then v_b.base_ptr = &v
    'base_ptr' : i8,

    # A recounted reference to the base_ptr (only nonzero on derefs i.e. v.B)
    'base_ptr_ref' : ptr_t,

    # A pointer to the Var instance which is the NOT() of this Var
    'conj_ptr' : i8,

    # The name of the Var 
    'alias_' : unicode_type,
    'deref_attrs_str' : types.optional(unicode_type),

    # The byte offsets for each attribute relative to the previous resolved
    #  fact. E.g.  if attr "B" is at offset 10 in the type assigned to "B"
    #  then v.B.B.deref_infos = [10,10]
    'deref_infos': deref_info_type[::1],

    # The t_ids of the fact that the base Var is meant to match.
    'base_t_id' : u2,

    'head_t_id' : u2,
    # If true then instead of testing for the existence of the Var
    #  we test that the Var does not exist.
    'is_not' : u1,
    # 'base_type_name': unicode_type,
    # 'head_type_name': unicode_type,
    'base_type': types.Any,
    'head_type': types.Any,

    # # Put the hashes of the head  
    # 'literal_base_hash' : types.Any,
    # 'literal_head_hash' : types.Any,
}

var_fields =  [(k,v) for k,v, in var_fields_dict.items()]

class VarTypeClass(CREObjTypeClass):
    t_id = T_ID_VAR
    def preprocess_fields(self, fields):
        f_dict = {k:v for k,v in fields}
        if(f_dict["head_type"] != types.Any):
            self._head_type = f_dict.get("head_type",None)
            self._base_type = f_dict.get("base_type",None)

        return fields
    def __str__(self):
        base_type = getattr(self,"_base_type",None)
        head_type = getattr(self,"_head_type",None)
        if(base_type is None and head_type is None):
            return f"cre.GenericVarType"
        else:
            return f"cre.Var[base_type={base_type.instance_type}, head_type={head_type.instance_type}])"

# @lower_cast(VarTypeClass, CREObjType)
# def upcast(context, builder, fromty, toty, val):
#     return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)



# Manually register the type to avoid automatic getattr overloading 
default_manager.register(VarTypeClass, models.StructRefModel)

GenericVarType = VarTypeClass([(k,v) for k,v in var_fields_dict.items()])
register_global_default("Var", GenericVarType)

# Allow typed Var instances to be upcast to GenericVarType
@lower_cast(VarTypeClass, GenericVarType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)

class Var(CREObjProxy):
    t_id = T_ID_VAR
    def __new__(cls, typ, alias="", skip_assign_alias=False):
        # if(not isinstance(typ, types.StructRef)): typ = typ.fact_type
        typ = _standardize_type(typ, cre_context())
        base_type_name = str(typ)
                
        base_t_id = cre_context().get_t_id(_type=typ)

        if(getenv("CRE_SPECIALIZE_VAR_TYPE",default=False)):
            raise ValueError("THIS SHOULDN'T HAPPEN")
            typ_ref = types.TypeRef(typ)
            struct_type = get_var_type(typ_ref,typ_ref)
            st = var_ctor(struct_type, base_t_id, alias)
        else:
            st = var_ctor_generic(base_t_id, alias)
        
        st._base_type = typ
        st._head_type = typ
        st._derefs_str = ""

        # if(alias):
            # import inspect, ctypes
            # if(alias is not None): 
            #     # Binds this instance globally in the calling python context 
            #     #  so that it is bound to a variable named whatever alias was set to
            #     # print(inspect.stack()[2][0].f_locals)
            #     # Get the calling frame
            #     frame = inspect.stack()[2][0] 
            #     # Assign the Var to it's alias
            #     frame.f_locals[alias] = st
            #     # Update locals()
            #     ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))
            # # assign_to_alias_in_parent_frame(st, alias)

        return st
        
    def _handle_deref(self, attr_or_ind):
        '''Helper function that... '''
        
        # assert(isinstance(self.base_type,Fact))
        # with PrintElapse("Startup"):
        base_type = self.base_type
        base_type_name = str(base_type)
        # print("<<",attr_or_ind)
        _derefs_str = self.get_derefs_str()
        # print("_derefs_str", _derefs_str, type(_derefs_str))
        if(isinstance(attr_or_ind, str)):
            # ATTR case
            curr_head_type = self.head_type
            attr = attr_or_ind
            # fd = curr_head_type.field_dict
            # print("<<", attr_or_ind)
            head_type = curr_head_type.clean_spec[attr]['type']
            if(isinstance(head_type, DeferredFactRefType)):
                head_type = cre_context().name_to_type[head_type._fact_name]
            a_id = curr_head_type.get_attr_a_id(attr) #list(fd.keys()).index(attr)
            offset = curr_head_type.get_attr_offset(attr)#curr_head_type._attr_offsets[a_id]
            deref_info_type = DEREF_TYPE_ATTR
            # _derefs_str += f".{attr}"
        else:
            # LIST case
            assert isinstance(self.head_type, ListType), \
                f'__getitem__() not supported for Var with head_type {type(self.head_type)}'

            head_type = self.head_type.item_type

            attr = str(attr_or_ind)
            a_id = u4(attr_or_ind)
            print(self.head_type)
            item_size = listtype_sizeof_item(self.head_type)
            offset = int(attr_or_ind)*item_size
            # print(int(attr_or_ind), item_size)
            deref_info_type = DEREF_TYPE_LIST
            # _derefs_str += f"[{attr_or_ind}]"


        # head_t_id = getattr(head_type, "t_id", -1)
        # head_type_name = str(head_type)

        # with PrintElapse("new"):
        if(getenv("CRE_SPECIALIZE_VAR_TYPE",default=False)):

            if(deref_info_type == DEREF_TYPE_ATTR):
                struct_type = get_var_type(base_type, head_type)
            else:
                raise NotImplemented("Haven't implemented getitem() when CRE_SPECIALIZE_VAR_TYPE=true.")
            new_var = var_append_deref(self, attr_or_ind)#self, attr, a_id, offset, head_type_name, t_id, deref_info_type)
        else:
            head_t_id = cre_context().get_t_id(_type=head_type)
            # head_t_id = getattr(head_type, "t_id", -1)
            # print("type, a_id, offset, head_t_id", deref_info_type, a_id, offset, head_t_id)
            new_var = generic_var_append_deref(self, a_id, offset, head_t_id, typ=deref_info_type)
            # new = generic_var_append_deref(self, attr, a_id, offset, head_type_name, t_id, deref_info_type)
            # struct_type = GenericVarType
        #CHECK THAT PTRS ARE SAME HERE
            

            
        new_var._base_type = base_type
        new_var._head_type = head_type
        # new_var._derefs_str = _derefs_str

        # new = var_ctor(struct_type, str(type_name), var_get_alias(self))
        # var_memcopy(self, new)
        # var_append_deref(new, attr, offset)
        return new_var





    def __getattr__(self, attr):
        if(attr in ['_head_type', '_base_type','_derefs_str', '_deref_attrs']): return None
        if(attr == 'deref_attrs_str'):
            if('_deref_attrs_str' not in self.__dict__):
                self._deref_attrs_str = "".join(
                    [f"[{a}]" if a.isdigit() else "." + a for a in self.deref_attrs]
                )
            return self._deref_attrs_str
        elif(attr == 'deref_attrs'):
            if('_deref_attrs' not in self.__dict__):
                self._deref_attrs = resolve_deref_attrs(self)
            return self._deref_attrs
        elif(attr == 'base_type'):
            # Otherwise we need to resolve the type from the current context
            if('_base_type' not in self.__dict__):
                base_type_ref = self._numba_type_.field_dict['base_type']
                if(base_type_ref != types.Any):
                    # If the Var is type specialize then grab its base_type
                    self._base_type = base_type_ref.instance_type 
                else:
                    context = cre_context()
                    self._base_type = context.get_type(t_id=self.base_t_id)
            return self._base_type
        elif(attr == 'head_type'):
            if('_head_type' not in self.__dict__):
                head_type_ref = self._numba_type_.field_dict['head_type']
                if(head_type_ref != types.Any):
                    self._head_type = head_type_ref.instance_type 
                else:
                    context = cre_context()
                    self._head_type = context.get_type(t_id=self.head_t_id)
            return self._head_type
        elif(attr == 'is_not'):
            return var_get_is_not(self)
        
        #     return var_get_deref_attrs(self)
        elif(attr == 'deref_infos'):
            return var_get_deref_infos(self)
        elif(attr == 'alias'):
            return var_get_alias(self)
        elif(attr == 'base_t_id'):
            return var_get_base_t_id(self)
        elif(attr == 'head_t_id'):
            return var_get_head_t_id(self)
        elif(attr == 'base_ptr'):
            return var_get_base_ptr(self)
        elif(True): 
            return self._handle_deref(attr)

    def __getitem__(self,ind):
        return self._handle_deref(ind)

    def __str__(self):
        if(self.alias == ""):
            return self.__repr__()
        if(self.is_not):
            return f'NOT({self.alias}){self.deref_attrs_str}'
        else:
            return f'{self.alias}{self.deref_attrs_str}'

    def __repr__(self):
        prefix = "NOT" if(self.is_not) else "Var"
        base_name = self.base_type._fact_name if hasattr(self.base_type,'_fact_name') else str(self.base_type)
        if(self.alias != ""):
            base = f'{prefix}({base_name},{self.alias!r})'
        else: 
            base = f'{prefix}({base_name})'
        # print(self.deref_attrs)
        return f'{base}{self.deref_attrs_str}'

    def _cmp_helper(self,op_str,other,negate):
        check_legal_cmp(self, op_str, other)
        opt_str = types.literal(types.unliteral(op_str))
        # print(other)
        
        if(not isinstance(other,(VarTypeClass,Var))):
            # print("other",other)
            if(isinstance(other,(bool,))): other = int(other)
            return var_cmp_alpha(self,op_str,other, negate)
        else:
            return var_cmp_beta(self,op_str,other, negate)
    

    def __lt__(self, other): 
        from cre.default_ops import LessThan, FactIdrecsLessThan
        if(isinstance(other,Var) and isinstance(other.head_type,Fact)):
            return FactIdrecsLessThan(self,other)
        else:
            return LessThan(self, other)
    def __le__(self, other): 
        from cre.default_ops import LessThanEq
        return LessThanEq(self, other)
            
    def __gt__(self, other): 
        from cre.default_ops import GreaterThan, FactIdrecsLessThan
        if(isinstance(other,Var) and isinstance(other.head_type,Fact)):
            return FactIdrecsLessThan(other,self)
        else:
            return GreaterThan(self, other)

    def __ge__(self, other):
        from cre.default_ops import GreaterThanEq
        return GreaterThanEq(self, other)
    def __eq__(self, other): 
        from cre.default_ops import Equals, ObjEquals, ObjIsNone
        from cre.conditions import op_to_cond

        # with PrintElapse("new_ptr_op"):
        #     npo = ObjIsNone(self)

        # with PrintElapse("op_to_cond"):
        #     op_to_cond(npo)

        if(other is None):
            # with PrintElapse("new_ObjIsNone"):
            return op_to_cond(ObjIsNone(self))
        if(isinstance(other,Var) and isinstance(other.head_type,Fact)):
            # print("ObjEquals")
            # with PrintElapse("new_ObjEquals"):
            return op_to_cond(ObjEquals(self,other))
        # print("Equals")
        return Equals(self, other)
    def __ne__(self, other): 
        return ~(self == other)
        # from cre.default_ops import Equals, ObjEquals, ObjIsNone
        # if(other is None):
        #     return ~ObjIsNone(other)
        # if(isinstance(other,Var) and isinstance(other.head_type,Fact)):
        #     return ~ObjEquals(self,other)
        # return ~Equals(self, other)

    def __add__(self, other):
        from cre.default_ops import Add
        return Add(self, other)

    def __radd__(self, other):
        from cre.default_ops import Add
        return Add(other, self)

    def __sub__(self, other):
        from cre.default_ops import Subtract
        return Subtract(self, other)

    def __rsub__(self, other):
        from cre.default_ops import Subtract
        return Subtract(other, self)

    def __mul__(self, other):
        from cre.default_ops import Multiply
        return Multiply(self, other)

    def __rmul__(self, other):
        from cre.default_ops import Multiply
        return Multiply(other, self)

    def __truediv__(self, other):
        from cre.default_ops import Divide
        return Divide(self, other)

    def __rtruediv__(self, other):
        from cre.default_ops import Divide
        return Divide(other, self)

    def __floordiv__(self, other):
        from cre.default_ops import FloorDivide
        return FloorDivide(self, other)

    def __rfloordiv__(self, other):
        from cre.default_ops import FloorDivide
        return FloorDivide(other, self)

    def __pow__(self, other):
        from cre.default_ops import Power
        return Power(self, other)

    def __rpow__(self, other):
        from cre.default_ops import Power
        return Power(other, self)

    def __mod__(self, other):
        from cre.default_ops import Modulus
        return Modulus(other, self)

    def __and__(self, other):
        from cre.conditions import conditions_and, op_to_cond
        from cre.op import Op
        if(isinstance(other,Op)): other = op_to_cond(other)
        return conditions_and(self, other)

    def __or__(self, other):
        from cre.conditions import conditions_or, op_to_cond
        from cre.op import Op
        if(isinstance(other,Op)): other = op_to_cond(other)
        return conditions_or(self, other)

    def __invert__(self):
        from cre.conditions import _var_NOT
        return _var_NOT(self)
    
    # Explicitly defining these allows for pickling w/o invoking __getattr__()
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, d):
        self.__dict__ = d

    def __hash__(self):
        from cre.dynamic_exec import _cre_obj_hash
        # ptr = 0#get_var_ptr(self)
        return _cre_obj_hash(self)

    def get_base_ptr(self):
        return var_get_base_ptr(self)

    def get_ptr(self):
        return get_var_ptr(self)

    def get_ptr_incref(self):
        return get_var_ptr_incref(self)

    def get_derefs_str(self):
        if(self._derefs_str is None):
            self._derefs_str = "".join([f"[{x}]" if x.isdigit() else f".{x}" for x in self.deref_attrs])#str_var_derefs(self)
        return self._derefs_str

#### Get Attribute Overloading ####

@infer_getattr
class StructAttribute(AttributeTemplate):
    key = VarTypeClass
    def generic_resolve(self, typ, attr):
        if(attr == "alias"): return unicode_type
        from numba.cpython.hashing import _Py_hash_t
        if(attr == "__hash__"): return types.FunctionType(_Py_hash_t(CREObjType,))
        

        if attr in typ.field_dict:

            attrty = typ.field_dict[attr]
            # print("is attr", attrty)
            return attrty
        # print("AAAAA", typ.field_dict)
        head_type = typ.field_dict['head_type'].instance_type 
        #TODO Should check that alld subtype references are valid
        if(not hasattr(head_type,'field_dict')):
            raise AttributeError(f"Cannot dereference attribute '{attr}' of {typ}.")

        base_type = typ.field_dict['base_type']
        if(attr in head_type.field_dict):
            head_type = types.TypeRef(head_type.field_dict[attr])
            field_dict = {
                **var_fields_dict,
                **{"base_type" : base_type,
                 "head_type" : head_type}
            }
            attrty = VarTypeClass([(k,v) for k,v, in field_dict.items()])
            return attrty
        else:
            raise AttributeError(f"Var[{base_type}] has no attribute '{attr}'")

@lower_getattr_generic(VarTypeClass)
def var_getattr_impl(context, builder, typ, val, attr):
    
    #If the attr is 'alias' then retrieve the base var's alias
    if(attr == "alias"):
        st = cgutils.create_struct_proxy(typ)(context, builder, value=val)._getvalue()

        def get_alias(self):
            if(self.base_ptr != _raw_ptr_from_struct(self)):
                base = _struct_from_ptr(GenericVarType, self.base_ptr)
            else:
                base = _cast_structref(GenericVarType, self)
            return base.alias_

        ret = context.compile_internal(builder, get_alias, unicode_type(typ,), (st,))
        # context.nrt.incref(builder, unicode_type, ret)
        return ret

    #If the attribute is one of the var struct fields then retrieve it.
    elif(attr in var_fields_dict):
        # p
        # print("GETATTR", attr)
        utils = _Utils(context, builder, typ)
        dataval = utils.get_data_struct(val)
        ret = getattr(dataval, attr)
        fieldtype = typ.field_dict[attr]
        return imputils.impl_ret_borrowed(context, builder, fieldtype, ret)

    #Otherwise return a new instance with a new 'attr' and 'offset' append 
    else:
        # print("APPEND")
        base_type = typ.field_dict['base_type'].instance_type 
        head_type = typ.field_dict['head_type'].instance_type.clean_spec[attr]['type']
        # head_type = typ.field_dict['head_type'].instance_type 
        head_type_name = str(head_type)
        # print(">>", head_type_name)
        fd = base_type.field_dict
        a_id = base_type.get_attr_a_id(attr)#list(fd.keys()).index(attr)
        offset = base_type.get_attr_offset(attr)#base_type._attr_offsets[list(fd.keys()).index(attr)]
        ctor = cgutils.create_struct_proxy(typ)
        st = ctor(context, builder, value=val)._getvalue()
        t_id = getattr(head_type, "t_id", -1)

        new_var_type = get_var_type(base_type,head_type)

        def new_var_and_append(self):
            return var_append_deref(self, attr)
            # st = new(typ)
            # var_memcopy(self,st)
            # var_append_deref(st,attr,offset)
        # return st

        ret = context.compile_internal(builder, new_var_and_append, new_var_type(typ,), (st,))
        context.nrt.incref(builder, typ, ret)
        return ret



def resolve_deref_attrs(self):
    '''Gets the chain of attribute strings for a Var e.g. x.A.B.C -> ['A','B','C']
    '''
    # print("Q")
    context = cre_context()
    deref_infos = var_get_deref_infos(self)
    # print("Y")
    # print(deref_infos)
    deref_attrs = []
    typ = self.base_type
    # print("\nbase", typ)
    # print("deref_infos", deref_infos)
    for i, x in enumerate(deref_infos):
        # print(i, "X")
        # print(i, typ)
        if(isinstance(typ, ListType)):
            deref_attrs.append(str(x['a_id']))
        else:
            try:
                deref_attrs.append(typ.get_attr_from_a_id(x['a_id']))
            except IndexError as e:
                # print(self.base_t_id)
                chain_so_far = ''.join([f'[{a}]' if a.isdigit() else f'.{a}' for a in deref_attrs])
                raise ValueError(
                "Could resolve next attribute in chain after Var(" +\
                f"{typ},{self.alias!r}){chain_so_far}. No a_id {x['a_id']}." +\
                f"Chain: {deref_infos}"
                )

        # print(i, "U")
        typ = context.get_type(t_id=x['t_id'])
        # print(i, typ, x['t_id'])
        # print(i, "R")
    # print("Z")
    # print("<<", deref_attrs)
    return deref_attrs

### Methods that require python interpreter ### 
# Note: could get around this is there was some way to load cre_context.context_data
#  as some kind of global variable within the numba runtime

@generated_jit
@overload_method(VarTypeClass, "get_head_type_name")
def get_head_type_name(self):
    def impl(self):
        # print("GET HEAD")
        with objmode(head_type_name=unicode_type):
            context = cre_context()
            head_type_name = str(context.get_type(t_id=self.head_t_id))
        return head_type_name
    return impl

@generated_jit
@overload_method(VarTypeClass, "get_base_type_name")
def get_base_type_name(self):
    def impl(self):
        with objmode(base_type_name=unicode_type):
            context = cre_context()
            base_type = context.get_type(t_id=self.base_t_id)
            base_type_name = getattr(base_type,'_fact_name',str(base_type))
        return base_type_name
    return impl

@generated_jit
@overload_method(VarTypeClass, "get_deref_attrs")
def get_deref_attrs(self):
    str_list_type = ListType(unicode_type)
    def impl(self):
        # print("GET DEREF ATTRS")
        with objmode(deref_attrs=str_list_type):
            context = cre_context()
            deref_attrs = List.empty_list(unicode_type)
            typ = self.base_type
            for i,x in enumerate(self.deref_infos):
                if(isinstance(typ, ListType)):
                    deref_attrs.append(f"{x['a_id']}")
                else:
                    deref_attrs.append(typ.get_attr_from_a_id(x['a_id']))
                typ = context.get_type(t_id=x['t_id'])
        return deref_attrs
    return impl

@generated_jit
@overload_method(VarTypeClass, "get_deref_attr_str")
def get_deref_attrs_str(self):
    def impl(self):
        # If self.deref_attrs_str is None then get it from object mode.
        if(self.deref_attrs_str is None):
            with objmode(deref_attrs_str=unicode_type):
                deref_attrs_str = self.deref_attrs_str
            self.deref_attrs_str = deref_attrs_str
        return self.deref_attrs_str
    return impl

@generated_jit(cache=True)
@overload(str)
def var_str(self):
    if(not isinstance(self,VarTypeClass)): return
    def impl(self):
        return f'{self.alias}{get_deref_attrs_str(self)}' 
    return impl

@njit(cache=True)    
def get_var_ptr(self):
    return _raw_ptr_from_struct(self)


@njit(cache=True)    
def get_var_ptr_incref(self):
    return _ptr_from_struct_incref(self)




# def var_cmp_alpha(left_var, op_str, right_var,negated):
#     from cre.conditions import pt_to_cond, gen_pterm_ctor_alpha, gen_pterm_ctor_beta
#     # Treat None as 0 for comparing against a fact ref
#     print("***", isinstance(left_var.head_type, Fact),isinstance(left_var.head_type, types.StructRef), left_var.head_type)
#     if(right_var is None and isinstance(left_var.head_type, types.StructRef)): right_var = 0
#     right_var_type = types.unliteral(types.literal(right_var)) #if (isinstance(right_var, types.NoneType)) else types.int64
#     ctor = gen_pterm_ctor_alpha(left_var, op_str, right_var_type)
#     pt = ctor(left_var, op_str, right_var)
#     lbv = cast_structref(GenericVarType,left_var)
#     return pt_to_cond(pt, lbv, None, negated)
    

# def var_cmp_beta(left_var, op_str, right_var, negated):
#     from cre.conditions import pt_to_cond, gen_pterm_ctor_alpha, gen_pterm_ctor_beta
#     ctor = gen_pterm_ctor_beta(left_var, op_str, right_var)
#     pt = ctor(left_var, op_str, right_var)
#     lbv = cast_structref(GenericVarType,left_var)
#     rbv = cast_structref(GenericVarType,right_var)
#     return pt_to_cond(pt, lbv, rbv, negated)


def check_legal_cmp(var, op_str, other_var):
    # if(isinstance(var,Var)): var = var
    # if(isinstance(other_var,Var)): other_var = other_var
    if(op_str != "=="):
        head_type = var.head_type
        other_head_type = None
        if(isinstance(other_var, (VarTypeClass,Var))):
            other_head_type = other_var.head_type

        # if(hasattr(head_type, '_fact_name') or hasattr(other_head_type, '_fact_name')):
        #     raise AttributeError("Inequality not valid comparitor for Fact types.")


@njit(cache=True)
def var_get_is_not(self):
    return self.is_not

# @njit(cache=True)
# def var_get_deref_attrs(self):
    # return self.deref_attrs

@njit(cache=True)
def var_get_deref_infos(self):
    return self.deref_infos

@njit(cache=True)
def var_get_alias(self):
    return self.alias

@njit(cache=True)
def var_get_base_ptr(self):
    return self.base_ptr

@njit(cache=True)
def var_get_head_t_id(self):
    return self.head_t_id

@njit(cache=True)
def var_get_base_t_id(self):
    return self.base_t_id

# Manually define the boxing to avoid constructor overloading
define_boxing(VarTypeClass,Var)


var_type_cache = {}
def get_var_type(base_type, head_type=None):
    if(head_type is None): head_type = base_type
    t = (base_type, head_type)
    if(t not in var_type_cache):
        # print((str(t[0]),str(t[1])), t[0].t_id)
        d = {**var_fields_dict,**{
            'base_type': types.TypeRef(base_type),
            'head_type': types.TypeRef(head_type),
            }}

        struct_type = VarTypeClass([(k,v) for k,v, in d.items()])
        var_type_cache[t] = struct_type
        return struct_type
    else:
        # print("RETREIVED", (str(t[0]),str(t[1])),t[0].t_id)
        return var_type_cache[t]

@njit(cache=True)
def var_ctor(var_struct_type, base_t_id, alias=""):
    st = new(var_struct_type)
    lower_setattr(st,'idrec', encode_idrec(T_ID_VAR,0,0xFF))
    lower_setattr(st,'is_not', u1(0))
    lower_setattr(st,'conj_ptr', i8(0))
    lower_setattr(st,'base_t_id', base_t_id)
    lower_setattr(st,'head_t_id', base_t_id)
    lower_setattr(st,'base_ptr', i8(_raw_ptr_from_struct(st)))
    lower_setattr(st,'base_ptr_ref', ptr_t(0))
    lower_setattr(st,'alias_', alias)
    lower_setattr(st,'deref_attrs_str', None)
    lower_setattr(st,'deref_infos', np.empty(0,dtype=deref_info_type))
    # base_type_name = lower_getattr(self,"base_type_name")
    

    # st.idrec = encode_idrec(T_ID_VAR,0,0xFF)
    # st.is_not = u1(0)
    # st.conj_ptr = i8(0)
    # st.base_t_id = base_t_id
    # st.head_t_id = base_t_id
    # st.base_ptr = i8(_raw_ptr_from_struct(st))
    # st.base_ptr_ref = ptr_t(0)
    # st.alias =  "" if(alias is  None) else alias
    # st.deref_attrs_str = None
    # # st.deref_attrs = List.empty_list(unicode_type)
    # st.deref_infos = np.empty(0,dtype=deref_info_type)
    return st


@njit(GenericVarType(u2, unicode_type), cache=True)
def var_ctor_generic(base_t_id, alias):
    return var_ctor(GenericVarType, base_t_id, alias)


@overload(Var)
def overload_Var(typ,alias=""):
    _typ = typ.instance_type
    struct_type = get_var_type(_typ,_typ)
    base_t_id = cre_context().get_t_id(_type=_typ)
    # print("@@ IMPL VAR :: ", _typ, base_t_id)
    def impl(typ, alias=""):
        return var_ctor(struct_type, base_t_id, alias)

    return impl

# @njit(cache=True)
# def repr_var(self):
#     alias_part = ", '" + self.alias + "'" if len(self.alias) > 0 else ""
#     s = "Var(" + self.base_type_name + "Type" + alias_part + ")"
#     return s + str_var_derefs(self)


# @overload(repr)
# def overload_repr_var(self):
#     if(not isinstance(self, VarTypeClass)): return
#     return lambda self: repr_var(self)
    

# @njit(cache=True)
# def str_var_derefs(self):
#     s = ""
#     for i in range(len(self.deref_infos)):
#         attr = self.deref_attrs[i]
#         deref = self.deref_infos[i]
#         if(deref.type == DEREF_TYPE_ATTR):
#             s += f".{attr}"
#         else:
#             s += f"[{attr}]"
    # return s



# @njit(cache=True)
# def str_var(self):
#     s = self.alias
#     if (len(s) > 0):
#         return s + str_var_derefs(self)
#     else:
#         return repr(self)

# @njit(cache=True)
# def str_var_ptr(ptr):
#     return str_var(_struct_from_ptr(GenericVarType,ptr))

# @njit(cache=True)
# def str_var_ptr_derefs(ptr):
#     return str_var_derefs(_struct_from_ptr(GenericVarType,ptr))


# @overload(str)
# def overload_str_var(self):
#     if(not isinstance(self, VarTypeClass)): return
#     return lambda self: str_var(self)





#### getattr and dereferencing ####

@njit(types.void(GenericVarType, GenericVarType),cache=True)
def var_memcopy(self,st):
    # new_deref_attrs = List.empty_list(unicode_type)
    # new_deref_infos = np.empty(len(),dtype=deref_info_type)
    # for x in lower_getattr(self,"deref_attrs"):
    #     new_deref_attrs.append(x)
    # old_deref_infos
    # for i,y in enumerate(lower_getattr(self,"deref_infos")):
    #     new_deref_infos[i] = y
    lower_setattr(st,'idrec', lower_getattr(self,"idrec"))
    lower_setattr(st,'is_not', lower_getattr(self,"is_not"))
    lower_setattr(st,'base_ptr', lower_getattr(self,"base_ptr"))
    lower_setattr(st,'base_ptr_ref', lower_getattr(self,"base_ptr_ref"))
    # lower_setattr(st,'alias_', lower_getattr(self,"alias_"))
    # lower_setattr(st,'deref_attrs',new_deref_attrs)
    lower_setattr(st,'deref_infos', lower_getattr(self,"deref_infos").copy())
    # base_type_name = lower_getattr(self,"base_type_name")
    lower_setattr(st,'base_t_id',lower_getattr(self,"base_t_id"))
    lower_setattr(st,'head_t_id',lower_getattr(self,"head_t_id"))


# @generated_jit(cache=True)
# def var_append_deref(self,deref):
    


@njit(types.void(GenericVarType, u4, i4, u2, u1), cache=True)
def _var_append_deref(self, a_id, offset, head_t_id, typ):
    # lower_getattr(self,"deref_attrs").append(attr)
    old_deref_infos = lower_getattr(self,"deref_infos")
    L = len(old_deref_infos)
    new_deref_infos = np.empty(L+1,dtype=deref_info_type)
    new_deref_infos[:L] = old_deref_infos
    if(typ == DEREF_TYPE_ATTR):
        new_deref_infos[L].type = u1(DEREF_TYPE_ATTR)
    elif(typ == DEREF_TYPE_LIST):
        new_deref_infos[L].type = u1(DEREF_TYPE_LIST)

    new_deref_infos[L].a_id = u4(a_id)
    new_deref_infos[L].offset = i4(offset)
    new_deref_infos[L].t_id = u2(head_t_id)

    lower_setattr(self,'deref_infos', new_deref_infos)
    lower_setattr(self,'head_t_id', head_t_id)
    # lower_setattr(self,'base_ptr_ref', ptr_t(lower_getattr(self,"base_ptr")))


@generated_jit(cache=True)
def var_append_deref(self, attr):
    SentryLiteralArgs(['attr']).for_function(var_append_deref).bind(self,attr)
    if(self is GenericVarType):
        raise ValueError("var_append_deref() doesn't work on GenericVarType. Use generic_var_append_deref()")

    old_var_type = self
    old_head_type = self.field_dict['head_type'].instance_type
    attr = attr.literal_value
    a_id = old_head_type.get_attr_a_id(attr)
    offset = old_head_type.get_attr_offset(attr)
    # print(old_head_type.instance_type.field_dict)
    head_type = old_head_type.field_dict[attr]
    base_type = old_var_type.field_dict['base_type'].instance_type
    head_t_id = cre_context().get_t_id(_type=head_type)
    
    var_struct_type = get_var_type(base_type, head_type)

    # print("CONSTR VAR", base_type, head_type, self.name)
    # print("<<", var_struct_type)
    if(isinstance(attr,int)):
        typ = DEREF_TYPE_ATTR
    else:
        typ = DEREF_TYPE_LIST

    # print("AFT")
    def impl(self, attr):
        st = new(var_struct_type)
        var_memcopy(_cast_structref(GenericVarType, self),st)
        _var_append_deref(st, a_id, offset, head_t_id, typ=DEREF_TYPE_ATTR)
        lower_setattr(st, 'base_ptr_ref', _ptr_from_struct_incref(self))
        return st
    return impl


# def deref_var(struct_type, base_var, attr, a_id, offset, head_type_name, t_id=-1, typ=DEREF_TYPE_ATTR):
#     # _incref_structref(base_var)
#     st = new(struct_type)
#     var_memcopy(base_var,st)
#     _var_append_deref(st,attr, a_id, offset, t_id, head_type_name, typ)
#     lower_setattr(st, 'base_ptr_ref', _ptr_from_struct_incref(base_var))
#     return st

@njit(GenericVarType(GenericVarType, u4, i8, i8, i8), cache=True)
def generic_var_append_deref(self, a_id, offset, head_t_id, typ=DEREF_TYPE_ATTR):
    # _incref_structref(base_var)
    st = new(GenericVarType)
    var_memcopy(self, st)
    _var_append_deref(st, a_id, offset, head_t_id, typ=typ)
    lower_setattr(st, 'base_ptr_ref', _ptr_from_struct_incref(self))
    return st


    
    






@lower_setattr_generic(VarTypeClass)
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


@njit(types.void(GenericVarType, unicode_type), cache=True)
def var_assign_alias(var, alias):
    base = _struct_from_ptr(GenericVarType, var.base_ptr)
    base.alias_ = alias

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




