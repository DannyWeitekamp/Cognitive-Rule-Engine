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
from cre.context import cre_context, get_cre_context_data
from cre.structref import define_structref, define_boxing, define_structref_template, CastFriendlyStructref
from cre.fact import (define_fact, BaseFact, cast_fact,
            DeferredFactRefType, Fact, _standardize_type, resolve_deref_data_ptr)
from cre.utils import (cast, PrintElapse, ptr_t,
    decode_idrec, lower_getattr,  lower_setattr, lower_getattr,
    _decref_ptr, _incref_ptr, _incref_structref, _ptr_from_struct_incref,
    _load_ptr, _ptr_to_data_ptr)
from cre.utils import assign_to_alias_in_parent_frame, encode_idrec, _obj_cast_codegen
from cre.vector import VectorType
from cre.obj import cre_obj_field_dict,CREObjType, CREObjTypeClass, CREObjProxy, set_chr_mbrs
from cre.type_conv import ptr_to_var_name
# from cre.predicate_node import BasePredicateNode,BasePredicateNodeType, get_alpha_predicate_node_definition, \
# get_beta_predicate_node_definition, deref_attrs, define_alpha_predicate_node, define_beta_predicate_node, AlphaPredicateNode, BetaPredicateNode
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.experimental.structref import StructRefProxy


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

class VarTypeClass(CREObjTypeClass):
    t_id = T_ID_VAR
    type_cache = {}

    def __new__(cls, base_type=None, head_type=None):
        if(head_type is None): head_type = base_type
        
        unq_tup = (base_type, head_type)

        self = cls.type_cache.get(unq_tup, None)
        if(self is not None):
            return self

        self = super().__new__(cls)
        self.base_type = base_type        
        self.head_type = head_type
        cls.type_cache[unq_tup] = self

        if(base_type is None or head_type is None):
            field_dict = {**var_fields_dict}
        else:
            field_dict = {**var_fields_dict,
                "base_type" : types.TypeRef(base_type),
                "head_type" : types.TypeRef(head_type)
            }
        types.StructRef.__init__(self,[(k,v) for k,v in field_dict.items()])
        
        # print(self.field_dict)
        self.name = repr(self)
        return self

    def __init__(self,*args,**kwargs):
        pass

    def __str__(self):
        if(self.base_type is None and self.head_type is None):
            return f"VarType"
        elif(self.base_type == self.head_type):
            return f"VarType[{self.base_type}]"
        else:
            return f"VarType[{self.base_type}->{self.head_type}]"
    __repr__ = __str__

# @lower_cast(VarTypeClass, CREObjType)
# def upcast(context, builder, fromty, toty, val):
#     return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)



# Manually register the type to avoid automatic getattr overloading 
default_manager.register(VarTypeClass, models.StructRefModel)

VarType = VarTypeClass()
register_global_default("Var", VarType)

# Allow typed Var instances to be upcast to VarType
@lower_cast(VarTypeClass, VarType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)

class Var(StructRefProxy):
    t_id = T_ID_VAR
    def __new__(cls, typ, alias="", skip_assign_alias=False):
        # if(not isinstance(typ, types.StructRef)): typ = typ.fact_type
        typ = _standardize_type(typ, cre_context())
        # base_type_name = str(typ)
        # print(base_type_name)
                
        base_t_id = cre_context().get_t_id(_type=typ)

        if(getenv("CRE_SPECIALIZE_VAR_TYPE",default=False)):
            raise ValueError("THIS SHOULDN'T HAPPEN")
            typ_ref = types.TypeRef(typ)
            struct_type = VarTypeClass(typ_ref,typ_ref)
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
                struct_type = VarTypeClass(base_type, head_type)
            else:
                raise NotImplemented("Haven't implemented getitem() when CRE_SPECIALIZE_VAR_TYPE=true.")
            new_var = var_append_deref(self, attr_or_ind)#self, attr, a_id, offset, head_type_name, t_id, deref_info_type)
        else:
            head_t_id = cre_context().get_t_id(_type=head_type)
            # head_t_id = getattr(head_type, "t_id", -1)
            # print("type, a_id, offset, head_t_id", deref_info_type, a_id, offset, head_t_id)
            new_var = generic_var_append_deref(self, u4(a_id), offset, head_t_id, typ=deref_info_type)
            # new = generic_var_append_deref(self, attr, a_id, offset, head_type_name, t_id, deref_info_type)
            # struct_type = VarType
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
        elif(attr == 'head_type' or attr == 'return_type'):
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
        from cre.default_funcs import LessThan, FactIdrecsLessThan
        if(isinstance(other,Var) and isinstance(other.head_type,Fact)):
            return FactIdrecsLessThan(self,other)
        else:
            return LessThan(self, other)
    def __le__(self, other): 
        from cre.default_funcs import LessThanEq
        return LessThanEq(self, other)
            
    def __gt__(self, other): 
        from cre.default_funcs import GreaterThan, FactIdrecsLessThan
        if(isinstance(other,Var) and isinstance(other.head_type,Fact)):
            return FactIdrecsLessThan(other,self)
        else:
            return GreaterThan(self, other)

    def __ge__(self, other):
        from cre.default_funcs import GreaterThanEq
        return GreaterThanEq(self, other)
    def __eq__(self, other): 
        from cre.default_funcs import Equals, ObjEquals, ObjIsNone
        from cre.conditions import to_cond

        # with PrintElapse("new_ptr_op"):
        #     npo = ObjIsNone(self)

        # with PrintElapse("op_to_cond"):
        #     op_to_cond(npo)
        
        if(other is None):
            # with PrintElapse("new_ObjIsNone"):
            return to_cond(ObjIsNone(self))
        if(isinstance(other,Var) and isinstance(other.head_type,Fact)):
            # print("ObjEquals")
            # with PrintElapse("new_ObjEquals"):
            return to_cond(ObjEquals(self,other))
        # print("Equals")
        return Equals(self, other)
    def __ne__(self, other): 
        return ~(self == other)
        # from cre.default_funcs import Equals, ObjEquals, ObjIsNone
        # if(other is None):
        #     return ~ObjIsNone(other)
        # if(isinstance(other,Var) and isinstance(other.head_type,Fact)):
        #     return ~ObjEquals(self,other)
        # return ~Equals(self, other)

    def __add__(self, other):
        from cre.default_funcs import Add
        return Add(self, other)

    def __radd__(self, other):
        from cre.default_funcs import Add
        return Add(other, self)

    def __sub__(self, other):
        from cre.default_funcs import Subtract
        return Subtract(self, other)

    def __rsub__(self, other):
        from cre.default_funcs import Subtract
        return Subtract(other, self)

    def __mul__(self, other):
        from cre.default_funcs import Multiply
        return Multiply(self, other)

    def __rmul__(self, other):
        from cre.default_funcs import Multiply
        return Multiply(other, self)

    def __truediv__(self, other):
        from cre.default_funcs import Divide
        return Divide(self, other)

    def __rtruediv__(self, other):
        from cre.default_funcs import Divide
        return Divide(other, self)

    def __floordiv__(self, other):
        from cre.default_funcs import FloorDivide
        return FloorDivide(self, other)

    def __rfloordiv__(self, other):
        from cre.default_funcs import FloorDivide
        return FloorDivide(other, self)

    def __pow__(self, other):
        from cre.default_funcs import Power
        return Power(self, other)

    def __rpow__(self, other):
        from cre.default_funcs import Power
        return Power(other, self)

    def __mod__(self, other):
        from cre.default_funcs import Modulus
        return Modulus(other, self)

    def __and__(self, other):
        from cre.conditions import conditions_and, to_cond
        from cre.func import CREFunc
        # if(isinstance(other,CREFunc)):
        #     other = to_cond(other)
        out = conditions_and(self, other)
        return out

    def __or__(self, other):
        from cre.conditions import conditions_or, to_cond
        from cre.func import CREFunc
        # if(isinstance(other,CREFunc)): other = to_cond(other)
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

    @classmethod
    def from_ptr(cls, ptr):
        return var_from_ptr(ptr)

    def copy(self):
        return var_copy(self)

    def __copy__(self):
        return var_copy(self)

    def with_alias(self, alias):
        return var_with_alias(self, alias)

    def __call__(self, fact):
        deref_infos = var_get_deref_infos(self)
        if(len(deref_infos) > 0):
            data_ptr = resolve_deref_data_ptr(fact, deref_infos)
            load_ptr_i8 = get_load_ptr_impl(i8)
            if(load_ptr_i8(data_ptr) == 0):
                return None
            t_id = var_get_head_t_id(self)
            load_ptr_fact = get_load_ptr_impl(cre_context().get_type(t_id=t_id))            
            return load_ptr_fact(data_ptr)
        else:
            return fact
        



# Manually define the boxing to avoid constructor overloading
define_boxing(VarTypeClass, Var)

#### Get Attribute Overloading ####
from numba.core.errors import NumbaAttributeError
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

        # print("<<", attr, typ.field_dict)
        head_type = typ.head_type
        #TODO Should check that alld subtype references are valid
        if(not hasattr(head_type,'field_dict')):
            raise NumbaAttributeError(f"Cannot dereference attribute '{attr}' of {typ}.")

        # base_type = typ.field_dict['base_type']
        if(attr in head_type.field_dict):
            # head_type = types.TypeRef(head_type.field_dict[attr])
            # field_dict = {
            #     **var_fields_dict,
            #     **{"base_type" : base_type,
            #      "head_type" : head_type}
            # }
            attrty = VarTypeClass(typ.base_type, head_type.field_dict[attr])
            # print("NEW", attrty)
            # attrty = VarTypeClass([(k,v) for k,v, in field_dict.items()])
            return attrty
        else:
            raise NumbaAttributeError(f"{typ} has no attribute '{attr}'")

@lower_getattr_generic(VarTypeClass)
def var_getattr_impl(context, builder, typ, val, attr):
    
    #If the attr is 'alias' then retrieve the base var's alias
    if(attr == "alias"):
        st = cgutils.create_struct_proxy(typ)(context, builder, value=val)._getvalue()

        def get_alias(self):
            if(self.base_ptr != cast(self, i8)):
                base = cast(self.base_ptr, VarType)
            else:
                base = cast(self, VarType)
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

        new_var_type = VarTypeClass(base_type,head_type)

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
    context = cre_context()
    deref_infos = var_get_deref_infos(self)
    deref_attrs = []
    typ = self.base_type
    for i, x in enumerate(deref_infos):
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

        typ = context.get_type(t_id=x['t_id'])
    return deref_attrs

# ### Methods that require python interpreter ### 
# # Note: could get around this is there was some way to load cre_context.context_data
# #  as some kind of global variable within the numba runtime

# @generated_jit
# @overload_method(VarTypeClass, "get_head_type_name")
# def get_head_type_name(self):
#     def impl(self):
#         # print("GET HEAD")
#         with objmode(head_type_name=unicode_type):
#             context = cre_context()
#             head_type_name = str(context.get_type(t_id=self.head_t_id))
#         return head_type_name
#     return impl

# @generated_jit
# @overload_method(VarTypeClass, "get_base_type_name")
# def get_base_type_name(self):
#     def impl(self):
#         with objmode(base_type_name=unicode_type):
#             context = cre_context()
#             base_type = context.get_type(t_id=self.base_t_id)
#             base_type_name = getattr(base_type,'_fact_name',str(base_type))
#         return base_type_name
#     return impl


# @njit(ListType(unicode_type)(VarType), cache=True)
@njit(cache=True)
def get_deref_attrs(self):
    deref_attrs = List.empty_list(unicode_type)
    context_data = get_cre_context_data()
    t_id = self.base_t_id
    for di in self.deref_infos:
        if(di.type == DEREF_TYPE_LIST):
            deref_attrs.append(f"[{di.a_id}]")
        else:
            deref_attrs.append(context_data.attr_names[(u2(t_id), u1(di.a_id))])
        t_id = di.t_id
    for di in self.deref_infos:
        deref_attrs.append(context_data.attr_names[(u2(t_id), u1(di.a_id))])
        t_id = di.t_id
    return deref_attrs

# @njit(unicode_type(VarType), cache=True)
@njit(cache=True)
def get_deref_attrs_str(self):
    context_data = get_cre_context_data()
    t_id = self.base_t_id
    s = ""
    for di in self.deref_infos:
        if(di.type == DEREF_TYPE_LIST):
            s += f"[{di.a_id}]"
        else:
            s += f".{context_data.attr_names[(u2(t_id), u1(di.a_id))]}"
        t_id = di.t_id
    return s


@njit(unicode_type(VarType), cache=True)
def var_str(self):
    alias = self.alias
    deref_str = get_deref_attrs_str(self)
    if(len(alias)==0):
        return f'{ptr_to_var_name(self.base_ptr)}{deref_str}'
    else:
        return f'{self.alias}{deref_str}' 

# @generated_jit(cache=True)
@overload_method(VarTypeClass, '__str__')
@overload(str)
def var_str_overload(self):
    if(not isinstance(self,VarTypeClass)): return
    def impl(self):
        return var_str(self)
    return impl






# def var_cmp_alpha(left_var, op_str, right_var,negated):
#     from cre.conditions import pt_to_cond, gen_pterm_ctor_alpha, gen_pterm_ctor_beta
#     # Treat None as 0 for comparing against a fact ref
#     print("***", isinstance(left_var.head_type, Fact),isinstance(left_var.head_type, types.StructRef), left_var.head_type)
#     if(right_var is None and isinstance(left_var.head_type, types.StructRef)): right_var = 0
#     right_var_type = types.unliteral(types.literal(right_var)) #if (isinstance(right_var, types.NoneType)) else types.int64
#     ctor = gen_pterm_ctor_alpha(left_var, op_str, right_var_type)
#     pt = ctor(left_var, op_str, right_var)
#     lbv = cast_structref(VarType,left_var)
#     return pt_to_cond(pt, lbv, None, negated)
    

# def var_cmp_beta(left_var, op_str, right_var, negated):
#     from cre.conditions import pt_to_cond, gen_pterm_ctor_alpha, gen_pterm_ctor_beta
#     ctor = gen_pterm_ctor_beta(left_var, op_str, right_var)
#     pt = ctor(left_var, op_str, right_var)
#     lbv = cast_structref(VarType,left_var)
#     rbv = cast_structref(VarType,right_var)
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



@njit(VarType(i8), cache=True)
def var_from_ptr(ptr):
    return cast(ptr, VarType)

@njit(cache=True)    
def get_var_ptr(self):
    return cast(self, i8)

@njit(cache=True)    
def get_var_ptr_incref(self):
    return _ptr_from_struct_incref(self)


# var_type_cache = {}
# def VarTypeClass(base_type, head_type=None):
#     if(head_type is None): head_type = base_type
#     t = (base_type, head_type)
#     if(t not in var_type_cache):
#         # print((str(t[0]),str(t[1])), t[0].t_id)
#         # d = {**var_fields_dict,**{
#         #     'base_type': types.TypeRef(base_type),
#         #     'head_type': types.TypeRef(head_type),
#         #     }}

#         struct_type = VarTypeClass(base_type, head_type)
#         var_type_cache[t] = struct_type
#         return struct_type
#     else:
#         # print("RETREIVED", (str(t[0]),str(t[1])),t[0].t_id)
#         return var_type_cache[t]

@njit(cache=True)
def var_ctor_generic(base_t_id, alias=""):
    st = new(VarType)
    lower_setattr(st,'idrec', encode_idrec(T_ID_VAR,0,0xFF))
    lower_setattr(st,'is_not', u1(0))
    lower_setattr(st,'conj_ptr', i8(0))
    lower_setattr(st,'base_t_id', base_t_id)
    lower_setattr(st,'head_t_id', base_t_id)
    lower_setattr(st,'base_ptr', cast(st, i8))
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
    # st.base_ptr = i8(cast(st,i8))
    # st.base_ptr_ref = ptr_t(0)
    # st.alias =  "" if(alias is  None) else alias
    # st.deref_attrs_str = None
    # # st.deref_attrs = List.empty_list(unicode_type)
    # st.deref_infos = np.empty(0,dtype=deref_info_type)
    return st


@njit
def var_ctor(var_type, base_t_id, alias):
    return cast(var_ctor_generic(base_t_id, alias),var_type)


@overload(Var)
def overload_Var(typ,alias=""):
    if(hasattr(typ, 'instance_type')):
        _typ = typ.instance_type
        struct_type = VarTypeClass(_typ,_typ)
        base_t_id = cre_context().get_t_id(_type=_typ)
        # print("@@ IMPL VAR :: ", _typ, base_t_id)
        def impl(typ, alias=""):
            return var_ctor(struct_type, base_t_id, alias)
    else:
        # Case when typ is t_id
        def impl(typ, alias=""):
            base_t_id = typ
            return var_ctor(VarType, base_t_id, alias)
    return impl


#### getattr and dereferencing ####

# @njit(types.void(VarType, VarType),cache=True)
# def var_memcopy(self,st):
#     # new_deref_attrs = List.empty_list(unicode_type)
#     # new_deref_infos = np.empty(len(),dtype=deref_info_type)
#     # for x in lower_getattr(self,"deref_attrs"):
#     #     new_deref_attrs.append(x)
#     # old_deref_infos
#     # for i,y in enumerate(lower_getattr(self,"deref_infos")):
#     #     new_deref_infos[i] = y
    

# @generated_jit(cache=True)
# def var_append_deref(self,deref):

# ------------------------
# : Copying, extending, and renaming Vars
#   Note: We use lower_get/setattr to get around the fact that Var().whatever 
#     creates a new var extended with a dereference.        

@njit(VarType(VarType), cache=True)
def var_copy(self):
    st = new(VarType)

    deref_infos = lower_getattr(self,"deref_infos").copy()
    lower_setattr(st,'deref_infos', deref_infos)

    if(len(deref_infos) == 0):
        # If we are copying a base Var then it is its own base.
        lower_setattr(st,'base_ptr', cast(st, i8))
        lower_setattr(st,'base_ptr_ref', ptr_t(0))
        lower_setattr(st,'alias_', lower_getattr(self,"alias_"))
    else:
        # If we are copying a Var with derefs then we need to borrow a
        #  reference to the original base.
        base_ptr_ref = lower_getattr(self, "base_ptr_ref")
        if(not i8(base_ptr_ref) == 0):
            _incref_ptr(base_ptr_ref)
        lower_setattr(st,'base_ptr_ref', base_ptr_ref)
        lower_setattr(st,'base_ptr', lower_getattr(self,"base_ptr"))
    
    lower_setattr(st,'idrec', lower_getattr(self,"idrec"))
    lower_setattr(st,'is_not', lower_getattr(self,"is_not"))
    lower_setattr(st,'base_t_id', lower_getattr(self,"base_t_id"))
    lower_setattr(st,'head_t_id', lower_getattr(self,"head_t_id"))
    lower_setattr(st,'deref_attrs_str', lower_getattr(self,"deref_attrs_str"))
    return st    

@njit(VarType(VarType, unicode_type), cache=True)
def var_with_alias(self, alias):
    st = var_copy(self)
    if(len(self.deref_infos) == 0):
        lower_setattr(st, "alias_", alias)
    else:
        # If realiasing a Var with derefs then we need to make a new base
        old_base = cast(lower_getattr(self,"base_ptr"), VarType)
        new_base = var_copy(old_base)
        lower_setattr(new_base, "alias_", alias)
        # Borrow a reference to the new base
        lower_setattr(st, "base_ptr_ref", _ptr_from_struct_incref(new_base))
        lower_setattr(st, "base_ptr", cast(new_base,i8))
    return st


@njit(types.void(VarType, u4, i4, u2, u1), cache=True)
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


# Stub function
def var_append_deref(self, attr):
    pass

@overload(var_append_deref)
def overload_var_append_deref(self, attr):
    SentryLiteralArgs(['attr']).for_function(var_append_deref).bind(self,attr)
    if(self is VarType):
        raise ValueError("var_append_deref() doesn't work on VarType. Use generic_var_append_deref()")

    old_var_type = self
    old_head_type = self.field_dict['head_type'].instance_type
    attr = attr.literal_value
    a_id = old_head_type.get_attr_a_id(attr)
    offset = old_head_type.get_attr_offset(attr)
    # print(old_head_type.instance_type.field_dict)
    head_type = old_head_type.field_dict[attr]
    # base_type = old_var_type.field_dict['base_type'].instance_type
    head_t_id = cre_context().get_t_id(_type=head_type)
    
    # var_struct_type = VarTypeClass(base_type, head_type)

    # print("CONSTR VAR", base_type, head_type, self.name)
    # print("<<", var_struct_type)
    if(isinstance(attr,int)):
        typ = DEREF_TYPE_ATTR
    else:
        typ = DEREF_TYPE_LIST

    # print("AFT")
    def impl(self, attr):
        # st = new(var_struct_type)
        # var_memcopy(cast(self, VarType),st)
        # lower_setattr(st, 'base_ptr_ref', _ptr_from_struct_incref(self))
        st = var_copy(self)
        _var_append_deref(st, a_id, offset, head_t_id, typ=DEREF_TYPE_ATTR)
        was_base = len(self.deref_infos) == 0
        if(was_base):
            # If was a base var then borrow its pointer
            lower_setattr(st, "base_ptr", cast(self, i8))
            lower_setattr(st, 'base_ptr_ref', _ptr_from_struct_incref(self))
        lower_setattr(st,'deref_attrs_str', None)
        return st
    return impl


# def deref_var(struct_type, base_var, attr, a_id, offset, head_type_name, t_id=-1, typ=DEREF_TYPE_ATTR):
#     # _incref_structref(base_var)
#     st = new(struct_type)
#     var_memcopy(base_var,st)
#     _var_append_deref(st,attr, a_id, offset, t_id, head_type_name, typ)
#     lower_setattr(st, 'base_ptr_ref', _ptr_from_struct_incref(base_var))
#     return st

@njit(VarType(VarType, u4, i8, i8, i8), cache=True)
def generic_var_append_deref(self, a_id, offset, head_t_id, typ=DEREF_TYPE_ATTR):
    # _incref_structref(base_var)
    st = var_copy(self)
    # st = new(VarType)
    was_base = len(self.deref_infos) == 0
    # var_memcopy(self, st)
    if(was_base):
        # If was a base var then borrow its pointer
        lower_setattr(st, "base_ptr", cast(self, i8))
        lower_setattr(st, 'base_ptr_ref', _ptr_from_struct_incref(self))
    _var_append_deref(st, a_id, offset, head_t_id, typ=typ)
    lower_setattr(st,'deref_attrs_str', None)
    

    return st

@njit(VarType(VarType, deref_info_type[::1]), cache=True)
def var_extend(self, deref_infos):
    st = var_copy(self)
    was_base = len(self.deref_infos) == 0
    if(was_base):
        # If was a base var then borrow its pointer
        lower_setattr(st, "base_ptr", cast(self, i8))
        lower_setattr(st, 'base_ptr_ref', _ptr_from_struct_incref(self))
    # st = new(VarType)
    # was_base = len(self.deref_infos) == 0
    # var_memcopy(self, st)
    # if(was_base):
    #     # If was a base var then borrow its pointer
    #     lower_setattr(st, 'base_ptr_ref', _ptr_from_struct_incref(self))

    new_deref_infos = np.empty(len(self.deref_infos) + len(deref_infos), dtype=deref_info_type)
    new_deref_infos[:len(self.deref_infos)] = self.deref_infos
    new_deref_infos[len(self.deref_infos):] = deref_infos
    lower_setattr(st, 'deref_infos', new_deref_infos)
    lower_setattr(st, 'head_t_id', new_deref_infos[-1].t_id)
    lower_setattr(st,'deref_attrs_str', None)
    
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


@njit(types.void(VarType, unicode_type), cache=True)
def var_assign_alias(var, alias):
    base = cast(var.base_ptr, VarType)
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

# @njit(cache=True)
# def _var_apply_deref(self, fact):
#     data_ptr = resolve_deref_data_ptr(fact, self.deref_infos)
#     return data_ptr, self.head_t_id


from numba.core.typing.typeof import typeof
load_ptr_overloads = {}
def get_load_ptr_impl(nb_val_type):
    ''' Implementation for loading a value 'val_type' from a data pointer'''
    if(nb_val_type is Fact):
        nb_val_type = BaseFact
    if(nb_val_type not in load_ptr_overloads):
        @njit(nb_val_type(i8),cache=True)
        def load_ptr_impl(data_ptr):
            val = _load_ptr(nb_val_type, data_ptr)
            return val
        load_ptr_overloads[nb_val_type] = load_ptr_impl
    return load_ptr_overloads[nb_val_type]
