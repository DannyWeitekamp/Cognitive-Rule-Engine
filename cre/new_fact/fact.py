import numpy as np
from numba import types, njit, i8, i4, i2, i1, u1, u2, f8
from numba.types import unicode_type, boolean
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple, Array
from numba.extending import (models, register_model, type_callable,
        typeof_impl, models, register_model, make_attribute_wrapper,
        overload_attribute, lower_builtin, unbox, box, NativeValue,
        overload_method, overload, intrinsic)
from numba.core.datamodel.registry import register_default
from numba.core import cgutils
from numba.cpython.unicode import _malloc_string, _strncpy, _kind_to_byte_width
from llvmlite import ir
from llvmlite import binding as ll
from llvmlite.ir import types as ll_types
from llvmlite.ir import Constant

from cre.core import DEFAULT_TYPE_T_IDS, DEFAULT_T_ID_TYPES, T_ID_CONDITIONS, T_ID_LITERAL, T_ID_FUNC, T_ID_FACT, T_ID_VAR, T_ID_UNDEFINED, T_ID_BOOL, T_ID_INT, T_ID_FLOAT, T_ID_STR, T_ID_TUPLE_FACT
from cre.utils import cast, PrintElapse, get_ep, _meminfo_from_struct
from cre.obj import CREObjType, CREObjProxy
from numba.core.datamodel import default_manager, models
from numba.experimental.structref import _Utils
from numba.experimental.function_type import _get_wrapper_address
from numba.core.typing import signature

from numba.core.imputils import (lower_builtin, lower_getattr,
                                 lower_getattr_generic,
                                 lower_setattr_generic,
                                 lower_cast, lower_constant,
                                 iternext_impl, impl_ret_borrowed,
                                 impl_ret_new_ref, impl_ret_untracked,
                                 RefType)

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
from numba.core.typing.templates import AttributeTemplate
from numba.core.datamodel import default_manager, models
from numba.core import cgutils, utils as numba_utils
from numba.experimental.structref import _Utils, imputils
from numba.typed.typedobjectutils import _nonoptional


import operator
from cre.utils import encode_idrec, _decref_ptr
from cre.context import cre_context
from cre.new_fact.member import MemberTypeClass, MemberType, Member
from cre.new_fact.fact_intrinsics import define_boxing

## NEW FACT STUFF

from cre.obj import cre_obj_field_dict, CREObjTypeClass

fact_fields_dict = {
    **cre_obj_field_dict,
    "length" : i8,
}


class FactTypeClass(CREObjTypeClass, types.SimpleIterableType):
    def __init__(self, name=None, spec={}, hash_code=None):
        if(isinstance(spec, dict)):
            mbr_fields = [(k, MemberTypeClass(v['type'])) for k, v in spec.items()]
        else:
            mbr_fields = [(k, MemberTypeClass(t)) for k, t in spec]
        fact_fields = [(k,v) for k,v in fact_fields_dict.items()]
        
        if(name is None):            
            self.name = f'FactType'
        else:
            self._hash_code = hash_code if hash_code else unique_hash_v([name, spec_fields])        
            self.name = f'{name}_{hash_code}'
            self._fact_name = name

        # A required internals for StructRef
        self._mbr_fields = mbr_fields 
        self._fields = fact_fields + mbr_fields 
        self._typename = self.__class__.__qualname__
        context = cre_context()
        self._mbr_t_ids = np.array([context.get_t_id(mtyp.val_type) for _, mtyp in mbr_fields], dtype=np.uint16)
        self.spec = spec

    def __call__(self, *args, **kwargs):
        ''' If a fact_type is called with types return a signature 
            otherwise use it's ctor to return a new instance'''
        if len(args) > 0 and isinstance(args[0], types.Type):
            return signature(self, *args)

        # Ensure that any lists are converted to typed lists
        # args = [List(x) if(isinstance(x,list)) else x for x in args]
        # kwargs = {k: (List(v) if(isinstance(v,list)) else v) for k,v in kwargs.items()}
        fact_meminfo = new_fact_meminfo(self._mbr_t_ids)
        proxy_cls = getattr(self, '_proxy_class', FactProxy)
        fact = proxy_cls._numba_box_no_recover_(self, fact_meminfo)
        # print(self, fact_meminfo)
        for i, val in enumerate(args):
            if(isinstance(val,list)):
                val = List(val)
            fact[i] = val
        for attr, val in kwargs.items():
            if(isinstance(val,list)):
                val = List(val)
            setattr(fact, attr, val)
        return fact

        # return self._ctor[0](*args, **kwargs)

    def __str__(self):
        if(hasattr(self, '_fact_name')):
            return f'{self._fact_name}_{self._hash_code[:10]}'
        else:
            return "FactType"

    # @property
    # def field_dict_keys(self):
    #     if(not hasattr(self, "_field_dict_keys")):
    #         self._field_dict_keys = [x[0] for x in self._fields]
    #     return self._field_dict_keys

    @property
    def iterator_type(self):
        return FactIteratorType

    @property
    def clean_spec(self):
        if(not hasattr(self,'_clean_spec')):
            from cre.new_fact.spec import clean_spec
            self._clean_spec = clean_spec(self.spec)
        return self._clean_spec

    def __getstate__(self):
        d = self.__dict__.copy()
        if('_clean_spec' in d): del d['_clean_spec']

        # NOTE: While ops still use py_classes we need clean out 'conversions' from the spec
        if('spec' in d):

            d_spec = d['spec'].copy()
            
            for attr, attr_spec in d_spec.items():
                if('conversions' in attr_spec):
                    attr_spec_copy = attr_spec.copy()
                    attr_spec_copy['conversions'] = tuple([(attr_spec['type'], attr, typ) for typ in attr_spec_copy['conversions']])
                    d_spec[attr] = attr_spec_copy
            d['spec'] = d_spec
            # print(d)
        if('_code' in d): del d['_code']
                    
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    

# Manually register the type to avoid automatic getattr overloading 
default_manager.register(FactTypeClass, models.StructRefModel)
FactType = FactTypeClass()

# Allow any fact to be upcast to FactType
@lower_cast(FactTypeClass, FactType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)



# register_global_default("Fact", FactType)

# ----------------------------------
# FactProxy

###### Fact Definition #######
class FactProxy(CREObjProxy):
    # '''Essentially the same as numba.experimental.structref.StructRefProxy 0.51.2
    #     except that __new__ is not defined to statically define the constructor.
    # '''
    @classmethod
    def _numba_box_(cls, mi):
        """Called by boxing logic, the conversion of Numba internal
        representation into a PyObject.

        Parameters
        ----------
        mi :
            a wrapped MemInfoPointer.

        Returns
        -------
        instance :
             a FactProxy instance.
        """
        inst = super(FactProxy,cls)._numba_box_(FactType,mi)
        return inst

    @classmethod
    def _numba_box_no_recover_(cls, ty, mi):
        '''Same as StructRefProxy'''
        instance = ty.__new__(cls)
        instance._type = ty
        instance._meminfo = mi
        return instance

    @property
    def _numba_type_(self):
        """Returns the Numba type instance for this structref instance.

        Subclasses should NOT override.
        """
        return self._fact_type

    def restore(self,context=None):
        context = cre_context(context)

    def get_ptr(self):
        return fact_to_ptr(self)

    def get_ptr_incref(self):
        return fact_to_ptr_incref(self)

    def _gen_val_var_possibilities(self, self_var):
        for attr, config in self._fact_type.clean_spec.items():
            typ = config['type']
            # with PrintElapse("getattr_var"):
            val = getattr(self,attr)
            # with PrintElapse("getattr_var"):
            attr_var = getattr(self_var, attr)
            if(isinstance(val, List)):
                for i in range(len(val)):
                    item_var = attr_var[i]
                    item_val = val[i]
                    yield (item_val, item_var)
            else:
                yield (val, attr_var)

    def as_conditions(self, fact_ptr_to_var_map=None, keep_null=True, add_implicit_neighbor_self_refs=True, neigh_count = 0):
        from cre.default_funcs import Equals
        from cre.utils import as_typed_list
        from cre.dynamic_exec import var_eq
        from cre.var import Var

        self_ptr = self.get_ptr()

        if(fact_ptr_to_var_map is None):
             fact_ptr_to_var_map = {self_ptr : Var(self._fact_type, "X")}
            

        self_var = fact_ptr_to_var_map[self_ptr]
        one_lit_conds = []
        
        # with PrintElapse("CONSTRUCTS"):       
        # for attr, config in self._fact_type.spec.items():
        for attr_val, attr_var in self._gen_val_var_possibilities(self_var):
            if(isinstance(attr_val, FactProxy)):
                # Fact case
                attr_val_fact_ptr = attr_val.get_ptr()
                if(attr_val_fact_ptr not in fact_ptr_to_var_map):
                    if(add_implicit_neighbor_self_refs):
                        nbr_var = Var(attr_val._fact_type, f"Nbr{neigh_count}")
                        fact_ptr_to_var_map[attr_val_fact_ptr] = nbr_var
                        neigh_count += 1
                    else:
                        continue


                val_var = fact_ptr_to_var_map[attr_val_fact_ptr]
                #   FIXME: use cre_obj.__eq__()
                
                    # str(attr_var) == str(val_var)
                # print("<<", str(attr_var), str(val_var), var_eq(attr_var, val_var))
                if(add_implicit_neighbor_self_refs and str(attr_var) == str(val_var)):
                    # for case like x.next == x.next, try make conditions like x == x.next.prev
                    # with PrintElapse("LOOP"):
                        #     list(attr_val._gen_val_var_possibilities(attr_var))
                    for attr_val2, attr_var2 in attr_val._gen_val_var_possibilities(attr_var):

                        if(isinstance(attr_val2, FactProxy) and 
                            attr_val2.get_ptr() == self_ptr):
                            one_lit_conds.append(self_var==attr_var2)

                else:
                    # with PrintElapse("NEW LIT"):
                    one_lit_conds.append(attr_var==fact_ptr_to_var_map[attr_val_fact_ptr])
                
            else:
                # Note: Making literals with primitives is slow w/ current Op 
                #  implmenentation since it compiles compositions. 

                # Primitive case
                if(not keep_null and attr_val is None): continue
                one_lit_conds.append(attr_var==attr_val)

        # with PrintElapse("ANDS"):    
        # print(fact_ptr_to_var_map)
        _vars = list({v.get_ptr():v for v in fact_ptr_to_var_map.values()}.values())
        # print(_vars)   
        conds = _vars[0]
        for i in range(1, len(_vars)):
            conds = conds & _vars[i]

        for c in one_lit_conds:
            conds = conds & c

        return conds

    def isa(self, typ):
        return isa(self,typ)

    def asa(self, typ):
        if(not isa(self,typ)):
            raise TypeError(f"Cannot cast fact '{str(self)}' to '{str(typ)}.'")
        return super().asa(typ)

    def __repr__(self):
        return str(self)


    def resolve_deref(self, derefs):
        if(not isinstance(derefs, np.ndarray)):
            derefs = derefs.deref_infos
        ctx = cre_context()
        out_type = ctx.get_type(t_id=derefs[-1]['t_id'])
        return resolve_deref(self, derefs, out_type)

    # def __getitem__(self, i):
    #     self._getters[i](self)

    # def __setitem__(self, i, val):
    #     print(self, self._setters[i])
    #     self._setters[i](self, val)

define_boxing(FactTypeClass, FactProxy)


# ----------------------------------
# Fact Deconstructor 


def impl_fact_dtor(context, module):
    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = ir.FunctionType(ir.VoidType(),
                                 [llvoidptr, llsize, llvoidptr])

    fname = "CRE_Fact_Dtor"
    dtor_fn = cgutils.get_or_insert_function(module, dtor_ftype, fname)
    if dtor_fn.is_declaration:
        # Define
        builder = ir.IRBuilder(dtor_fn.append_basic_block())
        # data_ptr = builder.ptrtoint(dtor_fn.args[0], cgutils.intp_t)

        alloc_fe_type = FactType.get_data_type()
        alloc_type = context.get_value_type(alloc_fe_type)

        ptr = builder.bitcast(dtor_fn.args[0], alloc_type.as_pointer())
        data = context.make_helper(builder, alloc_fe_type, ref=ptr)

        extra_fn_type = ir.FunctionType(ir.VoidType(),[llvoidptr])
        extra_fn = cgutils.get_or_insert_function(module, extra_fn_type, "CRE_Fact_Decref_Members")
        builder.call(extra_fn, [builder.bitcast(dtor_fn.args[0], cgutils.voidptr_t)])

        # Decref any non-Member fields
        context.nrt.decref(builder, alloc_fe_type, data._getvalue())

        builder.ret_void()

    return dtor_fn

# ----------------------------------
# Fact Constructor 

@intrinsic
def new_fact(typingctx, length):
    def codegen(context, builder, signature, args):
        payload_type = context.data_model_manager[FactType.get_data_type()].get_value_type()
        mbr_val_type = context.data_model_manager[MemberType].get_value_type()
        
        fact_size = context.get_constant(types.intp, context.get_abi_sizeof(payload_type))
        mbr_size = context.get_constant(types.intp, context.get_abi_sizeof(mbr_val_type))

        length = args[0]

        alloc_size = builder.add(fact_size, builder.mul(mbr_size, length))

        # Allocate
        meminfo = context.nrt.meminfo_alloc_dtor(
            builder,
            alloc_size,
            impl_fact_dtor(context, builder.module),
        )
        data_pointer = context.nrt.meminfo_data(builder, meminfo)

        # Nullify all data
        cgutils.memset(builder, data_pointer, alloc_size, 0)

        # Assign meminfo
        inst_struct = context.make_helper(builder, FactType)
        inst_struct.meminfo = meminfo

        # Assign idrec, Length
        utils = _Utils(context, builder, FactType)
        dataval = utils.get_data_struct(inst_struct._getvalue())
        idrec = context.get_constant(types.intp, encode_idrec(T_ID_FACT,0,u1(-1)))
        setattr(dataval, 'idrec', length)
        setattr(dataval, 'length', length)

        return inst_struct._getvalue()

    sig = FactType(length)
    return sig, codegen



@njit(cache=True)
def new_fact_meminfo(t_ids):
    fact = new_fact(len(t_ids))
    if(t_ids is not None):
        for i, t_id in enumerate(t_ids):
            _set_fact_member_t_id(fact, i, t_id)
    return _meminfo_from_struct(fact)


## --------------------------
# : getitem / setitem codegen + intrinsics

def _fact_get_mbr_data_ptr(context, builder, fact, index, fact_typ=FactType, mbr_typ=MemberType):
    payload_type = context.data_model_manager[FactType.get_data_type()].get_value_type()
    mbr_val_type = context.data_model_manager[mbr_typ].get_value_type()
    fact_size = context.get_constant(types.intp, context.get_abi_sizeof(payload_type))
    mbr_size = context.get_constant(types.intp, context.get_abi_sizeof(mbr_val_type))

    fact = context.make_helper(builder, fact_typ, value=fact)                
    data_ptr = context.nrt.meminfo_data(builder, fact.meminfo)
    data_ptr = builder.ptrtoint(data_ptr, cgutils.intp_t)

    start = builder.add(data_ptr, fact_size)
    offset = builder.mul(mbr_size, index)
    member_ptr = builder.add(start, offset)
    member_ptr = builder.inttoptr(member_ptr, ll_types.PointerType(mbr_val_type))
    return member_ptr

def _fact_get_attr_data_ptr(context, builder, fact, attr, fact_typ):
    field_type = fact_typ.field_dict[attr]
    utils = _Utils(context, builder, fact_typ)

    dataval = utils.get_data_struct(fact)
    return dataval._get_ptr_by_name(attr)

def _resolve_field_type_ptr(context, builder, fact, attr, attr_type, fact_typ, member_type):
    if(isinstance(attr_type, types.Literal)):
        lv = attr_type.literal_value
        if(isinstance(lv, int)):
            index = context.get_constant(types.intp, lv)
            dataptr = _fact_get_mbr_data_ptr(context, builder, fact, index, FactType, member_type)
        else:
            dataptr = _fact_get_attr_data_ptr(context, builder, fact, lv, fact_typ)
    else:
        index = attr
        dataptr = _fact_get_mbr_data_ptr(context, builder, fact, index, fact_typ, member_type)
    return dataptr


def _to_mbr(context, builder, val, val_type, member_type): 
    field_type = member_type.val_type
    
    # print("&&", attr, field_type)
    if(isinstance(field_type, (ListType,))):
        dtype = field_type.dtype
        # if(isinstance(field_type, Fact)):
        if(isinstance(val_type, types.Optional)):
            # If list member assigned to none just instantiate an empty list
            def cast_obj(x):
                if(x is None):
                    return Member(field_type)
                return Member(_cast_list(field_type, _nonoptional(x)))
        else:
            def cast_obj(x):
                if(x is None):
                    return Member(field_type)
                return Member(_cast_list(field_type, x))
        casted = context.compile_internal(builder, cast_obj, member_type(val_type,), (val,))

    elif(isinstance(field_type, (FactTypeClass,))):
        # If fact member assigned to none just assign to NULL pointer
        if(isinstance(val_type, types.Optional)):
            def cast_obj(x):
                if(x is None):
                    return Member(field_type)#cast(0,field_type)
                return Member(cast(_nonoptional(x), field_type))
        else:
            def cast_obj(x):
                if(x is None):
                    return Member(field_type)
                return Member(cast(x, field_type))

        casted = context.compile_internal(builder, cast_obj, member_type(val_type,), (val,))
    else:
        # print(val_type, ">", field_type)

        casted = context.cast(builder, val, val_type, field_type)
        def cast_obj(x):
            return Member(x)
        casted = context.compile_internal(builder, cast_obj, member_type(field_type,), (casted,))
        
    return casted

def _field_type_from_attr_type(fact_typ, attr_type):
    if(isinstance(attr_type, types.Literal)):
        lv = attr_type.literal_value
        if(isinstance(lv, str)):
            field_type = fact_typ.field_dict[lv]
        elif(isinstance(lv, int)):
            field_type = fact_typ._mbr_fields[lv][1]
    elif(isinstance(attr_type, types.Integer)):
        field_type = MemberType
    return field_type


def fact_setattr_codegen(context, builder, sig, args, attr=None, mutability_protected=False):
    # from cre.fact import Fact
    if(len(args) == 2):
        [fact_typ, val_type] = sig.args
        [fact, val] = args
        attr_type = types.literal(attr)
    else:
        [fact_typ, attr_type, val_type] = sig.args
        [fact, attr, val] = args
        if(isinstance(attr_type, types.Literal)):
            attr = attr_type.literal_value

    utils = _Utils(context, builder, fact_typ)
    # print(instance, fact_typ, default_manager[fact_typ].__dict__)
    dataval = utils.get_data_struct(fact)

    if(mutability_protected):
        idrec = getattr(dataval, "idrec")
        # If (idec & 0xFF) != 0, throw an error 
        idrec_set = builder.icmp_unsigned('==', builder.and_(idrec, idrec.type(0xFF)), idrec.type(0))
        with builder.if_then(idrec_set):
            msg =("Facts objects are immutable once declared. Use mem.modify instead.",)
            context.call_conv.return_user_exc(builder, AttributeError, msg)
    
    field_type = _field_type_from_attr_type(fact_typ, attr_type)
    is_member = isinstance(field_type, MemberTypeClass)
    if(is_member):
        if(field_type.val_type is None): 
            field_type = MemberTypeClass(val_type)

        # Make Member (note: increfs underlying value)
        new_value = _to_mbr(context, builder, val, val_type, field_type)        
        dataptr = _resolve_field_type_ptr(context, builder, fact, attr, attr_type, fact_typ, field_type)

        # If old value was object-like then decref it         
        old_value = cgutils.create_struct_proxy(MemberType)(context, builder, ref=dataptr)
        is_obj = builder.icmp_unsigned('>=', old_value.t_id, old_value.t_id.type(T_ID_STR))
        not_null = builder.icmp_unsigned('!=', old_value.val, old_value.val.type(0))

        # TODO Don't check for typed fact assignment
        with builder.if_then(builder.and_(is_obj, not_null)):
            meminfo = builder.inttoptr(old_value.val, cgutils.voidptr_t)
            context.nrt.decref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        # Store the member
        builder.store(new_value, dataptr)

    else:
        new_value = context.cast(builder, val, val_type, field_type)
        # incref new value
        context.nrt.incref(builder, field_type, new_value)

        # decref old value (must be last in case new value is old value)
        old_value = getattr(dataval, attr)
        context.nrt.decref(builder, field_type, old_value)

        # write new
        setattr(dataval, attr, new_value)
    

    if(mutability_protected):
        # Make sure that hash_val is 0 to force it to be recalculated
        setattr(dataval, "hash_val", cgutils.intp_t(0))
    return dataval
        # ret = _obj_cast_codegen(context, builder, ret, field_type, ret_type, False)

    # return imputils.impl_ret_borrowed(context, builder, ret_type, ret)
@intrinsic
def fact_lower_setattr(typingctx, inst_type, attr_type, val_type):
    if (isinstance(attr_type, (types.Literal,types.Integer)) and 
        isinstance(inst_type, types.StructRef)):
        # print("BB", isinstance(inst_type, types.StructRef), inst_type, attr_type)
        
        # attr = attr_type.literal_value
        def codegen(context, builder, sig, args):
            fact_setattr_codegen(context, builder, sig, args)
  
        sig = types.void(inst_type, attr_type, val_type)
        # print(sig)
        return sig, codegen


@intrinsic
def fact_mutability_protected_setattr(typingctx, inst_type, attr_type, val_type):
    # print("<<", inst_type, attr_type, val_type)
    if (isinstance(attr_type, (types.Literal,types.Integer)) and 
        isinstance(inst_type, types.StructRef)):
        # print("BB", isinstance(inst_type, types.StructRef), inst_type, attr_type)
        
        # attr = attr_type.literal_value
        def codegen(context, builder, sig, args):
            fact_setattr_codegen(context, builder, sig, args, 
                mutability_protected=True)
  
        sig = types.void(inst_type, attr_type, val_type)
        # print(sig)
        return sig, codegen

# -----------------
# : getattr

def _to_val(context, builder, ret, ret_type, member_type):#val, val_type, member_type):
    option_ret_type = types.optional(ret_type)

    # print("RT", ret_type)

    if(isinstance(ret_type, (FactTypeClass, ListType))):
        # If a fact member is Null then return None
        def cast_obj(mbr):
            if(mbr.is_none()):
                return None
            return mbr.get_val(ret_type)
        ret = context.compile_internal(builder, cast_obj, option_ret_type(member_type,), (ret,))
        return ret
    elif(not isinstance(ret_type, MemberTypeClass)):
        def cast_obj(mbr):
            return mbr.get_val(ret_type)
        ret = context.compile_internal(builder, cast_obj, ret_type(member_type,), (ret,))
        return ret
    # For untyped Facts just return a Member object
    else:
        return ret


def fact_getattr_codegen(context, builder, sig, args):
    # from cre.fact import Fact
    ret_type = sig.return_type
    fact_typ, attr_type = sig.args
    fact, attr = args

    if(isinstance(attr_type, types.Literal)):
        attr = attr_type.literal_value
    
    # Extract unoptional part of type
    if(isinstance(ret_type, types.Optional)):
        ret_type = ret_type.type

    field_type = _field_type_from_attr_type(fact_typ, attr_type)
    is_member = isinstance(field_type, MemberTypeClass)

    if(is_member):
        dataptr = _resolve_field_type_ptr(context, builder, fact, attr, attr_type, fact_typ, field_type)
        ret = builder.load(dataptr)
        ret = _to_val(context, builder, ret, ret_type, field_type)
    else:
        utils = _Utils(context, builder, fact_typ)
        dataval = utils.get_data_struct(fact)
        ret = getattr(dataval, attr)

    return ret

def _fact_lower_getattr(typingctx, inst_type, attr_type):
    ret_type = resolve_fact_getattr_type(inst_type, attr_type)
    def codegen(context, builder, sig, args):
        return fact_getattr_codegen(context, builder, sig, args)

    sig = ret_type(inst_type, attr_type)
    return sig, codegen

@intrinsic
def fact_lower_getattr(typingctx, inst_type, attr_type):
    if (isinstance(attr_type, (types.Literal,types.Integer)) and 
        isinstance(inst_type, types.StructRef)):
        return _fact_lower_getattr(typingctx, inst_type, attr_type)
        

@intrinsic
def fact_literal_lower_getattr(typingctx, inst_type, attr_type):
    if (isinstance(attr_type, types.Literal) and 
        isinstance(inst_type, types.StructRef)):
        return _fact_lower_getattr(typingctx, inst_type, attr_type)

def resolve_fact_getattr_type(typ, attr):
    from cre.fact import DeferredFactRefType

    if(isinstance(attr, types.Literal)):
        attr = attr.literal_value

    if(isinstance(attr, str)):
        if(attr not in typ.field_dict):
            return
        attrty = typ.field_dict[attr]
    elif(isinstance(attr, int)):
        if(attr >= len(typ._fields)):
            return
        attrty = typ._mbr_fields[attr][1]
    elif(isinstance(attr, types.Integer)):
        attrty = MemberType

    if(isinstance(attrty, MemberTypeClass) and 
        attrty.val_type is not None):
        attrty = attrty.val_type

    # print("B", attrty)
    if(isinstance(attrty, DeferredFactRefType)):
        attrty = attrty.get()

    if(isinstance(attrty,FactTypeClass)):
        attrty = types.optional(attrty)
    if(isinstance(attrty, ListType)):
        if(isinstance(attrty.dtype, DeferredFactRefType)):
            attrty = ListType(attrty.dtype.get())
        attrty = types.optional(attrty)
    
    # print(">>>", typ, attr, attrty)
    return attrty
    # if attr in typ.field_dict:
    #     attrty = typ.field_dict[attr]
    #     # print("<<",attr, attrty)
    #     return attrty


def define_attributes(struct_typeclass):
    # from cre.fact import Fact, base_fact_fields
    """
    Copied from numba.experimental.structref 0.51.2, but added protected mutability
    """
    # print("REGISTER FACT")
    @infer_getattr
    class StructAttribute(AttributeTemplate):
        key = struct_typeclass

        def generic_resolve(self, _typ, attr):
            typ = resolve_fact_getattr_type(_typ, attr)
            # print(attr, _typ,"!>>", typ)
            return typ

    @lower_getattr_generic(struct_typeclass)
    def struct_getattr_impl(context, builder, inst_type, val, attr):
        ret_type = resolve_fact_getattr_type(inst_type, attr)
        sig = ret_type(inst_type,types.literal(attr))
        args = (val, attr)
        return fact_getattr_codegen(context, builder, sig, args)
        

    @lower_setattr_generic(struct_typeclass)
    def struct_setattr_impl(context, builder, sig, args, attr):
        fact_setattr_codegen(context, builder, sig, args, attr, mutability_protected=True)
        
define_attributes(FactTypeClass)

@intrinsic
def _get_fact_member(typingctx, fact, index):
    def codegen(context, builder, signature, args):
        mbr_ptr = _fact_get_mbr_data_ptr(context, builder, args[0], args[1])
        ret = builder.load(mbr_ptr)
        return ret
    return MemberType(fact,index), codegen

@intrinsic
def _set_fact_member(typingctx, fact, index, val):
    def codegen(context, builder, signature, args):
        fact_typ,_,val_typ = signature.args
        mbr_typ = MemberTypeClass(val_typ)

        mbr_ptr = _fact_get_mbr_data_ptr(context, builder, args[0], args[1], fact_typ, mbr_typ)
        
        def make_mbr(val):
            return Member(val)

        mbr = context.compile_internal(builder, make_mbr, mbr_typ(val_typ,), (args[2],))
        builder.store(mbr, mbr_ptr)
    return types.void(fact, index, val), codegen

@intrinsic
def _set_fact_member_t_id(typingctx, fact, index, t_id):
    def codegen(context, builder, signature, args):
        fact_typ,_,val_typ = signature.args
        mbr_typ = MemberTypeClass(val_typ)

        mbr_ptr = _fact_get_mbr_data_ptr(context, builder, args[0], args[1], fact_typ, mbr_typ)
        
        mbr = cgutils.create_struct_proxy(MemberType)(context, builder)
        mbr.t_id = args[2]
        builder.store(mbr._getvalue(), mbr_ptr)
    return types.void(fact, index, t_id), codegen



# ------
# : getitem

@overload(operator.getitem, prefer_literal=True)
def impl_getitem(self, index):
    if not isinstance(self, FactTypeClass):
        return
    if(isinstance(index, types.Literal)):
        def impl(self, index):
            return fact_literal_lower_getattr(self, index)
    else:
        def impl(self, index):
            return fact_lower_getattr(self, index)
    return impl

@overload_method(FactTypeClass, 'get')
def overload_get(self, index, val_typ=None):
    if(val_typ is None or isinstance(val_typ, types.Omitted)):
        def impl(self, index, val_typ=None):
            return _get_fact_member(self, index)
    else:
        def impl(self, index, val_typ=None):
            mbr = _get_fact_member(self, index)
            return mbr.get_val(val_typ)
    return impl


@njit(boolean(FactType, i8), cache=True)
def fact_get_bool(fact, index):
    return fact.get(index, boolean)

@njit(i8(FactType, i8), cache=True)
def fact_get_int(fact, index):
    return fact.get(index, i8)

@njit(f8(FactType, i8), cache=True)
def fact_get_float(fact, index):
    return fact.get(index, f8)

@njit(unicode_type(FactType, i8), cache=True)
def fact_get_str(fact, index):
    return fact.get(index, unicode_type)

# @njit(types.optional(FactType)(FactType, i8), cache=True)
# def fact_get_fact(fact, index):
#     return fact.get(index, FactType)

@njit(cache=True)
def fact_get(fact, index, typ):
    return fact.get(index, unicode_type)


def fact_getitem_impl(typ):
    if(isinstance(typ, types.Boolean)):
        return get_ep(fact_get_bool)
    elif(isinstance(typ, types.Integer)):
        return get_ep(fact_get_int)
    elif(isinstance(typ, types.Float)):
        return get_ep(fact_get_float)
    elif(isinstance(typ, types.UnicodeType)):
        return get_ep(fact_get_str)
    else:
        return lambda fact, index : fact_get(fact, index, typ)

# ------
# : setitem

@overload(operator.setitem)
def impl_setitem(self, index, val):
    if not isinstance(self, FactTypeClass):
        return

    def impl(self, index, val):
        return fact_mutability_protected_setattr(self, index, val)
    return impl

@overload_method(FactTypeClass, 'set')
def overload_getitem(self, index, val):
    def impl(self, index, val):
        fact_mutability_protected_setattr(self, index, val)
    return impl


#   Strong Typed Versions
@njit(types.void(FactType, i8, boolean), cache=True)
def fact_set_bool(fact, index, val):
    fact_mutability_protected_setattr(fact, index, val)

@njit(types.void(FactType, i8, i8), cache=True)
def fact_set_int(fact, index, val):
    fact_mutability_protected_setattr(fact, index, val)

@njit(types.void(FactType, i8, f8), cache=True)
def fact_set_float(fact, index, val):
    fact_mutability_protected_setattr(fact, index, val)

@njit(types.void(FactType, i8, unicode_type), cache=True)
def fact_set_str(fact, index, val):
    fact_mutability_protected_setattr(fact, index, val)

# @njit(types.optional(FactType)(FactType, i8), cache=True)
# def fact_get_fact(fact, index):
#     return fact.get(index, FactType)

#   Multiple Dispatch version
@njit(cache=True)
def fact_set(fact, index, val):
    fact_mutability_protected_setattr(fact, index, val)


#   Implementation Chooser
def fact_setitem_impl(typ):
    if(isinstance(typ, types.Boolean)):
        return get_ep(fact_set_bool)
    elif(isinstance(typ, types.Integer)):
        return get_ep(fact_set_int)
    elif(isinstance(typ, types.Float)):
        return get_ep(fact_set_float)
    elif(isinstance(typ, types.UnicodeType)):
        return get_ep(fact_set_str)
    else:
        return fact_set(fact, index, val)

# Used by FactProxy to generate property fields
def make_proxy_properties(proxy_cls, fields) :
    getters = []
    setters = []
    for i, (attr, typ) in enumerate(fields):

        _g = fact_getitem_impl(typ)
        _s = fact_setitem_impl(typ)
        getters.append(_g)
        setters.append(_s)

        def make_property(_g, _s, i):
            g = lambda self : _g(self, i)
            s = lambda self, val : _s(self, i, val)
            return property(g, s)

        # Set the attribute property 
        setattr(proxy_cls, attr, make_property(_g, _s, i))
    setattr(proxy_cls, '_getters' , getters)
    setattr(proxy_cls, '_setters' , setters)


# ---------------------------------
# : Iterate Members

class FactIteratorTypeClass(types.SimpleIteratorType):
    def __init__(self):
        name = "iter(Fact)"
        super(FactIteratorTypeClass, self).__init__(name, MemberType)

FactIteratorType = FactIteratorTypeClass()


@register_default(FactIteratorTypeClass)
class FactIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        members = [('index', types.EphemeralPointer(types.uintp)),
                   ('fact', FactType)]
        super(FactIteratorModel, self).__init__(dmm, fe_type, members)

make_attribute_wrapper(FactIteratorTypeClass, 'index', 'index')


@lower_builtin('getiter', FactTypeClass)
def getiter_fact(context, builder, sig, args):
    [fact] = args

    iterobj = context.make_helper(builder, sig.return_type)

    zero = context.get_constant(types.intp, 0)
    indexptr = cgutils.alloca_once_value(builder, zero)

    iterobj.index = indexptr
    iterobj.fact = fact

    # Incref array
    if context.enable_nrt:
        context.nrt.incref(builder, FactType, fact)

    res = iterobj._getvalue()

    # Note: a decref on the iterator will dereference all internal MemInfo*
    out = impl_ret_new_ref(context, builder, sig.return_type, res)
    return out



# @lower_builtin('iternext', FactTypeClass)

@lower_builtin('iternext', FactIteratorTypeClass)
@iternext_impl(RefType.BORROWED)
def iternext_factiter(context, builder, sig, args, result):
    iterobj = context.make_helper(builder, FactIteratorType, value=args[0])

    fact = context.make_helper(builder, FactType, value=iterobj.fact)
    utils = _Utils(context, builder, FactType)
    dataval = utils.get_data_struct(iterobj.fact)
    length = getattr(dataval, 'length')

    index = builder.load(iterobj.index)
    is_valid = builder.icmp_signed('<', index, length)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        member_ptr = _fact_get_mbr_data_ptr(context, builder, iterobj.fact, index)
        value = builder.load(member_ptr)

        result.yield_(value)

        # Increment
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)


# --------------------------
# : Decref Members

SIZEOF_NRT_MEMINFO = 48
@njit(types.void(i8),cache=True)
def fact_decref_members(fact_data_ptr):
    fact = cast(fact_data_ptr-SIZEOF_NRT_MEMINFO, FactType)
    for i in range(fact.length):
        mbr = fact[i]
        

fact_decref_members_addr = _get_wrapper_address(fact_decref_members, types.void(i8))
ll.add_symbol("CRE_Fact_Decref_Members", fact_decref_members_addr)



if __name__ == "__main__":
    # new_fact_meminfo(8)

    @njit(cache=True)
    def goo():
        f = new_fact(6)
        f[0] = 1
        f[1] = "123"

        it = iter(f)
        for x in it:
            if(x.t_id == T_ID_STR):
                print(x.get_val(unicode_type))
            elif(x.t_id == T_ID_INT):
                print(x.get_val(i8))

    goo()


'''
Desired Behavior of Facts
On Instantiation 
  bool = False
  int = 0
  float = 0.0
  str = ""
  Facts = None
  List = None
  Dicts = None
  Otherwise = None

Facts can be:
    Typed/Untyped
    Mutable/Immutable

On Set 
  If has been declared
    raise IsImmutableError()
  If None then nullify member values
    -If typed keep t_id
    -Otherwise t_id = 0
  If Wrong Value Type
    raise error
  Otherwise 
    Set value (incref new / decref old)


  
  
  
  

'''
