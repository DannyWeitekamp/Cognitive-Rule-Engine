from numba import njit, u1,u2, i4, i8, u8, types, literal_unroll, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.fact import base_fact_field_dict, BaseFact, FactProxy, Fact, add_to_type_registry
from cre.fact_intrinsics import fact_lower_setattr, _register_fact_structref
from cre.cre_object import cre_obj_get_item, CREObjType, CREObjProxy, CREObjTypeTemplate, member_info_type
from cre.utils import _struct_get_attr_offset, _sizeof_type, _struct_get_data_ptr, _load_ptr, _struct_get_attr_offset, _struct_from_ptr, _cast_structref, _obj_cast_codegen, encode_idrec, decode_idrec, _incref_structref, _get_member_offset
# from cre.primitive import Primitive
from cre.context import cre_context
from numba.core.imputils import (lower_cast, numba_typeref_ctor)
from numba.experimental.structref import new, define_boxing, define_attributes, StructRefProxy
from numba.core.extending import overload, intrinsic, lower_builtin, type_callable
from numba.core.typing.typeof import typeof
from cre.core import T_ID_TUPLE_FACT, register_global_default
import cloudpickle

##### NOTE NOTE NTOE TODO -- Need distribute "members" across in order to getitem

# tup_fact_field_dict = {
#     **base_fact_field_dict,
#     # "members": types.Any,
#     # "chr_mbrs_infos" : types.UniTuple(member_info_type,1),
# }

# tup_fact_fields = [(k,v) for k,v in tup_fact_field_dict.items()]

def _up_cast_helper(x):
    if(isinstance(x, CREObjTypeTemplate)):
        return CREObjType
    else:
        return x

def tf_field_dict_from_types(member_types):   
    member_types = member_types.types if isinstance(member_types, types.Tuple) else member_types
    member_types = tuple([x for x in member_types])
    member_info_tup_type = types.UniTuple(member_info_type,len(member_types))
    # member_t_ids = tuple([_resolve_t_id_helper(x) for x in member_types])
    member_field_dict = {f"a{i}":m for i,m in enumerate(member_types)} 
    # tf_d = {**tup_fact_field_dict,"members" : types.Tuple(member_types),"chr_mbrs_infos":member_info_tup_type}
    tf_d = {**base_fact_field_dict,**member_field_dict}#,**{"chr_mbrs_infos":member_info_tup_type}}
    # print("tf_d", tf_d)
    return tf_d

def tf_get_item(tf,typ, index):
    return cre_obj_get_item(tf.asa(CREObjType),typ,index)

from cre.fact import _gen_props, _gen_getter_jit
def gen_tuple_fact_source(member_types, TF_T_ID, specialization_name=None, ind='    '):    
    # attr_offsets = get_offsets_from_member_types(member_types)
    base_fields = [(k,v) for k,v in base_fact_field_dict.items()]
    getter_jits = "\n".join([_gen_getter_jit("TupleFact",t,attr) for attr,t in base_fields])
    properties = "\n".join([_gen_props("TupleFact",attr) for attr,t in base_fields])
    init_fields = f'\n{ind}'.join([f"fact_lower_setattr(st, 'a{i}', members[{i}])" for i in range(len(member_types))])
    # print("::", member_types)
# attr_offsets = np.array({attr_offsets!r},dtype=np.int16)    
# SpecializedTF({"types.StarArgTuple(member_types)" if len(member_types) > 0 else ""})
    return (
f'''from numba import njit, u1, u8, types
from numba.types import UniTuple
from numba.experimental.structref import new
from numba.extending import overload, lower_cast
from cre.fact import uint_to_inheritance_bytes
from cre.fact_intrinsics import define_boxing, get_fact_attr_ptr, _register_fact_structref, fact_mutability_protected_setattr, fact_lower_setattr, _fact_get_chr_mbrs_infos
from cre.tuple_fact import TupleFactClass, TupleFactProxy, tf_field_dict_from_types, tf_get_item, T_ID_TUPLE_FACT{", TupleFact" if specialization_name else ""}
from cre.utils import encode_idrec, _get_member_offset, _cast_structref
from cre.cre_object import member_info_type, set_chr_mbrs
import cloudpickle
TF_T_ID = T_ID_TUPLE_FACT#{TF_T_ID}
member_types = cloudpickle.loads({cloudpickle.dumps(member_types)})
n_members = len(member_types)
tf_fields = [(k,v) for k,v in tf_field_dict_from_types(member_types).items()]

inheritance_bytes = tuple(list(uint_to_inheritance_bytes(T_ID_TUPLE_FACT)) + [u1(0)] + list(uint_to_inheritance_bytes({TF_T_ID}))) 
num_inh_bytes = len(inheritance_bytes)

@_register_fact_structref
class SpecializedTFClass(TupleFactClass):
    t_id = TF_T_ID
    def __str__(self):
        return '{specialization_name if specialization_name else "TupleFact"}'

SpecializedTF = fact_type = SpecializedTFClass(tf_fields)
SpecializedTF_w_mbr_infos = SpecializedTFClass(tf_fields+
[("chr_mbrs_infos", UniTuple(member_info_type,{len(member_types)})),
 ("num_inh_bytes", u1),
 ("inh_bytes", UniTuple(u1, num_inh_bytes))])



SpecializedTF._fact_name = "TupleFact"
SpecializedTF.t_id = TF_T_ID
# SpecializedTF._attr_offsets = attr_offsets
SpecializedTF._fact_type_class = SpecializedTFClass
{f'SpecializedTF._specialization_name = "{specialization_name}"' if(specialization_name is not None) else ''}

{(f"""SpecializedTF.parent_type = TupleFact
@lower_cast(SpecializedTF, TupleFact)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)                        
""") if specialization_name is not None else ""
}

default_idrec  = encode_idrec(TF_T_ID, 0, 0xFF)
@njit(cache=True)
def ctor({"*members" if len(member_types) > 0 else ""}):
    st = new(SpecializedTF_w_mbr_infos)
    fact_lower_setattr(st, 'idrec', default_idrec)
    fact_lower_setattr(st, 'hash_val', 0)
    set_chr_mbrs(st, ({", ".join([f"'a{i}'" for i in range(len(member_types))])}))
    fact_lower_setattr(st,'num_inh_bytes', num_inh_bytes)
    fact_lower_setattr(st,'inh_bytes', inheritance_bytes)
    {init_fields}
    return _cast_structref(SpecializedTF,st)

SpecializedTFClass._ctor = (ctor,)

{getter_jits}

class SpecializedTFProxy(TupleFactProxy):
    __numba_ctor = ctor
    _fact_type = SpecializedTF
    _fact_type_class = SpecializedTFClass
    t_id = TF_T_ID

    def __repr__(self):
        return f"TupleFact({{', '.join([repr(tf_get_item(self,mt,i)) for i,mt in enumerate(member_types)])}})"

    def __str__(self):
        return f"TF({{', '.join([str(tf_get_item(self,mt,i)) for i,mt in enumerate(member_types)])}})"
{properties}

@overload(SpecializedTFProxy, prefer_literal=False)
def overload_tup_fact_ctor(*args):
    def impl(*args):
        return ctor(*args)
    return impl

SpecializedTF._fact_proxy = SpecializedTFProxy

define_boxing(SpecializedTFClass,SpecializedTFProxy)

''')

# f'TupleFact({', '.join([repr_tf_item(self,member_types[i],i) for i in range()])})'

def define_tuple_fact(member_types, context=None, return_proxy=False, return_type_class=False):   
    if(len(member_types) > 0):
        typ_assigments = ", ".join([str(t) for t in member_types])
        specialization_name = f"TupleFact({typ_assigments})"
    else:
        specialization_name = None

    TF_hash_code = unique_hash(["TupleFact",member_types])
    if(not source_in_cache("TupleFact",TF_hash_code)):
        # from cre.core import T_ID_TUPLE_FACT
        TF_T_ID = add_to_type_registry("TupleFact", TF_hash_code)

        # TupleFacts all have the same t_id
        source = gen_tuple_fact_source(member_types, TF_T_ID, specialization_name)
        source_to_cache("TupleFact", TF_hash_code, source)
    tf_type = import_from_cached("TupleFact", TF_hash_code, ["SpecializedTF"])["SpecializedTF"]

    if(specialization_name is not None):
        context = cre_context(context)
        context._register_fact_type(specialization_name, tf_type, inherit_from=TupleFact)


    out = [tf_type]
    if(return_proxy): out.append(tf_type._fact_proxy)
    if(return_type_class): out.append(tf_type._fact_type_class)
    return tuple(out) if len(out) > 1 else out[0]

@_register_fact_structref
class TupleFactClass(Fact):
    _tf_type_cache = {}
    def __new__(cls, *args):
        # If args is a bunch of types then generate the a definition for the
        #   TupleFact with members_types==args. e.g. tf_type = TupleFact(i8,i8)
        if(len(args) == 0 or isinstance(args[0],types.Type)):
            return define_tuple_fact(args)

        # If args contains one list then assume it is the field list 
        #    and return the corresponding specialization
        if(len(args)==1 and isinstance(args[0],(list,tuple))):
            fields = args[0]
            self = super().__new__(cls)
            super().__init__(self,fields)
            return self

        else:
            # Otherwise make a new TupleFact
            typ = define_tuple_fact(tuple(typeof(x) for x in args))
            inst = typ._ctor[0](*args)
            return inst

    def __init__(self, *args):
        pass

    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

    # def __str__(self):

    #     if(hasattr(self,"_specialization_name")):
    #         return self._specialization_name
    #     else:
    #         return "TupleFact"
    #     print(self._fact_name)
    #     raise NotImplemented()
    #     return f"cre.{self._fact_name}"#f"cre.TupleFact({",".join(self.field_dict.)})"#self.name#"cre.PredType"




# Ensure that there is some placeholder for TupleFact in the cache 
#  so that we can use the registry to define TF_FACT_NUM



# TF_hash_code = unique_hash(["TupleFact",tup_fact_fields])
# if(not source_in_cache("TupleFact",TF_hash_code)):
#     TF_FACT_NUM = add_to_fact_registry("TupleFact", TF_hash_code)
#     source_to_cache("TupleFact", TF_hash_code, gen_tuple_fact_source(TF_FACT_NUM))
# TF_FACT_NUM = import_from_cached("TupleFact", TF_hash_code, ["TF_FACT_NUM"])["TF_FACT_NUM"]
# TupleFact._fact_num = TF_FACT_NUM
# TupleFactClass._hash_code = TF_hash_code

# print("TupleFact", TF_FACT_NUM)
    
# define_attributes(TupleFact)


@lower_cast(TupleFactClass, BaseFact)
@lower_cast(TupleFactClass, CREObjType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)

# @njit(cache=True)
# def init_members():
#     return List.empty_list(CREObjType)

# @njit(types.void(ListType(CREObjType),i8),cache=True)
# def add_member_from_ptr(members, ptr):
#     members.append(_struct_from_ptr(CREObjType,ptr))

# @njit(CREObjType(i8),cache=True)
# def cre_object_from_ptr(ptr):
#     return _struct_from_ptr(CREObjType,ptr)

# Not sure why giving signatures produces wrong type
# @njit(PredType(CREObjType, ListType(CREObjType)),cache=False)


# print("<<", decode_idrec(default_idrec))





# from cre.cre_object import _resolve_t_id_helper
# def _resolve_t_id_helper(x):
#     if(isinstance(x, types.Boolean)):
#         return T_ID_BOOL
#     elif(isinstance(x, types.Integer)):
#         return T_ID_INT
#     elif(isinstance(x, types.Float)):
#         return T_ID_FLOAT
#     elif(x is types.unicode_type):
#         return T_ID_STR
#     return T_ID_UNRESOLVED
    

from numba.experimental.structref import _Utils, imputils
from numba.core import cgutils, utils as numba_utils

@intrinsic
def _tup_fact_get_chr_mbrs_infos(typingctx, tf_type):
    from cre.context import CREContext
    context = CREContext.get_default_context()

    members_type = [v for k,v in tf_type._fields if k == 'members'][0]
    t_ids = [context.get_t_id(_type=x) for x in members_type.types]

    count = members_type.count
    member_infos_out_type = types.UniTuple(member_info_type, count)

    def codegen(context, builder, sig, args):
        [tf,] = args

        utils = _Utils(context, builder, tf_type)

        baseptr = utils.get_data_pointer(tf)
        baseptr_val = builder.ptrtoint(baseptr, cgutils.intp_t)
        dataval = utils.get_data_struct(tf)
        index_of_members = dataval._datamodel.get_field_position("members")

        member_infos = []
        for i in range(count):
            member_ptr = builder.gep(baseptr, [cgutils.int32_t(0), cgutils.int32_t(index_of_members), cgutils.int32_t(i)], inbounds=True)
            member_ptr = builder.ptrtoint(member_ptr, cgutils.intp_t)
            offset = builder.trunc(builder.sub(member_ptr, baseptr_val), cgutils.ir.IntType(16))
            t_id = context.get_constant(u2, t_ids[i])
            member_infos.append(context.make_tuple(builder, member_info_type, (t_id, offset) ))

        ret = context.make_tuple(builder,member_infos_out_type, member_infos)
        return ret

    sig = member_infos_out_type(tf_type)
    return sig, codegen

# @njit
# def get_member_offsets(typingctx, pred, member_info):
#     _pred_get_member_offsets(pred)
    # for i,x in enumerate(literal_unroll(members)):
    #     I = literally(i)



# from cre.core import T_ID_TUPLE_FACT
# default_idrec  = encode_idrec(T_ID_TUPLE_FACT,0,0xFF)

# @generated_jit(cache=True, nopython=True)
# def tup_fact_ctor(*members):
#     # print(members)
#     member_types = members[0]
#     # print(member_types)
#     tf_type = tf_type_from_member_types(member_types)
#     # print("<<", tf_type)

#     def impl(*members):
#         st = new(tf_type)
#         # Force zero to avoid immutability issues
#         fact_lower_setattr(st, 'idrec', u8(-1))

#         # Fill in everything
#         fact_lower_setattr(st, 'hash_val', 0)
#         fact_lower_setattr(st, 'fact_num', TF_FACT_NUM)
#         fact_lower_setattr(st, 'num_chr_mbrs', len(members))
#         fact_lower_setattr(st, 'chr_mbrs_infos', _tup_fact_get_chr_mbrs_infos(st))
#         fact_lower_setattr(st, 'chr_mbrs_infos_offset', _get_member_offset(st,'chr_mbrs_infos'))
#         fact_lower_setattr(st, 'members', members)
#         # print(st.members)
#         # Set this last to avoid immutability
#         fact_lower_setattr(st, 'idrec', default_idrec)
#         # print("**",  _struct_get_attr_offset(st,"members"), _struct_get_attr_offset(st,"member_info") + st.length)
#         return st
#     return impl

# TupleFactClass._ctor = (tup_fact_ctor,)

class TupleFactProxy(FactProxy):
    def __new__(cls, *args):
        if(len(args) == 0 or isinstance(args[0],types.Type)):
            return define_tuple_fact(args)
        else:
            ctor = getattr(cls,"_ctor",define_tuple_fact(tuple(typeof(x) for x in args))._ctor)[0]
            return ctor(*args)

    def __init__(self,*args):
        pass
        # super().__init__(self)
        # print(repr(self))

    # def __str__(self):
    #     # print(self.__dict__, type(self).__dict__)
    #     return "TupleFact({})"

    def __repr__(self):
        return str(self)


define_boxing(TupleFactClass, TupleFactProxy)


TupleFact = TupleFactClass()
# print(TupleFact.__module__)
# TF_FACT_NUM = TupleFact._fact_num

register_global_default("TupleFact", TupleFact)

@generated_jit(cache=True)
def assert_cre_obj(x):
    if(isinstance(x, types.Literal)): return
    if(isinstance(x, CREObjTypeTemplate)):
        def impl(x):
            return _cast_structref(CREObjType, x)
        return impl
    else:
        def impl(x):
            prim = Primitive(x)
            return _cast_structref(CREObjType, prim)
        return impl

# USE_TREF_CTOR = False

# if(USE_TREF_CTOR):
#     # NOTE This is better than implementing overload() for TupleFactProxy
#     #   since the end user would be able to just use TupleFact as a constructor 
#     #   and type but there is a bug in numba with using *arg in the typer.
#     @type_callable(TupleFactClass)
#     def tf_type_callable(context):
#         def typer(*args):
#             typ = TupleFactClass(*args)
#             return typ
#         return typer

#     @overload(numba_typeref_ctor)
#     def tf_numba_typeref_ctor(self, *args):
#         # print("BEF", self)
#         if(not isinstance(self.instance_type, TupleFactClass)): return
#         # print("##", A, B)
        
#         def impl(self, *args):
#             return tup_fact_ctor(*args)
#         return impl

#     TF = TupleFact
# else:

@overload(TupleFactProxy, prefer_literal=False,)
def overload_tup_fact_ctor(*args):
    ctor = define_tuple_fact(args)._ctor[0]
    def impl(*args):
        return ctor(*args)
    return impl

TF = TupleFactProxy

# @lower_builtin(TupleFact, types.VarArg(types.Any))
# def apply_tuple_fact_ctor(context, builder, sig, args):
#     print("SHOULD BE APPLIED")
#     cls = sig.return_type

#     def call_ctor(cls, *args):
#         return tup_fact_ctor(*args)

#     # Pack arguments into a tuple for `*args`
#     ctor_args = types.Tuple.from_types(sig.args)
#     # Make signature T(TypeRef[T], *args) where T is cls
#     sig = typing.signature(cls, types.TypeRef(cls), ctor_args)
#     if len(ctor_args) > 0:
#         args = (context.get_dummy_value(),   # Type object has no runtime repr.
#                 context.make_tuple(builder, ctor_args, args))
#     else:
#         args = (context.get_dummy_value(),   # Type object has no runtime repr.
#                 context.make_tuple(builder, ctor_args, ()))

#     return context.compile_internal(builder, call_ctor, sig, args)


# Pred("HI", 1, 1)

# @njit
# def test_pred_item_offset():
#     print(_pred_get_member_offsets(Pred("HI", 1, 2, "HI", 3, 4)))

# test_pred_item_offset()

# raise ValueError()




# @njit(i8(GenericPredType), cache=True)
# def pred_get_length(x):
#     return x.length

# @njit(i8(GenericPredType), cache=True)
# def pred_get_member_info_ptr(x):
#     data_ptr = _struct_get_data_ptr(x.chr_mbrs_infos)
#     member_info_offset = _struct_get_attr_offset(x.chr_mbrs_infos,"data")
#     return _struct_get_data_ptr(x.chr_mbrs_infos) + member_info_offset

# @njit(i8(GenericPredType), cache=True)
# def pred_get_members_ptr(x):
#     l = i8(pred_get_length(x))
#     data_ptr = _struct_get_data_ptr(x)
#     # member_info_offset = _struct_get_attr_offset(x,"member_info")
#     return data_ptr + x.members_offset + l

# @njit(u8(GenericPredType, i8), cache=True)
# def pred_get_member_info(x, i):
#     member_info_ptr = pred_get_member_info_ptr(x)
#     return _load_ptr(member_info_type, member_info_ptr + i*3)

# @njit(u8(GenericPredType, i8), cache=True)
# def pred_get_item(x, typ, i):
#     member_info_ptr = pred_get_member_info_ptr(x)
#     return _load_ptr(u1, data_ptr + member_info_ptr + i)

# @njit(types.UniTuple(i8,2)(i8, i8), cache=True)
# def pred_get_next_ptrs(t_id_ptr, item_ptr):
#     t_id = _load_ptr(u1,t_id_ptr)
#     diff = _sizeof_type(i8) if t_id != T_ID_STR else _sizeof_type(unicode_type)
#     # print(t_id, diff)
#     return t_id_ptr+1, item_ptr+diff


    # print()


# pred_iter_t_id_item_ptrs()


# print(">>", isinstance(2,CREObjType))



# @intrinsic 
# def 
