from numba import njit, u1,u2, i4, i8, u8, types, literal_unroll, generated_jit
from numba.typed import List
from numba.types import ListType, unicode_type
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.fact import base_fact_field_dict, BaseFact, FactProxy, Fact, lines_in_type_registry, add_to_type_registry, add_type_pickle
from cre.fact_intrinsics import fact_lower_setattr, _register_fact_structref
from cre.cre_object import cre_obj_get_item, CREObjType, CREObjProxy, CREObjTypeClass, member_info_type
from cre.utils import _struct_get_attr_offset, _sizeof_type, _struct_get_data_ptr, _load_ptr, _struct_get_attr_offset, _obj_cast_codegen, encode_idrec, decode_idrec, _incref_structref, _get_member_offset
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
    if(isinstance(x, CREObjTypeClass)):
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

from cre.fact import _gen_props, _gen_getter_jit, _gen_setter_jit
def gen_tuple_fact_source(member_types, TF_T_ID, specialization_name=None, ind='    '):    
    # attr_offsets = get_offsets_from_member_types(member_types)
    base_fields = [(k,v) for k,v in base_fact_field_dict.items()]
    getter_jits = "\n".join([_gen_getter_jit("SpecializedTF",t,attr) for attr,t in base_fields])
    setter_jits = "\n".join([_gen_setter_jit("SpecializedTF",attr,a_id) for a_id, (attr,t) in enumerate(base_fields)])
    properties = "\n".join([_gen_props(attr) for attr,t in base_fields])
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
from cre.utils import cast, encode_idrec, _get_member_offset
from cre.cre_object import member_info_type, set_chr_mbrs
import cloudpickle
TF_T_ID = T_ID_TUPLE_FACT#{TF_T_ID}
member_types = cloudpickle.loads({cloudpickle.dumps(member_types)})
n_members = len(member_types)
field_list = [(k,v) for k,v in tf_field_dict_from_types(member_types).items()]

inheritance_bytes = tuple(list(uint_to_inheritance_bytes(T_ID_TUPLE_FACT)) + [u1(0)] + list(uint_to_inheritance_bytes({TF_T_ID}))) 
num_inh_bytes = len(inheritance_bytes)

@_register_fact_structref
class SpecializedTFClass(TupleFactClass):
    t_id = TF_T_ID
    def __str__(self):
        return '{specialization_name if specialization_name else "TupleFact"}'

    __repr__ = __str__

SpecializedTF = fact_type = SpecializedTFClass(field_list)
SpecializedTF_w_mbr_infos = SpecializedTFClass(field_list+
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
    return cast(st, SpecializedTF)

SpecializedTFClass._ctor = (ctor,)

{getter_jits}

{setter_jits}

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
SpecializedTF._proxy_class = SpecializedTFProxy

define_boxing(SpecializedTFClass,SpecializedTFProxy)

''')

# f'TupleFact({', '.join([repr_tf_item(self,member_types[i],i) for i in range()])})'

def define_tuple_fact(member_types, context=None, return_proxy=False, return_type_class=False):   
    from cre.cre_func import CREFuncTypeClass, CREFuncType
    # print("::", member_types)
    
    member_types = list(member_types)
    for i, x in enumerate(member_types):
        # Don't specialize CREFunc types 
        if(isinstance(x, CREFuncTypeClass)):
            member_types[i] = CREFuncType
    # print("::>", member_types)


    if(len(member_types) > 0):
        typ_assigments = ", ".join([str(t) for t in member_types])
        specialization_name = f"TupleFact({typ_assigments})"
    else:
        specialization_name = None

    TF_hash_code = unique_hash(["TupleFact",member_types])
    if(not source_in_cache("TupleFact",TF_hash_code)):
        # Possible for other types to be defined while running the Fact source
        #  so preregister the t_id then add the pickle later.
        TF_T_ID = add_to_type_registry("TupleFact", TF_hash_code)

        # TupleFacts all have the same t_id
        source = gen_tuple_fact_source(member_types, TF_T_ID, specialization_name)
        source_to_cache("TupleFact", TF_hash_code, source)
        tf_type = import_from_cached("TupleFact", TF_hash_code, ["SpecializedTF"])["SpecializedTF"]
        add_type_pickle(tf_type, TF_T_ID)
        
    else:
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
            super().__init__(self,'TupleFact',fields)
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



@lower_cast(TupleFactClass, BaseFact)
@lower_cast(TupleFactClass, CREObjType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)

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


class TupleFactProxy(FactProxy):
    def __new__(cls, *args):
        if(len(args) == 0 or isinstance(args[0],types.Type)):
            return define_tuple_fact(args)
        else:
            ctor = getattr(cls,"_ctor",define_tuple_fact(tuple(typeof(x) for x in args))._ctor)[0]
            return ctor(*args)

    def __init__(self,*args):
        pass

    def __repr__(self):
        return str(self)


define_boxing(TupleFactClass, TupleFactProxy)

TupleFact = TupleFactClass()

register_global_default("TupleFact", TupleFact)

@overload(TupleFactProxy, prefer_literal=False,)
def overload_tup_fact_ctor(*args):
    ctor = define_tuple_fact(args)._ctor[0]
    def impl(*args):
        return ctor(*args)
    return impl

TF = TupleFactProxy

