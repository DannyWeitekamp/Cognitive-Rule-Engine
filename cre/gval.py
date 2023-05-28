import numpy as np
from numba import types, f8, i8, u8, boolean, generated_jit, objmode
from numba.extending import overload, overload_method
from numba.typed import List
from numba.types import ListType, unicode_type
from cre.obj import CREObjType, cre_obj_get_item, cre_obj_get_member_t_ids
from cre.fact import define_fact
from cre.core import short_name, T_ID_TUPLE_FACT, T_ID_FUNC, T_ID_VAR
from cre.context import cre_context
from cre.utils import cast, decode_idrec
from cre.var import get_deref_attrs_str, VarType
# from cre.op import CREFuncType
from cre.func import CREFuncType


# Define the base_type 'gval' for grounded values
gval_spec = {
    "head" : CREObjType,
    "flt" : f8,
    "nom" : u8,
    "val" : types.Any,
}
gval, GvalTypeClass = define_fact("gval", gval_spec, return_type_class=True)

_gval_types = {}
def get_gval_type(val_type, context=None):
    if(isinstance(val_type, types.Literal)):
        val_type = val_type.literal_type
    context = cre_context(context)
    tup = (context.name,val_type)
    if(tup not in _gval_types):
        spec  = {"inherit_from" : {"type": gval, "unified_attrs": ["head", "val"]},
                **gval_spec, **{"val" : val_type}}
        _type, proxy = define_fact(f"gval_{short_name(val_type)}", spec, return_proxy=True)
        _type._val_type = val_type
        _gval_types[tup] =  _type
        proxy.__str__ = lambda x : gval_str(x)
    return _gval_types[tup]

# Just prebuild these to avoid weird print ordering
get_gval_type(boolean)
get_gval_type(i8)
get_gval_type(f8)
get_gval_type(types.unicode_type)


@generated_jit(cache=True)
def new_gval(head, val, flt=None, nom=0):
    typ = get_gval_type(val)
    ctor = typ._ctor[0]
    def impl(head, val, flt=None, nom=0):
        if(flt is None): flt = np.NaN
        return ctor(head=head, flt=flt, nom=nom, val=val)
    return impl


@generated_jit(cache=True)
def _val_to_str(val):
    if(val is unicode_type):
        def impl(val):
            return f"'{val}'"
    else:
        def impl(val):
            return str(val)
    return impl

# @njit(u2(CREObjType),cache=True)
# def cre_obj_get_t_id(obj):
#     t_id, _, _ = decode_idrec(obj.idrec)
#     return t_id


@generated_jit(cache=True,nopython=True)
@overload_method(GvalTypeClass, "__str__")
@overload(str)
def gval_str(gval):
    if('gval' not in getattr(gval, '_fact_name','')): return
    def impl(gval):
        head = gval.head

        head_t_id, _, _ = decode_idrec(head.idrec)
        # print("head.idrec", head_t_id,x0,x1,T_ID_TUPLE_FACT)
        if(head_t_id == T_ID_TUPLE_FACT):
            t_ids = cre_obj_get_member_t_ids(head)
            if(t_ids[0] == T_ID_FUNC and np.all(t_ids[:1]==T_ID_VAR)):
                op = cre_obj_get_item(head, CREFuncType, 0)
                v_strs = List.empty_list(unicode_type)
                for i in range(1,head.num_chr_mbrs):
                    v = cre_obj_get_item(head, VarType, i)
                    s = get_deref_attrs_str(v)
                    v_strs.append(v.alias + get_deref_attrs_str(v))
                head_str = op.origin_data.expr_template.format(v_strs)
            else:
                with objmode(head_str=unicode_type):
                    head_str = str(head)

                # Get rid of the TF
                head_str = head_str[2:]
        else:
            v = cast(head, VarType)
            head_str = v.alias + get_deref_attrs_str(v)

        return f"{head_str} == {_val_to_str(gval.val)}" 
    return impl


# # BOOP = define_fact("BOOP", {"A":i8,"B":i8})
# # print(new_gval(BOOP(1,2),1))
# from cre.tuple_fact import TupleFact

# @generated_jit(cache=True)
# def gval_str(gval):
#     gval_type = gval
#     # val_type = gval_type._val_type
#     def impl(gval):
#         head = gval.head
#         if(head.isa(TupleFact))
#         if(gval)
#     return impl

# gval_str()
