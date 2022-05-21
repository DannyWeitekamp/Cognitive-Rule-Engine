from numba import types, f8, i8, generated_jit
from cre.cre_object import CREObjType
from cre.fact import define_fact
from cre.core import short_name
from cre.context import cre_context


# Define the base_type 'gval' for grounded values
gval_spec = {"head" : CREObjType, "val" : types.Any}
gval = define_fact("gval", gval_spec)

_gval_types = {}
def get_gval_type(val_type, context=None):
    context = cre_context(context)
    tup = (context.name,val_type)
    if(tup not in _gval_types):
        spec  = {"inherit_from" : {"type": gval, "unified_attrs": ["head", "val"]},
                **gval_spec, **{"val" : val_type}}
        _gval_types[tup] =  define_fact(f"gval_{short_name(val_type)}", spec)
    return _gval_types[tup]


@generated_jit(cache=True)
def new_gval(head, val):
    typ = get_gval_type(val)
    ctor = typ._ctor[0]
    def impl(head, val):
        return ctor(head=head, val=val)
    return impl
