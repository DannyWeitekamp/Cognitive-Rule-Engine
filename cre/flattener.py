import numpy as np
from numba import njit, generated_jit, types, literal_unroll, u8, i8, f8, u1, u2
from numba.types import unicode_type,  intp, Tuple,  Tuple, DictType, ListType
from numba.typed import Dict, List
from numba.experimental.structref import new
from cre.core import short_name
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.cre_object import CREObjType
from cre.fact import define_fact, UntypedFact, call_untyped_fact, BaseFact
from cre.fact_intrinsics import fact_lower_getattr, resolve_fact_getattr_type
from cre.tuple_fact import TupleFact, TF
from cre.context import cre_context
from cre.default_ops import Add, Subtract, Divide
from cre.var import Var, GenericVarType, var_append_deref, get_var_type
from cre.op import GenericOpType
from cre.utils import _func_from_address, _cast_structref, _obj_cast_codegen, _func_from_address, _incref_structref
from cre.structref import define_structref
from cre.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor
from cre.memory import Memory, MemoryType
from cre.structref import CastFriendlyStructref, define_boxing
from numba.experimental import structref
from numba.extending import overload_method, overload, lower_cast, SentryLiteralArgs
from numba.experimental.function_type import _get_wrapper_address
import cloudpickle
from cre.gval import gval_of_type, new_gval

# gval = define_fact('gval')

def get_semantic_visibile_fact_attrs(fact_types):
    ''' Takes in a set of fact types and returns all fact attribute pairs'''
    context = cre_context()

    sem_vis_fact_attrs = {}
    for ft in fact_types:
        ft = ft.instance_type if (isinstance(ft, types.TypeRef)) else ft
        parents = context.parents_of.get(ft._fact_name,[])
        for attr, d in ft.spec.items():
            if(d.get("is_semantic_visible", False)):
                is_new = True
                for p in parents:
                    if((p,attr) in sem_vis_fact_attrs):
                        is_new = False
                        break
                if(is_new): sem_vis_fact_attrs[(ft,attr)] = True

    return tuple([(fact_type,types.literal(attr)) for fact_type, attr in sem_vis_fact_attrs.keys()])

# Flattener Definition
a_id_addr_pair_type = Tuple((u1,i8))
a_id_addr_pair_list_type = ListType(a_id_addr_pair_type)

flattener_fields = {
    **incr_processor_fields,    
    "out_mem" : MemoryType,
    "update_addrs" : ListType(a_id_addr_pair_list_type),
    "idrec_map" : DictType(u8,ListType(u8)),
    "inv_idrec_map" : DictType(u8,ListType(u8)),
    "base_var_map" : DictType(Tuple((u2,unicode_type)), GenericVarType),
    "var_map" : DictType(Tuple((unicode_type,unicode_type)), GenericVarType),
    "fact_visible_attr_pairs" : types.Any,
}

@structref.register
class FlattenerTypeClass(CastFriendlyStructref):
    pass

GenericFlattenerType = FlattenerTypeClass([(k,v) for k,v in flattener_fields.items()])

@lower_cast(FlattenerTypeClass, GenericFlattenerType)
@lower_cast(FlattenerTypeClass, IncrProcessorType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)

@generated_jit(cache=True)
@overload_method(FlattenerTypeClass,'get_changes')
def self_get_changes(self, end=-1, exhaust_changes=True):
    def impl(self, end=-1, exhaust_changes=True):
        incr_pr = _cast_structref(IncrProcessorType, self)
        return incr_pr.get_changes(end=end, exhaust_changes=exhaust_changes)
    return impl

@njit(cache=True)
def update_addrs_add(update_addrs, t_id, a_id, addr):
    for i in range(len(update_addrs),t_id+1):
        update_addrs.append(List.empty_list(a_id_addr_pair_type))
    update_addrs[t_id].append((u1(a_id), addr))


@njit(cache=True)
def new_update_addrs(n):
    update_addrs = List.empty_list(a_id_addr_pair_list_type)
    for i in range(n):
        update_addrs.append(List.empty_list(a_id_addr_pair_type))
    return update_addrs

unicode_pair_type = Tuple((unicode_type,unicode_type))
t_id_alias_pair_type = Tuple((u2,unicode_type))

from numba.core.typing.typeof import typeof
@generated_jit(cache=True, nopython=True)    
def flattener_ctor(fact_types, in_mem, context_data, update_addrs, out_mem):
    fact_visible_attr_pairs = get_semantic_visibile_fact_attrs(fact_types)
    fact_visible_attr_pairs_type = typeof(fact_visible_attr_pairs)
    fields = {**flattener_fields, "fact_visible_attr_pairs" : fact_visible_attr_pairs_type}
    f_type = FlattenerTypeClass([(k,v) for k,v in fields.items()])

    def impl(fact_types, in_mem, context_data, update_addrs, out_mem):
        st = new(f_type)
        init_incr_processor(st, in_mem)
        st.fact_visible_attr_pairs = fact_visible_attr_pairs
        st.update_addrs = update_addrs
        st.base_var_map = Dict.empty(t_id_alias_pair_type, GenericVarType)
        st.var_map = Dict.empty(unicode_pair_type, GenericVarType)
        st.out_mem = out_mem 
        return st
    return impl


def gen_flattener_update_src(fact_type, id_attr, attr):
    if(isinstance(id_attr,types.Literal)): id_attr = id_attr.literal_value
    if(isinstance(attr,types.Literal)): attr = attr.literal_value
    return (
f'''from numba import njit, types
from cre.flattener import flattener_update_for_attr, update_sig
from cre.utils import _cast_structref
import cloudpickle
# from cre.gval import gval

fact_type = cloudpickle.loads({cloudpickle.dumps(fact_type)})
id_attr = "{id_attr}"
attr = "{attr}"
@njit(update_sig,cache=True)
def flattener_update_fact(self,fact):
    return flattener_update_for_attr(self, _cast_structref(fact_type,fact), id_attr, attr)
    ''')

class Flattener(structref.StructRefProxy):
    def __new__(cls, fact_types, in_mem=None, out_mem=None, context=None):
        context = cre_context(context)
        context_data = context.context_data    
        fact_visible_attr_pairs = get_semantic_visibile_fact_attrs(fact_types)
        id_attr = "A"
        update_addrs = new_update_addrs(len(context.name_to_type)+1)

        for fact_type, attr in fact_visible_attr_pairs:
            if(isinstance(attr, types.Literal)): attr = attr.literal_value

            hash_code = unique_hash(['flattener_update',fact_type, id_attr, attr])
            if(not source_in_cache('flattener_update',hash_code)):
                source = gen_flattener_update_src(fact_type, id_attr, attr)
                source_to_cache('flattener_update', hash_code, source)
            update_func = import_from_cached('flattener_update', hash_code, ['flattener_update_fact'])['flattener_update_fact']

            t_id = context.get_t_id(_type=fact_type)

            update_addr = _get_wrapper_address(update_func, update_sig)
            update_addrs_add(update_addrs, t_id, fact_type.get_attr_a_id(attr), update_addr)
            for c_t_id in context.get_child_t_ids(t_id=t_id):
                c_fact_type = context.get_type(t_id=c_t_id)
                if(t_id != c_t_id):
                    update_addrs_add(update_addrs, c_t_id, c_fact_type.get_attr_a_id(attr), update_addr)

        if(in_mem is None): in_mem = Memory(context);
        if(out_mem is None): out_mem = Memory(context);
        
        self = flattener_ctor(fact_types, in_mem, context_data, update_addrs, out_mem)
        self._out_mem = out_mem
        return self

    @property
    def in_mem(self):
        return get_in_mem(self)

    @property
    def out_mem(self):
        return self._out_mem

    def apply(self, in_mem=None):
        if(in_mem is not None):
            set_in_mem(self, in_mem)
        self.update()
        return self._out_mem

    def update(self):
        flattener_update(self)

define_boxing(FlattenerTypeClass, Flattener)

@njit(types.void(GenericFlattenerType,MemoryType),cache=True)
def set_in_mem(self, x):
    self.in_mem = x

@njit(MemoryType(GenericFlattenerType),cache=True)
def get_in_mem(self):
    return self.in_mem

# @njit(MemoryType(GenericFlattenerType),cache=True)
# def get_out_mem(self):
#     return self.out_mem


# # @lower_cast(GenericFlattenerType, GenericFlattenerType)
# @lower_cast(GenericFlattenerType, IncrProcessorType)
# def upcast(context, builder, fromty, toty, val):
#     return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)

# @generated_jit(cache=True)
# @overload_method(FlattenerClass,'get_changes')
# def self_get_changes(self, end=-1, exhaust_changes=True):
#     def impl(self, end=-1, exhaust_changes=True):
#         incr_pr = _cast_structref(IncrProcessorType, self)
#         return incr_pr.get_changes(end=end, exhaust_changes=exhaust_changes)
#     return impl





update_sig = types.void(GenericFlattenerType, BaseFact)
update_func_type = types.FunctionType(update_sig)


def get_ground_type(head_type,val_type):
    return 
# from cre.gval import gval
@generated_jit(cache=True, nopython=True)    
def flattener_update_for_attr(self, fact, id_attr, attr):#, context_name="cre_default"):    
    print("flattener_update_for_attr:: ", fact, id_attr, attr)
    attr = attr.literal_value
    fact_type = fact
    t_id = fact_type.t_id
    base_var_type = get_var_type(fact_type)

    def impl(self, fact, id_attr, attr):
        # upfront cost of ~2.5 ms to handle change_events
        identifier = fact_lower_getattr(fact, id_attr) # negligible
        v = fact_lower_getattr(fact, attr) # negligible

        # Get or make a base_var for 'identifier'.'attr'.  
        tup = (identifier,attr)
        if(tup not in self.var_map):

            # Get or make a base_var for 'identifier' 
            btup = (t_id,identifier)
            if(btup not in self.base_var_map):

                # Make the base_var and cache it
                self.base_var_map[btup] = Var(fact_type, identifier)

            # Apply .attr to the base_var and cache it
            base_var = _cast_structref(base_var_type, self.base_var_map[btup])
            var = var_append_deref(base_var,attr)
            self.var_map[tup] = _cast_structref(GenericVarType, var)

        g = new_gval(head=self.var_map[tup], val=v)
        self.out_mem.declare(g)
    return impl


@generated_jit(cache=True, nopython=True)    
def flattener_update(self):
    def impl(self):
        for change_event in self.get_changes():
            if(change_event.was_declared or change_event.was_modified):
                t_id = change_event.t_id


                
                update_addrs = self.update_addrs[t_id]
                # print("change_event t_id : ", t_id, update_addrs)
                for a_id, update_addr in update_addrs:
                    if(change_event.was_modified and a_id not in change_event.a_ids):
                        continue

                    if(update_addr != 0):
                        update = _func_from_address(update_func_type, update_addr)
                        fact = self.in_mem.get_fact(change_event.idrec)
                        update(self, self.in_mem.get_fact(change_event.idrec))

            if(change_event.was_retracted):
                pass
    return impl










# def new_flattener(fact_types, in_mem, out_mem=None, context=None):
#     context = cre_context(context)
#     context_data = context.context_data    
#     fact_visible_attr_pairs = get_semantic_visibile_fact_attrs(fact_types)
#     id_attr = "A"
#     update_addrs = new_update_addrs(len(context.name_to_type)+1)

#     for fact_type, attr in fact_visible_attr_pairs:
#         if(isinstance(attr, types.Literal)): attr = attr.literal_value

#         hash_code = unique_hash(['flattener_update',fact_type, id_attr, attr])
#         if(not source_in_cache('flattener_update',hash_code)):
#             source = gen_flattener_update_src(fact_type, id_attr, attr)
#             source_to_cache('flattener_update', hash_code, source)
#         update_func = import_from_cached('flattener_update', hash_code, ['flattener_update_fact'])['flattener_update_fact']

#         t_id = context_data.fact_num_to_t_id[fact_type._fact_num]

#         update_addr = _get_wrapper_address(update_func, update_sig)
#         update_addrs_add(update_addrs, t_id, fact_type.get_attr_a_id(attr), update_addr)
#         for c_fact_type in context.children_of[fact_type._fact_name]:
#             c_t_id = context_data.fact_num_to_t_id[c_fact_type._fact_num]
#             update_addrs_add(update_addrs, c_t_id, c_fact_type.get_attr_a_id(attr), update_addr)


#     if(out_mem is None): out_mem = Memory(context);

#     flattener = flattener_ctor(fact_types, in_mem, context_data, update_addrs, out_mem)
#     return flattener




# class Flattener():
#     def __init__(self):

### THINKING THINKING THINKING ###
'''
Makeing Var's is a bit slow... a big part is probably needing to make
lists 
'''
