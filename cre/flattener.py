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
from cre.gval import gval_of_type, new_gval, gval as gval_type

def get_semantic_visibile_fact_attrs(fact_types):
    ''' Takes in a set of fact types and returns all (fact, attribute) pairs
        for "is_semantic_visible" attributes. 
    '''
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

# Flattener Struct Definition
flattener_fields = {
    **incr_processor_fields,    
    "out_mem" : MemoryType,
    "idrec_map" : DictType(u8,ListType(u8)),
    "inv_idrec_map" : DictType(u8,ListType(u8)),
    "base_var_map" : DictType(Tuple((u2,unicode_type)), GenericVarType),
    "var_map" : DictType(Tuple((u2,unicode_type,unicode_type)), GenericVarType),
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


t_id_identifier_alias_tup_type = Tuple((u2,unicode_type,unicode_type))
t_id_alias_pair_type = Tuple((u2,unicode_type))
u8_list = ListType(u8)

def get_flattener_type(fact_types):
    fact_visible_attr_pairs = get_semantic_visibile_fact_attrs(fact_types)
    fact_visible_attr_pairs_type = typeof(fact_visible_attr_pairs)
    field_dict = {**flattener_fields, "fact_visible_attr_pairs" : fact_visible_attr_pairs_type}
    f_type = FlattenerTypeClass([(k,v) for k,v in field_dict.items()])
    f_type._fact_visible_attr_pairs = fact_visible_attr_pairs
    return f_type



from numba.core.typing.typeof import typeof
@generated_jit(cache=True, nopython=True)    
def flattener_ctor(flattener_type, in_mem, context_data, out_mem):
    fact_visible_attr_pairs = flattener_type.instance_type._fact_visible_attr_pairs
    def impl(flattener_type, in_mem, context_data, out_mem):
        st = new(flattener_type)
        init_incr_processor(st, in_mem)
        st.fact_visible_attr_pairs = fact_visible_attr_pairs
        st.base_var_map = Dict.empty(t_id_alias_pair_type, GenericVarType)
        st.var_map = Dict.empty(t_id_identifier_alias_tup_type, GenericVarType)
        st.idrec_map = Dict.empty(u8,u8_list)
        st.inv_idrec_map = Dict.empty(u8,u8_list)
        st.out_mem = out_mem 
        return st
    return impl


class Flattener(structref.StructRefProxy):
    def __new__(cls, fact_types, in_mem=None, out_mem=None, context=None):
        context = cre_context(context)
        context_data = context.context_data    
        if(in_mem is None): in_mem = Memory(context);
        if(out_mem is None): out_mem = Memory(context);
        
        flattener_type = get_flattener_type(fact_types)
        self = flattener_ctor(flattener_type, in_mem, context_data, out_mem)
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

def get_ground_type(head_type,val_type):
    return 

@generated_jit(cache=True, nopython=True)    
def flattener_update_for_attr(self, fact, id_attr, attr):
    '''Implements the flattening of a particular attribute of a fact'''

    # Make sure that 'id_attr','attr' are treated as literals (i.e. constants)
    #  that will force a unique implementation for this function.
    SentryLiteralArgs(['id_attr','attr']).for_function(flattener_update_for_attr).bind(self, fact, id_attr, attr) 

    # Iplementation Constants
    attr = attr.literal_value
    fact_type = fact
    t_id = fact_type.t_id
    base_var_type = get_var_type(fact_type)

    def impl(self, fact, id_attr, attr):
        # upfront cost of ~2.5 ms/10k to handle change_events.

        # Get the identifier ('id_attr') and value ('attr') from fact.
        identifier = fact_lower_getattr(fact, id_attr) # time negligible
        v = fact_lower_getattr(fact, attr) # time negligible

        ###### Start Var(...).attr #####
        # Get or make a base_var for 'identifier'.'attr'.  
        tup = (u2(t_id), identifier, attr)
        if(tup not in self.var_map):

            # Get or make a base_var for 'identifier'.
            btup = (u2(t_id), identifier)
            if(btup not in self.base_var_map):

                # Make the base_var and cache it.
                self.base_var_map[btup] = Var(fact_type, identifier)

            # Apply .attr to the base_var and cache it.
            base_var = _cast_structref(base_var_type, self.base_var_map[btup])
            var = var_append_deref(base_var,attr)
            self.var_map[tup] = _cast_structref(GenericVarType, var)

        ###### End Var(...).attr #####

        # Make the gval.
        g = new_gval(head=self.var_map[tup], val=v)
        idrec = self.out_mem.declare(g)

        # Map the fact's idrec in 'in_mem' to the gval's idrec in 'out_mem' 
        if(fact.idrec not in self.idrec_map):
            self.idrec_map[fact.idrec] = List.empty_list(u8)
        self.idrec_map[fact.idrec].append(idrec)

        # if(idrec not in self.inv_idrec_map):
        #     self.idrec_map[idrec] = List.empty_list(u8)
        # self.idrec_map[idrec].append(fact.idrec)

    return impl

# @njit(cache=True)
# def clear_modify():



@generated_jit(cache=True, nopython=True)    
def flattener_update(self):
    # One set of arguments typ:Type, t_ids:tuple(ints), attr:str, a_id:int
    #  for each semantic visible attribute 
    c = cre_context()
    impl_args = []
    for typ, attr in self._fact_visible_attr_pairs:
        args = (typ, tuple([*c.get_child_t_ids(_type=typ)]), typ.get_attr_a_id(attr.literal_value), attr)
        impl_args.append(args)
    impl_args = tuple(impl_args)
    print([tuple(str(x) for x in y) for y in impl_args])
    def impl(self):
        out = self.out_mem
        idrec = u8(0)
        # For each change event that occured since the last call to update() 
        for change_event in self.get_changes():
            # On RETRACT.
            if(change_event.was_retracted):
                if(change_event.idrec in self.idrec_map):
                    for idrec in self.idrec_map[change_event.idrec]:
                        out.retract(idrec)
                    del self.idrec_map[change_event.idrec]

            # On DECLARE or MODIFY.
            if(change_event.was_declared or change_event.was_modified):
                _t_id = change_event.t_id

                # Apply the implementation for each 'attr' as needed. 
                for args in literal_unroll(impl_args):
                    typ, t_ids, a_id, attr  = args

                    #Only apply for fact types that have the attr
                    if(change_event.t_id in t_ids):
                        # On MODIFY.
                        if(change_event.was_modified):
                            # If it's a modify event make sure the a_id matches this implementation
                            if(a_id in change_event.a_ids):
                                if(change_event.idrec in self.idrec_map):
                                    for idrec in self.idrec_map[change_event.idrec]:
                                        # Retract any old stuff 
                                        var = _cast_structref(GenericVarType, out.get_fact(idrec).asa(gval_type).head)
                                        if(var.deref_offsets[0].a_id == a_id):
                                            out.retract(idrec)
                                            break
                                    if(idrec != 0):
                                        self.idrec_map[change_event.idrec].remove(idrec)
                            else:
                                continue

                        # if(okay):
                        fact = _cast_structref(typ, self.in_mem.get_fact(change_event.idrec))
                        flattener_update_for_attr(self,fact,"A", attr)

    return impl



### THINKING THINKING THINKING ###
'''
Makeing Var's is a bit slow... a big part is probably needing to make
lists 
'''
