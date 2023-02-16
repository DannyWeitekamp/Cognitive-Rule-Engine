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
# from cre.default_ops import Add, Subtract, Divide
from cre.var import Var, VarType, var_append_deref, VarTypeClass
# from cre.op import GenericOpType
from cre.utils import cast, _func_from_address,  _obj_cast_codegen, _func_from_address, _incref_structref
from cre.structref import define_structref
from cre.memset import MemSet, MemSetType
from cre.structref import CastFriendlyStructref, define_boxing
from cre.transform.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor
from cre.transform.enumerizer import Enumerizer, EnumerizerType
from numba.experimental import structref
from numba.extending import overload_method, overload, lower_cast, SentryLiteralArgs
from numba.experimental.function_type import _get_wrapper_address
import cloudpickle
from cre.gval import get_gval_type, new_gval, gval as gval_type
from cre.fact import DeferredFactRefType


@njit(cache=True)
def get_ptr(x):
    return cast(x, i8)

def get_visibile_fact_attrs(fact_types):
    ''' Takes in a set of fact types and returns all (fact, attribute) pairs
        for "visible" attributes. 
    '''
    context = cre_context()

    vis_fact_attrs = {}
    for ft in fact_types:
        ft = ft.instance_type if (isinstance(ft, (types.TypeRef,))) else ft
        parents = context.parents_of.get(ft._fact_name,[])
        for attr in ft.filter_spec("visible"):
            is_new = True
            for p in parents:
                if((p,attr) in vis_fact_attrs):
                    is_new = False
                    break
            if(is_new): vis_fact_attrs[(ft,attr)] = True

    return tuple([(fact_type,types.literal(attr)) for fact_type, attr in vis_fact_attrs.keys()])

# Flattener Struct Definition
flattener_fields = {
    **incr_processor_fields,    
    "out_memset" : MemSetType,
    "idrec_map" : DictType(u8,ListType(u8)),
    # "inv_idrec_map" : DictType(u8,ListType(u8)),
    "base_var_map" : DictType(Tuple((u2,unicode_type)), VarType),
    "var_map" : DictType(Tuple((u2,unicode_type,unicode_type)), VarType),
    "enumerizer" : EnumerizerType,
    "fact_visible_attr_pairs" : types.Any,
    "id_attr" : types.Any,
    
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
        incr_pr = cast(self, IncrProcessorType)
        return incr_pr.get_changes(end=end, exhaust_changes=exhaust_changes)
    return impl


t_id_identifier_alias_tup_type = Tuple((u2,unicode_type,unicode_type))
t_id_alias_pair_type = Tuple((u2,unicode_type))
u8_list = ListType(u8)

def get_flattener_type(fact_types,id_attr):
    fact_visible_attr_pairs = get_visibile_fact_attrs(fact_types)
    fact_visible_attr_pairs_type = typeof(fact_visible_attr_pairs)
    field_dict = {**flattener_fields,
                 "fact_visible_attr_pairs" : fact_visible_attr_pairs_type,
                 "id_attr" : types.literal(id_attr)
                 }
    f_type = FlattenerTypeClass([(k,v) for k,v in field_dict.items()])
    # print("<<", fact_visible_attr_pairs)
    f_type._fact_visible_attr_pairs = fact_visible_attr_pairs
    f_type._id_attr = id_attr
    return f_type



from numba.core.typing.typeof import typeof
@generated_jit(cache=True, nopython=True)    
def flattener_ctor(flattener_type, in_memset, out_memset, enumerizer=None):
    fact_visible_attr_pairs = flattener_type.instance_type._fact_visible_attr_pairs
    def impl(flattener_type, in_memset, out_memset, enumerizer=None):
        st = new(flattener_type)
        init_incr_processor(st, in_memset)
        st.fact_visible_attr_pairs = fact_visible_attr_pairs
        st.base_var_map = Dict.empty(t_id_alias_pair_type, VarType)
        st.var_map = Dict.empty(t_id_identifier_alias_tup_type, VarType)
        st.idrec_map = Dict.empty(u8,u8_list)
        # st.inv_idrec_map = Dict.empty(u8,u8_list)
        st.out_memset = out_memset 
        st.enumerizer = enumerizer
        return st
    return impl


class Flattener(structref.StructRefProxy):
    def __new__(cls, fact_types, in_memset=None, id_attr="id",
                 out_memset=None, enumerizer=None, context=None):
        context = cre_context(context)
        fact_types = tuple(fact_types)

        # Make new in_memset and out_memset if they are not provided.
        if(in_memset is None): in_memset = MemSet(context);
        if(out_memset is None): out_memset = MemSet(context);

        # Inject an enumerizer instance into the context object
        #  if one does not exist. So it is shared across various
        #  processing steps.
        if(enumerizer is None):
            enumerizer = getattr(context,'enumerizer', Enumerizer()) 
            context.enumerizer = enumerizer

        # Make a flattener_type from fact_types + id_attr and instantiate
        #  the flattener. Keep a reference to out_memset around.
        flattener_type = get_flattener_type(fact_types, id_attr)
        self = flattener_ctor(flattener_type, in_memset, out_memset, enumerizer)
        self._out_memset = out_memset
        return self

    @property
    def in_memset(self):
        return get_in_memset(self)

    @property
    def out_memset(self):
        return self._out_memset

    def transform(self, in_memset=None):
        if(in_memset is not None):
            if(not check_same_in_memset(self, in_memset)):
                set_in_memset(self, in_memset)
                flattener_clear(self)
                self._out_memset = MemSet()
                set_out_memset(self, self._out_memset)
        self.update()
        return self._out_memset

    def __call__(self, in_memset=None):
        return self.transform(in_memset)


    def update(self):
        flattener_update(self)

define_boxing(FlattenerTypeClass, Flattener)

@njit(types.void(GenericFlattenerType),cache=True)
def flattener_clear(self):
    self.idrec_map = Dict.empty(u8,u8_list)
    self.change_queue_head = 0
    # self.out_memset = MemSet(self.out_memset.context_data)

@njit(types.void(GenericFlattenerType,MemSetType),cache=True)
def set_in_memset(self, in_memset):
    self.in_memset = in_memset

@njit(types.void(GenericFlattenerType,MemSetType),cache=True)
def set_out_memset(self, out_memset):
    self.out_memset = out_memset

@njit(MemSetType(GenericFlattenerType),cache=True)
def get_in_memset(self):
    return self.in_memset

@njit(types.boolean(GenericFlattenerType, MemSetType),cache=True)
def check_same_in_memset(self,in_memset):
    return cast(self.in_memset, i8) == cast(in_memset, i8)







@generated_jit(cache=True, nopython=True)    
def flattener_update_for_attr(self, fact, id_attr, attr):
    '''Implements the flattening of a particular attribute of a fact'''

    # Make sure that 'id_attr','attr' are treated as literals (i.e. constants)
    #  that will force a unique implementation for this function.
    SentryLiteralArgs(['id_attr','attr']).for_function(flattener_update_for_attr).bind(self, fact, id_attr, attr) 

    # Implement Constants
    attr = attr.literal_value
    fact_type = fact
    t_id = fact_type.t_id
    base_var_type = VarTypeClass(fact_type)


    def impl(self, fact, id_attr, attr):
        # NOTE: upfront cost of ~2.5 ms/10k to handle change_events.

        # Get the identifier 'id_attr' and value 'attr' from fact.
        identifier = fact_lower_getattr(fact, id_attr) # time negligible
        v = fact_lower_getattr(fact, attr) # time negligible

        ###### Start Var(...).attr #####
        # Get or make a base_var for 'identifier'.'attr'.  
        tup = (u2(t_id), identifier, attr)
        if(tup not in self.var_map): # 2 ms / 10k whole clause

            # Get or make a base_var for 'identifier'.
            btup = (u2(t_id), identifier)
            if(btup not in self.base_var_map):

                # Make the base_var and cache it.
                self.base_var_map[btup] = Var(fact_type, identifier)

            # Apply .attr to the base_var and cache it.
            base_var = cast(self.base_var_map[btup], base_var_type)
            var = var_append_deref(base_var,attr)
            self.var_map[tup] = cast(var, VarType)

        ###### End Var(...).attr #####

        nom = self.enumerizer.enumerize(v) #2.3 ms / 10k 

        # Make the gval.
        g = new_gval(head=self.var_map[tup], val=v, nom=nom) #2.6ms / 10k 
        idrec = self.out_memset.declare(g) #3.1ms / 10k 

        # Map the fact's idrec in 'in_memset' to the gval's idrec in 'out_memset' 
        if(fact.idrec not in self.idrec_map):
            self.idrec_map[fact.idrec] = List.empty_list(u8)
        self.idrec_map[fact.idrec].append(idrec) #3 ms / 10k 

    return impl

@njit(cache=True)
def clean_a_id(self, change_idrec, a_id):
    '''Cleans the gvals associated with the fact at 'change_idrec' with 'a_id' ''' 
    if(change_idrec in self.idrec_map):
        for idrec in self.idrec_map[change_idrec]:
            gval = self.out_memset.get_fact(idrec).asa(gval_type)
            var = cast(gval.head, VarType)
            if(var.deref_infos[0].a_id == a_id):
                self.out_memset.retract(idrec)
                self.idrec_map[change_idrec].remove(idrec)
                break
            
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

    id_attr = self._id_attr
    # print([tuple(str(x) for x in y) for y in impl_args])
    def impl(self):
        # For each change event that occured since the last call to update() 
        for change_event in self.get_changes():
            # On RETRACT: Remove downstream gvals and clean out of idrec_map
            if(change_event.was_retracted):
                if(change_event.idrec in self.idrec_map):
                    for idrec in self.idrec_map[change_event.idrec]:
                        self.out_memset.retract(idrec)
                    del self.idrec_map[change_event.idrec]

            # On DECLARE or MODIFY.
            if(change_event.was_declared or change_event.was_modified):
                # Apply the implementation for each 'attr' as needed. 
                for args in literal_unroll(impl_args):
                    typ, child_t_ids, a_id, attr  = args

                    #Only apply for fact types that have 'attr'
                    if(change_event.t_id in child_t_ids):
                        if(change_event.was_modified):
                            if(a_id in change_event.a_ids):
                                # Clean out old gval
                                clean_a_id(self, change_event.idrec, a_id)
                            else:
                                # Skip if MODIFY a_id doesn't match this implementation
                                continue

                        fact = cast(self.in_memset.get_fact(change_event.idrec), typ)
                        flattener_update_for_attr(self,fact, id_attr, attr)

    return impl

