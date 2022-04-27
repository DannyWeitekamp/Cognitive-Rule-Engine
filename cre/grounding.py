import numpy as np
from numba import njit, generated_jit, types, literal_unroll, u8, i8, f8, u1
from numba.types import unicode_type,  intp, Tuple,  Tuple, DictType, ListType
from numba.typed import Dict, List
from numba.experimental.structref import new
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.cre_object import CREObjType
from cre.fact import define_fact, UntypedFact, call_untyped_fact, BaseFact
from cre.fact_intrinsics import fact_lower_getattr, resolve_fact_getattr_type
from cre.tuple_fact import TupleFact, TF
from cre.context import cre_context
from cre.default_ops import Add, Subtract, Divide
from cre.var import Var, GenericVarType
from cre.op import GenericOpType
from cre.utils import _func_from_address, _cast_structref, _obj_cast_codegen, _func_from_address, _incref_structref
from cre.structref import define_structref
from cre.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor
from cre.memory import Memory, MemoryType
from numba.extending import overload_method, overload, lower_cast, SentryLiteralArgs
from numba.experimental.function_type import _get_wrapper_address
import cloudpickle
# from collections import OrderedSet
# types.MakeFunctionLiteral



gval = define_fact("gval")


a_id_addr_pair_type = Tuple((u1,i8))
a_id_addr_pair_list_type = ListType(a_id_addr_pair_type)

flattener_fields = {
    **incr_processor_fields,    
    "out_mem" : MemoryType,
    "update_addrs" : ListType(a_id_addr_pair_list_type),
    "fact_visible_attr_pairs" : types.Any,
    "idrec_map" : DictType(u8,ListType(u8)),
    "inv_idrec_map" : DictType(u8,ListType(u8)),

}

GenericFlattener, GenericFlattenerType, FlattenerClass  = define_structref("Flattener", flattener_fields, return_template=True) 


# @lower_cast(GenericFlattenerType, GenericFlattenerType)
@lower_cast(GenericFlattenerType, IncrProcessorType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)

@generated_jit(cache=True)
@overload_method(FlattenerClass,'get_changes')
def self_get_changes(self, end=-1, exhaust_changes=True):
    def impl(self, end=-1, exhaust_changes=True):
        incr_pr = _cast_structref(IncrProcessorType, self)
        return incr_pr.get_changes(end=end, exhaust_changes=exhaust_changes)
    return impl





update_sig = types.void(GenericFlattenerType, BaseFact)
update_func_type = types.FunctionType(update_sig)


@generated_jit(cache=True, nopython=True)    
def flattener_update_for_attr(self, fact, id_attr, attr):#, context_name="cre_default"):
    print("flattener_update_for_attr:: ", fact, id_attr, attr)
    grounding_types = []
    # for fact_ref, attr in fact_visible_attr_pairs:
        # print(fact_ref, attr)
    # head_type = Tuple((GenericVarType,fact_type.instance_type))
    head_type = GenericVarType
    # head_type = fact_type.instance_type
    attr = attr.literal_value
    fact_type = fact

    # Construct the 'grounding_type' directly and get it's 'ctor'
    #  since for now we can't use gval as it's own constructor
    val_type = resolve_fact_getattr_type(fact_type, attr)        
    grounding_type = gval(head=head_type, val=val_type)#, flt_val=f8, nom_val=i8)
    gval_ctor = grounding_type._ctor[0]

    # Make a special constructor for a Var(fact_type, unique_id).attr
    # NOTE: should probably just wrap this into it's own thing in Var
    a_id = fact_type.get_attr_a_id(attr)
    offset = fact_type.get_attr_offset(attr)
    fact_num = fact_type._fact_num
    var_head_type_name = str(fact_type.spec[attr]['type'])
    from cre.var import new_appended_var_generic, DEREF_TYPE_ATTR
    @njit
    def var_w_attr_dref(v_name, attr):
        v = _cast_structref(GenericVarType, Var(fact_type, v_name))
        return new_appended_var_generic(v, attr, a_id, offset, var_head_type_name, fact_num, DEREF_TYPE_ATTR)

    def impl(self, fact, id_attr, attr):
        # for fact in self.in_mem.get_facts(fact_type):
        identifier = fact_lower_getattr(fact, id_attr)
        # print("identifier", identifier)
        v = fact_lower_getattr(fact, attr)
        # print("v", v)
        g = gval_ctor(head=var_w_attr_dref(identifier, attr),val=v)
        # print(g)
        # print(self.in_mem)
        # print(self.out_mem)
        self.out_mem.declare(g)
        # print(g)
    return impl


@generated_jit(cache=True, nopython=True)    
def flattener_update(self):#, context_name="cre_default"):
    # print()
    # print("flattener_update:: ")
    # print(self.__dict__)

    # print(self.hrepr())
    # for attr,t in self._fields:
    #     print()
    #     print(attr, t)
    # print("::")
    # print(self.fields)

    # grounding_types = []
    # for fact_ref, attr in fact_visible_attr_pairs:
    #     # print(fact_ref, attr)
    #     val_type = resolve_fact_getattr_type(fact_ref.instance_type, attr.literal_value)        
    #     grounding_type = gval(head=head_type, value=val_type)#, flt_val=f8, nom_val=i8)
    #     grounding_types.append(grounding_type)
    # grounding_types = tuple(grounding_types)
    # ctor = grounding_type._ctor[0]
    # print(fact_visible_attr_pairs)
    # if(not context): context
    def impl(self):#fact_visible_attr_pairs, in_mem, out_mem):
        # i = 0
        for change_event in self.get_changes():
            if(change_event.was_declared or change_event.was_modified):
                t_id = change_event.t_id
                update_addrs = self.update_addrs[t_id]

                # print(t_id, update_addrs)                
                for a_id, update_addr in update_addrs:

                    if(change_event.was_modified and a_id not in change_event.a_ids):
                        continue

                    if(update_addr != 0):
                        update = _func_from_address(update_func_type, update_addr)
                        update(self, self.in_mem.get_fact(change_event.idrec))



            if(change_event.was_retracted):
                pass


            # for tup in literal_unroll(self.fact_visible_attr_pairs):
            #     fact_type, attr = tup
            #     flattener_update_for_attr(self, fact_type, attr)
            # # print(fact_type, attr)
            # # print("--", fact_type, "--")
            # print("--", i, ":", attr, "--")
            # for fact in in_mem.get_facts(fact_type):
            #     v = fact_lower_getattr(fact, attr)
            #     g = gval(head=(Var(fact_type,fact.A),fact), val=v)
            #     print(g)
            # i += 1
            # print()

        # return st
    return impl

from numba.core.typing.typeof import typeof
@generated_jit(cache=True, nopython=True)    
def flattener_ctor(fact_types, in_mem, context_data, update_addrs, out_mem):
    fact_visible_attr_pairs = get_semantic_visibile_fact_attrs(fact_types)
    fact_visible_attr_pairs_type = typeof(fact_visible_attr_pairs)
    # print(fact_visible_attr_pairs_type)
    # fact_visible_attr_pairs_type = types.Tuple()
    fields = {**flattener_fields, "fact_visible_attr_pairs" : fact_visible_attr_pairs_type}
    fl_proxy, f_type, f_type_class  = define_structref("Flattener", fields, return_template=True) 

    @lower_cast(f_type, GenericFlattenerType)
    @lower_cast(f_type, IncrProcessorType)
    def upcast(context, builder, fromty, toty, val):
        return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)

    @generated_jit(cache=True)
    @overload_method(f_type_class,'get_changes')
    def self_get_changes(self, end=-1, exhaust_changes=True):
        def impl(self, end=-1, exhaust_changes=True):
            incr_pr = _cast_structref(IncrProcessorType, self)
            return incr_pr.get_changes(end=end, exhaust_changes=exhaust_changes)
        return impl

    # print("&&", fact_visible_attr_pairs)
    # print("<<", f_type.hrepr())

    def impl(fact_types, in_mem, context_data, update_addrs, out_mem):
        st = new(f_type)
        init_incr_processor(st, in_mem)
        st.fact_visible_attr_pairs = fact_visible_attr_pairs
        st.update_addrs = update_addrs


         # = update_addrs
        # st.in_mem = in_mem
        # print("??", out_mem)
        # Dunno why but this is necessary
        # _incref_structref(out_mem)
        st.out_mem = out_mem #if(out_mem is not None) else Memory(context_data)

        # print("??", st.out_mem)
        return st
        # st.fact_visible_attr_pairs = fact_visible_attr_pairs
    return impl


def gen_flattener_update_src(fact_type, id_attr, attr):
    if(isinstance(id_attr,types.Literal)): id_attr = id_attr.literal_value
    if(isinstance(attr,types.Literal)): attr = attr.literal_value
    return (
f'''from numba import njit, types
from cre.grounding import flattener_update_for_attr, update_sig
from cre.utils import _cast_structref
import cloudpickle

fact_type = cloudpickle.loads({cloudpickle.dumps(fact_type)})
id_attr = "{id_attr}"
attr = "{attr}"
@njit(update_sig,cache=True)
def flattener_update(self,fact):
    return flattener_update_for_attr(self, _cast_structref(fact_type,fact), id_attr, attr)
    ''')


from cre.vector import new_vector



@njit(cache=True)
def update_addrs_add(update_addrs, t_id, a_id, addr):
    update_addrs[t_id].append((u1(a_id), addr))


@njit(cache=True)
def new_update_addrs(n):
    update_addrs = List.empty_list(a_id_addr_pair_list_type)
    for i in range(n):
        update_addrs.append(List.empty_list(a_id_addr_pair_type))
    return update_addrs



def new_flattener(fact_types, in_mem, out_mem=None, context=None):
    context = cre_context(context)
    context_data = context.context_data    
    fact_visible_attr_pairs = get_semantic_visibile_fact_attrs(fact_types)

    id_attr = "A"
    #new_vector(len(context.type_registry)+1)
    # for in range()
    update_addrs = new_update_addrs(len(context.type_registry)+1)

    #np.zeros((len(context.type_registry)+1,), dtype=np.int64)
    for fact_type, attr in fact_visible_attr_pairs:
        if(isinstance(attr, types.Literal)): attr = attr.literal_value

        hash_code = unique_hash(['flattener_update',fact_type, id_attr, attr])
        if(not source_in_cache('flattener_update',hash_code)):
            source = gen_flattener_update_src(fact_type, id_attr, attr)
            source_to_cache('flattener_update', hash_code, source)
        update_func = import_from_cached('flattener_update', hash_code, ['flattener_update'])['flattener_update']

        t_id = context_data.fact_num_to_t_id[fact_type._fact_num]

        update_addr = _get_wrapper_address(update_func, update_sig)
        update_addrs_add(update_addrs, t_id, fact_type.get_attr_a_id(attr), update_addr)
        # print("children", context.children_of[fact_type._fact_name])
        for c_fact_type in context.children_of[fact_type._fact_name]:
            c_t_id = context_data.fact_num_to_t_id[c_fact_type._fact_num]
            update_addrs_add(update_addrs, c_t_id, c_fact_type.get_attr_a_id(attr), update_addr)

    # print(update_addrs)
        
        # update_addrs[t_id] = _get_wrapper_address(update_func, update_sig)

    if(out_mem is None): out_mem = Memory(context);

    flattener = flattener_ctor(fact_types, in_mem, context_data, update_addrs, out_mem)
    return flattener


def get_semantic_visibile_fact_attrs(fact_types):
    context = cre_context()

    sem_vis_fact_attrs = {}
    for ft in fact_types:
        ft = ft.instance_type if (isinstance(ft, types.TypeRef)) else ft
        parents = context.parents_of.get(ft._fact_name,[])
        # print()
        # print(ft.spec)
        for attr, d in ft.spec.items():
            if(d.get("is_semantic_visible", False)):
                is_new = True
                for p in parents:
                    if((p,attr) in sem_vis_fact_attrs):
                        is_new = False
                        break
                if(is_new): sem_vis_fact_attrs[(ft,attr)] = True

    return tuple([(fact_type,types.literal(attr)) for fact_type, attr in sem_vis_fact_attrs.keys()])
                # print([(ft._fact_name, attr) for ft, attr in visible_semantic_attrs])
    # print([(ft._fact_name, attr) for ft, attr in visible_semantic_attrs])
                # print(ft._fact_name, attr)

# if(False):

#     @njit(cache=True)
#     def flattener_update(self):
#         for change_event in self.get_changes():
#             # print("CHANGE",change_event)
#             comp = spp.mem.get_fact(change_event.idrec, ComponentType)
            # sources.append(comp)




    # @lower_cast(ShortestPathProcessorType, IncrProcessorType)
    # def upcast(context, builder, fromty, toty, val):
    #     return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)


    # define_structref()
    # raise ValueError()


enumerizer_fields = {
    **incr_processor_fields,
    # "ops" 
    "head_to_inds" : types.Any,
    "float_to_enum" : DictType(f8,i8),
    "string_to_enum" : DictType(f8,i8),
    "obj_to_enum" : DictType(CREObjType,i8),
    "num_noms" : i8
}

Enumerizer, EnumerizerType, EnumerizerTypeClass  = define_structref("Enumerizer", enumerizer_fields, return_template=True) 

# @njit(cache=True)    
# def flattener_ctor():
#     st = new(EnumerizerType)
#     return st


@generated_jit(cache=True, nopython=True)
@overload_method(EnumerizerTypeClass, "add_grounding")
def new_op_grounding(self, op, return_type, *args):
    '''Creates a new gval Fact for an op and a set of arguments'''

    # Fix annoying numba bug where StarArgs becomes a tuple() of cre.Tuples
    if(isinstance(args,tuple) and isinstance(args[0],types.BaseTuple)):
        args = args[0]
    
    # Because of numba issue: https://github.com/numba/numba/issues/7973
    #  inlining gval(head=???, value=??) w/ arguments doesn't work
    #  so just grab it's ctor and use that directly
    # head_type = TF(GenericOpType,TF(*args))
    head_type = Tuple((GenericOpType,args))
    # print(head_type)
    grounding_type = gval(head=head_type, value=return_type.instance_type, flt_val=f8, nom_val=i8)
    ctor = grounding_type._ctor[0]
    
    # Find the types for the op's check and call
    call_sig = return_type.instance_type(*args)
    check_sig = types.bool_(*args)
    call_f_type = types.FunctionType(call_sig)
    check_f_type = types.FunctionType(check_sig)

    def impl(self, op, return_type, *args):
        # head = TF(op, TF(*args))        
        head = (op, args)       
        if(op.check_addr != 0):
            check = _func_from_address(check_f_type, op.check_addr)
            check_ok = check(*args)
            if(not check(*args)): raise ValueError("gval attempt failed check().")

        call = _func_from_address(call_f_type, op.call_addr)
        value = call(*args)
        grounding = ctor(head=head, value=value)
        return grounding
    return impl


def ground_op(op,*args):
    return_type = op.signature.return_type
    print(op, return_type)
    return new_op_grounding(op, return_type, *args)

if __name__ == "__main__":
    import faulthandler; faulthandler.enable()
    g = ground_op(Add(Var(f8),Var(f8)), 1., 7.)
    print("<<g", g)
    g = ground_op(Add(Var(unicode_type), Var(unicode_type)),  "1", "7")
    print("<<g", g)
