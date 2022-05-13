import numpy as np
from numba import njit, generated_jit, types, literal_unroll, u8, i8, f8, u1, u2
from numba.types import unicode_type,  intp, Tuple,  Tuple, DictType, ListType
from numba.typed import Dict, List
from numba.experimental.structref import new
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.cre_object import CREObjType
from cre.fact import define_fact, UntypedFact, call_untyped_fact, BaseFact
from cre.fact_intrinsics import fact_lower_getattr, resolve_fact_getattr_type
from cre.context import cre_context
from cre.tuple_fact import TupleFact, TF
from cre.default_ops import Add, Subtract, Divide
from cre.var import Var, GenericVarType
from cre.op import GenericOpType
from cre.utils import PrintElapse, _func_from_address, _cast_structref, _obj_cast_codegen, _func_from_address, _incref_structref, _struct_from_ptr, decode_idrec, _ptr_to_data_ptr, _load_ptr, _struct_tuple_from_pointer_arr
from cre.structref import define_structref
from numba.experimental import structref
from cre.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor
from cre.memory import Memory, MemoryType
from cre.structref import CastFriendlyStructref, define_boxing
from numba.extending import overload_method, overload, lower_cast, SentryLiteralArgs
from numba.experimental.function_type import _get_wrapper_address
import cloudpickle
from cre.gval import get_gval_type, new_gval, gval as gval_type
from cre.vector import VectorType
from cre.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor, ChangeEventType
from itertools import chain

enumerizer_fields = {
    **incr_processor_fields,
    # "ops" 
    "head_to_inds" : types.Any,
    # "float_to_enum" : DictType(f8,i8),
    # "string_to_enum" : DictType(f8,i8),
    # "obj_to_enum" : DictType(CREObjType,i8),
    "num_noms" : i8
}

Enumerizer, EnumerizerType, EnumerizerTypeClass  = define_structref("Enumerizer", enumerizer_fields, return_type_class=True) 

print(ListType(u2[::1]))

# Flattener Struct Definition
feature_applier_fields = {
    **incr_processor_fields,    
    "out_mem" : MemoryType,
    "idrec_map" : DictType(u8,ListType(u8)),
    "inv_idrec_map" : DictType(u8,ListType(u8)),
    "ops" : ListType(GenericOpType),
    "fact_vecs" : ListType(ListType(VectorType)),
    "gval_t_ids" : ListType(u2[::1]),
    "current_change_events" : ListType(ChangeEventType),
    "fact_vecs_needs_update" : types.bool_
    # "unq_arg_types":  types.Any,
    # "unq_return_types":  types.Any,

    # "base_var_map" : DictType(Tuple((u2,unicode_type)), GenericVarType),
    # "var_map" : DictType(Tuple((u2,unicode_type,unicode_type)), GenericVarType),
    # "fact_visible_attr_pairs" : types.Any,
}

@structref.register
class FeatureApplierTypeClass(CastFriendlyStructref):
    pass

GenericFeatureApplierType = FeatureApplierTypeClass([(k,v) for k,v in feature_applier_fields.items()])

@lower_cast(FeatureApplierTypeClass, GenericFeatureApplierType)
@lower_cast(FeatureApplierTypeClass, IncrProcessorType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)

@generated_jit(cache=True)
@overload_method(FeatureApplierTypeClass,'get_changes')
def feature_applier_get_changes(self, end=-1, exhaust_changes=True):
    def impl(self, end=-1, exhaust_changes=True):
        incr_pr = _cast_structref(IncrProcessorType, self)
        return incr_pr.get_changes(end=end, exhaust_changes=exhaust_changes)
    return impl

u2_arr = u2[::1]
u8_list = ListType(u8)
vec_list_type = ListType(VectorType)
list_vec_list_type = ListType(ListType(VectorType))
op_list_type = ListType(GenericOpType)

# def get_feature_applier_type(ops):
    # unq_return_types = Tuple(tuple(set([op.signature.return_type for op in ops])))
    # unq_arg_types = Tuple(tuple(set(chain(*[op.signature.args for op in ops]))))
    # print(unq_return_types, unq_arg_types)
    # field_dict = {**feature_applier_fields, "unq_arg_types" : unq_arg_types,  "unq_return_types" : unq_return_types}
    # f_type = FeatureApplierTypeClass([(k,v) for k,v in feature_applier_fields.items()])
    # f_type._unq_return_types = unq_return_types
    # f_type._unq_arg_types = unq_arg_types
    # return f_type


@njit(cache=True)
def get_op_fact_vecs(self, op_ind):
    op = self.ops[op_ind]
    gval_t_ids = self.gval_t_ids[op_ind]
    fact_vecs = List.empty_list(VectorType)
    for i,v in enumerate(op.base_vars):
        # gval_t_id = self.val_t_id_to_gval_t_id[u2(v.base_t_id)]
        ptr = self.in_mem.mem_data.facts[gval_t_ids[i]]
        # print(gval_t_id, ptr)
        # Skip if any of the argument vectors are uninitialized
        if(ptr == 0): 
            self.fact_vecs_needs_update = True
            return fact_vecs
        fact_vec = _struct_from_ptr(VectorType,ptr)
        fact_vecs.append(fact_vec)
    return fact_vecs

@njit(cache=True)
def update_fact_vecs(self):
    for op_ind in range(len(self.ops)):
        self.fact_vecs[op_ind] = get_op_fact_vecs(self,op_ind)

@njit(cache=True)
def add_op(self, op, gval_t_ids):
    self.ops.append(op)
    self.gval_t_ids.append(gval_t_ids)
    self.fact_vecs.append(get_op_fact_vecs(self,len(self.ops)-1))

@njit(cache=True)
def assign_current_change_events(self):
    self.current_change_events = self.get_changes()

from numba.core.typing.typeof import typeof
    # def impl(in_mem, out_mem):
        
    # return impl

@njit(cache=True)
def get_fact_vecs_needs_update(self):
    return self.fact_vecs_needs_update


class FeatureApplier(structref.StructRefProxy):
    def __new__(cls, ops, in_mem=None, out_mem=None, context=None):
        context = cre_context(context)
        if(in_mem is None): in_mem = Memory(context);
        if(out_mem is None): out_mem = Memory(context);

        # feature_applier_type = get_feature_applier_type(ops)
        self = feature_applier_ctor(in_mem, out_mem)
        # raise ValueError()
        self.ops = []
        self.update_impls = []
        for op in ops:
            self.add_op(op)
        self._out_mem = out_mem
        return self

    @property
    def in_mem(self):
        return get_in_mem(self)

    @property
    def out_mem(self):
        return self._out_mem

    @property
    def fact_vecs_needs_update(self):
        return get_fact_vecs_needs_update(self)

    def get_changes(self):
        return feature_applier_get_changes(self)

    def apply(self, in_mem=None):
        if(in_mem is not None):
            set_in_mem(self, in_mem)
        self.update()
        return self._out_mem

    def update(self):
        feature_applier_update(self)

    def add_op(self,op):
        self.ops.append(op)
        gval_t_ids = np.array([context.get_t_id(_type=get_gval_type(x)) for x in op.signature.args],dtype=np.uint16)
        add_op(self, op, gval_t_ids)
        ret_type = op.signature.return_type
        arg_types = op.signature.args
        
        @njit(types.void(GenericFeatureApplierType, i8),cache=True)
        def update_impl(self,op_ind):
            update_for_op_ind(self,ret_type, arg_types, op_ind)

        self.update_impls.append(update_impl)


define_boxing(FeatureApplierTypeClass, FeatureApplier)

@njit(GenericFeatureApplierType(MemoryType,MemoryType),cache=True)    
def feature_applier_ctor(in_mem, out_mem):    
    st = new(GenericFeatureApplierType)
    init_incr_processor(st, in_mem)
    st.ops = List.empty_list(GenericOpType)
    st.fact_vecs = List.empty_list(vec_list_type)
    st.fact_vecs_needs_update = False
    st.idrec_map = Dict.empty(u8, u8_list)
    st.inv_idrec_map = Dict.empty(u8, u8_list)
    st.out_mem = out_mem 
    st.gval_t_ids = List.empty_list(u2_arr)
    st.current_change_events = List.empty_list(ChangeEventType)

    return st


# @njit(types.void(GenericFeatureApplierType, MemoryType),cache=True)
@njit(cache=True)
def set_in_mem(self, x):
    self.in_mem = x
    update_fact_vecs(self)


##### Update Functions #####

@njit(cache=True)
def product_iter_w_const(lengths, const_position, const_ind):
    ''' Yields the iteration product from zero to 'lengths' 
        (e.g. lengths=(2,2) -> [[0,0],[0,1],[1,0],[1,1]])
        but holds 'const_position' constant with value 'const_ind'
        (e.g. const_position=0, const_ind=1, lengths=(2,2) -> [[1,0],[1,1]])
    ''' 
    inds = np.zeros((len(lengths),),dtype=np.int64)

    inds[const_position] = const_ind
    lengths[const_position] = const_ind

    max_k = len(lengths)-1
    if(const_position == max_k): max_k -= 1
    k = max_k
    done = False
    while(not done):
        yield inds
        # Gets next set of indices holding const_position to const_ind
        inds[k] += 1
        while(inds[k] >= lengths[k]):
            inds[k] = 0
            k -= 1
            if(k == const_position): k -= 1;    
            if(k < 0): done = True; break;
            inds[k] += 1
        k = max_k



@generated_jit(cache=True)
def iter_arg_gval_ptrs(self, op_ind, arg_gval_t_ids, t_id, f_id):
    def impl(self, op_ind, arg_gval_t_ids, t_id, f_id):
        op = self.ops[op_ind]

        # Extract the fact_vecs and their lengths
        fact_vecs = self.fact_vecs[op_ind]
        lengths = np.array([len(x) for x in fact_vecs],dtype=np.int64)
        gval_ptrs = np.empty((len(lengths),),dtype=np.int64)

        for const_pos, var in enumerate(op.base_vars):
            # Skip op args that don't match the change_event's t_id
            # if(self.val_t_id_to_gval_t_id[u2(var.base_t_id)] != t_id): continue
            if(arg_gval_t_ids[const_pos] != t_id): continue

            for inds in product_iter_w_const(lengths, const_pos, f_id):
                # Extract the pointers to gvals check if okay (non-zero)
                okay = True
                for i, ind in enumerate(inds):
                    ptr = fact_vecs[i].data[ind]
                    if(ptr == 0): okay = False; break;
                    gval_ptrs[i] = ptr

                # Ensure that they are unique
                if(okay and len(np.unique(gval_ptrs)) == len(gval_ptrs)):
                    yield gval_ptrs
    return impl

    
head_offset = gval_type.get_attr_offset('head')
val_offset = gval_type.get_attr_offset('val')
    
@generated_jit(nopython=True)
def update_for_op_ind(self, return_type, arg_types, op_ind):
    '''Updates the FeatureApplier for the op at 'op_ind'. Goes through
        if one of op's arguments is associated with t_id, '''
    # print(self, return_type, arg_types, op_ind, change_events)
    context = cre_context()
    return_type = return_type.instance_type
    arg_types = tuple([x.instance_type for x in arg_types])
    n_args = len(arg_types)
        
    # return_type = op.signature.return_type
    call_head_ptrs_func_type = types.FunctionType(return_type(i8[::1]))
    gval_type = get_gval_type(return_type)
    arg_gval_t_ids = tuple([context.get_t_id(_type=get_gval_type(x)) for x in arg_types])

    range_n_args = tuple([x for x in range(n_args)])
    n_var_type = tuple([GenericVarType for x in range(n_args)])


    def impl(self, return_type, arg_types, op_ind):
        # return
        op = self.ops[op_ind]
        for change_event in self.current_change_events:
            if(change_event.was_retracted):
                pass

            if(change_event.was_declared or change_event.was_modified):
                pass

            t_id, f_id, _ = decode_idrec(change_event.idrec)

            call_head_ptrs = _func_from_address(call_head_ptrs_func_type, op.call_head_ptrs_addr)
            val_ptrs = np.empty((n_args,),dtype=np.int64)
            var_ptrs = np.empty((n_args,),dtype=np.int64)

            # Get unique sets of pointers to the gval instances in 'in_mem' such that the 
            #  gval for this event is held fixed and the others are taken in every permutation.
            for gval_ptrs in iter_arg_gval_ptrs(self, op_ind, arg_gval_t_ids, t_id, f_id):

                # For each gval get the address for 'head' and val' slot
                for i in literal_unroll(range_n_args):
                    var_ptr = _ptr_to_data_ptr(gval_ptrs[i])
                    var_ptrs[i] = _ptr_to_data_ptr(gval_ptrs[i])+head_offset
                    val_ptrs[i] = _ptr_to_data_ptr(gval_ptrs[i])+val_offset

                # Make a gval with head=TupleFact(op,*vars) and val=call(*vals)
                tf = TF(op,*_struct_tuple_from_pointer_arr(n_var_type, var_ptrs))
                v = call_head_ptrs(val_ptrs)
                gval = new_gval(head=tf,val=v)
                self.out_mem.declare(gval)
                
    return impl

##### End #####


def feature_applier_update(self):
    ''' The main FeatureApplier update function. Update one op at a time
        to simplify typing
    '''
    if(self.fact_vecs_needs_update): update_fact_vecs(self)
    assign_current_change_events(self)
    for op_ind, (op, update_impl) in enumerate(zip(self.ops, self.update_impls)):
        # ret_type = op.signature.return_type
        # arg_types = op.signature.args
        # update_impl(self, ret_type, arg_types, op_ind)
        update_impl(self, op_ind)


            


if(__name__ == "__main__"):
    from cre.flattener import Flattener
    from cre.default_ops import Equals
    with cre_context("test_flatten") as context:
        spec1 = {"A" : {"type" : "string", "is_semantic_visible" : True}, 
                 "B" : {"type" : "number", "is_semantic_visible" : False}}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1,
                 "C" : {"type" : "number", "is_semantic_visible" : True}}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2,
                 "D" : {"type" : "number", "is_semantic_visible" : True}}
        BOOP3 = define_fact("BOOP3", spec3)

        eq_f8 = Equals(f8, f8)
        eq_str = Equals(unicode_type, unicode_type)
        fa = FeatureApplier([eq_f8,eq_str],Memory())
        fa.apply()


        mem = Memory()
        a = BOOP1("A", 1)
        b = BOOP1("B", 2)
        c = BOOP2("C", 3, 13)
        d = BOOP2("D", 4, 13)
        e = BOOP3("E", 5, 14, 106)
        f = BOOP3("A", 6, 16, 106)

        a_idrec = mem.declare(a)
        b_idrec = mem.declare(b)
        c_idrec = mem.declare(c)
        d_idrec = mem.declare(d)
        e_idrec = mem.declare(e)
        f_idrec = mem.declare(f)
        print("-------")

        fl = Flattener((BOOP1, BOOP2, BOOP3), mem)
        
        flat_mem = fl.apply()

        

        fa = FeatureApplier([eq_f8,eq_str],flat_mem)
        with PrintElapse("POOP"):
            feat_mem = fa.apply()
        with PrintElapse("POOP"):
            feat_mem = fa.apply()
        print(feat_mem)


# @njit(cache=True)
# def bar(arg_ind, f_id, lens):
#     for x in foo(arg_ind, f_id, lens):
#         print(x)
# # foo(1, 2, np.array((10,10,10)))
# bar(0, 5, np.array((10,10,10)))



# @generated_jit(cache=True)
# def update_gval_multi(return_type, arg_types, call_addr, check_addr, arg_ind, in_mem, out_mem):
#     nargs = len(arg_types)
#     gval_argtypes = tuple([get_gval_type(t) for t in arg_types])


#     # def impl():
#     #     # literal_unroll

#     in_mem.get_facts(arg_types)




    # for i, arg in enumerate(args):
    #     src += f'{c_ind}for i{i} in range(stride[{i}][0],stride[{i}][1]):\n'

# @njit(cache=True)
# def list_of_n(n):
#     l = List.empty_list(i8)
#     for i in range(n):
#         l.append(i)
#     return l

# @njit(cache=True)
# def _apply_whatever(*iters):
#     print(iters)


# @njit(cache=True)
# def apply_whatever(ranges):
#     # n_var = len(ranges)
#     # inds = ([0]*len(ranges))
#     # k = n_var-1
#     tup = ()
#     i = 0
#     for x in literal_unroll(ranges):

#         i += 1
#     print(tup)

# apply_whatever((1,2,3))








# if __name__ == "__main__":
#     import faulthandler; faulthandler.enable()
#     g = ground_op(Add(Var(f8),Var(f8)), 1., 7.)
#     print("<<g", g)
#     g = ground_op(Add(Var(unicode_type), Var(unicode_type)),  "1", "7")
#     print("<<g", g)
