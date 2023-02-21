import numpy as np
from numba import njit, generated_jit, types, literal_unroll, u8, i8, f8, u1, u2
from numba.types import unicode_type,  intp, Tuple,  Tuple, DictType, ListType
from numba.typed import Dict, List
from numba.experimental.structref import new
from cre.cre_object import copy_cre_obj, CREObjType
from cre.fact import define_fact, UntypedFact, call_untyped_fact, BaseFact
from cre.fact_intrinsics import fact_lower_getattr, resolve_fact_getattr_type
from cre.context import cre_context
from cre.tuple_fact import TupleFact, TF
# from cre.builtin_cre_funcs import Add, Subtract, Divide
from cre.var import Var, VarType
# from cre.op import CREFuncType
from cre.cre_func import CREFuncType, get_best_call_self, get_return_val_impl, set_base_arg_val_impl, CFSTATUS_TRUTHY
from cre.utils import cast, PrintElapse,encode_idrec, _func_from_address, _obj_cast_codegen, _func_from_address, _incref_structref, decode_idrec, _ptr_to_data_ptr, _load_ptr, _struct_tuple_from_pointer_arr, _incref_ptr
from cre.structref import define_structref
from numba.experimental import structref
from cre.memset import MemSet, MemSetType
from cre.structref import CastFriendlyStructref, define_boxing
from numba.extending import overload_method, overload, lower_cast, SentryLiteralArgs
from numba.experimental.function_type import _get_wrapper_address
import cloudpickle
from cre.gval import get_gval_type, new_gval, gval as gval_type
from cre.vector import VectorType, new_vector
from cre.transform.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor, ChangeEventType
from cre.transform.enumerizer import Enumerizer, EnumerizerType
from itertools import chain

# Flattener Struct Definition
feature_applier_fields = {
    **incr_processor_fields,    
    "out_memset" : MemSetType,
    "idrec_map" : DictType(u8,ListType(u8)),
    "ops" : ListType(CREFuncType),

    "fact_vecs" : DictType(u2, Tuple((VectorType,VectorType, VectorType))),

    "gval_t_ids" : ListType(u2[::1]),
    # "current_change_events" : ListType(ChangeEventType),
    "enumerizer" : EnumerizerType,
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
        incr_pr = cast(self, IncrProcessorType)
        return incr_pr.get_changes(end=end, exhaust_changes=exhaust_changes)
    return impl

u2_arr = u2[::1]
u8_list = ListType(u8)
vec_list_type = ListType(VectorType)
list_vec_list_type = ListType(ListType(VectorType))
op_list_type = ListType(CREFuncType)


class FeatureApplier(structref.StructRefProxy):
    def __new__(cls, ops, in_memset=None, out_memset=None,
                 enumerizer=None, context=None):
        context = cre_context(context)
        if(in_memset is None): in_memset = MemSet(context);
        if(out_memset is None): out_memset = MemSet(context);

        # Inject an enumerizer instance into the context object
        #  if one does not exist. So it is shared across various
        #  processing steps.
        if(enumerizer is None):
            enumerizer = getattr(context,'enumerizer', Enumerizer()) 
            context.enumerizer = enumerizer

        self = feature_applier_ctor(in_memset, out_memset, enumerizer)
        self.ops = []
        self.update_impls = []
        for op in ops:
            self.add_op(op)
        self._out_memset = out_memset
        return self

    @property
    def in_memset(self):
        return get_in_memset(self)

    @property
    def out_memset(self):
        return self._out_memset

    def get_changes(self):
        return feature_applier_get_changes(self)

    def transform(self, in_memset=None):
        if(in_memset is not None):
            if(not check_same_in_memset(self, in_memset)):
                set_in_memset(self, in_memset)
                feature_applier_clear(self)
                self._out_memset = MemSet()
                set_out_memset(self, self._out_memset)
        self.update()
        return self._out_memset

    def __call__(self, in_memset=None):
        return self.transform(in_memset)


    def update(self):
        ''' The main FeatureApplier update function. Update one op at a time
            to simplify typing
        '''
        update_fact_vecs(self)
        for op_ind, (op, update_impl) in enumerate(zip(self.ops, self.update_impls)):
            update_impl(self, op_ind)
        assign_new_f_ids(self)

    def add_op(self,op):
        context = cre_context()
        print("add op:", op, type(op))
        self.ops.append(op)
        gval_t_ids = np.array([context.get_t_id(_type=get_gval_type(x)) for x in op.signature.args],dtype=np.uint16)
        add_op(self, op, gval_t_ids)
        ret_type = op.signature.return_type
        arg_types = op.signature.args
        
        @njit(types.void(GenericFeatureApplierType, i8),cache=True)
        def update_impl(self,op_ind):
            update_op(self, ret_type, arg_types, op_ind)

        self.update_impls.append(update_impl)


define_boxing(FeatureApplierTypeClass, FeatureApplier)

vec_triple_type = Tuple((VectorType,VectorType,VectorType))

@njit(GenericFeatureApplierType(MemSetType,MemSetType,EnumerizerType),cache=True)    
def feature_applier_ctor(in_memset, out_memset, enumerizer):    
    st = new(GenericFeatureApplierType)
    init_incr_processor(st, in_memset)
    st.ops = List.empty_list(CREFuncType)
    st.fact_vecs = Dict.empty(u2, vec_triple_type)
    st.idrec_map = Dict.empty(u8, u8_list)
    st.out_memset = out_memset 
    st.gval_t_ids = List.empty_list(u2_arr)
    # st.current_change_events = List.empty_list(ChangeEventType)
    st.enumerizer = enumerizer

    return st


@njit(types.void(GenericFeatureApplierType, CREFuncType, u2[::1]), cache=True)
def add_op(self, op, gval_t_ids):
    self.ops.append(op)
    self.gval_t_ids.append(gval_t_ids)
    for t_id in gval_t_ids:
        if(t_id not in self.fact_vecs):
            self.fact_vecs[u2(t_id)] = (new_vector(1),new_vector(1), new_vector(1))







##### Update Functions #####

# @njit(cache=True)
# def range_product(lengths):
#     ''' Yields the iteration product from zero to 'lengths' 
#         (e.g. lengths=(2,2) -> [[0,0],[0,1],[1,0],[1,1]])
#     ''' 
#     if(not np.all(lengths)): return
#     inds = np.zeros((len(lengths),),dtype=np.int64)
#     max_k = k = len(lengths)-1
#     done = False
#     while(not done):
#         yield inds
#         # Gets next set of indices holding const_position to const_ind
#         inds[k] += 1
#         while(inds[k] >= lengths[k]):
#             inds[k] = 0
#             k -= 1
#             if(k < 0): done = True; break;
#             inds[k] += 1
#         k = max_k

# NOTE: Generator based implementation is probably way more efficient 
#  but causes memleak see numba issue 6993
@njit(cache=True)
def range_product(lengths):
    ''' Yields the iteration product from zero to 'lengths' 
        (e.g. lengths=(2,2) -> [[0,0],[0,1],[1,0],[1,1]])
    ''' 
    if(not np.all(lengths)): 
        return np.zeros((0, len(lengths),),dtype=np.int64)

    n_inds = int(np.prod(lengths))

    inds = np.zeros((len(lengths),),dtype=np.int64)
    out = np.zeros((n_inds, len(lengths),),dtype=np.int64)
    max_k = k = len(lengths)-1
    done = False
    c = 0
    while(not done):
        out[c,:] = inds[:]
        # yield inds
        # Gets next set of indices holding const_position to const_ind
        inds[k] += 1
        while(inds[k] >= lengths[k]):
            inds[k] = 0
            k -= 1
            if(k < 0): done = True; break;
            inds[k] += 1
        k = max_k
        c += 1
    return out


@njit(cache=True)
def arr_is_unique(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if(i != j and arr[i]==arr[j]):
                return False
    return True
    
head_offset = gval_type.get_attr_offset('head')
val_offset = gval_type.get_attr_offset('val')


@njit(cache=True)
def update_fact_vecs(self):
    for change_event in self.get_changes():
        t_id, f_id = change_event.t_id, change_event.f_id

        # Remove any facts in out_memset derived from the retracted fact.
        if(change_event.was_modified or change_event.was_retracted):
            if(change_event.idrec in self.idrec_map):
                for idrec in self.idrec_map[change_event.idrec]:
                    self.out_memset.retract(idrec)
                if(change_event.was_retracted):
                    del self.idrec_map[change_event.idrec]        
                else:
                    self.idrec_map[change_event.idrec] = ListType.empty_list(u8)

        # Copy any of the contents of in_memset to out_memset
        if(change_event.was_declared or change_event.was_modified):
            fact = self.in_memset.get_fact(change_event.idrec)
            fact_copy = copy_cre_obj(fact)
            copy_idrec = self.out_memset.declare(fact_copy)
            if(change_event.idrec not in self.idrec_map):
                self.idrec_map[change_event.idrec] = ListType.empty_list(u8)
            self.idrec_map[change_event.idrec].append(copy_idrec)

        if(t_id not in self.fact_vecs): continue
        new_f_ids, new_fact_ptrs, old_fact_ptrs = self.fact_vecs[u2(t_id)]

        if(change_event.was_retracted):
            old_fact_ptrs[f_id] = 0

        if(change_event.was_declared or change_event.was_modified):
            # Prepare new_f_ids and new_fact_ptrs for update_op.
            in_memset_facts = cast(self.in_memset.facts[i8(t_id)], VectorType)
            new_f_ids.add(f_id)
            new_fact_ptrs.add(in_memset_facts[f_id])


i8_arr = i8[::1]
# NOTE: Generator based implementation is probably way more efficient 
#  but causes memleak see numba issue 6993
@njit(cache=True)
def iter_arg_gval_ptrs(self, arg_gval_t_ids):
    n_args = len(arg_gval_t_ids)
    lengths = np.empty((n_args,),dtype=np.int64)
    gval_ptrs = np.empty((n_args,),dtype=np.int64)

    # Fill these vectors 
    new_fact_ptrs_list = List.empty_list(VectorType)
    olds_facts_list = List.empty_list(VectorType)
    facts_lists = List.empty_list(VectorType)
    for gval_t_id in arg_gval_t_ids:
        new_f_ids, new_fact_ptrs, old_fact_ptrs = self.fact_vecs[u2(gval_t_id)]
        new_fact_ptrs_list.append(new_fact_ptrs)
        olds_facts_list.append(old_fact_ptrs)
        facts_lists.append(old_fact_ptrs)

    gval_ptrs_seq = List.empty_list(i8_arr)
    for is_new in range_product(2*np.ones(n_args)):
        if(not np.any(is_new)): continue
        for i in range(n_args):
            facts_lists[i] = new_fact_ptrs_list[i] if is_new[i] else olds_facts_list[i]
            lengths[i] = len(facts_lists[i])
        # print("::",is_new,lengths)
        for inds in range_product(lengths):
            okay = True
            for i, ind in enumerate(inds):
                ptr = facts_lists[i][ind]
                if(ptr == 0): okay = False; break;
                gval_ptrs[i] = ptr

            # print(is_new, inds, lengths, gval_ptrs, okay, arr_is_unique(gval_ptrs))
            if(okay and arr_is_unique(gval_ptrs)):
                gval_ptrs_seq.append(gval_ptrs.copy())
                # yield gval_ptrs
    return gval_ptrs_seq


@generated_jit(nopython=True)
def update_op(self, return_type, arg_types, op_ind):
    '''Updates the FeatureApplier for the op at 'op_ind'. Goes through
        if one of op's arguments is associated with t_id, '''
    context = cre_context()
    return_type = return_type.instance_type
    arg_types = tuple([x.instance_type for x in arg_types])
    n_args = len(arg_types)
        
    # call_head_ptrs_func_type = types.FunctionType(return_type(i8[::1]))
    gval_type = get_gval_type(return_type)
    arg_gval_t_ids = tuple([context.get_t_id(_type=get_gval_type(x)) for x in arg_types])

    range_n_args = tuple([x for x in range(n_args)])
    n_var_tup_type = tuple([VarType for x in range(n_args)])

    get_ret = get_return_val_impl(return_type)
    set_bases = tuple([set_base_arg_val_impl(a) for a in arg_types])

    for i in range(1, len(set_bases)):
        if(set_bases[i] != set_bases[0]):
            raise NotImplementedError("Cannot Currently use feature applier with CREFunc with more than unique arg_type.")
    set_base = set_bases[0]
    arg_type = arg_types[0]

    def impl(self, return_type, arg_types, op_ind):
        op = self.ops[op_ind]
        val_ptrs = np.empty((n_args,),dtype=np.int64)
        var_ptrs = np.empty((n_args,),dtype=np.int64)

        call_self_func = get_best_call_self(op,True)

        # call_head_ptrs = _func_from_address(call_head_ptrs_func_type, op.call_head_ptrs_addr)
        enum_d = self.enumerizer.dict_for_type(return_type)
        # print("A")
        for gval_ptrs in iter_arg_gval_ptrs(self, arg_gval_t_ids):
            # For each gval get the address for 'head' and val' slot
            for i in literal_unroll(range_n_args):
                var_ptrs[i] = _load_ptr(i8, _ptr_to_data_ptr(gval_ptrs[i])+head_offset) 
                set_base(op, u8(i), _load_ptr(arg_type, _ptr_to_data_ptr(gval_ptrs[i])+val_offset))
                # val_ptrs[i] = _ptr_to_data_ptr(gval_ptrs[i])+val_offset
            # print("B")
            # Make a gval with head=TupleFact(op,*vars) and val=call(*vals)
            tf = TF(op,*_struct_tuple_from_pointer_arr(n_var_tup_type, var_ptrs))
            status = call_self_func(op)
            if(status > CFSTATUS_TRUTHY):
                continue
            v = get_ret(op)
            # print(v)
            nom = self.enumerizer.enumerize(v,enum_d)
            gval = new_gval(head=tf,val=v,nom=nom)
            idrec = self.out_memset.declare(gval)
            # print("C")
            # Map each arg's idrecs in 'in_memset' to the gval's idrec in 'out_memset' 
            for ptr in gval_ptrs:
                fact = cast(ptr, BaseFact)
                if(fact.idrec not in self.idrec_map):
                    self.idrec_map[u8(fact.idrec)] = List.empty_list(u8)
                self.idrec_map[u8(fact.idrec)].append(idrec)

    return impl


@njit(cache=True)
def assign_new_f_ids(self):
    for t_id in self.fact_vecs:
        new_f_ids, new_fact_ptrs, old_fact_ptrs = self.fact_vecs[u2(t_id)]
        for i in range(len(new_f_ids)):
            f_id = new_f_ids[i]
            old_fact_ptrs.set_item_safe(f_id,new_fact_ptrs[i])

        self.fact_vecs[u2(t_id)] = (new_vector(1),new_vector(1), old_fact_ptrs)


@njit(types.boolean(GenericFeatureApplierType, MemSetType),cache=True)
def check_same_in_memset(self,in_memset):
    return cast(self.in_memset, i8) == cast(in_memset, i8)

@njit(types.void(GenericFeatureApplierType, MemSetType), cache=True)
def set_in_memset(self, in_memset):
    self.in_memset = in_memset
    # update_fact_vecs(self)

@njit(MemSetType(GenericFeatureApplierType),cache=True)
def get_in_memset(self):
    return self.in_memset

@njit(MemSetType(GenericFeatureApplierType),cache=True)
def get_out_memset(self):
    return self.out_memset

@njit(types.void(GenericFeatureApplierType,MemSetType),cache=True)
def set_out_memset(self, out_memset):
    self.out_memset = out_memset


@njit(types.void(GenericFeatureApplierType),cache=True)
def feature_applier_clear(self):
    self.idrec_map = Dict.empty(u8,u8_list)
    self.change_queue_head = 0
    for t_id in self.fact_vecs:
        self.fact_vecs[u2(t_id)] = (new_vector(1), new_vector(1), new_vector(1))

    # self.out_memset = MemSet(self.out_memset.context_data)




            
                


##### End #####




            


# if(__name__ == "__main__"):
#     import faulthandler; faulthandler.enable()
#     from cre.flattener import Flattener
#     from cre.builtin_cre_funcs import Equals
#     with cre_context("test_feature_applier") as context:
#         spec1 = {"A" : {"type" : "string", "visible" : True}, 
#                  "B" : {"type" : "number", "visible" : False}}
#         BOOP1 = define_fact("BOOP1", spec1)
#         spec2 = {"inherit_from" : BOOP1,
#                  "C" : {"type" : "number", "visible" : True}}
#         BOOP2 = define_fact("BOOP2", spec2)
#         spec3 = {"inherit_from" : BOOP2,
#                  "D" : {"type" : "number", "visible" : True}}
#         BOOP3 = define_fact("BOOP3", spec3)

#         eq_f8 = Equals(f8, f8)
#         eq_str = Equals(unicode_type, unicode_type)
#         fa = FeatureApplier([eq_f8,eq_str],MemSet())
#         fa.apply()


#         mem = MemSet()
#         a = BOOP1("A", 1)
#         b = BOOP1("B", 2)
#         c = BOOP2("C", 3, 13)
#         d = BOOP2("D", 4, 13)
#         e = BOOP3("E", 5, 14, 106)
#         f = BOOP3("A", 6, 16, 106)

#         a_idrec = mem.declare(a)
#         b_idrec = mem.declare(b)
#         c_idrec = mem.declare(c)
#         d_idrec = mem.declare(d)
#         e_idrec = mem.declare(e)
#         f_idrec = mem.declare(f)
#         print("-------")

#         fl = Flattener((BOOP1, BOOP2, BOOP3), mem, id_attr="A")
        
#         flat_mem = fl.apply()

        

#         fa = FeatureApplier([eq_f8,eq_str],flat_mem)
#         with PrintElapse("POOP"):
#             feat_mem = fa.apply()
#         with PrintElapse("POOP"):
#             feat_mem = fa.apply()
#         print(feat_mem)


#         mem = MemSet()
#         for i in range(3):
#             mem.declare(BOOP2(str(i),i,i))

#         fl = Flattener((BOOP1, BOOP2, BOOP3), mem, id_attr="A")
#         flat_mem = fl.apply()

#         fa = FeatureApplier([eq_str,eq_f8],flat_mem)
#         with PrintElapse("100x100 str"):
#             feat_mem = fa.apply()

#         fa = FeatureApplier([eq_f8],flat_mem)
#         with PrintElapse("100x100 f8"):
#             feat_mem = fa.apply()





# @njit(cache=True)
# def bar(arg_ind, f_id, lens):
#     for x in foo(arg_ind, f_id, lens):
#         print(x)
# # foo(1, 2, np.array((10,10,10)))
# bar(0, 5, np.array((10,10,10)))



# @generated_jit(cache=True)
# def update_gval_multi(return_type, arg_types, call_addr, check_addr, arg_ind, in_memset, out_memset):
#     nargs = len(arg_types)
#     gval_argtypes = tuple([get_gval_type(t) for t in arg_types])


#     # def impl():
#     #     # literal_unroll

#     in_memset.get_facts(arg_types)




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
#     for inds in range_product(np.array([3,3,3],dtype=np.int64)):
#         print(inds)

#     import faulthandler; faulthandler.enable()
#     g = ground_op(Add(Var(f8),Var(f8)), 1., 7.)
#     print("<<g", g)
#     g = ground_op(Add(Var(unicode_type), Var(unicode_type)),  "1", "7")
#     print("<<g", g)
