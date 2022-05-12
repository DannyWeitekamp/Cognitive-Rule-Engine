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
from cre.utils import _func_from_address, _cast_structref, _obj_cast_codegen, _func_from_address, _incref_structref, _struct_from_ptr, decode_idrec, _ptr_to_data_ptr, _load_ptr
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
from cre.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor
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



# Flattener Struct Definition
feature_applier_fields = {
    **incr_processor_fields,    
    "out_mem" : MemoryType,
    "idrec_map" : DictType(u8,ListType(u8)),
    "inv_idrec_map" : DictType(u8,ListType(u8)),
    "ops_by_ret_t_id" : DictType(u2,ListType(GenericOpType)),
    "fact_vecs_needs_update" : types.bool_,
    "fact_vecs" : DictType(u2,ListType(ListType(VectorType))),
    "val_t_id_to_gval_t_id" : DictType(u2,u2),
    "unq_arg_types":  types.Any,
    "unq_return_types":  types.Any

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
def self_get_changes(self, end=-1, exhaust_changes=True):
    def impl(self, end=-1, exhaust_changes=True):
        incr_pr = _cast_structref(IncrProcessorType, self)
        return incr_pr.get_changes(end=end, exhaust_changes=exhaust_changes)
    return impl

u8_list = ListType(u8)
vec_list_type = ListType(VectorType)
list_vec_list_type = ListType(ListType(VectorType))
op_list_type = ListType(GenericOpType)

def get_feature_applier_type(ops):
    unq_return_types = Tuple(tuple(set([op.signature.return_type for op in ops])))
    unq_arg_types = Tuple(tuple(set(chain(*[op.signature.args for op in ops]))))
    print(unq_return_types, unq_arg_types)
    field_dict = {**feature_applier_fields, "unq_arg_types" : unq_arg_types,  "unq_return_types" : unq_return_types}
    f_type = FeatureApplierTypeClass([(k,v) for k,v in field_dict.items()])
    f_type._unq_return_types = unq_return_types
    f_type._unq_arg_types = unq_arg_types
    return f_type


@njit(cache=True)
def update_fact_vecs(self):
    self.fact_vecs = Dict.empty(u2, list_vec_list_type)
    print(self.fact_vecs)
    for ret_t_id, ops in self.ops_by_ret_t_id.items():
        fact_vecs_for_t_id = List.empty_list(vec_list_type)
        print(ret_t_id)
        for op in ops:
            fact_vecs = List.empty_list(VectorType)
            for i,v in enumerate(op.base_vars):
                print(v.base_t_id, self.val_t_id_to_gval_t_id)
                gval_t_id = self.val_t_id_to_gval_t_id[u2(v.base_t_id)]
                ptr = self.in_mem.mem_data.facts[gval_t_id]
                # print(gval_t_id, ptr)
                # Skip if any of the argument vectors are uninitialized
                if(ptr == 0): 
                    self.fact_vecs_needs_update = True
                    return
                fact_vec = _struct_from_ptr(VectorType,ptr)
                fact_vecs.append(fact_vec)
            fact_vecs_for_t_id.append(fact_vecs)
        self.fact_vecs[u2(ret_t_id)] = fact_vecs_for_t_id
    self.fact_vecs_needs_update = False
    # print("***********")
    


from numba.core.typing.typeof import typeof
@generated_jit(cache=True, nopython=True)    
def feature_applier_ctor(st_type, ops, in_mem, out_mem):
    # print(ops, in_mem, out_mem)
    context = cre_context()

    unq_arg_t_ids = []
    unq_gval_t_ids = []
    for arg_typ in st_type.instance_type._unq_arg_types:
        gval_type = get_gval_type(arg_typ)
        unq_arg_t_ids.append(context.get_t_id(_type=arg_typ))
        unq_gval_t_ids.append(context.get_t_id(_type=gval_type))

    arg_gval_t_ids = tuple([(a,b) for a,b in zip(unq_arg_t_ids,unq_gval_t_ids)])
    
    def impl(st_type, ops, in_mem, out_mem):
        st = new(st_type)
        init_incr_processor(st, in_mem)
        ops_by_ret_t_id = Dict.empty(u2,op_list_type)
        for op in ops:
            if(op.return_t_id not in ops_by_ret_t_id):
                ops_by_ret_t_id[op.return_t_id] = List.empty_list(GenericOpType)
            ops_by_ret_t_id[op.return_t_id].append(op)
        st.ops_by_ret_t_id = ops_by_ret_t_id

        # print(ops_by_ret_t_id)

        # st.ops = ops
        st.idrec_map = Dict.empty(u8, u8_list)
        st.inv_idrec_map = Dict.empty(u8, u8_list)
        st.out_mem = out_mem 

        st.val_t_id_to_gval_t_id = Dict.empty(u2,u2)
        for tup in literal_unroll(arg_gval_t_ids):
            arg_t_id, gval_t_id = tup
            st.val_t_id_to_gval_t_id[u2(arg_t_id)] = u2(gval_t_id)

        # print("*****:")
        update_fact_vecs(st)

        # print(st.fact_vecs)

        return st
    return impl


class FeatureApplier(structref.StructRefProxy):
    def __new__(cls, ops, in_mem=None, out_mem=None, context=None):
        context = cre_context(context)
        if(in_mem is None): in_mem = Memory(context);
        if(out_mem is None): out_mem = Memory(context);


        feature_applier_type = get_feature_applier_type(ops)
        self = feature_applier_ctor(feature_applier_type, List(ops), in_mem, out_mem)
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
        feature_applier_update(self)

define_boxing(FeatureApplierTypeClass, FeatureApplier)

@njit(types.void(GenericFeatureApplierType, MemoryType),cache=True)
def set_in_mem(self, x):
    self.in_mem = x
    update_fact_vecs(self)


@generated_jit(cache=True, nopython=True)
# @overload_method(EnumerizerTypeClass, "add_grounding")
def new_op_grounding(op, return_type, *args):
    '''Creates a new gval Fact for an op and a set of arguments'''
    # print(return_type,f8, return_type.__dict__, type(return_type))
    if(hasattr(return_type,'instance_type')):
        return_type = return_type.instance_type

    # Fix annoying numba bug where StarArgs becomes a tuple() of cre.Tuples
    if(isinstance(args,tuple) and isinstance(args[0],types.BaseTuple)):
        args = args[0]

    gval_type = get_gval_type(return_type)
    
    # Find the types for the op's check and call
    call_sig = return_type(*args)
    check_sig = types.bool_(*args)
    call_f_type = types.FunctionType(call_sig)
    check_f_type = types.FunctionType(check_sig)

    def impl(op, return_type, *args):
        # If check exists try it first 
        if(op.check_addr != 0):
            check = _func_from_address(check_f_type, op.check_addr)
            check_ok = check(*args)
            if(not check(*args)): raise ValueError("gval attempt failed check().")


        call = _func_from_address(call_f_type, op.call_addr)
        v = call(*args)

        gval = new_gval(head=TF(op,*args),val=v)
        # grounding = ctor(head=head, value=value)
        return gval
    return impl


def ground_op(op,*args):
    return_type = op.signature.return_type
    return new_op_grounding(op, return_type, *args)


# @njit(cache=True)
# def add(a,b):
#     return a+b




# def gen_src_apply_gval_multi(op,ind='    '):
#     arg_types = op.signature
#     n_args = len(arg_types)
#     has_check = (op.check_addr != 0)
#     src = f'''import cloudpickle
# sig = cloudpickle.loads({cloudpickle.dumps(op.signature)})
# arg_types = sig.args
# check_f_type = types.bool(*arg_types)
# {f"{', '.join([f'arg_type{i}' for i in range(n_args)])} = arg_types" }

# def update_gval_multi(op, in_mem, out_mem):
#     {"check = _func_from_address(check_f_type, op.check_addr)" if has_check else ""}
#     call = _func_from_address(sig, op.check_addr)"
# '''
#     for i in range(n_args):
#         src += f'{ind}gvals{i} = in_mem.get_facts(arg_types{i})'
#     c_ind = ind
#     src += f'''
#     lengths = ({",".join([f"len(gvals{i})" for i in range(n_args)])})
#     for const_pos in range({n_args}):
#         for inds in product_iter_w_const(lengths, const_post, 0):

#     '''
    
#     # src += "
#     for i in range(n_args):
#         src += f'{c_ind}for i{i} in range(len(gvals)):\n'
#         c_ind  += ind
#     # src += 

# @njit(cache=True)
# def extract_fact_vecs(op):
    
val_offset = gval_type.get_attr_offset('val')
    
@generated_jit
def iter_val_ptrs_for_change(self, ret_t_id, t_id, f_id):
    def impl(self, ret_t_id, t_id, f_id):
        for op_ind, op in enumerate(self.ops_by_ret_t_id[ret_t_id]):
            for const_pos, v in enumerate(op.base_vars):
                # Skip op args that don't match the change_event's t_id
                if(self.val_t_id_to_gval_t_id[u2(v.base_t_id)] != t_id): continue

                # Extract the fact_vecs and their lengths
                fact_vecs = self.fact_vecs[ret_t_id][op_ind]
                lengths = np.array([len(x) for x in fact_vecs],dtype=np.int64)
                gval_ptrs = np.empty((len(lengths),),dtype=np.int64)
                val_ptrs = np.empty((len(lengths),),dtype=np.int64)
                for inds in product_iter_w_const(lengths, const_pos, f_id):
                    # Extract the pointers to gvals check if okay (non-zero)
                    okay = True
                    for i, ind in enumerate(inds):
                        ptr = fact_vecs[i].data[ind]

                        # Skip holes in the fact_vec
                        if(ptr != 0):
                            gval_ptrs[i] = ptr
                        else:
                            okay = False; 
                            break;

                    if(okay):
                        if(len(np.unique(gval_ptrs)) == len(gval_ptrs)):
                            for i,ptr in enumerate(gval_ptrs):
                                val_ptrs[i] = _ptr_to_data_ptr(ptr)+val_offset
                                print(_load_ptr(f8,val_ptrs[i]))
                                # eq_f8


                            print(t_id, inds, val_ptrs)
                            yield val_ptrs
                            
    return impl



@generated_jit
def feature_applier_update(self):
    context = cre_context()
    return_t_ids = tuple([context.get_t_id(_type=typ) for typ in self._unq_return_types])
    print("<<", return_t_ids)
    def impl(self):
        if(self.fact_vecs_needs_update): update_fact_vecs(self)
        for change_event in self.get_changes():
            if(change_event.was_retracted):
                pass

            if(change_event.was_declared or change_event.was_modified):
                pass

            t_id, f_id, _ = decode_idrec(change_event.idrec)

            for ret_t_id in literal_unroll(return_t_ids):
                for val_ptrs in iter_val_ptrs_for_change(self, ret_t_id, t_id, f_id):
                    pass
                # update_for_ret_t_id(self,ret_t_id)
                # print(ret_t_id, self.ops_by_ret_t_id)
                
                    

    return impl

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


        mem = Memory()
        a = BOOP1("A", 1)
        b = BOOP1("B", 2)
        c = BOOP2("C", 3, 13)
        d = BOOP2("D", 4, 14)
        e = BOOP3("E", 5, 15, 105)
        f = BOOP3("F", 6, 16, 106)

        a_idrec = mem.declare(a)
        b_idrec = mem.declare(b)
        c_idrec = mem.declare(c)
        d_idrec = mem.declare(d)
        e_idrec = mem.declare(e)
        f_idrec = mem.declare(f)
        print("-------")

        fl = Flattener((BOOP1, BOOP2, BOOP3), mem)
        
        flat_mem = fl.apply()

        eq_f8 = Equals(f8, f8)
        eq_str = Equals(unicode_type, unicode_type)

        feat_mem = FeatureApplier([eq_f8],flat_mem)
        feat_mem.update()



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
