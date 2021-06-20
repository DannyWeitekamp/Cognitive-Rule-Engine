import operator
import numpy as np
from numba.np.arrayobj import _getitem_array_single_int, make_array
from numba import types, njit, i8, u8, i4, u1, i8, literally, generated_jit, literal_unroll
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple, UniTuple, optional
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from cre.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_pointer,  lower_setattr, lower_getattr, _pointer_from_struct
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from cre.condition_node import Conditions, ConditionsType, initialize_conditions, get_linked_conditions_instance

from cre.var import *
from cre.predicate_node import GenericAlphaPredicateNodeType, GenericBetaPredicateNodeType

from operator import itemgetter
from copy import copy



i8_arr = i8[::1]
i8_arr_2d = i8[:,::1]
list_i8_arr = ListType(i8_arr)
# list_i8_arr_2d = ListType(optional(i8_arr_2d))

i8_i8_arr_tuple = Tuple((i8,i8[:,::1]))
list_i8_i8_arr_tuple = ListType(i8_i8_arr_tuple)

i8_x2 = UniTuple(i8,2)


@njit(cache=True)
def filter_alpha(term, inds):
    png = cast_structref(GenericAlphaPredicateNodeType, term.pred_node)
    mi = _meminfo_from_struct(png)
    return png.filter_func(mi, term.link_data, inds, term.negated)


@njit(cache=True)
def filter_beta(term, l_inds, r_inds):
    png = cast_structref(GenericBetaPredicateNodeType, term.pred_node)
    mi = _meminfo_from_struct(png)
    return png.filter_func(mi, term.link_data, l_inds, r_inds, term.negated)



@njit(cache=True)
def get_alpha_inds(facts_per_var, alpha_conjuncts, conds):
    n_vars = len(conds.vars)
    # alpha_inds_list = List.empty_list(list_i8_arr)
    # for alpha_conjuncts, _, _ in conds.distr_dnf:

    # kb = _struct_from_pointer(KnowledgeBaseType,conds.kb_ptr)
    # kb_data = kb.kb_data
    # # Using this dictionary might be a little slow, but it's low frequency
    # t_id_map = kb.context_data.fact_to_t_id


    alpha_inds = List.empty_list(i8_arr)
    # for _ in range(n_vars):
    #     alpha_inds.append(np.arange(5))

    for i, (facts,alpha_conjunct) in enumerate(zip(facts_per_var,alpha_conjuncts)):
        inds = np.arange(len(facts))
        for term in alpha_conjunct:
            inds = filter_alpha(term, inds)
        
        alpha_inds.append(inds)

            # print(i, alpha_inds[i])
    # print("alpha_inds", alpha_inds)
    # alpha_inds_list.append(alpha_inds)
    return alpha_inds

@njit(cache=True)
def get_pair_matches(alpha_inds, beta_conjuncts, beta_inds, conds):
    n_vars = len(conds.vars)
    pair_matches = List([List.empty_list(i8_i8_arr_tuple) for _ in range(n_vars)])
    # print(beta_inds)
    # print(beta_inds)
    for i in range(n_vars):
        pair_matches_i = pair_matches[i]
        for j in range(n_vars):

            if(beta_inds[i,j] != -1):
                terms_ij = beta_conjuncts[beta_inds[i,j]]
                # print(len(terms_ij))
                if(len(terms_ij) > 1):
                    # When two or more beta nodes act on the same pair of variables
                    #   we need to find the intersection of the pairs they form
                    #   Note: this could probably be rethought to be more performant
                    #     but it is probably a rare enough case that it doesn't matter
                    pair_set = None
                    for term in terms_ij:
                        pairs = filter_beta(term, alpha_inds[i], alpha_inds[j])
                        _pair_set = Dict.empty(i8_x2, u1)
                        if(pair_set is None):
                            for pair in pairs:
                                 _pair_set[(pair[0],pair[1])] = u1(1)
                        else:
                            for pair in pairs:
                                t = (pair[0], pair[1])
                                if(t in pair_set): _pair_set[t] = u1(1)
                        pair_set = _pair_set        
                    pairs = np.empty((len(pair_set),2),dtype=np.int64)        
                    for k,(a,b) in enumerate(pair_set.keys()):
                        pairs[k][0] = a
                        pairs[k][1] = b


                else:
                    # The more common case: there is just one beta term for this
                    #  pair of variables  
                    term = terms_ij[0]
                    pairs = filter_beta(term, alpha_inds[i], alpha_inds[j])
                    # print(pairs)

                pair_matches_i.append((j, pairs))

    # print(pair_matches)
    return pair_matches



@njit(cache=True)
def fill_pairs_at(partial_matches, i, pair_matches):
    for j,pair_matches_i in pair_matches[i]:
        new_pms = List()
        #Apply the following to each partial_match "pm" so far:
        for pm in partial_matches:
            altered = False
            already_consistent = False
            
            # For each consistent pair (e_i, e_j) of elements matching concepts
            #   i and j respectively:
            for pair in pair_matches_i:
                e_i, e_j = pair[0], pair[1]
                okay = True
                # Bind the pair to partial match "pm" if:                
                #  var_i is unbound or var_i is bound to e_i and var_j is unbound
                if(not (pm[i] == -1 or (pm[i] == e_i and pm[j] == -1))): okay = False
                # and var_j is unbound or var_j is bound to e_j and var_i is unbound
                if(not (pm[j] == -1 or (pm[j] == e_j and pm[i] == -1))): okay = False
                
                # and e_i and e_j are not bound elsewhere in partial match "pm"
                for p in range(len(pm)):
                    if(p != i and p != j and (pm[p] == e_i or pm[p] == e_j)):
                        okay = False; break;

                # Mark the partial match as consistent for this iteration if it
                #   is consistent with this pair.
                if(pm[j] == e_j and pm[i] == e_i): already_consistent = True
                
                # Then append the new partial match to the set of partial matches
                if(okay):
                    new_pm = pm.copy()
                    new_pm[i], new_pm[j] = e_i, e_j
                    new_pms.append(new_pm)
                    altered = True

            # If partial match "pm" hasn't picked up any new bindings to concepts i and j
            #   but at least one was not yet bound, then no further binding can be found 
            #   for "pm" and it is rejected. Otherwise as long as pm[i] is consistent 
            #   with pm[j] keep pm for the next iteration.
            if(not altered and pm[i] != -1 and pm[j] != -1 and already_consistent):
                new_pms.append(pm)
        partial_matches = new_pms
    return partial_matches

@njit(nogil=False, parallel=False,fastmath=False,cache=True)
def fill_singles_at(partial_matches,i, candidate_inds, is_not):
    '''
    For var_i which is free floating and without any beta
    constraints, bind the candidates for this variable
    to every partial match in partial_matches. 
    '''

    new_pms = List.empty_list(i8_arr)
    for pm in partial_matches:
        if(is_not):
            if(len(candidate_inds) == 0 and pm[i] == -1):
                new_pms.append(pm)
        else:
            if(pm[i] == -1):
                for ind in candidate_inds:
                    # if(not (inds == pm).any()):
                    new_pm = pm.copy()
                    new_pm[i] = ind
                    new_pms.append(new_pm)
            else:
                new_pms.append(pm)
    return new_pms



@njit(cache=True)
def _get_fact_vectors(conds):
    n_vars = len(conds.vars)
    kb = _struct_from_pointer(KnowledgeBaseType,conds.kb_ptr)
    kb_data = kb.kb_data

    # Using this dictionary might be a little slow, but it's low frequency
    t_id_map = kb.context_data.fact_to_t_id

    facts = List.empty_list(VectorType)    

    for i in range(n_vars):
        t_id = t_id_map[conds.vars[i].type_name]
        facts.append(facts_for_t_id(kb_data, t_id))
    return facts

@njit(i8[:,::1](ConditionsType),cache=True)
def _get_ptr_matches(conds):
    '''Takes a linked Conditions object and gets sets of pointers to 
       facts that match
    '''
    if(not conds.kb_ptr): raise RuntimeError("Cannot match unlinked conditions object.")
    if(not conds.is_initialized): initialize_conditions(conds)

    n_vars = len(conds.vars)
    fact_vectors = _get_fact_vectors(conds)

    is_not = np.empty(n_vars,dtype=np.uint8)
    for i in range(n_vars): 
        is_not[i] = conds.vars[i].is_not

    # partial_matches_set = Dict(i8_arr)
    for alpha_conjuncts, beta_conjuncts, beta_inds in conds.distr_dnf:
        alpha_inds = get_alpha_inds(fact_vectors, alpha_conjuncts, conds)
        pair_matches = get_pair_matches(alpha_inds, beta_conjuncts, beta_inds, conds)

        partial_matches = List.empty_list(i8_arr)
        partial_matches.append(-np.ones((n_vars,),dtype=np.int64))

        # print("A")
        for i in range(n_vars):
            partial_matches = fill_pairs_at(partial_matches,i,pair_matches)

        # print("B")
        for i in range(n_vars):
            if(len(pair_matches[i]) == 0):
                partial_matches = fill_singles_at(partial_matches,i,alpha_inds[i], is_not[i])
        # print("C", partial_matches)

    #Turn indicies into fact pointers 
    # Time Negligible
    matching_fact_ptrs = np.empty((len(partial_matches),n_vars-np.sum(is_not)),dtype=np.int64)
    for i,match in enumerate(partial_matches):
        j = 0
        for k, ind in enumerate(match):
            if(not is_not[k]):
                if(ind == -1):
                    return np.zeros((0,n_vars),dtype=np.int64)
                matching_fact_ptrs[i][j] = fact_vectors[k][ind]
                j +=1
    print("D", matching_fact_ptrs)
    return matching_fact_ptrs


@njit(cache=True)
def ensure_linked(conds, kb=None):
    if(conds.kb_ptr == 0):
        if(kb is None):
            raise RuntimeError("Cannot match Condtions that are not yet linked to a KnowledgeBase.")
        conds = get_linked_conditions_instance(conds,kb, True)
    return conds

@njit(cache=True)
def get_ptr_matches(conds, kb=None):
    conds = ensure_linked(conds, kb)
    return _get_ptr_matches(conds)



@intrinsic
def _struct_tuple_from_pointer_arr(typingctx, struct_types, ptr_arr):
    ''' Takes a tuple of fact types and a ptr_array i.e. an i8[::1] and outputs 
        the facts pointed to, casted to the appropriate types '''
    print(">>",struct_types)
    if(isinstance(struct_types, UniTuple)):
        typs = tuple([struct_types.dtype.instance_type] * struct_types.count)
        out_type =  UniTuple(struct_types.dtype.instance_type,struct_types.count)
    else:
        print(struct_types.__dict__)
        typs = tuple([x.instance_type for x in struct_types.types])
        out_type =  Tuple(typs)
    print(out_type)
    
    sig = out_type(struct_types,i8[::1])
    def codegen(context, builder, sig, args):
        _,ptrs = args

        vals = []
        ary = make_array(i8[::1])(context, builder, value=ptrs)
        for i, inst_type in enumerate(typs):
            i_val = context.get_constant(types.intp, i)

            # Same as _struct_from_pointer
            raw_ptr = _getitem_array_single_int(context,builder,i8,i8[::1],ary,i_val)
            meminfo = builder.inttoptr(raw_ptr, cgutils.voidptr_t)

            st = cgutils.create_struct_proxy(inst_type)(context, builder)
            st.meminfo = meminfo

            context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

            vals.append(st._getvalue())


        
        return context.make_tuple(builder,out_type,vals)

    return sig,codegen

# @generated_jit(cache=True,nopython=True)
# def _get_matches(conds, struct_types, kb=None):
#     print(type(struct_types))
#     print(struct_types)
#     if(isinstance(struct_types, UniTuple)):
#         typs = tuple([struct_types.dtype.instance_type] * struct_types.count)
#         out_type =  UniTuple(struct_types.dtype.instance_type,struct_types.count)
#     else:
#         raise NotImplemented("Need to write intrinsic for multi-type ")

    # print(typs)
@njit(cache=True, locals={"ptr_set" : i8[::1]})
def _get_matches(conds, struct_types, kb=None):
    

    
    ptr_matches = get_ptr_matches(conds, kb)

    out = List()

    for i in range(len(ptr_matches)):
        ptr_set = ptr_matches[i]
        facts = _struct_tuple_from_pointer_arr(struct_types,ptr_set)
        # print("Z")
        out.append(facts)
        # for i,fact_type in enumerate(literal_unroll(fact_types)):
        #     _struct_from_pointer(fact_type,ptr_set[i])

    return out

    # return impl




        # for k in pair_matches:







