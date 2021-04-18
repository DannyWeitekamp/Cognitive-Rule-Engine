import operator
import numpy as np
from numba import types, njit, i8, u8, i4, u1, i8, literally, generated_jit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple, UniTuple, optional
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from numbert.experimental.utils import _struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_pointer,  lower_setattr, lower_getattr, _pointer_from_struct
from numbert.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache
from numbert.experimental.condition_node import Conditions, ConditionsType, initialize_conditions

from numbert.experimental.var import *
from numbert.experimental.predicate_node import GenericAlphaPredicateNodeType, GenericBetaPredicateNodeType

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
def get_alpha_inds(alpha_conjuncts, conds):
    n_vars = len(conds.vars)
    # alpha_inds_list = List.empty_list(list_i8_arr)
    # for alpha_conjuncts, _, _ in conds.distr_dnf:
    alpha_inds = List.empty_list(i8_arr)
    for _ in range(n_vars):
        alpha_inds.append(np.arange(5))

    for i, alpha_conjunct in enumerate(alpha_conjuncts):
        for term in alpha_conjunct:
            alpha_inds[i] = filter_alpha(term, alpha_inds[i])

            print(i, alpha_inds[i])
    # alpha_inds_list.append(alpha_inds)
    return alpha_inds

@njit(cache=True)
def get_pair_matches(alpha_inds, beta_conjuncts, beta_inds, conds):
    n_vars = len(conds.vars)
    pair_matches = List([List.empty_list(i8_i8_arr_tuple) for _ in range(n_vars)])
    print(beta_inds)
    # print(beta_inds)
    for i in range(n_vars):
        pair_matches_i = pair_matches[i]
        for j in range(n_vars):

            if(beta_inds[i,j] != -1):
                terms_ij = beta_conjuncts[beta_inds[i,j]]
                print(len(terms_ij))
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
                                 _pair_set[(pair[0],pair[1])] = 1
                        else:
                            for pair in pairs:
                                t = (pair[0], pair[1])
                                if(t in pair_set): _pair_set[t] = 1
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
                    print(pairs)

                pair_matches_i.append((j, pairs))
    return pair_matches



@njit(cache=True)
def fill_pairs_at(partial_matches, i, pair_matches):
    for j,pair_matches_i in pair_matches[i]:
        new_pms = List()
        #Apply the following to each partial_match "pm" so far:
        for pm in partial_matches:
            altered = False
            already_consistent = False
            
            #For each consistent pair (e_i, e_j) of elements matching concepts
            #   i and j respectively:
            for pair in pair_matches_i:
                e_i, e_j = pair[0], pair[1]
                okay = True
                #Bind the pair to partial match "pm" if:                
                #concept_i is unbound or concept_i is bound to e_i and concept_j is unbound
                if(not (pm[i] == -1 or (pm[i] == e_i and pm[j] == -1))): okay = False
                #and concept_j is unbound or concept_j is bound to e_j and concept_i is unbound
                if(not (pm[j] == -1 or (pm[j] == e_j and pm[i] == -1))): okay = False
                
                #and e_i and e_j are not bound elsewhere in partial match "pm"
                for p in range(len(pm)):
                    if(p != i and p != j and (pm[p] == e_i or pm[p] == e_j)):
                        okay = False; break;

                #Mark the partial match as consistent for this iteration if it
                #   is consistent with this pair.
                if(pm[j] == e_j and pm[i] == e_i): already_consistent = True
                
                #Then append the new partial match to the set of partial matches
                if(okay):
                    new_pm = pm.copy()
                    new_pm[i], new_pm[j] = e_i, e_j
                    new_pms.append(new_pm)
                    altered = True

            #If partial match "pm" hasn't picked up any new bindings to concepts i and j
            #   but at least one was not yet bound, then no further binding can be found 
            #   for "pm" and it is rejected. Otherwise as long as pm[i] is consistent 
            #   with pm[j] keep pm for the next iteration.
            if(not altered and pm[i] != -1 and pm[j] != -1 and already_consistent):
                new_pms.append(pm)
        partial_matches = new_pms
    return partial_matches

@njit(cache=True)
def conditions_get_matches(conds):
    if(not conds.is_linked): raise RuntimeError("Cannot match unlinked conditions object.")
    if(not conds.is_initialized): initialize_conditions(conds)

    n_vars = len(conds.vars)

    

    for alpha_conjuncts, beta_conjuncts, beta_inds in conds.distr_dnf:
        alpha_inds =  get_alpha_inds(alpha_conjuncts, conds)
        pair_matches = get_pair_matches(alpha_inds, beta_conjuncts, beta_inds, conds)

        partial_matches = List()
        partial_matches.append(-np.ones((n_vars,),dtype=np.int64))

        for i in range(n_vars):
            partial_matches = fill_pairs_at(partial_matches,i,pair_matches)
            print(partial_matches)


@njit(cache=True)
def get_matches(conds,kb=None):
    if(kb is not None):
        conds = get_linked_conditions_instance(conds,kb, True)
    return conditions_get_matches(conds)




        # for k in pair_matches:







