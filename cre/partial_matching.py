import numpy as np
import numba
from numba import types, njit, i8, u8, i2, i4, u1, u2, i8, f8, f4, literally, generated_jit
from numba.extending import SentryLiteralArgs
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple
from numba.experimental.structref import new, define_boxing, StructRefProxy
import numba.experimental.structref as structref
from cre.utils import (cast, wptr_t, ptr_t, _dict_from_ptr, _get_array_raw_data_ptr,
         _ptr_from_struct_incref, decode_idrec, CastFriendlyMixin,
        encode_idrec, deref_info_type, DEREF_TYPE_ATTR, DEREF_TYPE_LIST, _obj_cast_codegen,
         _ptr_to_data_ptr, _list_base_from_ptr, _load_ptr, PrintElapse, meminfo_type,
         _decref_structref, _decref_ptr, cast_structref, _struct_tuple_from_pointer_arr, _meminfo_from_struct,
         lower_getattr, lower_setattr, ptr_to_meminfo, _memcpy, _incref_ptr, _incref_structref)
from cre.structref import define_structref, StructRefType
from cre.caching import gen_import_str, unique_hash_v, import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.memset import MemSetType
from cre.vector import VectorType
from cre.var import VarType
# from cre.op import CREFuncType
from cre.func import (CREFuncType, CFSTATUS_TRUTHY, CFSTATUS_FALSEY, CFSTATUS_NULL_DEREF

        , get_best_call_self, set_base_arg_val_impl, REFKIND_UNICODE, REFKIND_STRUCTREF)
from cre.conditions import LiteralType, build_distributed_dnf, ConditionsType
from cre.vector import VectorType, new_vector
from cre.fact import BaseFact, resolve_deref_data_ptr
from cre.why_not import new_why_not, why_not_type, WN_VAR_TYPE, WN_BAD_DEREF, WN_INFER_UNPROVIDED, WN_FAIL_MATCH
import cloudpickle

from numba.core.imputils import (lower_cast)

from cre.matching import (matcher_graph_ctor, MatcherGraph, MatcherGraphType, 
                        make_root_nodes, init_root_nodes, make_literal_nodes,
                        update_graph, NodeIOType)


@njit(cache=True)
def build_tspm_graph(ms, c):
    # Make all of the Matcher nodes
    root_nodes = make_root_nodes(ms, c)
    global_t_id_root_map = init_root_nodes(root_nodes)
    nodes_by_nargs, global_modify_map = \
         make_literal_nodes(ms, c)
    var_end_join_ptrs = np.zeros((len(c.vars),len(c.vars)),dtype=np.int64)
    end_nodes = root_nodes.copy()#Dict.empty(i8,CorgiNodeType)
    n_nodes = 0
    total_weight = float(len(root_nodes))

    for i, nodes in enumerate(nodes_by_nargs):
        for node in nodes:
            # Set the inputs to be the root nodes' outputs
            inputs = List.empty_list(NodeIOType)
            for j, var_ind in enumerate(node.var_inds):
                inputs.append(root_nodes[var_ind].outputs[0])
            node.inputs = inputs

            total_weight += node.lit.weight 
            n_nodes += 1

    return matcher_graph_ctor(ms, c, nodes_by_nargs, n_nodes, 
                root_nodes, global_modify_map, global_t_id_root_map,
                 total_weight)
            
@njit(cache=True)
def get_partial_matcher_graph(ms, conds):
    needs_new_graph = False

    if(conds.partial_matcher_inst is None):
        needs_new_graph = True
    else:
        graph = cast(conds.partial_matcher_inst, MatcherGraphType)
        if(cast(ms, i8) != cast(graph.memset, i8)):
            needs_new_graph = True

    print("Needs new graph", needs_new_graph)

    graph = build_tspm_graph(ms, conds)
    conds.partial_matcher_inst = cast(graph, StructRefType)

    
    return graph


@njit(cache=True)
def argmax_sm(score_matrix):
    fixed_inds = -np.ones(len(score_matrix), dtype=np.int64)
    for i, score_row in enumerate(score_matrix):
        fixed_inds[i] = np.argmax(score_row)
    return fixed_inds

@njit
def _score_mapping(graph, mapping):
    score = u2(0)
    for n, nodes in enumerate(graph.nodes_by_nargs):

        # Alpha Case
        if(n == 0):
            for node in nodes:
                # print(node.lit)
                inp_state = node.input_state_buffers[0]
                v_ind0 = node.var_inds[0]
                fix0 = mapping[v_ind0]
                if(fix0 >= 0 and inp_state[fix0].true_count > 0):
                    # print("YES", node.lit)
                    score += 1
                # else:
                    # print("NO", node.lit)
        elif(n == 1):
            for node in nodes:
                # print(node.lit)
                tt = node.truth_table
                v_ind0 = node.var_inds[0]
                v_ind1 = node.var_inds[1]
                fix0 = mapping[v_ind0]
                fix1 = mapping[v_ind1]
                if( fix0 >= 0 and fix1 >= 0 and
                    tt[mapping[v_ind0], mapping[v_ind1]]):
                    # print("YES", node.lit)
                    score += 1
                # else:
                #     print("NO", node.lit)
    return score

@njit(cache=True)
def get_best_ind_iter(score_matrix, fixed_inds):
    # print_sm(score_matrix)
    # print("INP", fixed_inds)
    best_iter = None
    best_unamb = (-1, 0.0)
    for i in range(len(score_matrix)):
        # Skip if already assigned
        if(fixed_inds[i] != -2):
            continue

        row = score_matrix[i].astype(np.int32)
        inds = np.argsort(-row)
        # print("inds", inds)
        # Don't make iterators for rows of all zeros
        if(row[inds[0]] == 0):
            continue

        inds = inds[:np.argmin(row[inds])]
        scores = row[inds]
        max_diff = scores[0] - scores[1:]
        
        if(len(max_diff) > 0):
            # NOTE: Maybe harmonic mean is better?
            # amb = (len(scores)-1)/np.mean(1/(1+max_diff))
            unamb = (scores[0], np.mean(scores[0] - scores[1:]))
        else:
            unamb = (scores[0], scores[0])
        # print("unamb", unamb)
        if(unamb > best_unamb):
            best_iter = (i, inds)
            best_unamb = unamb
    # print("END", best_iter)
    return best_iter

@njit
def _get_fixed_inds(graph, conds, match_ptrs=None):
    fixed_inds = np.full(len(conds.vars), -2, dtype=np.int64)
    if(match_ptrs is not None):        
        for i, ptr in enumerate(match_ptrs):
            if(ptr != 0):
                fact = cast(ptr, BaseFact)
                root_inp = graph.root_nodes[i].inputs[0]
                fixed_inds[i] = root_inp.idrecs_to_inds[fact.idrec]
    return fixed_inds


score_mapping_type = Tuple((f4, i8[::1]))
stack_item_type = Tuple((i8,i8, i8[::1], i8[::1], f4))

@njit(cache=True)
def partial_match(ms, conds, match_ptrs, tolerance, max_loops=5000):
    # Partial matching graph is flat, none of the 
    #  literals are considered dependances of each other
    graph = get_partial_matcher_graph(ms, conds) 
    update_graph(graph)

    fixed_inds = _get_fixed_inds(graph, conds, match_ptrs)
    results = List.empty_list(score_mapping_type)
    iter_stack = List.empty_list(stack_item_type)
    it = None
    c = 0
    score, best_score, score_bound = f4(0), f4(0), f4(0)
    n_var = len(conds.vars)
    # results = List.empty_list(

    # Outer loop handles generation of iterators over ambiguous
    #  variable assignments. 
    n_loops = 0
    while(n_loops < max_loops):
        n_loops += 1
        
        # c += 1
        # Inner loop recalcs score matrix, from current fixed_inds.
        #  Loops multiple times if new scores imply some previously
        #  unfixed variable now has an unambiguous mapping.
        while(True):        
            # Recalculate the score matrix w/ fixed_inds
            score_matrix, beta_matrix = \
                    _calc_remap_score_matrices(graph, fixed_inds)

            # print("score_matrix")
            # print_sm(score_matrix)

            # Look for unambiguous mapping in the new matrix
            fixed_inds, unamb_cnt, _ = get_unambiguous_inds(score_matrix, fixed_inds)

            # print("FIX",fixed_inds, unamb_cnt)

            if(unamb_cnt == 0):
                break

        score_bound = bound_mapping(
            argmax_sm(score_matrix),
            score_matrix, beta_matrix
        )

        # print("loop", n_loops, score_bound, fixed_inds)

        # print(f"BEST={best_score}", f"BOUND={score_bound}")
        backtrack = False

        # Case: Abandon if the upper bound on the current assignment's 
        #   score is less than the current best score. 
        if(score_bound < best_score * (1.0-tolerance)):
            # print("backtrack", score_bound)
            backtrack = True
        

        # Case: All vars fixed so store mapping. Then backtrack. 
        elif(np.all(fixed_inds != -2)):
            mapping = fixed_inds.copy()
            
            score = _score_mapping(graph, mapping)
            results.append((score, mapping))
            # print()
            # print("S", score, score_bound, mapping)
            # return 
            if(score > best_score):
                best_result = mapping#(mapping, matched_mask)
                best_score = score

            backtrack = True

        

        if(backtrack):
            # print("B")
            

            while(len(iter_stack) > 0):
                i, c, js, old_fixed_inds, sb = iter_stack.pop()

                # Keep popping off stack until the score bound
                #  could feasibly produce a better result.
                if(sb < best_score * (1.0-tolerance)):
                    continue
                
                # Get fixed inds for the popped iterator
                #  and push its next state to the stack
                fixed_inds = old_fixed_inds.copy()
                fixed_inds[i] = js[c]
                c += 1
                if(c < len(js)):
                    iter_stack.append((i, c, js, old_fixed_inds, sb))
                    # print("PUSH", i, c, js, old_fixed_inds)
                    break

            if(len(iter_stack) == 0):
                # Case: All iterators exhausted (i.e. Finished)
                # print("FINISHED")
                break
        else:
            # Case: some assignments ambiguous so try to make next iter.
            #  'fixed_inds' is set to the first choice of var_i -> ind_j.
            #  Iterator for rest are pushed to stack. 
            # print("A")
            best_iter = get_best_ind_iter(score_matrix, fixed_inds)
            if(best_iter is not None):
                (i,js) = best_iter
                iter_stack.append((i, 1, js, fixed_inds.copy(), score_bound))
                fixed_inds[i] = js[0]
                # print("PUSH", i, 1, js, fixed_inds.copy())
                
            else:
                # Case: Making iterator failed because only unconstrained
                #  assignments remain. Set all unassigned (-2s) to
                #  unconstrained (-1s)
                fixed_inds[fixed_inds==-2] = -1
                # print("UNFIXED", fixed_inds)

            # Case: fixed_inds has been set by popping from stack
        # else:
    scores = np.empty(len(results), dtype=np.float32)
    match_ptrs = np.empty((len(results),n_var), dtype=np.int64)
    c = 0
    for score, mapping in results:
        if(score >= best_score*(1.0-tolerance)):

            # Map the indicies to ptrs
            ptrs = np.empty(n_var, dtype=np.int64)
            for j, ind in enumerate(mapping):
                if(ind >= 0):
                    inp = graph.root_nodes[j].inputs[0]
                    idrec = inp.match_idrecs[ind]
                    t_id, f_id, _  = decode_idrec(idrec)
                    facts = cast(ms.facts[t_id], VectorType)
                    ptrs[j] = facts.data[f_id]
                else:
                    ptrs[j] = 0

            match_ptrs[c] = ptrs
            scores[c] = (score + n_var) / graph.total_weight
            c += 1

    order = np.argsort(-scores[:c])
    # for s, m in zip(scores[order], mappings[order]):
    #     print(s,m)
    return scores[order], match_ptrs[order]



f4_arr = f4[::1]
@njit
def _calc_remap_score_matrices(graph, fixed_inds):
    score_matrix = List.empty_list(f4_arr)
    beta_matrix = List.empty_list(f4_arr)
    for var_ind, root_inp in enumerate(graph.root_nodes):
        W = len(root_inp.inputs[0].match_idrecs)
        score_row = np.zeros(W, dtype=np.float32)
        beta_row = np.zeros(W, dtype=np.float32)
        score_matrix.append(score_row)
        beta_matrix.append(beta_row)

    
    # N = len(graph.root_nodes)
    # M = len(root_inp.match_idrecs)#inp_widths[0]
    # print("N,M", N, M)
    # score_matrix = np.zeros((N, M), dtype=np.uint16)
    # beta_matrix = np.zeros((N, M), dtype=np.uint16)

    # fixed_vars = np.full(M, -2, dtype=np.int16)
    # for i,j in enumerate(a_fixed_inds):
    #     if(j != -2):
    #         b_fixed_inds[j] = i

    for n, nodes in enumerate(graph.nodes_by_nargs):
        # Alpha Case
        if(n == 0):
            for node in nodes:
                w = node.lit.weight
                inp_state = node.input_state_buffers[0]
                # tt = node.truth_table[0]

                # print("Node", node.lit)
                # print(inp_state)
                v_ind0 = node.var_inds[0]
                fix0 = fixed_inds[v_ind0]
                score_row = score_matrix[v_ind0]

                if(fix0 < 0):
                    for i in range(len(score_row)):
                        score_row[i] += w*(inp_state[i].true_count > 0)
                else:
                    score_row[fix0] += w*(inp_state[fix0].true_count > 0)
                # print(v_ind0, "->", score_row)

        # Beta Case
        elif(n == 1):
            # pass
            for node in nodes:
                w = node.lit.weight
                # print("Node", node.lit)
                # tt = node.truth_table
                v_ind0 = node.var_inds[0]
                v_ind1 = node.var_inds[1]
                fix0 = fixed_inds[v_ind0]
                fix1 = fixed_inds[v_ind1]
                tt = node.truth_table
                # print(node.lit, tt.shape)
                if(fix0 >= 0 and fix1 >= 0):
                    is_true = tt[fix0,fix1]
                    score_matrix[v_ind0][fix0] += is_true
                    score_matrix[v_ind1][fix1] += is_true
                    beta_matrix[v_ind0][fix0] += is_true
                    beta_matrix[v_ind1][fix1] += is_true
                else:
                    if(fix0 >= 0):
                        inp_state = node.input_state_buffers[0]
                        tr0 = w*(inp_state[fix0].true_count > 0)
                        tr1 = w*tt[fix0, :]
                        score_matrix[v_ind0][fix0] += tr0
                        score_matrix[v_ind1] += tr1
                        beta_matrix[v_ind0][fix0] += tr0
                        beta_matrix[v_ind1] += tr1


                    elif(fix1 >= 0):
                        inp_state = node.input_state_buffers[1]
                        tr1 = w*(inp_state[fix1].true_count > 0)
                        tr0 = w*tt[:, fix1]
                        score_matrix[v_ind1][fix1] += tr1
                        score_matrix[v_ind0] += tr0
                        beta_matrix[v_ind1][fix1] += tr1
                        beta_matrix[v_ind0] += tr0

                    else:
                        inp_state = node.input_state_buffers[0]
                        score_row = score_matrix[v_ind0]
                        beta_row = beta_matrix[v_ind0]
                        for i in range(len(score_row)):
                            tr = w*(inp_state[i].true_count > 0)
                            score_row[i] += tr
                            beta_row[i] += tr

                        inp_state = node.input_state_buffers[1]
                        score_row = score_matrix[v_ind1]
                        beta_row = beta_matrix[v_ind1]
                        for i in range(len(score_row)):
                            tr = w*(inp_state[i].true_count > 0)
                            score_row[i] += tr
                            beta_row[i] += tr
                                                    

                    # inp_state = node.input_state_buffers[0]
                    # score_row = score_matrix[v_ind0]
                    # if(fix0 < 0):
                    #     for i in range(len(score_row)):
                    #         score_row[i] += inp_state[i].true_count > 0
                    # else:
                    #     score_row[fix0] += inp_state[fix0].true_count > 0

                    # # print(v_ind0, "->", score_row)

                    # inp_state = node.input_state_buffers[1]
                    # score_row = score_matrix[v_ind1]
                    # if(fix1 < 0):
                    #     for i in range(len(score_row)):
                    #         score_row[i] += inp_state[i].true_count > 0
                    # else:
                    #     score_row[fix1] += inp_state[fix1].true_count > 0

                # print(v_ind1, "->", score_row)                


            #     if(fix0 >= 0):
            #         print("NOT Implemented")
            #     else:
            #         pass

        else:
            raise Exception("Not Implmented")

    return score_matrix, beta_matrix
    # for lst in graph.nodes_by_nargs:
    #     for node in lst:
    #         print(node.lit)
    #         print(node.truth_table)
    #         print()


@njit(cache=True)
def get_unambiguous_inds(score_matrix, fixed_inds):#, drop_unconstr):
    unamb_inds = fixed_inds.copy()
    unconstr_mask = np.zeros(len(fixed_inds),dtype=np.uint8)
    new_unamb = 0
    # N, M = score_matrix.shape
    for var_ind, score_row in enumerate(score_matrix):
        # Don't touch if already assigned  
        if(fixed_inds[var_ind] != -2):
            continue

        # Find any assignments with non-zero score
        cnt = 0
        nz_f_id = -1
        for f_id in range(len(score_row)):
            if(score_row[f_id] != 0):
                cnt += 1
                nz_f_id = f_id

        # If there is exactly one assignment with a non-zero
        #  score then apply that assignment.
        if(cnt == 1):
            all_cnt = 0
            for v_ind_other in range(len(score_matrix)):
                # TODO: Should also check t_id here
                if(score_matrix[v_ind_other][nz_f_id] != 0):
                    all_cnt += 1

            if(all_cnt == 1):
                new_unamb += 1
                unamb_inds[var_ind] = nz_f_id

        # Or if they all have a score of zero then mark
        #  them as unconstrainted.
        elif(cnt == 0):
            unconstr_mask[var_ind] = 1

    return unamb_inds, new_unamb, unconstr_mask


@njit(cache=True)
def bound_mapping(mapping, score_matrix, beta_matrix):
    score = u2(0)
    for i, j in enumerate(mapping):
        s = score_matrix[i][j]
        b = beta_matrix[i][j]
        score += s - (b / 2.0)# + u2(s != 0)
    return score

@njit(cache=True)
def print_sm(score_matrix):
    for score_row in score_matrix:
        print(score_row)


class PartialMatchIterator:
    def __init__(self, ms, conds, match_ptrs=None, tolerance=0.0,
                    return_scores=False):#, kind="fact", recover_types=False):
        self.return_scores = return_scores
        var_base_types = conds.var_base_types
        self.proxy_types = [x._fact_proxy for x in var_base_types]
        # self.kind = kind
        # self.recover_types = recover_types

        scores, ptrs = partial_match(ms, conds, match_ptrs, tolerance)
        self.match_scores = scores 
        self.match_ptrs = ptrs
        var_base_types = conds.var_base_types
        
        self.curr_ind = 0

    def __next__(self):
        if(self.curr_ind >= len(self.match_ptrs)):
            raise StopIteration()
        ptrs = self.match_ptrs[self.curr_ind]
        score = self.match_scores[self.curr_ind]
        self.curr_ind += 1
        arr = []
        for ptr, proxy_typ in zip(ptrs, self.proxy_types):
            if(ptr != 0):
                mi = ptr_to_meminfo(ptr)
                instance = super(StructRefProxy,proxy_typ).__new__(proxy_typ)
                instance._type = proxy_typ
                instance._meminfo = mi
                arr.append(instance)
            else:
                arr.append(None)
            
        if(self.return_scores):
            return (score, arr)
        else:
            return arr

    def __iter__(self):
        return self


        # NOTE: Only implementing generic version for now
    
