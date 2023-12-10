import numpy as np
import numba
from numba import types, njit, i8, u8, i4, u1, u2, i8, f8, f4, literally, generated_jit
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
import cloudpickle

from numba.core.imputils import (lower_cast)


RETRACT = u1(0xFF)# u1(0)
DECLARE = u1(0)


# -----------------------------------------------------------------------
# : CorgiNode


# -----------------------------------------------------------------------
# : NodeIO (i.e. input/output to node)

node_io_field_dict = {
    # Whether or not is a first (i.e. most upstream) input.
    "is_root" : types.boolean,

    # Note: these are pairs of buffer + slices into them

    # Indicies changed in this match cycle.
    "change_buffer" : i8[::1],
    "change_inds" : i8[::1],

    # Indices removed in this match cycle.
    "remove_buffer" : i8[::1],
    "remove_inds" : i8[::1],
    
    # The set of idrecs held by this input/output.
    "match_idrecs_buffer" : u8[::1], 
    "match_idrecs" : u8[::1], 

    # The indicies in the input of associated output facts.
    "match_inp_inds_buffer" : i8[::1], 
    "match_inp_inds" : i8[::1], 

    # A mapping of idrecs to their indicies in the output.
    "idrecs_to_inds" : DictType(u8,i8),

    # A vector that keeps track indicies of holes in the output.  
    "match_holes" : VectorType,

    # The size of the output including holes.  
    "width" : i8,

    # Weak pointer to the node that owns this memory.
    "parent_node_ptr" : i8,
}

NodeIO, NodeIOType = define_structref("NodeIO", node_io_field_dict)


@njit(cache=True)
def new_NodeIO():
    # print("NEW NODE")
    st = new(NodeIOType)
    st.change_buffer = np.empty(8,dtype=np.int64)#Dict.empty(u8,u1)
    st.change_inds = st.change_buffer[:0]

    st.remove_buffer = np.empty(8,dtype=np.int64)#Dict.empty(u8,u1)
    st.remove_inds = st.remove_buffer[:0]
    
    st.match_idrecs_buffer = np.empty(8,dtype=np.uint64)
    st.match_idrecs = st.match_idrecs_buffer[:0]
    st.match_inp_inds_buffer = np.empty(8,dtype=np.int64)
    st.match_inp_inds = st.match_inp_inds_buffer[:0]

    st.idrecs_to_inds = Dict.empty(u8,i8)
    st.match_holes = new_vector(8)
    st.width = 0
    # st.match_holes = new_vector(2)#Dict.empty(u8,i8)
    st.is_root = False
    st.parent_node_ptr = 0
    return st


_input_state_type = np.dtype([
    # The input fact's idrec
    ('idrec', np.int64),

    # For beta nodes the number of facts paired with this one. 
    ('true_count', np.int64),

    # The index of input fact in the associated output.
    ('output_ind', np.int64),

    # Some change to this input fact has occured this cycle.
    ('changed', np.uint8),

    # Indicates that the input fact is not removed and its relevant 
    #  attributes have been successfully dereferenced.
    ('is_valid', np.uint8), 
    
    # The input fact was inserted in this match cycle.
    ('recently_inserted', np.uint8),

    # The input fact was modified in this match cycle.
    ('recently_modified', np.uint8),

    # The input fact was removed or invalidated in this match cycle.
    ('recently_invalid', np.uint8),
    
    # The input fact was matched in the previous match cycle 
    ('true_was_nonzero', np.uint8), 

    # The input fact has ever been a match. Used to keep track of 
    #  where holes should be kept in this input's corresponding output. 
    ('true_ever_nonzero', np.uint8),

    # Pad to align w/ i8[:4]
    ('_padding0', np.uint8),
    # ('_padding1', np.uint8),
    # ('_padding2', np.uint8),
])
    
input_state_type = numba.from_dtype(_input_state_type)
dict_i8_u1_type = DictType(i8,u1)
deref_dep_typ = DictType(u8,DictType(ptr_t,u1))

base_corgi_node_field_dict = {
    
    # A weak ptr to the working memory for this graph 
    "memset_ptr" : i8, 

    # The Literal associated with this node
    "lit" : types.optional(LiteralType),

    # The Op for the node's literal
    "op" : types.optional(CREFuncType),

    # ???
    "deref_depends" : deref_dep_typ, 
    
    # The number of Vars (1 for alpha or 2 for beta)
    "n_vars" : i8,

    # The var_inds for the vars handled by this node 
    "var_inds" : i8[::1],

    # The t_ids for the vars handled by this node 
    "t_ids" : u2[::1],

    # The widths of each othe node's inputs.
    "inp_widths" : i8[::1],

    # For each input the resolved head pointers (i.e. pointers to 
    #  the heads of deref chains like A.next.next.value).  
    "head_ptr_buffers" : ListType(i8[:,::1]),

    # For each input a buffer holding the set of input_state structs 
    #  for each of the input's input facts. 
    "input_state_buffers" : ListType(input_state_type[::1]),
        
    # For each input a buffer from which removed_inds, changed_inds, 
    #   and unchanged_inds are slices.
    "inds_change_buffers" : ListType(i8[::1]),

    # For each input the indicies of facts that have changed 
    #  or been inserted upstream in this match cycle.
    "changed_inds" : ListType(i8[::1]),

    # For each input the indicies of facts that have not changed
    "unchanged_inds" : ListType(i8[::1]),

    # For each input the indicies of facts that been removed upstream
    #  or been invalidated in this match cycle.
    "removed_inds" : ListType(i8[::1]),

    # The 1 or 2 inputs.
    "inputs" : ListType(NodeIOType),

    # The associated 1 or 2 outputs.
    "outputs" : ListType(NodeIOType), #DictType(u8,DictType(u8,u1))

    # For beta nodes a size (width[0], width[1]) boolean table
    #  indicating whether a pair of match candidates match.
    "truth_table" : u1[:, ::1],

    # True if the inputs to this node both come from the same beta node
    # e.g. like in the case of (a.v < b.v) & (a.v != b.v)
    "upstream_same_parents" : u1,

    # True if 'upstream_same_parents' and the inputs are the same 
    #  with respect to which one is the 0th/1st input.
    "upstream_aligned" : u1,

    # A weak pointer to the node upstream to this one
    "upstream_node_ptr" : i8,

    # A vector of idrecs that is filled directly in parse_change_events()
    #  on each match cycle to indicate a modify() relevant to this node.
    "modify_idrecs" : ListType(VectorType),

}

CorgiNode, CorgiNodeType = define_structref("CorgiNode", base_corgi_node_field_dict, define_constructor=False)


u8_arr_typ = u8[::1]
i8_arr_typ = i8[::1]
i8_x2_arr_typ = i8[:,::1]
input_state_arr_type = input_state_type[::1]

@njit(cache=True)
def node_ctor(ms, t_ids, var_inds,lit=None):
    st = new(CorgiNodeType)
    st.memset_ptr = cast(ms, i8)
    st.deref_depends = Dict.empty(u8,dict_i8_u1_type)
    st.modify_idrecs = List.empty_list(VectorType)
    st.var_inds = var_inds
    st.t_ids = t_ids

    st.head_ptr_buffers = List.empty_list(i8_x2_arr_typ)
    st.input_state_buffers = List.empty_list(input_state_arr_type)

    st.inds_change_buffers = List.empty_list(i8_arr_typ)
    st.changed_inds = List.empty_list(i8_arr_typ)
    st.unchanged_inds = List.empty_list(i8_arr_typ)
    st.removed_inds = List.empty_list(i8_arr_typ)

    INIT_BUFF_LEN = 8
    st.lit = lit
    if(lit is not None):
        st.op = op = lit.op
        n_vars = st.n_vars = op.n_args
        for i in range(n_vars):
            l = op.head_ranges[i].end-op.head_ranges[i].start
            st.head_ptr_buffers.append(np.empty((INIT_BUFF_LEN,l),dtype=np.int64))
            st.input_state_buffers.append(np.zeros(INIT_BUFF_LEN, dtype=input_state_type))

            ind_change_buff = np.empty(INIT_BUFF_LEN, dtype=np.int64)
            st.inds_change_buffers.append(ind_change_buff)
            placeholder = np.empty(INIT_BUFF_LEN, dtype=np.int64)
            st.changed_inds.append(ind_change_buff)
            st.removed_inds.append(ind_change_buff)
            st.unchanged_inds.append(ind_change_buff)
            st.modify_idrecs.append(new_vector(INIT_BUFF_LEN))

    else:
        st.op = None
        n_vars = 1

    outputs = List.empty_list(NodeIOType)
    

    self_ptr = cast(st, i8)
    for i in range(n_vars):
        node_out = new_NodeIO()
        node_out.parent_node_ptr = self_ptr
        outputs.append(node_out)
        
    st.outputs = outputs
    st.truth_table = np.zeros((8,8), dtype=np.uint8)
    st.inp_widths = np.zeros(2,dtype=np.int64)

    # Just make False by default, can end up being True after linking
    st.upstream_same_parents = False
    st.upstream_aligned = False
    st.upstream_node_ptr = 0


    return st


# -----------------------------------------------------------------------
# : CorgiGraph

node_arg_pair_type = Tuple((CorgiNodeType,i8))
node_arg_list_type = ListType(node_arg_pair_type)
node_io_list_type = ListType(NodeIOType)
corgi_graph_field_dict = {
    # The change_head of the working memory at the last graph update.
    "change_head" : i8,

    # The working memory memset.
    "memset" : MemSetType,

    # All graph nodes organized by [[...alphas],[...betas],[...etc]]
    "nodes_by_nargs" : ListType(ListType(CorgiNodeType)),

    # The number of nodes in the graph
    "n_nodes" : i8,

    # Maps a var_ind to its associated root node (i.e. the node that 
    #  holds all match candidates for a fact_type before filtering).
    "var_root_nodes" : DictType(i8,CorgiNodeType),

    # TODO: Replace with list/array for speed?
    # The map a var_ind to the most downstream node that constrains that var.
    "var_end_nodes" : DictType(i8,CorgiNodeType),

    # A matrix of size (n_var, n_var) with weak pointers to the most 
    #  downstream beta nodes connecting each pair of vars. 
    "var_end_join_ptrs" : i8[::,::1],

    # Maps (t_id, 0, a_id) idrec patterns to (node,arg_ind) that should be 
    #  rechecked based on that pattern.
    "global_modify_map" : DictType(u8, node_arg_list_type),

    # Maps t_ids to the root node outputs associated with facts of that t_id.
    "global_t_id_root_map" : DictType(u2, NodeIOType),

    # A reference to the prototype instance for match iterators on this graph.
    "match_iter_prototype_inst" : types.optional(StructRefType), #NOTE: Should really use deferred type

    # The sum of weights of all literals in the graph
    "total_weight" : f4,
}


CorgiGraph, CorgiGraphType = define_structref("CorgiGraph", corgi_graph_field_dict, define_constructor=False)



@njit(cache=True)
def corgi_graph_ctor(ms, conds, nodes_by_nargs, n_nodes, var_root_nodes, var_end_nodes,
                var_end_join_ptrs, global_modify_map, global_t_id_root_map, total_weight):
    st = new(CorgiGraphType)
    st.change_head = 0
    st.memset = ms
    st.nodes_by_nargs = nodes_by_nargs
    st.n_nodes = n_nodes
    st.var_root_nodes = var_root_nodes
    st.var_end_nodes = var_end_nodes
    st.var_end_join_ptrs = var_end_join_ptrs
    st.global_modify_map = global_modify_map
    st.global_t_id_root_map = global_t_id_root_map
    # st.match_iter_prototype_ptr = 0
    st.total_weight = total_weight
    
    return st


@njit(cache=False)
def conds_get_corgi_graph(self):
    if(self.matcher_inst is not None):
        return cast(self.matcher_inst, CorgiGraphType)
    return None


# --------------------------------------
# : build_corgi_graph()

@njit(cache=True)
def _get_degree_order(c, index_map):
    ''' Order vars by decreasing beta degree --- the number of other 
        vars that they share beta literals with. Implements heuristic
        that the most beta-constrained nodes are matched first. 
     '''
    has_pairs = np.zeros((len(c.vars),len(c.vars)),dtype=np.uint8)
    for distr_conjunct in c.distr_dnf:
        for j, var_conjuct in enumerate(distr_conjunct):
            for lit in var_conjuct:
                var_inds = np.empty((len(lit.base_var_ptrs),),dtype=np.int64)
                for i, base_var_ptr in enumerate(lit.base_var_ptrs):
                    var_inds[i] = index_map[i8(base_var_ptr)]
                if(len(var_inds) > 1):
                    has_pairs[var_inds[0],var_inds[1]] = 1
                    has_pairs[var_inds[1],var_inds[0]] = 1
    degree = has_pairs.sum(axis=1)
    degree_order = np.argsort(-degree)
    return degree_order


@njit(cache=True)
def _ensure_long_enough(nodes_by_nargs, nargs):
    while(len(nodes_by_nargs) <= nargs-1):
        nodes_by_nargs.append(List.empty_list(CorgiNodeType))

@njit(cache=True)
def _mod_map_insert(idrec, mod_map, node, arg_ind):
    if(idrec not in mod_map):
        mod_map[idrec] = List.empty_list(node_arg_pair_type)
    mod_map[idrec].append((node, arg_ind))

CorgiNode_List_type = ListType(CorgiNodeType)

@njit(cache=True,locals={})
def _make_corgi_nodes(ms, c, index_map):
    nodes_by_nargs = List.empty_list(CorgiNode_List_type)
    nodes_by_nargs.append(List.empty_list(CorgiNodeType))
    global_modify_map = Dict.empty(u8, node_arg_list_type)

    # Make an identity node (i.e. lit,op=None) so there are always alphas
    for j in range(len(c.vars)):
        t_ids = np.empty((1,),dtype=np.uint16)
        var_inds = np.empty((1,),dtype=np.int64)
        base_var = c.vars[j]
        t_ids[0] = base_var.base_t_id
        var_inds[0] = index_map[i8(base_var.base_ptr)]
        nodes_by_nargs[0].append(node_ctor(ms, t_ids, var_inds,lit=None))

    degree_order = _get_degree_order(c, index_map)

    # print("degree_order", degree_order)

    for distr_conjunct in c.distr_dnf:
        for j, var_ind in enumerate(degree_order):
            var_conjuct = distr_conjunct[var_ind]

            for lit in var_conjuct:
                nargs = len(lit.base_var_ptrs)
                _ensure_long_enough(nodes_by_nargs, nargs)
                # print("A")
                t_ids = np.empty((nargs,),dtype=np.uint16)
                var_inds = np.empty((nargs,),dtype=np.int64)
                for i, base_var_ptr in enumerate(lit.base_var_ptrs):
                    var_inds[i] = index_map[i8(base_var_ptr)]
                    base_var = cast(base_var_ptr, VarType)
                    t_id = base_var.base_t_id
                    t_ids[i] = t_id
                # print("B")
                # print("t_ids", t_ids)
                node = node_ctor(ms, t_ids, var_inds, lit)
                nodes_by_nargs[nargs-1].append(node)
                # print("<< aft", lit.op.head_var_ptrs)
                for i, hi, in enumerate(lit.op.head_infos):
                    head_var = cast(hi.var_ptr, VarType)
                    arg_ind = np.min(np.nonzero(lit.base_var_ptrs==i8(head_var.base_ptr))[0])
                    t_id = t_ids[arg_ind]
                    # print("START")
                    for d_offset in head_var.deref_infos:
                        idrec1 = encode_idrec(u2(t_id),0,u1(d_offset.a_id))
                        _mod_map_insert(idrec1, global_modify_map, node, arg_ind)

                        t_id = d_offset.t_id
                        idrec2 = encode_idrec(u2(t_id),0,0)
                        _mod_map_insert(idrec2, global_modify_map, node, arg_ind)
                                
    return nodes_by_nargs, global_modify_map

optional_node_mem_type = types.optional(NodeIOType)

@njit(cache=True)
def arr_is_unique(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if(i != j and arr[i]==arr[j]):
                return False
    return True


@njit(cache=True)
def build_corgi_graph(ms, c):
    # Build a map from base var ptrs to indicies
    index_map = Dict.empty(i8, i8)
    for i, v in enumerate(c.vars):
        index_map[i8(v.base_ptr)] = i

    # Reorganize the dnf into a distributed dnf with one dnf per var
    if(not c.has_distr_dnf):
        build_distributed_dnf(c,index_map)
    
    # Make all of the CORGI nodes
    nodes_by_nargs, global_modify_map = \
         _make_corgi_nodes(ms, c, index_map)

    global_t_id_root_map = Dict.empty(u2, NodeIOType)
    var_end_join_ptrs = np.zeros((len(c.vars),len(c.vars)),dtype=np.int64)
    var_end_nodes = Dict.empty(i8,CorgiNodeType)
    var_root_nodes = Dict.empty(i8,CorgiNodeType)
    n_nodes = 0
    total_weight = 0

    # Link nodes together. 'nodes_by_nargs' should already be ordered
    # so that alphas are before 2-way, 3-way, etc. betas. 
    for i, nodes in enumerate(nodes_by_nargs):
        for node in nodes:
            # print("node", node.lit)
            inputs = List.empty_list(NodeIOType)
            for j, var_ind in enumerate(node.var_inds):
                # print("ind", ind)
                if(var_ind in var_end_nodes):
                    # Set the .inputs of this node to be the NodeIOs from the
                    #   .outputs of the last node in the graph to check against 'var_ind' 
                    e_node = var_end_nodes[var_ind]
                    om = e_node.outputs[np.min(np.nonzero(e_node.var_inds==var_ind)[0])]
                    inputs.append(om)
                    node.upstream_node_ptr = cast(e_node, i8)
                    # print("wire", var_ind, e_node.lit, '[', np.min(np.nonzero(e_node.var_inds==var_ind)[0]),']', "->",  node.lit, "[", j, "]")
                else:

                    t_id = node.t_ids[j]
                    if(t_id not in global_t_id_root_map):
                        root = new_NodeIO()
                        root.is_root = True

                        global_t_id_root_map[t_id] = root
                    root = global_t_id_root_map[t_id]

                    inputs.append(root)
                    var_root_nodes[var_ind] = node

            node.inputs = inputs

            if(len(inputs) == 2):
                p1, p2 = inputs[0].parent_node_ptr, inputs[1].parent_node_ptr
                node.upstream_same_parents = (p1 != 0 and p1 == p2)
                if(node.upstream_same_parents):
                    parent = cast(p1, CorgiNodeType)
                    node.upstream_aligned = np.all(node.var_inds==parent.var_inds)   

                # Fill in end joins
                vi_a, vi_b = node.var_inds[0],node.var_inds[1]
                var_end_join_ptrs[vi_a, vi_b] = cast(node, i8)
                var_end_join_ptrs[vi_b, vi_a] = cast(node, i8)

                # print("Assign end join:", vi_a if vi_a < vi_b else vi_b,
                #      vi_b if vi_a < vi_b else vi_a, node.lit)
                

            # Short circut the input to the output for identity nodes
            if(node.lit is None):
                node.outputs = node.inputs
                total_weight += 1.0 
            else:
                total_weight += node.lit.weight 
            

            n_nodes += 1
            

            # Make this node the new end node for the vars it takes as inputs
            for var_ind in node.var_inds:
                var_end_nodes[var_ind] = node
            
    return corgi_graph_ctor(ms, c, nodes_by_nargs, n_nodes, var_root_nodes, var_end_nodes,
              var_end_join_ptrs, global_modify_map, global_t_id_root_map, total_weight)


# -----------------------------------------------------------------------
# : update_graph()

# -------------------------------------------
# : parse_change_events()

@generated_jit(cache=True)
def setitem_buffer(st, buffer_name, k, val):
    SentryLiteralArgs(['buffer_name']).for_function(setitem_buffer).bind(st, buffer_name, k, val)
    def impl(st, buffer_name, k, val):
        buffer = lower_getattr(st, buffer_name)
        buff_len = len(buffer)
        if(k >= buff_len):
            new_len = max(k+1, 2*buff_len)
            new_buffer = np.empty((new_len,), dtype=np.int64)
            new_buffer[:buff_len] = buffer
            lower_setattr(st, buffer_name, new_buffer) 
            buffer = new_buffer
        buffer[k] = val        
    return impl
    

@njit(cache=True)
def node_memory_insert_match_buffers(self, k, idrec, ind):
    buff_len = len(self.match_idrecs_buffer)
    if(k >= buff_len):
        expand = max(k-buff_len, buff_len)

        new_idrecs_buffer = np.empty(expand+buff_len, dtype=np.uint64)
        new_idrecs_buffer[:buff_len] = self.match_idrecs_buffer
        self.match_idrecs_buffer = new_idrecs_buffer

        new_inds_buffer = np.empty(expand+buff_len, dtype=np.int64)
        new_inds_buffer[:buff_len] = self.match_inp_inds_buffer
        self.match_inp_inds_buffer = new_inds_buffer

        
    self.match_idrecs_buffer[k] = idrec
    self.match_inp_inds_buffer[k] = ind


from cre.change_event import accumulate_change_events

@njit(cache=True,locals={"t_id" : u2, "f_id":u8, "a_id":u1})
def parse_change_events(r_graph):

    # Extract values used below
    global_modify_map = r_graph.global_modify_map
    global_t_id_root_map = r_graph.global_t_id_root_map
    change_queue = r_graph.memset.change_queue

    
    for t_id, root_mem in global_t_id_root_map.items():
        root_mem.change_inds = root_mem.change_buffer[:0]
        root_mem.remove_inds = root_mem.remove_buffer[:0]


    change_events = accumulate_change_events(change_queue, r_graph.change_head, -1)

    zeros = List.empty_list(u1)
    zeros.append(u1(0))

    for change_event in change_events:
        # print("change_event", change_event)
        t_id, f_id, _ = decode_idrec(change_event.idrec)

        # Add this idrec to change_inds of root nodes
        if(t_id not in global_t_id_root_map): continue
        root_mem = global_t_id_root_map[t_id]

        # Modify Case
        if(change_event.was_modified):
            for a_id in change_event.a_ids:
                # Add this idrec to relevant deref idrecs
                idrec_pattern = encode_idrec(t_id, 0, a_id)
                node_arg_pairs = global_modify_map.get(idrec_pattern,None)
                if(node_arg_pairs is not None):
                    for (node,arg_ind) in node_arg_pairs:
                        # print("added: ", node.lit, 't_id=', t_id, 'f_id=', f_id, 'a_id=', a_id)
                        node.modify_idrecs[arg_ind].add(encode_idrec(t_id, f_id, a_id))

        # Declare/Modify Case
        if(change_event.was_declared or change_event.was_modified):
            k = len(root_mem.change_inds)
            setitem_buffer(root_mem, 'change_buffer', k, i8(f_id))
            root_mem.change_inds = root_mem.change_buffer[:k+1]

            if(change_event.was_declared):
                idrec = encode_idrec(t_id,f_id,0)# if(a_id != RETRACT) else u8(0)
                root_mem.idrecs_to_inds[idrec] = i8(f_id)

                node_memory_insert_match_buffers(root_mem, i8(f_id), idrec, i8(f_id))
                if(i8(f_id) >= len(root_mem.match_idrecs)):
                    root_mem.match_idrecs = root_mem.match_idrecs_buffer[:i8(f_id)+1]
                    root_mem.match_inp_inds = root_mem.match_inp_inds_buffer[:i8(f_id)+1]
        else:            
            k = len(root_mem.remove_inds)
            setitem_buffer(root_mem, 'remove_buffer', k, i8(f_id))
            root_mem.remove_inds = root_mem.remove_buffer[:k+1]

            # print("RETRACT", t_id,f_id, root_mem.remove_inds)

            node_memory_insert_match_buffers(root_mem, i8(f_id), u8(0), i8(0))
                
    # print("END ROOT")

    r_graph.change_head = change_queue.head

# -------------------------------------------
# : update_input_changes()

deref_record_field_dict = {
    # Weak Pointers to the Dict(i8,u1)s inside deref_depends
    "parent_ptrs" : i8[::1], 
    "arg_ind" : i8,
    "base_idrec" : u8,
    "was_successful" : u1,
}

DerefRecord, DerefRecordType = define_structref("DerefRecord", deref_record_field_dict)

@njit(cache=True)
def invalidate_head_ptr_rec(rec):
    r_ptr = cast(rec, i8)
    for ptr in rec.parent_ptrs:
        parent = _dict_from_ptr(dict_i8_u1_type, ptr)
        del parent[r_ptr]
        _decref_structref(rec)

@njit(wptr_t(deref_dep_typ, u8, ptr_t),cache=True)
def make_deref_record_parent(deref_depends, idrec, r_ptr):
    p = deref_depends.get(idrec,None)
    if(p is None): 
        p = deref_depends[idrec] = Dict.empty(ptr_t,u1)
    p[r_ptr] = u1(1)
    return cast(p, i8)

@njit(i8(deref_info_type,i8),inline='never',cache=True)
def deref_once(deref, inst_ptr):
    if(deref.type == u1(DEREF_TYPE_ATTR)):
        return _ptr_to_data_ptr(inst_ptr)
    else:
        return _list_base_from_ptr(inst_ptr)


# @njit(i8(CorgiNodeType, u2, u8, deref_info_type[::1]),cache=True)
@njit(cache=True)
def resolve_head_ptr(self, arg_ind, base_t_id, f_id, deref_infos):
    '''Try to get the head_ptr of 'f_id' in input 'arg_ind'. Inject a DerefRecord 
         regardless of the result Keep in mind that a head_ptr is the pointer
         to the address where the data is stored not the data itself.
    '''
    memset = cast(self.memset_ptr, MemSetType)
    facts = cast(memset.facts[base_t_id], VectorType)
    if(len(deref_infos) > 0):
        inst_ptr = facts.data[f_id]
        if(len(deref_infos) > 1):
            rel_idrecs = np.empty(len(deref_infos)-1, dtype=np.uint64)
            for k in range(len(deref_infos)-1):
                if(inst_ptr == 0): break;
                deref = deref_infos[k]
                data_ptr = deref_once(deref, inst_ptr)
                # print(f"{deref.type} inst_ptr {inst_ptr} -> {data_ptr+deref.offset}")
                # print("data_ptr", k, data_ptr)
                # Note: This won't retreive an idrec for lists 
                rel_idrecs[k] = _load_ptr(u8, data_ptr)
                inst_ptr = _load_ptr(i8, data_ptr+deref.offset)
                # print("inst_ptr", k, inst_ptr)

            # Inject a deref record so that we can track changes to intermediate facts
            parent_ptrs = rel_idrecs.astype(i8) 
            r = DerefRecord(parent_ptrs, arg_ind, encode_idrec(base_t_id,f_id,0), inst_ptr != 0) #7us
            r_ptr = _ptr_from_struct_incref(r)
            for i, idrec in enumerate(rel_idrecs): #21 us
                parent_ptrs[i] = i8(make_deref_record_parent(self.deref_depends, idrec, r_ptr))            

        if(inst_ptr != 0):
            deref = deref_infos[-1]
            data_ptr = deref_once(deref, inst_ptr)
            # print(f"{deref.type} inst_ptr {inst_ptr} -> {data_ptr+deref.offset}")
            # print("data_ptr", -1, data_ptr,data_ptr+deref.offset)
            return data_ptr+deref.offset
        else:
            return 0
    else:
        return _get_array_raw_data_ptr(facts.data) + (f_id * 8) #assuming 8 byte ptrs

# @njit(void(CorgiNodeType, i8,u8,u1),locals={"f_id" : u8}, cache=True)
@njit(locals={"f_id" : u8, "a_id" : u8}, cache=True)
def validate_head_or_retract(self, arg_ind, idrec, head_ptrs, r):
    '''Update the head_ptr dictionaries by following the deref
     chains of DECLARE/MODIFY changes, and make retractions
    for an explicit RETRACT or a failure in the deref chain.'''

    t_id, f_id, a_id = decode_idrec(idrec)
    is_valid = True
    if(a_id != RETRACT):
        for i in range(r.start,r.end):
            head_var = cast(self.op.head_infos[i].var_ptr, VarType)
            deref_infos = head_var.deref_infos
            head_ptr = resolve_head_ptr(self, arg_ind, t_id, f_id, deref_infos)
            if(head_ptr == 0): 
                is_valid=False;
                break; 
            head_ptrs[i-r.start] = head_ptr

    else:
        is_valid = False
    return is_valid
            


# NOTE: locals={'ind' : i8} necessary because typing bug w/ dict.getitem
@njit(cache=True, locals={'ind' : i8})
def update_input_changes(self):
    ''' 
    Updated the input_state structs for each fact in each input and 
    populates for each input the set of removed_inds, changed_inds, and 
    unchanged_inds that are used decide which facts or fact pairs need to 
    be rechecked in update_matches(). removed_inds consists of those facts  
    that were removed upstream or that are now invalid for this node
    because of a head validation failure (i.e. cannot dereference attribute 
    chains in one of the vars). changed_inds corresponds to facts that
    were newly inserted upstream, or to facts that had an attribute modified 
    that is relevant to this node, or to facts that have a new successful
    validation in the current match cylce. unchanged_inds comprises the rest.
    NOTE: Might not actually need to fill 'removed_inds'
    '''
    for i, inp in enumerate(self.inputs):
        self.inp_widths[i] = len(inp.match_idrecs)

        # Extract values used below
        head_ptr_buffers_i = self.head_ptr_buffers[i]
        input_state_buffers_i = self.input_state_buffers[i]
        inds_change_buffers_i = self.inds_change_buffers[i]
        head_range_i = self.op.head_ranges[i]
        modify_idrecs_i = self.modify_idrecs[i]

        # Ensure various buffers are large enough (3 us).
        curr_len, curr_w = head_ptr_buffers_i.shape
        if(self.inp_widths[i] > curr_len):
            new_len = max(self.inp_widths[i], curr_len+min(64,curr_len))
            new_head_ptr_buff = np.empty((new_len, curr_w),dtype=np.int64)
            new_head_ptr_buff[:curr_len] = head_ptr_buffers_i
            head_ptr_buffers_i = self.head_ptr_buffers[i] = new_head_ptr_buff

            new_input_state_buff = np.zeros(new_len,dtype=input_state_type)
            new_input_state_buff[:curr_len] = input_state_buffers_i
            input_state_buffers_i = self.input_state_buffers[i] = new_input_state_buff

            new_inds_chng_buff = np.empty(new_len,dtype=np.int64)
            new_inds_chng_buff[:curr_len] = inds_change_buffers_i
            inds_change_buffers_i = self.inds_change_buffers[i] = new_inds_chng_buff


        # Clear input_states of properties only meant to last one cycle
        for k in range(len(inp.match_idrecs)):
            input_state = input_state_buffers_i[k]
            input_state.recently_invalid = False
            input_state.recently_inserted = False
            input_state.recently_modified = False
            input_state.changed = False

        # NOTE: Keep prints for debugging
        # print("i :", i)
        # print("inp", np.array([decode_idrec(x)[1] for x in inp.match_idrecs]))
        # print("insrt", np.array([decode_idrec(inp.match_idrecs[x])[1] for x in  inp.change_inds]))
        # print("remove", np.array([decode_idrec(inp.match_idrecs[x])[1] for x in  inp.remove_inds]))
        # print("modify", np.array([decode_idrec(x)[1] for x in modify_idrecs_i.data[:modify_idrecs_i.head] ]))
        
        n_rem = 0
        # Update input_states with any upstream removals
        for ind in inp.remove_inds:
            input_state = input_state_buffers_i[ind]

            # Only counts as a change if creates a new removal
            if(input_state.is_valid):
                n_rem += 1
                # input_state.true_was_nonzero = False
                input_state.recently_invalid = True
            input_state.is_valid = False
            input_state.changed = True

        change_inds = np.empty(len(inp.change_inds)+len(modify_idrecs_i),dtype=np.int64)
        c = len(inp.change_inds)
        for ind in inp.change_inds:
            input_state_buffers_i[ind].changed = True
        change_inds[:c] = inp.change_inds

        # Add to change_inds any modifies specifically routed to this node. 
        mod_cutoff = c
        for k in range(modify_idrecs_i.head):
            t_id, f_id, a_id = decode_idrec(modify_idrecs_i.data[k])
            idrec = encode_idrec(t_id, f_id, 0)
            ind = inp.idrecs_to_inds.get(idrec,-1)            
            if(ind != -1):
                input_state = input_state_buffers_i[ind]
                # Don't add the modify if fact changed/removed upstream.
                if(not input_state.changed):
                    change_inds[c] = ind; c += 1;
                    input_state.changed = True
                    
        # Buffer is consumed -> set head to zero.
        modify_idrecs_i.head = 0

        change_inds = change_inds[:c]
        
        # For each changed_ind apply deref chains associated with the base var 
        #  for this input. Mark as a newly inserted/removed input as appropriate.
        for k, ind in enumerate(change_inds):
            idrec = inp.match_idrecs[ind]

            is_modify = (k >= mod_cutoff)
            input_state = input_state_buffers_i[ind]

            # Check if the deref chain(s) are valid.
            head_ptrs = head_ptr_buffers_i[ind]
            was_valid = input_state.is_valid
            is_valid = validate_head_or_retract(self, i, idrec, head_ptrs, head_range_i)
            recently_inserted = (~was_valid & is_valid)
            recently_modified = (is_modify & is_valid)

            # Assign changes to to the input_state struct.
            input_state.idrec = idrec
            if(was_valid and not is_valid):
                n_rem += 1
                input_state.recently_invalid = True    
            input_state.recently_inserted = recently_inserted
            input_state.recently_modified = recently_modified
            input_state.is_valid = is_valid

            # NOTE: Keep print for debugging
            # print(self.lit, "is_valid", decode_idrec(idrec)[1], is_valid, input_state.recently_inserted, input_state.recently_modified)

        
        # Fill the insert, remove, and unchanged inds arrays (<1 us).
        c, r, u = 0, 0, 0
        for j in range(len(inp.match_idrecs)):
            input_state = input_state_buffers_i[j]
            if(input_state.recently_inserted or input_state.recently_modified):
                inds_change_buffers_i[n_rem+c] = j; c += 1;
            elif(input_state.recently_invalid):
                inds_change_buffers_i[r] = j; r += 1;

        for j in range(len(inp.match_idrecs)):
            input_state = input_state_buffers_i[j]
            if(not (input_state.recently_inserted or input_state.recently_modified)
               and not input_state.recently_invalid):
                inds_change_buffers_i[r+c+u] = j; u += 1;

        self.removed_inds[i] = inds_change_buffers_i[:r]
        self.changed_inds[i] = inds_change_buffers_i[r:r+c]
        self.unchanged_inds[i] = inds_change_buffers_i[r+c:r+c+u]

        # NOTE: Keep prints for debugging
        # print("removed_inds_i: ", self.removed_inds[i])
        # print("changed_inds_i: ", self.changed_inds[i])
        # print("unchanged_inds_i: ", self.unchanged_inds[i])
        # print()

# -------------------------------------------
# : update_matches()


@njit(cache=True)
def resize_truth_table(self):
    ''' Ensures that the truth table is large enough for the widths of 
        the inputs.
    '''
    s0,s1 = self.truth_table.shape
    expand0 = max(0,self.inp_widths[0]-s0)
    expand1 = max(0,self.inp_widths[1]-s1)
    if(expand0 and expand0):
        expand0 = max(s0,expand0)
        expand1 = max(s1,expand1)

        new_truth_table = np.empty((s0+expand0, s1+expand1),dtype=np.uint8)
        new_truth_table[:s0,:s1] = self.truth_table
        new_truth_table[s0:] = u1(0)
        new_truth_table[:, s1:] = u1(0)
        self.truth_table = new_truth_table

    elif(expand0):
        expand0 = max(s0,expand0)

        new_truth_table = np.empty((s0+expand0, s1),dtype=np.uint8)
        new_truth_table[:s0,:s1] = self.truth_table
        new_truth_table[s0:] = u1(0)
        self.truth_table = new_truth_table

    elif(expand1):
        expand1 = max(s1,expand1)
        new_truth_table = np.empty((s0,s1+expand1),dtype=np.uint8)
        new_truth_table[:s0,:s1] = self.truth_table
        new_truth_table[:, s1:] = u1(0)
        self.truth_table = new_truth_table

            
from cre.utils import _func_from_address
match_heads_f_type = types.FunctionType(u1(i8[::1],))

u8_i8_dict_type = DictType(u8,i8)



@njit(cache=True)
def beta_matches_to_str(matches):
    s = ""
    for match0, others_ptr in matches.items():
        others_str = ""
        t_id0, f_id0, a0 = decode_idrec(match0)
        others = _dict_from_ptr(u8_i8_dict_type, others_ptr)
        for match1, ok in others.items():
            t_id1, f_id1, _ = decode_idrec(match1)
            if(ok): others_str += f"({t_id1},{f_id1}),"
            
        s +=f"({t_id0},{f_id0}) : [{others_str[:-1] if len(others_str) > 0 else ''}]\n"

    return s


@njit(cache=True)
def alpha_matches_to_str(matches):
    s = "["
    for match0, others_ptr in matches.items():
        t_id0, f_id0, a0 = decode_idrec(match0)
        s +=f"({t_id0},{f_id0}),"

    return s[:-1] + "]"

@njit(cache=True, inline='always')
def _upstream_true(u_tt, aligned, pind_i, pind_j):
    if(aligned):
        return u_tt[pind_i, pind_j]
    else:
        return u_tt[pind_j, pind_i]


@njit(cache=True, inline='always')
def _update_truth_table(tt, is_match, match_inp_inds, inp_state_i, inp_state_j):
    # Set the truth table and modify 'true_count' as appropriate
    ind0, ind1 = match_inp_inds[0], match_inp_inds[1]
    was_match = tt[ind0, ind1]
    tt[ind0, ind1] = u1(is_match)
    count_diff = i8(is_match) - i8(was_match)
    inp_state_i.true_count += count_diff
    inp_state_j.true_count += count_diff


@njit(types.void(CREFuncType, u8, i8[::1]), cache=True, locals={"i" : u8,"k":u8, "j": u8})
def set_heads_from_data_ptrs(cf, i, data_ptrs):
    for k, j in enumerate(range(cf.head_ranges[i].start, cf.head_ranges[i].end)):
        hi = cf.head_infos[j]
        if(hi.ref_kind==REFKIND_UNICODE): 
            # _decref_structref(_load_ptr(unicode_type, hi.head_data_ptr))
            _incref_structref(_load_ptr(unicode_type, data_ptrs[k]))
        elif(hi.ref_kind==REFKIND_STRUCTREF):
            # _decref_ptr(_load_ptr(i8, hi.head_data_ptr))
            _incref_ptr(_load_ptr(i8, data_ptrs[k]))

        _memcpy(hi.head_data_ptr, data_ptrs[k], hi.head_size)

@njit(cache=True)
def update_alpha_matches(self):
    
    # Get the best call_self() implmentation that ignores dereferencing
    op = self.op
    call_self_func = get_best_call_self(op,True)
    tt = self.truth_table
    negated = self.lit.negated    
    input_state_buffers_i = self.input_state_buffers[0]
    head_ptrs_i = self.head_ptr_buffers[0]
    changed_inds_i = self.changed_inds[0]
            
    # Go through all of the changed candidates 
    for ind_i in changed_inds_i:
        inp_state_i = input_state_buffers_i[ind_i]

        # Check if the op matches this candidate
        set_heads_from_data_ptrs(op, 0, head_ptrs_i[ind_i])
        is_match = (call_self_func(op) == CFSTATUS_TRUTHY) ^ negated
        inp_state_i.true_count = i8(is_match)



@njit(cache=True)
def update_beta_matches(self):
    
    # Make sure the truth_table is big enough.
    resize_truth_table(self)    
    
    # Get the best call_self() implmentation that ignores dereferencing
    op = self.op
    call_self_func = get_best_call_self(op,True)

    # Buffers that will be loaded with ptrs, indrecs, or inds of candidates.
    # match_inp_ptrs = np.zeros(len(self.op.head_infos),dtype=np.int64)
    match_inp_inds = np.zeros(len(self.var_inds),dtype=np.int64)
    tt = self.truth_table
    negated = self.lit.negated

    # Loop through the 2 variables for this node.
    for i in range(2):
        # Extract various things used below
        head_ptrs_i = self.head_ptr_buffers[i]
        changed_inds_i = self.changed_inds[i]
    
        # For every change in a candidate to the first variable we'll 
        #  update against every other possible candidate for the second.
        #  i.e. for i=0 we fill  whole rows in our truth table.
        #  [?--?--?]
        #  [X--X--X]
        #  [?--?--?]
        
        # For every change in a candidate to the second variable we'll 
        #  update against only the unchanged candidates for the first.
        #  i.e. for i=1 we fill columns in our truth table, but skip
        #   cells that were already updated.
        #  [?--X--?]
        #  [?--?--?]
        #  [?--X--?]

        # When i=1, j=0 and vise versa
        j = 1 if i == 0 else 0
        
        # Extract various things used below
        head_ptrs_j = self.head_ptr_buffers[j]
        inp_buffers_i = self.input_state_buffers[i]
        inp_buffers_j = self.input_state_buffers[j]
        pinds_i = self.inputs[i].match_inp_inds
        pinds_j = self.inputs[j].match_inp_inds
        aligned = self.upstream_aligned
        same_parent = self.upstream_same_parents
        
        # Go through all of the changed candidates
        for ind_i in changed_inds_i:
            # Extract various things associated with 'ind_i'
            inp_state_i  = inp_buffers_i[ind_i]
            pind_i = pinds_i[ind_i]
            
            # Fill in the 'i' part of ptrs, idrecs, inds 
            set_heads_from_data_ptrs(op, i, head_ptrs_i[ind_i])
            match_inp_inds[i] = ind_i

            # NOTE: Sections below have lots of repeated code. Couldn't find a way to inline
            #  without incurring a ~4x slowdown.
            
            # If there is a beta node upstream of to this one that shares the same
            #  variables then we'll need to make sure we also check its truth table.
            if(self.upstream_same_parents):
                upstream_node = cast(self.upstream_node_ptr, CorgiNodeType)
                u_tt = upstream_node.truth_table

                if(j > i):
                    # Update the whole row/column
                    for ind_j in range(self.inp_widths[j]):
                        input_state_j = inp_buffers_j[ind_j]
                        if(not _upstream_true(u_tt, aligned, pind_i, pinds_j[ind_j])):
                            match_inp_inds[j] = ind_j
                            _update_truth_table(tt, u1(0), match_inp_inds, inp_state_i, input_state_j)    
                            continue

                        if(input_state_j.is_valid):
                            match_inp_inds[j] = ind_j

                            set_heads_from_data_ptrs(op, j, head_ptrs_j[ind_j])
                            is_match = (call_self_func(op) == CFSTATUS_TRUTHY) ^ negated
                        else:
                            is_match = u1(0)
                        _update_truth_table(tt, is_match, match_inp_inds, inp_state_i, input_state_j)
                else:
                    # Check just the unchanged parts, so to avoid repeat checks 
                    for ind_j in self.unchanged_inds[j]:
                        input_state_j = inp_buffers_j[ind_j]
                        if(not _upstream_true(u_tt, aligned, pind_i, pinds_j[ind_j])):
                            match_inp_inds[j] = ind_j
                            _update_truth_table(tt, u1(0), match_inp_inds, inp_state_i, input_state_j)    
                            continue

                        if(input_state_j.is_valid):
                            match_inp_inds[j] = ind_j
                            set_heads_from_data_ptrs(op, j, head_ptrs_j[ind_j])
                            is_match = (call_self_func(op) == CFSTATUS_TRUTHY) ^ negated
                        else:
                            is_match = u1(0)
                        _update_truth_table(tt, is_match, match_inp_inds, inp_state_i, input_state_j)

            # If no 'upstream_same_parents' then just update the truth table 
            #  with the match values for all relevant pairs.
            else:
                if(j > i):
                    # Update the whole row/column
                    for ind_j in range(self.inp_widths[j]):
                        input_state_j = inp_buffers_j[ind_j]
                        if(input_state_j.is_valid):
                            match_inp_inds[j] = ind_j
                            set_heads_from_data_ptrs(op, j, head_ptrs_j[ind_j])
                            is_match = (call_self_func(op) == CFSTATUS_TRUTHY) ^ negated
                        else:
                            is_match = u1(0)
                        _update_truth_table(tt, is_match, match_inp_inds, inp_state_i, input_state_j)
                else:
                    # Check just the unchanged parts, so to avoid repeat checks 
                    for ind_j in self.unchanged_inds[j]:
                        input_state_j = inp_buffers_j[ind_j]
                        if(input_state_j.is_valid):
                            match_inp_inds[j] = ind_j
                            set_heads_from_data_ptrs(op, j, head_ptrs_j[ind_j])
                            is_match = (call_self_func(op) == CFSTATUS_TRUTHY) ^ negated
                        else:
                            is_match = u1(0)
                        _update_truth_table(tt, is_match, match_inp_inds, inp_state_i, input_state_j)

# -------------------------------------------
# : update_output_changes()

@njit(cache=True)
def update_output_changes(self):
    # Update each of (the at most 2) outputs (one for each Var).
    for i, out_i in enumerate(self.outputs):
        change_ind = 0
        remove_ind = 0
        input_state_buffers_i = self.input_state_buffers[i]
        idrecs_to_inds_i = out_i.idrecs_to_inds
        
        # Update each slot k for match candidates for the ith Var.
        #  Performance Note: We almost always need to recheck every input
        #  for at least one var so gating with true_ever_nonzero instead of
        #  iterating over changed_inds / removed_inds makes sense.
        for k in range(self.inp_widths[i]):
            # Extract t_id, f_id, a_id for the input at k.
            input_state_k = input_state_buffers_i[k]
            idrec_k = input_state_k.idrec

            # Determine if we've ever found a match for this candidate
            true_is_nonzero = (input_state_k.true_count != 0)
            input_state_k.true_ever_nonzero |= true_is_nonzero

            if(input_state_k.true_ever_nonzero):
                # If a candidate is invalidated on this cycle or goes 
                #  from matching to unmatching mark as removed.
                if( input_state_k.recently_invalid or
                   (input_state_k.true_was_nonzero and not true_is_nonzero)):

                    output_ind = input_state_k.output_ind
                    setitem_buffer(out_i, "remove_buffer", remove_ind, output_ind)
                    node_memory_insert_match_buffers(out_i, output_ind, u8(0), k)
                    out_i.match_holes.add(output_ind)
                    
                    idrecs_to_inds_i[idrec_k] = -1
                    input_state_k.output_ind = -1
                    remove_ind += 1

                # If a candidate goes from unmatching to matching or was 
                #  recently inserted and matches then mark as change.
                elif(not input_state_k.true_was_nonzero and true_is_nonzero or
                    (true_is_nonzero and input_state_k.recently_inserted)):

                    if(len(out_i.match_holes) > 0):
                        output_ind = out_i.match_holes.pop()
                    else:
                        output_ind = out_i.width
                        out_i.width += 1

                    setitem_buffer(out_i, "change_buffer", change_ind, output_ind)
                    node_memory_insert_match_buffers(out_i, output_ind, idrec_k, k)
                    
                    idrecs_to_inds_i[idrec_k] = output_ind
                    input_state_k.output_ind = output_ind
                    change_ind += 1

            input_state_k.true_was_nonzero = true_is_nonzero
            
        
        # To avoid reallocations the arrays in each output are slices of larger buffers.
        out_i.match_idrecs = out_i.match_idrecs_buffer[:out_i.width]
        out_i.match_inp_inds = out_i.match_inp_inds_buffer[:out_i.width]
        out_i.change_inds = out_i.change_buffer[:change_ind]
        out_i.remove_inds = out_i.remove_buffer[:remove_ind]

        # Note: Keep these print statements for debugging
        # print("i :", i)
        # print("idrec_to_inds", out_i.idrecs_to_inds)
        # print("match_idrecs.f_id", np.array([decode_idrec(x)[1] for x in out_i.match_idrecs]))
        # print("match_inp_inds", out_i.match_inp_inds)
        # print("change_inds.f_id", np.array([decode_idrec(out_i.match_idrecs[x])[1] for x in out_i.change_inds]))
        # print("remove_inds.f_id", np.array([decode_idrec(out_i.match_idrecs[x])[1] for x in out_i.remove_inds]))
        # print(self.truth_table)


@njit(cache=True)
def update_node(self):
    '''Updates a graph node. Updates the node's outputs based on upstream
         change to its inputs or from directly routed modifications. For 
         beta nodes their truth table is updated in the process. Each node
         output (one for each var) holds the current matching idrecs, 
         the their indicies in the node's input, and sets of change_inds,
         and remove_inds to signal how downstream nodes should update based
         on the output changes.'''

    # If the node is an identity node then skip since the input is hardwired to the output
    if(self.op is None): return

    # Note: Keep these print statements for debugging
    # print("-----------------------")
    # print("Update", self.lit)
    

    # Phase 1: Updates input_states and produces changed_inds and unchanged_inds.
    #  also updates the head ptrs of for dereference chains.
    update_input_changes(self)

    # Phase 2: Update true_count / truth_table
    if(len(self.op.head_ranges) == 1):
        #  For Alpha nodes just check the changed_inds. Fills true_count 
        # (i.e. number of consistent match pairs) for each input fact.
        update_alpha_matches(self)
    else:
        #For beta nodes recheck the changed_inds for one var against the
        #   changed_inds and unchanged_inds of the other.
        update_beta_matches(self)

    # Ensure that idrecs for facts with true_count > 0 represented in 
    #  outputs. Track removals/changes for updating downstream nodes. 
    update_output_changes(self)
            




@njit(cache=True)
def update_graph(graph):
    # print("START UP")
    parse_change_events(graph)
    # print("PARSED")
    for lst in graph.nodes_by_nargs:
        for node in lst:
            # print(node)
            update_node(node)        


# ----------------------------------------------------------------------
# : MatchIterator

match_iterator_node_field_dict = {
    # "graph" : CorgiGraphType,
    "node" : CorgiNodeType,
    "associated_arg_ind" : i8,
    "var_ind" : i8,
    "m_node_ind" : i8,
    # "is_exhausted": boolean,
    "curr_ind": i8,
    "idrecs" : u8[::1],
    "other_idrecs" : u8[::1],

    # Indicies of downstream m_nodes on which this m_node depends.
    "dep_m_node_inds" : i8[::1],

    # Pointers to downstream nodes on which this m_node depends
    #  NOTE: the dependant node might not be the .node of the
    #  corresponding dependant m_node. This points to the most
    #  downstream join for a particular var pair. 
    "dep_node_ptrs" : i8[::1],

    # The arg_ind in each node in dep_node_ptrs for the var this
    #  m_node is associated with. 
    "dep_arg_inds" : i8[::1],
}



MatchIterNode, MatchIterNodeType = define_structref("MatchIterNode", match_iterator_node_field_dict, define_constructor=False)

match_iterator_field_dict = {
    "graph" : CorgiGraphType,
    "iter_nodes" : ListType(MatchIterNodeType),
    "is_empty" : types.boolean,
    "iter_started" : types.boolean,
    "output_types" : types.Any
    # "curr_match" : u8[::1],
}
match_iterator_fields = [(k,v) for k,v, in match_iterator_field_dict.items()]


# NOTE: Probably uncessary now that Fact types recover their original type on unboxing.
def gen_match_iter_source(output_types):
    ''' Generates source code for '''
    return f'''import cloudpickle
from numba import njit
from cre.matching import MatchIteratorType, MatchIteratorTypeClass
from cre.utils import cast
output_types = cloudpickle.loads({cloudpickle.dumps(output_types)})
m_iter_type = MatchIteratorTypeClass(output_types)

@njit(m_iter_type(MatchIteratorType),cache=True)
def specialize_m_iter(self):
    return cast(self, m_iter_type)
    '''

MATCH_ITER_FACT_KIND = 0
MATCH_ITER_PTR_KIND = 1
MATCH_ITER_IDREC_KIND = 2
match_iter_kinds = {
    "fact" : MATCH_ITER_FACT_KIND,
    "ptr" : MATCH_ITER_PTR_KIND,
    "idrec" : MATCH_ITER_IDREC_KIND,
}


class MatchIterator(structref.StructRefProxy):
    ''' '''
    __slots__ = ('kind', 'recover_types' , 'output_types', 'proxy_types')
    m_iter_type_cache = {}
    def __new__(cls, ms, conds, kind="fact", recover_types=False):
        # Make a generic MatchIterator (reuses graph if conds already has one)
        generic_m_iter = get_match_iter(ms, conds)
        kind = match_iter_kinds[kind]
        var_base_types = conds.var_base_types

        if(kind == MATCH_ITER_FACT_KIND and recover_types):
            #Cache 'output_types' and 'specialized_m_iter_type'
            
            if(var_base_types not in cls.m_iter_type_cache):
                hash_code = unique_hash_v([var_base_types])
                if(not source_in_cache('MatchIterator', hash_code)):
                    output_types = types.TypeRef(types.Tuple([types.TypeRef(x) for x in conds.var_base_types]))
                    source = gen_match_iter_source(output_types)
                    source_to_cache('MatchIterator', hash_code, source)
                l = import_from_cached('MatchIterator', hash_code, ['specialize_m_iter', 'output_types'])
                output_types, specialize_m_iter = cls.m_iter_type_cache[var_base_types] = l['output_types'], l['specialize_m_iter']
            else:
                output_types, specialize_m_iter  = cls.m_iter_type_cache[var_base_types]

            # Specialize the match iter so that it outputs conds.var_base_types 
            self = specialize_m_iter(generic_m_iter)
            self.output_types = output_types#tuple([types.TypeRef(x) for x in conds.var_base_types])
        else:
            self = generic_m_iter
            if(kind == MATCH_ITER_FACT_KIND):
                self.proxy_types = [x._fact_proxy for x in var_base_types]                

        self.kind = kind
        self.recover_types = recover_types
        return self
            
        
    def __next__(self):
        # Note: try/except appears to only be necessary for profiling quirk.
        if(match_iter_is_empty(self)): raise StopIteration()
        try:
            if(self.kind == MATCH_ITER_FACT_KIND):
                if(self.recover_types):
                    empty, out = match_iter_next(self)
                else:
                    empty, ptrs = match_iter_next_ptrs(self)
                    if(empty): raise StopIteration()

                    arr = []
                    for ptr, proxy_typ in zip(ptrs, self.proxy_types):
                        mi = ptr_to_meminfo(ptr)
                        instance = super(StructRefProxy,proxy_typ).__new__(proxy_typ)
                        instance._type = proxy_typ
                        instance._meminfo = mi
                        arr.append(instance)
                    return arr
            elif(self.kind == MATCH_ITER_PTR_KIND):
                empty, out = match_iter_next_ptrs(self)
            else:
                empty, out = match_iter_next_idrecs(self)
            if(empty): raise StopIteration()
            return out

        # Catch system errors. Needed for Cprofiling to work
        except SystemError:
            raise StopIteration()



    def __iter__(self):
        return self


@structref.register
class MatchIteratorTypeClass(CastFriendlyMixin, types.StructRef):

    def __new__(cls, output_types=None):
        self = super().__new__(cls)
        self.output_types = output_types
        if(output_types is None):
            field_dict = match_iterator_field_dict
        else:
            field_dict= {**match_iterator_field_dict,
                "output_types": types.LiteralType(output_types)
            }
        types.StructRef.__init__(self,[(k,v) for k,v in field_dict.items()])
        self.name = repr(self)
        return self

    def __init__(self,*args,**kwargs):
        pass

    def __str__(self):
        if(self.output_types is None):
            return "MatchIteratorType"
        else:
            return f"MatchIteratorType[{self.output_types}]"

    __repr__ = __str__
        


    # def preprocess_fields(self, fields):
    #     return tuple((name, types.unliteral(typ)) for name, typ in fields)


define_boxing(MatchIteratorTypeClass, MatchIterator)
MatchIteratorType = MatchIteratorTypeClass()

# Allow any specialization of MatchIteratorTypeClass to be upcast to MatchIteratorType
@lower_cast(MatchIteratorTypeClass, MatchIteratorType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty)


@njit(MatchIteratorType(MatchIteratorType, CorgiGraphType),cache=True)
def copy_match_iter(m_iter, graph):
    '''Makes a copy a prototype match iterator'''
    m_iter_nodes = List.empty_list(MatchIterNodeType)
    for i,m_node in enumerate(m_iter.iter_nodes):
        new_m_node = new(MatchIterNodeType)
        # new_m_node.graph = m_node.graph
        new_m_node.node = m_node.node
        new_m_node.var_ind = m_node.var_ind
        new_m_node.m_node_ind = m_node.m_node_ind
        new_m_node.associated_arg_ind = m_node.associated_arg_ind
        new_m_node.dep_m_node_inds = m_node.dep_m_node_inds
        new_m_node.dep_arg_inds = m_node.dep_arg_inds
        new_m_node.dep_node_ptrs = m_node.dep_node_ptrs
        new_m_node.curr_ind = m_node.curr_ind
        if(m_node.curr_ind != -1):
            new_m_node.idrecs = m_node.idrecs
        m_iter_nodes.append(new_m_node)

    new_m_iter = new(MatchIteratorType)
    new_m_iter.graph = graph 
    new_m_iter.iter_nodes = m_iter_nodes 
    new_m_iter.is_empty = m_iter.is_empty
    new_m_iter.iter_started = m_iter.iter_started

    return new_m_iter

@njit(MatchIteratorType(CorgiGraphType), cache=True)
def new_match_iter(graph):
    '''Produces a new MatchIterator for a graph.'''

    # Make an iterator prototype for this graph if one doesn't exist
    #  a copy of the prototype will be built when __iter__ is called.
    if(graph.match_iter_prototype_inst is None):
        m_iter_nodes = List.empty_list(MatchIterNodeType)
        handled_vars = Dict.empty(i8,MatchIterNodeType)

        #Make a list of var_inds
        var_inds = List.empty_list(i8)
        for i in range(len(graph.var_end_nodes)-1,-1,-1):
            var_inds.append(i) 

        # Loop downstream to upstream through the end nodes. Build a MatchIterNode
        #  for each end node to help us iterate over valid matches in the graph.
        # for var_ind in range(len(graph.var_end_nodes)-1,-1,-1):
        m_node_ind = 0;
        while(len(var_inds) > 0):
            # print_ind = np.empty(len(handled_vars))
            # for q, k in enumerate(handled_vars):
            #     print_ind[q] = k

            var_ind = var_inds.pop()
            node = graph.var_end_nodes[var_ind]

            # Instantiate a prototype m_node for the end node for this Var.
            m_node = new(MatchIterNodeType)
            m_node.node = node
            # The index the m_node will have after reversal
            m_node.m_node_ind = m_node_ind
            m_node.var_ind = var_ind
            m_node.associated_arg_ind = np.argmax(node.var_inds==var_ind)#node.outputs[np.argmax(node.var_inds==i)]
            m_node.curr_ind = -1;
            # m_node.dep_m_node_ind = -1
            # m_node.dep_arg_ind = -1
            # other_var_ind = node.var_inds[1 if m_node.associated_arg_ind==0 else 0]

            # print("END NODE", var_ind, m_node.m_node_ind, node.lit, m_node.associated_arg_ind)
            
            dep_m_node_inds = np.empty((len(graph.var_end_nodes)),dtype=np.int64)
            dep_arg_inds = np.empty((len(graph.var_end_nodes)),dtype=np.int64)
            dep_node_ptrs = np.empty((len(graph.var_end_nodes)),dtype=np.int64)
            c = 0
            for j, ptr in enumerate(graph.var_end_join_ptrs[m_node.var_ind]):
                if(ptr != 0 and j in handled_vars):
                    dep_m_node = handled_vars[j]
                    dep_m_node_inds[c] = dep_m_node.m_node_ind
                    dep_node_ptrs[c] = ptr

                    dep_node = cast(ptr, CorgiNodeType)
                    dep_arg_inds[c] = np.argmax(dep_node.var_inds==j)
                    # print(j, "Make", m_node.node.lit, m_node.associated_arg_ind, "dep on", dep_node.lit, dep_arg_inds[c])
                    c +=1
            m_node.dep_m_node_inds = dep_m_node_inds[:c]
            m_node.dep_arg_inds = dep_arg_inds[:c]
            m_node.dep_node_ptrs = dep_node_ptrs[:c]

            m_node_ind += 1

            # Mark 'var_ind' as being handled by m_node. This allows 
            #   downstream m_nodes to depend on it.
            if(var_ind not in handled_vars):
                handled_vars[var_ind] = m_node

            m_iter_nodes.append(m_node)

        # Instantiate the prototype and link it to the graph
        m_iter = new(MatchIteratorType)
        m_iter.iter_nodes = m_iter_nodes 
        m_iter.is_empty = False
        m_iter.iter_started = False
        graph.match_iter_prototype_inst = cast(m_iter, StructRefType)

    # Return a copy of the prototype 
    prototype = cast(graph.match_iter_prototype_inst, MatchIteratorType)
    m_iter = copy_match_iter(prototype,graph)
    return m_iter


@njit(unicode_type(MatchIteratorType),cache=True)
def repr_match_iter_dependencies(m_iter):
    rep = ""
    for i, m_node in enumerate(m_iter.iter_nodes):
        base_var_ptr_i = m_node.node.lit.base_var_ptrs[m_node.associated_arg_ind]
        s = f'({str(cast(base_var_ptr_i, VarType).alias)}'
        for j, dep_m_node_ind in enumerate(m_node.dep_m_node_inds):
            dep_m_node = m_iter.iter_nodes[dep_m_node_ind]
            base_var_ptr_j = dep_m_node.node.lit.base_var_ptrs[dep_m_node.associated_arg_ind]
            # s += f",dep={str(dep_m_node.var_ind)}"
            s += f",dep={str(cast(base_var_ptr_j, VarType).alias)}"
        s += f")"

        rep += s
        if(i < len(m_iter.iter_nodes)-1): rep += " "

    return rep

@njit(types.void(MatchIteratorType, MatchIterNodeType),cache=True)
def update_from_upstream_match(m_iter, m_node):
    ''' Updates the list of `other_idrecs` for a beta m_node. If an 
          m_node's 'curr_ind' would have it yield a next match `A` for its 
          associated Var then the 'other_idrecs' are the idrecs
          for the matching facts of the other (non-associated) Var.
    '''
    multiple_deps = len(m_node.dep_m_node_inds) > 1
    if(multiple_deps):
        idrecs_set = Dict.empty(u8,u1)
    
    # print("--", cast(m_node.node.lit.base_var_ptrs[m_node.associated_arg_ind], VarType).alias, "-UPDATE FROM UPSTREAM:--")

    for i, dep_m_node_ind in enumerate(m_node.dep_m_node_inds):
        # Each dep_node is the terminal beta node (i.e. a graph node not an iter node) 
        #  between the vars iterated by m_node and dep_m_node, and might not be the same
        #  as dep_m_node.node.
        dep_node = cast(m_node.dep_node_ptrs[i], CorgiNodeType)
        dep_arg_ind = m_node.dep_arg_inds[i]
        dep_m_node = m_iter.iter_nodes[dep_m_node_ind]
        assoc_arg_ind = 1 if dep_arg_ind == 0 else 0

        # print("\tdep on", dep_node.lit, dep_arg_ind)
        
        # NOTE: Below is a potential optimization where instead of using a
        #  hash-table lookup for idec->internal_ind we instead get the information
        #  directly. Didn't work because curr_ind indexes .idrecs of an m_node which
        #  is only a subset of the matches for the corresponding graph node.
        
        # if(cast(m_node.node, i8) == cast(dep_node, i8)):
        #     dep_output = dep_node.outputs[dep_arg_ind]
        #     if(dep_m_node.curr_ind >= len(dep_output.match_inp_inds)):
        #         m_node.idrecs = np.empty((0,), dtype=np.uint64)
        #         return

        #     print("D0 f_id:", decode_idrec(dep_m_node.idrecs[dep_m_node.curr_ind])[1])
        #     print(dep_output.match_inp_inds, dep_m_node.curr_ind)
        #     fixed_intern_ind = dep_output.match_inp_inds[dep_m_node.curr_ind]

        # TODO: Potentially still a way to add bookeeping that enables a more
        #  direct method than hash-table lookup

        # -----
        # : Getting fixed_intern_ind
        # Determine the index of the fixed downstream match within this node.
        
        # Extract the idrec for the current fixed dependency value in dep_m_node
        dep_idrec = dep_m_node.idrecs[dep_m_node.curr_ind]

        # Use idrecs_to_inds to find the index of the fixed value in dep_node.
        dn_idrecs_to_inds = dep_node.inputs[dep_arg_ind].idrecs_to_inds

        # If failed to retrieve then idrecs for this node should be empty
        if(dep_idrec not in dn_idrecs_to_inds):
            m_node.idrecs = np.empty((0,), dtype=np.uint64)
            return
        
        fixed_intern_ind = dn_idrecs_to_inds[dep_idrec]
        # -----

        # Note: Debug Print Statement
        fixed_var = cast(dep_node.lit.base_var_ptrs[dep_arg_ind], VarType)
        fixed_idrec = dep_node.input_state_buffers[dep_arg_ind][fixed_intern_ind].idrec
        # print("\tFixed:",  f"{fixed_var.alias}:{dep_m_node.curr_ind}", f"f_id:{decode_idrec(fixed_idrec)[1]}")#m_iter.graph.memset.get_fact(fixed_idrec))
        
        # Consult the dep_node's truth_table to fill 'idrecs'.
        inp_states = dep_node.input_state_buffers[assoc_arg_ind]        
        truth_table = dep_node.truth_table
        k = 0
        idrecs = np.empty((truth_table.shape[assoc_arg_ind],), dtype=np.uint64)
        if(assoc_arg_ind == 1):
            for j, t in enumerate(truth_table[fixed_intern_ind, :]):
                if(t): idrecs[k] = inp_states[j].idrec; k += 1
        else:        
            for j, t in enumerate(truth_table[:, fixed_intern_ind]):
                if(t): idrecs[k] = inp_states[j].idrec; k += 1

        # Assign idrecs as just the parts of the buffer we filled. 
        idrecs = idrecs[:k]

        # If the var covered by m_node has multiple dependencies then 
        #  find the set of idrecs common between them. 
        if(multiple_deps):
            if(i == 0):
                for idrec in idrecs:
                    idrecs_set[idrec] = u1(1)
            else:
                # Intersect with previous idrecs_set
                new_idrecs_set = Dict.empty(u8,u1)
                for idrec in idrecs:
                    if(idrec in idrecs_set):
                        new_idrecs_set[idrec] = u1(1)
                idrecs_set = new_idrecs_set

            # Only for print Statement
            # set_f_ids = np.empty(len(idrecs_set),dtype=np.int64)
            # for j, x in enumerate(idrecs_set):
            #     set_f_ids[j] = decode_idrec(x)[1]
            # print(":", i, 'f_ids:', np.array([decode_idrec(x)[1] for x in idrecs]), "f_ids:", set_f_ids)
        else:
            m_node.idrecs = idrecs
            # print(":", "-", 'node:', np.array([decode_idrec(x)[1] for x in idrecs]))

    
    # If multiple dependencies, copy remaining contents of idrecs_set into an array.
    if(multiple_deps):
        idrecs = np.empty((len(idrecs_set),), dtype=np.uint64)
        for i, idrec in enumerate(idrecs_set):
            idrecs[i] = idrec
        m_node.idrecs = idrecs



@njit(types.boolean(MatchIterNodeType), cache=True)
def update_no_depends(m_node):
    # From the output associated with m_node make a copy of match_idrecs
    #  that omits all of the zeros. 
    # print("UPDA TERMINAL")
    cnt = 0
    associated_output = m_node.node.outputs[m_node.associated_arg_ind]
    matches = associated_output.match_idrecs
    idrecs = np.empty((len(matches)),dtype=np.uint64)
    for j, idrec in enumerate(matches):
        if(idrec == 0): continue
        idrecs[cnt] = idrec; cnt += 1;
    # print(m_node.node.lit, m_node.associated_arg_ind, idrecs)
    # If the output only had zeros then prematurely mark m_node as empty.
    if(cnt == 0):
        return False

    # Otherwise update the idrecs for the matches of m_node.
    m_node.idrecs = idrecs[:cnt]
    # print("END UPDA TERMINAL")
    return True

@njit(types.boolean(MatchIteratorType),cache=True)
def match_iter_is_empty(m_iter):
    return m_iter.is_empty


@njit(Tuple((types.boolean,u8[::1]))(MatchIteratorType),cache=True)
def match_iter_next_idrecs(m_iter):
    # print()
    n_vars = len(m_iter.iter_nodes)
    # Increment the m_iter nodes until satisfying all upstream
    while(not m_iter.is_empty):
        
        most_upstream_overflow = -1
        # On a fresh iterator skip incrementating and 
        #  just make sure upstream updates are applied
        if(not m_iter.iter_started):
            most_upstream_overflow = 0
            m_iter.iter_started = True

        # Otherwise increment from downstream to upstream
        else:
            most_upstream_overflow = -1

            # For each m_node from downstream to upstream
            for i in range(n_vars-1,-1,-1):
                m_node = m_iter.iter_nodes[i]
                # Increment curr_ind if m_node the most downstream or if
                #  a downstream m_node overflowed.
                if(i == n_vars-1 or most_upstream_overflow == i+1):
                    m_node.curr_ind += 1

                # Track whether incrementing also overflowed the ith m_node.
                if(m_node.curr_ind >= len(m_node.idrecs)):
                    m_node.curr_ind = 0
                    most_upstream_overflow = i

            # If the 0th m_node overflows then iteration is finished.
            if(most_upstream_overflow == 0):
                m_iter.is_empty = True

        # Starting with the most upstream overflow and moving downstream set m_node.idrecs
        #  to be the set of idrecs consistent with its upstream dependencies.
        idrec_sets_are_nonzero = True
        for i in range(most_upstream_overflow, n_vars):
            m_node = m_iter.iter_nodes[i]

            # Only Update if has dependencies
            # print("L",len(m_node.dep_m_node_inds))
            if(len(m_node.dep_m_node_inds)):
                update_from_upstream_match(m_iter, m_node)

            if(len(m_node.idrecs) == 0): idrec_sets_are_nonzero = False;

        # Note: Keep print for debugging
        # print("<< it: ", np.array([y.curr_ind for y in m_iter.iter_nodes]) , "/", np.array([len(m_iter.iter_nodes[i].idrecs) for i in range(n_vars)]))

        # If each m_node has a non-zero idrec set we can yield a match
        #  otherwise we need to keep iterating
        if(idrec_sets_are_nonzero): break



    


    # Fill in the matched idrecs
    idrecs = np.empty(n_vars,dtype=np.uint64) 
    if(m_iter.is_empty):
        return True, idrecs

    for i in range(n_vars-1,-1,-1):
        m_node = m_iter.iter_nodes[i]
        idrecs[m_node.var_ind] = m_node.idrecs[m_node.curr_ind]
    return False, idrecs

@njit(Tuple((types.boolean, i8[::1]))(MatchIteratorType), cache=True)
def match_iter_next_ptrs(m_iter):
    ms, graph = m_iter.graph.memset, m_iter.graph
    empty, idrecs = match_iter_next_idrecs(m_iter)
    
    ptrs = np.empty(len(idrecs),dtype=np.int64)
    if(empty):
        return empty, ptrs

    for i, idrec in enumerate(idrecs):
        t_id, f_id, _  = decode_idrec(idrec)
        facts = cast(ms.facts[t_id], VectorType)
        ptrs[i] = facts.data[f_id]
        # print("END")
    # print("SLOOP")
    return empty, ptrs

@njit(cache=True)
def match_iter_next(m_iter):
    empty, ptrs = match_iter_next_ptrs(m_iter)
    # print("PTRS", ptrs)
    tup = _struct_tuple_from_pointer_arr(m_iter.output_types, ptrs)
    # print("tup")
    # print(tup)
    return empty, tup


@njit(cache=True)
def fact_ptrs_as_tuple(typs, ptr_arr):
    # print("BEFORE")
    tup = _struct_tuple_from_pointer_arr(typs, ptr_arr)
    # print("AFTER",tup)
    return tup


@njit(cache=True)
def get_graph(ms, conds):
    needs_new_graph = False

    if(conds.matcher_inst is None):
        needs_new_graph = True
    else:
        graph = cast(conds.matcher_inst, CorgiGraphType)
        if(cast(ms, i8) != cast(graph.memset, i8)):
            needs_new_graph = True

    graph = build_corgi_graph(ms, conds)
    conds.matcher_inst = cast(graph, StructRefType)
    return graph

@njit(MatchIteratorType(MemSetType, ConditionsType), cache=True)
def get_match_iter(ms, conds):

    # Performance Note: In integration benchmarks w/ AL get_graph  
    #  takes up about half of the time for a cold match.  
    corgi_graph = get_graph(ms, conds) 

    update_graph(corgi_graph)
    m_iter = new_match_iter(corgi_graph)
    # print("DEPS:", repr_match_iter_dependencies(m_iter))
    if(len(m_iter.iter_nodes) == 0):
        m_iter.is_empty = True
        return m_iter
    for i in range(len(m_iter.iter_nodes)):
        m_node = m_iter.iter_nodes[i]
        m_node.curr_ind = 0
        if(len(m_node.dep_m_node_inds) == 0):
            ok = update_no_depends(m_node)
            if(not ok):
                m_iter.is_empty = True
                break

    return m_iter


# ----------------------------------------------------------------------
# : score_match, check_match

@njit(cache=True)
def match_ptrs_from_idrecs(ms, match_idrecs, length):
    match_ptrs = np.zeros(length,dtype=np.int64)
    for i, idrec in enumerate(match_idrecs):
        if(idrec != 0):
            t_id, f_id, _  = decode_idrec(idrec)
            facts = cast(ms.facts[t_id], VectorType)
            match_ptrs[i] = facts.data[f_id]
    return match_ptrs

@njit(cache=True)
def _infer_unprovided(known_ptr, op, arg_ind):
    ''' Take a known pointer to fact and an "ObjEquals" CREFunc in a beta 
        literal and the base variable index that the fact is supposed to be a 
        match for and return the pointer to the other fact in the Equals relation.
    '''
    # Only bother for beta-like Equals 
    if(known_ptr == 0 or 
       op.origin_data.name != "ObjEquals" or op.n_args != 2):
        # Skip Case
        return 0, False

    # Only bother if there is only one head associated with arg_ind.
    hr = op.head_ranges[arg_ind]
    if(hr.end-hr.start == 1):
        v = cast(op.head_infos[hr.start].var_ptr, VarType)
        if(len(v.deref_infos) == 0):
            return 0, True
        other_data_ptr = resolve_deref_data_ptr(cast(known_ptr,BaseFact), v.deref_infos)
        other_ptr = _load_ptr(i8, other_data_ptr)
        return other_ptr, False

    # print("OTHER FAIL")
    # Fail Case
    return 0, True

WN_VAR_TYPE = u8(1)
WN_BAD_DEREF = u8(2)
WN_INFER_UNPROVIDED = u8(3)
WN_FAIL_MATCH = u8(4)

np_why_not_type = np.dtype([
    # Enum for ATTR or LIST
    ('ptr', np.int64),
    ('var_ind0', np.int64),
    ('var_ind1', np.int64),
    ('d_ind', np.int64),
    ('c_ind', np.int64),
    ('kind', np.uint64),
])

why_not_type = numba.from_dtype(np_why_not_type)

@njit
def new_why_not(ptr, var_ind0, var_ind1=-1, d_ind=-1, c_ind=-1, kind=0):
    arr = np.empty(1,dtype=why_not_type)
    arr[0].ptr = i8(ptr)
    arr[0].var_ind0 = i8(var_ind0)
    arr[0].var_ind1 = i8(var_ind1)
    arr[0].d_ind = i8(d_ind)
    arr[0].c_ind = i8(c_ind)
    arr[0].kind = u8(kind)
    return arr[0]

# Compile implementation of set_base_arg for BaseFact type
set_base_fact_arg = set_base_arg_val_impl(BaseFact)

@njit(cache=True)
def _score_match(conds, match_ptrs, why_nots=None, zero_on_fail=False):
    # Get the instance pointers from match_idrecs
    cum_weight, total_weight = 0.0, 0.0
    # Pad match_ptrs with zeros to account for variables in the conditions
    #  for which no match has been provided
    _match_ptrs = np.zeros(len(conds.vars), np.int64)
    _match_ptrs[:len(match_ptrs)] = match_ptrs
    match_ptrs = _match_ptrs

    # Give weight for at least having the right var types
    bad_vars = np.zeros(len(conds.vars), dtype=np.uint8)    
    for i, v in enumerate(conds.vars):
        ptr_i = match_ptrs[i]
        if(ptr_i == 0):
            continue

        m = cast(ptr_i, BaseFact)
        t_id, _, _ = decode_idrec(m.idrec)

        total_weight += 1.0

        # NOTE: Need to handle inheritance here... isa()
        if(v.base_t_id != t_id):
            bad_vars[i] = u1(1)
            if(why_nots is not None):
                why_nots.append(new_why_not(
                    cast(v,i8), i, kind=WN_VAR_TYPE))
                    
            continue

        cum_weight += 1.0
        
    # NOTE: MAKE WORK FOR OR
    for d_ind, conj in enumerate(conds.dnf):
        for c_ind, lit in enumerate(conj):
            total_weight += lit.weight

            # Set arguments
            # TODO: Should set up so never need to skip (every literal always evaluated)
            skip = False
            infer_fail = False
            for i, var_ind in enumerate(lit.var_inds):
                if(bad_vars[var_ind]):
                    continue
                # print("VAR_IND", var_ind)
                ptr_i = match_ptrs[var_ind]
                # print("ptr_i", ptr_i)
                # If a match candidate wasn't provided in match_idrecs
                #  then try to resolve it.
                if(ptr_i == 0 and lit.op.n_args == 2):
                    j = 1 if i == 0 else 0
                    known_ptr = match_ptrs[lit.var_inds[j]]
                    ptr_i, skip = _infer_unprovided(known_ptr, lit.op, j)
                    match_ptrs[var_ind] = ptr_i

                if(skip):
                    # print("SKIP", var_ind)
                    continue

                if(ptr_i == 0):
                    # print("ZERO PTR", var_ind)
                    if(why_nots is not None):
                        why_nots.append(new_why_not(cast(lit,i8), 
                            var_ind, kind=WN_INFER_UNPROVIDED))
                    infer_fail = True
                    continue

                # print(i, var_ind, repr(cast(ptr_i, BaseFact)), cast(ptr_i, BaseFact).idrec)
                set_base_fact_arg(lit.op, i, cast(ptr_i, BaseFact))
                
            if(skip):
                # print("SKIP", node.op, node.lit.weight)
                continue

            if(infer_fail):
                if(zero_on_fail):
                    return 0.0, 0.0    
                continue

            # Call
            call_self = get_best_call_self(lit.op, False)
            status = call_self(lit.op)

            # Add weight if match
            if((status == CFSTATUS_TRUTHY) ^ lit.negated):
                # print("OKAY", node.op, node.lit.weight)
                cum_weight += lit.weight
            else:
                if(why_nots is not None):
                    why_nots.append(new_why_not(
                        cast(lit,i8),
                        lit.var_inds[0],
                        var_ind1=-1 if len(lit.var_inds) < 2 else lit.var_inds[1],
                        d_ind=d_ind, c_ind=c_ind,
                        kind=WN_FAIL_MATCH))

                if(zero_on_fail):
                # print("BAD Status", lit.op, status, CFSTATUS_TRUTHY)
                    return 0.0, 0.0
            # else:
                # print("FAIL", node.op, node.lit.weight)
    # print("END")
    return cum_weight, total_weight

@njit(MemSetType(ConditionsType, types.optional(MemSetType)), cache=True)
def _ensure_ms(conds, ms):
    if(ms is None):
        if(conds.matcher_inst is None):
            raise ValueError("Cannot check/score matches on Conditions object without assigned MemSet.")
        corgi_graph = cast(conds.matcher_inst, CorgiGraphType)
        return corgi_graph.memset
    return ms


@njit(cache=True)
def score_match(conds, match_ptrs):
    cum_weight, total_weight = _score_match(conds, match_ptrs, None, False)
    return cum_weight / total_weight

@njit(cache=True)
def check_match(conds, match_ptrs):
    w, _ = _score_match(conds, match_ptrs, None, True)
    return w != 0.0


# why_not_type = Tuple((LiteralType, u1))
@njit(cache=True)
def why_not_match(conds, match_ptrs):
    why_nots = List.empty_list(why_not_type)
    w, _ = _score_match(conds, match_ptrs, why_nots, False)
    why_not_arr = np.empty(len(why_nots), dtype=why_not_type)
    for i, wn in enumerate(why_nots):
        why_not_arr[i] = wn
    return why_not_arr




            



    # # Note should probably cache this too
    # index_map = Dict.empty(i8, i8)
    # for i, v in enumerate(c.vars):
    #     index_map[i8(v.base_ptr)] = i



    # # Reorganize the dnf into a distributed dnf with one dnf per var
    # if(not c.has_distr_dnf):
    #     build_distributed_dnf(c,index_map)

    # c.distr_dnf




if __name__ == "__main__":
    pass
    # deref_depends = Dict.empty(i8, DictType(i8,u1))

    # node_ctor()


