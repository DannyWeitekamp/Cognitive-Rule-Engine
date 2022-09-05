import numpy as np
import numba
from numba import njit, i8,i4,u1,u2,u4,u8, f8, f4, generated_jit, types, literal_unroll
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, Tuple
from cre.fact import define_fact
from cre.structref import define_structref, define_structref_template
from numba.experimental import structref
from numba.experimental.structref import new, define_attributes
from numba.extending import lower_cast, overload, overload_method
from cre.memset import MemSet,MemSetType
from cre.utils import  lower_setattr, _raw_ptr_from_struct, _ptr_from_struct_incref, decode_idrec, _listtype_sizeof_item, _cast_structref, _obj_cast_codegen, DEREF_TYPE_ATTR, DEREF_TYPE_LIST, _memcpy_structref
from cre.vector import VectorType
from cre.transform.incr_processor import IncrProcessorType, ChangeEventType, incr_processor_fields, init_incr_processor
from cre.structref import CastFriendlyStructref, define_boxing
from cre.context import cre_context, CREContextDataType
from cre.fact import Fact, BaseFact, DeferredFactRefType
from cre.gval import gval as gval_type
from cre.var import GenericVarType
from cre.vector import new_vector


def _get_base_type(fact_type):
    parent = None
    while(hasattr(fact_type,'parent_type')):
        parent = fact_type = fact_type.parent_type
    return parent if parent is not None else fact_type

def assert_mutual_base(fact_types, id_attr):
    if(len(fact_types) == 1): return fact_types[0]
    base = None
    for fact_type in fact_types:
        f_base = _get_base_type(fact_type)
        if(base is not None):
            assert f_base == base, "RelativeEncoder requires that all fact types inherit from a common base type."
        
        base = f_base

    assert id_attr in base.spec, "Base type must define an identifier attribute (e.g. like id='fact_instance_name')"
    return base



def get_relational_fact_attrs(fact_types):
    ''' Takes in a set of fact types and returns all (fact, attribute) pairs
        for "relational" attributes. 
    '''
    context = cre_context()

    rel_fact_attrs = {}
    for ft in fact_types:
        ft = ft.instance_type if (isinstance(ft, types.TypeRef)) else ft
        parents = context.parents_of.get(ft._fact_name,[])
        for attr, attr_spec in ft.filter_spec("relational").items():
            # relational = False
            attr_t = attr_spec['type']
            # attr_t = attr_t.instance_type if (isinstance(attr_t, (types.TypeRef,DeferredFactRefType))) else attr_t
            # if(isinstance(attr_t, Fact)): relational = True
            # if(isinstance(attr_t, types.ListType)):
            #     item_t = attr_t.item_type
            #     item_t = item_t.instance_type if (isinstance(item_t, (types.TypeRef,DeferredFactRefType))) else item_t
            #     if(isinstance(item_t, Fact)): relational = True
            #     attr_t = types.ListType(item_t)

            # relational = d.get("relational", relational)
                
            # if(relational):
            is_new = True
            for p in parents:
                if((p,attr) in rel_fact_attrs):
                    is_new = False
                    break
            if(is_new): rel_fact_attrs[(ft,attr)] = attr_t

    return tuple([(fact_type, attr_t, types.literal(attr)) for (fact_type, attr), attr_t in rel_fact_attrs.items()])


#### Relative Encoder ####
from cre.utils import deref_info_type, np_deref_info_type
np_ind_dist_deref = np.dtype([
    ("ind", np.int32),
    ("dist", np.float32),
    ("attr_id", np.uint32),
    ("item_ind", np.int32),
    # ('deref_info1', np_deref_info_type),
    # ('deref_info2', np_deref_info_type),
    ])
ind_dist_deref = numba.from_dtype(np_ind_dist_deref)

relative_encoder_fields = {
    **incr_processor_fields,
    "context_data" : CREContextDataType,
    "dist_matrix" : ind_dist_deref[:, ::1],
    "idrec_to_ind" : DictType(u8, i4),
    "visited_inds" : DictType(i8, u1),
    "facts" : ListType(BaseFact),
    "deref_info_matrix" : deref_info_type[:,::1],
    "lateral_w" : f8,
    "id_to_ind" : DictType(unicode_type, i4),
    "var_cache" : DictType(Tuple((i4,i4,i4)),GenericVarType),
    "valid_t_ids" : VectorType,
    "keep_floating_facts" : types.boolean,
    "needs_rebuild" : types.boolean,
    "fact_types" : types.Any,
    "relational_attrs" : types.Any,
    "base_fact_type" : types.Any,
    "id_attr" : types.Any,
}

@structref.register
class RelativeEncoderTypeClass(CastFriendlyStructref):
    pass

GenericRelativeEncoderType = RelativeEncoderTypeClass([(k,v) for k,v in relative_encoder_fields.items()])

from numba.core.typing.typeof import typeof
def get_relative_encoder_type(fact_types, id_attr):
    relational_fact_attrs = get_relational_fact_attrs(fact_types)

    relational_fact_attrs_type = typeof(relational_fact_attrs)
    field_dict = {**relative_encoder_fields,
                 "fact_types" : Tuple(fact_types),
                 "relational_attrs" : relational_fact_attrs_type,
                 "base_fact_type" : types.TypeRef(assert_mutual_base(fact_types, id_attr)), 
                 "id_attr" : types.literal(id_attr),
                 }
                 
    re_type = RelativeEncoderTypeClass([(k,v) for k,v in field_dict.items()])
    re_type._fact_types = fact_types
    re_type._relational_attrs = relational_fact_attrs
    return re_type

class RelativeEncoder(structref.StructRefProxy):
    def __new__(cls, fact_types, in_memset=None, id_attr='id', context=None, keep_floating_facts=True):
        # Make new in_ms and out_ms if they are not provided.
        context = cre_context(context)
        fact_types=tuple(fact_types)
        if(in_memset is None): in_memset = MemSet(context);
        re_type = get_relative_encoder_type(fact_types,id_attr)
        self = RelativeEncoder_ctor(context.context_data, re_type, in_memset, keep_floating_facts)
        return self

    def encode_relative_to(self, ms, sources, source_vars):
        assert len(sources) == len(source_vars), f"{sources},{source_vars}"

        source_idrecs = np.empty(len(sources),dtype=np.uint64)
        for i, src in enumerate(sources): 
            source_idrecs[i] = src.idrec

        var_ptrs = np.empty(len(sources),dtype=np.int64)
        for i, v in enumerate(source_vars): 
            var_ptrs[i] = v.get_ptr()

        out_ms = encode_relative_to(self, ms, source_idrecs, var_ptrs)
        return out_ms

    @property
    def in_memset(self):
        return get_in_memset(self)

    def set_in_memset(self, in_memset):
        if(not check_same_in_memset(self,in_memset)):
            set_in_memset(self,in_memset)
            # RelativeEncoder_reinit(self)

    def update(self):
        RelativeEncoder_update(self)

define_boxing(RelativeEncoderTypeClass, RelativeEncoder)

@lower_cast(RelativeEncoderTypeClass, GenericRelativeEncoderType)
@lower_cast(RelativeEncoderTypeClass, IncrProcessorType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)


@njit(cache=True)
def new_dist_matrix(n, old_dist_matrix=None):
    dist_matrix = np.empty((n,n),dtype=ind_dist_deref)
    for i in range(n):
        for j in range(n):
            if(i == j):
                dist_matrix[i,j].ind = i4(-1)
                dist_matrix[i,j].dist = f4(0)
                dist_matrix[i,j].attr_id = u4(0)
                dist_matrix[i,j].item_ind = i4(-1)
            else:
                dist_matrix[i,j].ind = i4(-1)
                dist_matrix[i,j].dist = f4(np.inf)
                dist_matrix[i,j].attr_id = u4(0)
                dist_matrix[i,j].item_ind = i4(-1)
    if(old_dist_matrix is not None):
        o_n, o_m = old_dist_matrix.shape
        dist_matrix[:o_n, :o_m] = old_dist_matrix
    return dist_matrix



@generated_jit(cache=False, nopython=True)
def _fill_rel_derefs(deref_info_matrix, rel_derefs, start=0):
    if(len(rel_derefs) > 0):
        def impl(deref_info_matrix, rel_derefs, start=0):
            attr_id = start
            for d1 in literal_unroll(rel_derefs):
                # d1 = rel_derefs[i]
                deref_info_matrix[attr_id][0].type = d1[0]
                deref_info_matrix[attr_id][0].a_id = d1[1]
                deref_info_matrix[attr_id][0].t_id = d1[2]
                deref_info_matrix[attr_id][0].offset = d1[3]
                attr_id += 1
    else:
        def impl(deref_info_matrix, rel_derefs, start=0):
            pass
    return impl


@generated_jit(cache=False, nopython=True)
def _fill_list_derefs(deref_info_matrix, list_derefs, start=0):
    if(len(list_derefs) > 0):
        def impl(deref_info_matrix, list_derefs, start=0):
            attr_id = start
            for tup in literal_unroll(list_derefs):
                attr_t, d1, d2 = tup

                deref_info_matrix[attr_id][0].type = d1[0]
                deref_info_matrix[attr_id][0].a_id = d1[1]
                deref_info_matrix[attr_id][0].t_id = d1[2]
                deref_info_matrix[attr_id][0].offset = d1[3]
                deref_info_matrix[attr_id][1].type = d2[0]
                deref_info_matrix[attr_id][1].a_id = 0
                deref_info_matrix[attr_id][1].t_id = d2[2]
                item_size = _listtype_sizeof_item(attr_t)
                deref_info_matrix[attr_id][1].offset = item_size
                attr_id += 1
    else:
        def impl(deref_info_matrix, list_derefs, start=0):
            pass
    return impl

@generated_jit(cache=True, nopython=True)
def _build_valid_t_ids(self):
    context = cre_context()
    t_ids = set()
    for fact_type in self._fact_types:
        t_ids.add(context.get_t_id(_type=fact_type))
    val_t_ids = tuple(t_ids)
    max_t_id = int(max(val_t_ids))

    def impl(self):
        valid_t_ids = new_vector(max_t_id+1)
        for i in range(len(val_t_ids)):
            valid_t_ids.set_item_safe(val_t_ids[i], 1) 
        return valid_t_ids
    return impl

@generated_jit(cache=True, nopython=True)
def _build_deref_info_matrix(self):
    context = cre_context()
    re_type = self
    rel_derefs, list_derefs = [], []

    for base_t, attr_t, attr in re_type._relational_attrs:
        deref_info1 = base_t.get_attr_deref_info(attr.literal_value)
        if(not isinstance(attr_t,types.ListType)):
            rel_derefs.append(deref_info1.tolist())
        else:
            deref_info2 = np.zeros((),dtype=deref_info_type)
            deref_info2['type'] = DEREF_TYPE_LIST
            deref_info2['t_id'] = context.get_t_id(_type=attr_t.item_type)
            d1, d2 = deref_info1.tolist(), deref_info2.tolist()
            list_derefs.append((attr_t, deref_info1.tolist(), deref_info2.tolist()))        

    rel_derefs = tuple(rel_derefs)
    list_derefs = tuple(list_derefs)
    n_attrs = len(rel_derefs) + len(list_derefs)

    def impl(self):
        # Fill in the deref_info_matrix which we use to build Var instances 
        deref_info_matrix = np.zeros((n_attrs,2),dtype=deref_info_type)
        
        _fill_list_derefs(deref_info_matrix, list_derefs)
        _fill_rel_derefs(deref_info_matrix, rel_derefs,start=len(list_derefs))
        return deref_info_matrix

    return impl

var_cache_key_type = Tuple((i4,i4,i4))

@njit(cache=True)
def RelativeEncoder_reinit(self,size=32):
    self.dist_matrix = new_dist_matrix(size)
    self.idrec_to_ind = Dict.empty(u8,i4)
    self.visited_inds = Dict.empty(i8,u1)
    self.facts = List.empty_list(BaseFact)
    self.id_to_ind = Dict.empty(unicode_type,i4)
    self.var_cache = Dict.empty(var_cache_key_type,GenericVarType)

# print("A")
@njit(cache=True)
def RelativeEncoder_ctor(context_data, re_type, in_memset, keep_floating_facts):
    # def impl(re_type, in_ms):
    st = new(re_type)
    init_incr_processor(st, in_memset)
    RelativeEncoder_reinit(st)
    st.context_data = context_data
    st.lateral_w = f8(1.0)
    st.deref_info_matrix = _build_deref_info_matrix(st)
    st.valid_t_ids = _build_valid_t_ids(st)
    st.needs_rebuild = False
    st.keep_floating_facts = keep_floating_facts
    return st

    
    return k+1
from cre.fact_intrinsics import fact_lower_getattr

# @njit(cache=True)
# def fill_adj_inds(self, fact, k, adj_inds, attr_id, item_ind):
#     ind = self.idrec_to_ind[fact.idrec]

#     adj_inds[k].ind = i4(ind)
#     adj_inds[k].dist = f4(self.lateral_w)
#     adj_inds[k].attr_id = u4(attr_id)
#     adj_inds[k].item_ind = i4(item_ind)

#     return k+1

@njit(cache=True)
def fill_adj_inds(self, fact, k, adj_inds, attr_id, item_ind):
    ind = self.idrec_to_ind[fact.idrec]
    adj_inds[k].ind = i4(ind)
    adj_inds[k].dist = f4(self.lateral_w - (1.0/(2<<(attr_id+1)*3) if attr_id < 7 else 0.0))
    adj_inds[k].attr_id = u4(attr_id)
    adj_inds[k].item_ind = i4(item_ind)
    return k+1

@njit(cache=True)
def fill_adj_list_inds(self, lst, k, adj_inds, attr_id):
    for i, item in enumerate(lst):
        fact = _cast_structref(self.base_fact_type, item)
        # k = fill_adj_inds(self, base_item, k, adj_inds, attr_id, i4(i))
        ind = self.idrec_to_ind[fact.idrec]
        adj_inds[k].ind = i4(ind)
        adj_inds[k].dist = f4(self.lateral_w - (1.0/(2<<(attr_id+1)*3) if attr_id < 7 else 0.0))
        adj_inds[k].attr_id = u4(attr_id)
        adj_inds[k].item_ind = i4(i)
        k += 1
    return k


@generated_jit(cache=True,nopython=True)
def _next_list_adj(self, fact):
    list_attrs = []
    n_rel = 0
    for base_t, attr_t, attr in self._relational_attrs:
        if(isinstance(attr_t,types.ListType)):
            list_attrs.append((attr, base_t))
        else:
            n_rel += 1
    list_attrs = tuple(list_attrs)

    if(len(list_attrs) == 0): 
        return lambda self, fact : (np.empty(n_rel, dtype=ind_dist_deref), 0, 0)
    
    def impl(self, fact):
        # Make a buffer to fill adj_inds
        n_ladj = 0
        for ltup in literal_unroll(list_attrs):
            attr, base_t = ltup
            if(fact.isa(base_t)):
                typed_fact = _cast_structref(base_t, fact)
                n_ladj += len(fact_lower_getattr(typed_fact, attr))
        adj_inds = np.empty(n_rel+n_ladj, dtype=ind_dist_deref)
        # adj_facts = List.empty_list(self.base_fact_type)
        attr_id, k = 0, 0

        # Fill in list relative attributes
        for ltup in literal_unroll(list_attrs):
            attr, base_t = ltup
            if(fact.isa(base_t)):
                typed_fact = _cast_structref(base_t, fact)

                # Member is a list so go through it's items
                lst = fact_lower_getattr(typed_fact, attr)
                k = fill_adj_list_inds(self, lst, k, adj_inds, attr_id)
                # for i, item in enumerate(member):
                #     base_item = _cast_structref(self.base_fact_type,item)
                #     k = fill_adj_inds(self, base_item, k, adj_inds, attr_id, i4(i))
            attr_id += 1
        return adj_inds, attr_id, k
    return impl

@generated_jit(cache=True,nopython=True)
def _next_rel_adj(self, fact, adj_inds, attr_id, k):
    rel_attrs = []
    n_rel = 0
    for base_t, attr_t, attr in self._relational_attrs:
        if(not isinstance(attr_t,types.ListType)):
            rel_attrs.append((attr, base_t))
    rel_attrs = tuple(rel_attrs)
    # print("<<<<", rel_attrs)

    if(len(rel_attrs) == 0): 
        return lambda self, fact, adj_inds, attr_id, k : (adj_inds, attr_id, k)

    def impl(self, fact, adj_inds, attr_id, k):
        # Fill in list relative attributes
        for tup in literal_unroll(rel_attrs):
            attr, base_t = tup
            if(fact.isa(base_t)):
                typed_fact = _cast_structref(base_t, fact)
                member = fact_lower_getattr(typed_fact, attr)

                if(member is not None):
                    base_member = _cast_structref(self.base_fact_type, member)
                    k = fill_adj_inds(self, base_member, k, adj_inds, attr_id, -1)
            attr_id += 1
        return adj_inds, attr_id, k
    return impl
        
@njit(cache=True)
def next_adjacent(self, fact):
    adj_inds, attr_id, k = _next_list_adj(self, fact)
    adj_inds, attr_id, k = _next_rel_adj(self, fact, adj_inds, attr_id, k)
    return adj_inds[:k]


@njit(cache=True)
def update_relative_to(self , sources):
    dist_matrix = self.dist_matrix
    frontier_inds = Dict.empty(i8,u1)
    next_frontier_inds = Dict.empty(i8,u1)

    for src in sources:
        s_ind = self.idrec_to_ind[src.idrec]
        frontier_inds[i8(s_ind)] = u1(1)

    while(len(frontier_inds) > 0):
        for b_ind in frontier_inds:
            b = self.facts[b_ind]
            for c_ind_dist in next_adjacent(self,b):

                c_ind = i8(c_ind_dist.ind)

                # Skip if does not exist
                if(c_ind == -1): continue
                    
                # If the dist_matrix doesn't have b->c then fill it in
                if(dist_matrix[b_ind, c_ind].ind == -1):
                    #Assign new heading and dist
                    dist_matrix[b_ind, c_ind].ind = c_ind
                    dist_matrix[b_ind, c_ind].dist = c_ind_dist.dist 
                    dist_matrix[b_ind, c_ind].attr_id = c_ind_dist.attr_id
                    dist_matrix[b_ind, c_ind].item_ind = c_ind_dist.item_ind

                    next_frontier_inds[c_ind] = u1(1)

                    # print("CONNECT:", f'{b_ind}', "->", f'{c_ind}', c_ind_dist.dist)
                    # print("CONNECT:", f'{b.id}', "->", f'{c.id}', c_ind_w_pair.dist)

                for a_ind in self.visited_inds:
                    if( a_ind == b_ind or a_ind == c_ind): continue

                    new_dist = dist_matrix[a_ind, b_ind].dist  + dist_matrix[b_ind, c_ind].dist 
                    if(new_dist < dist_matrix[a_ind, c_ind].dist):
                        # print("CONNECT:", f'{self.comps[a_ind].id}', "->", f'{c.id},', "via", f'{b.id}', ":", new_dist, "<", dist_matrix[a_ind, c_ind].dist)
                        dist_matrix[a_ind, c_ind].ind = dist_matrix[a_ind, b_ind].ind 
                        dist_matrix[a_ind, c_ind].dist = new_dist 
                        dist_matrix[a_ind, c_ind].attr_id = dist_matrix[a_ind, b_ind].attr_id
                        dist_matrix[a_ind, c_ind].item_ind = dist_matrix[a_ind, b_ind].item_ind
                        next_frontier_inds[c_ind] = u1(1)
                
            self.visited_inds[b_ind] = u1(1)
        frontier_inds = next_frontier_inds
        next_frontier_inds = Dict.empty(i8,u1)

    # print("B")
    # print(dist_matrix[:len(self.idrec_to_ind),:len(self.idrec_to_ind)])


@generated_jit(cache=True, nopython=True)
@overload_method(RelativeEncoderTypeClass,'get_changes')
def incr_pr_accumulate_change_events(self, end=-1, exhaust_changes=True):
    def impl(self, end=-1, exhaust_changes=True):
        incr_pr = _cast_structref(IncrProcessorType, self)
        return incr_pr.get_changes(end=end, exhaust_changes=exhaust_changes)
    return impl


@generated_jit(cache=True,nopython=True)
def _check_needs_rebuild(self, change_events):
    context = cre_context()
    # Make base_t, a_id pairs for relative attributes
    base_a_id_pairs = []
    for base_t, attr_t, attr in self._relational_attrs:
        a_id = base_t.get_attr_a_id(attr.literal_value)
        base_a_id_pairs.append((base_t,a_id))
    base_a_id_pairs = tuple(base_a_id_pairs)

    def impl(self, change_events):
        # Need rebuild if there were any retractions or modifications to
        #  'relational' attributes.
        need_rebuild = False
        for ce in change_events:
            # Ensure that the t_ids belong to provided fact_types.
            if(ce.t_id <= len(self.valid_t_ids) and self.valid_t_ids[i8(ce.t_id)]):
                if(ce.was_retracted): 
                    need_rebuild = True; break;
                if(ce.was_modified):
                    fact = self.in_memset.get_fact(ce.idrec, self.base_fact_type)
                    for tup in literal_unroll(base_a_id_pairs):
                        base_t, a_id = tup
                        if(a_id in ce.a_ids and fact.isa(base_t)):
                            need_rebuild = True; break;
            else:
                raise ValueError(
"Relative Encoder in_memset should only contain facts of given fact_types. \
Ensure that RelativeEncoder is not initialized with a flattened MemSet."
                    )
            
        return need_rebuild
            
    return impl


@njit(cache=True)
def _insert_fact(self,fact):
    ind = i4(len(self.idrec_to_ind))
    self.idrec_to_ind[fact.idrec] = ind
    self.id_to_ind[fact_lower_getattr(fact, self.id_attr)] = ind
    self.facts.append(fact)


@njit(cache=True)
def _ensure_dist_matrix_size(self):
    if(len(self.dist_matrix) < len(self.facts)):
        self.dist_matrix = new_dist_matrix(len(self.facts)*2, self.dist_matrix)
    


@njit(cache=True)
def RelativeEncoder_update(self):
    change_events = self.get_changes()
    
    # Need rebuild if there were any retractions or modifications to
    #  'relative' attributes. TODO: rebuilds could probably triggered
    #   more conservatively.
    if(not self.needs_rebuild):
        self.needs_rebuild = _check_needs_rebuild(self,change_events)
    # print("<<", "need_rebuild", need_rebuild)
    if(self.needs_rebuild):
        # If need_rebuild start bellman-ford shortest path from scratch
        RelativeEncoder_reinit(self)
        sources = self.in_memset.get_facts(self.base_fact_type)
        for fact in sources:
            _insert_fact(self, fact)
        _ensure_dist_matrix_size(self)
        update_relative_to(self, sources)    
    else:
        # Otherwise we can do an incremental update
        # TODO: For some reason slower on second run. Might be in here
        #  although could also be a cache locality thing.
        sources = List.empty_list(self.base_fact_type)
        for ce in change_events:
            if(ce.was_declared or ce.was_modified):
                if(ce.was_declared):
                    fact = self.in_memset.get_fact(ce.idrec, self.base_fact_type)
                    _insert_fact(self,fact)
                    sources.append(fact)
        _ensure_dist_matrix_size(self)
        update_relative_to(self, sources)    

    self.needs_rebuild = False

@njit(cache=True)
def _closest_source(self, ind, s_inds):
    '''Finds the closest source index among 's_inds' to the fact 
         at index 'ind'. Distance is src->fact, not other way around.'''
    min_i = -1
    min_s_ind = -1
    min_dist = np.inf
    for i, s_ind in enumerate(s_inds):
        dist = self.dist_matrix[i8(s_ind)][i8(ind)].dist
        if(dist < min_dist): 
            min_dist = dist
            min_s_ind = s_ind
            min_i = i

    return min_s_ind, min_i

@njit(cache=True)
def _new_rel_var(source_var, head_t_id, deref_infos):
    ''' A constructor for a new Var instance with base source_var'''
    # source_var = source_vars[s_i]    
    # new_var = _memcpy_structref(source_var)
    new_var = new(GenericVarType)
    new_var.idrec = source_var.idrec
    new_var.alias = source_var.alias
    new_var.base_t_id = source_var.base_t_id
    new_var.head_t_id = head_t_id
    new_var.base_ptr = source_var.base_ptr
    new_var.base_ptr_ref = _ptr_from_struct_incref(source_var)
    new_var.is_not = u1(0)
    new_var.conj_ptr = i8(0)
    new_var.deref_infos = deref_infos
    new_var.hash_val = u8(0)
    return new_var

@njit(cache=True)
def _make_rel_var(self, f_ind, s_ind, s_var, extra_derefs=None):
    '''Builds a Var instance with a dereference chain 
         id==id_str. Encodes relative to the closest source. '''

    # Follow the shortest path encoded in the Bellman-Ford dist_matrix 
    #  to build the deref_info sequence for the var
    deref_info_buffer = np.empty((2*len(self.facts),),dtype=deref_info_type)
    k = 0; ind = s_ind;
    while(True):
        dm_entry = self.dist_matrix[i8(ind)][i8(f_ind)]
        ind = dm_entry.ind

        if(ind == i4(-1)): break

        deref_infos = self.deref_info_matrix[dm_entry.attr_id]
        deref_info_buffer[k:k+1] = deref_infos[0:1]
        k += 1
        if(deref_infos[1].type == DEREF_TYPE_LIST):
            deref_info_buffer[k].type = DEREF_TYPE_LIST
            deref_info_buffer[k].a_id = dm_entry.item_ind
            deref_info_buffer[k].t_id = decode_idrec(self.facts[ind].idrec)[0]#deref_infos[1].t_id
            deref_info_buffer[k].offset = dm_entry.item_ind*deref_infos[1].offset
            k += 1
        else:
            deref_info_buffer[k-1].t_id = decode_idrec(self.facts[ind].idrec)[0]

    # Add any extra deref_infos. e.g.'.value' at the end of a deref chain.
    if(extra_derefs is not None):
        for i in range(len(extra_derefs)):
            deref_info_buffer[k:k+1] = extra_derefs[i:i+1]
            k += 1

    # Make sure that the last t_id is the actual t_id of the fact
    #  so that we are certain we can dereference the final value.
    if(k >= 2):
        # print("&&", deref_info_buffer[k-2])
        # p = k-2 if deref_info_buffer[k-2].type == DEREF_TYPE_ATTR else k-3
        deref_info_buffer[k-2].t_id = decode_idrec(self.facts[f_ind].idrec)[0]
        # print("&&", deref_info_buffer[k-2].t_id)
    
    # Make the Var
    head_t_id = deref_info_buffer[k-1].t_id if(k > 0) else s_var.base_t_id
    # print(deref_info_buffer[:k].copy(), s_var, s_var.base_t_id, head_t_id)
    r_var =  _new_rel_var(s_var, head_t_id, deref_info_buffer[:k].copy())
    # print(r_var.base_t_id)
    return  r_var


@njit(cache=True)
def f_ind_from_id(self, id_str):
    if(id_str not in self.id_to_ind): 
        print(f"No fact with {self.id_attr}={id_str}")
        raise ValueError(f"Tried to re-encode fact with an id unknown to the RelativeEncoder.")
    f_ind = self.id_to_ind[id_str]
    return f_ind

@njit(cache=True,locals={"tup": var_cache_key_type})
def get_rel_var(self, f_ind, source_vars, s_inds, extra_derefs=None):
    '''Builds a Var using the relative encoder for the fact instance with
         id==id_str. Var has dereference chain that starts at the closest source. '''
    s_ind, s_i = _closest_source(self, f_ind, s_inds)

    if(s_ind == -1):
        return None

    s_var = source_vars[s_i]

    offset, cachable = i4(-1), True
    if(extra_derefs is not None):
        if(len(extra_derefs) > 1):
            cachable = False
        elif(len(extra_derefs) == 1):
            offset = extra_derefs[0].offset

    if(cachable):
        tup = (i4(s_ind),i4(f_ind), offset)
        if(tup not in self.var_cache):
            rel_var = _make_rel_var(self, f_ind, s_ind, s_var, extra_derefs)
            self.var_cache[tup] = rel_var
            return rel_var
        else:
            return self.var_cache[tup]
    else:
        return _make_rel_var(self, f_ind, s_ind, s_var, extra_derefs)

    # return _make_rel_var(self, f_ind, s_ind, source_vars[s_i])
from cre.core import T_ID_TUPLE_FACT, T_ID_VAR
from cre.cre_object import copy_cre_obj, _iter_mbr_infos, cre_obj_set_item, CREObjType, PRIMITIVE_MBR_ID, OBJECT_MBR_ID
from cre.tuple_fact import TupleFact
from cre.utils import _load_ptr, _struct_from_ptr

@njit(cache=True, locals={'rel_var' : GenericVarType})
def rel_encode_tf(self, tf, source_vars, s_inds):
    tf_copy = copy_cre_obj(tf)
    tf_cre_obj = _cast_structref(CREObjType, tf)
    for i, (t_id, m_id, data_ptr) in enumerate(_iter_mbr_infos(tf_cre_obj)):
        if(t_id == T_ID_VAR):
            v = _struct_from_ptr(GenericVarType, _load_ptr(i8,data_ptr))
            f_ind = f_ind_from_id(self, v.alias)
            _rel_var = get_rel_var(self, f_ind, source_vars, s_inds, v.deref_infos)
            if(_rel_var is None): 
                _rel_var = v
                is_floating = True
            rel_var = _rel_var
            cre_obj_set_item(tf_copy, i, rel_var)
            # cre_obj_set_member_info(tf_copy, i, (T_ID_VAR, OBJECT_MBR_ID))

        elif(m_id != PRIMITIVE_MBR_ID):

            # Fact case 
            fact = _struct_from_ptr(CREObjType, _load_ptr(i8,data_ptr))
            if(fact.idrec in self.idrec_to_ind):
                f_ind = self.idrec_to_ind[fact.idrec]
                _rel_var = get_rel_var(self, f_ind, source_vars, s_inds)
                if(_rel_var is not None):
                    rel_var = _rel_var
                    cre_obj_set_item(tf_copy, i, rel_var)
                    # cre_obj_set_member_info(tf_copy, i, (T_ID_VAR, OBJECT_MBR_ID))




    return tf_copy


@njit(cache=True, locals={"source_idrecs" : u8[::1]})
def encode_relative_to(self, gval_ms, source_idrecs, source_var_ptrs):
    # Ensure that it is up-to-date.
    RelativeEncoder_update(self)
    # Get the inds associated with the source_idrecs
    source_vars = List.empty_list(GenericVarType)
    for i, var_ptr in enumerate(source_var_ptrs):
        source_vars.append(_struct_from_ptr(GenericVarType, var_ptr))

    s_inds = np.empty((len(source_idrecs,)),dtype=np.int32)
    for i, s_idrec in enumerate(source_idrecs):
        if(s_idrec not in self.idrec_to_ind):
            raise ValueError("Source fact not declared to input MemSet.")
        s_inds[i] = self.idrec_to_ind[s_idrec]

    rel_ms = MemSet(self.context_data)

    for fact in gval_ms.get_facts(gval_type):
        fact_copy = copy_cre_obj(fact)

        head = fact.head
        head_t_id, _, _ = decode_idrec(head.idrec)
        is_floating = False
        # For TupleFacts replace each Var member with a relatively encoded one.
        if(head_t_id == T_ID_TUPLE_FACT):
            tf = _cast_structref(TupleFact, head)
            new_tf = rel_encode_tf(self, tf, source_vars, s_inds)
            # print(tf, "->", new_tf)
            fact_copy.head = _cast_structref(CREObjType, new_tf)

        elif(head_t_id == T_ID_VAR):
            v = _cast_structref(GenericVarType, head)
            f_ind = f_ind_from_id(self, v.alias)
            rel_var = get_rel_var(self, f_ind, source_vars, s_inds, v.deref_infos)
            if(rel_var is None): 
                rel_var = v
                is_floating = True
            fact_copy.head = _cast_structref(CREObjType, rel_var)
        


        if(self.keep_floating_facts or not is_floating):
            rel_ms.declare(fact_copy)
    return rel_ms
            

@njit(types.boolean(GenericRelativeEncoderType, MemSetType),cache=True)
def check_same_in_memset(self,in_memset):
    return _raw_ptr_from_struct(self.in_memset) == _raw_ptr_from_struct(in_memset)

@njit(types.void(GenericRelativeEncoderType, MemSetType), cache=True)
def set_in_memset(self, in_memset):
    self.in_memset = in_memset
    self.needs_rebuild = True
    # update_fact_vecs(self)

@njit(MemSetType(GenericRelativeEncoderType),cache=True)
def get_in_memset(self):
    return self.in_memset
