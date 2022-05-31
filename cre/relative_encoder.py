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
from cre.utils import  lower_setattr, _ptr_from_struct_incref, decode_idrec, listtype_sizeof_item, _cast_structref, _obj_cast_codegen, DEREF_TYPE_ATTR, DEREF_TYPE_LIST, _memcpy_structref
from cre.vector import VectorType
from cre.incr_processor import IncrProcessorType, ChangeEventType, incr_processor_fields, init_incr_processor
from cre.structref import CastFriendlyStructref, define_boxing
from cre.context import cre_context
from cre.fact import Fact, BaseFact, DeferredFactRefType
from cre.gval import gval as gval_type
from cre.var import GenericVarType


def _get_base_type(fact_type):
    parent = None
    while(hasattr(fact_type,'parent_type')):
        parent = fact_type = fact_type.parent_type
    return parent

def assert_mutual_base(fact_types, id_attr):
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
        for attr, d in ft.spec.items():
            relational = False
            attr_t = d['type']
            attr_t = attr_t.instance_type if (isinstance(attr_t, (types.TypeRef,DeferredFactRefType))) else attr_t
            if(isinstance(attr_t, Fact)): relational = True
            if(isinstance(attr_t, types.ListType)):
                item_t = attr_t.item_type
                item_t = item_t.instance_type if (isinstance(item_t, (types.TypeRef,DeferredFactRefType))) else item_t
                if(isinstance(item_t, Fact)): relational = True
                attr_t = types.ListType(item_t)

            relational = d.get("relational", relational)
                
            if(relational):
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
    "dist_matrix" : ind_dist_deref[:, ::1],
    "idrec_to_ind" : DictType(u8, i4),
    "visited_inds" : DictType(i8, u1),
    "facts" : ListType(BaseFact),
    "deref_info_matrix" : deref_info_type[:,::1],
    "lateral_w" : f8,
    "id_to_ind" : DictType(unicode_type, i4),
    "var_cache" : DictType(Tuple((i4,i4,i4)),GenericVarType),
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
                 "relational_attrs" : relational_fact_attrs_type,
                 "base_fact_type" : types.TypeRef(assert_mutual_base(fact_types, id_attr)), 
                 "id_attr" : types.literal(id_attr),
                 }
    re_type = RelativeEncoderTypeClass([(k,v) for k,v in field_dict.items()])
    re_type._relational_attrs = relational_fact_attrs
    return re_type

class RelativeEncoder(structref.StructRefProxy):
    def __new__(cls, fact_types, in_ms=None, id_attr='id', context=None):
        # Make new in_ms and out_ms if they are not provided.
        if(in_ms is None): in_ms = MemSet(context);
        re_type = get_relative_encoder_type(fact_types,id_attr)
        self = RelativeEncoder_ctor(re_type, in_ms)
        return self

    def encode_relative_to(self, ms, sources, source_vars):
        if(isinstance(source_vars,list)):
            _source_vars = List.empty_list(GenericVarType)
            for v in source_vars:
                _source_vars.append(v)
            source_vars = _source_vars
        source_idrecs = np.array([u8(x.idrec) for x in sources],dtype=np.uint64)
        # print("source_idrecs", [decode_idrec(x) for x in source_idrecs])
        return encode_relative_to(self, ms, source_idrecs, source_vars)

    def update(self):
        RelativeEncoder_update(self)

define_boxing(RelativeEncoderTypeClass, RelativeEncoder)


@lower_cast(RelativeEncoderTypeClass, GenericRelativeEncoderType)
@lower_cast(RelativeEncoderTypeClass, IncrProcessorType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)


# # component_fields = list(Component.field_dict.keys())
# TO_LEFT_A_ID = Component.get_attr_a_id("to_left")#component_fields.index("to_left")
# TO_RIGHT_A_ID = Component.get_attr_a_id("to_right")#component_fields.index("to_right")
# ABOVE_A_ID = Component.get_attr_a_id("above")#component_fields.index("above")
# BELOW_A_ID = Component.get_attr_a_id("below")#component_fields.index("below")
# PARENTS_A_ID = Component.get_attr_a_id("parents")#component_fields.index("parents")

# # container_fields = list(Container.field_dict.keys())
# CHILDREN_A_ID = Container.get_attr_a_id("children")#container_fields.index("children")


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



@generated_jit(cache=True)
def _new_deref_info_matrix(self):
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
    # print("___________")
    # print(rel_derefs)
    # print(list_derefs)

    def impl(self):
        # Fill in the deref_info_matrix which we use to build Var instances 
        deref_info_matrix = np.zeros((n_attrs,2),dtype=deref_info_type)
        attr_id = 0
        for i in range(len(rel_derefs)):
            d1 = rel_derefs[i]
            deref_info_matrix[attr_id][0].type = d1[0]
            deref_info_matrix[attr_id][0].a_id = d1[1]
            deref_info_matrix[attr_id][0].t_id = d1[2]
            deref_info_matrix[attr_id][0].offset = d1[3]
            attr_id += 1
        for i in range(len(list_derefs)):
            attr_t, d1, d2 = list_derefs[i]
            # print("___________")
            # print(d1,d2)
            deref_info_matrix[attr_id][0].type = d1[0]
            deref_info_matrix[attr_id][0].a_id = d1[1]
            deref_info_matrix[attr_id][0].t_id = d1[2]
            deref_info_matrix[attr_id][0].offset = d1[3]
            deref_info_matrix[attr_id][1].type = d2[0]
            deref_info_matrix[attr_id][1].a_id = 0
            deref_info_matrix[attr_id][1].t_id = d2[2]
            deref_info_matrix[attr_id][1].offset = listtype_sizeof_item(attr_t)
            attr_id += 1
        # print(deref_info_matrix)
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
@njit(cache=True, nopython=True)
def RelativeEncoder_ctor(re_type, in_ms):
    # def impl(re_type, in_ms):
    st = new(re_type)
    init_incr_processor(st, in_ms)
    RelativeEncoder_reinit(st)
    st.lateral_w = f8(1.0)
    st.deref_info_matrix = _new_deref_info_matrix(st)
    
    return st
    # return impl


# @njit(inline='always')
# def _try_append_nxt(re, facts, inds_ws, obj, deref_info1, deref_info2, k):
#     if(obj.idrec not in self.idrec_to_ind):
#         self.idrec_to_ind[obj.idrec] = len(self.idrec_to_ind)
#         self.facts.append(obj)
#     ind = self.idrec_to_ind[obj.idrec]
#     # if(ind not in self.covered_inds):
#     facts.append(obj)
#     inds_ws[k].ind = ind
#     inds_ws[k].dist = self.lateral_w
#     inds_ws[k].deref_info1 = deref_info1
#     # inds_ws[k].a_id = a_id
    
    
    return k+1
from cre.fact_intrinsics import fact_lower_getattr

@njit(cache=True)
def fill_adj_inds(self, fact, k, adj_inds, attr_id, item_ind):
    # idrec = fact.idrec
    # if(idrec not in self.idrec_to_ind):
    #     ind = i4(len(self.idrec_to_ind))
    #     self.idrec_to_ind[idrec] = ind
    #     self.id_to_ind[fact_lower_getattr(fact, self.id_attr)] = ind
    #     self.facts.append(fact)
    ind = self.idrec_to_ind[fact.idrec]

    adj_inds[k].ind = i4(ind)
    adj_inds[k].dist = f4(self.lateral_w)
    adj_inds[k].attr_id = u4(attr_id)
    adj_inds[k].item_ind = i4(item_ind)

    return k+1
    # adj_inds[k].deref_info1.type = u1(d1[0])
    # adj_inds[k].deref_info1.a_id = u4(d1[1])
    # adj_inds[k].deref_info1.t_id = u2(d1[2])
    # adj_inds[k].deref_info1.offset = i4(d1[3])
    # adj_inds[k].deref_info2.type = u1(d2[0])
    # adj_inds[k].deref_info2.a_id = u4(d2[1])
    # adj_inds[k].deref_info2.t_id = u2(d2[2])
    # adj_inds[k].deref_info2.offset = i4(d2[3])


@generated_jit(cache=True,nopython=True)
def next_adjacent(self, fact):
    rel_attrs, list_attrs = [], []
    context = cre_context()

    for base_t, attr_t, attr in self._relational_attrs:
        literal_attr = attr
        if(hasattr(attr,'literal_value')): attr = attr.literal_value
        if(isinstance(attr_t,types.ListType)):
            list_attrs.append((literal_attr, base_t))
        else:
            rel_attrs.append((literal_attr, base_t))

    list_attrs = tuple(list_attrs)
    rel_attrs = tuple(rel_attrs)

    def impl(self, fact):
        # Make a buffer to fill adj_inds
        n_adj = len(rel_attrs)
        for ltup in literal_unroll(list_attrs):        
            attr, base_t = ltup
            if(fact.isa(base_t)):
                typed_fact = _cast_structref(base_t, fact)
                n_adj += len(fact_lower_getattr(typed_fact, attr))
        adj_inds = np.empty(n_adj, dtype=ind_dist_deref)
        # adj_facts = List.empty_list(self.base_fact_type)
        k = 0

        attr_id = 0 
        # Fill in non-list relative attributes
        for tup in literal_unroll(rel_attrs):
            attr, base_t = tup
            if(fact.isa(base_t)):
                typed_fact = _cast_structref(base_t, fact)
                member = fact_lower_getattr(typed_fact, attr)

                if(member is not None):
                    base_member = _cast_structref(self.base_fact_type, member)
                    k = fill_adj_inds(self, base_member, k, adj_inds, attr_id, -1)
                    # adj_facts.append(base_member)
            attr_id += 1
                # print(member)

        # Fill in list relative attributes
        for ltup in literal_unroll(list_attrs):
            attr, base_t = ltup
            if(fact.isa(base_t)):
                typed_fact = _cast_structref(base_t, fact)
                # Member is a list so go through it's items
                member = fact_lower_getattr(typed_fact, attr)
                for i, item in enumerate(member):
                    base_item = _cast_structref(self.base_fact_type,item)
                    k = fill_adj_inds(self, base_item, k, adj_inds, attr_id, i4(i))
                    # adj_facts.append(base_item)
            attr_id += 1
        # return adj_facts, adj_inds[:len(adj_facts)]
        return adj_inds[:k]
    return impl


# @njit(cache=True)
# def next_adj_comps_inds_weights(re, c):
#     if(c.isa(Container)):
#         cont = _cast_structref(Container,c)
#         adj_inds = np.empty(4+len(cont.parents)+len(cont.children),dtype=ind_dist_deref)
#     else:   
#         adj_inds = np.empty(4,dtype=ind_dist_deref)
#     adj_comps = List.empty_list(BaseFact)

#     k = 0
#     if(c.to_left is not None): k = _try_append_nxt(re, adj_comps, adj_inds, c.to_left, TO_LEFT_A_ID, k, self.lateral_w)
#     if(c.to_right is not None): k = _try_append_nxt(re, adj_comps, adj_inds, c.to_right, TO_RIGHT_A_ID, k, self.lateral_w)
#     if(c.below is not None): k = _try_append_nxt(re, adj_comps, adj_inds, c.below, k, BELOW_A_ID, self.lateral_w)
#     if(c.above is not None): k = _try_append_nxt(re, adj_comps, adj_inds, c.above, k, ABOVE_A_ID, self.lateral_w)
#     if(c.parents is not None):
#         _parents = c.parents
#         for parent in c.parents:
#             k = _try_append_nxt(re, adj_comps, adj_inds, parent, PARENTS_A_ID, k, self.lateral_w)

#     if(c.isa(Container)):
#         c_cont = _cast_structref(Container,c)
#         if(c_cont.children is not None):
#             for child in c_cont.children:
#                 k = _try_append_nxt(re, adj_comps, adj_inds, child, CHILDREN_A_ID, k, self.lateral_w)            
#     return adj_comps, adj_inds[:len(adj_comps)]


@njit(cache=True)
def update_relative_to(self , sources):
    dist_matrix = self.dist_matrix
    frontier_inds = Dict.empty(i8,u1)
    next_frontier_inds = Dict.empty(i8,u1)

    for src in sources:
        # if(src.idrec not in self.idrec_to_ind):
        #     ind = len(self.idrec_to_ind)
        #     self.idrec_to_ind[src.idrec] = ind
        #     self.id_to_ind[fact_lower_getattr(src, self.id_attr)] = ind
        #     self.facts.append(src)
        s_ind = self.idrec_to_ind[src.idrec]
        frontier_inds[i8(s_ind)] = u1(1)


    while(len(frontier_inds) > 0):
        print(": -----")
        print("visited",List(self.visited_inds.keys()))
        print("frontier",List(frontier_inds.keys()))
        for b_ind in frontier_inds:
            b = self.facts[b_ind]
            # adj_ind_dists = next_adjacent(self,b)#next_adj_comps_inds_weights(re, b)

            for c_ind_dist in next_adjacent(self,b):

            # for i,c in enumerate(adj_facts):
                # c_ind_dist = adj_ind_dists[i]
                c_ind = i8(c_ind_dist.ind)
                # print(c_ind_dist, b_ind, c_ind)

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

                    # print("CONNECT:", f'{b_ind}', "->", f'{c_ind}', c_ind_dist_pair.dist)
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

            # print("A", b_ind)
                
            self.visited_inds[b_ind] = u1(1)
        # print("C")
        frontier_inds = next_frontier_inds
        next_frontier_inds = Dict.empty(i8,u1)

    # print("B")
    print(dist_matrix[:len(self.idrec_to_ind),:len(self.idrec_to_ind)])


@generated_jit(cache=True)
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
            if(ce.was_retracted): 
                need_rebuild = True; break;
            if(ce.was_modified):
                fact = self.in_memset.get_fact(ce.idrec, self.base_fact_type)
                for tup in literal_unroll(base_a_id_pairs):
                    base_t, a_id = tup
                    if(a_id in ce.a_ids and fact.isa(base_t)):
                        need_rebuild = True; break;
        return need_rebuild
    return impl


@njit(cache=True)
def _insert_fact(self,fact):
    ind = i4(len(self.idrec_to_ind))
    self.idrec_to_ind[fact.idrec] = ind
    self.id_to_ind[fact_lower_getattr(fact, self.id_attr)] = ind
    self.facts.append(fact)


@njit(cache=True)
def RelativeEncoder_update(self):
    change_events = self.get_changes()
    # Need rebuild if there were any retractions or modifications to
    #  'relative' attributes. TODO: rebuilds could probably triggered
    #   more conservatively.
    need_rebuild = _check_needs_rebuild(self,change_events)

    if(need_rebuild):
        # If need_rebuild start bellman-ford shortest path from scratch
        RelativeEncoder_reinit(self)
        sources = self.in_memset.get_facts(self.base_fact_type)
        for fact in sources:
            _insert_fact(self, fact)
        update_relative_to(self, sources)    
    else:
        # Otherwise we can do an incremental update
        sources = List.empty_list(self.base_fact_type)
        for change_event in change_events:
            if(change_event.was_declared or change_event.was_modified):
                fact = self.in_memset.get_fact(change_event.idrec, self.base_fact_type)
                if(change_event.was_declared): _insert_fact(self,fact)
                sources.append(fact)
        update_relative_to(self, sources)    


    


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
            deref_info_buffer[k].t_id = deref_infos[1].t_id
            deref_info_buffer[k].offset = dm_entry.item_ind*deref_infos[1].offset
            k += 1

    # Add any extra deref_infos. e.g.'.value' at the end of a deref chain.
    if(extra_derefs is not None):
        for i in range(len(extra_derefs)):
            deref_info_buffer[k:k+1] = extra_derefs[i:i+1]
            k += 1

    # Make the Var
    head_t_id = deref_info_buffer[k-1].t_id if(k > 0) else s_var.base_t_id
    return  _new_rel_var(s_var, head_t_id, deref_info_buffer[:k].copy())

@njit(cache=True,locals={"tup": var_cache_key_type})
def rel_var_for_id(self, id_str, source_idrecs, source_vars, s_inds=None, extra_derefs=None):
    '''Builds a Var using the relative encoder for the fact instance with
         id==id_str. Var has dereference chain that starts at the closest source. '''
    if(s_inds is None):
        s_inds = np.empty((len(source_idrecs,)),dtype=np.int32)
        for i, s_idrec in enumerate(source_idrecs):
            s_inds[i] = self.idrec_to_ind[s_idrec]

    if(id_str not in self.id_to_ind): 
        print(f"No fact with {self.id_attr}={id_str}")
        raise ValueError(f"Tried to re-encode fact with an id unknown to the RelativeEncoder.")

    f_ind = self.id_to_ind[id_str]
    s_ind, s_i = _closest_source(self, f_ind, s_inds)
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
from cre.cre_object import copy_cre_obj, cre_obj_iter_t_id_item_ptrs, cre_obj_set_item, CREObjType
from cre.tuple_fact import TupleFact
from cre.utils import _load_ptr, _struct_from_ptr

@njit(cache=True, locals={"source_idrecs" : u8[::1]})
def encode_relative_to(self, ms, source_idrecs, source_vars):
    # Get the inds associated with the source_idrecs
    s_inds = np.empty((len(source_idrecs,)),dtype=np.int32)
    for i, s_idrec in enumerate(source_idrecs):
        s_inds[i] = self.idrec_to_ind[s_idrec]

    # mem = MemSet()

    # l = List.empty_list(GenericVarType)
    # seq_buffer = np.empty((len(self.dist_matrix),),dtype=np.int64)
    # deref_info_buffer = np.empty((2*len(self.dist_matrix),),dtype=deref_info_type)
    for fact in ms.get_facts(gval_type):
        fact_copy = copy_cre_obj(fact)

        head = fact.head
        head_t_id, _, _ = decode_idrec(head.idrec)
        if(head_t_id == T_ID_TUPLE_FACT):
            tf = _cast_structref(TupleFact, head)
            tf_copy = copy_cre_obj(tf)
            

            for i, (t_id, m_id, data_ptr) in enumerate(cre_obj_iter_t_id_item_ptrs(tf)):
                if(t_id == T_ID_VAR):
                    v = _struct_from_ptr(GenericVarType, _load_ptr(i8,data_ptr))
                    rel_var = rel_var_for_id(self, v.alias, source_idrecs, source_vars, s_inds, v.deref_infos)
                    cre_obj_set_item(tf_copy, i, rel_var)
                    # print("\t", rel_var)
            fact_copy.head = _cast_structref(CREObjType, tf_copy)
            # print("TF", tf)
            # print(">>", fact_copy)

                


        elif(head_t_id == T_ID_VAR):
            v = _cast_structref(GenericVarType, head)
            rel_var = rel_var_for_id(self, v.alias, source_idrecs, source_vars, s_inds, v.deref_infos)
            fact_copy.head = _cast_structref(CREObjType, rel_var)
            # print("Var", v)
        print(">>", fact_copy)
            



        # print(fact.head)
        # fact_id = fact_lower_getattr(fact, self.id_attr)
        # rel_var = rel_var_for_id(self, fact_id, source_idrecs, source_vars, s_inds)

        # print(rel_var.deref_infos)
        # l.append(rel_var)
        # return rel_var
        # print("DONE")
        # print(fact_id, rel_var)
        # print(rel_var.idrec)
    # return l 
        # print(fact,fact.idrec)
        # print(self.idrec_to_ind)
        # ind = f_ind = self.idrec_to_ind[fact.idrec]
        # s_ind = _closest_source_ind(self, f_ind, s_inds)

        # k = 0
        # while(True):
        #     dm_entry = self.dist_matrix[ind][s_ind]
        #     ind = dm_entry.ind

        #     if(ind == -1):break
        #     # seq_buffer[k] = ind
        #     # k += 1

        #     # attr_id = dm_entry.attr_id
        #     # item_ind = dm_entry.item_ind


        #     deref_infos = self.deref_info_matrix[dm_entry.attr_id]
        #     # print(attr_id, deref_infos)
        #     deref_info_buffer[k:k+1] = deref_infos[0:1]
        #     k += 1
        #     if(deref_infos[1].type == DEREF_TYPE_LIST):
        #         deref_info_buffer[k].type = DEREF_TYPE_LIST
        #         deref_info_buffer[k].a_id = dm_entry.item_ind
        #         deref_info_buffer[k].t_id = deref_infos[1].t_id
        #         deref_info_buffer[k].offset = dm_entry.item_ind*deref_infos[1].offset
        #         k += 1

        # base_t_id,_,_ = decode_idrec(source_idrecs[s_ind])
        # head_t_id = base_t_id
        # if(k > 0):
        #     head_t_id = deref_info_buffer[k-1].t_id

        # print(base_t_id, head_t_id, deref_info_buffer[:k])

            
        # print(fact.idrec, seq_buffer[:k])







        
        

    # print("<<HEAD:",cq.head)

# import logging
# logging.basicConfig(level=logging.DEBUG)
# numba_logger = logging.getLogger('numba')
# numba_logger.setLevel(logging.INFO)




'''PLANNING PLANNING PLANNING

Incremental bellman-ford for free floating objects
Input: a bunch of objects with adjacency relationships, left, right, above, below, parent
Output: For each pair of objects, the effective minimum distance needed to follow
        adjacencies from A->B.

Incremental Input: More objects

Thoughts, it may make sense to keep around all paths of equivalent distance.
But, favor relationships that include parent->child jumps. For instance

c1 c2 c3
-- -- --
A1 A2 A3
B1 B2 B3
C1 C2 C3

B2 from A1 can be represented as
-A1.r.b    (2 steps)
-A1.b.r    (2 steps)
-A1.p.r[1] (3 steps)
-A1.p.r[-2] (3 steps)

Considering these relationships, some of the parent child relationships are fragile
for instance in the case of:

c1 c2 c3
-- -- --
   Z2   
A1 A2 A3
B1 B2 B3
C1 C2 C3
   C3

The first two will succeed, the last two will fail.

Now lets consider an example from algebra:

          Eq 2x + 1 = 5

add-expr 2x + 1;        op =; expr 5

term 2x; op +; term 1;

coef-term 2; var-term x

What would the 'selection' look like in this case... the whole next line?
I suppose it must be, we'll call it "NL". Then coef from NL is.

NL.a.lexpr.terms[0].coef

So from these two examples it isn't necessarily true that the parent relationships 
need be preferred, as long as in this case we haven't parsed the state in a
weird way. 

What if we consider going up the tree to be almost free. Then any way down the
tree from a parent will be free-ish. For long division:


128/5

      2 ? 
    -----
5 | 1 2 8
    1 0  
    -----
      2 8
      2 5
      ---


One way to break this down is that each line divides a table, which 
can either be viewed as a row or column first table. The whole thing,
is then a sort of bifurcated table of tables. 

It seems that upward traversals are tricky (there can be multiple parents), 
so we should probably just consider all objects found along an upward traversal 
to be sources. This makes this a little bit tricky, but perhaps a little bit easier, 
since for tree-like structures the bellman-ford will be trivial. 

So in this case it's fine if the BF data structure is not symmetric
some child objects will not have paths to their parents, but parents
will have paths to their children. So what happens then in this case 
when the where-learning mechanism generalizes away the problem root?
I think in principle the root should always stick around, it should
be possible for there to be holes in the tree, so long as decendants
of a common ancestor are used.

It seems this means we will need in our feature space fairly
general predicates such as is_ancestor(A, B), is_decendant(A, B), etc.


Speaking of predicates what would an invented predicate look like.
The classic case is:
ancestor(A,C) =: ancestor(A,B) ^ child(B,C)
ancestor(A,B) =: child(A,B)

This needs to be determined by a unification process right???
How much would we get out of recursion here? Seems computing this,
let alone learning this would be complex. 


Seems like some kind of use of loop-like statements that evaluate to
objects or lists would be more useful like:
All_Until(X, Pred(X), Next(X)) -> [x for x in generator(X, Next) if Pred(x)]

But to feasibly consider using such a statement, we need some heuristics.

Consider the problems:

1000
-  7
-----

10000
-   7
-----

Where we need to borrow from previous digits


--------

Back to Bellman-Ford 3/24

Typical bellmanford
1. Fill in src as zero others as inf
2. Go through all verticies |V| - 1 times trying to find better connections
3. Check for negative weights (we don't need this)

For our case we need to find the shortest dist from everything to everything else
-The distance from thing to itself is 0
-Should fill in a Dict[idrec]-> (Dict[idrec] -> Struct(dist,heading))
-Should be a procedure on Component class, might need to check if Component is a Container


BFClass_fields = {
       
}



dist_ind_pair = ...

_try_append_nxt(re, comps, inds_ws, obj, k, w):
    ind = self.idrec_to_ind[obj.idrec]
    if(ind not in self.covered_inds):
        adj_comps.append(obj)
        adj_inds[k].ind = ind
        adj_inds[k].weight = w
        return k+1
    return k

NextAdjacentComponents_IndsWeights(re : BellmanFordObj, c : Component):
    //Extract indicies for adjacent elements 
    if(c.isa(Container)):
        cont = _cast_structref(Container,c)
        adj_inds = np.empty(4+len(cont.parents)+len(cont.children),dtype=ind_weight_pair)
    else:   
        adj_inds = np.empty(4,dtype=ind_weight_pair)

    adj_comps = List.empt_list(Component)

    k = 0
    if(c.to_left): k = _try_append_nxt(re, adj_comps, adj_inds, c.to_left, k, self.lateral_w)
    if(c.to_right): k = _try_append_nxt(re, adj_comps, adj_inds, c.to_right, k, self.lateral_w)
    if(c.below): k = _try_append_nxt(re, adj_comps, adj_inds, c.below, k, self.lateral_w)
    if(c.above): k = _try_append_nxt(re, adj_comps, adj_inds, c.above, k, self.lateral_w)
        
    for parent in c.parents:
        k = _try_append_nxt(re, adj_comps, adj_inds, parent, k, self.lateral_w)

    if(c.isa(Container)):
        for child in _cast_structref(Container,c).children:
            k = _try_append_nxt(re, adj_comps, adj_inds, child, k, self.lateral_w)            
    return adj_comps, adj_inds[:len(adj_comps)]


//Need to define
updateRERelativeTo(re : BellmanFordObj, src : Component):
    dist_matrix = self.dist_matrix
    s_ind = self.idrec_to_ind[src.idrec]

    head_components = List([src])
    covered_inds = Dict.empty(i8,u1)
    covered_inds[0] = u1(1)

    next_components = List.empty_list(Component)
    while(len(head_components) > 0):
        for b_ind in head_components:
            adj_comps, adj_indweights = GetAdjacentInds(re,c)
            for i, c_ind_w_pair in enumerate(adj_indweights):
                c_ind = c_ind_w_pair.ind

                //Skip if does not exist
                if(c_ind == -1): continue

                did_update = False

                item_ind = self.idrec_to_ind[item.idrec]
                
                ab_dist, ab_heading = dist_matrix[b_ind][c_ind]

                if(ab_heading == -1):
                    dist_matrix[b_ind, c_ind] = (c_ind_w_pair.weight, c_ind)
                    dist_matrix[c_ind, b_ind] = (c_ind_w_pair.weight, b_ind)
    
                for a_ind in covered_inds:
                    ca_dist, ca_heading = dist_matrix[a_ind][b_ind]
                    n_dist = ca_dist + c_ind_w_pair.weight
                    if(n_dist < ca_dist):
                        dist_matrix[a_ind, c_ind] = (n_dist, ca_heading)
                        dist_matrix[c_ind, a_ind] = (n_dist, b_ind)
                
                next_components.append(c_ind)

            covered_inds[b_ind] = u1(1)


            
                
        head_components = next_components
        next_components = List.empty_list(Component)


######## PLANNING PLANNING PLANNING 4/7 ####### 

What is an incremental processor? Why do we need one when we have rules?

An incremental processor allows us to do incremental processing in a more imperative manner
than rules. We can directly force an incremental processor to update from python, keep/update datastructures
within it that are cumbersome or impossible to distribute across facts stored in a working memory.
For instance datastructures like hash maps, queues etc. aren't currently supported in CRE facts. An imperative
incremental processor that lives outside of a ruleset also alleviates the complexity of fitting 




5/29
Gvals don't keep a reference to the object that they came from...
But ought they? Probably would make sense. Although raises questions of 
reference lifetime... are gvals meant to be serializable? Need all
fact references be within the same MemSet? Not sure it matters if 
I don't intend to apply matching Conditions on that MemSet. 

Ultimately we are in a situation of needing to map from either the idrec or id
to an index. 








'''
