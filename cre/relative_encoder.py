import numpy as np
import numba
from numba import njit, i8,i4,u1,u2,u4,u8, f8, f4, generated_jit, types, literal_unroll
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type
from cre.fact import define_fact
from cre.structref import define_structref, define_structref_template
from numba.experimental import structref
from numba.experimental.structref import new, define_attributes
from numba.extending import lower_cast, overload, overload_method
from cre.memory import Memory,MemoryType
from cre.utils import listtype_sizeof_item, _cast_structref, _obj_cast_codegen, DEREF_TYPE_ATTR, DEREF_TYPE_LIST
from cre.vector import VectorType
from cre.incr_processor import IncrProcessorType, ChangeEventType, incr_processor_fields, init_incr_processor
from cre.structref import CastFriendlyStructref, define_boxing
from cre.context import cre_context
from cre.fact import Fact, BaseFact, DeferredFactRefType




import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100, 
    formatter=dict(float=lambda x: "%.3g" % x))

Component = define_fact("Component", {
    "id" : unicode_type,
    "above" : "Component", "below" : "Component",
    "to_left": "Component", "to_right" : "Component",
    "parents" : "List(Component)"
    })

Container = define_fact("Container", {
    "inherit_from" : "Component",
    "children" : "List(Component)"
})


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
    ("attr_ind", np.uint32),
    # ('deref_info1', np_deref_info_type),
    # ('deref_info2', np_deref_info_type),
    ])
ind_dist_deref = numba.from_dtype(np_ind_dist_deref)

relative_encoder_fields = {
    **incr_processor_fields,
    "dist_matrix" : ind_dist_deref[:, ::1],
    "idrec_to_ind" : DictType(u8, i4),
    "visited_inds" : DictType(i4, u1),
    "facts" : ListType(BaseFact),
    "lateral_w" : f8,
    "relational_attrs" : types.Any
}

@structref.register
class RelativeEncoderTypeClass(CastFriendlyStructref):
    pass

GenericRelativeEncoderType = RelativeEncoderTypeClass([(k,v) for k,v in relative_encoder_fields.items()])

from numba.core.typing.typeof import typeof
def get_relative_encoder_type(fact_types):
    relational_fact_attrs = get_relational_fact_attrs(fact_types)
    relational_fact_attrs_type = typeof(relational_fact_attrs)
    field_dict = {**relative_encoder_fields,
                 "relational_attrs" : relational_fact_attrs_type,
                 }
    re_type = RelativeEncoderTypeClass([(k,v) for k,v in field_dict.items()])
    re_type._relational_attrs = relational_fact_attrs
    return re_type

class RelativeEncoder(structref.StructRefProxy):
    def __new__(cls, fact_types, in_mem=None,context=None):
        # Make new in_mem and out_mem if they are not provided.
        if(in_mem is None): in_mem = Memory(context);
        re_type = get_relative_encoder_type(fact_types)
        self = RelativeEncoder_ctor(re_type, in_mem)
        return self

    def encode_relative_to(self, mem, sources):
        source_idrecs = np.array([x.idrec for x in sources],dtype=np.uint64)
        encode_relative_to(self, mem, source_idrecs)

    def update(self):
        relative_encoder_update(self)

define_boxing(RelativeEncoderTypeClass, RelativeEncoder)


@lower_cast(RelativeEncoderTypeClass, GenericRelativeEncoderType)
@lower_cast(RelativeEncoderTypeClass, IncrProcessorType)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)


# component_fields = list(Component.field_dict.keys())
TO_LEFT_A_ID = Component.get_attr_a_id("to_left")#component_fields.index("to_left")
TO_RIGHT_A_ID = Component.get_attr_a_id("to_right")#component_fields.index("to_right")
ABOVE_A_ID = Component.get_attr_a_id("above")#component_fields.index("above")
BELOW_A_ID = Component.get_attr_a_id("below")#component_fields.index("below")
PARENTS_A_ID = Component.get_attr_a_id("parents")#component_fields.index("parents")

# container_fields = list(Container.field_dict.keys())
CHILDREN_A_ID = Container.get_attr_a_id("children")#container_fields.index("children")


@njit(cache=True)
def new_dist_matrix(n, old_dist_matrix=None):
    dist_matrix = np.empty((n,n),dtype=ind_dist_deref)
    for i in range(n):
        for j in range(n):
            if(i == j):
                dist_matrix[i,j].ind = -1
                dist_matrix[i,j].attr_ind = 0
                dist_matrix[i,j].dist = 0
            else:
                dist_matrix[i,j].ind = -1
                dist_matrix[i,j].attr_ind = 0
                dist_matrix[i,j].dist = np.inf
    if(old_dist_matrix is not None):
        o_n, o_m = old_dist_matrix.shape
        dist_matrix[:o_n, :o_m] = old_dist_matrix
    return dist_matrix



# print("A")
@generated_jit(cache=True, nopython=True)
def RelativeEncoder_ctor(re_type, in_mem):
    def impl(re_type, in_mem):
        st = new(re_type)
        init_incr_processor(st, in_mem)
        st.dist_matrix = new_dist_matrix(32)
        st.idrec_to_ind = Dict.empty(i8,i4)
        st.visited_inds = Dict.empty(i4,u1)
        st.facts = List.empty_list(BaseFact)
        st.lateral_w = f8(1.0)
        return st
    return impl


@njit(inline='always')
def _try_append_nxt(re, facts, inds_ws, obj, deref_info1, deref_info2, k):
    if(obj.idrec not in re.idrec_to_ind):
        re.idrec_to_ind[obj.idrec] = len(re.idrec_to_ind)
        re.facts.append(obj)
    ind = re.idrec_to_ind[obj.idrec]
    # if(ind not in re.covered_inds):
    facts.append(obj)
    inds_ws[k].ind = ind
    inds_ws[k].dist = re.lateral_w
    inds_ws[k].deref_info1 = deref_info1
    # inds_ws[k].a_id = a_id
    
    
    return k+1
from cre.fact_intrinsics import fact_lower_getattr

@njit(cache=True)
def fill_adj_inds(re, fact, k, adj_inds, attr_ind):
    idrec = fact.idrec
    if(idrec not in re.idrec_to_ind):
        re.idrec_to_ind[idrec] = i4(len(re.idrec_to_ind))
        re.facts.append(fact)
    ind = re.idrec_to_ind[idrec]

    adj_inds[k].ind = i4(ind)
    adj_inds[k].dist = f4(re.lateral_w)
    adj_inds[k].attr_ind = u4(attr_ind)

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
def next_adjacent(re, fact):
    rel_attrs = []
    n_non_list = 0
    list_attrs = []
    context = cre_context()

    for base_t, attr_t, attr in re._relational_attrs:
        literal_attr = attr
        if(hasattr(attr,'literal_value')): attr = attr.literal_value

        deref_info1 = base_t.get_attr_deref_info(attr)
        deref_info2 = np.zeros((),dtype=deref_info_type)
        is_list = types.literal(False)
        if(isinstance(attr_t,types.ListType)):
            deref_info2['type'] = DEREF_TYPE_LIST
            deref_info2['t_id'] = context.get_t_id(_type=attr_t)
            d1, d2 = deref_info1.tolist(), deref_info2.tolist()
            list_attrs.append((literal_attr, base_t, attr_t, d1, d2))
        else:
            d1, d2 = deref_info1.tolist(), deref_info2.tolist()
            rel_attrs.append((literal_attr, base_t, d1, d2))
        # print(deref_info1, deref_info2)

    list_attrs = tuple(list_attrs)
    rel_attrs = tuple(rel_attrs)

    def impl(re, fact):
        # Get the upper bound on the number of adjacent facts
        n_adj = n_non_list
        for ltup in literal_unroll(list_attrs):        
            attr, base_t, attr_t, d1, d2 = ltup
            if(fact.isa(base_t)):
                typed_fact = _cast_structref(base_t, fact)
                n_adj += len(fact_lower_getattr(typed_fact, attr))
        adj_inds = np.empty(n_adj, dtype=ind_dist_deref)
        adj_facts = List.empty_list(BaseFact)
        k = 0

        attr_ind = 0 
        # Fill in non-list relative attributes
        for tup in literal_unroll(rel_attrs):
            attr, base_t, d1, d2 = tup
            # print(_cast_structref(Component,fact).id, "hasa", attr, fact.isa(base_t))
            if(fact.isa(base_t)):
                typed_fact = _cast_structref(base_t, fact)
                member = fact_lower_getattr(typed_fact, attr)

                if(member is not None):
                    base_member = _cast_structref(BaseFact, member)
                    # print(k)
                    k = fill_adj_inds(re, base_member, k, adj_inds, attr_ind)
                    adj_facts.append(base_member)
            attr_ind += 1
                # print(member)

        # Fill in list relative attributes
        for ltup in literal_unroll(list_attrs):
            attr, base_t, attr_t, d1, d2 = ltup
            # print(_cast_structref(Component,fact).id, "hasa", attr, fact.isa(base_t))
            if(fact.isa(base_t)):
                typed_fact = _cast_structref(base_t, fact)
                member = fact_lower_getattr(typed_fact, attr)
                for i, item in enumerate(member):
                    # print("LIST", i,_cast_structref(Component,item).id)
                    new_d2 = (d2[0],i,d2[1],i*listtype_sizeof_item(attr_t))
                    base_item = _cast_structref(BaseFact,item)
                    # print(k)
                    k = fill_adj_inds(re, base_item, k, adj_inds, attr_ind)
                    adj_facts.append(base_item)
                    # k += 1
            attr_ind += 1
        # print(k)
        # print("END")
        # print("LENS", k, len(adj_facts))
        return adj_facts, adj_inds[:k]
    return impl


@njit(cache=True)
def next_adj_comps_inds_weights(re, c):
    if(c.isa(Container)):
        cont = _cast_structref(Container,c)
        adj_inds = np.empty(4+len(cont.parents)+len(cont.children),dtype=ind_dist_deref)
    else:   
        adj_inds = np.empty(4,dtype=ind_dist_deref)
    adj_comps = List.empty_list(BaseFact)

    k = 0
    if(c.to_left is not None): k = _try_append_nxt(re, adj_comps, adj_inds, c.to_left, TO_LEFT_A_ID, k, re.lateral_w)
    if(c.to_right is not None): k = _try_append_nxt(re, adj_comps, adj_inds, c.to_right, TO_RIGHT_A_ID, k, re.lateral_w)
    if(c.below is not None): k = _try_append_nxt(re, adj_comps, adj_inds, c.below, k, BELOW_A_ID, re.lateral_w)
    if(c.above is not None): k = _try_append_nxt(re, adj_comps, adj_inds, c.above, k, ABOVE_A_ID, re.lateral_w)
    if(c.parents is not None):
        _parents = c.parents
        for parent in c.parents:
            k = _try_append_nxt(re, adj_comps, adj_inds, parent, PARENTS_A_ID, k, re.lateral_w)

    if(c.isa(Container)):
        c_cont = _cast_structref(Container,c)
        if(c_cont.children is not None):
            for child in c_cont.children:
                k = _try_append_nxt(re, adj_comps, adj_inds, child, CHILDREN_A_ID, k, re.lateral_w)            
    return adj_comps, adj_inds[:len(adj_comps)]


@njit(cache=True)
def update_relative_to(re , sources):
    dist_matrix = re.dist_matrix
    frontier_inds = Dict.empty(i8,u1)
    next_frontier_inds = Dict.empty(i8,u1)

    for src in sources:
        if(src.idrec not in re.idrec_to_ind):
            re.idrec_to_ind[src.idrec] = len(re.idrec_to_ind)
            re.facts.append(src)
        s_ind = re.idrec_to_ind[src.idrec]
        frontier_inds[i8(s_ind)] = u1(1)


    while(len(frontier_inds) > 0):
        print(": -----")
        print("visited",List(re.visited_inds.keys()))
        print("frontier",List(frontier_inds.keys()))
        for b_ind in frontier_inds:
            b = re.facts[b_ind]
            adj_facts, adj_ind_dists = next_adjacent(re,b)#next_adj_comps_inds_weights(re, b)
            # print("<<", b_ind, adj_ind_dists)


            # for f in adj_facts:
            #     print("ID", _cast_structref(Component,f).id)
            # print(">>", adj_facts)
            for i, (c, c_ind_dist_pair) in enumerate(zip(adj_facts, adj_ind_dists)):
                # print(i, c_ind_dist_pair)
                c_ind = i8(c_ind_dist_pair.ind)

                # Skip if does not exist
                if(c_ind == -1): continue
                    
                # If the dist_matrix doesn't have b->c then fill it in
                if(dist_matrix[b_ind, c_ind].ind == -1):
                    #Assign new heading and dist
                    dist_matrix[b_ind, c_ind].ind = c_ind
                    # dist_matrix[b_ind, c_ind].a_id = c_ind_w_pair.a_id
                    dist_matrix[b_ind, c_ind].dist = c_ind_dist_pair.dist 

                    next_frontier_inds[c_ind] = u1(1)

                    # print("CONNECT:", f'{b_ind}', "->", f'{c_ind}', c_ind_dist_pair.dist)
                    # print("CONNECT:", f'{b.id}', "->", f'{c.id}', c_ind_w_pair.dist)

                for a_ind in re.visited_inds:
                    if( a_ind == b_ind or a_ind == c_ind): continue

                    ab_dist  = dist_matrix[a_ind, b_ind].dist 
                    # ab_a_id = dist_matrix[a_ind, b_ind].a_id
                    ab_heading = dist_matrix[a_ind, b_ind].ind 
                    new_dist = ab_dist + dist_matrix[b_ind, c_ind].dist 
                    if(new_dist < dist_matrix[a_ind, c_ind].dist):
                        # print("CONNECT:", f'{re.comps[a_ind].id}', "->", f'{c.id},', "via", f'{b.id}', ":", new_dist, "<", dist_matrix[a_ind, c_ind].dist)
                        dist_matrix[a_ind, c_ind].ind = ab_heading
                        # dist_matrix[a_ind, c_ind].a_id = ab_a_id
                        dist_matrix[a_ind, c_ind].dist = new_dist 
                        next_frontier_inds[c_ind] = u1(1)

            # print("A", b_ind)
                
            re.visited_inds[b_ind] = u1(1)
        # print("C")
        frontier_inds = next_frontier_inds
        next_frontier_inds = Dict.empty(i8,u1)

    # print("B")
    print(dist_matrix[:len(re.idrec_to_ind),:len(re.idrec_to_ind)])


@generated_jit(cache=True)
@overload_method(RelativeEncoderTypeClass,'get_changes')
def incr_pr_accumulate_change_events(self, end=-1, exhaust_changes=True):
    def impl(self, end=-1, exhaust_changes=True):
        incr_pr = _cast_structref(IncrProcessorType, self)
        return incr_pr.get_changes(end=end, exhaust_changes=exhaust_changes)
    return impl


@njit(cache=True)
def relative_encoder_update(rel_enc):
    # cq = re.mem.mem_data.change_queue
    # incr_pr = _cast_structref(IncrProcessorType, re)
    # changes = incr_pr.get_changes()
    # changes = incr_pr_accumulate_change_events(incr_pr)
    sources = List.empty_list(BaseFact)
    for change_event in rel_enc.get_changes():
        print("CHANGE",change_event)
        comp = rel_enc.in_mem.get_fact(change_event.idrec, BaseFact)
        sources.append(comp)
    update_relative_to(rel_enc, sources)    


@njit(cache=True)
def _closest_source_ind(self,ind,s_inds):
    # Find the closest source
    min_s_ind = -1
    min_dist = np.inf
    for s_ind in s_inds:
        dist = self.dist_matrix[ind][s_ind].dist
        if(dist < min_dist): 
            min_dist = dist
            min_s_ind = s_ind

    return min_s_ind





@njit(cache=True)
def encode_relative_to(self, mem, source_idrecs):
    # print(source_idrecs)
    s_inds = np.empty((len(source_idrecs,)),dtype=np.int64)
    for i, s_idrec in enumerate(source_idrecs):
        s_inds[i] = self.idrec_to_ind[s_idrec]

    seq_buffer = np.empty((len(self.dist_matrix),),dtype=np.int64)
    for fact in mem.get_facts(BaseFact):
        # print(fact,fact.idrec)
        # print(self.idrec_to_ind)
        ind = f_ind = self.idrec_to_ind[fact.idrec]
        s_ind = _closest_source_ind(self,f_ind, s_inds)
        # print("closest", s_ind)

        k = 0
        while(True):
            # print(s_ind, ind)
            
            dm_entry = self.dist_matrix[ind][s_ind]
            # print(ind_dist_deref)
            ind = dm_entry.ind
            # a_id = dm_entry.a_id

            if(ind == -1):break
            seq_buffer[k] = ind
            k += 1

            
        print(fact.idrec, seq_buffer[:k])







        
        

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
    ind = re.idrec_to_ind[obj.idrec]
    if(ind not in re.covered_inds):
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
    if(c.to_left): k = _try_append_nxt(re, adj_comps, adj_inds, c.to_left, k, re.lateral_w)
    if(c.to_right): k = _try_append_nxt(re, adj_comps, adj_inds, c.to_right, k, re.lateral_w)
    if(c.below): k = _try_append_nxt(re, adj_comps, adj_inds, c.below, k, re.lateral_w)
    if(c.above): k = _try_append_nxt(re, adj_comps, adj_inds, c.above, k, re.lateral_w)
        
    for parent in c.parents:
        k = _try_append_nxt(re, adj_comps, adj_inds, parent, k, re.lateral_w)

    if(c.isa(Container)):
        for child in _cast_structref(Container,c).children:
            k = _try_append_nxt(re, adj_comps, adj_inds, child, k, re.lateral_w)            
    return adj_comps, adj_inds[:len(adj_comps)]


//Need to define
updateRERelativeTo(re : BellmanFordObj, src : Component):
    dist_matrix = re.dist_matrix
    s_ind = re.idrec_to_ind[src.idrec]

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

                item_ind = re.idrec_to_ind[item.idrec]
                
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












'''
