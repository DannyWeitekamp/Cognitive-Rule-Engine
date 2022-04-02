import numpy as np
import numba
from numba import njit, i8,i4,u1, f8, f4
from numba.typed import List, Dict
from numba.types import ListType, DictType
from cre.fact import define_fact
from cre.structref import define_structref
from numba.experimental.structref import new
from cre.memory import Memory,MemoryType
from cre.utils import _cast_structref



import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100, 
    formatter=dict(float=lambda x: "%.3g" % x))

Component, ComponentType = define_fact("Component", {
    "name" : str,
    "above" : "Component", "below" : "Component",
    "to_left": "Component", "to_right" : "Component",
    "parents" : "List(Component)"
    })

Container, ContainerType = define_fact("Container", {
    "inherit_from" : "Component",
    "children" : "List(Component)"
})


# raise ValueError()


# print(mem)

###########
_ind_fltval_pair = np.dtype([("ind", np.int32), ('a_id', np.uint8), ("val", np.float32)])
ind_fltval_pair = numba.from_dtype(_ind_fltval_pair)


bfpm_fields = {
    "wm" : MemoryType,
    "dist_matrix" : ind_fltval_pair[:, ::1],
    "idrec_to_ind" : DictType(i8, i4),
    "visited_inds" : DictType(i4, u1),
    # "covered_inds" : DictType(i4, u1),
    "comps" : ListType(ComponentType),
    "lateral_w" : f8
}

BellmanFordPathMap, BellmanFordPathMapType = define_structref("BellmanFordPathMap", bfpm_fields, define_constructor=False) 


component_fields = list(ComponentType.field_dict.keys())
TO_LEFT_A_ID = component_fields.index("to_left")
TO_RIGHT_A_ID = component_fields.index("to_right")
ABOVE_A_ID = component_fields.index("above")
BELOW_A_ID = component_fields.index("below")
PARENTS_A_ID = component_fields.index("parents")

container_fields = list(ContainerType.field_dict.keys())
CHILDREN_A_ID = container_fields.index("children")


@njit(cache=True)
def new_dist_matrix(n, old_dist_matrix=None):
    dist_matrix = np.empty((n,n),dtype=ind_fltval_pair)
    for i in range(n):
        for j in range(n):
            if(i == j):
                dist_matrix[i,j].ind = 0
                dist_matrix[i,j].a_id = 0
                dist_matrix[i,j].val = 0
            else:
                dist_matrix[i,j].ind = -1
                dist_matrix[i,j].a_id = 0
                dist_matrix[i,j].val = np.inf
    if(old_dist_matrix is not None):
        o_n, o_m = old_dist_matrix.shape
        dist_matrix[:o_n, :o_m] = old_dist_matrix
    return dist_matrix



print("A")
@njit(cache=True)
def BellmanFordPathMap_ctor(wm):
    st = new(BellmanFordPathMapType)
    st.wm = wm
    st.dist_matrix = new_dist_matrix(32)
    # print(st.dist_matrix)
    st.idrec_to_ind = Dict.empty(i8,i4)
    st.visited_inds = Dict.empty(i4,u1)
    # st.covered_inds = Dict.empty(i4,u1)
    st.comps = List.empty_list(ComponentType)
    st.lateral_w = 1.0
    return st


print("B")



# _ind_weight_pair = np.dtype([("ind", np.int32),("weight", np.float32)])
# ind_weight_pair = nb.from_dtype(_ind_weight_pair)


@njit(inline='always')
def _try_append_nxt(bf, comps, inds_ws, obj, a_id, k, w):
    if(obj.idrec not in bf.idrec_to_ind):
        bf.idrec_to_ind[obj.idrec] = len(bf.idrec_to_ind)
        bf.comps.append(obj)
    ind = bf.idrec_to_ind[obj.idrec]
    # if(ind not in bf.covered_inds):
    comps.append(obj)
    inds_ws[k].val = w
    inds_ws[k].a_id = a_id
    inds_ws[k].ind = ind
    
    return k+1
    # return k

print("C")

@njit(cache=True)
def nextAdjacentComponents_IndsWeights(bf, c):
    # Extract indicies for adjacent elements 
    # print("zA")
    if(c.isa(ContainerType)):
        cont = _cast_structref(ContainerType,c)
        adj_inds = np.empty(4+len(cont.parents)+len(cont.children),dtype=ind_fltval_pair)
    else:   
        adj_inds = np.empty(4,dtype=ind_fltval_pair)
    # print("zB")
    adj_comps = List.empty_list(ComponentType)

    k = 0
    if(c.to_left is not None): k = _try_append_nxt(bf, adj_comps, adj_inds, c.to_left, TO_LEFT_A_ID, k, bf.lateral_w)
    if(c.to_right is not None): k = _try_append_nxt(bf, adj_comps, adj_inds, c.to_right, TO_RIGHT_A_ID, k, bf.lateral_w)
    if(c.below is not None): k = _try_append_nxt(bf, adj_comps, adj_inds, c.below, k, BELOW_A_ID, bf.lateral_w)
    if(c.above is not None): k = _try_append_nxt(bf, adj_comps, adj_inds, c.above, k, ABOVE_A_ID, bf.lateral_w)
    # print("zC", c.parents)
    if(c.parents is not None):
        _parents = c.parents
        # print("zC'", c.parents, len(c.parents))
        # for _ in _parents:
        #     print(_)
        for parent in c.parents:
            # print(parent)
            k = _try_append_nxt(bf, adj_comps, adj_inds, parent, PARENTS_A_ID, k, bf.lateral_w)

    if(c.isa(ContainerType)):
        c_cont = _cast_structref(ContainerType,c)
        if(c_cont.children is not None):
            for child in c_cont.children:
                k = _try_append_nxt(bf, adj_comps, adj_inds, child, CHILDREN_A_ID, k, bf.lateral_w)            
    # print("zD")
    return adj_comps, adj_inds[:len(adj_comps)]


@njit(cache=True)
def updateBFRelativeTo(bf , src):
    dist_matrix = bf.dist_matrix
    if(src.idrec not in bf.idrec_to_ind):
        bf.idrec_to_ind[src.idrec] = len(bf.idrec_to_ind)
        bf.comps.append(src)
    s_ind = bf.idrec_to_ind[src.idrec]

    frontier_inds = Dict.empty(i8,u1)
    frontier_inds[i8(s_ind)] = u1(1)
    next_frontier_inds = Dict.empty(i8,u1)

    while(len(frontier_inds) > 0):
        print(": -----")
        print("visited",List(bf.visited_inds.keys()))
        print("frontier",List(frontier_inds.keys()))
        for b_ind in frontier_inds:
            b = bf.comps[b_ind]
            adj_comps, adj_indweights = nextAdjacentComponents_IndsWeights(bf, b)
            for i, (c, c_ind_w_pair) in enumerate(zip(adj_comps, adj_indweights)):
                c_ind = i8(c_ind_w_pair.ind)

                # Skip if does not exist
                if(c_ind == -1): continue
                    
                # If the dist_matrix doesn't have b->c then fill it in
                if(dist_matrix[b_ind, c_ind].ind == -1):
                    #Assign new heading and weights
                    dist_matrix[b_ind, c_ind].ind = c_ind
                    dist_matrix[b_ind, c_ind].a_id = c_ind_w_pair.a_id
                    dist_matrix[b_ind, c_ind].val = c_ind_w_pair.val 

                    next_frontier_inds[c_ind] = u1(1)

                    print("CONNECT:", f'{b.name}', "->", f'{c.name}', c_ind_w_pair.val)

                for a_ind in bf.visited_inds:
                    if( a_ind == b_ind or a_ind == c_ind): continue

                    ab_dist  = dist_matrix[a_ind, b_ind].val 
                    ab_a_id = dist_matrix[a_ind, b_ind].a_id
                    ab_heading = dist_matrix[a_ind, b_ind].ind 
                    new_dist = ab_dist + dist_matrix[b_ind, c_ind].val 
                    if(new_dist < dist_matrix[a_ind, c_ind].val):
                        print("CONNECT:", f'{bf.comps[a_ind].name}', "->", f'{c.name},', "via", f'{b.name}', ":", new_dist, "<", dist_matrix[a_ind, c_ind].val)
                        dist_matrix[a_ind, c_ind].ind = ab_heading
                        dist_matrix[a_ind, c_ind].a_id = ab_a_id
                        dist_matrix[a_ind, c_ind].val = new_dist 
                        next_frontier_inds[c_ind] = u1(1)
                
            bf.visited_inds[b_ind] = u1(1)
            
        frontier_inds = next_frontier_inds
        next_frontier_inds = Dict.empty(i8,u1)


    print(dist_matrix[:len(bf.idrec_to_ind),:len(bf.idrec_to_ind)])

                    

a = Component(name="A")
b = Component(name="B")
c = Component(name="C")
a.to_right = b
b.to_right = c
b.to_left = a
c.to_left = b

p1 = Container(name="P1", children=List([a,b,c]))
a.parents = List([p1])
b.parents = List([p1])
c.parents = List([p1])
print("<<",*[p for p in a.parents])

p2 = Container(name="P2", children=List([p1]))
p1.parents = List([p2])

p3 = Container(name="P3", children=List([p2]))
p2.parents = List([p3])

print(a,b,c)
print(p1)
print(p2)

mem = Memory()
mem.declare(a)
mem.declare(b)
mem.declare(c)
mem.declare(p1)
mem.declare(p2)
mem.declare(p3)
bf = BellmanFordPathMap_ctor(mem)
updateBFRelativeTo(bf, a)


# d = 
# mem.declare(p3)
# print(p1)
# print(p2)




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

_try_append_nxt(bf, comps, inds_ws, obj, k, w):
    ind = bf.idrec_to_ind[obj.idrec]
    if(ind not in bf.covered_inds):
        adj_comps.append(obj)
        adj_inds[k].ind = ind
        adj_inds[k].weight = w
        return k+1
    return k

NextAdjacentComponents_IndsWeights(bf : BellmanFordObj, c : Component):
    //Extract indicies for adjacent elements 
    if(c.isa(ContainerType)):
        cont = _cast_structref(ContainerType,c)
        adj_inds = np.empty(4+len(cont.parents)+len(cont.children),dtype=ind_weight_pair)
    else:   
        adj_inds = np.empty(4,dtype=ind_weight_pair)

    adj_comps = List.empt_list(ComponentType)

    k = 0
    if(c.to_left): k = _try_append_nxt(bf, adj_comps, adj_inds, c.to_left, k, bf.lateral_w)
    if(c.to_right): k = _try_append_nxt(bf, adj_comps, adj_inds, c.to_right, k, bf.lateral_w)
    if(c.below): k = _try_append_nxt(bf, adj_comps, adj_inds, c.below, k, bf.lateral_w)
    if(c.above): k = _try_append_nxt(bf, adj_comps, adj_inds, c.above, k, bf.lateral_w)
        
    for parent in c.parents:
        k = _try_append_nxt(bf, adj_comps, adj_inds, parent, k, bf.lateral_w)

    if(c.isa(Container)):
        for child in _cast_structref(ContainerType,c).children:
            k = _try_append_nxt(bf, adj_comps, adj_inds, child, k, bf.lateral_w)            
    return adj_comps, adj_inds[:len(adj_comps)]


//Need to define
UpdateBFRelativeTo(bf : BellmanFordObj, src : Component):
    dist_matrix = bf.dist_matrix
    s_ind = bf.idrec_to_ind[src.idrec]

    head_components = List([src])
    covered_inds = Dict.empty(i8,u1)
    covered_inds[0] = u1(1)

    next_components = List.empty_list(ComponentType)
    while(len(head_components) > 0):
        for b_ind in head_components:
            adj_comps, adj_indweights = GetAdjacentInds(bf,c)
            for i, c_ind_w_pair in enumerate(adj_indweights):
                c_ind = c_ind_w_pair.ind

                //Skip if does not exist
                if(c_ind == -1): continue

                did_update = False

                item_ind = bf.idrec_to_ind[item.idrec]
                
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
        next_components = List.empty_list(ComponentType)


                    












'''