import numpy as np
import numba
from numba import f8, i8, njit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type
from cre.memory import Memory
from cre.var import Var
from cre.flattener import Flattener
from cre.feature_applier import FeatureApplier
from cre.relative_encoder import _check_needs_rebuild, RelativeEncoder, get_relational_fact_attrs, next_adjacent
from cre.utils import deref_info_type
from cre.fact import define_fact
from cre.default_ops import Equals
import cre
# v1 = Var(Container).children[0]
# v2 = Var(Container).children[1]
# print(v1, v1.deref_infos, v1.deref_infos)
# print(v2, v2.deref_infos, v2.deref_infos)

eq_f8 = Equals(f8, f8)
eq_str = Equals(unicode_type, unicode_type)


Component = define_fact("Component", {
    "id" : unicode_type,
    "value" : {"type" : unicode_type, "is_semantic_visible" : True},
    "above" : "Component", "below" : "Component",
    "to_left": "Component", "to_right" : "Component",
    "parents" : "List(Component)"
    })

Container = define_fact("Container", {
    "inherit_from" : "Component",
    "children" : "List(Component)"
})

deref_infos = Container.get_attr_deref_info("children")
print("<<", type(deref_infos.tolist()))

@njit(cache=True)
def foo():
    arr = np.zeros(1,dtype=deref_info_type)
    arr[0].a_id = deref_infos.a_id
    arr[0].a_id = 200
    print("Foo", arr)

foo()

re = RelativeEncoder((Component,Container))
next_adjacent(re,Component())


np.set_printoptions(linewidth=100000)
# raise ValueError()

# for a,b,c in get_relational_fact_attrs((Component,Container)):
#     print(a,b,c)
# print()

# from cre.utils import np_deref_info_type
# np_dtype = np.dtype([("ind", np.int32), ("dist", np.float32), ('deref_info', np_deref_info_type)])
# dtype = numba.from_dtype(np_dtype)

# a = np.array((1,2,(3,4,5,6)),dtype=dtype)
# print(a)
# print(a.dtype)
# print(dtype)

# @njit(cache=True)
# def foo():
#     a = np.zeros(2, dtype=dtype)
#     a[0].deref_info.a_id = 1
#     print(a)

# foo()

# raise ValueError()


def test_relative_encoder():
    import faulthandler; faulthandler.enable()
    ### Make Structure ### 
    #     p3
    #     p2
    #     p1
    #  [a,b,c]

    a = Component(id="A")
    b = Component(id="B")
    c = Component(id="C")
    a.to_right = b
    b.to_right = c
    b.to_left = a
    c.to_left = b

    p1 = Container(id="P1", children=List([a,b,c]))
    a.parents = List([p1])
    b.parents = List([p1])
    c.parents = List([p1])
    # print("<<",*[p for p in a.parents])

    p2 = Container(id="P2", children=List([p1]))
    p1.parents = List([p2])

    p3 = Container(id="P3", children=List([p2]))
    p2.parents = List([p3])

    # print(a,b,c)
    # print(p1)
    # print(p2)

    mem = Memory()
    mem.declare(a)
    mem.declare(b)
    mem.declare(c)
    mem.declare(p1)
    mem.declare(p2)
    mem.declare(p3)

    re = RelativeEncoder((Component,Container),mem)

    @njit(cache=True)
    def get_changes(re):
        return re.get_changes()

    _check_needs_rebuild(re, get_changes(re))

    fl = Flattener((Component,Container,), mem, id_attr="id")
    flat_mem = fl.apply()
    fa = FeatureApplier([eq_f8,eq_str], flat_mem)
    feat_mem = fa.apply()



    
    # rel_mem = re.encode_relative_to(feat_mem,[p1])

    ### Revise Structure ### 
    #     p4
    #     p3
    #     p2
    #     p1
    #   [a,b,c]

    p4 = Container(id="P4", children=List([p3]))
    p4.parents = List([p3])
    mem.declare(p4)
    mem.modify(p3,'parents', List([p4]))
    print()
    print("-----------4-------------")
    print()

    re.update()
    fl.update()
    fa.update()

    print()
    print("-----------5-------------")
    print()

    # print(flat_mem)


    src_vars = [Var(Container,'p1')]
    print(src_vars)
    re.encode_relative_to(feat_mem,[p1], src_vars)
    # l = re.encode_relative_to(mem,[p1], src_vars)
    # print("<<", l)


    print()
    print("-----------6-------------")
    print()

    # re.encode_relative_to(feat_mem,[p1,p2,p3,p4])

    src_vars = [Var(Container,'p1'),Var(Container,'p2'),Var(Container,'p3'),Var(Container,'p4')]
    re.encode_relative_to(feat_mem,[p1,p2,p3,p4], src_vars)
    # l = re.encode_relative_to(mem,[p1,p2,p3,p4],src_vars)
    # print("<<", l)

if __name__ == "__main__":
    test_relative_encoder()
