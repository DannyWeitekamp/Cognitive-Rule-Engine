import numpy as np
import numba
from numba import f8, i8, njit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type
from cre.memset import MemSet
from cre.var import Var
from cre.transform.flattener import Flattener
from cre.transform.feature_applier import FeatureApplier
from cre.transform.relative_encoder import _check_needs_rebuild, RelativeEncoder, get_relational_fact_attrs, next_adjacent
from cre.utils import PrintElapse, deref_info_type
from cre.fact import define_fact
from cre.default_ops import Equals
from cre.context import cre_context
import cre
from numba.core.runtime.nrt import rtsys
import gc
# v1 = Var(Container).children[0]
# v2 = Var(Container).children[1]
# print(v1, v1.deref_infos, v1.deref_infos)
# print(v2, v2.deref_infos, v2.deref_infos)




with cre_context("test_relative_encoder"):
    eq_f8 = Equals(f8, f8)
    eq_str = Equals(unicode_type, unicode_type)


def def_fact_types():
    Component = define_fact("Component", {
        "id" : unicode_type,
        "value" : {"type" : unicode_type, "visible" : True},
        "above" : "Component", "below" : "Component",
        "to_left": "Component", "to_right" : "Component",
        "parents" : "List(Component)"
    })

    Container = define_fact("Container", {
        "inherit_from" : "Component",
        "children" : "List(Component)"
    })

    TestLL = define_fact("TestLL",{
        "id": "string",
        "value" : {"type": "string", "visible" : True},
        "nxt" : "TestLL",
        "prev" : "TestLL",
    })
    return Component, Container, TestLL



def setup_encoder_w_heir_state():
    with cre_context("test_relative_encoder"):
        Component, Container, TestLL = def_fact_types()
        ### Make Structure ### 
        #     p3
        #     p2
        #     p1
        #  [a,b,c]

        a = Component(id="A",value="a")
        b = Component(id="B",value="b")
        c = Component(id="C",value="c")
        a.to_right = b
        b.to_right = c
        b.to_left = a
        c.to_left = b

        p1 = Container(id="P1", children=List([a,b,c]))
        a.parents = List([p1])
        b.parents = List([p1])
        c.parents = List([p1])

        p2 = Container(id="P2", children=List([p1]))
        p1.parents = List([p2])

        p3 = Container(id="P3", children=List([p2]))
        p2.parents = List([p3])

        ms = MemSet()
        ms.declare(a)
        ms.declare(b)
        ms.declare(c)
        ms.declare(p1)
        ms.declare(p2)
        ms.declare(p3)

        fl = Flattener((Component,Container,), ms, id_attr="id")
        flat_ms = fl()
        fa = FeatureApplier([eq_f8,eq_str], flat_ms)
        feat_ms = fa()

        re = RelativeEncoder((Component,Container),ms)
        vs = [Var(Container,'p1'),Var(Container,'p2'),Var(Container,'p3')]
        return (ms,feat_ms,fl,fa,re,[p1,p2,p3], vs), {}



def test_relative_encoder():
    with cre_context("test_relative_encoder"):
        import faulthandler; faulthandler.enable()

        (ms,feat_ms,fl,fa,re,ps,vs),_ = setup_encoder_w_heir_state()
        p1, p2, p3  = ps[0],ps[1],ps[2]
        vp1,vp2,vp2 = vs[0],vs[1],vs[2]

        print(feat_ms)
        re.update()

        with PrintElapse("encode_relative"):
            rel_ms = re.encode_relative_to(feat_ms,[p1], [vp1])
        print(rel_ms)

        with PrintElapse("encode_relative"):
            rel_ms = re.encode_relative_to(feat_ms,[p1,p2,p3], [vp1,vp2,vp2])
        print(rel_ms)
    


def setup_update():
    with cre_context("test_relative_encoder"):
        Component, Container, TestLL = def_fact_types()

        ms = MemSet()


        # init_bytes = used_bytes()  
        
        fl = Flattener((TestLL,), in_memset=ms, id_attr="id",)
        end = first = TestLL(id="q",value="1")
        
        flat_ms = fl()
        fa = FeatureApplier([eq_f8,eq_str], flat_ms)
        feat_ms = fa()

        re = RelativeEncoder((TestLL,), ms, id_attr="id")
        re.update()

        ms.declare(end)
        # print("first", first._meminfo.refcount)



        # init_bytes = used_bytes()
        for i in range(100):
            # erc0 = end._meminfo.refcount
            new_end = TestLL(str(i),str(i),end)
            # erc1, nrc1 = end._meminfo.refcount, new_end._meminfo.refcount 
            # print(f'{erc0}->{erc1}', f'{nrc1}')
            ms.modify(end, "prev",new_end)
            ms.declare(new_end)
            # erc2, nrc2 = end._meminfo.refcount, new_end._meminfo.refcount 
            # print(f'{erc1}->{erc2}', f'{nrc1}->{nrc2}')
            end = new_end

        flat_ms = fl()
        feat_ms = fa()
        feat_ms, fa = None, None

    return (re,ms,end), {}

def setup_update_plus_1():
    with cre_context("test_relative_encoder"):
        Component, Container, TestLL = def_fact_types()
        (re,ms,end),_ = setup_update()
        re.update()
        new_end = TestLL("plus1","plus1",end)
        ms.modify(end, "prev",new_end)
        ms.declare(new_end)
        
        return (re,ms,end), {}    



def do_update(re,ms,end):
    re.update()

def do_encode_rel(ms,feat_ms,fl,fa,re,ps,vs):
    rel_ms = re.encode_relative_to(feat_ms, ps,vs)



def used_bytes(garbage_collect=True):
    if(garbage_collect): gc.collect()
    stats = rtsys.get_allocation_stats()
    # print(stats)
    return stats.alloc-stats.free

# NOTE: Requires revisiting. Definitely partial leak here.
#   The relative encoder seems to be aquiring a reference
#   to each of the facts in the memset at instantiation.
def test_re_mem_leaks():
    with cre_context("test_re_mem_leaks"):
        for i in range(5):
            args, kwargs = setup_update()
            (re,ms,end) = args
            # first = ms.get_facts()[0]

            # print(re._meminfo.refcount)
            # print(ms._meminfo.refcount)
            # print("first", first._meminfo.refcount)
            # print("end", end._meminfo.refcount)
            ms.clear_refs()
            
            # print("Free first", first._meminfo.refcount)
            # print("Free end", end._meminfo.refcount)

            ms =None; gc.collect()

            # print("msnull first", first._meminfo.refcount)
            # print("msnull end", end._meminfo.refcount)

            # NOTE: really ought to be fine when == 0
            if(i <= 1):
                init_bytes = used_bytes()
            else:
                print("<<", used_bytes()-init_bytes)

        # Commenting out for now until larger memleak issue is resolved for facts
        assert used_bytes()-init_bytes == 0


def test_downcasted_heads():
    with cre_context("test_downcasted_heads"):
        Component = define_fact("Component", {
            "id" : str,
            "x" : {"type" : float, "visible" : False},
            "y" : {"type" : float, "visible" : False},
            "width" : {"type" : float, "visible" : False},
            "height" : {"type" : float, "visible" : False},
            "above" : "Component", 
            "below" : "Component",
            "left": "Component", 
            "right" : "Component",
            "parents" : "List(Component)"
        })

        TextField = define_fact("TextField", {
            "inherit_from" : "Component",
            "value" : {"type" : str, "visible" : True, "semantic" : True},
            "locked" : {"type" : bool, "visible" : True},
        })

        Button = define_fact("Button", {
            "inherit_from" : "Component",
        })

        Container = define_fact("Container", {
            "inherit_from" : "Component",
            "children" : "List(Component)"
        })

        a = TextField(id="A",value="a")
        b = TextField(id="B",value="b")
        c = TextField(id="C",value="c")
        a.right = b
        b.right = c
        b.left = a
        c.left = b

        p1 = Container(id="P1", children=List([a,b,c]))
        a.parents = List([p1])
        b.parents = List([p1])
        c.parents = List([p1])

        p2 = Container(id="P2", children=List([p1]))
        p1.parents = List([p2])

        p3 = Container(id="P3", children=List([p2]))
        p2.parents = List([p3])

        ms = MemSet()
        ms.declare(a)
        ms.declare(b)
        ms.declare(c)
        ms.declare(p1)
        ms.declare(p2)
        ms.declare(p3)

        fl = Flattener((TextField,Container,), ms, id_attr="id")
        flat_ms = fl()
        fa = FeatureApplier([eq_f8,eq_str], flat_ms)
        feat_ms = fa()

        print(feat_ms)

        re = RelativeEncoder((TextField,Container),ms)
        vs = [Var(Container,'p1'),Var(Container,'p2'),Var(Container,'p3')]

        vp1,vp2,vp2 = vs[0],vs[1],vs[2]


        print(ms)
        re.update()

        with PrintElapse("encode_relative"):
            rel_ms = re.encode_relative_to(feat_ms,[p1], [vp1])
        print(rel_ms)

        with PrintElapse("encode_relative"):
            rel_ms = re.encode_relative_to(feat_ms,[p1,p2,p3], [vp1,vp2,vp2])
        print(rel_ms)

        with PrintElapse("encode_relative"):
            rel_ms = re.encode_relative_to(feat_ms,[a,c], [Var(TextField,'a'),Var(TextField,'c')])
        print(rel_ms)

    
    
        


def test_b_rel_enc_update_100x100(benchmark):
    with cre_context("test_relative_encoder"):
        benchmark.pedantic(do_update,setup=setup_update, warmup_rounds=1, rounds=10)

def test_b_rel_enc_100x100_update_plus_1(benchmark):
    with cre_context("test_relative_encoder"):
        benchmark.pedantic(do_update,setup=setup_update_plus_1, warmup_rounds=1, rounds=10)


def test_b_rel_enc_100x100_encode(benchmark):
    with cre_context("test_relative_encoder"):
        benchmark.pedantic(do_encode_rel,setup=setup_encoder_w_heir_state, warmup_rounds=1, rounds=10)


if __name__ == "__main__":
    test_re_mem_leaks()

    # test_downcasted_heads()
    # test_relative_encoder()

    # np.set_printoptions(linewidth=1000)
    # test_relative_encoder()
    # (re,ms,_), _ = setup_update()
    # with PrintElapse("Elapse"):
    #     re.update()
    # with PrintElapse("Elapse 2"):
    #     re.update()

    # (re,ms,_), _ = setup_update_plus_1()
    # with PrintElapse("Elapse_plus 1"):
    #     re.update()
