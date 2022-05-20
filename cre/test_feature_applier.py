from numba import njit, f8, types
from numba.types import unicode_type
from cre.utils import decode_idrec, PrintElapse
from cre.context import cre_context 
from cre.memory import Memory 
from cre.flattener import Flattener, get_semantic_visibile_fact_attrs, flattener_update
from cre.fact import define_fact
import pytest_benchmark
from cre.default_ops import Equals
from cre.feature_applier import FeatureApplier

eq_f8 = Equals(f8, f8)
eq_str = Equals(unicode_type, unicode_type)


# def test_product_iter_w_const():
#     print()

def count_true_false(flat_mem):
    from cre.gval import gval
    t_ids = set([decode_idrec(x.idrec)[0] for x in flat_mem.get_facts(gval)])
    t_count, f_count = 0, 0
    for x in flat_mem.get_facts(gval):
        if(x.val):
            t_count += 1
        else:
            f_count += 1
    return (t_count, f_count)


def test_feature_apply():
    with cre_context("test_feature_apply") as context:
        spec1 = {"id" : {"type" : "string", "is_semantic_visible" : False},
                 "u" : {"type" : "string", "is_semantic_visible" : True}, 
                 "v" : {"type" : "number", "is_semantic_visible" : False}}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1,
                 "q" : {"type" : "number", "is_semantic_visible" : True}}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2,
                 "x" : {"type" : "number", "is_semantic_visible" : True}}
        BOOP3 = define_fact("BOOP3", spec3)
   
        fa = FeatureApplier([eq_f8,eq_str],Memory())
        fa.apply()


        mem = Memory()
        a = BOOP1(id="A",u="1", v=1)
        b = BOOP1(id="B",u="2", v=2)
        c = BOOP2(id="C",u="3", v=3, q=13)
        d = BOOP2(id="D",u="1", v=4, q=13)
        e = BOOP3(id="E",u="2", v=5, q=14, x=106)
        f = BOOP3(id="F",u="3", v=6, q=16, x=106)

        a_idrec = mem.declare(a)
        b_idrec = mem.declare(b)
        c_idrec = mem.declare(c)
        d_idrec = mem.declare(d)
        e_idrec = mem.declare(e)
        f_idrec = mem.declare(f)
        print("-------")

        fl = Flattener((BOOP1, BOOP2, BOOP3), "id", mem)
        flat_mem = fl.apply()

        fa = FeatureApplier([eq_f8, eq_str],flat_mem)
        feat_mem = fa.apply()
        print(flat_mem)
        print(feat_mem)
        print(len(feat_mem.get_facts()))
        print(count_true_false(feat_mem))

        # 5*5 .u, 2*5 .v, 2*5 .x
        assert len(feat_mem.get_facts()) == ((5*6) + (5*6))
        assert count_true_false(feat_mem)[0] == 10

        mem.retract(c)
        mem.retract(d)

        flat_mem = fl.apply()
        feat_mem = fa.apply()
        print(flat_mem)
        print(feat_mem)
        print(len(feat_mem.get_facts()))
        print(count_true_false(feat_mem))


        assert len(feat_mem.get_facts()) == ((3*4) + (3*4))
        assert count_true_false(feat_mem)[0] == 4

        mem.modify(e, "x", 777)
        mem.modify(f, "u", "Z")

        flat_mem = fl.apply()
        feat_mem = fa.apply()
        print(flat_mem)
        print(feat_mem)
        print(len(feat_mem.get_facts()))

        print()
        assert len(feat_mem.get_facts()) == ((3*4) + (3*4))
        assert count_true_false(feat_mem)[0] == 2


with cre_context("feat_apply_100x100"):
    spec ={ "A" : {"type" : "string", "is_semantic_visible" : True},
            "B" : {"type" : "number", "is_semantic_visible" : True}
          }
    BOOP = define_fact("BOOP", spec)
            # return (BOOP,), {}

    @njit(cache=True)
    def _b_dec_100(mem):
        for i in range(100):
            b = BOOP(str(i%10),i%5)
            mem.declare(b)

def setup_feat_apply_100x100():
    with cre_context("feat_apply_100x100"):
        mem = Memory()
        _b_dec_100(mem)
        fl = Flattener((BOOP,),"A",mem)
        flat_mem = fl.apply()
        fa = FeatureApplier([eq_f8,eq_str],flat_mem)
        feat_mem = fa.apply()
    return (fa, feat_mem), {}

def do_feat_apply(fa,mem):
    fa.update()
    return fa.out_mem

def test_b_feat_apply(benchmark):
    benchmark.pedantic(do_feat_apply,setup=setup_feat_apply_100x100, warmup_rounds=1, rounds=10)



if(__name__ == "__main__"):
    import faulthandler; faulthandler.enable()
    # test_product_iter_w_const()
    test_feature_apply()
    # with PrintElapse("elapse"):
    #     do_feat_apply(*setup_feat_apply_100x100()[0])
    # with PrintElapse("elapse"):
    #     do_feat_apply(*setup_feat_apply_100x100()[0])
#     import faulthandler; faulthandler.enable()
#     # from cre.flattener import Flattener
    
    

#         mem = Memory()
#         for i in range(100):
#             mem.declare(BOOP2(str(i),i,i))

#         fl = Flattener((BOOP1, BOOP2, BOOP3), mem)
#         flat_mem = fl.apply()

#         # print(flat_mem)

#         fa = FeatureApplier([eq_str,eq_f8],flat_mem)
#         with PrintElapse("100x100 str"):
#             feat_mem = fa.apply()

#         fa = FeatureApplier([eq_f8],flat_mem)
#         with PrintElapse("100x100 f8"):
#             feat_mem = fa.apply()

#         # print(feat_mem)

