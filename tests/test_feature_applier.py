from numba import njit, f8, types
from numba.types import unicode_type
from cre.utils import decode_idrec, PrintElapse, _raw_ptr_from_struct
from cre.context import cre_context 
from cre.memset import MemSet, MemSetType 
from cre.fact import define_fact
from cre.var import Var
from cre.builtin_cre_funcs import Equals
from cre.transform.flattener import Flattener, flattener_update
from cre.transform.feature_applier import FeatureApplier
from numba.core.runtime.nrt import rtsys
import gc
from cre.cre_object import copy_cre_obj
import pytest

var_a, var_b = Var(f8,'a'), Var(f8,'b')
eq_f8 = Equals(var_a, var_b)
eq_str = Equals(unicode_type, unicode_type)


# def test_product_iter_w_const():
#     print()

def count_true_false(flat_mem):
    from cre.gval import gval
    t_ids = set([decode_idrec(x.idrec)[0] for x in flat_mem.get_facts(gval)])
    t_count, f_count = 0, 0
    for x in flat_mem.get_facts(gval):
        if(isinstance(x.val,bool) and x.val):
            t_count += 1
        else:
            f_count += 1
    return (t_count, f_count)

@njit(cache=True)
def get_ptr(x):
    return _raw_ptr_from_struct(x)


def test_feature_apply():
    with cre_context("test_feature_apply") as context:
        spec1 = {"id" : {"type" : "string", "visible" : False},
                 "u" : {"type" : "string", "visible" : True}, 
                 "v" : {"type" : "number", "visible" : False}}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1,
                 "q" : {"type" : "number", "visible" : True}}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2,
                 "x" : {"type" : "number", "visible" : True}}
        BOOP3 = define_fact("BOOP3", spec3)
   
        fa = FeatureApplier([eq_f8,eq_str],MemSet())
        fa()


        ms = MemSet()
        a = BOOP1(id="A",u="1", v=1)
        b = BOOP1(id="B",u="2", v=2)
        c = BOOP2(id="C",u="3", v=3, q=13)
        d = BOOP2(id="D",u="1", v=4, q=13)
        e = BOOP3(id="E",u="2", v=5, q=14, x=106)
        f = BOOP3(id="F",u="3", v=6, q=16, x=106)

        a_idrec = ms.declare(a)
        b_idrec = ms.declare(b)
        c_idrec = ms.declare(c)
        d_idrec = ms.declare(d)
        e_idrec = ms.declare(e)
        f_idrec = ms.declare(f)
        print("-------")

        fl = Flattener((BOOP1, BOOP2, BOOP3), ms, id_attr="id")
        flat_ms = fl()

        fa = FeatureApplier([eq_f8, eq_str],flat_ms)
        print("::", eq_f8._meminfo.refcount, var_a._meminfo.refcount, var_b._meminfo.refcount)
        feat_ms = fa()
        print(flat_ms)
        print("::", eq_f8._meminfo.refcount, var_a._meminfo.refcount, var_b._meminfo.refcount)
        print(feat_ms)
        print(len(feat_ms.get_facts()))
        print(count_true_false(feat_ms))

        # 5*5 .u, 2*5 .v, 2*5 .x
        assert len(feat_ms.get_facts()) == ((5*6) + (5*6)) + (6+6)
        assert count_true_false(feat_ms)[0] == 10

        ms.retract(c)
        ms.retract(d)

        flat_ms = fl()
        feat_ms = fa()
        print(flat_ms)
        print(feat_ms)
        print(len(feat_ms.get_facts()))
        print(count_true_false(feat_ms))


        assert len(feat_ms.get_facts()) == ((3*4) + (3*4)) + (4+4)
        assert count_true_false(feat_ms)[0] == 4

        ms.modify(e, "x", 777)
        ms.modify(f, "u", "Z")

        flat_ms = fl()
        feat_ms = fa()
        print(flat_ms)
        print(feat_ms)
        print(len(feat_ms.get_facts()))

        print()
        assert len(feat_ms.get_facts()) == ((3*4) + (3*4)) + (4+4)
        assert count_true_false(feat_ms)[0] == 2

    
def setup_feat_apply_100x100():
    with cre_context("feat_apply_100x100"):
        spec ={ "A" : {"type" : "string", "visible" : True},
            "B" : {"type" : "number", "visible" : True}
              }
        BOOP = define_fact("BOOP", spec)
                # return (BOOP,), {}

        @njit(types.void(MemSetType), cache=True)
        def _b_dec_100(ms):
            for i in range(100):
                b = BOOP(str(i%10),i%5)
                ms.declare(b)

        ms = MemSet()
        _b_dec_100(ms)
        fl = Flattener((BOOP,),ms,id_attr="A")
        flat_ms = fl()
        fa = FeatureApplier([eq_f8,eq_str],flat_ms)
        feat_ms = fa()
        return (fa, feat_ms), {}

def do_feat_apply(fa,ms):
    fa.update()
    return fa.out_memset


def used_bytes(garbage_collect=True):
    if(garbage_collect): gc.collect()
    stats = rtsys.get_allocation_stats()
    # print(stats)
    return stats.alloc-stats.free

def test_fa_mem_leaks():
    with cre_context("test_fa_mem_leaks"):
        spec ={ "A" : {"type" : "string", "visible" : True},
            "B" : {"type" : "number", "visible" : True}
              }
        BOOP = define_fact("BOOP", spec)

        for k in range(5):
            ms = MemSet()
            for i in range(20):
                ms.declare(BOOP(str(i),i))

            fl = Flattener((BOOP,),ms,id_attr="A")
            flat_ms = fl()
            fa = FeatureApplier([eq_f8,eq_str],flat_ms)
            feat_ms = fa()

            if(k <= 1):
                init_bytes = used_bytes()
            else:
                print("<<", used_bytes()-init_bytes)

        assert used_bytes()-init_bytes == 0

@pytest.mark.benchmark(group="transform")
def test_b_feat_apply_100x100(benchmark):
    with cre_context("feat_apply_100x100"):
        benchmark.pedantic(do_feat_apply,setup=setup_feat_apply_100x100, warmup_rounds=1, rounds=10)



if(__name__ == "__main__"):
    import faulthandler; faulthandler.enable()
    # test_fa_mem_leaks()
    # test_product_iter_w_const()
    test_feature_apply()
    # with PrintElapse("elapse"):
    #     do_feat_apply(*setup_feat_apply_100x100()[0])
    # with PrintElapse("elapse"):
    #     do_feat_apply(*setup_feat_apply_100x100()[0])
#     import faulthandler; faulthandler.enable()
#     # from cre.flattener import Flattener
    
    

#         mem = MemSet()
#         for i in range(100):
#             mem.declare(BOOP2(str(i),i,i))

#         fl = Flattener((BOOP1, BOOP2, BOOP3), mem)
#         flat_mem = fl()

#         # print(flat_mem)

#         fa = FeatureApplier([eq_str,eq_f8],flat_mem)
#         with PrintElapse("100x100 str"):
#             feat_mem = fa()

#         fa = FeatureApplier([eq_f8],flat_mem)
#         with PrintElapse("100x100 f8"):
#             feat_mem = fa()

#         # print(feat_mem)

