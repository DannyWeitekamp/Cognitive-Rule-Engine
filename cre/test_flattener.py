from numba import njit
from cre.utils import decode_idrec 
from cre.context import cre_context 
from cre.memset import MemSet 
from cre.flattener import Flattener, get_semantic_visibile_fact_attrs, flattener_update
from cre.fact import define_fact
import pytest_benchmark

def flat_ms_vals(flat_ms):
    from cre.gval import gval
    t_ids = set([decode_idrec(x.idrec)[0] for x in flat_ms.get_facts(gval)])
    values = set([x.val for x in flat_ms.get_facts(gval)])
    return values


def test_flatten():
    with cre_context("test_flatten") as context:
        spec1 = {"A" : {"type" : "string", "visible" : True}, 
                 "B" : {"type" : "number", "visible" : False}}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1,
                 "C" : {"type" : "number", "visible" : True}}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2,
                 "D" : {"type" : "number", "visible" : True}}
        BOOP3 = define_fact("BOOP3", spec3)


        ms = MemSet()
        a = BOOP1("A", 1)
        b = BOOP1("B", 2)
        c = BOOP2("C", 3, 13)
        d = BOOP2("D", 4, 14)
        e = BOOP3("E", 5, 15, 105)
        f = BOOP3("F", 6, 16, 106)

        a_idrec = ms.declare(a)
        b_idrec = ms.declare(b)
        c_idrec = ms.declare(c)
        d_idrec = ms.declare(d)
        e_idrec = ms.declare(e)
        f_idrec = ms.declare(f)
        print("-------")

        fl = Flattener((BOOP1, BOOP2, BOOP3), ms, id_attr="A")
        
        out_ms = fl.apply()
        values = flat_ms_vals(out_ms)
        print(out_ms)
        print(values)
        assert values == {"A", "B", "C", "D", "E" ,"F", 13., 14., 15., 16., 105.,106.}

        ms.retract(c)
        ms.retract(d)

        out_ms = fl.apply()
        values = flat_ms_vals(out_ms)
        print(values)
        assert values == {"A", "B", "E" ,"F", 15., 16., 105.,106.}

        ms.modify(e, "D", 777)
        ms.modify(f, "A", "Z")

        out_ms = fl.apply()
        values = flat_ms_vals(out_ms)
        print(values)
        assert values == {"A", "B", "E" ,"Z", 15., 16., 777., 106.}



# with cre_context("flat") as context:

with cre_context("flatten_10000"):
    spec ={ "A" : {"type" : "string", "visible" : True},
            "B" : {"type" : "number", "visible" : True}
          }
    BOOP = define_fact("BOOP", spec)
            # return (BOOP,), {}

    @njit(cache=True)
    def _b_dec_10000(ms):
        for i in range(10000):
            b = BOOP("HI",i)
            ms.declare(b)

def setup_flatten():
    with cre_context("flatten_10000"):
        ms = MemSet()
        ms.declare(BOOP("HI",-1))
        fl = Flattener((BOOP,),in_memset=ms,id_attr="A",)
        fl.update()
        _b_dec_10000(ms)
        
    return (fl,ms), {}

def do_flatten(fl,ms):
    fl.update()
    return fl.out_memset

def test_b_flatten_10000(benchmark):
    benchmark.pedantic(do_flatten,setup=setup_flatten, warmup_rounds=1, rounds=10)

if(__name__ == "__main__"):
    import faulthandler; faulthandler.enable()
    test_flatten()
    # from cre.utils import PrintElapse
    # fl = setup_flatten()[0][0]
    # with PrintElapse("ABC"):
    #     fl.update()
    # fl = setup_flatten()[0][0]
    # with PrintElapse("ABC"):
    #     fl.update()


# ms.declare(BOOP1("zA", 1))
# ms.declare(BOOP1("zB", 2))
# ms.declare(BOOP2("zC", 3, 13))
# ms.declare(BOOP2("zD", 4, 14))
# ms.declare(BOOP3("zE", 5, 15, 105))
# ms.declare(BOOP3("zF", 6, 16, 106))

# flattener_update(fl)

# print(ms,out_ms)
# flattener_ctor((BOOP1, BOOP2, BOOP3))


# raise ValueError()
