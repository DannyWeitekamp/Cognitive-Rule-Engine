from numba import njit
from cre.utils import decode_idrec 
from cre.context import cre_context 
from cre.memory import Memory 
from cre.flattener import Flattener, get_semantic_visibile_fact_attrs, flattener_update
from cre.fact import define_fact
import pytest_benchmark


def test_flatten():
    with cre_context("test_flatten") as context:
        spec1 = {"A" : {"type" : "string", "is_semantic_visible" : True}, 
                 "B" : {"type" : "number", "is_semantic_visible" : False}}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1,
                 "C" : {"type" : "number", "is_semantic_visible" : True}}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2,
                 "D" : {"type" : "number", "is_semantic_visible" : True}}
        BOOP3 = define_fact("BOOP3", spec3)


        mem = Memory()
        mem.declare(BOOP1("A", 1))
        mem.declare(BOOP1("B", 2))
        mem.declare(BOOP2("C", 3, 13))
        mem.declare(BOOP2("D", 4, 14))
        mem.declare(BOOP3("E", 5, 15, 105))
        mem.declare(BOOP3("F", 6, 16, 106))
        print(mem)

        fl = Flattener((BOOP1, BOOP2, BOOP3), mem)
        out_mem = fl.apply()
        out_mem = fl.apply()

        from cre.gval import gval
        t_ids = set([decode_idrec(x.idrec)[0] for x in out_mem.get_facts(gval)])
        print(t_ids,gval.t_id)
        values = set([x.val for x in out_mem.get_facts(gval)])
        print(values)
        


        
        print(fl)
        print("-------")
        print(out_mem)
        print(out_mem)
        print(repr(out_mem))

        assert values == {"A", "B", "C", "D", "E" ,"F", 13., 14., 15., 16., 105.,106.}

# with cre_context("flat") as context:

with cre_context("flatten_10000"):
    spec ={ "A" : {"type" : "string", "is_semantic_visible" : True},
            "B" : {"type" : "number", "is_semantic_visible" : True}
          }
    BOOP = define_fact("BOOP", spec)
            # return (BOOP,), {}

    @njit(cache=True)
    def _b_dec_10000(mem):
        for i in range(10000):
            b = BOOP("HI",i)
            mem.declare(b)

def setup_flatten():
    with cre_context("flatten_10000"):
        mem = Memory()
        mem.declare(BOOP("HI",-1))
        fl = Flattener((BOOP,),mem)
        fl.update()
        _b_dec_10000(mem)
        
    return (fl,mem), {}

def do_flatten(fl,mem):
    fl.update()
    return fl.out_mem

def test_b_flatten(benchmark):
    benchmark.pedantic(do_flatten,setup=setup_flatten, warmup_rounds=1, rounds=10)

if(__name__ == "__main__"):
    test_flatten()
    # from cre.utils import PrintElapse
    # fl = setup_flatten()[0][0]
    # with PrintElapse("ABC"):
    #     fl.update()
    # fl = setup_flatten()[0][0]
    # with PrintElapse("ABC"):
    #     fl.update()


# mem.declare(BOOP1("zA", 1))
# mem.declare(BOOP1("zB", 2))
# mem.declare(BOOP2("zC", 3, 13))
# mem.declare(BOOP2("zD", 4, 14))
# mem.declare(BOOP3("zE", 5, 15, 105))
# mem.declare(BOOP3("zF", 6, 16, 106))

# flattener_update(fl)

# print(mem,out_mem)
# flattener_ctor((BOOP1, BOOP2, BOOP3))


# raise ValueError()
