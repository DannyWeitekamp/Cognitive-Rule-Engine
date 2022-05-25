from numba import njit
from cre.utils import decode_idrec 
from cre.context import cre_context 
from cre.memory import Memory 
from cre.flattener import Flattener, get_semantic_visibile_fact_attrs, flattener_update
from cre.fact import define_fact
import pytest_benchmark

def flat_mem_vals(flat_mem):
    from cre.gval import gval
    t_ids = set([decode_idrec(x.idrec)[0] for x in flat_mem.get_facts(gval)])
    values = set([x.val for x in flat_mem.get_facts(gval)])
    return values


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
        a = BOOP1("A", 1)
        b = BOOP1("B", 2)
        c = BOOP2("C", 3, 13)
        d = BOOP2("D", 4, 14)
        e = BOOP3("E", 5, 15, 105)
        f = BOOP3("F", 6, 16, 106)

        a_idrec = mem.declare(a)
        b_idrec = mem.declare(b)
        c_idrec = mem.declare(c)
        d_idrec = mem.declare(d)
        e_idrec = mem.declare(e)
        f_idrec = mem.declare(f)
        print("-------")

        fl = Flattener((BOOP1, BOOP2, BOOP3), mem, id_attr="A")
        
        out_mem = fl.apply()
        values = flat_mem_vals(out_mem)
        print(out_mem)
        print(values)
        assert values == {"A", "B", "C", "D", "E" ,"F", 13., 14., 15., 16., 105.,106.}

        mem.retract(c)
        mem.retract(d)

        out_mem = fl.apply()
        values = flat_mem_vals(out_mem)
        print(values)
        assert values == {"A", "B", "E" ,"F", 15., 16., 105.,106.}

        mem.modify(e, "D", 777)
        mem.modify(f, "A", "Z")

        out_mem = fl.apply()
        values = flat_mem_vals(out_mem)
        print(values)
        assert values == {"A", "B", "E" ,"Z", 15., 16., 777., 106.}



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
        fl = Flattener((BOOP,),in_mem=mem,id_attr="A",)
        fl.update()
        _b_dec_10000(mem)
        
    return (fl,mem), {}

def do_flatten(fl,mem):
    fl.update()
    return fl.out_mem

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
