from numba import njit, f8
from numba.types import unicode_type, boolean
from cre.utils import decode_idrec 
from cre.context import cre_context 
from cre.memory import Memory 
from cre.flattener import Flattener, get_semantic_visibile_fact_attrs, flattener_update
from cre.fact import define_fact
import pytest_benchmark
from cre.default_ops import Equals
from cre.feature_applier import FeatureApplier
from cre.vectorizer import Vectorizer


eq_f8 = Equals(f8, f8)
eq_str = Equals(unicode_type, unicode_type)


def test_vectorizer():
    with cre_context("test_vectorizer") as context:
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
        vr = Vectorizer((f8,unicode_type,boolean))

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
        print(feat_mem)
        
        facts = feat_mem.get_facts()
        # for fact in facts:
        #     print(fact)
        print("LEN", len(facts))
        # print(feat_mem)    
        vr.apply(feat_mem)
        # continuous, nominal = vr.apply(feat_mem)
        # print(continuous, nominal)
        return

        # mem.retract(c)
        # mem.retract(d)

        # flat_mem = fl.apply()
        # feat_mem = fa.apply()
        # print(flat_mem)
        # print(feat_mem)
        # print(len(feat_mem.get_facts()))
        # print(count_true_false(feat_mem))


        # assert len(feat_mem.get_facts()) == ((3*4) + (3*4))
        # assert count_true_false(feat_mem)[0] == 4

        # mem.modify(e, "x", 777)
        # mem.modify(f, "u", "Z")

        # flat_mem = fl.apply()
        # feat_mem = fa.apply()
        # print(flat_mem)
        # print(feat_mem)
        # print(len(feat_mem.get_facts()))

        # print()
        # assert len(feat_mem.get_facts()) == ((3*4) + (3*4))
        # assert count_true_false(feat_mem)[0] == 2

if(__name__ == "__main__"):
    import faulthandler; faulthandler.enable()
    test_vectorizer()
