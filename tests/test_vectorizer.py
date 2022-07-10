import numpy as np
from numba import njit, f8
from numba.types import unicode_type, boolean
from cre.utils import decode_idrec 
from cre.context import cre_context 
from cre.memset import MemSet 
from cre.transform.flattener import Flattener, flattener_update
from cre.fact import define_fact
import pytest_benchmark
from cre.default_ops import Equals
from cre.transform.feature_applier import FeatureApplier
from cre.transform.vectorizer import Vectorizer
from pprint import pprint



eq_f8 = Equals(f8, f8)
eq_str = Equals(unicode_type, unicode_type)


def test_vectorizer():
    with cre_context("test_vectorizer") as context:
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
        vr = Vectorizer((f8,unicode_type,boolean))

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
        feat_ms = fa()
        # print(feat_ms)
        
        facts = feat_ms.get_facts()
        orig_len = len(facts)
        # for fact in facts:
            # print(fact)
        print("LEN", len(facts))
        # print(feat_ms)    
        # print()

        
        # Make sure that the vector has same size as set of gval facts
        floats, noms = vr(feat_ms)

        print(noms)

        assert len(noms) == orig_len
        assert not np.any(noms == 0)

    
        # Make sure that there are holes when retract facts
        ms.retract(c)
        ms.retract(d)

        flat = fl()
        feat = fa()
        floats, noms = vr(feat_ms)

        assert len(noms) == orig_len
        assert np.any(noms == 0)

        print(noms)

        # Make sure that vectors grow when add new facts
        ms.declare(BOOP3(id="Z",u="q", v=77, q=16, x=106))
        ms.declare(BOOP3(id="W",u="q", v=77, q=17, x=108))

        flat = fl()
        feat = fa()
        floats, noms = vr(feat_ms)

        print(noms)

        assert len(noms) > orig_len
        assert np.any(noms == 0)

        # Make sure that if we refill holes with equivalent facts that 
        #  the vectors' holes are filled
        ms.declare(BOOP2(id="C",u="3", v=3, q=13))
        ms.declare(BOOP2(id="D",u="1", v=4, q=13))

        flat = fl()
        feat = fa()
        floats, noms = vr(feat_ms)

        print(noms)

        assert len(noms) > orig_len
        assert not np.any(noms == 0)




        # Make sure that get_inv_map works
        inv_map = vr.get_inv_map()





        # pprint({k:v for k,v in inv_map.items()})


        # continuous, nominal = vr(feat_ms)
        # print(continuous, nominal)
        # return

        # ms.retract(c)
        # ms.retract(d)

        # flat_ms = fl()
        # feat_ms = fa()
        # print(flat_ms)
        # print(feat_ms)
        # print(len(feat_ms.get_facts()))
        # print(count_true_false(feat_ms))


        # assert len(feat_ms.get_facts()) == ((3*4) + (3*4))
        # assert count_true_false(feat_ms)[0] == 4

        # ms.modify(e, "x", 777)
        # ms.modify(f, "u", "Z")

        # flat_ms = fl()
        # feat_ms = fa()
        # print(flat_ms)
        # print(feat_ms)
        # print(len(feat_ms.get_facts()))

        # print()
        # assert len(feat_ms.get_facts()) == ((3*4) + (3*4))
        # assert count_true_false(feat_ms)[0] == 2

if(__name__ == "__main__"):
    import faulthandler; faulthandler.enable()
    test_vectorizer()
