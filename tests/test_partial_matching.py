import numpy as np
from numba import njit, f8
from numba.typed import List
from cre.conditions import *
from cre.memset import MemSet
from cre.context import cre_context
from cre.utils import used_bytes, NRTStatsEnabled
from cre.utils import PrintElapse, _struct_from_ptr, _list_base,_list_base_from_ptr,_load_ptr, _incref_structref, _raw_ptr_from_struct
from numba.core.runtime.nrt import rtsys
import gc
from cre.partial_matching import partial_match
import pytest
from cre.transform.memset_builder import MemSetBuilder

from cre.matching import check_match, score_match

def test_partial_match():
    with cre_context("test_partial_match"):
        BOOP = define_fact("BOOP",{"name": str, "val" : float})

        bps = []
        ms = MemSet()
        for i in range(10):
            x = BOOP(str(i),i)
            ms.declare(x)
            bps.append(x)

        a = Var(BOOP,"a")
        b = Var(BOOP,"b")
        c = Var(BOOP,"c")

        conds = a & b & c & (a.name != "7") & (a.val < b.val) & (b.val < c.val)

        partial_match(ms, conds)


def test_mc_add_case():
    from test_processing_pipeline import new_mc_addition_state, setup_fact_types
    from cre.matching import repr_match_iter_dependencies

    with cre_context("test_condition_generalizing"):
        (Container, TextField,Component, Button) = setup_fact_types()

        dict_state = new_mc_addition_state(567,354)
        # pprint(dict_state)
        wm = MemSet()
        conv = MemSetBuilder(wm)
        wm, fact_map = conv(dict_state, return_map=True)

        print({decode_idrec(f.idrec)[1] : f.id for f in  fact_map.values()})

        # # -----------------
        # # : Add2
        varz = [Var(TextField,'Sel'), Var(TextField,'Arg0'), Var(TextField,'Arg1')]
        sel_a, arg_a0, arg_a1 = itemgetter("0_answer", "0_upper","0_lower")(fact_map)
        sel_b, arg_b0, arg_b1 = itemgetter("1_answer", "1_upper","1_lower")(fact_map)

        print("-------------------------")
        c_a = Conditions.from_facts([sel_a, arg_a0, arg_a1], varz)

        with PrintElapse("partial_match (load)"):
            partial_match(wm, c_a, None, .2)

        
        

        c_a = Conditions.from_facts([sel_a, arg_a0, arg_a1], varz)
        with PrintElapse("partial_match (first)"):
            partial_match(wm, c_a, None, .2)
        with PrintElapse("partial_match (second)"):
            partial_match(wm, c_a, None, .2)

        with PrintElapse("from py"):
            for score, match in c_a.get_partial_matches(wm,return_scores=True):
                print(score, [m.id for m in match])

        with PrintElapse("from py"):
            for score, match in c_a.get_partial_matches(wm,return_scores=True):
                print(score, [m.id for m in match])
        print()
        with PrintElapse("from py"):
            for score, match in c_a.get_partial_matches(wm,[sel_b, None, None],return_scores=True):
                print(score, [m.id for m in match])

        # # c_a = conditions_from_facts([sel_a, arg_a0, arg_a1], varz)
        # print(repr(c_a))
        # print(repr_match_iter_dependencies(c_a.get_matches(wm)))
        # match_names = [[x.id for x in match][:3] for match in c_a.get_matches(wm)]
        # print(match_names)
        # assert match_names == [['0_answer', '0_upper', '0_lower']]

        # print("-------------------------")
        # c_b = Conditions.from_facts([sel_b, arg_b0, arg_b1], varz)
        # # c_b = conditions_from_facts([sel_b, arg_b0, arg_b1], varz)
        # match_names = [[x.id for x in match][:3] for match in c_b.get_matches(wm)]
        # print(repr(c_b))
        # print(match_names)
        # assert match_names == [['1_answer', '1_upper', '1_lower']]

        # # Generalized verison
        # c_ab = varz[0] & varz[1] & varz[2] & c_a.antiunify(c_b)
        # # print("---------------------")
        # match_names = [[x.id for x in match][:3] for match in c_ab.get_matches(wm)]
        # print(repr(c_ab))
        # print(repr_match_iter_dependencies(c_ab.get_matches(wm)))
        # print(match_names)
        
        # # assert match_names == [['0_answer', '0_upper', '0_lower'], ['1_answer', '1_upper', '1_lower'], ['2_answer', '2_upper', '2_lower']]

        # # Modify the state to make some not match
        # wm.modify(fact_map['0_answer'],'value', '4')
        # wm.modify(fact_map['0_answer'],'locked', True)
        # wm.modify(fact_map['1_answer'],'value', '5')
        # wm.modify(fact_map['1_answer'],'locked', True)

        # print(wm)

        # match_names = [[x.id for x in match][:3] for match in c_ab.get_matches(wm)]
        # print(match_names)
        # # assert match_names == [['2_answer', '2_upper', '2_lower']]

        # # -----------------
        # # : Carry2
        # varz = [Var(TextField,'Sel'), Var(TextField,'Arg0'), Var(TextField,'Arg1')]
        # sel_a, arg_a0, arg_a1 = itemgetter("1_carry", "0_upper","0_lower")(fact_map)
        # sel_b, arg_b0, arg_b1 = itemgetter("2_carry", "1_upper","1_lower")(fact_map)

        
        # c_a = Conditions.from_facts([sel_a, arg_a0, arg_a1], varz)
        # # c_a = conditions_from_facts([sel_a, arg_a0, arg_a1], varz)
        # match_names = [[x.id for x in match][:3] for match in c_a.get_matches(wm)]
        # print("--c_a--")
        # print(repr(c_a))
        # print("----")
        # print(match_names)
        # assert match_names == [['1_carry', '0_upper', '0_lower']]

        # c_b = Conditions.from_facts([sel_b, arg_b0, arg_b1], varz)
        # # c_b = conditions_from_facts([sel_b, arg_b0, arg_b1], varz)
        # match_names = [[x.id for x in match][:3] for match in c_b.get_matches(wm)]
        # print("--c_b--")
        # print(repr(c_b))
        # print("----")
        # print(match_names)
        # assert match_names == [['2_carry', '1_upper', '1_lower']]

        # c_ab = c_a.antiunify(c_b)
        # match_names = [[x.id for x in match][:3] for match in c_ab.get_matches(wm)]
        # print("--c_ab--")
        # print(repr(c_ab))
        # print("----")
        # print(match_names)
        # assert ['1_carry', '0_upper', '0_lower'] in match_names 
        # assert ['2_carry', '1_upper', '1_lower'] in match_names 
        # # NOTE : ['3_carry', '2_lower', '2_upper'] might not be available at this point

        # sel_c, arg_c0, arg_c1 = itemgetter("3_carry", "2_upper","2_lower")(fact_map)
        # c_c = Conditions.from_facts([sel_c, arg_c0, arg_c1], varz)
        # # c_c = conditions_from_facts([sel_c, arg_c0, arg_c1], varz)
        # c_abc = c_ab.antiunify(c_c, drop_unconstr=True)

        # print(str(c_abc))
        
        # match_names = [[x.id for x in match][:3] for match in c_abc.get_matches(wm)]
        # print(match_names)
        # # print(repr_match_iter_dependencies(c_abc.get_matches(wm)))
        
        # assert ['1_carry', '0_upper', '0_lower'] in match_names 
        # assert ['2_carry', '1_upper', '1_lower'] in match_names 
        # assert ['3_carry', '2_upper', '2_lower'] in match_names 

        # # Check matching responds to modify
        # wm.modify(fact_map['1_carry'],'value', '1')
        # wm.modify(fact_map['1_carry'],'locked', True)
        # wm.modify(fact_map['2_carry'],'value', '1')
        # wm.modify(fact_map['2_carry'],'locked', True)

        # match_names = [[x.id for x in match][:3] for match in c_abc.get_matches(wm)]
        # print(match_names)
        # assert match_names == [['3_carry', '2_upper', '2_lower']]
        # # assert ['3_carry', '2_upper', '2_lower'] in match_names 

        # with PrintElapse("Q"):
        #     [match for match in c_abc.get_matches(wm)]
        # print("\n\n\n")

        # # Check matching responds to retractions
        # # wm.modify(fact_map['2_carry'],'value', '')
        # # wm.modify(fact_map['2_carry'],'locked', False)
        # wm.retract(fact_map['3_carry'])
        # wm.modify(fact_map['2_carry'],'to_left', None)
        # wm.modify(fact_map['hidey2'],'above', None)
        # # wm.retract(fact_map['2_carry'])
        # wm.modify(fact_map['1_carry'],'value', '')
        # wm.modify(fact_map['1_carry'],'locked', False)

        # match_names = [[x.id for x in match][:3] for match in c_abc.get_matches(wm)]
        # print(match_names)
        # print({decode_idrec(f.idrec)[1] : f.id for f in  fact_map.values()})
        # assert match_names == [['1_carry', '0_upper', '0_lower']] #??

        # c = Conditions.from_facts([sel_c, arg_c0, arg_c1], varz,
        # # c = conditions_from_facts([sel_c, arg_c0, arg_c1], varz,
        #      neighbor_req_n_adj=2, alpha_flags=("visible", "few_valued"))
        # print(str(c))
        # match_names = [[x.id for x in match][:3] for match in c_abc.get_matches(wm)]
        # print(match_names)
        # assert match_names == [['1_carry', '0_upper', '0_lower']] #??


if(__name__ == "__main__"):
    import faulthandler; faulthandler.enable()
    # test_partial_match()
    test_mc_add_case()


