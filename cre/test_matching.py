from numbert.experimental.condition_node import *
from numbert.experimental.kb import KnowledgeBase
from numbert.experimental.context import kb_context
from numbert.experimental.matching import conditions_get_matches


def test_matching():
    with kb_context("test_link"):
        BOOP, BOOPType = define_fact("BOOP",{"A": "string", "B" : "number"})
        l1, l2 = Var(BOOPType,"l1"), Var(BOOPType,"l2")
        r1, r2 = Var(BOOPType,"r1"), Var(BOOPType,"r2")

        c = (l1.B > 0) & (l1.B != 3) & (l1.B < 4) & (l2.B != 3) | \
            (l1.B == 3) 

        kb = KnowledgeBase()
        kb.declare(BOOP("0", 0))
        kb.declare(BOOP("1", 1))
        kb.declare(BOOP("2", 2))
        kb.declare(BOOP("3", 3))
        kb.declare(BOOP("4", 4))

        cl = get_linked_conditions_instance(c, kb)

        conditions_get_matches(cl)

        print("----------")

        c = (l1.B <= 1) & (l1.B < l2.B) & (l2.B <= r1.B)

        cl = get_linked_conditions_instance(c, kb)

        conditions_get_matches(cl)

















if(__name__ == "__main__"):
    test_matching()
