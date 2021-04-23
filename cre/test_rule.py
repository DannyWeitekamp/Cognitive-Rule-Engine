import numpy as np
from numba import njit
from cre.fact import define_fact
from cre.rule import Rule, RuleEngine
from cre.kb import KnowledgeBase
from cre.var import Var
from cre.utils import _pointer_from_struct,_pointer_from_struct_incref


StopLight, StopLightType = define_fact("StopLight",{"color": "string"})

class GreenToYellow(Rule):
    def when():
        s = Var(StopLightType,"l1")
        c = (s.color == "green")
        return c

    def then(kb,s):
        b = s.color
        kb.modify(s,"color",'yellow')
        print(b+ "->" + s.color)
        # kb.declare(BOOP("???",1000+r2.B))
        # kb.halt()


class YellowToRed(Rule):
    def when():
        s = Var(StopLightType,"l1")
        c = (s.color == "yellow")
        return c

    def then(kb,s):
        b = s.color
        kb.modify(s,"color",'red')
        print(b+ "->" + s.color)
        # kb.declare(BOOP("???",1000+r2.B))
        kb.halt()

from time import time_ns

def foo():
    kb = KnowledgeBase()
    kb.declare(StopLight("green"))
    rule_engine = RuleEngine(kb, [GreenToYellow,YellowToRed])
    rule_engine.start()


import timeit
N=100
def time_ms(f):
    f() #warm start
    return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))

print(time_ms(foo))
