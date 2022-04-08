import time

t0 = time.time_ns()

import numpy as np
from numba import njit
from cre.fact import define_fact
from cre.rule import Rule, RuleEngine
from cre.memory import Memory
from cre.var import Var
from cre.utils import _pointer_from_struct,_pointer_from_struct_incref

t1 = time.time_ns()
print("Import time:",(t1-t0)/1e6)

# raise ValueError()

StopLight = define_fact("StopLight",{"color": "string"})

t2 = time.time_ns()
print("Define Fact time:",(t2-t1)/1e6)

global_value = 11

class GreenToYellow(Rule):
    # cache_then = False
    def when():
        s = Var(StopLight,"l1")
        c = (s.color == "green")
        return c

    def then(mem,s):
        b = s.color
        mem.modify(s,"color",'yellow')
        print(b+ "->" + s.color)
        # print(np.abs(global_value))
        # mem.declare(BOOP("???",1000+r2.B))
        # mem.halt()


class YellowToRed(Rule):
    def when():
        s = Var(StopLight,"l1")
        c = (s.color == "yellow")
        return c

    def then(mem,s):
        b = s.color
        mem.modify(s,"color",'red')
        print(b+ "->" + s.color)
        # mem.declare(BOOP("???",1000+r2.B))
        mem.halt()

t3 = time.time_ns()


print("Def Rules time",(t3-t2)/1e6)


# @njit(cache=True)
def start_up():
    mem = Memory()
    mem.declare(StopLight("green"))
    return mem

mem = Memory()

t4 = time.time_ns()
print("Inst mem",(t4-t3)/1e6)

sl = StopLight("green")

t5 = time.time_ns()
print("Inst StopLight",(t5-t4)/1e6)
mem.declare(sl)
t6 = time.time_ns()
print("Declare",(t6-t5)/1e6)


rule_engine = RuleEngine(mem, [GreenToYellow,YellowToRed])
t7 = time.time_ns()
print("Init Rule Engine",(t7-t6)/1e6)

rule_engine.start()
t8 = time.time_ns()
print("Rule Engine Start",(t8-t7)/1e6)



def foo():
    mem = start_up()
    rule_engine = RuleEngine(mem, [GreenToYellow,YellowToRed])
    rule_engine.start()


import timeit
N=10
def time_ms(f):
    f() #warm start
    return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))

print(time_ms(foo))
