import numpy as np
from numba import njit
from cre.fact import define_fact
from cre.rule import Rule, RuleEngine
from cre.kb import KnowledgeBase
from cre.var import Var
from cre.condition_node import NOT
from cre.utils import _pointer_from_struct

@njit(cache=True)
def ptr(x):
    return _pointer_from_struct(x)



Button, ButtonType = define_fact("Button", {"name": str})
TextField, TextFieldType = define_fact("TextField", {
    "name": str,
    "value" : str,
    "to_left" : "TextField",
    "to_right" : "TextField",
    "above" : "TextField",
    "below" : "TextField",
    "enabled" : float
 })

PhaseHandler, PhaseHandlerType = define_fact("PhaseHandler", {
    "phase" : float,
    "cycle" : float
})




Match, MatchType = define_fact("Match", {
    "skill" : str,
    "rhs" : str,
    "sel" : str,
    "action" : str,
    "input" : str,
    "args" : str, # < Should be list
    "full_fired" : float,
    "is_correct" : float,
})

ConflictSet, ConflictSetType = define_fact("ConflictSet", {
    "conflict_set" : float, # < Should be list
})


CustomField, CustomFieldType = define_fact("CustomField", {
    "name" : str, 
    "value" : str, 
})

interfaceElement, interfaceElementType = define_fact("interfaceElement", {
    "name" : str, 
    "value" : str,
})


StudentValues, StudentValuesType = define_fact("StudentValues", {
    "selection" : str, 
    "action" : str,
    "input" : str,
})


TPA, TPAType = define_fact("TPA", {
    "selection" : str, 
    "action" : str,
    "input" : str,
})

Hint, HintType = define_fact("Hint", {
    "precedence" : float, 
    "msg" : str,
})


IsHintMatch, IsHintMatchType = define_fact("IsHintMatch",{})


Skill, SkillType = define_fact("Skill",{
    "name" : str,
    "category" : str,
})


# Phase Control flow: 0)Start-> 1)Match-> 2)Resolve-> 3)Check-> 4)Report-> Reset
class startMatching(Rule):
    # cache_then = False
    def when():
        return (Var(PhaseHandler,"ph").phase == 0)

    def then(kb, ph):
        # TODO: clear conflict set
        print("START MATCHING")
        # kb.focus("phase1")
        kb.modify(ph, "phase", 1)   

class startResolving(Rule):
        # cache_then = False
    def when():
        return (Var(PhaseHandler,"ph").phase == 1)

    def then(kb, ph):
        # TODO: clear conflict set
        print("START Resolving")
        # kb.focus("phase2")
        kb.modify(ph, "phase", 2)


class startChecking(Rule):
        # cache_then = False
    def when():
        return (Var(PhaseHandler,"ph").phase == 2)

    def then(kb, ph):
        # TODO: clear conflict set
        print("START Checking")
        # kb.focus("phase3")
        kb.modify(ph, "phase", 3)

class reportCorrectness(Rule):
        # cache_then = False
    def when():
        return (Var(PhaseHandler,"ph").phase == 3)

    def then(kb, ph):
        print("START Correctness")
        # kb.focus("phase4")
        kb.modify(ph, "phase", 4)

class resetting(Rule):
        # cache_then = False
    def when():
        return (Var(PhaseHandler,"ph").phase == 4) & \
               Var(ConflictSet,"cfs");

    def then(kb, ph, cfs):
        print("START Reset")
        # kb.focus("phase0")
        # kb.modify(cfs,'conflict_set',conflict_set);
        kb.modify(ph, "phase", 0)
        kb.modify(ph, "cycle", ph.cycle+1)
        kb.halt();

##############  RULES START #################


class Add2(Rule):
    def when():
        ph = Var(PhaseHandler,"ph")
        sel, arg0, arg1 = Var(TextField,'sel'), Var(TextField,'arg1'), Var(TextField,'arg2')
        c = ph & \
            (sel.enabled == True) & \
            (arg0.enabled == True) & (arg0.value != "") & \
            (arg1.enabled == True) & (arg1.value != "") & \
            (sel.above == arg1) & \
            (arg0.below == arg1) & \
            (arg1.above == arg0) & \
            (arg1.below == sel)
        return c

    def then(kb, ph, sel, arg0, arg1):
        v = "?"#str((float(arg0.value) + float(arg1.value)) % 10);
        if(True): # used to check for NaN?
            print("Add2", v, sel.name, arg0.name, arg1.name);
            match = Match("Add2","Add2", sel.name, "UpdateTextField",
                         v,
                         # [arg0.name,arg1.name]);
                         arg0.name + "," + arg1.name)
            kb.declare(match);


class resolveAdd2(Rule):
    def when():
        match = Var(Match,'match')
        sel = Var(TextField,'sel')
        # sel_r = NOT(Var(TextField,'sel_r'))
        # m1 = NOT(Var(Match,'m1'))
        # m2 = NOT(Var(Match,'m2'))
        # m3 = NOT(Var(Match,'m3'))
        c = (match.rhs == "Add2") & (match.full_fired == False) & \
            (match.sel == sel.name) & \
            NOT(TextField,'sel_r') & (sel.to_right == sel_r) & (sel_r.enabled == True) & \
            NOT(Match,'m1') & (m1.rhs == "Carry2") & (m1.sel == sel.above.above.above.name) & (m1.input == "1") & \
            NOT(Match,'m2') & (m2.rhs == "Carry3") & (m2.sel == sel.above.above.above.name) & (m2.input == "1") & \
            NOT(Match,'m3') & (m3.rhs == "Add3") & (m3.input == "1")#(m3.sel == sel.above.above.above) & 
        print(repr(c))
        print([ptr(var) for var in c.vars])
        return c

    def then(kb, match, sel):
        kb.modify(match,"full_fired",1)
        print("ResolveAdd2")




def bootstrap():
    kb = KnowledgeBase()
    kb.declare(PhaseHandler())
    kb.declare(ConflictSet())
    return kb

kb = bootstrap()
r_eng = RuleEngine(kb,[startMatching,startResolving,startChecking,reportCorrectness,resetting,Add2, resolveAdd2])
r_eng.start()
