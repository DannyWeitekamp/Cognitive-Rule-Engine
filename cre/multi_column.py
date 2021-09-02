import numpy as np
from numba import njit
from numba.typed import List
from cre.fact import define_fact
from cre.rule import Rule, RuleEngine
from cre.memory import Memory
from cre.var import Var
from cre.conditions import NOT
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
    "args" : "ListType(str)", # < Should be list
    "full_fired" : float,
    "is_correct" : float,
})
print(Match.spec)

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

# match = Var(Match,'match')
# sel = Var(TextField,'sel')
# (match.sel == sel.name)
# raise ValueError()

# Phase Control flow: 0)Start-> 1)Match-> 2)Resolve-> 3)Check-> 4)Report-> Reset
class startMatching(Rule):
    # cache_then = False
    def when():
        return (Var(PhaseHandler,"ph").phase == 0)

    def then(mem, ph):
        # TODO: clear conflict set
        print("START MATCHING",ph.phase)
        # mem.focus("phase1")
        mem.modify(ph, "phase", 1)   

class startResolving(Rule):
        # cache_then = False
    def when():
        return (Var(PhaseHandler,"ph").phase == 1)

    def then(mem, ph):
        # TODO: clear conflict set
        print("START Resolving")
        # mem.focus("phase2")
        mem.modify(ph, "phase", 2)


class startChecking(Rule):
        # cache_then = False
    def when():
        return (Var(PhaseHandler,"ph").phase == 2)

    def then(mem, ph):
        # TODO: clear conflict set
        print("START Checking")
        # mem.focus("phase3")
        mem.modify(ph, "phase", 3)

class reportCorrectness(Rule):
        # cache_then = False
    def when():
        return (Var(PhaseHandler,"ph").phase == 3)

    def then(mem, ph):
        print("START Correctness")
        # mem.focus("phase4")
        mem.modify(ph, "phase", 4)

class resetting(Rule):
        # cache_then = False
    def when():
        return (Var(PhaseHandler,"ph").phase == 4) & \
               Var(ConflictSet,"cfs");

    def then(mem, ph, cfs):
        print("START Reset")
        # mem.focus("phase0")
        # mem.modify(cfs,'conflict_set',conflict_set);
        mem.modify(ph, "phase", 0)
        mem.modify(ph, "cycle", ph.cycle+1)
        mem.halt();

##############  RULES START #################


class Add2(Rule):
    def when():
        ph = Var(PhaseHandler,"ph")
        sel, arg0, arg1 = Var(TextField,'sel'), Var(TextField,'arg1'), Var(TextField,'arg2')
        return (
            (ph.phase == 1) & 
            (sel.enabled == True) & 
            (arg0.enabled == False) & (arg0.value != "") & 
            (arg1.enabled == False) & (arg1.value != "") & 
            (sel.above == arg1) & 
            (arg0.below == arg1) &
            (arg1.above == arg0) & 
            (arg1.below == sel)
        )

    def then(mem, ph, sel, arg0, arg1):
        # print("APPLESAUCE")
        v = "?"#str((float(arg0.value) + float(arg1.value)) % 10);
        if(True): # used to check for NaN?
            print("Add2", v, sel.name, arg0.name, arg1.name);
            match = Match(skill="Add2",rhs="Add2", sel=sel.name, action="UpdateTextField",
                         input=v,
                         args=List([arg0.name,arg1.name]),
                         full_fired=False);
                         # arg0.name + "," + arg1.name)
            mem.declare(match);


class ResolveAdd2(Rule):
    def when():
        ph = Var(PhaseHandler,"ph")
        match = Var(Match,'match')
        sel = Var(TextField,'sel')
        return (
            (ph.phase == 2) & match & sel & 
            (match.rhs == "Add2") & (match.full_fired == False) & 
            (match.sel == sel.name) & 
            NOT(TextField,'sel_r') & (sel.to_right == sel_r) & (sel_r.enabled == True) & 
            NOT(Match,'m1') & (m1.rhs == "Carry2") & (m1.sel == sel.above.above.above.name) & (m1.input == "1") & 
            NOT(Match,'m2') & (m2.rhs == "Carry3") & (m2.sel == sel.above.above.above.name) & (m2.input == "1") & 
            NOT(Match,'m3') & (m3.rhs == "Add3") #& (m3.args[0] == sel.above.above.above.name)#(m3.sel == sel.above.above.above) & 
        )

    def then(mem, ph, match, sel):
        mem.modify(match,"full_fired",True)
        print("ResolveAdd2", match.sel, sel.name)

class Add3(Rule):
    def when():
        return (
            Var(PhaseHandler,'ph') & 
            # Sel editable, args not editable or empty
            Var(TextField,'sel') & (sel.enabled == True) & 
            Var(TextField,'arg0') & (arg0.enabled == False) & (arg0.value != "") & 
            Var(TextField,'arg1') & (arg1.enabled == False) & (arg1.value != "") & 
            Var(TextField,'arg2') & (arg2.enabled == False) & (arg2.value != "") &
            # Arranged vertically w/ sel on bottom
            (arg0.below == arg1) & (arg1.above == arg0) & 
            (arg1.below == arg2) & (arg2.above == arg1) & 
            (sel.above == arg2) & (arg2.below == sel)
        )

    def then(mem, ph, sel, arg0, arg1, arg2):
        v = "?"#String((Number(arg0.value) + Number(arg1.value) + Number(arg2.value)) % 10);
        # if(!isNaN(v)){
        #     console.log("Add3", v, sel.name, arg0.name, arg1.name, arg2.name);
        #     match = new Match("Add3","Add3", sel.name, "UpdateTextField",
        #                  v,[arg0.name,arg1.name,arg2.name]);
        #     assert(match);
        # }
        match = Match(skill="Add3",rhs="Add3", sel=sel.name, action="UpdateTextField",
                     input=v,
                     args=List([arg0.name,arg1.name]));
                     # arg0.name + "," + arg1.name)
        mem.declare(match);
print("C")        
        
class ResolveAdd3(Rule):
    def when():
        return (
            Var(Match,'match') & (match.rhs == "Add3") & (match.full_fired == False) & 
            Var(TextField,'sel') & (match.sel == sel.name) & 
            NOT(TextField, 'sel_r') & (sel.to_right == sel_r) & (sel_r.enabled == True)
        )

    def then(mem, match, sel):
        mem.modify(match,'full_fired', 1)

class Carry2_1(Rule):
    def when():
        return (
            Var(PhaseHandler, "ph") & (ph.phase == 1) & 
            Var(TextField, "sel")  & (sel.enabled == True) & 
            Var(TextField, "arg0") & (sel.enabled == False) & (arg0.value != "") & 
            Var(TextField, "arg1") & (sel.enabled == False) & (arg1.value != "") &
            Var(TextField, "neigh0") & (neigh0.value != "") & 
            (sel.to_right == neigh0) & 
            (neigh0.below == arg0) & 
            (arg0.above == neigh0) & 
            (arg0.below == arg1) & 
            (arg1.above == arg0) #& 
            # NOT()
        )
    def then(mem,ph,sel, arg0, arg1, neigh0):
        v = "?"
        if(True):
            match = Match(skill="Carry2",rhs="Carry2", sel=sel.name, action="UpdateTextField",
                     input=v,
                     args=List([arg0.name,arg1.name]));
            mem.declare(match);


class ResolveCarry2(Rule):
    def when():
        return (
            Var(Match,'match') & (match.rhs == "Carry2") & (match.full_fired == False) & 
            Var(TextField,'sel') & (match.sel == sel.name) & 
            (match.input != "0") & 
            # ()
            # TODO: Maybe add exists
        )
    def then(mem, match, sel):
        mem.modify("full_fired", True)
        print("RESOLVE Carry2", match.sel)


class Carry3(Rule):
    def when():
        return (
            Var(PhaseHandler, "ph") & (ph.phase == 1) & 
            Var(TextField, "sel")  & (sel.enabled == True) & 
            Var(TextField, "arg0") & (sel.enabled == False) & (arg0.value != "") & 
            Var(TextField, "arg1") & (sel.enabled == False) & (arg1.value != "") &
            Var(TextField, "arg2") & (sel.enabled == False) & #(arg2.value != "") &
            Var(TextField, "neigh0") & (neigh0.value != "") & 
            (sel.to_right == arg0) & 
            (arg0.below == arg1) & 
            (arg0.above == arg0) & 
            (arg1.below == arg2) & 
            (arg2.above == arg1)# & 
            # (arg1.above == arg0) #& 
            # NOT()
        )
    def then(mem,ph,sel, arg0, arg1, neigh0):
        v = "?"
        if(True):
            match = Match(skill="Carry2",rhs="Carry2", sel=sel.name, action="UpdateTextField",
                     input=v,
                     args=List([arg0.name,arg1.name]));
            mem.declare(match);
# print("D")

#######
## C3 C2 C1 C0
## A3 A2 A1 A0
## B3 B2 B1 B0
## O3 O2 O1 O0



def bootstrap():
    mem = Memory()
    mem.declare(PhaseHandler())
    mem.declare(ConflictSet())

    row_names = ["C","A","B","O"]
    enabled_rows = [True,False,False,True]
    MAX_LEN = 4
    rows = []
    for i in range(MAX_LEN):
        row_i = []
        for j in range(len(row_names)):
            row_i.append(TextField(**{
                "name" : row_names[j] + str((MAX_LEN-1) - i),
                # "type" : "TextField",
                "value" : "" if enabled_rows[j] else "7",
                "enabled" : enabled_rows[j]
            }))
        rows.append(row_i)
    for i in range(MAX_LEN):
        for j in range(len(row_names)):
            if(i-1 >= 0): 
                rows[i][j].to_left = rows[i-1][j]
                rows[i-1][j].to_right = rows[i][j]
            if(j-1 >= 0): 
                rows[i][j].above = rows[i][j-1]
                rows[i][j-1].below = rows[i][j]


    rows[0][2].value = "1"
    rows[0][2].enabled = False

    for i in range(MAX_LEN):
        for j in range(len(row_names)):
            e = rows[i][j]
            print(rows[i][j].name, 
                "to_left", e.to_left.name if e.to_left else "", ',',
                "to_right", e.to_right.name if e.to_right else "", ',',
                "above", e.above.name if e.above else "", ',',
                "below", e.below.name if e.below else "",
                )
            mem.declare(rows[i][j])



    return mem

mem = bootstrap()
r_eng = RuleEngine(mem,[Add2, ResolveAdd2, Add3, ResolveAdd3,
    startMatching, startResolving, startChecking, reportCorrectness, resetting])
r_eng.start()






# Mod10(Add(X1,X2))

# Y1 = Add(X1,X2)
# output = Mod10(Y1)


# Y1 = Number(X1) + Number(X2)
# output = Y1 % 10






# @custom_condition((TextField,))
# def is_prime(a):
#     ...


# is_prime(x) && (x > 1)
