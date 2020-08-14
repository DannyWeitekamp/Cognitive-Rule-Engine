from numbert.rule import BaseRule, RuleEngine, declare,modify,retract
from numbert.numbalizer import Numbalizer


class Add2Rule(BaseRule):
    patterns = {
        "sel": {
            "type" : "TextField",
            "pos_pattern" : {
                "above": "arg1",
            }
        },
        "arg0": {
            "type" : "TextField",
            "pos_pattern" : {
                "below": "arg1",
            }
        },
        "arg1": {
            "type" : "TextField",
            "pos_pattern" : {
                "above": "arg0",
                "below": "sel",
            }
        }
    }
    def conditions(sel,arg0,arg1):
        return (sel.value == "") & ((arg0.value == "1") | (arg0.value == "3"))

    def forward(re,sel,arg0,arg1):
        # Add3()
        declare(re,"Match2",
            {"composition": "Add2(?,?)",
            "sel" : sel.name,
            "arg0" : arg0.name,
            "arg1" : arg1.name,
            "input_value": float(arg0.obj.value) + float(arg1.obj.value),
            "satisfied" : False
            })
        modify(re,sel,{'value': 4})
        # return "halt"

        # declare(delta, "TextField",  TextField())
        # modify(delta,"TextField",)
        # halt(delta)
        # backtrack(delta)

        # return delta
        # kb << halt()

class Add2Resolve(BaseRule):
    patterns = {
        "partial_match": {
            "type" : "Match2",
            "pos_pattern" : {
                "sel": "sel",
                "arg0": "arg0",
                "arg1": "arg1",
                "satisfied" : False
            }
        },
        "sel": {
            "type" : "TextField",
        },
        "arg0": {
            "type" : "TextField",
        },
        "arg1": {
            "type" : "TextField",
        }
    }
    def conditions(partial_match,sel,arg0,arg1):
        return sel.right.value != ""
    def forward(re,partial_match,sel,arg0,arg1):
        print("partial_match")
        print(partial_match.obj)
        print()
        modify(re,partial_match,{'satisfied' : True})


numbalizer = Numbalizer()
numbalizer.register_specification("TextField",{
    "value" : "string",
    "above" : {"type" : "string", "flags" : ["reference"]},
    "below" : {"type" : "string", "flags" : ["reference"]},
    "to_left" : {"type" : "string", "flags" : ["reference"]},
    "to_right" : {"type" : "string", "flags" : ["reference"]},
    })
numbalizer.register_specification("Match2",{
    "composition" : "string",
    "sel" : {"type" : "string", "flags" : ["reference"]},
    "arg0" : {"type" : "string", "flags" : ["reference"]},
    "arg1" : {"type" : "string", "flags" : ["reference"]},
    "input_value" : "string",
    "satisfied" : "number"
    })


state1 = {
        "A1": {
            "type" : "TextField",
            "value": 1,
            "above": None,
            "below": "B1",
            "to_left" : "A2",
            "to_right": None,
        },
        "A2": {
            "type" : "TextField",
            "value": 2,
            "above": None,
            "below": "B2",
            "to_left" : "A3",
            "to_right": "A1",
        },
        "A3": {
            "type" : "TextField",
            "value": 3,
            "above": None,
            "below": "B3",
            "to_left" : "A4",
            "to_right": "A2",
        },
        "A4": {
            "type" : "TextField",
            "value": 3,
            "above": None,
            "below": "C4",
            "to_left" : None,
            "to_right": "A3",
        },
        "B1": {
            "type" : "TextField",
            "value": 4,
            "above": "A1",
            "below": "C1",
            "to_left" : "B2",
            "to_right": None,
        },
        "B2": {
            "type" : "TextField",
            "value": 5,
            "above": "A2",
            "below": "C2",
            "to_left" : "B3",
            "to_right": "B1",
        },
        "B3": {
            "type" : "TextField",
            "value": 6,
            "above": "A3",
            "below": "C3",
            "to_left" : None,
            "to_right": "B2",
        },
        "C1": {
            "type" : "TextField",
            "value": "",
            "above": "B1",
            "below": None,
            "to_left" : "C2",
            "to_right": None,
        },
        "C2": {
            "type" : "TextField",
            "value": "",
            "above": "B2",
            "below": None,
            "to_left" : "C3",
            "to_right": "C1",
        },
        "C3": {
            "type" : "TextField",
            "value": "",
            "above": "B3",
            "below": None,
            "to_left" : "C4",
            "to_right": "C2",
        },
        "C4": {
            "type" : "TextField",
            "value": "",
            "above": "A4",
            "below": None,
            "to_left" : None,
            "to_right": "C3",
        }
    }

nb_state1 = numbalizer.state_to_nb_objects(state1)

eng = RuleEngine(numbalizer,[Add2Rule,Add2Resolve])
eng.set_state(nb_state1)
eng.forward()
eng.forward()

print(eng.state['Match2'])


save_rule(Add2Rule,".")
rule = load_rule("./Add3Rule.pkl")
print(rule)