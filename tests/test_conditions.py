import numpy as np
from numbert.conditions import Conditions, conds_apply, AlphaBindable
from numbert.numbalizer import Numbalizer
import json


#STATE 1:
# A4 A3 A2 A1
#    B3 B2 B1
# C4 C3 C2 C1
state1 = {
    "A1": {
        "type" : "TextField",
        "value": 1,
        "above": "",
        "below": "B1",
        "to_left" : "A2",
        "to_right": "",
    },
    "A2": {
        "type" : "TextField",
        "value": 2,
        "above": "",
        "below": "B2",
        "to_left" : "A3",
        "to_right": "A1",
    },
    "A3": {
        "type" : "TextField",
        "value": 3,
        "above": "",
        "below": "B3",
        "to_left" : "A4",
        "to_right": "A2",
    },
    "A4": {
        "type" : "TextField",
        "value": 3,
        "above": "",
        "below": "C4",
        "to_left" : "",
        "to_right": "A3",
    },
    "B1": {
        "type" : "TextField",
        "value": 4,
        "above": "A1",
        "below": "C1",
        "to_left" : "B2",
        "to_right": "",
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
        "to_left" : "",
        "to_right": "B2",
    },
    "C1": {
        "type" : "TextField",
        "value": 7,
        "above": "B1",
        "below": "",
        "to_left" : "C2",
        "to_right": "",
    },
    "C2": {
        "type" : "TextField",
        "value": 8,
        "above": "B2",
        "below": "",
        "to_left" : "C3",
        "to_right": "C1",
    },
    "C3": {
        "type" : "TextField",
        "value": 9,
        "above": "B3",
        "below": "",
        "to_left" : "C4",
        "to_right": "C2",
    },
    "C4": {
        "type" : "TextField",
        "value": 9,
        "above": "A4",
        "below": "",
        "to_left" : "",
        "to_right": "C3",
    }
}

numbalizer = Numbalizer()
numbalizer.register_specification("TextField",{
    "value" : "string",
    "above" : {"type" : "string", "flags" : ["reference"]},
    "below" : {"type" : "string", "flags" : ["reference"]},
    "to_left" : {"type" : "string", "flags" : ["reference"]},
    "to_right" : {"type" : "string", "flags" : ["reference"]},
    })
state1_enumerized = numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects(state1))

def test_literal():
    sel = AlphaBindable("sel", "TextField",numbalizer)
    arg0 = AlphaBindable("arg0", "TextField",numbalizer)
    arg1 = AlphaBindable("arg1", "TextField",numbalizer)

    sym_conds = (sel.value == "7") & (arg0.value == "1") & (arg1.value == "4")
    s_conditions = Conditions(sym_conds,["sel",'arg0','arg1'],numbalizer=numbalizer)

    config = {
        "bindables" : {
            "sel" : [
                ["value"]
                ],
            "arg0" : [
                ["value"]
                ],
            "arg1" : [
                ["value"]
                ]
        },
        "relations" : [
            ["EQUAL", True , 0 ,"7"],
            ["EQUAL", True , 1 ,"1"],
            ["EQUAL", True , 2 ,"4"],
        ],
        "clause" : [
            ["AND", 0, 1, 2]
        ]
    }
    assert json.dumps(s_conditions.get_config()) == json.dumps(config)

    conditions = Conditions(numbalizer=numbalizer,config=config)
    match = numbalizer.enumerize(np.array(['C1', 'A1', 'B1'],dtype="U"))
    applies = conds_apply(match, conditions.conds_data, state1_enumerized,
         numbalizer.string_enums, numbalizer.attr_inds_by_type)
    # print(applies)
    assert applies == 1

def test_pair():
    arg0 = AlphaBindable("arg0", "TextField",numbalizer)
    arg1 = AlphaBindable("arg1", "TextField",numbalizer)

    sym_conds = arg0.value == arg1.value
    s_conditions = Conditions(sym_conds,['arg0','arg1'],numbalizer=numbalizer)

    config = {
        "bindables" : {
            "arg0" : [
                ["value"]
                ],
            "arg1" : [
                ["value"]
                ]
        },
        "relations" : [
            ["EQUAL", False , 0 ,1],
        ],
        "clause" : []
    }
    assert json.dumps(s_conditions.get_config()) == json.dumps(config)

    conditions = Conditions(numbalizer=numbalizer,config=config)
    match = numbalizer.enumerize(np.array(['C3', 'C4'],dtype="U"))
    applies = conds_apply(match, conditions.conds_data, state1_enumerized,
         numbalizer.string_enums, numbalizer.attr_inds_by_type)
    # print(applies)
    assert applies == 1

def test_dereference():
    sel = AlphaBindable("sel", "TextField",numbalizer)
    arg0 = AlphaBindable("arg0", "TextField",numbalizer)

    sym_conds = (sel.to_right.value == arg0.to_left.value)
    s_conditions = Conditions(sym_conds,["sel","arg0"],numbalizer=numbalizer)

    config = {
        "bindables" : {
            "sel" : [
                ['to_right', "value"],
            ],
            "arg0" : [
                ['to_left', "value"]
            ]
        },
        "relations" : [
            ["EQUAL", False , 0 ,1],
        ],
        "clause" : []
    }

    print(json.dumps(s_conditions.get_config()))
    print(json.dumps(config))
    assert json.dumps(s_conditions.get_config()) == json.dumps(config)


    conditions = Conditions(numbalizer=numbalizer,config=config)
    match = numbalizer.enumerize(np.array(['C4','C2'],dtype="U"))
    applies = conds_apply(match, conditions.conds_data, state1_enumerized,
         numbalizer.string_enums, numbalizer.attr_inds_by_type)
    print(applies)
    assert applies == 1


#Tests to write: Truthy Relation, NotEq, lessthan greater than (equal)

# test_literal()
# test_pair()
# test_dereference()