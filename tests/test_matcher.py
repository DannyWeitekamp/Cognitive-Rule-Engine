import pytest
from numbert.numbalizer import Numbalizer
from numbert.matcher import Matcher
import numpy as np

# @pytest.fixture
# def state1():
    
#     return




def matches_equal(a,b):
    a = np.sort(np.array(a,dtype="U"),axis=0)
    b = np.sort(np.array(b,dtype="U"),axis=0)
    return (a==b).all()


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


def test_direct_pattern_assign():
    matcher = Matcher(numbalizer,["?sel"],["TextField"],[np.zeros(5,dtype=np.uint32)])
    matches = matcher.get_matches(state1_enumerized)
    assert sorted([match[0] for match in matches]) == sorted([name for name in state1.keys()])

def test_match3():
    config = {
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
    matcher = Matcher(numbalizer,config=config)
    matches = matcher.get_matches(state1_enumerized)
    print(matches)
    assert matches_equal(matches,
        [['C1', 'A1', 'B1'],
         ['C2', 'A2', 'B2'],
         ['C3', 'A3', 'B3']]
    )


def test_unbound3():
    config = {
        "sel": {
            "type" : "TextField",
            "pos_pattern" : {
                "below": "",
            }
        },
        "arg0": {
            "type" : "TextField",
            "pos_pattern" : {
                "below": "arg1",
                "above": "",
            }
        },
        "arg1": {
            "type" : "TextField",
            "pos_pattern" : {
                "above": "arg0",
            }
        }
    }
    print(state1_enumerized)
    matcher = Matcher(numbalizer,config=config)
    matches = matcher.get_matches(state1_enumerized)
    print(matches)
    print(matcher.pos_patterns)
    print(matcher.neg_patterns)

    ground_truth = [
         ['C1', 'A1', 'B1'],
         ['C2', 'A1', 'B1'],
         ['C3', 'A1', 'B1'],
         ['C4', 'A1', 'B1'],
         ['C1', 'A2', 'B2'],
         ['C2', 'A2', 'B2'],
         ['C3', 'A2', 'B2'],
         ['C4', 'A2', 'B2'],
         ['C1', 'A3', 'B3'],
         ['C2', 'A3', 'B3'],
         ['C3', 'A3', 'B3'],
         ['C4', 'A3', 'B3'],
         ['C1', 'A4', 'C4'],
         ['C2', 'A4', 'C4'],
         ['C3', 'A4', 'C4'],
         ]

    assert len(matches) == len(ground_truth), "Incorrect number of matches"
    assert matches_equal(matches, ground_truth)

    

