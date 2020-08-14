import os, sys
from numbert.conditions import Conditions, AlphaBindable
from numbert.matcher import Matcher
from numbert.numbalizer import Numbalizer
from numba import njit
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple 

from types import FunctionType
import dill



class Binding(object):
    def __init__(self,typ,name,obj):
        self.type = typ
        self.name = name
        self.obj  = obj

 
class RuleEngine(object):
    def __init__(self,numbalizer,rules):
        self.numbalizer = numbalizer
        self.rules = rules
        for rule in self.rules:
            rule._init(numbalizer)
        self.filler_count = 0

    def set_state(self,state):
        self.state = state

    def forward(self):
        # matches = []
        # print("BEFORE")
        # print(self.state)
        # print("FORWARD")
        enumerized_state = self.numbalizer.nb_objects_to_enumerized(self.state)
        any_matches = True
        while(any_matches):

            any_matches = False
            for rule in self.rules:

                matches = rule.matcher.get_matches(enumerized_state)
                print(matches)
                if(len(matches) > 0): any_matches = True
                for match in matches:
                    bindings = []
                    for b_name, typ, name in zip(rule.matcher.names,rule.matcher.types, match):
                        # print(b_name, typ, name)
                        bindings.append(Binding(typ,name,self.state[typ][name]))
                    delta = {}
                    # print(bindings)
                    out = rule.forward(self,*bindings)
                    enumerized_state = self.numbalizer.nb_objects_to_enumerized(self.state)
                    if(out == "halt"):
                        break
        print(self.state)

    def next_filler_name(self):
        self.filler_count += 1
        return "fact%s" % self.filler_count



            # print(matches)
            # matches.append(rule, matches)
        # print("AFTER")
        # print(self.state)

# def __init__(self,numbalizer,names=None,types=None,pos_patterns=None,
#                     neg_patterns=None,conditions=None,config=None):
@njit(cache=True)
def update_state(curr, update):
    for name, obj in update.items():
        curr[name] = obj

def modify(re,bound,change):
    print("modify",bound.name,change)
    numbalizer = re.numbalizer
    bound_as_dict = bound.obj._asdict()
    # print(bound_as_dict)
    n_keys=  len(bound_as_dict)
    bound_as_dict.update(change)
    assert n_keys == len(bound_as_dict), "Modification %r invalid for type %r" % (change, bound.type)
    bound_as_dict['type'] = bound.type

    changes = numbalizer.state_to_nb_objects({bound.name: bound_as_dict})

    # print(changes)
    update_state(re.state[bound.type],changes[bound.type])
    print(re.state)

def declare(re, typ, fact, name=None):
    print("declare",fact)
    if(name == None): name = re.next_filler_name()
    if("type" not in fact): fact['type'] = typ
    re.numbalizer.enumerize_value(name)
    changes = re.numbalizer.state_to_nb_objects({name: fact})
    print(changes)
    print()

    if(not typ in re.state):
        nb_jitstruct = re.numbalizer.jitstructs[typ].numba_type
        re.state[typ] = Dict.empty(unicode_type,nb_jitstruct)
    update_state(re.state[typ],changes[typ])

def retract(re, bound):
    del re.state[bound.type][bound.name]


def load_rule(path):
    with open(path,'rb') as f:
        config = dill.load(f)
    name = config['__name__']
    del config['__name__']

    rule = type(name, (BaseRule,),config)
    return rule

def save_rule(rule,path):
    rule.save(path)


class BaseRule():
    initialized = False

    @classmethod
    def _init(cls,numbalizer):
        if(not cls.initialized):
            cls.numbalizer = numbalizer
            cls._init_matcher(numbalizer)
            cls.initialized = True

    @classmethod
    def _init_matcher(cls,numbalizer):
        bindables = []
        bindable_names = []

        #Parse cls.conditions into Conditions() type
        if(hasattr(cls,"conditions")):
            if(isinstance(cls.conditions,FunctionType)):
                assert hasattr(cls,"patterns"), "'patterns' attr required for symbolically defined conditions."
                for name, obj in cls.patterns.items():
                    bindable_names.append(name)
                    bindables.append(AlphaBindable(name,obj['type'],numbalizer))
                symbolic_conds = cls.conditions(*bindables)
                cls.conditions = Conditions(symbolic_conds,bindable_names,numbalizer)
            elif(isinstance(cls.conditions,dict)):
                cls.conditions = Conditions(numbalizer=numbalizer,config=cls.conditions)
            elif(not isinstance(cls.conditions,Conditions)):
                raise "Conditions should be function, config or, Condition() object, but got %s" % (type(cls.conditions))
        

        if(hasattr(cls,"matcher")):
            if(isinstance(cls.matcher,dict)):
                cls.matcher = Matcher(numbalizer=numbalizer, config=cls.matcher)
            elif(not isinstance(cls.matcher,Matcher)):
                raise "Matcher should be config or, Matcher() object, but got %s" % (type(cls.matcher))
        elif(hasattr(cls,"patterns")):
            cls.matcher = Matcher(numbalizer)
            # cls.matcher.set_patterns(cls.patterns)
            config = {"patterns" : cls.patterns}
            if(hasattr(cls,"conditions")):
                config["conditions"] = cls.conditions
            print("config")
            print(config)
            cls.matcher.set_config(config)
            print(cls.matcher)

        else:
            raise ValueError("Class attributes insufficient to generate Matcher.")

    def __init_subclass__(cls, **kwargs):
        pass

    @classmethod
    def save(cls, path):
        config = {"matcher" : cls.matcher.get_config(), "__name__" : cls.__name__, "forward" : cls.forward}
        path = os.path.join(path,"%s.pkl"%cls.__name__)
        with open(path,'wb') as f:
            dill.dump(config,f)


        





# Add3Rule.save_rule(".")
# eng._init(numbalizer)