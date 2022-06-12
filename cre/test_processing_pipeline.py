import numpy as np
import numba
from numba import f8, i8, njit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, boolean
from cre.memset import MemSet
from cre.var import Var
from cre.utils import PrintElapse, deref_info_type
from cre.fact import define_fact
from cre.default_ops import Equals
from cre.context import cre_context
from pprint import pprint
from cre.flattener import Flattener
from cre.feature_applier import FeatureApplier
from cre.relative_encoder import _check_needs_rebuild, RelativeEncoder, get_relational_fact_attrs, next_adjacent
from cre.vectorizer import Vectorizer
from cre.default_ops import Equals

eq_f8 = Equals(f8, f8)
eq_str = Equals(unicode_type, unicode_type)

pprint._sorted = lambda x:x
# pprint = lambda x : pprint(x, sort_dicts=False)

Component = define_fact("Component", {
    "id" : str,
    # "x" : {"type" : float, "visible" : False},
    # "y" : {"type" : float, "visible" : False},
    # "width" : {"type" : float, "visible" : False},
    # "height" : {"type" : float, "visible" : False},
    "above" : "Component", "below" : "Component",
    "to_left": "Component", "to_right" : "Component",
    "parents" : "List(Component)"
})

TextField = define_fact("TextField", {
    "inherit_from" : "Component",
    "value" : {"type" : str, "visible" : True},
    "locked" : {"type" : bool, "visible" : True},
})

Button = define_fact("Button", {
    "inherit_from" : "Component",
    # "locked" : {"type" : bool, "visible" : True},
})

Container = define_fact("Container", {
    "inherit_from" : "Component",
    "children" : "List(Component)"
})

def encode_neighbors(objs, l_str='to_left', r_str="to_right", a_str="above", b_str="below", strip_attrs=["x", "y", "width", "height"]):
  # objs = list(_objs.values()) if(isinstance(_objs,dict)) else _objs
  objs_list = list(objs.values())

  rel_objs = []
  for i, obj in enumerate(objs):
    rel_objs.append({
      l_str : [],
      r_str : [], 
      a_str : [],
      b_str : [],
    })

  for i, a_obj in enumerate(objs_list):
    for j, b_obj in enumerate(objs_list):
      if(i != j):
        if(a_obj['y'] > b_obj['y'] and
           a_obj['x'] < b_obj['x'] + b_obj['width'] and
           a_obj['x'] + a_obj['width'] > b_obj['x']):
            dist = a_obj['y'] - b_obj['y'];
            rel_objs[i][a_str].append((dist, j));
            rel_objs[j][b_str].append((dist, i));

        if(a_obj['x'] < b_obj['x'] and
           a_obj['y'] + a_obj['height'] > b_obj['y'] and
           a_obj['y'] < b_obj['y'] + b_obj['height']):
            dist = b_obj['x'] - a_obj['x']
            rel_objs[i][r_str].append((dist, j));
            rel_objs[j][l_str].append((dist, i));

  strip_attrs_set = set(strip_attrs)
  out = {}   
  for (_id, obj), rel_obj in zip(objs.items(), rel_objs):
    # print(_id, obj["x"],obj["y"],obj["width"],obj["height"])
    new_obj = {k:v for k,v in obj.items() if k not in strip_attrs}
    new_obj[l_str] = objs_list[sorted(rel_obj[l_str])[0][1]]["id"] if len(rel_obj[l_str]) > 0 else ""
    new_obj[r_str] = objs_list[sorted(rel_obj[r_str])[0][1]]["id"] if len(rel_obj[r_str]) > 0 else ""
    new_obj[a_str] = objs_list[sorted(rel_obj[a_str])[0][1]]["id"] if len(rel_obj[a_str]) > 0 else ""
    new_obj[b_str] = objs_list[sorted(rel_obj[b_str])[0][1]]["id"] if len(rel_obj[b_str]) > 0 else ""
    out[_id] = new_obj

  # if(any([obj.get('value',"") != "" and obj.get('value',"")]))  
  # print()

  return out

# def from_dict(d):



def new_mc_addition_state(upper, lower):
    upper, lower = str(upper), str(lower)
    n = max(len(upper),len(lower))

    tf_config = {"type": "TextField", "width" : 100, "height" : 100, "value" : ""}
    comp_config = {"type": "Component", "width" : 100, "height" : 100}

    d_state = {
        "operator" : {"id" : "operator", "x" :-110,"y" : 220 , **comp_config},
        # "line" :     {"id" : "line", "x" :0,   "y" : 325 , **comp_config, "height" : 5},
        "done" :     {"id" : "done", "x" :0, "y" : 440 , **comp_config, "type": "Button"},
        "hidey1" :   {"id" : "hidey1", "x" :n * 110, "y" : 0 , **comp_config},
        "hidey2" :   {"id" : "hidey2", "x" :0,   "y" : 110 , **comp_config},
        "hidey3" :   {"id" : "hidey3", "x" :0,   "y" : 220 , **comp_config},
    }

    for i in range(n):
        offset = (n - i) * 110
        d_state.update({
            f"{i}_carry": {"id" : f"{i}_carry", "x" :offset,   "y" : 0 , **tf_config},
            f"{i}_upper": {"id" : f"{i}_upper", "x" :offset,   "y" : 110 , **tf_config},
            f"{i}_lower": {"id" : f"{i}_lower", "x" :offset,   "y" : 220 , **tf_config},
            f"{i}_answer": {"id" : f"{i}_answer", "x" :offset,   "y" : 330 , **tf_config},
        })

    del d_state["0_carry"]

    d_state.update({
        f"{n}_carry": {"id" : f"{n}_carry", "x" :0,   "y" : 0 , **tf_config},
        f"{n}_answer": {"id" : f"{n}_answer", "x" :0,   "y" : 330 , **tf_config},
    })

    for i,c in enumerate(reversed(upper)):
        d_state[f'{i}_upper']['value'] = c

    for i,c in enumerate(reversed(lower)):
        d_state[f'{i}_lower']['value'] = c


    d_state = encode_neighbors(d_state)

    # pprint(d_state)
    return d_state

class DictMemSetConverter():
    def __init__(self, in_memset=None, context=None):
        self.context = cre_context(context)
        self.non_relational_attrs = {}
        self.relational_attrs = {}
        self.in_memset = in_memset

    def get_non_relational(self, type_name):
        if(type_name not in self.non_relational_attrs):
            fact_type = self.context.get_type(name=type_name)
            attrs = list(fact_type.filter_spec("~relational").keys())
            self.non_relational_attrs[type_name] = attrs
        return self.non_relational_attrs[type_name]

    def get_relational(self, type_name):
        if(type_name not in self.relational_attrs):
            fact_type = self.context.get_type(name=type_name)
            attrs = list(fact_type.filter_spec("relational").keys())
            self.relational_attrs[type_name] = attrs
        return self.relational_attrs[type_name]


    def apply(self, state_dict, return_map=False):
        '''Converts a dictionary of dictionaries each representing a fact into a MemSet'''
        # Make each fact instance, but skip setting any relational members.
        fact_instances = {}
        for _id, config in state_dict.items():
            fact_type = self.context.get_type(name=config['type'])
            non_rel_attrs = self.get_non_relational(config['type'])
            kwargs = {attr:config[attr] for attr in non_rel_attrs if attr in config} 
            fact = fact_type(**kwargs)
            fact_instances[_id] = fact

        # Now that all facts exist set any relational attributes
        for _id, fact in fact_instances.items():
            config = state_dict[_id]
            rel_attrs = self.get_relational(config['type'])

            for attr in rel_attrs:
                if(attr not in config): continue
                val_name = config[attr]
                if(not val_name): continue # i.e. skip if None or empty str
                if(val_name not in fact_instances):
                    raise ValueError(f"Reference to unspecified fact {val_name}.")
                setattr(fact, attr, fact_instances[val_name])

        # Declare each fact to a new MemSet
        ms = self.in_memset if(self.in_memset) else MemSet()
        for _id, fact in fact_instances.items():
            ms.declare(fact)

        if(return_map):
            return ms, fact_instances
        else:
            return ms

fact_types = [Container, TextField,Component, Button]
feat_types = [eq_f8, eq_str]
val_types = [f8,unicode_type,boolean]

def setup_pipeline():
    wm = MemSet()
    conv = DictMemSetConverter(wm)
    fl = Flattener(fact_types, wm, id_attr="id")
    fa = FeatureApplier(feat_types)
    re = RelativeEncoder(fact_types, wm, id_attr="id")
    vr = Vectorizer(val_types)
    pipeline = (conv,fl,fa,re,vr)
    match_names = ['1_carry', '0_upper', '0_lower']
    vars = [Var(TextField, 'sel'), Var(TextField, 'arg0'), Var(TextField, 'arg1')]
    return (wm, pipeline, match_names, vars), {}
        

def setup_pipeline_first_run():
    dict_state = new_mc_addition_state(777,777)
    (wm, pipeline, match_names, vars), _ = setup_pipeline()
    return (dict_state, wm, pipeline, match_names, vars), {}

def pipeline_first_run(dict_state, wm, pipeline, match_names, vars):
    conv, fl, fa, re, vr = pipeline
    wm, fact_map = conv.apply(dict_state, return_map=True) # 1.23 ms
    matches = [fact_map[x] for x in match_names]
    flat_ms = fl.apply(wm) # 0.07ms
    feat_ms = fa.apply(flat_ms) # 0.14ms
    rel_ms = re.encode_relative_to(feat_ms, matches, vars) # 0.30ms
    vec = vr.apply(rel_ms) # 0.12ms
    return vec, matches, fact_map

def setup_pipeline_second_run():
    args,_ = setup_pipeline_first_run()
    vec, matches, fact_map = pipeline_first_run(*args)
    (dict_state, wm, pipeline, match_names, vars) = args
    wm.modify(matches[0], "value", "4")
    return (wm, pipeline, matches, vars), {}

def pipeline_second_run(wm, pipeline, matches, vars):
    conv, fl, fa, re, vr = pipeline
    flat_ms = fl.apply(wm) # 0.01ms
    feat_ms = fa.apply(flat_ms) # 0.02ms
    rel_ms = re.encode_relative_to(feat_ms, matches, vars) # 0.78ms
    vecs = vr.apply(rel_ms) # 0.14ms
    return vecs



def test_pipeline():
    args,_ = setup_pipeline_first_run()
    (nom, flt), matches, fact_map = pipeline_first_run(*args)
    (dict_state, wm, pipeline, match_names, vars) = args
    conv, fl, fa, re, vr = pipeline

    wm.modify(fact_map["0_answer"], "value", "4")
    (nom, new_flt) = pipeline_second_run(wm,pipeline, matches, vars)
    assert len(np.unique(new_flt)) > len(np.unique(flt))
    flt = new_flt

    wm.modify(fact_map["1_carry"], "value", "1")
    (nom, new_flt) = pipeline_second_run(wm,pipeline, matches, vars)
    assert len(np.unique(new_flt)) > len(np.unique(flt))
    flt = new_flt


def test_b_pipeline_1st_run(benchmark):
    benchmark.pedantic(pipeline_first_run,setup=setup_pipeline_first_run, warmup_rounds=1, rounds=10)

def test_b_pipeline_2nd_run(benchmark):
    benchmark.pedantic(pipeline_second_run,setup=setup_pipeline_second_run, warmup_rounds=1, rounds=10)


if __name__ == "__main__":
    import faulthandler; faulthandler.enable()

    test_pipeline()
    # for i in range(2):
    #     args,_ = setup_pipeline_first_run()
    #     with PrintElapse("first_run"):
    #         pipeline_first_run(*args)

    # for i in range(2):
    #     args,_ = setup_pipeline_second_run()
    #     with PrintElapse("second_run"):
    #         pipeline_second_run(*args)



    


# ms, facts = conv.apply(ds,return_map=True)

# ms.modify(facts["0_answer"], "value", "4")
# ms.modify(facts["1_carry"], "value", "1")
# ms.modify(facts["1_answer"], "value", "5")
# ms.modify(facts["2_carry"], "value", "1")
# ms.modify(facts["2_answer"], "value", "5")
# ms.modify(facts["3_carry"], "value", "1")
# ms.modify(facts["3_answer"], "value", "1")

# print(ms)

'''TODO
1) [X] Make bool work 
2) [] Make printout better 
'''
#TODO: 

# with PrintElapse("Elapse"):
#     ms = conv.dict_to_memset(ds)

# print("----------------")
# with PrintElapse("Elapse"):
#     ms = conv.dict_to_memset(ds)

# with PrintElapse("Elapse"):
#     ms = conv.dict_to_memset(ds)

# with PrintElapse("Elapse"):
#     ms = conv.dict_to_memset(ds)
# print(ms)


