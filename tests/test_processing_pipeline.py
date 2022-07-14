import numpy as np
import numba
from numba import f8, i8, njit
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, boolean
from cre.memset import MemSet
from cre.var import Var
from cre.utils import PrintElapse, deref_info_type, decode_idrec
from cre.fact import define_fact, FactProxy
from cre.default_ops import Equals
from cre.context import cre_context
from pprint import pprint
from cre.transform.flattener import Flattener
from cre.transform.feature_applier import FeatureApplier
from cre.transform.relative_encoder import _check_needs_rebuild, RelativeEncoder, get_relational_fact_attrs, next_adjacent
from cre.transform.vectorizer import Vectorizer
from cre.transform.memset_builder import MemSetBuilder
from cre.default_ops import Equals
from cre.conditions import Conditions

eq_f8 = Equals(f8, f8)
eq_str = Equals(unicode_type, unicode_type)

pprint._sorted = lambda x:x
# pprint = lambda x : pprint(x, sort_dicts=False)

# with cre_context("test_processing_pipeline"):
    

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



def new_mc_addition_state(upper, lower, ):
    upper, lower = str(upper), str(lower)
    n = max(len(upper),len(lower))

    tf_config = {"type": "TextField", "width" : 100, "height" : 100, "value" : ""}
    comp_config = {"type": "Component", "width" : 100, "height" : 100}
    hidden_config = {**tf_config, 'locked' : True}

    d_state = {
        "operator" : {"id" : "operator", "x" :-110,"y" : 220 , **comp_config},
        # "line" :     {"id" : "line", "x" :0,   "y" : 325 , **comp_config, "height" : 5},
        "done" :     {"id" : "done", "x" :0, "y" : 440 , **comp_config, "type": "Button"},
        "hidey1" :   {"id" : "hidey1", "x" :n * 110, "y" : 0 , **hidden_config},
        "hidey2" :   {"id" : "hidey2", "x" :0,   "y" : 110 , **hidden_config},
        "hidey3" :   {"id" : "hidey3", "x" :0,   "y" : 220 , **hidden_config},
    }

    for i in range(n):
        offset = (n - i) * 110
        d_state.update({
            f"{i}_carry": {"id" : f"{i}_carry", "x" :offset,   "y" : 0 , **tf_config},
            f"{i}_upper": {"id" : f"{i}_upper", "x" :offset,   "y" : 110 , "locked" : True, **tf_config},
            f"{i}_lower": {"id" : f"{i}_lower", "x" :offset,   "y" : 220 , "locked" : True, **tf_config},
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


def setup_fact_types():
    context = cre_context()
    if("Component" in context.name_to_type):
        print("Retrieved!!!")
        return (context.get_type(name="Component"),
                context.get_type(name="TextField"),
                context.get_type(name="Button"),
                context.get_type(name="Container"),
                )
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
    return (Container, TextField, Component, Button)



def setup_pipeline():
    Container, TextField, Component, Button = setup_fact_types()
    print(TextField.spec)

    fact_types = [Container, TextField,Component, Button]
    feat_types = [eq_f8, eq_str]
    val_types = [f8,unicode_type,boolean]

    wm = MemSet()
    conv = MemSetBuilder(wm)
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
    wm, fact_map = conv(dict_state, return_map=True) # 1.23 ms
    matches = [fact_map[x] for x in match_names]
    flat_ms = fl(wm) # 0.07ms
    feat_ms = fa(flat_ms) # 0.14ms
    rel_ms = re.encode_relative_to(feat_ms, matches, vars) # 0.30ms
    vec = vr(rel_ms) # 0.12ms
    return vec, matches, fact_map

def setup_pipeline_second_run():
    args,_ = setup_pipeline_first_run()
    vec, matches, fact_map = pipeline_first_run(*args)
    (dict_state, wm, pipeline, match_names, vars) = args
    wm.modify(matches[0], "value", "4")
    return (wm, pipeline, matches, vars), {}

def pipeline_second_run(wm, pipeline, matches, vars):
    conv, fl, fa, re, vr = pipeline
    flat_ms = fl(wm) # 0.01ms
    print(flat_ms)
    feat_ms = fa(flat_ms) # 0.02ms
    print(feat_ms)
    rel_ms = re.encode_relative_to(feat_ms, matches, vars) # 0.78ms
    # print(rel_ms)
    vecs = vr(rel_ms) # 0.14ms
    return vecs

def test_pipeline():
    with cre_context("test_pipeline"):
        args,_ = setup_pipeline_first_run()
        (flt, nom), matches, fact_map = pipeline_first_run(*args)
        (dict_state, wm, pipeline, match_names, vars) = args
        conv, fl, fa, re, vr = pipeline

        wm.modify(fact_map["0_answer"], "value", "4")
        (new_flt, new_nom) = pipeline_second_run(wm, pipeline, matches, vars)
        assert len(np.unique(new_nom)) > len(np.unique(nom))
        nom = new_nom

        wm.modify(fact_map["1_carry"], "value", "1")
        (new_flt, new_nom) = pipeline_second_run(wm, pipeline, matches, vars)
        assert len(np.unique(new_nom)) > len(np.unique(nom))
        nom = new_nom

def _add_adjacent(src_fact, _, fact, src_attr_var, conds,
         fact_ptr_map, attr_flags, add_back_relation=False, var_prefix=None):
    src_ptr = src_fact.get_ptr()
    fact_ptr = fact.get_ptr()

    
    # Make sure that the adjacent fact is in 'fact_ptr_map'
    if(fact_ptr not in fact_ptr_map):
        fact_var = Var(fact._fact_type, f"{var_prefix}{len(conds)}")
        fact_ptr_map[fact_ptr] = fact_var

    fact_var = fact_ptr_map[fact_ptr]
    src_fact_var = fact_ptr_map[src_ptr]
    _conds = conds.get(fact_ptr, [])
    
    if(add_back_relation):
        for attr, config in fact._fact_type.filter_spec(*attr_flags).items():
            attr_val = getattr(fact, attr)
            if(not isinstance(attr_val, FactProxy) or
                attr_val.get_ptr() != src_ptr):
                continue
            attr_var = getattr(fact_var, attr)
            _conds.append(attr_var==src_fact_var)

    # Add a condition from val to 
    _conds.append(src_attr_var==fact_var)            
    conds[fact_ptr] = _conds


from itertools import chain

def conditions_from_facts(facts, _vars=None, add_neighbors=True,
     add_neighbor_holes=False, neighbor_back_relation=False, neighbor_req_n_adj=1, 
     alpha_flags=('visible',), parent_flags=('parent',), beta_flags=('relational',)):
    
    if(_vars is None):
        _vars = [Var(x._fact_type, f'A{i}') for i, x in enumerate(facts)]

    fact_ptr_map = {fact.get_ptr() : var for fact, var in zip(facts, _vars)}
    inp_fact_ptrs = set(fact_ptr_map.keys())
    
    # Make flag sets based on 
    beta_not_parent_flags = [*beta_flags,*[f"~{f}" for f in parent_flags]]
    alpha_candidate_flags = alpha_flags
    if(add_neighbor_holes):
        alpha_candidate_flags = [*alpha_flags, *parent_flags, *beta_flags]
    
    # Make Alphas (i.e. nvar = 1)
    cond_set = []
    nbr_conds = {}
    parent_conds = {}
    beta_conds = {}
    for fact, var in zip(facts,_vars):
        for attr, config in fact._fact_type.filter_spec(*alpha_candidate_flags).items():
            val = getattr(fact, attr)
            if(isinstance(val, FactProxy)): continue
            attr_var = getattr(var, attr)
            cond_set.append(attr_var==val)

        # TODO: Make parents 
        
        nxt_parents = [fact]
        while(len(nxt_parents) > 0):
            for nxt_parent in nxt_parents:
                spec = nxt_parent._fact_type.filter_spec(*parent_flags)
                for attr, config in spec.items():
                    pass
            nxt_parents = []

        # Make Betas (i.e. n_var=2) + Neighbors
        
        for attr, config in fact._fact_type.filter_spec(*beta_not_parent_flags).items():
            attr_val =  getattr(fact, attr)
            if(not isinstance(attr_val, FactProxy)): continue
            attr_var = getattr(var, attr)

            # Add a beta between the input facts.
            if(attr_val.get_ptr() in inp_fact_ptrs):
                _add_adjacent(fact, var, attr_val, attr_var, beta_conds,
                        fact_ptr_map, beta_not_parent_flags, var_prefix=None)

            # Add a beta between an input fact and a non-input neighbor.
            elif(add_neighbors):
                _add_adjacent(fact, var, attr_val, attr_var, nbr_conds,
                 fact_ptr_map, beta_not_parent_flags, add_back_relation=neighbor_back_relation,
                var_prefix="Nbr")

    if(neighbor_req_n_adj > 1):
        for ptr, lst in list(nbr_conds.items()): 
            if len(lst) < neighbor_req_n_adj:
                del nbr_conds[ptr]
                del fact_ptr_map[ptr]
        # TODO: Rename aliases of remaining vars to 0,1,2...
        # for i, ptr in enumerate(nbr_conds): 
        #     v = fact_ptr_map[ptr]
        #     v.alias = f"Nbr{i}"
        #     print(v.alias)

    cond_set = [*cond_set, 
                *chain(*parent_conds.values()),
                *chain(*beta_conds.values()),    
                *chain(*nbr_conds.values())
                ]

    _vars = list({v.get_ptr():v for v in fact_ptr_map.values()}.values())
    # print(_vars)   
    conds = _vars[0]
    for i in range(1, len(_vars)):
        conds = conds & _vars[i]

    for c in cond_set:
        conds = conds & c

    # print(conds)
    return conds

from operator import itemgetter
def test_condition_generalizing():
    from cre.rete import repr_match_iter_dependencies

    with cre_context("test_condition_generalizing"):
        (Container, TextField,Component, Button) = setup_fact_types()

        dict_state = new_mc_addition_state(567,354)
        # pprint(dict_state)
        wm = MemSet()
        conv = MemSetBuilder(wm)
        wm, fact_map = conv(dict_state, return_map=True)

        print({decode_idrec(f.idrec)[1] : f.id for f in  fact_map.values()})

        # # -----------------
        # # : Add2
        varz = [Var(TextField,'Sel'), Var(TextField,'Arg0'), Var(TextField,'Arg1')]
        sel_a, arg_a0, arg_a1 = itemgetter("0_answer", "0_upper","0_lower")(fact_map)
        sel_b, arg_b0, arg_b1 = itemgetter("1_answer", "1_upper","1_lower")(fact_map)

        print("-------------------------")
        # c_a = Conditions.from_facts([sel_a, arg_a0, arg_a1], varz)
        c_a = conditions_from_facts([sel_a, arg_a0, arg_a1], varz)
        print(repr(c_a))
        print(repr_match_iter_dependencies(c_a.get_matches(wm)))
        match_names = [[x.id for x in match][:3] for match in c_a.get_matches(wm)]
        print(match_names)
        assert match_names == [['0_answer', '0_upper', '0_lower']]

        print("-------------------------")
        # c_b = Conditions.from_facts([sel_b, arg_b0, arg_b1], varz)
        c_b = conditions_from_facts([sel_b, arg_b0, arg_b1], varz)
        match_names = [[x.id for x in match][:3] for match in c_b.get_matches(wm)]
        print(repr(c_b))
        print(match_names)
        assert match_names == [['1_answer', '1_upper', '1_lower']]

        # Generalized verison
        c_ab = varz[0] & varz[1] & varz[2] & c_a.antiunify(c_b)
        # print("---------------------")
        match_names = [[x.id for x in match][:3] for match in c_ab.get_matches(wm)]
        print(repr(c_ab))
        print(repr_match_iter_dependencies(c_ab.get_matches(wm)))
        print(match_names)
        
        # assert match_names == [['0_answer', '0_upper', '0_lower'], ['1_answer', '1_upper', '1_lower'], ['2_answer', '2_upper', '2_lower']]

        # Modify the state to make some not match
        wm.modify(fact_map['0_answer'],'value', '4')
        wm.modify(fact_map['0_answer'],'locked', True)
        wm.modify(fact_map['1_answer'],'value', '5')
        wm.modify(fact_map['1_answer'],'locked', True)

        print(wm)

        match_names = [[x.id for x in match][:3] for match in c_ab.get_matches(wm)]
        print(match_names)
        # assert match_names == [['2_answer', '2_upper', '2_lower']]

        # -----------------
        # : Carry2
        varz = [Var(TextField,'Sel'), Var(TextField,'Arg0'), Var(TextField,'Arg1')]
        sel_a, arg_a0, arg_a1 = itemgetter("1_carry", "0_upper","0_lower")(fact_map)
        sel_b, arg_b0, arg_b1 = itemgetter("2_carry", "1_upper","1_lower")(fact_map)

        # c_a = Conditions.from_facts([sel_a, arg_a0, arg_a1], varz)
        print("--------CARRRY------------")
        c_a = conditions_from_facts([sel_a, arg_a0, arg_a1], varz)
        match_names = [[x.id for x in match][:3] for match in c_a.get_matches(wm)]
        print("--c_a--")
        print(repr(c_a))
        print("----")
        print(match_names)
        assert match_names == [['1_carry', '0_upper', '0_lower']]

        # c_b = Conditions.from_facts([sel_b, arg_b0, arg_b1], varz)
        c_b = conditions_from_facts([sel_b, arg_b0, arg_b1], varz)
        match_names = [[x.id for x in match][:3] for match in c_b.get_matches(wm)]
        print("--c_b--")
        print(repr(c_b))
        print("----")
        print(match_names)
        assert match_names == [['2_carry', '1_upper', '1_lower']]

        c_ab = c_a.antiunify(c_b)
        match_names = [[x.id for x in match][:3] for match in c_ab.get_matches(wm)]
        print("--c_ab--")
        print(repr(c_ab))
        print("----")
        print(match_names)
        assert ['1_carry', '0_upper', '0_lower'] in match_names 
        assert ['2_carry', '1_upper', '1_lower'] in match_names 
        # NOTE : ['3_carry', '2_lower', '2_upper'] might not be available at this point

        sel_c, arg_c0, arg_c1 = itemgetter("3_carry", "2_upper","2_lower")(fact_map)
        # c_c = Conditions.from_facts([sel_c, arg_c0, arg_c1], varz)
        c_c = conditions_from_facts([sel_c, arg_c0, arg_c1], varz)
        c_abc = c_ab.antiunify(c_c)

        print(repr(c_abc))
        
        match_names = [[x.id for x in match][:3] for match in c_abc.get_matches(wm)]
        print(match_names)
        # print(repr_match_iter_dependencies(c_abc.get_matches(wm)))
        
        assert ['1_carry', '0_upper', '0_lower'] in match_names 
        assert ['2_carry', '1_upper', '1_lower'] in match_names 
        assert ['3_carry', '2_upper', '2_lower'] in match_names 

        # Check matching responds to modify
        wm.modify(fact_map['1_carry'],'value', '1')
        wm.modify(fact_map['1_carry'],'locked', True)
        wm.modify(fact_map['2_carry'],'value', '1')
        wm.modify(fact_map['2_carry'],'locked', True)

        match_names = [[x.id for x in match][:3] for match in c_abc.get_matches(wm)]
        print(match_names)
        assert match_names == [['3_carry', '2_upper', '2_lower']]
        # assert ['3_carry', '2_upper', '2_lower'] in match_names 

        with PrintElapse("Q"):
            [match for match in c_abc.get_matches(wm)]
        print("\n\n\n")

        # Check matching responds to retractions
        # wm.modify(fact_map['2_carry'],'value', '')
        # wm.modify(fact_map['2_carry'],'locked', False)
        wm.retract(fact_map['3_carry'])
        wm.modify(fact_map['2_carry'],'to_left', None)
        wm.modify(fact_map['hidey2'],'above', None)
        # wm.retract(fact_map['2_carry'])
        wm.modify(fact_map['1_carry'],'value', '')
        wm.modify(fact_map['1_carry'],'locked', False)

        match_names = [[x.id for x in match][:3] for match in c_abc.get_matches(wm)]
        print(match_names)
        print({decode_idrec(f.idrec)[1] : f.id for f in  fact_map.values()})
        assert match_names == [['1_carry', '0_upper', '0_lower']] #??
        
        # c = Conditions.from_facts([sel_c, arg_c0, arg_c1], varz,
        c = conditions_from_facts([sel_c, arg_c0, arg_c1], varz,
             neighbor_req_n_adj=2, alpha_flags=("visible", "few_valued"))
        print(c)
        match_names = [[x.id for x in match][:3] for match in c_abc.get_matches(wm)]
        print(match_names)
        assert match_names == [['1_carry', '0_upper', '0_lower']] #??



def test_b_pipeline_1st_run(benchmark):
    with cre_context("test_b_pipeline_1st_run"):
        benchmark.pedantic(pipeline_first_run,setup=setup_pipeline_first_run, warmup_rounds=1, rounds=10)

def test_b_pipeline_2nd_run(benchmark):
    with cre_context("test_b_pipeline_2nd_run"):
        benchmark.pedantic(pipeline_second_run,setup=setup_pipeline_second_run, warmup_rounds=1, rounds=10)


if __name__ == "__main__":
    import faulthandler; faulthandler.enable()

    # test_pipeline()
    test_condition_generalizing()
    # for i in range(2):
    #     args,_ = setup_pipeline_first_run()
    #     with PrintElapse("first_run"):
    #         pipeline_first_run(*args)

    # for i in range(2):
    #     args,_ = setup_pipeline_second_run()
    #     with PrintElapse("second_run"):
    #         pipeline_second_run(*args)



    


# ms, facts = conv(ds,return_map=True)

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


