import numpy as np
import numba
from numba import njit
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.types import UniTuple, ListType, unicode_type, UnicodeCharSeq
from numbert.numbalizer import Numbalizer
from numbert.core import STRING_DTYPE
from numbert.conditions import Conditions, conds_apply



u4_list_type = ListType(u4)
u4_array = u4[:]
@njit(cache=True)
def get_enumerized_elems_and_candidates(enumerized_state, pos_spec_patterns,
             pattern_types, pattern_names_to_inds, ref_flags, string_enums):
    '''
    
    enumerized_state -- A state enumerized with generated from Numbalizer.nb_object_to_enumerized() {typ : {name: enum, ....},...}
    pos_spec_patterns -- A list of patterns that we will use to find candidates for those patterns.
    pattern_types -- A list of struct types for each pattern (e.g. TextField)
    string_enums -- The mapping of strings to their enumerized values from Numbalizer.string_enums
    '''
    

    n_elems = 0
    for typ,enum_dicts in enumerized_state.items():
        n_elems += len(enum_dicts)

    elems = List.empty_list(u4_array)
    elem_types = List.empty_list(unicode_type)
    elem_names = np.empty((n_elems,),dtype=np.uint32)
    elem_name_dict = Dict.empty(u4,u1)
    i = 0
    for typ,objs in enumerized_state.items():
        for name,enums in objs.items():
            v = string_enums[name]
            elem_names[i] = v
            elem_name_dict[v] = i
            elems.append(enums.astype(np.uint32))
            elem_types.append(typ)
            i += 1

    #Make a version of each pattern which zeros out references 
    #   to bindable elemenets (i.e. pair constraints are zeroed).
    scrubbed_ps = List()
    for i,ps_i in enumerate(pos_spec_patterns):
        is_ref = ref_flags[str(pattern_types[i])]
        ps_is = ps_i.copy()
        for j in range(len(ps_is)):
            if(is_ref[j] and ps_i[j] in pattern_names_to_inds):
                ps_is[j] = 0
        scrubbed_ps.append(ps_is)

    
    candidate_lists = List.empty_list(u4_list_type)
    
    #Prefill candiditate lists
    for _ in range(len(pattern_types)):
        candidate_lists.append(List.empty_list(u4))

    #Go through enumerized state and fill candidates 
    elm_count = 0
    for typ, objs in enumerized_state.items():
        # is_ref = ref_flags[typ]
        applicable_patterns = np.where(np.array([et == typ for et in pattern_types],dtype=np.uint32))[0]
        for name, _enums in objs.items():
            enums = _enums.astype(np.uint32)
            #Add the enumerized elements into a flat list of all of them
            # elems.append(enums)
            for i in applicable_patterns:
                ps_is = scrubbed_ps[i]
                #Mark as candidate if for all features the scurbbed pattern is zeroed
                #    or the feature value matches the element's value at that feature
                pattern_applies = ((ps_is == 0) | (ps_is == enums)).all()
                if(pattern_applies): candidate_lists[i].append(elm_count)
            elm_count += 1

    #Pack candidates into numpy arrays
    candidates = List.empty_list(u4_array)
    for cand_list in candidate_lists:
        cands = np.empty((len(cand_list),),dtype=np.uint32)
        for i,c in enumerate(cand_list):
            cands[i] = c
        candidates.append(cands)
    return elems, elem_names, candidates





@njit(nogil=False, parallel=False,fastmath=False,cache=True)
def fill_pairs_at(partial_matches,i,pair_matches):
    ''' 
    Adds new bindings to each partial match in list partial_matches, by
    trying to apply consistent pairs of elements associated with pattern_i.
    '''
    #For every pattern_j adjacent to pattern_i
    for j, pair_matches_ij in pair_matches[i].items():
        new_pms = List()
        #Apply the following to each partial_match "pm" so far:
        for pm in partial_matches:
            altered = False
            already_consistent = False
            
            #For each consistent pair (e_i, e_j) of elements matching patterns
            #   i and j respectively:
            for (k, e_i, e_j) in pair_matches_ij:
                okay = True
                #Bind the pair to partial match "pm" if:                
                #pattern_i is unbound or pattern_i is bound to e_i and pattern_j is unbound
                if(not (pm[i] == 0 or (pm[i] == e_i and pm[j] == 0))): okay = False
                #and pattern_j is unbound or pattern_j is bound to e_j and pattern_i is unbound
                if(not (pm[j] == 0 or (pm[j] == e_j and pm[i] == 0))): okay = False
                
                #and e_i and e_j are not bound elsewhere in partial match "pm"
                for p in range(len(pm)):
                    if(p != i and p != j and (pm[p] == e_i or pm[p] == e_j)):
                        okay = False; break;

                #Mark the partial match as consistent for this iteration if it
                #   is consistent with this pair.
                if(pm[j] == e_j and pm[i] == e_i): already_consistent = True
                
                #Then append the new partial match to the set of partial matches
                if(okay):
                    new_pm = pm.copy()
                    new_pm[i], new_pm[j] = e_i, e_j
                    new_pms.append(new_pm)
                    altered = True

            #If partial match "pm" hasn't picked up any new bindings to patterns i and j
            #   but at least one was not yet bound, then no further binding can be found 
            #   for "pm" and it is rejected. Otherwise as long as pm[i] is consistent 
            #   with pm[j] keep pm for the next iteration.
            if(not altered and pm[i] != 0 and pm[j] != 0 and already_consistent):
                new_pms.append(pm)
        partial_matches = new_pms
    return partial_matches

@njit(nogil=False, parallel=False,fastmath=False,cache=True)
def fill_singles_at(partial_matches,i,cand_names):
    '''
    For pattern_i which is free floating and without a pattern
    tracked neighbor, bind the candidates for this pattern
    to every partial match in partial_matches. 
    '''
    new_pms = List()
    for pm in partial_matches:
        if(pm[i] == 0):
            for cn in cand_names:
                if(not (cn == pm).any()):
                    new_pm = pm.copy()
                    new_pm[i] = cn
                    new_pms.append(new_pm)
        else:
            new_pms.append(pm)
    return new_pms


@njit(nogil=False, parallel=False,fastmath=False,cache=True,locals={"k": u4})
def find_consistent_pairs(patterns, elems, candidates,
                        elem_names, pattern_names_to_inds):
    '''
    Builds the list "pair_matches" which holds a dictionary for each
    pattern_i with keys for the indicies of adjacent patterns_j and 
    values are lists of pairs of elements which are mutually 
    consistent candidates for pattern_i and pattern_j respectively.
    '''
    pair_matches = List()
    for i,(pattern_i, cands_inds_i) in enumerate(zip(patterns,candidates)):
        pair_matches_i = Dict.empty(i8,list_of_u4_triple)
        #Loop through every feature of pattern_i
        for k, f_k in enumerate(pattern_i):

            #If a reference to pattern_j is present at feature k of pattern_i
            if(f_k in pattern_names_to_inds):
                j = pattern_names_to_inds[f_k]
                elem_names_j = elem_names[candidates[j]]
                pair_matches_ij = List.empty_list(u4_triple)

                #Find pairs of consistent candidates between the two patterns
                for c1_ind in cands_inds_i:
                    f_k_of_c1 = elems[c1_ind][k]
                    for c2_val in elem_names_j:
                        if(f_k_of_c1 == c2_val):
                            #Add the triple (k, e_i,e_j) as a candidate pair
                            pair_matches_ij.append((k,elem_names[c1_ind],c2_val))
                pair_matches_i[j] = pair_matches_ij
        pair_matches.append(pair_matches_i)
    return pair_matches

# pos_spec_patterns,elems,candidates,elem_names, pattern_names_to_inds)
u4_triple = UniTuple(u4,3)
list_of_u4_triple = ListType(u4_triple)
@njit(nogil=False, parallel=False,fastmath=False,cache=True)
def match_iterative(patterns,elems,candidates,elem_names, pattern_names_to_inds):
    '''
    Produces every consistent matching of elems to patterns.
    '''
    n_patterns = len(patterns)
    pair_matches = find_consistent_pairs(patterns, elems, candidates,
                        elem_names, pattern_names_to_inds)
    partial_matches = List()
    partial_matches.append(np.zeros((n_patterns,),dtype=np.uint32))

    for i in range(n_patterns):
        partial_matches = fill_pairs_at(partial_matches,i,pair_matches)

    for i in range(n_patterns):
        if(len(pair_matches[i]) == 0):
            partial_matches = fill_singles_at(partial_matches,i,[elem_names[c] for c in candidates[i]])
    return partial_matches

@njit(cache=True)#NOTE: conds_data,attr_inds_by_type,
def get_matches(enumerized_state, pos_spec_patterns, pattern_names, pattern_types, conds_data, ref_flags, string_enums, string_backmap, attr_inds_by_type):
    pattern_names_to_inds = Dict.empty(u4,i8)
    for i,name in enumerate(pattern_names):
        pattern_names_to_inds[name] = i

    elems, elem_names, candidates = get_enumerized_elems_and_candidates(enumerized_state,
                                     pos_spec_patterns, pattern_types, pattern_names_to_inds,ref_flags,string_enums)
    
    enumerized_matches = match_iterative(pos_spec_patterns,elems,candidates,elem_names, pattern_names_to_inds)

    condition_passing_matches = List()
    for i, e_match in enumerate(enumerized_matches):
        if(conds_data is not None):
            applies = conds_apply(e_match, conds_data, \
                 enumerized_state, string_enums, attr_inds_by_type)
            if(not applies): continue
                
        condition_passing_matches.append(e_match)

    out = np.empty((len(condition_passing_matches),len(pattern_types)), dtype=pattern_types.dtype)
    for i,e_match in enumerate(condition_passing_matches):
        for j,name in enumerate(e_match):
            out[i][j] = string_backmap[name]
    return out


class Matcher(object):
    names = None
    types = None
    pos_patterns = None
    neg_patterns = None
    conditions = None

    enumerized_names = None
    
    def __init__(self,numbalizer,names=None,types=None,pos_patterns=None,
                    neg_patterns=None,conditions=None,config=None):
        self.numbalizer = numbalizer

        if(config is not None):
            self.set_config(config)
            return

        n,t = names is not None, types is not None
        if(n and t):
            self.set_patterns(names=names,types=types,pos_patterns=pos_patterns,neg_patterns=neg_patterns)
        elif(n or t):
            raise ValueError("Must set names and types at the same time")

        if(conditions is not None):
            self.set_conditions(conditions)

    def _check_patterns(self,patterns):
        if(len(patterns) == 0): return None
        if(isinstance(patterns,(list,tuple))): patterns = List(patterns)
        # print(type(patterns))
        assert isinstance(patterns,List),  ""
        for i,pattern in enumerate(patterns):
            assert isinstance(pattern,np.ndarray), "patterns must be of type numpy.ndarray, got %s" % type(pattern)
            # if(len(pattern.shape)==1): pattern = patterns[i] = pattern.expand_dims(0)
            assert np.issubdtype(pattern.dtype, np.integer), "pattern must use integer dtype, got %s" % pattern.dtype
            assert len(pattern.shape) == 1, "pattern shape must be (d,), got shape %s" % str(pattern.shape)
        return patterns

    def set_patterns(self,names,types,pos_patterns=None,neg_patterns=None):
        # print("SET patterns")
        if(isinstance(names,(list,tuple))): names = np.array(names,dtype=STRING_DTYPE)
        if(isinstance(types,(list,tuple))): types = np.array(types,dtype=STRING_DTYPE)
        assert isinstance(names,np.ndarray), "names must be of type numpy.ndarray, got %s" % type(names)
        assert isinstance(types,np.ndarray), "types must be of type numpy.ndarray, got %s" % type(types)
        self.names = names
        self.types = types

        self.enumerized_names = self.numbalizer.enumerize(self.names).astype(np.uint32)

        if(pos_patterns is None): pos_patterns = None; return;
        self.pos_patterns = self._check_patterns(pos_patterns)

        if(neg_patterns is None): pos_patterns = None; return;
        self.neg_patterns = self._check_patterns(neg_patterns)

    def set_conditions(self,*args,**kwargs):
        if('numbalizer' not in kwargs): kwargs['numbalizer'] = self.numbalizer
        if(len(args) > 0 and isinstance(args[0],Conditions)):
            self.conditions = args[0]
        else:    
            self.conditions = Conditions(*args,**kwargs)
        
    def _unenum_v(self,x,t):
        return self.numbalizer.unenumerize_value(x,t)

    def get_config(self):
        pattern_config = {}
        for i, (name,t) in enumerate(zip(self.names,self.types)):
            attrs = self.numbalizer.registered_specs[t].keys()
            pattern_config[name] = {}
            pattern_config[name]['type'] = t
            if(self.pos_patterns is not None):
                pattern_config[name]["pos_pattern"] = {attr :self._unenum_v(self.pos_patterns[i][k],t) for k,attr in enumerate(attrs)}
            if(self.neg_patterns is not None):
                pattern_config[name]["neg_pattern"] = {attr :self._unenum_v(self.neg_patterns[i][k],t) for k,attr in enumerate(attrs)}
        out = {"patterns" : pattern_config}
        if(self.conditions is not None): out['conditions'] = self.conditions.get_config()
        return out

    def _parse_pattern(self,pattern_config,typ):
        assert typ in self.numbalizer.registered_specs, "%r is not a registered type." % typ
        spec = self.numbalizer.registered_specs[typ]
        pattern = []
        for attr,s_obj in spec.items():
            attr_typ = s_obj['type']
            if(attr in pattern_config):
                pattern.append(self.numbalizer.enumerize_value(pattern_config[attr],attr_typ))
            else:
                pattern.append(0)
        return np.array(pattern,dtype=np.uint32)
                


    def set_config(self,config):
        assert 'patterns' in config, "Matcher config requires 'patterns'" 
        pattern_configs = config['patterns']

        names = []
        types = []
        pos_patterns = []
        neg_patterns = []
        for name,pattern_config in pattern_configs.items():
            # print(pattern_config)
            names.append(name)
            assert 'type' in pattern_config, "Every pattern must have a type. %s:%s" % (name, pattern_config)
            typ = pattern_config['type']
            types.append(typ)
            # if("pos_pattern" in pattern_config):
            pos_patterns.append(self._parse_pattern(pattern_config.get('pos_pattern',{}),typ))
            # else:

            # if("neg_pattern" in pattern_config):
            neg_patterns.append(self._parse_pattern(pattern_config.get('neg_pattern',{}),typ))
        # assert len(names) == len(types), 
        assert len(pos_patterns) == 0 or len(pos_patterns) == len(types), "If using pos_patterns all patterns need it."
        assert len(pos_patterns) == 0 or len(pos_patterns) == len(types), "If using neg_patterns all patterns need it."
        # print(names, types, pos_patterns, neg_patterns)
        # print()
        self.set_patterns(names,types,pos_patterns,neg_patterns)        
        if('conditions' in config):
            conditions_config = config['conditions']
            print("conditions_config")
            print(conditions_config)
            if(isinstance(conditions_config,dict)):
                self.set_conditions(numbalizer=self.numbalizer,config=config['conditions'])
            elif(isinstance(conditions_config,Conditions)):
                print("SET CONDITIONS")
                self.set_conditions(conditions_config)
            else:
                raise ValueError("Condition should be config or Condition() object, but got %s" % (type(conditions_config)))


    def get_matches(self,enumerized_state):
        if(self.pos_patterns is None): raise Exception("Cannot get_matches with no pos_patterns.")
        #enumerized_state, pos_spec_patterns, pattern_names, pattern_types, conds_data, ref_flags, string_enums, string_backmap, attr_inds_by_type
        conds_data = self.conditions.conds_data if self.conditions is not None else None
        print("conds_data", conds_data)
        return get_matches(enumerized_state, self.pos_patterns, self.enumerized_names,
                        self.types, conds_data, self.numbalizer.spec_flags['reference'],
                        self.numbalizer.string_enums, self.numbalizer.string_backmap, self.numbalizer.attr_inds_by_type)

if __name__ == "__main__":
    state = {
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
            "value": 7,
            "above": "B1",
            "below": None,
            "to_left" : "C2",
            "to_right": None,
        },
        "C2": {
            "type" : "TextField",
            "value": 8,
            "above": "B2",
            "below": None,
            "to_left" : "C3",
            "to_right": "C1",
        },
        "C3": {
            "type" : "TextField",
            "value": 9,
            "above": "B3",
            "below": None,
            "to_left" : "C4",
            "to_right": "C2",
        },
        "C4": {
            "type" : "TextField",
            "value": 9,
            "above": "A4",
            "below": None,
            "to_left" : None,
            "to_right": "C3",
        }
    }
    config = {
        "patterns" : {
            "sel": {
                "type" : "TextField",
                "pos" : {
                    "above": "arg1",
                }
            },
            "arg0": {
                "type" : "TextField",
                "pos" : {
                    "below": "arg1",
                }
            },
            "arg1": {
                "type" : "TextField",
                "pos" : {
                    "above": "arg0",
                    "below": "sel",
                }
            }
        },
        "conditions" : {
            "bindables" : {
                "sel" : [
                        ["to_left" , "above", "value"],
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
                ["EQUAL", True , 0 ,1],
                ["EQUAL", False , 0 ,"2"],
            ],
            "clause" : [
                [["AND", 0, 0]]
            ]
        }
    }
    numbalizer = Numbalizer()
    matcher = Matcher(numbalizer,config=config)
    print(matcher.get_config())

    
    # numbalizer.register_specification("TextField",{
    #     "value" : "string",
    #     "above" : "string",
    #     "below" : "string",
    #     "to_left" : "string",
    #     "to_right" : "string",
    #     })

    # print(type(List()))
    # matcher = Matcher(numbalizer,["?sel"],["TextField"],[np.zeros(5,dtype=np.uint32)])
    # enumerized_state = numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects(state))
    
    # matches = matcher.get_matches(enumerized_state)    
    # print(matches)
    # print(matcher.get_config())

    # config = {
    #     "sel": {
    #         "type" : "TextField",
    #         "pos_pattern" : {
    #             "above": "arg1",
    #         }
    #     },
    #     "arg0": {
    #         "type" : "TextField",
    #         "pos_pattern" : {
    #             "below": "arg1",
    #         }
    #     },
    #     "arg1": {
    #         "type" : "TextField",
    #         "pos_pattern" : {
    #             "above": "arg0",
    #             "below": "sel",
    #         }
    #     }
    # }
    # matcher.set_config(config)
    # print(matcher.get_config())

    # matches = matcher.get_matches(enumerized_state)    
    # print(matches)
    # print(matcher.get_config())
    # raise ValueError()






