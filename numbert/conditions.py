import itertools
import numpy as np
from numba import njit
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.types import UniTuple, ListType, unicode_type, UnicodeCharSeq, optional, Array
from numbert.numbalizer import Numbalizer
from numbert.core import STRING_DTYPE


AND = 1
OR = 2

def _assert_cond(x):
    if(not isinstance(x,AlphaCondition)):
        return TRUTHY_Node(x)
    else:
        return x


class AlphaNode(object):    
    def __and__(A, B):
        A, B = _assert_cond(A), _assert_cond(B)
        AisAND, BisAND = isinstance(A, AND_Node), isinstance(B, AND_Node)
        if AisAND and BisAND:
            return AND_Node(*(A.args + B.args))
        elif AisAND:
            return AND_Node(*(A.args + [B]))
        elif BisAND:
            return AND_Node(*(A + B.args))
        else:
            return AND_Node(A,B)

    def __or__(A, B):
        A, B = _assert_cond(A), _assert_cond(B)
        AisOR, BisOR = isinstance(A, OR_Node), isinstance(B, OR_Node)
        if AisOR and BisOR:
            return OR_Node(*(A.args + B.args))
        elif AisOR:
            return OR_Node(*(A.args + [B]))
        elif BisOR:
            return OR_Node(*(A + B.args))
        else:
            return OR_Node(A,B)
    def __invert__(A):
        A = _assert_cond(A)
        return NOT_Node(A)
    __not__ = __invert__

    def __eq__(A, B):
        raise ValueError("Can only use == with variables and literals.")

    def as_tuple():
        raise NotImplemented()
        # return EQUALS_Node(A,B)

class AlphaBindable(object):
    def __init__(self,name,typ, numbalizer):
        self._name = name
        self._type = typ
        # self.numbalizer = numbalizer
        # assert self.type in self.numbalizer.registered_specs, "Type %s is not registered." % self.type
        # self.spec = self.numbalizer.registered_specs[self.type]
        # print(self.spec)
    def __getattr__(self, key):
        if(key in ['_name','_typ']): return self.__dict__.get(key, None)
        # if(key == '_name'): return self._name
        # if(key == '_typ'): return self._typ
        # assert key in self.spec, "Spec for %s has no attribute %s" % (self.type,key)
        # if("reference" in self.spec[key]['flags']):
        #     return AlphaDereference(self,key)
        # else:
        return AlphaVar(self,key)
    def __str__(self):
        return str(self._name)
    def __repr__(self):
        return "Bindable({},{})".format(self._name,self._type)
    def as_tuple(self):
        return (AlphaBindable,self._name,self._type)


class AlphaVar(AlphaNode):
    def __init__(self,parent,attr_name):
        assert isinstance(parent,(AlphaBindable,AlphaVar)), "Expecting parent of type AlphaBindable, but got %s." %  type(parent)
        self._parent = parent
        self._attr_name = attr_name
    def __getattr__(self, key):
        if(key in ['_parent','_attr_name']): return self.__dict__.get(key, None)
        return AlphaVar(self,key)
    def __eq__(A, B):
        return EQUALS_Node(A,B)
    def __ne__(A, B):
        return NOT_Node(EQUALS_Node(A,B))
    def __str__(self):
        return "{}.{}".format(str(self._parent),self._attr_name) 
    def as_tuple(self):
        _parent = self._parent
        derefs = []
        # print(type(_parent))
        while(isinstance(_parent,AlphaVar)):
            derefs.append(_parent._attr_name)
            _parent = _parent._parent

        # print(derefs)
        # print("MOOOP")
        # print(("AlphaVar",_parent._type,*reversed(derefs),self._attr_name))
        return (_parent._name,_parent._type,*reversed(derefs),self._attr_name)

# class AlphaDereference(AlphaVar):
#     def __getattr__(self, key):
#         assert key in self.spec, "Spec for %s has no attribute %s" % (self.type,key)
#         if("reference" in self.spec[key]['flags']):
#             return AlphaDereference(self,key)
#         else:
#             return AlphaVar(self,key)


class AlphaCondition(AlphaNode):
    enum = 0
    def __init__(self,*args):
        self.args = list(args)
    def as_tuple(self):
        return (self.enum,*[x.as_tuple() if isinstance(x,AlphaNode) else x for x in self.args])    

AND_enum = 1
OR_enum = 2
NOT_enum = 3
EQUALS_enum = 4
TRUTHY_enum = 5

class AND_Node(AlphaCondition):
    enum = AND_enum
    def __str__(self):
        return "({})".format(" & ".join([str(x) for x in self.args]))

class OR_Node(AlphaCondition):
    enum = OR_enum
    def __str__(self):
        return "({})".format(" | ".join([str(x) for x in self.args]))

class NOT_Node(AlphaCondition):
    enum = NOT_enum
    def __new__(cls,x):
        # if(isinstance(x, EQUALS_Node)):
        #     return NOT_EQUALS_Node(*x.args)
        # if(isinstance(x, NOT_EQUALS_Node)):
        #     return EQUALS_Node(*x.args)
        if(isinstance(x, NOT_Node)):
            return x.args[0]
        return super(NOT_Node, cls).__new__(cls) 

    def __str__(self):
        return "~{}".format(str(self.args[0]))

class EQUALS_Node(AlphaCondition):
    enum = EQUALS_enum
    def __str__(self):
        return "({})".format(" == ".join([str(x) for x in self.args]))

# class NOT_EQUALS_Node(AlphaCondition):
#     enum = NOT_EQUALS_enum
#     def __str__(self):
#         return "({})".format(" != ".join([str(x) for x in self.args]))

class TRUTHY_Node(AlphaCondition):
    enum = TRUTHY_enum
    def __str__(self):
        return str(self.args[0])




def assign_depths(node,depth_map={}):
    depth = 0
    node_tup = node.as_tuple() if isinstance(node, AlphaNode) else node
    if(isinstance(node,AlphaCondition)):
        max_depth = 0
        for arg in node.args:
            max_depth = max(assign_depths(arg,depth_map)+1,max_depth)
        depth = max_depth
    elif(isinstance(node,AlphaVar)):
        print(node, node_tup)
        depth = depth_map[node_tup] = 0
    
    prev_depth = depth_map.get(node_tup,0)
    depth_map[node_tup] = max(depth,prev_depth)
    return depth

def organize_by_depth(node_depths):
    out = [[] for _ in range(max(node_depths.values())+1)]
    for node_tup, depth in node_depths.items():
       out[depth].append(node_tup) 
    return out

def flatten_conditions(node,binding_names, numbalizer):
    node_depths = {}
    assign_depths(node,node_depths)
    value_map = {}
    index_map = {}
    offsets = []
    instructions = []

    value_counter = 0
    index_counter = 0
    offset_counter = 0
    instr_offset = 0
    derefs_by_bindable = List([List.empty_list(ListType(unicode_type)) for _ in range(len(binding_names))])
    relations = List()
    for i,layer in enumerate(organize_by_depth(node_depths)):
        # print("layer %s" % i)
        for node_tup in layer:
            if(i == 0):
                if(isinstance(node_tup, tuple)):
                    binding_name = node_tup[0] 
                    # print("binding_name")
                    j = binding_names.index(binding_name)
                    # print(binding_name, j)
                    derefs_by_bindable[j].append(List(node_tup[2:]))
                    value_map[node_tup] = value_counter
                    value_counter += 1
            if(i == 1):
                # print("value_map",value_map)
                op, left, right = node_tup
                is_literal = not isinstance(right, tuple)
                left = value_map[left]
                if(is_literal):
                    right = numbalizer.enumerize_value(right)
                else:
                    right = value_map[right]
                relations.append(np.array([op, is_literal,left,right],dtype=np.uint32))
                # print('relation',relations[-1])
            # print(node_tup)
            if(i > 0):
                index_map[node_tup] = index_counter
                index_counter += 1
            if(i > 1):
                instructions += [node_tup[0],len(node_tup)-1] + [index_map[node_tup[j]] for j in range(1,len(node_tup))]
                offsets.append(instr_offset)
                instr_offset = len(instructions)
        

    return derefs_by_bindable, relations, np.array(instructions,dtype=np.uint32), np.array(offsets,dtype=np.uint32)

@njit(cache=True)
def apply_op(op,args):
    # print("apply", op, args)
    if(op == 1):
        return args.all()
    elif(op == 2):
        return args.any()
    elif(op == 3):
        return not args[0]
    elif(op == 4):
        return args[0] == args[1]
    elif(op == 5):
        return args[0]

enum_to_op = {
    1 : "AND",
    2 : "OR",
    3 : "NOT",
    4 : "EQUAL",
    5 : "TRUTHY",
}
op_to_enum = {v:k for k,v in enum_to_op.items()}


@njit(cache=True)
def execute_instructions(inp, instructions,offsets):
    out = np.empty(len(inp)+len(offsets),np.uint8)
    out[:len(inp)] = inp
    o_off =len(inp)
    for i,i_off in enumerate(offsets):
        op = instructions[i_off]
        args_start = i_off+2
        args_len = instructions[i_off+1]
        # print(args_start,args_len)
        args = instructions[args_start:args_start+args_len]
        # print(o_off,args)
        out[o_off] = apply_op(op,out[args])
        o_off += 1
        # out[len(inp)]
    return out

class Conditions(object):
    def __init__(self,cond=None, binding_names=None, numbalizer=None, config=None):
        assert numbalizer is not None, "numbalizer is required argument"
        self.numbalizer = numbalizer
        if(cond is not None and binding_names is not None):
            self.conds_data = flatten_conditions(cond, binding_names, numbalizer)
            self.binding_names = binding_names
        if(config is not None):
            self.set_config(config)
    def set_config(self,config):
        assert "bindables" in config, "config missing 'bindables'"
        assert "relations" in config, "config missing 'relations': [...]"
        assert "clause" in config, "config missing 'clause' :[...]"
        config_bindables = config["bindables"]
        config_relations = config["relations"]
        config_clause = config["clause"]

        # print(config_bindables)
        self.binding_names = []
        derefs_by_bindable = List()
        for name,derefs in config_bindables.items():
            self.binding_names.append(name)
            derefs_by_bindable.append(List([List([x for x in y]) for y in derefs]))

        relations = List()
        for relation in config_relations:
            op,is_literal,left,right = relation
            op = op_to_enum[op]
            if(is_literal): right = self.numbalizer.enumerize_value(right)
            relations.append(np.array([op,is_literal,left,right],dtype=np.uint32))

        instructions = []
        offsets = []
        prev_offset = 0
        for sub_clause in config_clause:
            op = sub_clause[0]
            assert isinstance(op,str), "Operator should be string but got %r." % op
            args = sub_clause[1:]
            assert all([isinstance(x,int) for x in args]), "Args should be integers but got %r." % args
            instructions += [op_to_enum[op], len(args),*args]
            offsets.append(prev_offset)
            prev_offset = len(instructions)

        instructions = np.array(instructions,dtype=np.uint32)
        offsets = np.array(offsets,dtype=np.uint32)
        self.conds_data = (derefs_by_bindable, relations, instructions, offsets)

    def get_config(self):
        derefs_by_bindable, relations, instructions, offsets = self.conds_data
        config_bindables = {}
        for name, derefs in zip(self.binding_names,derefs_by_bindable):
            config_bindables[name] = [[x for x in y ] for y in derefs]

        config_relations = []
        for relation in relations:
            r = [int(x) for x in relation] 
            op, is_literal, left, right = r[0],bool(r[1]),r[2],r[3]
            op = enum_to_op[op]
            if(is_literal): right = self.numbalizer.unenumerize_value(right)
            config_relations.append([op,is_literal,left,right])

        config_clause = []
        for i in range(len(offsets)):
            offset = offsets[i]
            op = enum_to_op[instructions[offset]]
            n_args = instructions[offset+1]
            args = instructions[offset+2: offset+2+n_args]
            config_clause.append([op,*[int(x) for x in args]])
        return {"bindables" : config_bindables,
                "relations" : config_relations,
                "clause": config_clause}




'''
numbalizer = Numbalizer()
numbalizer.register_specification("TextField",{
    "value" : "string",
    "text" : "string",
    "poop" : "string",
    "above" : {"type" : "string", "flags" : ["reference"]},
    "below" : {"type" : "string", "flags" : ["reference"]},
    "to_left" : {"type" : "string", "flags" : ["reference"]},
    "to_right" : {"type" : "string", "flags" : ["reference"]},
})


sel = AlphaBindable("sel","TextField", numbalizer)
arg0 = AlphaBindable("arg0","TextField", numbalizer)
arg1 = AlphaBindable("arg1","TextField", numbalizer)

# print(sel.value == sel.arg0 == sel.arg1)
print(sel)
print(sel.value == sel.text)
print(~(sel.value == sel.text) & (sel.poop == sel.text))
print(~(sel.value == sel.text) | (sel.poop == sel.text))
print((sel.value != sel.text) | (sel.poop == sel.text))
print(type(sel.to_left))
print((sel.to_left.value != sel.text) | (sel.poop == sel.text))
# raise ValueError()
print(sel.value & sel.value)
print(sel.value | sel.value)

a = (sel.to_left.to_right.value == sel.value)
# print(a.as_tuple())
# raise ValueError()
b = (sel.value == arg0.value)
c = (arg0.value == arg1.value)
print()

big = (a & b & ~c) | (~a & ~~b & ~c)

c_obj = Conditions(big,['sel','arg0','arg1'],numbalizer)
print(c_obj.conds_data)
print("HERE1")
config = c_obj.get_config()
print("HERE2")
print(config)
c_obj2 = Conditions(numbalizer=numbalizer,config=config)
print(c_obj2.conds_data)
print("HERE3")

config2 = c_obj2.get_config()
print(config)
print(config2)
# raise ValueError()

conds_data = flatten_conditions(big,['sel','arg0','arg1'],numbalizer)
# print(deref_paths, relations,)
# print(instructions, offsets)
# print(execute_instructions(np.array([0,0,0],dtype=np.uint8),instructions, offsets))
# raise ValueError()

a = (sel.value == "9")
b = (arg1.value == "4")
c = (arg0.value == "6")
big = (a & b & ~c) | (~a & ~~b & ~c)
conds_data2 = flatten_conditions(big,['sel','arg0','arg1'],numbalizer)
# print(deref_paths, relations,)
# print(instructions, offsets)
# raise ValueError()
# print(execute_instructions(np.array([0,0,0],dtype=np.uint8),instructions, offsets))



state1 = {
    "A1": {
        "type" : "TextArea",
        "value": 1,
        "above": "",
        "below": "B1",
        "to_left" : "A2",
        "to_right": "",
    },
    "A2": {
        "type" : "TextArea",
        "value": 2,
        "above": "",
        "below": "B2",
        "to_left" : "A3",
        "to_right": "A1",
    },
    "A3": {
        "type" : "TextArea",
        "value": 3,
        "above": "",
        "below": "B3",
        "to_left" : "A4",
        "to_right": "A2",
    },
    "A4": {
        "type" : "TextArea",
        "value": 3,
        "above": "",
        "below": "C4",
        "to_left" : "",
        "to_right": "A3",
    },
    "B1": {
        "type" : "TextArea",
        "value": 4,
        "above": "A1",
        "below": "C1",
        "to_left" : "B2",
        "to_right": "",
    },
    "B2": {
        "type" : "TextArea",
        "value": 5,
        "above": "A2",
        "below": "C2",
        "to_left" : "B3",
        "to_right": "B1",
    },
    "B3": {
        "type" : "TextArea",
        "value": 6,
        "above": "A3",
        "below": "C3",
        "to_left" : "",
        "to_right": "B2",
    },
    "C1": {
        "type" : "TextArea",
        "value": 7,
        "above": "B1",
        "below": "",
        "to_left" : "C2",
        "to_right": "",
    },
    "C2": {
        "type" : "TextArea",
        "value": 8,
        "above": "B2",
        "below": "",
        "to_left" : "C3",
        "to_right": "C1",
    },
    "C3": {
        "type" : "TextArea",
        "value": 9,
        "above": "B3",
        "below": "",
        "to_left" : "C4",
        "to_right": "C2",
    },
    "C4": {
        "type" : "TextArea",
        "value": 9,
        "above": "A4",
        "below": "",
        "to_left" : "",
        "to_right": "C3",
    }
}

numbalizer = Numbalizer()
numbalizer.register_specification("TextArea",{
    "value" : "string",
    "above" : {"type" : "string", "flags" : ["reference"]},
    "below" : {"type" : "string", "flags" : ["reference"]},
    "to_left" : {"type" : "string", "flags" : ["reference"]},
    "to_right" : {"type" : "string", "flags" : ["reference"]},
    })
state1_enumerized = numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects(state1))


#A cond should be (attr_ind, literal)
# single_conds = List[bindables[all of its single conditions]]
# cond_types = List[bindables[all of its single conditions]]
'''

u4_list_type = ListType(u4)
def get_candidates(enumerized_state, single_conds,cond_types):
    #Go through enumerized state and fill candidates 

    candidate_lists = List.empty_list(u4_list_type)

    for _ in range(len(cond_types)):
        candidate_lists.append(List.empty_list(u4))

    elm_count = 0
    for typ, objs in enumerized_state.items():
        applicable_conds = np.where(np.array([et == typ for et in cond_types],dtype=np.uint32))[0]
        # print(applicable_conds)
        for name, obj in objs.items():
            for i in applicable_conds:
                
                # print(obj[attr_ind],literal)
                l1_consistency = np.array([obj[attr_ind] == literal 
                    for attr_ind, literal in single_conds[i]],dtype=np.uint8)

                if(consistent): candidate_lists[i].append(elm_count)
            elm_count += 1
    # print(candidate_lists)


#derefs_by_bindable = List[ List[np.array('to_left',...),...]]

#string-> index
#
@njit(cache=True)
def elem_ind_from_deref(elem,typ,dref_attr,elem_name_dict,attr_inds_by_type):
    return elem_name_dict.get(elem[attr_inds_by_type[typ][dref_attr]],np.uint32(-1))

u4_array = u4[:]
@njit(cache=True)
def get_enumerized_elems(enumerized_state,string_enums):
    '''
    
    enumerized_state -- A state enumerized with generated from Numbalizer.nb_object_to_enumerized() {typ : {name: enum, ....},...}
    pos_spec_patterns -- A list of patterns that we will use to find candidates for those patterns.
    pattern_types -- A list of struct types for each pattern (e.g. TextField)
    string_enums -- The mapping of strings to their enumerized values from Numbalizer.string_enums
    '''
    # print("BEFORE4")
    n_elems = 0
    for typ,enum_dicts in enumerized_state.items():
        n_elems += len(enum_dicts)

    elems = List.empty_list(u4_array)
    elem_types = List.empty_list(unicode_type)
    elem_names = np.empty((n_elems,),dtype=np.uint32)
    elem_name_dict = Dict.empty(u4,u1)
    i = 0
    for typ,objs in enumerized_state.items():
        for name,_enums in objs.items():
            enums = _enums.astype(np.uint32)
            v = string_enums[name]
            elem_names[i] = v
            elem_name_dict[v] = i
            elems.append(enums)
            elem_types.append(typ)
            i += 1

    return elems, elem_names, elem_name_dict, elem_types

@njit(cache=True)
def resolve_dereferences(match, derefs_by_bindable, elems, elem_names, elem_name_dict, elem_types, attr_inds_by_type):
    out = List()
    # print("BEGINNING", derefs_by_bindable)
    for i, (elm_name,derefs) in enumerate(zip(match,derefs_by_bindable)):
        elem, typ = elems[elem_name_dict[elm_name]], elem_types[elem_name_dict[elm_name]]
        # print("INP ELEM", elem_names[elm_index])
        # print([x for x in elem_name_dict.keys()][[x for x in elem_name_dict.values()]])
        attr_inds_map = attr_inds_by_type[typ]

        out_i = List.empty_list(u4)
        for j,deref in enumerate(derefs):
            # print("deref",deref)
            # print("j" , j)
            d_elem, d_elem_typ = elem, typ
            okay = True
            for d in deref[:-1]:
                # print('d',d)
                #elem_ind_from_deref(d_elem,d_elem_typ,d,elem_name_dict,attr_inds_by_type)
                attr_val = d_elem[attr_inds_by_type[d_elem_typ][d]]
                if(attr_val not in elem_name_dict):
                    okay = False
                    break
                d_ind = elem_name_dict[attr_val]
                d_elem, d_elem_typ = elems[d_ind], elem_types[d_ind]
            # print("okay", okay,deref[-1],attr_inds_by_type[d_elem_typ])
            #If every dereference up to the last one succeeded and the last one is in the final obj
            if(okay and deref[-1] in attr_inds_by_type[d_elem_typ]):
                # print("J")
                # print(attr_inds_by_type,d_elem_typ)
                # print(deref[-1])
                # print(attr_inds_by_type[d_elem_typ])
                # print(d_elem)
                value = d_elem[attr_inds_by_type[d_elem_typ][deref[-1]]]
                out_i.append(value)
            else:
                out_i.append(0)
        out.append(out_i)
    # print("END", out)
    return out

# print(numbalizer.unenumerize_value(19))
# match = np.array([8,9,10],dtype=np.uint32) #C2
# derefs_by_bindable = [[np.array(['to_left','above',"value"]),np.array(['to_right','above',"value"]),np.array(['to_right','below',"value"])]]

# def bleh(match,derefs_by_bindable,enumerized_state,numbalizer):
#     elems, elem_names, elem_name_dict, elem_types = get_enumerized_elems(enumerized_state,numbalizer.string_enums)
#     return resolve_dereferences(match,derefs_by_bindable, elems, elem_names, elem_name_dict, elem_types, numbalizer.attr_inds_by_type)

# blop = bleh(match,derefs_by_bindable,state1_enumerized,numbalizer)
# print([numbalizer.unenumerize_value(x) for x in blop[0]])
# raise ValueError()
# resolve_dereferences()

@njit(cache=True)
def apply_relation_op(rel_op, left, right):
    if(rel_op == 4):
        return left == right
    return 0


l1_conds = []

@njit(cache=True)
def conds_apply(match,conds_data, enumerized_state, string_enums, attr_inds_by_type):
    derefs_by_bindable, relations, cond_instructions, cond_offsets = conds_data
    # print("BEFORE2")
    elems, elem_names, elem_name_dict, elem_types = get_enumerized_elems(enumerized_state,string_enums)
    # print("BEFORE3")
    values_by_bindable = resolve_dereferences(match,derefs_by_bindable, elems, elem_names,
                                 elem_name_dict, elem_types, attr_inds_by_type)
    # print("AFTER2")
    values = List()
    for vs in values_by_bindable:
        for v in vs:
            values.append(v) 

    l1_inps = np.empty(len(relations),dtype=np.uint8)
    for i,relation in enumerate(relations):
        rel_op, is_literal, left, right = relation
        # print(values,left, right)
        left_value = values[left]
        right_value = right if(is_literal) else values[left] 
        # print(rel_op,left_value,right_value)
        l1_inps[i] = apply_relation_op(rel_op,left_value,right_value)
    # execute_instructions(inp, instructions,offsets):
    # print("cond_instructions",cond_instructions,cond_offsets)
    out = execute_instructions(l1_inps, cond_instructions, cond_offsets) 
    # print(l1_inps)
    # print(out)
    return out[-1]

'''
# match, derefs_by_bindable, enumerized_state, conds_data,string_enums,attr_inds_by_type
applies = conds_apply(match, conds_data, state1_enumerized, 
                        numbalizer.string_enums, numbalizer.attr_inds_by_type)
print(applies)

sel = AlphaBindable("sel","TextField", numbalizer)
arg0 = AlphaBindable("arg0","TextField", numbalizer)
arg1 = AlphaBindable("arg1","TextField", numbalizer)


a = sel.to_left.to_left.to_left.value == sel.to_left.to_left.value
b = arg1.to_left.value == "2"
c = arg0.to_left.value == "5"
conds = (a & b & c)
conds_data = flatten_conditions(conds,['sel','arg0','arg1'],numbalizer)
match = np.array([7,0,4],dtype=np.uint32)
applies = conds_apply(match, conds_data, state1_enumerized, 
    numbalizer.string_enums, numbalizer.attr_inds_by_type)
print(applies)






#NOTE: Should prevent OR Being Applied above pair constraints?
#NOTE: Maybe just use patterns to narrow down, then conditions after 

# print(state1_enumerized)
# cond_types = np.array(["TextArea","TextArea"], dtype="U")
# single_conds = List([List([(0,22)]),List([(0,10)])])
# print(single_conds)
# get_candidates(state1_enumerized, single_conds,cond_types)
            
'''