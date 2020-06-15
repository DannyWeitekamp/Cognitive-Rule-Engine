from numba import types, njit, guvectorize,vectorize,prange
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, unicode_type, float64, NamedTuple, NamedUniTuple 
from numba.cpython.unicode import  _set_code_point
from utils import cache_safe_exec
from collections import namedtuple
import numpy as np
import timeit
import types as pytypes
import sys
import __main__

N=1000
def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))

cast_map = {
	"string" : str,
	"number" : float,
}

numba_type_map = {
	"string" : unicode_type,
	"number" : float64,	
}

numpy_type_map = {
	"string" : '|S%s',
	"number" : np.float64,	
}

numba_type_ids = {k:i  for i,k in enumerate(numba_type_map)}


@njit(cache=True,fastmath=True,nogil=True)
def _assert_map(v,enum_map,back_map,count):
	if(v not in enum_map):
		enum_map[v] = count.item()
		back_map[count.item()] = v
		count += 1

@njit(cache=True)
def exp_fixed_width(x,_min=20): 
	out = np.empty_like(x,dtype=np.int64)
	for i in range(len(x)):
		n = int(np.ceil(x[i]/_min))
		if(n <= 1):#
			out[i] = _min
			continue
		count = 0; 

		while( n != 0): 
			n >>= 1
			count += 1
		count += count & 1 #just even powers of 2
		out[i] = (1 << count)*_min; 
	return out

@njit(cache=True)
def charseq_len(s,max_l=100):
	i = 0
	for i in range(max_l):
		try:
			v = s[i]
		except Exception:
			break
	return i

NULL = chr(0)

@njit
def charseq_to_str(x,max_l=100):
	l = charseq_len(x)
	if(l == 0):
		return ""
	else:
		s = NULL*(l+1)
		for i in range(l):
			_set_code_point(s,i,x[i])
		return s[:l]

def gen_source_get_enumerized(name,spec,ind='   '):

	arg_str = ind*3 + "string_enums, number_enums,\n"
	arg_str += ind*3 + "string_backmap,number_backmap,\n"
	arg_str += ind*3 + "enum_counter"

	# attr_str = ind*3 + ", ".join(spec.keys()) 

	header = "@njit(cache=True,fastmath=True,nogil=True)\n"
	# header += "def {}_get_enumerized(\n{},\n{},assert_maps=True):\n".format(name,attr_str,arg_str)
	header += "def {}_get_enumerized(x,\n{},assert_maps=True):\n".format(name,arg_str)

	strings = ", ".join(["(x.{},{})".format(k,i) for i,k in enumerate(spec.keys()) if spec[k] == 'string'])
	numbers = ", ".join(["(x.{},{})".format(k,i) for i,k in enumerate(spec.keys()) if spec[k] == 'number'])

	body = ind + "enumerized = np.empty(({},),np.int64)\n".format(len(spec.keys()))
	# if(strings != ""):
	# 	body += ind +"if(assert_maps):\n"
	# 	for i,k in enumerate(spec.keys()):
	# 		if spec[k] == 'string':
	# 			body += ind*2 +"_assert_map({}, string_enums, string_backmap, enum_counter)\n".format(k)
	# 			# body += ind +"enumerized[{}] = string_enums[{}]\n".format(i,k)
	if(strings != ""):
	# 	for i,k in enumerate(spec.keys()):
	# 		if spec[k] == 'string':
	# 			body += ind +"enumerized[{}] = string_enums[{}]\n".format(i,k)
		body += ind + "for v,i in [{strings}]:\n".format(strings=strings)
		body += ind*2 + "if(assert_maps): _assert_map(v, string_enums, string_backmap, enum_counter)\n"
		body += ind*2 + "enumerized[i] = string_enums[v]\n"
	if(numbers != ""):	
		body += ind + "for v,i in [{numbers}]:\n".format(numbers=numbers)
		body += ind*2 + "if(assert_maps): _assert_map(v, number_enums, number_backmap, enum_counter)\n"
		body += ind*2 + "enumerized[i] = number_enums[v]\n"
	body += ind + "return enumerized\n\n"

	

	source = header + body#  +c
	return source

def gen_source_enumerize_nb_objs(name,spec,ind='   '):

	arg_str = "string_enums, number_enums,\n"
	arg_str += ind*3 + "string_backmap,number_backmap,\n"
	arg_str += ind*3 + "enum_counter"

	header = "@njit(cache=True,fastmath=True,nogil=True)\n"
	# header += "def {}_get_enumerized(\n{},\n{},assert_maps=True):\n".format(name,attr_str,arg_str)
	header += "def {}_enumerize_nb_objs(inp,out,{}):\n".format(name,arg_str)
	body = ind + 'for k,v in inp.items():\n'
	body += ind*2 + 'out[k] = {}_get_enumerized(v,{})\n\n'.format(name,arg_str)
	source = header + body+("\n"*10)
	return source




def gen_source_tuple_defs(name,spec,ind='   '):
	
	tuple_defs = "{} = namedtuple('{}', {}, module=__name__)\n".format(name,name,["%s"%k for k in spec.keys() if k != 'type'])
	sv = list(spec.values())
	if(len(set(sv))==1):
		tuple_defs += "NB_{}_NamedTuple = NamedUniTuple({},{},{})\n".format(name,str(numba_type_map[sv[0]]),len(sv),name)
	else:
		typ_str = ", ".join([str(numba_type_map[x]) for x in spec.values()])
		tuple_defs += "NB_{}_NamedTuple = NamedTuple(({}),{})\n".format(name,typ_str,name)
	# tuple_defs += "{} = NB_{}_NamedTuple.instance_class\n".format(name,name)
	return tuple_defs


def gen_source_pack_from_numpy(name,spec,ind='   '):
	header = "@njit(cache=True,fastmath=True,nogil=True)\n"
	header += "def {}_pack_from_numpy(inp,mlens):\n".format(name)

	cast_map = {"string":"charseq_to_str(x.{},mlens[{lens}])", 'number': 'float(x.{})'}

	body = ind + "out = Dict.empty(unicode_type,NB_{}_NamedTuple)\n".format(name)
	# body = ""
	body += ind + "for i in range(inp.shape[0]):\n"
	body += ind*2 + "x = inp[i]\n"
	body += ind*2 + "__name__ = charseq_to_str(x.__name__,mlens[0])\n"
	for i,(attr, typ) in enumerate(spec.items()):
		body += ind*2 + ("_{} = " + cast_map[typ]+ "\n").format(attr,attr,lens=i+1)
	body += ind*2 +"out[__name__] = {}({})\n".format(name,", ".join(["_%s"%x for x in spec.keys()]))
	body += ind + "return out\n"

	source = header + body #+("\n"*10)
	return source


@njit(cache=False)
def enumerize_nb_objs(inp,out,enumerize_f, string_enums, number_enums,
					string_backmap,number_backmap,enum_counter):
	for k,v in inp.items():
		out[k] = enumerize_f(v,string_enums, number_enums,
					string_backmap,number_backmap,enum_counter)



class Numbalizer(object):
	registered_specs = {}
	jitstructs = {}
	string_enums = Dict.empty(unicode_type,i8)
	number_enums = Dict.empty(f8,i8)
	string_backmap = Dict.empty(i8,unicode_type)
	number_backmap = Dict.empty(i8,f8)
	enum_counter = np.array(0)

	@classmethod
	def __init__(cls):
		for x in ["<#ANY>",None,'','?sel']:
			cls.enumerize_value(x)

	@classmethod
	def state_to_nb_objects(cls,state):
		tm = numpy_type_map
		#Throw the attribute values into tuples and find the lengths of
		#	all of the strings.
		data_by_type = {}
		mlens_by_type = {}
		for name, elm in state.items():
			typ = elm['type']
			spec = cls.registered_specs[typ]
			elm_data = tuple([name] + [v for k,v in elm.items() if k != 'type'])

			data = data_by_type.get(typ,[])
			data.append(elm_data)
			data_by_type[typ] = data

			if(typ not in mlens_by_type):
				mlens_by_type[typ] = [0]*len(elm_data)
			mlens = mlens_by_type[typ]

			for i,(attr, typ) in enumerate(spec.items()):
				if(spec[attr] == 'string'):
					L = len(elm[attr])
					if(L > mlens[i+1]): mlens[i+1] = L

		#Make a fixed bitwidth numpy datatype for the tuples based off of
		#	the max string lengths per attribute.
		out = {}
		for spec_typ,len_arrs in mlens_by_type.items():
			#Pick string widths that fit into [20,80,320,...] to avoid jit recompiles
			spec = cls.registered_specs[spec_typ]
			mlens = exp_fixed_width(np.array(mlens_by_type[spec_typ]),20) 
			dtype = dtype = [('__name__', tm['string']%int(mlens[0]))]
			for i,(attr, typ) in enumerate(spec.items()):
				if(typ == 'string'):
					dtype.append( (attr, tm['string']%int(mlens[i+1])) )
				else:
					dtype.append( (attr, tm[spec[attr]]) )
			pack_from_numpy = cls.jitstructs[spec_typ].pack_from_numpy
			out[spec_typ] = pack_from_numpy(np.array(data_by_type[spec_typ],dtype=dtype),mlens)
		return out	

	@classmethod	
	def jitstruct_from_spec(cls,name,spec,ind="   "):
		
		#Dynamically generate the named tuple for this spec and a numba wrapper for it		
		tuple_def_source = gen_source_tuple_defs(name,spec,ind)
		print(tuple_def_source)
		nb_nt_name = 'NB_{}_NamedTuple'.format(name)
		l,g = cache_safe_exec(tuple_def_source,gbls=globals())
		nt = 				l[name]
		nb_nt = 			l[nb_nt_name]

		#Do this to make the namedtuple picklable (see https://stackoverflow.com/questions/16377215/how-to-pickle-a-namedtuple-instance-correctly)
		setattr(__main__, nt.__name__, nt)
		nt.__module__ = "__main__"

		#Dynamically generate njit functions, pass in the namedtuple and wrapper as globals
		source = gen_source_get_enumerized(name,spec,ind=ind)
		source += gen_source_pack_from_numpy(name,spec,ind=ind)
		
		print(source)
		l,g = cache_safe_exec(source,gbls={**globals(), **{name:nt,nb_nt_name:nb_nt} })
		get_enumerized = 	l['{}_get_enumerized'.format(name)]
		pack_from_numpy =	l['{}_pack_from_numpy'.format(name)]
		

		source = gen_source_enumerize_nb_objs(name,spec,ind=ind)
		l,g = cache_safe_exec(source,gbls={**globals(), **{'{}_get_enumerized'.format(name):get_enumerized} })
		enumerize_nb_objs =	l['{}_enumerize_nb_objs'.format(name)]

		def py_get_enumerized(self,assert_maps=True):
			return get_enumerized(self,
								   string_enums=cls.string_enums,
								   number_enums=cls.number_enums,
								   string_backmap=cls.string_backmap,
								   number_backmap=cls.number_backmap,
								   enum_counter=cls.enum_counter,
								   assert_maps=assert_maps)
		nt.get_enumerized = py_get_enumerized#pytypes.MethodType(_get_enumerized, self) 
		nt._get_enumerized = get_enumerized#pytypes.MethodType(_get_enumerized, self) 
		nt.pack_from_numpy = pack_from_numpy
		nt.enumerize_nb_objs = enumerize_nb_objs

		return nt


	@classmethod
	def register_specification(cls, name, spec):
		spec = {k:v.lower() for k,v in spec.items()}
		if(name in cls.registered_specs):
			assert cls.registered_specs[name] == spec, \
			"Specification redefinition not permitted. Attempted on %r" % name
		else:
			cls.registered_specs[name] = spec
			print("start jit")
			cls.jitstructs[name] = cls.jitstruct_from_spec(name,spec)
			print("end jit")

	@classmethod
	def register_specifications(cls, specs):
		for name, spec in specs.items():
			cls.register_specification(name,spec)


	@classmethod
	def object_to_nb_object(cls,name,obj_d):
		assert 'type' in obj_d, "Object %s missing required attribute 'type'" % name
		assert obj_d['type'] in object_specifications, \
				 "Object specification not defined for %s" % obj_d['type']
		spec = cls.registered_specs[obj_d['type']]
		o_struct_type = cls.jitstructs[obj_d['type']]
		args = []
		for x,t in spec.items():
			try:
				args.append(cast_map[t](obj_d[x]))
			except ValueError as e:
				raise ValueError("Cannot cast %r to %r in %r of %r" % (obj_d[x],cast_map[t],x,name)) from e

		obj = o_struct_type(*args)
		return obj

	# @classmethod
	# def state_to_nb_objects(cls,state):
	# 	out = {}
	# 	for name, obj_d in state.items():
	# 		# print(name)
	# 		out[name] = cls.object_to_nb_object(name,obj_d)
	# 	return out

	@classmethod
	def nb_objects_to_enumerized(cls,nb_objects):
		out = Dict.empty(unicode_type,i8[:])
		for typ,objs in nb_objects.items():
			enumerize_nb_objs = cls.jitstructs[typ].enumerize_nb_objs

			enumerize_nb_objs(objs,out, cls.string_enums, cls.number_enums,
         		cls.string_backmap, cls.number_backmap, cls.enum_counter)
		return out

	@classmethod
	def infer_type(cls,value):
		if(isinstance(value,str)):
			return "string"
		elif(isinstance(value,(float,int))):
			return "number"
		else:
			raise ValueError("Could not infer type of %s" % value)

	@classmethod
	def enumerize(cls,value, typ=None):
		if(isinstance(value,(list,tuple))):
			return [cls.enumerize_value(x) for x in value]
		else:
			return cls.enumerize_value(x)

	@classmethod
	def enumerize_value(cls,value, typ=None):
		if(typ is None): typ = cls.infer_type(value)
		if(typ == 'string'):
			value = str(value)
			_assert_map(value,cls.string_enums,cls.string_backmap,cls.enum_counter)
			return cls.string_enums[value]
		elif(typ == 'number'):
			value = float(value)
			_assert_map(value,cls.number_enums,cls.number_backmap,cls.enum_counter)
			return cls.number_enums[value]
		else:
			raise ValueError("Unrecognized type %r" % typ)

	@classmethod
	def unenumerize_value(cls,value, typ=None):
		if(value in cls.string_backmap):
			return cls.string_backmap[value]
		elif(value in cls.number_backmap):
			return cls.number_backmap[value]
		else:
			raise ValueError("No enum for %r." % value)







object_specifications = {
	"InterfaceElement" : {
		"id" : "String",
		"value" : "String",
		"above" : "String",#["String","Reference"],
		"below" : "String",#["String","Reference"],
		"to_left" : "String",#["String","Reference"],
		"to_right" : "String",# ["String","Reference"],
		"x" : "Number",
		"y" : "Number"
	},
	"Trajectory" : {
		"x" : "Number",
		"y" : "Number",
		"z" : "Number",
		"dx" : "Number",
		"dy" : "Number",
		"dz" : "Number",
		"a_x" : "Number",
		"a_y" : "Number",
		"a_z" : "Number",
		"a_dx" : "Number",
		"a_dy" : "Number",
		"a_dz" : "Number",
	}

}

Numbalizer.register_specifications(object_specifications)

STATE_SIZE = 40
_state = {
	"i0" : {
		"type" : "InterfaceElement",
		"id" : "i0",
		"value" : "9",
		"above" : "",
		"below" : "i1",
		"to_left" : "",
		"to_right" : "",
		"x" : 100,
		"y" : 100,
	},
	"i1" : {
		"type" : "InterfaceElement",
		"id" : "i1",
		"value" : "7",
		"above" : "i0",
		"below" : "",
		"to_left" : "",
		"to_right" : "",
		"x" : 100,
		"y" : 200,
	}
}

state = {"ie" + str(i) : _state['i0'] for i in range(STATE_SIZE)}

print()

# for k,ie in Numbalizer.state_to_nb_objects(state).items():
# 	print(ie)
# 	print("ENUM",ie.get_enumerized())


_state2 = {
	"a" : {
		"type" : "Trajectory",
		"x" : 1,
		"y" : 2,
		"z" : 3,
		"dx" : 5.5,
		"dy" : 5.9,
		"dz" : 0.4,
		"a_x" : 1,
		"a_y" : 2,
		"a_z" : 3,
		"a_dx" : 5.5,
		"a_dy" : 5.9,
		"a_dz" : 0.4,
	}
}

state2 = {"ie" + str(i) : _state2['a'] for i in range(STATE_SIZE)}


nb_objects = Numbalizer.state_to_nb_objects(state)
nb_objects2 = Numbalizer.state_to_nb_objects(state2)
obj1 = nb_objects['InterfaceElement']['ie0']
obj2 = nb_objects2['Trajectory']['ie0']
	

nb_objects_real = Numbalizer.state_to_nb_objects(state)
nb_objects_real2 = Numbalizer.state_to_nb_objects(state2)
print(Numbalizer.nb_objects_to_enumerized(nb_objects_real))
# print(nb_objects_real)
# print(nb_objects_real.keys())
# raise ValueError()

def enumerize_obj():
	obj1.get_enumerized()

def enumerize_obj_nocheck():
	obj1.get_enumerized(False)

def enumerize_obj_justnums():
	obj2.get_enumerized()


def enumerize_value_string():
	Numbalizer.enumerize_value("159")

def enumerize_value_number():
	Numbalizer.enumerize_value(159)

def unenumerize_value():
	Numbalizer.unenumerize_value(2)


state_10 = {"ie" + str(i) : _state['i0'] for i in range(10)}
nb_objects_10 = Numbalizer.state_to_nb_objects(state_10)
def b10_state_to_objs():
	Numbalizer.state_to_nb_objects(state_10)
def b10_enumerize_objs():
	Numbalizer.nb_objects_to_enumerized(nb_objects_10)


state_40 = {"ie" + str(i) : _state['i0'] for i in range(40)}
nb_objects_40 = Numbalizer.state_to_nb_objects(state_40)
def b40_state_to_objs():
	Numbalizer.state_to_nb_objects(state_40)
def b40_enumerize_objs():
	Numbalizer.nb_objects_to_enumerized(nb_objects_40)

state_200 = {"ie" + str(i) : _state['i0'] for i in range(200)}
nb_objects_200 = Numbalizer.state_to_nb_objects(state_200)
def b200_state_to_objs():
	Numbalizer.state_to_nb_objects(state_200)
def b200_enumerize_objs():
	Numbalizer.nb_objects_to_enumerized(nb_objects_200)


print("-----Single Objs------")
print("enumerize_obj:",time_ms(enumerize_obj))
print("enumerize_obj_nocheck:",time_ms(enumerize_obj_nocheck))
print("enumerize_obj_justnums:",time_ms(enumerize_obj_justnums))
print("enumerize_value_string:",time_ms(enumerize_value_string))
print("enumerize_value_number:",time_ms(enumerize_value_number))
print("unenumerize_value:",time_ms(enumerize_value_number))
print("-----State 10 Objs------")
print("state_to_objs:",time_ms(b10_state_to_objs))
print("enumerize_objs:",time_ms(b10_enumerize_objs))
print("-----State 40 Objs------")
print("state_to_objs:",time_ms(b40_state_to_objs))
print("enumerize_objs:",time_ms(b40_enumerize_objs))
print("-----State 200 Objs------")
print("state_to_objs:",time_ms(b200_state_to_objs))
print("enumerize_objs:",time_ms(b200_enumerize_objs))
