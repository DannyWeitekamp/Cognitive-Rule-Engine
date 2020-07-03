from numba import types, njit, guvectorize,vectorize,prange
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple 
from numba.cpython.unicode import  _set_code_point
from numbert.utils import cache_safe_exec
from collections import namedtuple
import numpy as np
import timeit
import types as pytypes
import sys
import __main__



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


attr_map_i_type = DictType(i8,i8)
attr_map_type = DictType(i8,attr_map_i_type)
attr_maps_type = DictType(unicode_type,attr_map_type)
bool_array = u1[:]
elm_present_type = DictType(unicode_type,bool_array)


VectInversionData = namedtuple('VectInversionData',['element_names','slice_nominal','slice_continous','width_nominal','width_continous','attr_records'])
# NBVectInversionData = NamedTuple([elm_present_type,UniTuple(i8,2),UniTuple(i8,2),i8,i8,attr_map_type],VectInversionData)
NBVectInversionData = NamedTuple([ListType(unicode_type),i8[::1],i8[::1],i8,i8,i8[:,::1]],VectInversionData)
# elm_present_by_type, type_slices_nominal, type_slices_continuous, type_widths_nominal, type_widths_continous, attr_maps_by_type):

@njit(cache=True)
def enumerized_to_vectorized(enumerized_states,nominal_maps,number_backmap,return_inversion_data=False):
	
	# nominals = Dict()
	# continuous = Dict()
	n_states = len(enumerized_states)

	#Fill elm_present_by_type, attr_maps, and type_widths_nominal
	#	elm_present_by_type: Dict<String, Dict<String, bool[:]>> keyed by Type then Elm_name
	#		and taking value of array of booleans indicating if an element of that type and name
	#		are present in states [0,...,n_states].
	#	attr_maps: Dict<String, Dict<Int, Dict<Int,Int>>> keyed by Type, then attr_position,
	#		then attr_value, and taking value of the offset of that attr_value in a one-hotted encoding
	#	type_widths_nominal: Dict<String, Int> keyed by Type and valued by length of an elemented of that
	#		type one-hot encoded 
	elm_present_by_type = Dict.empty(unicode_type,elm_present_type)
	attr_maps_by_type = Dict.empty(unicode_type,attr_map_type)
	type_widths_nominal = Dict.empty(unicode_type,i8)
	for k,state in enumerate(enumerized_states):
		for typ,elms in state.items():
			nominal_map = nominal_maps[typ]
			type_width_nominal = type_widths_nominal.get(typ,0)
			if(typ not in elm_present_by_type):
				elm_present_by_type[typ] = Dict.empty(unicode_type,bool_array)
			elm_present = elm_present_by_type[typ]
			if(typ not in attr_maps_by_type):
				attr_maps_by_type[typ] = Dict.empty(i8,attr_map_i_type)
			attr_map = attr_maps_by_type[typ]
			for j,(name,elm) in enumerate(elms.items()):
				if(j == 0):
					for i in range(len(elm)):
						if(nominal_map[i]):
							if(i not in attr_map):
								attr_map[i] = Dict.empty(i8,i8)
				if(name not in elm_present):
					elm_present[name] = np.zeros((n_states,),dtype=np.uint8)
				elm_present[name][k] = True

				for i,attr in enumerate(elm):
					if(nominal_map[i]):
						attr_map_i = attr_map[i]
						if(attr not in attr_map_i):
							attr_map_i[attr] = len(attr_map_i)
							type_width_nominal += 1
			type_widths_nominal[typ] = type_width_nominal
			elm_present_by_type[typ] = elm_present

	#Compute the sizes of the output arrays
	total_continuous_slots = 0
	type_widths_continous = Dict.empty(unicode_type,i8)
	type_slices_continuous = np.empty(len(elm_present_by_type)+1,dtype=np.int64)
	type_slices_continuous[0] = 0
	for i,typ in enumerate(elm_present_by_type.keys()):
		w = np.sum(nominal_maps[typ]==0)
		type_widths_continous[typ] = w
		total_continuous_slots += w*len(elm_present_by_type[typ])
		type_slices_continuous[i+1] = total_continuous_slots

	total_nominal_slots = 0

	type_slices_nominal = np.empty(len(elm_present_by_type)+1,dtype=np.int64)
	type_slices_nominal[0] = 0
	for i,(typ, type_width_nominal) in enumerate(type_widths_nominal.items()):
		total_nominal_slots += type_widths_nominal[typ]*len(elm_present_by_type[typ])
		type_slices_nominal[i+1] = total_nominal_slots

	#Instatiate the output arrays
	vect_nominals = np.empty((n_states,total_nominal_slots), dtype=np.uint8)#, order='F')
	vect_continuous = np.empty((n_states,total_continuous_slots), dtype=np.float64)#, order='F')
	

	#Fill the output arrays
	for k,state in enumerate(enumerized_states):
		offset_n,offset_c = 0,0
		for typ,elms in state.items():
			nominal_map, type_width_n,type_width_c  = nominal_maps[typ], type_widths_nominal[typ], type_widths_continous[typ]
			elm_present, attr_map = elm_present_by_type[typ],  attr_maps_by_type[typ]
			for name, is_present in elm_present.items():
				if(is_present[k]):
					vect_nominals[k,offset_n:offset_n+type_width_n] = 0
					elm_offset_n,elm_offset_c = offset_n, offset_c
					for i,attr in enumerate(elms[name]):
						if(nominal_map[i]):
							attr_map_i = attr_map[i]
							vect_nominals[k,elm_offset_n+attr_map[i][attr]] = True
							elm_offset_n += len(attr_map_i)
						else:
							vect_continuous[k,elm_offset_c] = number_backmap[attr]
							elm_offset_c += 1
				else:
					vect_nominals[k,offset_n:offset_n+type_width_n] = 255
					vect_continuous[k,offset_c:offset_c+type_width_c] = np.nan
				offset_n += type_width_n
				offset_c += type_width_c

	if(return_inversion_data):
		inversion_data = Dict.empty(unicode_type,NBVectInversionData)

		for j,typ in enumerate(elm_present_by_type.keys()):
			flat_attr_map = np.empty((type_widths_nominal[typ],3),dtype=np.int64)
			count = 0 
			for attr_ind in attr_map:
				attr_map_attr = attr_map[attr_ind]
				for val_ind,pos in attr_map_attr.items():
					flat_attr_map[count,0] = attr_ind
					flat_attr_map[count,1] = val_ind
					flat_attr_map[count,2] = pos
					count += 1

			elm_names = List.empty_list(unicode_type)
			for elm_name in elm_present_by_type[typ].keys():
				elm_names.append(elm_name)

			inv_dat = VectInversionData(elm_names,
						np.array([type_slices_nominal[j],type_slices_nominal[j+1]],dtype=np.int64),
						np.array([type_slices_continuous[j],type_slices_continuous[j+1]],dtype=np.int64),
						type_widths_nominal[typ],type_widths_continous[typ],
						flat_attr_map)
			inversion_data[typ] = inv_dat

		return vect_nominals,vect_continuous,inversion_data #(elm_present_by_type,type_slices_nominal,type_slices_continuous,type_widths_nominal,type_widths_continous,attr_maps_by_type)#elm_present_type#,type_slices_nominal,type_slices_continuous,type_widths_nominal,type_widths_continous, attr_maps_by_type
	else:
		return vect_nominals,vect_continuous,None

@njit(cache=True)
def decode_vectorized(indicies, isnominal, inversion_data):
	out = List()
	for i, i_is_nom in zip(indicies,isnominal):
		j = 0
		typ = ""
		offset = 0
		inv_dat = None
		for typ, inv_dat in inversion_data.items():
		# for j,typ in zip(range(1,len(type_slices)),type_widths_nominal.keys()):
			if(i_is_nom):
				slc_s, slc_e = inv_dat.slice_nominal[0],inv_dat.slice_nominal[1]
			else:
				slc_s, slc_e = inv_dat.slice_continous[0],inv_dat.slice_continous[1]
			if(i >= slc_s and i < slc_e):
				offset = slc_s
				break

		width = inv_dat.width_nominal if(i_is_nom) else inv_dat.width_continous
		# attr_map = attr_maps_by_type[typ]
		# print(i)
		i -= offset
		# print(i)
		elm_n = i // width
		i = i%width
		# print(i,elm_n)

		elm_name = "UNDEFINED"
		for e,name in enumerate(inv_dat.element_names):
			if(e == elm_n):
				elm_name = name
				break

		attr_ind = 0
		val_ind = 0
		count = 0
		# okay = True
		# for attr_record in inv_dat.attr_records:
		attr_record = inv_dat.attr_records[i]
		attr_ind = attr_record[0]
		val_ind = attr_record[1]
		position = attr_record[2]
			# if(count == i): break
		# for attr_ind in attr_map:
			# attr_map_attr = attr_map[attr_ind]
			# for val_ind in attr_map_attr:
				# print(count, "->", attr_ind,val_ind,attr_map_attr[val_ind])
				# if(count == i):
				# 	okay = False
				# count += 1
				# if(not okay): break
			# print("COUNT",count)
			# if(not okay): break

		# print(i,":",typ,elm_name,attr_ind,val_ind)
		out.append((typ,elm_name,attr_ind,val_ind))
	return out








def infer_type(value):
	if(isinstance(value,str)):
		return "string"
	elif(isinstance(value,(float,int))):
		return "number"
	else:
		raise ValueError("Could not infer type of %s" % value)

def infer_nb_type(value):
	return numba_type_map[infer_type(value)]



Dict_Unicode_to_Enums = DictType(unicode_type,i8[:])

class Numbalizer(object):
	registered_specs = {}
	jitstructs = {}
	string_enums = Dict.empty(unicode_type,i8)
	number_enums = Dict.empty(f8,i8)
	string_backmap = Dict.empty(i8,unicode_type)
	number_backmap = Dict.empty(i8,f8)
	enum_counter = np.array(0)
	nominal_maps = Dict.empty(unicode_type,u1[:])

	def __init__(self):
		for x in ["<#ANY>",'','?sel']:
			self.enumerize_value(x)

	def jitstruct_from_spec(self,name,spec,ind="   "):
		
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

		def py_get_enumerized(_self,assert_maps=True):
			return get_enumerized(_self,
								   string_enums=self.string_enums,
								   number_enums=self.number_enums,
								   string_backmap=self.string_backmap,
								   number_backmap=self.number_backmap,
								   enum_counter=self.enum_counter,
								   assert_maps=assert_maps)
		nt.get_enumerized = py_get_enumerized#pytypes.MethodType(_get_enumerized, self) 
		nt._get_enumerized = get_enumerized#pytypes.MethodType(_get_enumerized, self) 
		nt.pack_from_numpy = pack_from_numpy
		nt.enumerize_nb_objs = enumerize_nb_objs

		return nt

	def register_specification(self, name, spec):
		spec = {k:v.lower() for k,v in spec.items()}
		if(name in self.registered_specs):
			assert self.registered_specs[name] == spec, \
			"Specification redefinition not permitted. Attempted on %r" % name
		else:
			self.registered_specs[name] = spec
			print("start jit")
			self.jitstructs[name] = self.jitstruct_from_spec(name,spec)
			self.nominal_maps[name] = np.array([x == "string" for k,x in spec.items() if k != 'type'],dtype=np.uint8)
			print("end jit")

	def register_specifications(self, specs):
		for name, spec in specs.items():
			self.register_specification(name,spec)


	def state_to_nb_objects(self,state):

		tm = numpy_type_map
		#Throw the attribute values into tuples and find the lengths of
		#	all of the strings.
		data_by_type = {}
		mlens_by_type = {}
		for name, elm in state.items():
			typ = elm['type']
			spec = self.registered_specs[typ]
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
			spec = self.registered_specs[spec_typ]
			mlens = exp_fixed_width(np.array(mlens_by_type[spec_typ]),20) 
			dtype = dtype = [('__name__', tm['string']%int(mlens[0]))]
			for i,(attr, typ) in enumerate(spec.items()):
				if(typ == 'string'):
					dtype.append( (attr, tm['string']%int(mlens[i+1])) )
				else:
					dtype.append( (attr, tm[spec[attr]]) )
			pack_from_numpy = self.jitstructs[spec_typ].pack_from_numpy
			out[spec_typ] = pack_from_numpy(np.array(data_by_type[spec_typ],dtype=dtype),mlens)
		return out	

	def object_to_nb_object(self,name,obj_d):
		assert 'type' in obj_d, "Object %s missing required attribute 'type'" % name
		assert obj_d['type'] in object_specifications, \
				 "Object specification not defined for %s" % obj_d['type']
		spec = self.registered_specs[obj_d['type']]
		o_struct_type = self.jitstructs[obj_d['type']]
		args = []
		for x,t in spec.items():
			try:
				args.append(cast_map[t](obj_d[x]))
			except ValueError as e:
				raise ValueError("Cannot cast %r to %r in %r of %r" % (obj_d[x],cast_map[t],x,name)) from e

		obj = o_struct_type(*args)
		return obj

	def nb_objects_to_enumerized(self,nb_objects):
		out = Dict.empty(unicode_type,Dict_Unicode_to_Enums)
		for typ,objs in nb_objects.items():
			enumerize_nb_objs = self.jitstructs[typ].enumerize_nb_objs

			out_typ = out[typ] = Dict.empty(unicode_type,i8[:])
			enumerize_nb_objs(objs,out_typ, self.string_enums, self.number_enums,
         		self.string_backmap, self.number_backmap, self.enum_counter)
		return out


	def enumerize(self,value, typ=None):
		if(isinstance(value,(list,tuple))):
			return [self.enumerize_value(x) for x in value]
		else:
			return self.enumerize_value(x)

	def enumerize_value(self,value, typ=None):
		if(typ is None): typ = infer_type(value)
		if(typ == 'string'):
			value = str(value)
			_assert_map(value,self.string_enums,self.string_backmap,self.enum_counter)
			return self.string_enums[value]
		elif(typ == 'number'):
			value = float(value)
			_assert_map(value,self.number_enums,self.number_backmap,self.enum_counter)
			return self.number_enums[value]
		else:
			raise ValueError("Unrecognized type %r" % typ)

	def unenumerize_value(self,value, typ=None):
		if(value in self.string_backmap):
			return self.string_backmap[value]
		elif(value in self.number_backmap):
			return self.number_backmap[value]
		else:
			raise ValueError("No enum for %r." % value)

	def enumerized_to_vectorized(self,enumerized_state,return_inversion_data=False):

		nominal, continuous, inversion_data = enumerized_to_vectorized(enumerized_state,
										self.nominal_maps,self.number_backmap,
										return_inversion_data=return_inversion_data)
		out = {'nominal':nominal,'continuous':continuous, 'inversion_data': inversion_data}
		return out

	def remap_vectorized(self,vectorized):
		pass





# numbalizer = Numbalizer()


# object_specifications = {
# 	"InterfaceElement" : {
# 		"id" : "String",
# 		"value" : "String",
# 		"above" : "String",#["String","Reference"],
# 		"below" : "String",#["String","Reference"],
# 		"to_left" : "String",#["String","Reference"],
# 		"to_right" : "String",# ["String","Reference"],
# 		"x" : "Number",
# 		"y" : "Number"
# 	},
# 	"Trajectory" : {
# 		"x" : "Number",
# 		"y" : "Number",
# 		"z" : "Number",
# 		"dx" : "Number",
# 		"dy" : "Number",
# 		"dz" : "Number",
# 		"a_x" : "Number",
# 		"a_y" : "Number",
# 		"a_z" : "Number",
# 		"a_dx" : "Number",
# 		"a_dy" : "Number",
# 		"a_dz" : "Number",
# 	}

# }

# numbalizer.register_specifications(object_specifications)

# STATE_SIZE = 40
# _state = {
# 	"i0" : {
# 		"type" : "InterfaceElement",
# 		"id" : "i0",
# 		"value" : "9",
# 		"above" : "",
# 		"below" : "i1",
# 		"to_left" : "",
# 		"to_right" : "",
# 		"x" : 100,
# 		"y" : 100,
# 	},
# 	"i1" : {
# 		"type" : "InterfaceElement",
# 		"id" : "i1",
# 		"value" : "7",
# 		"above" : "i0",
# 		"below" : "",
# 		"to_left" : "",
# 		"to_right" : "",
# 		"x" : 100,
# 		"y" : 200,
# 	}
# }

# state = {"ie" + str(i) : _state['i0'] for i in range(STATE_SIZE)}

# print()

# # for k,ie in Numbalizer.state_to_nb_objects(state).items():
# # 	print(ie)
# # 	print("ENUM",ie.get_enumerized())


# _state2 = {
# 	"a" : {
# 		"type" : "Trajectory",
# 		"x" : 1,
# 		"y" : 2,
# 		"z" : 3,
# 		"dx" : 5.5,
# 		"dy" : 5.9,
# 		"dz" : 0.4,
# 		"a_x" : 1,
# 		"a_y" : 2,
# 		"a_z" : 3,
# 		"a_dx" : 5.5,
# 		"a_dy" : 5.9,
# 		"a_dz" : 0.4,
# 	}
# }

# state2 = {"ie" + str(i) : _state2['a'] for i in range(STATE_SIZE)}


# nb_objects = numbalizer.state_to_nb_objects(state)
# nb_objects2 = numbalizer.state_to_nb_objects(state2)
# obj1 = nb_objects['InterfaceElement']['ie0']
# obj2 = nb_objects2['Trajectory']['ie0']
	

# nb_objects_real = numbalizer.state_to_nb_objects(state)
# nb_objects_real2 = numbalizer.state_to_nb_objects(state2)


# enumerized = numbalizer.nb_objects_to_enumerized(nb_objects_real)
# enumerized_states = List()
# enumerized_states.append(enumerized)
# enumerized_states.append(numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects({"ie" + str(i+3) : _state['i1'] for i in range(STATE_SIZE)})))


# print("nominal maps")
# print(Numbalizer.nominal_maps)
# print("number backmaps")
# print(Numbalizer.number_backmap)


# nominals, continuous = enumerized_to_vectorized_legacy(enumerized_states,Numbalizer.nominal_maps,Numbalizer.number_backmap)
# nominals, continuous = enumerized_to_vectorized(enumerized_states,Numbalizer.nominal_maps,Numbalizer.number_backmap)


# np.set_printoptions(threshold=2000)
# print("nominals")
# print(nominals)
# print("continuous")
# print(continuous)
# # raise ValueError()
# # print(np.isfortran(nominals))
# # print(np.isfortran(continuous))

# # print(nb_objects_real)
# # print(nb_objects_real.keys())
# # raise ValueError()

# def enumerize_obj():
# 	obj1.get_enumerized()

# def enumerize_obj_nocheck():
# 	obj1.get_enumerized(False)

# def enumerize_obj_justnums():
# 	obj2.get_enumerized()


# def enumerize_value_string():
# 	numbalizer.enumerize_value("159")

# def enumerize_value_number():
# 	numbalizer.enumerize_value(159)

# def unenumerize_value():
# 	numbalizer.unenumerize_value(2)


# state_10 = {"ie" + str(i) : _state['i0'] for i in range(10)}
# nb_objects_10 = numbalizer.state_to_nb_objects(state_10)
# def b10_state_to_objs():
# 	numbalizer.state_to_nb_objects(state_10)
# def b10_enumerize_objs():
# 	numbalizer.nb_objects_to_enumerized(nb_objects_10)


# state_40 = {"ie" + str(i) : _state['i0'] for i in range(40)}
# nb_objects_40 = numbalizer.state_to_nb_objects(state_40)
# def b40_state_to_objs():
# 	numbalizer.state_to_nb_objects(state_40)
# def b40_enumerize_objs():
# 	numbalizer.nb_objects_to_enumerized(nb_objects_40)

# state_200 = {"ie" + str(i) : _state['i0'] for i in range(200)}
# nb_objects_200 = numbalizer.state_to_nb_objects(state_200)
# def b200_state_to_objs():
# 	numbalizer.state_to_nb_objects(state_200)
# def b200_enumerize_objs():
# 	numbalizer.nb_objects_to_enumerized(nb_objects_200)





# stateA = {"ie" + str(i) : _state['i0'] for i in range(40)}
# stateB = {"ie" + str(i+3) : _state['i1'] for i in range(40)}
# enumerized_A = numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects(stateA))
# enumerized_B = numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects(stateB))
# enumerized_states = List()
# enumerized_states.append(enumerized_A)
# enumerized_states.append(enumerized_B)

# def flatten_state(state):
# 	flat_state = {}
# 	for name,elm in state.items():
# 		for attr_name,attr in elm.items():
# 			flat_state[str((name,attr_name))] = attr
# 	return flat_state

# py_states = [flatten_state(stateA),flatten_state(stateB)]

# from sklearn.feature_extraction import DictVectorizer
# dv = DictVectorizer(sparse=False, sort=False)
# def b40_py_enumerized_to_vectorized():
# 	py_states = [flatten_state(stateA),flatten_state(stateB)]
# 	dv.fit_transform(py_states)

# def b40_nb_enumerized_to_vectorized():
# 	enumerized_to_vectorized(enumerized_states,numbalizer.nominal_maps,numbalizer.number_backmap)

# def b40_nb_enumerized_to_vectorized2():
# 	enumerized_to_vectorized_legacy(enumerized_states,numbalizer.nominal_maps,numbalizer.number_backmap)

# # def b40_py_enumerized_to_vectorized():
# # 	enumerized_to_vectorized(enumerized_states,Numbalizer.nominal_maps,Numbalizer.number_backmap)

# # b40_py_enumerized_to_vectorized()

# # print("-----Single Objs------")
# # print("enumerize_obj:",time_ms(enumerize_obj))
# # print("enumerize_obj_nocheck:",time_ms(enumerize_obj_nocheck))
# # print("enumerize_obj_justnums:",time_ms(enumerize_obj_justnums))
# # print("enumerize_value_string:",time_ms(enumerize_value_string))
# # print("enumerize_value_number:",time_ms(enumerize_value_number))
# # print("unenumerize_value:",time_ms(enumerize_value_number))
# # print("-----State 10 Objs------")
# # print("state_to_objs:",time_ms(b10_state_to_objs))
# # print("enumerize_objs:",time_ms(b10_enumerize_objs))
# # print("-----State 40 Objs------")
# # print("state_to_objs:",time_ms(b40_state_to_objs))
# # print("enumerize_objs:",time_ms(b40_enumerize_objs))
# # print("-----State 200 Objs------")
# # print("state_to_objs:",time_ms(b200_state_to_objs))
# # print("enumerize_objs:",time_ms(b200_enumerize_objs))


# print("-----States 2x40 Objs------")
# print("nb_enumerized_to_vectorized2:",time_ms(b40_nb_enumerized_to_vectorized2))
# print("nb_enumerized_to_vectorized:", time_ms(b40_nb_enumerized_to_vectorized))
# print("py_enumerized_to_vectorized:",time_ms(b40_py_enumerized_to_vectorized))



