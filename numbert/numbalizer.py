from numba import types, njit, guvectorize,vectorize,prange
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple 
from numba.cpython.unicode import  _set_code_point
from numbert.utils import cache_safe_exec
from numbert.core import TYPE_ALIASES, REGISTERED_TYPES, py_type_map, numba_type_map, numpy_type_map
from numbert.gensource import gen_source_standard_imports, gen_source_get_enumerized, \
							  gen_source_enumerize_nb_objs, \
							  gen_source_tuple_defs, gen_source_pack_from_numpy
from numbert.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache
from collections import namedtuple
import numpy as np
import timeit
import types as pytypes
import sys
import __main__


# numba_type_ids = {k:i  for i,k in enumerate(numba_type_map)}


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

# @njit(cache=True)
# def charseq_len(s,max_l=100):
# 	i = 0
# 	for i in range(max_l):
# 		try:
# 			v = s[i]
# 		except Exception:
# 			break
# 	return i

# NULL = chr(0)

# @njit
# def charseq_to_str(x,max_l=100):
# 	return str(x)
	# l = len(x)
	# if(l == 0):
	# 	return ""
	# else:
	# 	s = NULL*(l+1)
	# 	for i in range(l):
	# 		_set_code_point(s,i,x[i])
	# 	return s[:l]




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
	elif(hasattr(value,"_fields") and hasattr(type(value),"__name__")):
		#Is a namedtuple
		return type(value).__name__
	else:
		raise ValueError("Could not infer type of %s" % value)

def infer_nb_type(value):
	return numba_type_map[infer_type(value)]



Dict_Unicode_to_Enums = DictType(unicode_type,i8[:])

class Numbalizer(object):
	#Static Values
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
		
		#For the purposes of autogenerating code we need a clean alphanumeric name 
		name = "".join(x for x in name if x.isalnum())

		hash_code = unique_hash([name,spec])
		if(not source_in_cache(name,hash_code)):
			source = gen_source_standard_imports()
			source += gen_source_tuple_defs(name,spec)
			source += gen_source_get_enumerized(name,spec)
			source += gen_source_enumerize_nb_objs(name,spec)
			source += gen_source_pack_from_numpy(name,spec)
			source_to_cache(name,hash_code,source)
		# else:
		# 	source = source_from_cache(name,hash_code)
		# pack_from_numpy =	l['{}_pack_from_numpy'.format(name)]}
		out = import_from_cached(name,hash_code,[
			'{}_get_enumerized'.format(name),
			'{}_pack_from_numpy'.format(name),
			name,
			'NB_{}_NamedTuple'.format(name),
			'{}_enumerize_nb_objs'.format(name)
			]).values()

		get_enumerized, pack_from_numpy, nt, nb_nt, enumerize_nb_objs = tuple(out)
		# tuple_def_source = gen_source_tuple_defs(name,spec,ind)
		# print(tuple_def_source)
		# nb_nt_name = 'NB_{}_NamedTuple'.format(name)
		# l,g = cache_safe_exec(tuple_def_source,gbls=globals())
		# nt = 				l[name]
		# nb_nt = 			l[nb_nt_name]

		# #Do this to make the namedtuple picklable (see https://stackoverflow.com/questions/16377215/how-to-pickle-a-namedtuple-instance-correctly)
		# print("NAME",nt.__name__)
		# setattr(__main__, nt.__name__, nt)
		# nt.__module__ = "__main__"

		# #Dynamically generate njit functions, pass in the namedtuple and wrapper as globals
		# source = gen_source_get_enumerized(name,spec,ind=ind)
		# source += gen_source_pack_from_numpy(name,spec,ind=ind)
		
		# print(source)
		# l,g = cache_safe_exec(source,gbls={**globals(), **{name:nt,nb_nt_name:nb_nt} })
		# get_enumerized = 	l['{}_get_enumerized'.format(name)]
		# pack_from_numpy =	l['{}_pack_from_numpy'.format(name)]
		

		# source = gen_source_enumerize_nb_objs(name,spec,ind=ind)
		# l,g = cache_safe_exec(source,gbls={**globals(), **{'{}_get_enumerized'.format(name):get_enumerized} })
		# enumerize_nb_objs =	l['{}_enumerize_nb_objs'.format(name)]

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
		nt.numba_type = nb_nt

		return nt

	def register_specification(self, name, spec):
		spec = {k:v.lower() for k,v in spec.items()}
		if(name in self.registered_specs):
			assert self.registered_specs[name] == spec, \
			"Specification redefinition not permitted. Attempted on %r" % name
		else:
			self.registered_specs[name] = spec
			# print("start jit")
			jitstruct = self.jitstruct_from_spec(name,spec)
			self.jitstructs[name] = jitstruct
			self.nominal_maps[name] = np.array([x == "string" for k,x in spec.items() if k != 'type'],dtype=np.uint8)
			# print("end jit")

			REGISTERED_TYPES[name] = jitstruct.numba_type
			TYPE_ALIASES[name] = jitstruct.__name__

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
			# print(elm)
			assert 'type' in elm, "All objects need 'type' attribute to be numbalized."
			typ = elm['type']
			spec = self.registered_specs[typ]

			values = [elm[k] for k in spec.keys() if k != 'type']
			assert len(values) == len(spec), "Dict with keys [{}], cannot be cast to {} [{}]".format(
				",".join(elm.keys()),name,",",join(spec.keys())) 

			elm_data = tuple([name] + values)

			data = data_by_type.get(typ,[])
			data.append(elm_data)
			data_by_type[typ] = data

			if(typ not in mlens_by_type):
				mlens_by_type[typ] = [0]*len(elm_data)
			mlens = mlens_by_type[typ]

			for i,(attr, typ) in enumerate(spec.items()):
				cast_fn = py_type_map[typ]
				if(typ == 'string'):
					L = len(cast_fn(elm[attr]))
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
				args.append(py_type_map[t](obj_d[x]))
			except ValueError as e:
				raise ValueError("Cannot cast %r to %r in %r of %r" % (obj_d[x],py_type_map[t],x,name)) from e

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
			# print(value,typ)
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



