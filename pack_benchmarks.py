
from numba import types, njit, guvectorize,vectorize,prange
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba.core.extending import overload
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, unicode_type, float64, Tuple, NamedTuple
from utils import cache_safe_exec
from collections import namedtuple
import numpy as np
import timeit
import types as pytypes
import sys
import __main__
from numba.cpython.unicode import _empty_string, _set_code_point, _get_code_point, PY_UNICODE_1BYTE_KIND, _kind_to_byte_width,_malloc_string
from numba.cpython.charseq import  _get_code_impl,unicode_charseq_get_code,unicode_charseq_get_value
o_str =  optional(unicode_type)
nb_str = unicode_type

# @njit
# def init_bool():
# 	np.empty((5,),np.uint8)

# init_bool()


N=1000
def time_ms(f):
	f() #warm start
	return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))

STATE_SIZE = 2


# my_data = np.array([elm1]*10)

InterfaceElement_NamedTuple = namedtuple("InterfaceElement", ["id", "value", "above", "below", "to_left", "to_right","x","y"])
nt = InterfaceElement_NamedTuple
import __main__
setattr(__main__, nt.__name__, nt)
nt.__module__ = "__main__"
# NB_InterfaceElement_NamedTuple = NamedTuple((o_str, o_str, o_str, o_str, o_str, o_str, f8, f8),InterfaceElement_NamedTuple)
NB_InterfaceElement_NamedTuple = NamedTuple((nb_str, nb_str, nb_str, nb_str, nb_str, nb_str, f8, f8),InterfaceElement_NamedTuple)
# print(mydf,mydf.dtype)


@njit
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


numpy_type_map = {
	"string" : '|S%s',
	"number" : np.float64,	
}

FIXED_STRING = '|S%s' % 50
my_data = np.array([("ie1","Mooooooooooooooooooooooooop","value","1","2","3","4",10,1)]*STATE_SIZE,
 					dtype=[('__name__',FIXED_STRING),
					('id',FIXED_STRING),
					('value',FIXED_STRING),
					('above',FIXED_STRING),
					('below',FIXED_STRING),
					('to_right',FIXED_STRING),
					('to_left',FIXED_STRING),
					('x',np.float64),
					('y',np.float64)])

# @njit
# def exp_fixed_width(x,_min,_max=10000):
# 	for i in range(len(x)):
# 		if(x[i] > _min):
# 			x[i] = min(2**np.ceil((np.log2(x[i]/_min)))*_min, _max)
# 		else:
# 			x[i] = _min

# 	return x#(2**np.clip(np.ceil((np.log2(x/_min))),0,_max)*_min)


# def exp_fixed_width(x,_min,_max=10000):
# 	return (2**np.clip(np.ceil((np.log2(x/_min))),0,_max)*_min)

ie_spec = {
		"id" : "string",
		"value" : "string",
		"above" : "string",#["String","Reference"],
		"below" : "string",#["String","Reference"],
		"to_left" : "string",#["String","Reference"],
		"to_right" : "string",# ["String","Reference"],
		"x" : "number",
		"y" : "number"
	}

_state = {
	"i01" : {
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
	"i12" : {
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

state = {"ie" + str(i) : _state['i01'] for i in range(STATE_SIZE)}




	# print(len_arr_by_spec_type)
	# dtype = [('__name__', tm['string']%exp_fixed_width(len(name),50))]
	# for k,v in elm.items():
	# 	if(k == 'type'):continue
	# 	# assert k in spec, "attribute %r not found in spec for %r"%(k,elm['type'])
	# 	if(spec[k] == 'string'):
	# 		dtype.append( (k, tm['string']%exp_fixed_width(len(v),50)) )
	# 	else:
	# 		dtype.append( (k, tm[spec[k]]) )


	# 	print(dtype)
@njit
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

print(exp_fixed_width(np.array([5,19,40,60,150,300])))
# raise ValueError()

bloopy_mc_noodler = NB_InterfaceElement_NamedTuple.instance_class
# print(bloopy_mc_noodler)
# raise ValueError()

@njit(parallel=False)
def pack_from_numpy(inp,mlens):
	out = Dict.empty(unicode_type,NB_InterfaceElement_NamedTuple)
	for i in prange(inp.shape[0]):
		x = inp[i]
		__name__ = charseq_to_str(x.__name__,mlens[0])
		_id = charseq_to_str(x.id,mlens[1])
		_value = charseq_to_str(x.value,mlens[2])
		_above = charseq_to_str(x.above,mlens[3])
		_below = charseq_to_str(x.below,mlens[4])
		_to_right = charseq_to_str(x.to_right,mlens[5])
		_to_left = charseq_to_str(x.to_left,mlens[6])
		_x = float(x.x)
		_y = float(x.y)
		out[__name__] = bloopy_mc_noodler(_id,_value,_above,_below,_to_right,_to_left,_x,_y)
	return out

def pack_to_nb_via_numpy(state,spec):
	tm = numpy_type_map
	
	len_arr_by_spec_type = {}
	for name, elm in state.items():
		assert 'type' in elm, "All state elements must have a type, none found for %s" % name
		len_arrs = len_arr_by_spec_type.get(elm['type'],[])
		len_arr = [len(name)] + [len(v) if spec[k] == 'string' else None for k,v in elm.items() if k != "type"]
		len_arrs.append(len_arr)
		len_arr_by_spec_type[elm['type']] = len_arrs
	
	dtype_by_spec_type = {}
	mlens_by_spec_type = {}
	for spec_type,len_arrs in len_arr_by_spec_type.items():
		mlens = exp_fixed_width(np.max(np.array(len_arrs,dtype=np.float64), axis=0), _min=50)
		mlens_by_spec_type[spec_type] = mlens
		# spec = ?
		dtype = dtype = [('__name__', tm['string']%int(mlens[0]))]
		for i,(attr, typ) in enumerate(spec.items()):
			if(spec[attr] == 'string'):
				dtype.append( (attr, tm['string']%int(mlens[i+1])) )
			else:
				dtype.append( (attr, tm[spec[attr]]) )
		dtype_by_spec_type[spec_type] = dtype

	data_by_type = {}
	for name, elm in state.items():
		typ = elm['type']
		elm_data = tuple([name] + [v for k,v in elm.items() if k != 'type'])
		data = data_by_type.get(typ,[])
		data.append(elm_data)
		data_by_type[typ] = data

	out = {}
	for typ,data in data_by_type.items():
		out[typ] = pack_from_numpy(np.array(data,dtype=dtype_by_spec_type[typ]),mlens_by_spec_type[typ])
	return out

#https://www.geeksforgeeks.org/smallest-power-of-2-greater-than-or-equal-to-n/


# print(exp_fixed_width(np.array([51,4,186,201])))
# raise ValueError()
def pack_to_nb_via_numpy2(state,spec):
	tm = numpy_type_map

	#Throw the attribute values into tuples and find the lengths of
	#	all of the strings.
	data_by_type = {}
	mlens_by_type = {}
	for name, elm in state.items():
		# spec = ?
		typ = elm['type']
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
		mlens = exp_fixed_width(np.array(mlens_by_type[spec_typ]),20) 
		dtype = dtype = [('__name__', tm['string']%int(mlens[0]))]
		for i,(attr, typ) in enumerate(spec.items()):
			if(spec[attr] == 'string'):
				dtype.append( (attr, tm['string']%int(mlens[i+1])) )
			else:
				dtype.append( (attr, tm[spec[attr]]) )
		# print(np.array(data_by_type[spec_typ],dtype=dtype))
		# print(dtype)
		out[typ] = pack_from_numpy(np.array(data_by_type[spec_typ],dtype=dtype),mlens)

	return out	


def pack_to_nb_via_args(state):
	out = Dict.empty(unicode_type,NB_InterfaceElement_NamedTuple)
	for _name, elm in state.items():
		pack_from_args(_name,*[v for k,v in elm.items() if k != 'type'],out=out)	

@njit()
def pack_from_args(__name__,_id,_value,_above,_below,_to_right,_to_left,_x,_y, out,max_str=100):
	# print(__name__,_id,_value,_above,_below,_to_right,_to_left,_x,_y)
	ie = InterfaceElement_NamedTuple(_id,_value,_above,_below,_to_right,_to_left,float(_x),float(_y))
	out[__name__] = ie

def pack_to_nb_via_append(state):
	out = Dict.empty(unicode_type,NB_InterfaceElement_NamedTuple)
	for _name, elm in state.items():
		out[_name] = InterfaceElement_NamedTuple(elm['id'],elm['value'],
						elm['above'],elm['below'],elm['to_right'],elm['to_left'],
						float(elm['x']),float(elm['y']))


# def test_pack_from_args():


print(pack_from_numpy(my_data,np.array([50]*9)))
print("EYA",pack_to_nb_via_numpy(state,ie_spec))
print("EYA2",pack_to_nb_via_numpy2(state,ie_spec))



def b_pack_from_numpy():
	pack_from_numpy(my_data,np.array([50]*9))


state_10 = {"ie" + str(i) : _state['i01'] for i in range(10)}
def b10_pack_to_nb_via_numpy():
	pack_to_nb_via_numpy(state_10,ie_spec)
def b10_pack_to_nb_via_numpy2():
	pack_to_nb_via_numpy2(state_10,ie_spec)
def b10_pack_to_nb_via_args():
	pack_to_nb_via_args(state_10)

def b10_pack_to_nb_via_append():
	pack_to_nb_via_append(state_10)


state_40 = {"ie" + str(i) : _state['i01'] for i in range(40)}
def b40_pack_to_nb_via_numpy():
	pack_to_nb_via_numpy(state_40,ie_spec)
def b40_pack_to_nb_via_numpy2():
	pack_to_nb_via_numpy2(state_40,ie_spec)

def b40_pack_to_nb_via_args():
	pack_to_nb_via_args(state_40)

def b40_pack_to_nb_via_append():
	pack_to_nb_via_append(state_40)

state_200 = {"ie" + str(i) : _state['i01'] for i in range(200)}
def b200_pack_to_nb_via_numpy():
	pack_to_nb_via_numpy(state_200,ie_spec)
def b200_pack_to_nb_via_numpy2():
	pack_to_nb_via_numpy2(state_200,ie_spec)

def b200_pack_to_nb_via_args():
	pack_to_nb_via_args(state_200)

def b200_pack_to_nb_via_append():
	pack_to_nb_via_append(state_200)


print("pack_from_numpy",time_ms(b_pack_from_numpy))

print("10 items:")
print("pack_to_nb_via_numpy",time_ms(b10_pack_to_nb_via_numpy))
print("pack_to_nb_via_numpy2",time_ms(b10_pack_to_nb_via_numpy2))
print("pack_to_nb_via_args",time_ms(b10_pack_to_nb_via_args))
print("pack_to_nb_via_append",time_ms(b10_pack_to_nb_via_append))

print("40 items:")
print("pack_to_nb_via_numpy",time_ms(b40_pack_to_nb_via_numpy))
print("pack_to_nb_via_numpy2",time_ms(b40_pack_to_nb_via_numpy2))
print("pack_to_nb_via_args",time_ms(b40_pack_to_nb_via_args))
print("pack_to_nb_via_append",time_ms(b40_pack_to_nb_via_append))

print("200 items:")
print("pack_to_nb_via_numpy",time_ms(b200_pack_to_nb_via_numpy))
print("pack_to_nb_via_numpy2",time_ms(b200_pack_to_nb_via_numpy2))
print("pack_to_nb_via_args",time_ms(b200_pack_to_nb_via_args))
print("pack_to_nb_via_append",time_ms(b200_pack_to_nb_via_append))

print("Note to self: At the time of writing this for large states packing via_numpy2 " +
		"seems to be the fastest option, it is decently faster than packing by expanding " +
		"the arguments into an njitted function (a perhaps simpler approach). Performance changes " +
		"in later versions of numba (currently 0.50.0) might prompt a different decision." )

