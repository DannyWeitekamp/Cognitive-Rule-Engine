from numba import types, njit, guvectorize,vectorize,prange
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, unicode_type, float64
from utils import cache_safe_exec
import timeit

N=1000
def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))

cast_map = {
	"String" : str,
	"Number" : float,
}

numba_type_map = {
	"String" : unicode_type,
	"Number" : float64,	
}


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
	}
}


state = {
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


@jitclass([('id', unicode_type),
		   ('value', unicode_type),
		   ('above', unicode_type),
		   ('below', unicode_type),
		   ('to_left', unicode_type),
		   ('to_right', unicode_type),
		   ('x', f8),
		   ('y', f8),
		   ])
class InterfaceElement(object):
	def __init__(self, _id, _value, _above, _below, _to_left, _to_right, _x, _y):
		self.id = _id
		self.value = _value
		self.above = _above
		self.to_left = _to_left
		self.to_right = _to_right
		self.x = _x
		self.y = _y


def jitclass_from_spec(name,spec,ind="   "):
	header = "@jitclass([{}])\n"
	header += "class {}(object):\n".format(name)
	header += ind + "def __init__(self, {}):\n"
	args = []
	body = ""
	sig_items = ""
	for i,(k,v) in enumerate(spec.items()):
		sig_items += (i!=0)*(3*ind) + "('%s', %s),\n"%(k,numba_type_map[v])
		args.append("_%s" % k)
		body += ind*2 + "self.{0} = _{0}\n".format(k)

	args = ", ".join(args)
	header = header.format(sig_items,args)
	source = header + body
	print(source)
	l,g = cache_safe_exec(source,gbls=globals())
	return l[name]


jitclasses = {
	"InterfaceElement" : InterfaceElement,
}


def stateToObjs(state):
	for name, obj_d in state.items():
		assert 'type' in obj_d, "Object %s missing required attribute 'type'" % name
		assert obj_d['type'] in object_specifications, \
				 "Object specification not defined for %s" % obj_d['type']
		spec = object_specifications[obj_d['type']]
		o_class = jitclasses[obj_d['type']]
		args = []
		for x,t in spec.items():
			try:
				args.append(cast_map[t](obj_d[x]))
			except ValueError as e:
				raise ValueError("Cannot cast %r to %r in %r of %r" % (obj_d[x],cast_map[t],x,name)) from e

		obj = o_class(*args)
		# print(obj, obj.value)

def go():
	stateToObjs(state)

print("stateToObjs",time_ms(go))
print(jitclass_from_spec("InterfaceElement", object_specifications["InterfaceElement"]))


