from numbert.data_trans import Numbalizer
from sklearn.feature_extraction import DictVectorizer
from numba.typed import List, Dict
from numba import njit
import numpy as np
import timeit

N=1000
def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))


@njit(cache=True)
def enumerized_to_vectorized_legacy(enumerized_states,nominal_maps,number_backmap):
	'''
	enumerized_states : List<Dict<i8[:]>>
	'''
	elm_present = Dict()
	nominals = Dict()
	continuous = Dict()
	n_states = len(enumerized_states)
	for k,state in enumerate(enumerized_states):
		for typ,elms in state.items():
			nominal_map = nominal_maps[typ]
			for name,elm in elms.items():
				if(name not in elm_present):
					elm_present[name] = np.zeros((n_states,),dtype=np.uint8)
				elm_present[name][k] = True

				for i,attr in enumerate(elm):
					if(nominal_map[i]):
						tn = (name,i,attr)
						if(tn not in nominals):
							n_arr = np.zeros((n_states,),dtype=np.uint8)
							# n_arr.fill(255)
							nominals[tn] = n_arr
						nominals[tn][k] = True
					else:
						tc = (name,i)
						if(tc not in continuous):
							c_arr = np.empty((n_states,),dtype=np.float64)
							c_arr.fill(np.nan)
							continuous[tc] = c_arr
						continuous[tc][k] = number_backmap[attr]

	# Apparently filling it transposed and then transposing gives a fortran ordered array
	vect_nominals = np.empty((n_states,len(nominals)), dtype=np.uint8)#, order='F')
	vect_continuous = np.empty((n_states,len(continuous)), dtype=np.float64)#, order='F')

	for i,(tup,n_arr) in enumerate(nominals.items()):
		name,_,_ = tup
		# print(elm_present[name], np.where(elm_present[name], n_arr, 255) )
		vect_nominals[:,i] = np.where(elm_present[name], n_arr, 255) 

	for i,c_arr in enumerate(continuous.values()):
		vect_continuous[:,i] = c_arr

	return vect_nominals,vect_continuous

numbalizer = Numbalizer()


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

numbalizer.register_specifications(object_specifications)

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


nb_objects = numbalizer.state_to_nb_objects(state)
nb_objects2 = numbalizer.state_to_nb_objects(state2)
obj1 = nb_objects['InterfaceElement']['ie0']
obj2 = nb_objects2['Trajectory']['ie0']
	

nb_objects_real = numbalizer.state_to_nb_objects(state)
nb_objects_real2 = numbalizer.state_to_nb_objects(state2)


enumerized = numbalizer.nb_objects_to_enumerized(nb_objects_real)
enumerized_states = List()
enumerized_states.append(enumerized)
enumerized_states.append(numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects({"ie" + str(i+3) : _state['i1'] for i in range(STATE_SIZE)})))


print("nominal maps")
print(Numbalizer.nominal_maps)
print("number backmaps")
print(Numbalizer.number_backmap)


nominals, continuous = enumerized_to_vectorized_legacy(enumerized_states,Numbalizer.nominal_maps,Numbalizer.number_backmap)
nc = numbalizer.enumerized_to_vectorized(enumerized_states)


np.set_printoptions(threshold=2000)
print("nominal")
print(nc['nominal'])
print("continuous")
print(nc['continuous'])
# raise ValueError()
# print(np.isfortran(nominals))
# print(np.isfortran(continuous))

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
	numbalizer.enumerize_value("159")

def enumerize_value_number():
	numbalizer.enumerize_value(159)

def unenumerize_value():
	numbalizer.unenumerize_value(2)


state_10 = {"ie" + str(i) : _state['i0'] for i in range(10)}
nb_objects_10 = numbalizer.state_to_nb_objects(state_10)
def b10_state_to_objs():
	numbalizer.state_to_nb_objects(state_10)
def b10_enumerize_objs():
	numbalizer.nb_objects_to_enumerized(nb_objects_10)


state_40 = {"ie" + str(i) : _state['i0'] for i in range(40)}
nb_objects_40 = numbalizer.state_to_nb_objects(state_40)
def b40_state_to_objs():
	numbalizer.state_to_nb_objects(state_40)
def b40_enumerize_objs():
	numbalizer.nb_objects_to_enumerized(nb_objects_40)

state_200 = {"ie" + str(i) : _state['i0'] for i in range(200)}
nb_objects_200 = numbalizer.state_to_nb_objects(state_200)
def b200_state_to_objs():
	numbalizer.state_to_nb_objects(state_200)
def b200_enumerize_objs():
	numbalizer.nb_objects_to_enumerized(nb_objects_200)





stateA = {"ie" + str(i) : _state['i0'] for i in range(40)}
stateB = {"ie" + str(i+3) : _state['i1'] for i in range(40)}
enumerized_A = numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects(stateA))
enumerized_B = numbalizer.nb_objects_to_enumerized(numbalizer.state_to_nb_objects(stateB))
enumerized_states = List()
enumerized_states.append(enumerized_A)
enumerized_states.append(enumerized_B)

def flatten_state(state):
	flat_state = {}
	for name,elm in state.items():
		for attr_name,attr in elm.items():
			flat_state[str((name,attr_name))] = attr
	return flat_state

py_states = [flatten_state(stateA),flatten_state(stateB)]


dv = DictVectorizer(sparse=False, sort=False)
def b40_py_enumerized_to_vectorized():
	py_states = [flatten_state(stateA),flatten_state(stateB)]
	dv.fit_transform(py_states)

def b40_nb_enumerized_to_vectorized():
	numbalizer.enumerized_to_vectorized(enumerized_states)

def b40_nb_enumerized_to_vectorized_legacy():
	enumerized_to_vectorized_legacy(enumerized_states,numbalizer.nominal_maps,numbalizer.number_backmap)

b40_py_enumerized_to_vectorized()

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


print("-----States 2x40 Objs------")
print("nb_enumerized_to_vectorized_legacy:",time_ms(b40_nb_enumerized_to_vectorized_legacy))
print("nb_enumerized_to_vectorized:", time_ms(b40_nb_enumerized_to_vectorized))
print("py_enumerized_to_vectorized:",time_ms(b40_py_enumerized_to_vectorized))
