#From here: https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
from numba import types, njit, jit, prange
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, DictType, unicode_type, Array, Tuple
from numba.experimental import jitclass
import numpy as np
import re
from collections.abc import Iterable 
import timeit
from pprint import pprint
N = 10

def parse_signature(s):
	fn_match = re.match(r"(?P<out_type>\w+)\s?\((?P<arg_types>(?P<args>\w+(,\s?)?)+)\)", s)
	fn_dict = fn_match.groupdict()
	arg_types = [arg.strip() for arg in fn_dict['arg_types'].split(',')]
	return fn_dict['out_type'], arg_types

# def norm_check_types(s):





		# [time][op]

class BaseOperator(object):
	registered_operators = {}
	operators_by_uid = []
	# def __init__(self):
	# 	self.num_flt_inputs = 0;
	# 	self.num_str_inputs = 0;
	# 	self.out_arg_types = ["value"]
	# 	self.in_arg_types= ["value","value"]
	# 	self.commutative = False;
	# 	self.template = "BaseOperator"

	@classmethod
	def init_signature(cls):
		out_type, arg_types = parse_signature(cls.signature)
		out_type = TYPE_ALIASES.get(out_type,out_type)
		arg_types = [TYPE_ALIASES.get(x,x) for x in arg_types]
		print(arg_types)
		cls.out_type = out_type
		cls.arg_types = arg_types

		u_types,u_inds = np.unique(arg_types,return_inverse=True)
		cls.u_arg_types = u_types
		cls.u_arg_inds = u_inds

		if(isinstance(cls.commutes,bool)):
			cls.commutes = [np.where(i == u_inds)[0].tolist() for i in range(len(u_types))]
		else:
			assert(isinstance(cls.commutes,Iterable))

		right_commutes = {}
		for i in range(len(cls.commutes)):
			commuting_set =  cls.commutes[i]
			print(commuting_set)
			for j in range(len(commuting_set)-1,0,-1):
				right_commutes[commuting_set[j]] = commuting_set[0:j]
				for k in commuting_set[0:j]:
					assert u_inds[k] == u_inds[commuting_set[j]], \
					 "invalid 'commutes' argument, argument %s and %s have different types \
					  %s and %s" % (j, k, u_types[u_inds[j]], u_types[u_inds[k]])
		cls.right_commutes = right_commutes
		print(cls.right_commutes)
		print(cls.arg_types)


	@classmethod
	def register(cls):
		name = cls.__name__.lower()
		
		if(name in cls.registered_operators):
			raise Warning("Duplicate Operator Definition %s" % name)

		uid = len(cls.operators_by_uid)
		cls.uid = uid
		cls.operators_by_uid.append(cls)
		cls.registered_operators[name] = cls

	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		cls.init_signature()
		cls.register()
	


	# def forward(self, args):
	# 	raise NotImplementedError("Not Implemeneted")

	# def backward(self, args):
	# 	raise NotImplementedError("Not Implemeneted")
	# def search_mask(self,*args):
	# 	return args
	def __str__(self):
		# print(self.template)
		# print(self.in_arg_types)
		# print(tuple(["E" + str(i) for i in range(len(self.in_arg_types))]))
		return self.template.format(*["E" + str(i) for i in range(len(self.in_arg_types))])


TYPE_ALIASES = {
	"float" : 'f4',
	"string" : 'unicode_type'
}

# ALLOWED_TYPES = []

def compile_forward(op):
	_ = "    "
	
	exec_code = '@njit(nogil=True,fastmath=True) \n' + \
				'def {}({}): \n' #+ \
				# _ + 'L1,L2 = len(x1),len(x2)\n'+ \
				# _ + 'for i1 in range(len(x1)):\n'+ \
				# _*2 +	'for i2 in range(len(x1)):\n'+ \
				# _*3 +		'out[i2 * L1 + i1] = f(x1,x2)'

	loops = _+", ".join(["L%i"%i for i in range(len(op.u_arg_types))]) + " = " + \
			  ", ".join(["len(x%i)"%i for i in range(len(op.u_arg_types))]) + "\n"
	# ", ".join(["len(x%i)"%(np.where(i==op.u_arg_inds)[0][0]) 
	# 		  				for i in range(len(op.u_arg_types))]) + "\n"
	for i in range(len(op.arg_types)):
		l = _*(i+1) + "for i{} in range({},L{}):\n"

		start = 0
		if(len(op.right_commutes.get(i,[])) > 0):
			start = "i{}+1".format(op.right_commutes[i][-1])

		l = l.format(i,start,op.u_arg_inds[i])
		loops += l


	exec_code = exec_code.format(op.__name__+"_forward",
		 ",".join(["x%i"%i for i in range(len(op.u_arg_types))]) )
	# for 
	# for 

	print(exec_code + loops)

				# '	return {}({}) \n' + \
				# 'out_func = {}'

# def normalize_types():


class Add(BaseOperator):
	commutes = True
	signature = 'float(float,float)'
	def forward(x, y):
		return x + y

class Subtract(BaseOperator):
	commutes = False
	signature = 'float(float,float)'
	def forward(x, y):
		return x - y

class Concatenate(BaseOperator):
	commutes = True
	signature = 'string(string,string)'
	def forward(x, y):
		return x + y

# compile_forward(Add)

# def Multiply(x, y):
# 	return x * y

def forward(state, goal, operators):
	for op in operators:
		for i in range(len(state)):
			for j in range(len(state)):
				pass




# @njit(nogil=True,fastmath=True,parallel=True) 
# def Add_forward1(x0,x1): 

#     L0, L1 = len(x0), len(x1)
#     Total_Len = (L0)*(L0-1)*(L0-2)/(1*2*3) * \
#     			 L1 
#     Total_Len = int(Total_Len)    			 
#     # print(Total_Len)
#     # Total_Len = L0*L0*L0*L1
#     # out = np.empty((L0,L0,L1,L0))
#     out = np.empty((Total_Len,))
#     ind = 0
#     for i0 in prange(0,L0):
#         for i1 in prange(i0+1,L0):
#             for i2 in prange(0,L1):
#                 for i3 in prange(i1+1,L0):
#                 	# ind = i0*L0*L0*L1 + i1*L0*L1 + i2*L0 + i3
#                 	# out[i0,i1,i2,i3] = x0[i0] + x0[i1] + x0[i3] * x1[i2]
#                 	out[ind] = x0[i0] + x0[i1] + x0[i3] * x1[i2]
#                 	ind += 1
#     return out

bloop_type = ListType(u8[:])
sloop_type = u8[:]
@njit(nogil=True,fastmath=True,parallel=True,cache=True) 
def Grumbo_forward1(x0,x1): 
	L0, L1 = len(x0), len(x1)

	out = np.empty((L0,L0,L1,L0))
	da =[]
	for i0 in range(0,L0):
		da.append(Dict.empty(f8,i8))

	for i0 in prange(0,L0):
		d = da[i0]
		# ind = 0
		for i1 in range(i0+1,L0):
			for i2 in range(i1+1,L0):
				for i3 in range(0,L1):
					v = x0[i0] + x0[i1] + x0[i2] * x1[i3]
					if(v not in d):
						d[v] = 1

					out[i0,i1,i2,i3] = v
					# beezl = np.array([u4(i0),u4(i1),u4(i2),u4(i3)])

	d_out = Dict.empty(f8,i8)
	for i0 in range(0,L0):
		for v in da[i0]:
			if(v not in d_out):
				d_out[v] = 1

	u_vs = np.empty(len(d_out))
	for i,v in enumerate(d_out):
		u_vs[i] = v

	# sqeep = np.where(out == 1)
	# out[sqeep[0]] = 7
	return u_vs


# HE_deffered = deferred_type()
# @jitclass([('op_id', i8),
#            ('args',  i8[:]),
#            ('next', optional(HE_deffered))])
# class HistElm(object):
#     def __init__(self,op_id,args):
#         self.op_id = op_id
#         self.args = args
#         self.next = None
# # print(BinElem)
# HE = HistElm.class_type.instance_type
# HE_deffered.define(HE)


bloop_type = ListType(u8[:])
sloop_type = u8[:]
@njit(nogil=True,fastmath=True,parallel=False,cache=True) 
def Grumbo_forward2(x0,x1): 
	L0, L1 = len(x0), len(x1)

	out = np.empty((L0,L0,L1,L0))
	# da =[]
	# for i0 in range(0,L0):
	# 	da.append(Dict.empty(f8,i8))
	d = Dict.empty(f8,i8)
	uid = 0
	for i0 in range(0,L0):
		# d = da[i0]
		# uid = 0
		# d = Dict.empty(f8,i8)
		# for i1 in range(i0+1,L0):
		# 	for i2 in range(i1+1,L0):
		for i1 in range(0,L0):
			for i2 in range(0,L0):
				for i3 in range(0,L1):
					if(i1 > i0 and i2 > i1):
						v = x0[i0] + x0[i1] + x0[i2] * x1[i3]
						# v = x0[i0] + x0[i1]# + x0[i2] + x0[i3] 
						if(v not in d):
							d[v] =uid; uid +=1; 
						out[i0,i1,i2,i3] = d[v]
						# HistElm(0,np.array([i0,i1,i2,i3]))
					else:
						out[i0,i1,i2,i3] = 0
					
	# d_out = d
	# u_vs = np.empty(len(d_out))
	# for i,v in enumerate(d_out):
	# 	u_vs[i] = v

	# sqeep = np.where(out == 1)
	return out, d


@njit(nogil=True,fastmath=True,parallel=False,cache=True) 
def Add_forward(x0): 
	L0 = len(x0)
	out = np.empty((L0,L0),dtype=np.int64)
	d = Dict.empty(f4,i8)
	uid = 1
	for i0 in range(0,L0):
		for i1 in range(0,L0):
			if(i1 > i0):
				v = x0[i0] + x0[i1]
				if(v not in d):
					d[v] = uid; uid +=1; 
				out[i0,i1] = d[v]
			else:
				out[i0,i1] = 0
	return out, d


@njit(nogil=True,fastmath=True,parallel=False,cache=True) 
def Subtract_forward(x0): 
	L0 = len(x0)
	out = np.empty((L0,L0),dtype=np.int64)
	d = Dict.empty(f4,i8)
	uid = 1
	for i0 in range(0,L0):
		for i1 in range(0,L0):
			if(i1 != i0):
				v = x0[i0] - x0[i1]
				if(v not in d):
					d[v] = uid; uid +=1; 
				out[i0,i1] = d[v]
			else:
				out[i0,i1] = 0
	return out, d



@njit(nogil=True,fastmath=True,parallel=False,cache=True) 
def cat_forward(x0): 
	L0= len(x0)
	out = np.empty((L0,L0),dtype=np.int64)
	d = Dict.empty(unicode_type,i8)
	uid = 1
	for i0 in range(0,L0):
		# d = da[i0]
		for i1 in range(0,L0):
			# for i2 in range(i1+1,L0):
			# 	for i3 in range(0,L0):
			# if(i1 != i0):
				v = x0[i0] + x0[i1]# + x0[i2] + x0[i3] 
				if(v not in d):
					d[v] = uid; uid +=1; 
				out[i0,i1] = d[v]
			# else:
			# 	out[i0,i1] = 0
				# d[v] = 1
			# print(v)

			# out[i0,i1,i2,i3] = v

	# d_out = Dict.empty(unicode_type,i8)
	# for i0 in range(0,L0):
	# 	for v in da[i0]:
	# 		if(v not in d_out):
	# 			d_out[v] = 1
	# print(d_out)

	# u_vs = np.empty(len(d_out))
	# for i,v in enumerate(d_out):
	# 	u_vs[i] = v
	return out, d#, uid

@njit(nogil=True,fastmath=True,cache=True) 
def join_new_vals(vd,new_ds,depth):
	for d in new_ds:
		for v in d:
			if(v not in vd):
				vd[v] = depth
	return vd

@njit(nogil=True,fastmath=True,cache=True) 
def array_from_dict(d):
	out = np.empty(len(d))
	for i,v in enumerate(d):
		out[i] = v
	return out

@njit(nogil=True,fastmath=True,cache=True) 
def list_from_dict(d):
	out = List()
	for i,v in enumerate(d):
		out.append(v)
	return out
	
class NBRT_KnowledgeBase(object):
	def __init__(self):
		self.hists = {}
		self.hist_structs = {}
		self.curr_infer_depth = 0
		# self.vmaps = {}
		self.u_vds = {}
		self.u_vs = {}
		self.registered_types ={'f4': f4, 'unicode_type' : unicode_type}


# @njit(nogil=True,fastmath=True,parallel=False) 

def insert_record(kb,depth,op, btsr, vmap):
	typ = op.out_type
	if(typ not in kb.hist_structs):
		typ_cls = kb.registered_types[typ]
		kb.hist_structs[typ] = Tuple([i8,
									 i8[::1], i8[::1], ListType(unicode_type),
									 DictType(typ_cls,i8)])

	typ_store = kb.hists[typ] = kb.hists.get(typ,Dict.empty(i8,ListType(kb.hist_structs[typ])))
	tsd = typ_store[depth] = typ_store.get(depth, List.empty_list(kb.hist_structs[typ]))
	tsd.append(tuple([op.uid,
					  btsr.reshape(-1), np.array(btsr.shape,np.int64), List(op.arg_types),
					  vmap]))

	return tsd


Add.broadcast_forward = Add_forward
Subtract.broadcast_forward = Subtract_forward
Concatenate.broadcast_forward = cat_forward
def forward(kb,ops):
	output_typs = set([op.out_type for op in ops])
	new_records = {typ:[] for typ in output_typs}
	depth = kb.curr_infer_depth = kb.curr_infer_depth+1
	
	for op in ops:
		typ = op.out_type

		btsr, vmap = op.broadcast_forward(kb.u_vs[typ])
		records = insert_record(kb,depth,op,btsr,vmap)
		new_records[typ] = records
		
	for typ in output_typs:
		if(typ in new_records):
			vmaps = List([rec[4] for rec in new_records[typ]])
			kb.u_vds[typ] = join_new_vals(kb.u_vds[typ],vmaps,depth)

			if(typ == TYPE_ALIASES['float']):
				kb.u_vs[typ] = array_from_dict(kb.u_vds[typ])
			else:
				kb.u_vs[typ] = list_from_dict(kb.u_vds[typ])


HE_deffered = deferred_type()
@jitclass([('op_uid', i8),
		   ('args', i8[:])])
class HistElm(object):
	def __init__(self,op_uid,args):
		self.op_uid = op_uid
		self.args = args
HE = HistElm.class_type.instance_type
HE_deffered.define(HE)

def new_HE_list(n):
	out = List.empty_list(ListType(HE))
	for i in range(n):
		out.append(List.empty_list(HE))
	return out
def retrace_solutions(kb,ops,goal,g_typ):

	# depth = kb.u_vds[g_typ][goal]
	u_vds = kb.u_vds[g_typ]
	records = kb.hists[g_typ]
	goal = kb.registered_types[g_typ](goal)
	
	goals = List.empty_list(kb.registered_types[g_typ])
	goals.append(goal)

	# hist_elems, , new_inds = retrace_one(goals,records)
	# print(":0")
	hist_elems = List.empty_list(ListType(HE))#new_HE_list(1)#List([List.empty_list(HE)],listtype=ListType(HE))
	arg_inds = retrace_back_one(goals,records,u_vds,hist_elems)
	out = [{g_typ: hist_elems}]
	
	# goals = kb.u_vs[g_typ][arg_inds[g_typ]]
	finished, i = False, 1
	while(not finished):
		nxt = {}
		for typ in arg_inds:
			records,u_vds = kb.hists[typ], kb.u_vds[typ]
			hist_elems = List.empty_list(ListType(HE))#new_HE_list(len(goals))#List([List.empty_list(HE) for i in range(len(goals))])
			
			goals = kb.u_vs[typ][arg_inds[typ]]
			arg_inds = retrace_back_one(goals,records,u_vds,hist_elems)
			nxt[typ] = hist_elems
			if(len(arg_inds) == 0):
				finished = True
				# print("FINISHED")
				break
		out.append(nxt)
		assert i <= kb.curr_infer_depth, "Retrace has persisted past current infer depth."
		i += 1

			
	# for i in range(len(out)):
	# 	print(i)
	# 	pprint(out[i])

		



	# hist_elems = List.empty_list(ListType(HE))#new_HE_list(1)#List([List.empty_list(HE)],listtype=ListType(HE))
	# arg_inds = retrace_back_one(goals,records,u_vds,hist_elems)
	# print("1:",arg_inds)
	# for typ in arg_inds:
	# 	goals = kb.u_vs[typ][arg_inds[typ]]
	# 	records = kb.hists[typ]
	# 	hist_elems = List.empty_list(ListType(HE))#new_HE_list(len(goals))#List([List.empty_list(HE) for i in range(len(goals))])
	# 	u_vds = kb.u_vds[typ]
	# 	arg_inds = retrace_back_one(goals,records,u_vds,hist_elems)


	# 	print("2:",arg_inds)
		# arg_inds = retrace_back_one(u_vs,records,hist_elems)
		
	# new_goals = new

	# for typ in new_inds:
	# 	u_vs = kb.u_vs[typ][new_inds[typ]]
	# 	print(kb.u_vds[typ])
	# 	print(kb.u_vs[typ])
	# 	print(u_vs)
	# 	for v in u_vs:
	# 		print(kb.u_vds[typ][v])
	# print("moo")
	# for i,record in enumerate(records):
	# 	_hist, shape, vmap = record
	# 	hist = _hist.reshape(shape)

	# 	# print("MEEP")
	# 	# print(np.where(hist == 0))
		
	# 	# print(_hists,_vmaps)
	# 	# for hist,vmap in zip(_hists,_vmaps):
	# 	if(goal in vmap):
	# 		print(ops)
	# 		print(vmap[goal])
	# 		print(np.where(hist == vmap[goal]))

#Adapted from here: https://gitter.im/numba/numba?at=5dc1f9d13d669b28a0408463
@njit
def unravel_indicies(indicies, shape):
	sizes = np.zeros(len(shape), dtype=np.int64)
	result = np.zeros(len(shape), dtype=np.int64)
	sizes[-1] = 1
	for i in range(len(shape) - 2, -1, -1):
		sizes[i] = sizes[i + 1] * shape[i + 1]

	out = np.empty((len(indicies),len(shape)), dtype=np.int64)
	for j,index in enumerate(indicies):
		remainder = index
		for i in range(len(shape)):
			out[j,i] = remainder // sizes[i]
			remainder %= sizes[i]
	return out

i8_u1_dict = DictType(i8,u1)
i8_arr = i8[:]
@njit
def retrace_back_one(goals,records,u_vds,hist_elems):
	unq_arg_inds = Dict.empty(unicode_type, i8_u1_dict)
	pos_by_typ = Dict.empty(unicode_type, i8)


	# for record in records[0]:
	# 	op_name, _hist, shape, arg_types, vmap = record
	# 	for typ in arg_types:
	# 		if(typ not in unq_arg_inds):
	# 			unq_arg_inds[typ] = Dict.empty(i8,u1)
	# 			pos_by_typ[typ] = 0

	#Go through each goal in goals, and find applications of operations 
	#	which resulted in each subgoal. Add the indicies of args of each 
	#	goal satisficing operation application to the set of subgoals for 
	#	the next iteration.
	# _hist_elems = List.empty_list(ListType(HE))
	for goal in goals:
		hist_elems_k = List.empty_list(HE)
		hist_elems.append(hist_elems_k)
		shallowest_depth = u_vds[goal]
		# print([x for x in records.keys()],shallowest_depth)
		# print([x for x in u_vds.keys()])
		if(shallowest_depth == 0): continue
		_records = records[shallowest_depth]
		for record in _records:
			op_uid, _hist, shape, arg_types, vmap = record

			#Make a dictionary for each type to collect unique arg values
			# for _records in records:
			# 	print(_records)
			for typ in arg_types:
				if(typ not in unq_arg_inds):
					unq_arg_inds[typ] = Dict.empty(i8,u1)
					pos_by_typ[typ] = 0

			if(goal in vmap):
				wher = np.where(_hist == vmap[goal])[0]
				inds = unravel_indicies(wher,shape)
				
				for i in range(inds.shape[0]):
					arg_uids = np.empty(inds.shape[1],np.int64)
					for j in range(inds.shape[1]):
						d = unq_arg_inds[arg_types[j]]
						v = inds[i,j]
						if(v not in d):
							d[v] = pos_by_typ[typ]
							pos_by_typ[typ] += 1
						arg_uids[j] = d[v]
					
					# print(arg_uids)
					hist_elems_k.append(HistElm(op_uid,arg_uids))

	#Consolidate the dictionaries of unique arg indicies into arrays
	out_arg_inds = Dict.empty(unicode_type,i8_arr)
	for typ in unq_arg_inds:
		u_vals = out_arg_inds[typ] = np.empty((len(unq_arg_inds[typ])),np.int64)
		for i, v in enumerate(unq_arg_inds[typ]):
			u_vals[i] = v
	# print(out_arg_inds)

	return out_arg_inds







def how_search(kb,ops,goal,search_depth=1):
	if(isinstance(goal, (int,float))):
		g_typ = TYPE_ALIASES['float']
	elif(isinstance(goal, (str))):
		g_typ = TYPE_ALIASES['string']
	else:
		raise NotImplemented("Object goals not implemented yet")

	# depth = 0
	while(goal not in kb.u_vds[g_typ]):
		if(kb.curr_infer_depth > search_depth): break
		forward(kb,ops)
		# for typ in kb.registered_types:
		# 	print("MOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO",kb.curr_infer_depth)
		# 	print(typ)
		# 	print(kb.u_vs[typ])


	if(goal in kb.u_vds[g_typ]):
		retrace_solutions(kb,ops,goal,g_typ)
	# 	print("FOUND IT ", kb.curr_infer_depth)

		
	# else:
	# 	print("NOT HERE")




def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))




X = np.array(np.arange(10),np.float32)
y = 17
# forward(X,y,[Add,Multiply])
# print("OUT",Add_forward1(X,X))
# print(Add_forward2(X,X))

@njit
def make_S(offset):
	S = List.empty_list(unicode_type)
	for x in range(97+offset,97+offset+5):
		S.append(chr(x))
	return S#np.array(S)
S1 = make_S(0)
S2 = make_S(5)

def buildKB():
	kb = NBRT_KnowledgeBase()

	kb.u_vs['unicode_type'] = S1
	kb.u_vs['f4'] = X

	kb.u_vds['unicode_type'] = Dict.empty(unicode_type,i8)
	kb.u_vds['f4'] = Dict.empty(f4,i8)

	for s in S1:
		kb.u_vds['unicode_type'][s] = 0
	for x in X:
		kb.u_vds['f4'][x] = 0

	return kb

kb = buildKB()

# print("STAAAAART")
# for op in [Add,Subtract,Concatenate]:
# 	print(op, op.uid)
# for op in [Add,Subtract,Concatenate]:
# 	print(op, op.uid)
# print(BaseOperator.operators_by_uid)
# forward(kb,[Add,Subtract,Concatenate])
# forward(kb,[Add,Subtract,Concatenate])
how_search(kb,[Add,Subtract,Concatenate],21,2)
# forward(kb,[Add,Subtract,Concatenate])
# for typ in kb.registered_types:
# 	print(typ)
# 	print(kb.u_vs[typ])

# print(cat_forward1(S))

@njit
def g1():
	# uid = 1; d = Dict.empty(f8,i8)
	# Add_forward1(uid,d, X,X)	
	Grumbo_forward1(X,X)	

@njit
def g2():
	# uid = 1; d = Dict.empty(f8,i8)
	# Add_forward2(uid,d, X,X)	
	Grumbo_forward2(X,X)	

# @njit
def s1():
	# uid = 1; d = Dict.empty(unicode_type,i8)
	# cat_forward1(uid,d, S1)	
	cat_forward(S1)	

def f1():
	kb = buildKB()
	forward(kb,[Add,Subtract])
	forward(kb,[Add,Subtract])
	forward(kb,[Add,Subtract])
	forward(kb,[Add,Subtract])
	forward(kb,[Add,Subtract])
	forward(kb,[Add,Subtract])

def h1():
	kb = buildKB()
	how_search(kb,[Add,Subtract],21,2)


# print("g1", time_ms(g1))
# print("g2", time_ms(g2))
# print("s1", time_ms(s1))
# print("forward", time_ms(f1))
print("how_search", time_ms(h1))
