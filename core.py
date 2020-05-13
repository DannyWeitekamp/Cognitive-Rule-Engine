#From here: https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
from numba import types, njit, jit, prange
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, DictType, unicode_type
from numba.experimental import jitclass
import numpy as np
import re
from collections.abc import Iterable 
import timeit
N = 20

def parse_signature(s):
	fn_match = re.match(r"(?P<out_type>\w+)\s?\((?P<arg_types>(?P<args>\w+(,\s?)?)+)\)", s)
	fn_dict = fn_match.groupdict()
	arg_types = [arg.strip() for arg in fn_dict['arg_types'].split(',')]
	return fn_dict['out_type'], arg_types

# def norm_check_types(s):



class NBRT_KnowledgeBase(object):
	def __init__(self):
		pass

class BaseOperator(object):
	registered_operators = {}
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
		cls.registered_operators[cls.__name__.lower()] = cls

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
	signature = 'float(float,float,string,float)'
	def forward(x, y):
		return x + y

compile_forward(Add)

def Multiply(x, y):
	return x * y

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
@njit(nogil=True,fastmath=True,parallel=True) 
def Add_forward1(x0,x1): 
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

	sqeep = np.where(out == 1)
	return u_vs


@njit(nogil=True,fastmath=True,parallel=False) 
def cat_forward1(x0): 
	L0= len(x0)
	out = np.empty((L0,L0))
	d = Dict.empty(unicode_type,i8)
	ind = 0
	for i0 in range(0,L0):
		# d = da[i0]
		for i1 in range(i0+1,L0):
			# for i2 in range(i1+1,L0):
			# 	for i3 in range(0,L0):
			v = x0[i0] + x0[i1]# + x0[i2] + x0[i3] 
			if(v not in d):
				d[v] =ind; ind +=1; 
			out[i0,i1] = d[v]

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
	return 0


@njit(nogil=True,fastmath=True,parallel=True) 
def Add_forward2(x0,x1): 
	L0, L1 = len(x0), len(x1)
	Total_Len = (L0)*(L0-1)*(L0-2)/(1*2*3) * \
				  L1 
	Total_Len = int(Total_Len)    			 
	# print(Total_Len)
	# Total_Len = L0*L0*L0*L1
	# out = np.empty((L0,L0,L1,L0))
	# out = np.empty((Total_Len,))#.tolist()
	out = np.empty((Total_Len,))
	# out = List.empty_list(f8,Total_Len)
	# print(len(out))
	ind = 0
	for i0 in prange(0,L0):
		# m1 = i0*(L0)(L0-1)(L0-2)
		m1 = int(Total_Len-(((L0-i0)*(L0-i0-1)*(L0-i0-2))/6)*L1) #+ (((L0-(i1-i0))*(L0-(i1-i0)-1))/2))
		# print(i0,m1)
		ind = m1
		for i1 in range(i0+1,L0):
			for i2 in range(i1+1,L0):
				for i3 in range(0,L1):
					# i3=0
				
					# ind = i0*L0*L0*L1 + i1*L0*L1 + i2*L0 + i3
					# out[i0,i1,i2,i3] = x0[i0] + x0[i1] + x0[i3] * x1[i2]
					# b = i2*L0+i3+((i0)*(i0-1)/2)*(((i1)*(i1-1)/2))
					
					# print((L0-i0-1)*(L0-i0-2)/2  - ((i1-i0))*((i1-i0-1))/2)
					# m2 = ((L0-(i1-i0+1))*(L0-(i1-i0+1))/2)
					# m2 = (L0-i0-1)*(L0-i0-2)  - (L0-(i1-i0))*(L0-(i1-i0-1))/2
					# print(m1,((L0-i1)*((L0-i1)-1)/2))
					# print(ind, m1,m2,":" ,i0, i1-i0-1,i2-i1-1,i3)#, i0,i1,i2,i3)
					# print(ind,"\t ",i0, i1-i0-1,i2-i1-1,i3)
					# out.append(x0[i0] + x0[i1] + x0[i3] * x1[i2])
					out[ind] = x0[i0] + x0[i1] + x0[i3] * x1[i2] #+ m1 + m2
					# print(out[ind])
					ind += 1
	# print(ind,Total_Len)
	return out


def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))




X = np.array(np.arange(50),np.float32)
y = 17
forward(X,y,[Add,Multiply])
print("OUT",Add_forward1(X,X))
print(Add_forward2(X,X))

S = List.empty_list(unicode_type)
for x in range(97,97+50):
	S.append(chr(x))

print(cat_forward1(S))

def v1():
	Add_forward1(X,X)	

def v2():
	Add_forward2(X,X)	

def s1():
	cat_forward1(S)	


print("v1", time_ms(v1))
print("v2", time_ms(v2))
print("s1", time_ms(s1))
