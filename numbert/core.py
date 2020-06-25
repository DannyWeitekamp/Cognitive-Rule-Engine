#From here: https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
from numba import types, njit, jit, prange
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, DictType, unicode_type, Array, Tuple
from numba.experimental import jitclass
from numba.core.dispatcher import Dispatcher
from numba.core import sigutils 
import numba.typed.typedlist as tl_mod 
import numba.typed.typeddict as td_mod
import numba
from numba.core.dispatcher import Dispatcher
import numpy as np
import re
from collections.abc import Iterable 
from collections import deque
import timeit
from pprint import pprint
from numbert.utils import cache_safe_exec
import inspect
import time
from numbert.data_trans import infer_type, infer_nb_type

from numbert.caching import _UniqueHashable
import itertools
N = 10

print("START")


#Monkey Patch Numba so that the builtin functions for List() and Dict() cache between runs 
def monkey_patch_caching(mod,exclude=[]):
	for name, val in mod.__dict__.items():
		if(isinstance(val,Dispatcher) and name not in exclude):
			val.enable_caching()

#They promised to fix this by 0.51.0, so we'll only run it if an earlier release
if(tuple([int(x) for x in numba.__version__.split('.')]) < (0,51,0)):
	monkey_patch_caching(tl_mod,['_sort'])
	monkey_patch_caching(td_mod)



def parse_signature(s):
	fn_match = re.match(r"(?P<out_type>\w+)\s?\((?P<arg_types>(?P<args>\w+(,\s?)?)+)\)", s)
	fn_dict = fn_match.groupdict()
	arg_types = [arg.strip() for arg in fn_dict['arg_types'].split(',')]
	return fn_dict['out_type'], arg_types

# def norm_check_types(s):


		# [time][op]

class Var(object):
	def __init__(self,binding=None):
		self.binding = binding
	def __repr__(self):
		if(self.binding is None):
			return "?"
		else:
			return self.binding





class OperatorComposition(object):
	def __init__(self,tup):
		self.tup = tup
		
	def _gen_template(self,x):
		if(isinstance(x,(list,tuple))):
			rest = [self._gen_template(x[j]) for j in range(1,len(x))]
			return x[0].template.format(*rest,name=x[0].__name__)
		elif(isinstance(x,Var)):
			return "{}"
		else:
			return x

	def _execute_composition(self,tup,dq_args):
		L = len(tup)
		op = tup[0]
		resolved_args = []
		for i in range(1,L):
			t_i = tup[i]
			if(isinstance(t_i,tuple)):
				val = self._execute_composition(t_i,dq_args)
				resolved_args.append(val)
			elif(isinstance(t_i,Var)):
				try:
					a_i = dq_args.popleft()
				except IndexError:
					raise TypeError("Not Enough Arguments For {}".format(self))

				resolved_args.append(a_i)
			else:
				resolved_args.append(t_i)
		if(len(dq_args) > 0):
			raise TypeError("Too Many Arguments For {}".format(self))
		if(hasattr(op,'condition')):
			if(not op.condition(*resolved_args)):
				raise ValueError()
		return op.forward(*resolved_args)

	@property
	def template(self):
		if(not hasattr(self,"_template")):
			self._template = self._gen_template(self.tup)
		return self._template

	def _count_args(self,x):
		if(isinstance(x,(list,tuple))):
			return sum([self._count_args(y) for y in x])
		elif(isinstance(x,Var)):
			return 1
		else:
			return 0

	@property
	def n_args(self):
		if(not hasattr(self,"_n_args")):
			self._n_args = self._count_args(self.tup)
		return self._n_args
	

	def __repr__(self):
		return self.template.format(*(["?"]*self.n_args))


	def __call__(self,*args):
		value = self._execute_composition(self.tup,deque(args))
		return value

class BaseOperatorMeta(type):
	def __repr__(cls):
		return cls.template.format(*(['?']*len(cls.arg_types)),name=cls.__name__)

	# def __str__(cls):
	# 	return cls.template.format(,name=cls.__name__)



class BaseOperator(_UniqueHashable, metaclass=BaseOperatorMeta):
	# __metaclass__ = BaseOperatorMeta
	#Static Attributes
	registered_operators = {}
	operators_by_uid = [None] #Save space for no-op

	#Subclass Attributes
	commutes = False
	muted_exceptions = []

	hash_on = set(['commutes','forward','condition','signature','muted_exceptions'])

	@classmethod
	def _init_signature(cls):
		assert hasattr(cls,'signature'), "Operator %r missing signature." % cls.__name__
		out_type, arg_types = parse_signature(cls.signature)
		out_type = TYPE_ALIASES.get(out_type,out_type)
		arg_types = [TYPE_ALIASES.get(x,x) for x in arg_types]
		print(arg_types)
		cls.out_type = out_type
		cls.arg_types = arg_types
		cls.signature = "{}({})".format(out_type,",".join(arg_types))
		cls.cond_signature = "u1({})".format(",".join(arg_types))

		u_types,u_inds = np.unique(arg_types,return_inverse=True)
		cls.u_arg_types = u_types
		cls.u_arg_inds = u_inds

		if(isinstance(cls.commutes,bool)):
			if(cls.commutes == True):
				cls.commutes = [np.where(i == u_inds)[0].tolist() for i in range(len(u_types))]
			else:
				cls.commutes = []
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
	def _check_funcs(cls):
		assert hasattr(cls,'forward'), "Operator %r missing forward() function." % cls.__name__

		args = inspect.getargspec(cls.forward).args
		assert len(args) == len(cls.arg_types),\
		 	"%s.forward(%s) has %s arguments but signature %s has %s arguments." \
		 	%(cls.__name__,",".join(args),len(args),cls.signature,len(cls.arg_types))

		if(hasattr(cls,'condition')):
			args = inspect.getargspec(cls.condition).args
			assert len(args) == len(cls.arg_types),\
		 	"%s.condition(%s) has %s arguments but signature %s has %s arguments." \
		 	%(cls.__name__,",".join(args),len(args),cls.signature,len(cls.arg_types))


	@classmethod
	def _register(cls):
		name = cls.__name__.lower()
		
		if(name in cls.registered_operators):
			raise Warning("Duplicate Operator Definition %s" % name)

		uid = len(cls.operators_by_uid)
		cls.uid = uid
		cls.operators_by_uid.append(cls)
		cls.registered_operators[name] = cls

	@classmethod
	def _init_template(cls):
		if(not hasattr(cls,"template")):
			brks = ["{%i}"%i for i in range(len(cls.arg_types))]
			cls.template = "{name}("+",".join(brks)+")"

	
	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)

		t0 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
		cls._init_signature()
		cls._check_funcs()
		cls._register()
		cls._init_template()
		t1 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
		print("Init Stuff Time %.4f ms" % (t1-t0))

		compile_forward(cls)
		t2 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
		print("Compile Forward Time %.4f ms" % (t2-t1))

	@classmethod
	def _assert_cargs(cls,args,allow_zero=True):
		# cls = type(self)
		assert (allow_zero and len(args) == 0) or len(cls.arg_types) == len(args), \
			"incorrect number of arguments for signature: %s" % cls.signature
		return args if(len(args) != 0) else [None]*len(cls.arg_types)

	def _assert_iargs(self,args,allow_zero=True):
		assert (allow_zero and len(args) == 0) or len(self.arg_types) == len(args), \
			"incorrect number of arguments for signature: %s" % self.signature
		return args if(len(args) != 0) else [None]*len(self.arg_types)

	def __init__(self,*args):
		pass
		

		

	def __new__(cls,*args):
		raise NotImplemented()
		print(cls)

		args = cls._assert_cargs(args)
		arg_types = []
		arg_strs = []
		for typ, arg in zip(cls.arg_types,args):
			if(isinstance(arg,BaseOperator)):
				arg_types += arg.arg_types
				arg_strs.append(arg.__name__)
			elif(arg is None):
				arg_types.append(typ)
				arg_strs.append("?")
			else:
				arg_strs.append(arg)
		arg_types = arg_types
		signature = cls.out_type + "("+",".join(arg_types) +")"

		def f(*args):
			pass


		print(arg_strs)
		return type("MOOSE",(BaseOperator,),{'signature' : signature, 'forward' : f})


	# @classmethod
	def get_template(self,*args):
		iargs, cls = self._assert_iargs(args), type(self)

		arg_strs = []
		for arg in self.args:
			if(isinstance(arg,BaseOperator)):
				arg_strs.append(arg.get_template())
			elif(arg is None):
				arg_strs.append("{}")
			else:
				arg_strs.append(str(arg))

		temp = cls.template.format(*arg_strs,name=cls.__name__)
		iarg_strs = ["{}" if (arg is None) else str(arg) for arg in iargs ]
		print(temp, iarg_strs)
		return temp.format(*iarg_strs)

	@classmethod
	def get_hashable(cls):
		d = {k: v for k,v in vars(cls).items() if k in cls.hash_on}
		return d

	def __repr__(self):
		return self.get_template()

	def __call__(self,*args):
		self._assert_iargs(args,False)
		raise NotImplementedError()
	

LOOPLIFT_UNJITABLES = True
UID_START = 1
TYPE_ALIASES = {
	"float" : 'f8',
	"flt" : 'f8',
	"number" : 'f8',
	"string" : 'unicode_type',
	"str" : 'unicode_type'
}

# ALLOWED_TYPES = []

def compile_forward(op):
	_ = "    "
	nopython = True
	forward_func = njit(op.forward,cache=True)
	condition_func =  njit(op.condition,cache=True) if(hasattr(op,'condition')) else None
	try:
		forward_func.compile(op.signature)
	except Exception:
		forward_func = op.forward
		nopython= False

	if(condition_func != None):
		try:
			condition_func.compile(op.cond_signature)
		except Exception as e:
			raise e
			condition_func = op.condition
			nopython= False


	# condition_func, nopython = njit(op.forward,cache=True), True
	# try:
	# 	forward_func.compile(op.signature)
	# except Exception:
	# 	forward_func, nopython = op.forward, False
	time1 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)


	f_name = op.__name__+"_forward"
	if(nopython):
		header = '@jit(nogil=True, fastmath=True, cache=True) \n'
	elif(LOOPLIFT_UNJITABLES and len(op.muted_exceptions) == 0):
		header = '@jit(fastmath=True, looplift=True, forceobj=True) \n'
	else:
		header = ""
	func_def =	'def {}({}): \n' #+ \

	func_def = func_def.format(f_name,
		 ",".join(["x%i"%i for i in range(len(op.u_arg_types))]) )

	defs = _+", ".join(["L%i"%i for i in range(len(op.u_arg_types))]) + " = " + \
			  ", ".join(["len(x%i)"%i for i in range(len(op.u_arg_types))]) + "\n"

	defs += _+"out = np.empty((%s),dtype=np.int64)\n"%",".join(["L%s"%x for x in op.u_arg_inds])
	defs += _+"d = Dict.empty({},i8)\n".format(op.out_type)
	defs += _+"uid = {}\n".format(UID_START)
				
	loops = ""
	curr_indent = 1
	for i in range(len(op.arg_types)):
		curr_indent = i+1
		l = _*curr_indent + "for i{} in range(L{}):\n"
		l = l.format(i,op.u_arg_inds[i])
		loops += l
	

	all_indicies = ["i%s"%i for i in range(len(op.arg_types))]
	arg_terms = ["x{}[i{}]".format(op.u_arg_inds[i],i) for i in range(len(op.arg_types))]
	cond_expr = "{}\n"
	if(len(op.right_commutes) > 0 or condition_func != None):
		curr_indent += 1
		conds = []

		if(len(op.right_commutes) > 0):
			for i_a, i_bs in op.right_commutes.items():
				conds.append("i{} >= i{}".format(i_a,i_bs[-1]))
		if(condition_func != None):
			conds.append("c({})".format(",".join(arg_terms)))

		cond_expr =  _*curr_indent     + "if({}):\n".format(" and ".join(conds))
		cond_expr += "{}\n"#_*(curr_indent+1) + "{}\n"
		cond_expr += _*(curr_indent)   + "else:\n"
		cond_expr += _*(curr_indent+1) + "out[{}] =  0\n".format(",".join(all_indicies))
		print("COMMUTES", op.right_commutes)

	# use_try = False
	try_expr = "{}"
	if(len(op.muted_exceptions) > 0):
		# use_try = True
		try_expr = _*(curr_indent+1) + "try:\n"
		try_expr += "{}\n"
		try_expr += _*(curr_indent+1) + "except ({}):\n".format(",".join([x.__name__ for x in op.muted_exceptions]))
		try_expr += _*(curr_indent+2) + "out[{}] =  0\n".format(",".join(all_indicies))
		curr_indent += 1

	
	exec_code =  _*(curr_indent+1) +"v = f({})\n".format(",".join(arg_terms))
	exec_code += _*(curr_indent+1) +"if(v not in d):\n"
	exec_code += _*(curr_indent+2) +"d[v] = uid; uid +=1;\n"
	exec_code += _*(curr_indent+1) +"out[{}] = d[v]".format(",".join(all_indicies))


	exec_code = try_expr.format(exec_code)

	cond_expr = cond_expr.format(exec_code)
	ret_expr = _+"return out, d\n"
	source = header + func_def + defs +  loops + cond_expr+ret_expr

	time2 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
	print("Gen Source Time %.4f ms" % (time2-time1))

	print(source)
	print("END----------------------")
	l,g = cache_safe_exec(source,gbls={'f':forward_func,'c': condition_func,**globals()})
	print("TIS HERE:",l[f_name])
	if(nopython):
		op.broadcast_forward = l[f_name]
	else:
		print(op.__name__,"NOPYTHON")
		_bf = l[f_name]
		def bf(*args):
			global f
			f = forward_func
			return _bf(*args)
		op.broadcast_forward = bf
	time3 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
	print("Compile Source Time %.4f ms" % (time3-time2))
	

	# print(func_def + defs +  loops + cond_expr)

				# '	return {}({}) \n' + \
				# 'out_func = {}'

# def normalize_types():
t1 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
import math
#from here: https://stackoverflow.com/questions/18833759/python-prime-number-checker
@njit(cache=True)
def is_prime(n):
    if n % 2 == 0 and n > 2: 
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

class SquaresOfPrimes(BaseOperator):
	signature = 'float(float)'
	def condition(x):
		return is_prime(x)

	def forward(x):
		return x**2

class EvenPowersOfPrimes(BaseOperator):
	signature = 'float(float,float)'
	def condition(x,y):
		b = is_prime(x)
		a = (y % 2 == 0) and (y > 0) and (y == int(y))
		return a and b

	def forward(x,y):
		return x**y

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
	signature = 'string(string,string)'
	def forward(x, y):
		return x + y

class StrToFloat(BaseOperator):
	signature = 'float(string)'
	muted_exceptions = [ValueError]
	def forward(x):
		return float(x)

t2 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
print("Init all %.4f ms" % (t2-t1))

# a = Add(None,Add())
# print(type(a).signature , ":", a.signature )
# print(a.get_template(1,2,None))

# print(a.forward)
# a(1,2,3)
print(repr(Add))
# print(Add.__metaclass__)
v = Var()
t = (Add,v,(Subtract,v,v))
oc = OperatorComposition(t)

print(oc)
print(oc.template)
print(oc(1,2,3))

# Add(None)

# raise ValueError("STOP")
# compile_forward(Add)
# compile_forward(Subtract)
# compile_forward(Concatenate)

# def Multiply(x, y):
# 	return x * y

# def forward(state, goal, operators):
# 	for op in operators:
# 		for i in range(len(state)):
# 			for j in range(len(state)):
# 				pass




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
	d = Dict.empty(f8,i8)
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
	d = Dict.empty(f8,i8)
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
	'''
		hist : dict<str,Dict<int,Record>> stores a compact record of inferences
		hist_structs : dict<str,Record_Typ>> Stores the numba tuple types for records.
			keyed by type string valued by a numba tuple representing the tuple type
		curr_infer_depth: int The deepest inference depth in the knowledge base so far
		u_vds: dict<str,Dict<Typ,int>> Dictionaries for each type which map values to
			the deepest depth at which they have been inffered.
		u_vs: dict<str,Iterable<Typ>> Lists/Arrays for each type which simply hold the
			values() of their corresponding dictionary in u_vds. The purpose of this is
			is to make a contiguous copy (Dictionary keys are not gaurenteed to be contiguous)

		dec_u_vs: like dict<str,Iterable<Typ>> u_vs but is only for declared facts (i.e. depth=0)

		registered_types: Maps typ strings to their actual types
		hist_consistent: whether or not the current history of operation use is consistent
			with the declared facts.
	'''
	def __init__(self):
		self.hists = {}
		self.hist_structs = {}
		self.curr_infer_depth = 0
		# self.vmaps = {}
		self.u_vds = {}
		self.u_vs = {}

		self.dec_u_vs = {}
		
		self.registered_types ={'f8': f8, 'unicode_type' : unicode_type}
		self.hist_consistent = True
		self.declared_consistent = True

	def _assert_record_type(self,typ):
		if(typ not in self.hist_structs):
			typ_cls = self.registered_types[typ]


			#Type : (op_id, _hist, shape, arg_types, vmap)
			struct_typ = self.hist_structs[typ] = Tuple([i8,
										 i8[::1], i8[::1], ListType(unicode_type),
										 DictType(typ_cls,i8)])
			self.hists[typ] = self.hists.get(typ,Dict.empty(i8,ListType(struct_typ)))
		return self.hist_structs[typ]
	def _assert_declare_store(self,typ):
		struct_typ = self._assert_record_type(typ)
		typ_store = self.hists[typ]
		if(0 not in typ_store):
			typ_cls = self.registered_types[typ]
			tsd = typ_store[0] = typ_store.get(0, List.empty_list(struct_typ))
			tl = List();tl.append(typ);
			vmap = Dict.empty(typ_cls,i8)
			#Type : (0 (i.e. no-op), _hist, shape, arg_types, vmap)
			tsd.append( tuple([0, np.empty((0,),dtype=np.int64),
					   np.empty((0,),dtype=np.int64), tl,vmap]) )


	def _assert_declared_values(self):
		if(not self.declared_consistent):
			for typ in self.registered_types.keys():
				self._assert_declare_store(typ)
				record = self.hists[typ][0][0]
				_,_,_,_, vmap = record

				typ_cls = self.registered_types[typ]
				d = self.u_vds[typ] = Dict.empty(typ_cls,i8)
				for x in vmap:
					d[x] = 0
				if(typ == TYPE_ALIASES['float']):
					self.u_vs[typ] = array_from_dict(d)
				else:
					self.u_vs[typ] = list_from_dict(d)

				self.dec_u_vs[typ] = self.u_vs[typ].copy()
				print(self.u_vs)
			self.declared_consistent = True




	def declare(self,x,typ=None):
		if(typ is None): typ = TYPE_ALIASES[infer_type(x)]
		self._assert_declare_store(typ)
		record = self.hists[typ][0][0]
		_,_,_,_, vmap = record

		if(x not in vmap):
			vmap[x] = len(vmap)
			self.hist_consistent = False
			self.declared_consistent = False



# @njit(nogil=True,fastmath=True,parallel=False) 

def insert_record(kb,depth,op, btsr, vmap):
	# print('is')
	typ = op.out_type
	struct_typ = kb._assert_record_type(typ)
	# if(typ not in kb.hist_structs):
	# 	typ_cls = kb.registered_types[typ]
	# 	kb.hist_structs[typ] = Tuple([i8,
	# 								 i8[::1], i8[::1], ListType(unicode_type),
	# 								 DictType(typ_cls,i8)])

	typ_store = kb.hists[typ] = kb.hists.get(typ,Dict.empty(i8,ListType(struct_typ)))
	tsd = typ_store[depth] = typ_store.get(depth, List.empty_list(struct_typ))
	tsd.append(tuple([op.uid,
					  btsr.reshape(-1), np.array(btsr.shape,np.int64), List(op.arg_types),
					  vmap]))
	# print('istop')
	return tsd

# @njit(cache=True):
# def extract_vmaps():

# Add.broadcast_forward = Add_forward
# Subtract.broadcast_forward = Subtract_forward
# Concatenate.broadcast_forward = cat_forward
def forward(kb,ops):
	# print("F_start",kb.curr_infer_depth)
	# if(kb.curr_infer_depth == 0):
	kb._assert_declared_values()
	# 	kb.u_vds = kb.hists[]

	output_types = set([op.out_type for op in ops])
	new_records = {typ:[] for typ in output_types}
	depth = kb.curr_infer_depth = kb.curr_infer_depth+1
	
	for op in ops:
		typ = op.out_type
		args = [kb.u_vs[t] for t in op.u_arg_types]
		btsr, vmap = op.broadcast_forward(*args)
		records = insert_record(kb,depth,op,btsr,vmap)
		new_records[typ] = records
		
	for typ in output_types:
		if(typ in new_records):
			# print("A")
			vmaps = List([rec[4] for rec in new_records[typ]])
			# print("_A")
			kb.u_vds[typ] = join_new_vals(kb.u_vds[typ],vmaps,depth)

			if(typ == TYPE_ALIASES['float']):
				kb.u_vs[typ] = array_from_dict(kb.u_vds[typ])
			else:
				kb.u_vs[typ] = list_from_dict(kb.u_vds[typ])
	# print("F_end")

HE_deffered = deferred_type()
@jitclass([('op_uid', i8),
		   ('args', i8[:])])
class HistElm(object):
	def __init__(self,op_uid,args):
		self.op_uid = op_uid
		self.args = args
	# def __repr__(self):
	# 	return str(self.op_uid) + ":" + str(self.args) 
HE = HistElm.class_type.instance_type
HE_deffered.define(HE)

# def new_HE_list(n):
# 	out = List.empty_list(ListType(HE))
# 	for i in range(n):
# 		out.append(List.empty_list(HE))
# 	return out
@njit(cache=True)
def select_from_list(lst,sel):
	out = List()
	for s in sel:
		out.append(lst[s])
	return out

def select_from_collection(col,sel):
	if(isinstance(col,np.ndarray)):
		return col[sel]
	elif(isinstance(col,List)):
		return select_from_list(col,sel)


he_list = ListType(HE)
@njit(cache=True)
def HistElmListList():
	return List.empty_list(he_list)

def retrace_solutions(kb,ops,goal,g_typ,max_solutions=1):
	u_vds = kb.u_vds[g_typ]
	records = kb.hists[g_typ]

	goals = List.empty_list(kb.registered_types[g_typ])
	goals.append(goal)

	hist_elems = HistElmListList()#List.empty_list(ListType(HE))#new_HE_list(1)#List([List.empty_list(HE)],listtype=ListType(HE))
	arg_inds = retrace_back_one(goals,records,u_vds,hist_elems,max_solutions)

	out = [{g_typ: hist_elems}]
	all_arg_inds = [arg_inds]
	
	finished, i = False, 1
	while(not finished):
		nxt = {}
		# print("AQUI",[(k,type(v),v) for k,v in arg_inds.items()])
		for typ in arg_inds:
			records,u_vds = kb.hists[typ], kb.u_vds[typ]
			hist_elems = HistElmListList()#List.empty_list(ListType(HE))#new_HE_list(len(goals))#List([List.empty_list(HE) for i in range(len(goals))])
			
			goals = select_from_collection(kb.u_vs[typ],arg_inds[typ])
			arg_inds = retrace_back_one(goals,records,u_vds,hist_elems,max_solutions)
			nxt[typ] = hist_elems
			if(len(arg_inds) == 0):
				finished = True
				# print("FINISHED")
				break
		out.append(nxt)
		assert i <= kb.curr_infer_depth, "Retrace has persisted past current infer depth."
		i += 1

	# print("out")
	# print(out)
	# print("------------")
	tups = []
	for i in range(len(out)):
		# print(len(out)-i)
		tups_depth = {}
		for typ in out[len(out)-i-1].keys():
			# print(typ)
			# print(out[i][typ])
			tups_depth_typ = tups_depth[typ] = []
			for j,l in enumerate(out[len(out)-i-1][typ]):
				tups_depth_j = []
				for he in l:
					if(he.op_uid == 0):
						# print(kb.dec_u_vs[typ][he.args[0].item()])
						tups_depth_j.append(kb.dec_u_vs[typ][he.args[0].item()])
					else:
						op = BaseOperator.operators_by_uid[he.op_uid]
						prod_lists = [[op]] +[tups[i-1][typ][he.args[k]] for k,typ in enumerate(op.arg_types)]
						for t in itertools.product(*prod_lists):
							# print(t)
							tups_depth_j.append(t)
					# print([str(x.op_uid) + ":(" + str(x.args) +")" for x in l ])
				tups_depth_typ.append(tups_depth_j)
		tups.append(tups_depth)
	# pprint(tups[-1][g_typ][0])

	out = [OperatorComposition(t) for t in tups[-1][g_typ][0][:max_solutions]]

	return out
	# print("------------")

		



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
@njit(nogil=True,fastmath=True,cache=True) 
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

i8_i8_dict = DictType(i8,i8)
i8_arr = i8[:]
@njit(nogil=True,fastmath=True,cache=True) 
def retrace_back_one(goals,records,u_vds,hist_elems,max_solutions=1):
	unq_arg_inds = Dict.empty(unicode_type, i8_i8_dict)
	pos_by_typ = Dict.empty(unicode_type, i8)

	solution_quota = float(max_solutions-len(goals)) if max_solutions else np.inf
	#Go through each goal in goals, and find applications of operations 
	#	which resulted in each subgoal. Add the indicies of args of each 
	#	goal satisficing operation application to the set of subgoals for 
	#	the next iteration.
	for goal in goals:
		n_goal_solutions = 0
		hist_elems_k = List.empty_list(HE)
		hist_elems.append(hist_elems_k)


		#Determine the shallowest infer depth where the goal was encountered
		shallowest_depth = u_vds[goal]

		#If the goal was declared (i.e. it is present at depth 0) then
		#	 make a no-op history element for it
		if(shallowest_depth == 0):
			_,_,_,_, vmap = records[0][0]
			if(goal in vmap):
				arg_ind = np.array([vmap[goal]],dtype=np.int64)
				hist_elems_k.append(HistElm(0,arg_ind))

		#Otherwise the goal was infered from the declared values
		else:
			#For every record (i.e. history of an inference with a particular op) 
			_records = records[shallowest_depth]
			for record in _records:
				needs_more_solutions = True
				op_uid, _hist, shape, arg_types, vmap = record

				#Make a dictionary for each type to collect unique arg values
				for typ in arg_types:
					if(typ not in unq_arg_inds):
						unq_arg_inds[typ] = Dict.empty(i8,i8)
						pos_by_typ[typ] = 0

				#If the record shows that the goal was produced by the op associated
				#	with record. 
				if(goal in vmap):
					#Then find any set of arguments used to produce it
					wher = np.where(_hist == vmap[goal])[0]
					inds = unravel_indicies(wher,shape)
					
					#For every such combination of arguments
					for i in range(inds.shape[0]):
						#Build a mapping from each argument's index to a unique id
						arg_uids = np.empty(inds.shape[1],np.int64)
						for j in range(inds.shape[1]):
							d = unq_arg_inds[arg_types[j]]
							v = inds[i,j]
							if(v not in d):
								d[v] = pos_by_typ[typ]
								pos_by_typ[typ] += 1
							arg_uids[j] = d[v]
						
						#Store the op_uid and argument unique ids in a HistElm
						hist_elems_k.append(HistElm(op_uid,arg_uids))
						n_goal_solutions += 1
						if(n_goal_solutions >= 1 and solution_quota <= 0):
							needs_more_solutions = False
							break
				if(not needs_more_solutions): break


	#Consolidate the dictionaries of unique arg indicies into arrays.
	#	These will be used with select_from_collection to slice out goals
	#	for the next iteration.
	out_arg_inds = Dict.empty(unicode_type,i8_arr)
	for typ in unq_arg_inds:
		u_vals = out_arg_inds[typ] = np.empty((len(unq_arg_inds[typ])),np.int64)
		for i, v in enumerate(unq_arg_inds[typ]):
			u_vals[i] = v

	return out_arg_inds







def how_search(kb,ops,goal,search_depth=1,max_solutions=1):
	kb._assert_declared_values()
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
		for typ in kb.registered_types:
			pass
			# print("MOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO",kb.curr_infer_depth)
			# print(typ)
			# print(kb.u_vs[typ])
			


	if(goal in kb.u_vds[g_typ]):
		return retrace_solutions(kb,ops,goal,g_typ,max_solutions=max_solutions)
	return None
	# 	print("FOUND IT ", kb.curr_infer_depth)

		
	# else:
	# 	print("NOT HERE")

