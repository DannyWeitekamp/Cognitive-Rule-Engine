#From here: https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
from numba import types, njit, jit, prange
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, DictType, unicode_type, Array, Tuple, NamedTuple
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
from collections import namedtuple

# from numbert.caching import _UniqueHashable
import itertools
import warnings
import math
N = 10

print("START")

WARN_START = ("-"*26) + "WARNING" + ("-"*26) + "\n"
WARN_END = ("-"*24) + "END WARNING" + ("-"*24) + "\n"


#Monkey Patch Numba so that the builtin functions for List() and Dict() cache between runs 
def monkey_patch_caching(mod,exclude=[]):
	for name, val in mod.__dict__.items():
		if(isinstance(val,Dispatcher) and name not in exclude):
			val.enable_caching()

#They promised to fix this by 0.51.0, so we'll only run it if an earlier release
if(tuple([int(x) for x in numba.__version__.split('.')]) < (0,51,0)):
	monkey_patch_caching(tl_mod,['_sort'])
	monkey_patch_caching(td_mod)


def str_preserve_ints(x):
	if(isinstance(x,float) and int(x) == x):
		return str(int(x))
	return str(x)


def parse_signature(s):
	fn_match = re.match(r"(?P<out_type>\w+)\s?\((?P<arg_types>(?P<args>\w+(,\s?)?)+)\)", s)
	fn_dict = fn_match.groupdict()
	arg_types = [arg.strip() for arg in fn_dict['arg_types'].split(',')]
	return fn_dict['out_type'], arg_types


class Var(object):
	def __init__(self,binding=None,type=None):
		self.binding = binding
		self.type = type
		self.index = None
	def __repr__(self):
		if(self.binding is None):
			return "?"
		else:
			return str(self.binding)





class OperatorComposition(object):
	def __init__(self,tup):
		#Minimal initialization at instantiation to reduce overhead
		#	instead most initialization is via @property 
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
		if(isinstance(tup,(tuple,list))):
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
			if(hasattr(op,'condition')):
				if(not op.condition(*resolved_args)):
					raise ValueError("Condition Fail.")
			# if(isinstance(op,OperatorComposition)):
			# 	print(op.tup)
			return op.forward(*resolved_args)
		elif(isinstance(tup,Var)):
			try:
				return dq_args.popleft()
			except IndexError:
				raise TypeError("Not Enough Arguments For {}".format(self))			
		else:
			return tup


	@property
	def depth(self):
		if(not hasattr(self,"_depth")):
			_ = self.args
		return self._depth


	@property
	def template(self):
		if(not hasattr(self,"_template")):
			self._template = self._gen_template(self.tup)
		return self._template

	def _accum_args(self,x,arg_arr,typ_arr,depth):
		if(isinstance(x,(list,tuple))):
			arg_types = x[0].arg_types
			depth += 1
			for i, x_i in enumerate(x[1:]):
				if(isinstance(x_i,Var)):
					x_i.index = len(arg_arr)
					arg_arr.append(x_i)
					typ_arr.append(arg_types[i])
				else:
					self._accum_args(x_i,arg_arr,typ_arr,depth)
		elif(isinstance(x,Var)):
			arg_arr.append(x)
			typ_arr.append(None)
	@property
	def __name__(self):
		return repr(self)
	

	@property
	def args(self):
		if(not hasattr(self,"_args")):
			args_arr, type_arr = [],[]
			depth_arr = np.zeros(1,np.int64)
			self._accum_args(self.tup, args_arr,type_arr,depth_arr)
			self._args = args_arr#self._count_args(self.tup)
			self._arg_types = type_arr#self._count_args(self.tup)
			self._depth = depth_arr[0]#self._count_args(self.tup)
		return self._args

	@property
	def out_type(self):
		if(not hasattr(self,"_out_type")):
			if(isinstance(self.tup,(list,tuple))):
				self._out_type = self.tup[0].out_type			
			elif(isinstance(self.tup,Var)):
				self._out_type = None
		return self._out_type

	@property
	def arg_types(self):
		if(not hasattr(self,"_arg_types")):
			_ = self.args
		return self._arg_types

	@property
	def uid(self):
		if(not hasattr(self,"_uid")):
			operators_by_uid = BaseOperator.operators_by_uid
			registered_operators = BaseOperator.registered_operators

			name = self.__repr__(as_unbound=True)
			if(name not in registered_operators):
				uid = len(operators_by_uid)
				operators_by_uid.append(self)
				registered_operators[name] = self
			else:
				uid = registered_operators[name].uid
			self._uid = uid
		return self._uid
	
	def force_cast(self,type_str):
		self.cast_fn = py_type_map[type_str]
		if(self.cast_fn == str): self.cast_fn = str_preserve_ints
		self._out_type = TYPE_ALIASES[type_str]



	def unbind(self):
		for arg in self.args:
			arg.binding = None


	def __repr__(self,as_unbound=False):
		arg_strs = [repr(arg) for arg in self.args] if not as_unbound else ["?"]*len(self.args)
		return self.template.format(*arg_strs)

	def forward(self,*args):
		dq_args = deque(args)
		value = self._execute_composition(self.tup,dq_args)
		if(len(dq_args) > 0):
				raise TypeError("Too Many Arguments For {}. Resolved: {} Extra: {} ".format(self,args[:len(dq_args)],list(dq_args)))
		if(hasattr(self,'cast_fn')): value = self.cast_fn(value)
		return value

	def __call__(self,*args):
		return self.forward(*args)

class BaseOperatorMeta(type):
	def __repr__(cls):
		return cls.template.format(*(['?']*len(cls.arg_types)),name=cls.__name__)

	# def __str__(cls):
	# 	return cls.template.format(,name=cls.__name__)


# initial_right_commutes_by_uid =  List.empty_list(DictType(i8,i8[::1]))
# initial_right_commutes_by_uid.append(Dict.empty(i8,i8[::1]))


class BaseOperator(metaclass=BaseOperatorMeta):
	# __metaclass__ = BaseOperatorMeta
	#Static Attributes
	registered_operators = {}
	operators_by_uid = [None] #Save space for no-op
	# right_commutes_by_uid = initial_right_commutes_by_uid #Save space for no-op

	#Subclass Attributes
	commutes = False
	muted_exceptions = []
	nopython = True

	hash_on = set(['commutes','forward','condition','signature','muted_exceptions'])

	@classmethod
	def _init_signature(cls):
		assert hasattr(cls,'signature'), "Operator %r missing signature." % cls.__name__
		out_type, arg_types = parse_signature(cls.signature)
		out_type = TYPE_ALIASES.get(out_type,out_type)
		arg_types = [TYPE_ALIASES.get(x,x) for x in arg_types]
		# print(arg_types)
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

		right_commutes = Dict.empty(i8,i8[::1])
		for i in range(len(cls.commutes)):
			commuting_set =  cls.commutes[i]
			# print(commuting_set)
			for j in range(len(commuting_set)-1,0,-1):
				right_commutes[commuting_set[j]] = np.array(commuting_set[0:j],dtype=np.int64)
				for k in commuting_set[0:j]:
					assert u_inds[k] == u_inds[commuting_set[j]], \
					 "invalid 'commutes' argument, argument %s and %s have different types \
					  %s and %s" % (j, k, u_types[u_inds[j]], u_types[u_inds[k]])
		cls.right_commutes = right_commutes

		# print(len(cls.operators_by_uid), len(cls.right_commutes_by_uid))
		# assert len(cls.operators_by_uid) == len(cls.right_commutes_by_uid)
		# cls.right_commutes_by_uid.append(right_commutes)

		# print(cls.right_commutes)
		# print(cls.arg_types)

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
		print("%s: Init Stuff Time %.4f ms" % (cls.__name__, t1-t0))

		compile_forward(cls)
		# t2 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
		# print("%s: Compile Forward Time %.4f ms" % (cls.__name__, t2-t1))

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
			if(isinstance(arg,BaseOperatorMeta)):
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
		return type("MOOSE",(BaseOperatorMeta,),{'signature' : signature, 'forward' : f})


	# @classmethod
	def get_template(self,*args):
		iargs, cls = self._assert_iargs(args), type(self)

		arg_strs = []
		for arg in self.args:
			if(isinstance(arg,BaseOperatorMeta)):
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
numba_type_map = {
	"f8" : f8,
	"unicode_type" : unicode_type,
	"string" : unicode_type,
	"number" : f8,	
}

py_type_map = {
	"f8" : float,
	"unicode_type" : str,
	"string" : str,
	"number" : float,	
}

# ALLOWED_TYPES = []

def compile_forward(op):
	_ = "    "
	nopython = op.nopython
	if(nopython):
		forward_func = njit(op.forward,cache=True)
		condition_func =  njit(op.condition,cache=True) if(hasattr(op,'condition')) else None
		try:
			forward_func.compile(op.signature)
		except Exception as e:
			forward_func = op.forward
			nopython= False
			warnings.warn("\n"+ WARN_START + str(e) + ("\nWarning: Operator %s failed to compile forward() " +
				"in nopython mode. To remove this message add nopython=False in the class definition.\n" +
				WARN_END)%op.__name__)			

		if(condition_func != None):
			try:
				condition_func.compile(op.cond_signature)
			except Exception as e:
				# raise e
				condition_func = op.condition
				nopython= False
				warnings.warn("\n"+ WARN_START + str(e) + ("\nWarning: Operator %s failed to compile condition() " +
					"in nopython mode. To remove this message add nopython=False in the class definition.\n" +
					WARN_END)%op.__name__)			
	else:
		forward_func = op.forward
		condition_func = op.condition if(hasattr(op,'condition')) else None

	

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
		# print("COMMUTES", op.right_commutes)

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
	print("%s: Gen Source Time %.4f ms" % (op.__name__, time2-time1))

	# print(source)
	# print("END----------------------")
	l,g = cache_safe_exec(source,gbls={'f':forward_func,'c': condition_func,**globals()})
	# print("TIS HERE:",l[f_name])
	if(nopython):
		op.broadcast_forward = l[f_name]
	else:
		print(op.__name__,"NOPYTHON=False")
		_bf = l[f_name]
		def bf(*args):
			global f
			f = forward_func
			return _bf(*args)
		op.broadcast_forward = bf
	time3 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
	print("%s: Compile Source Time %.4f ms" % (op.__name__,time3-time2))
	

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
		out = is_prime(x)
		return out

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

class Multiply(BaseOperator):
	commutes = True
	signature = 'float(float,float)'
	def forward(x, y):
		return x * y

class Divide(BaseOperator):
	commutes = False
	signature = 'float(float,float)'
	def condition(x, y):
		return y != 0
	def forward(x, y):
		return x / y

class Equals(BaseOperator):
	commutes = False
	signature = 'float(float,float)'
	def forward(x, y):
		return x == y


class Add3(BaseOperator):
	commutes = True
	signature = 'float(float,float,float)'
	def forward(x, y, z):
		return x + y + z

class Mod10(BaseOperator):
	commutes = True
	signature = 'float(float)'
	def forward(x):
		return x % 10

class Div10(BaseOperator):
	commutes = True
	signature = 'float(float)'
	def forward(x):
		return x // 10

class Concatenate(BaseOperator):
	signature = 'string(string,string)'
	def forward(x, y):
		return x + y

class StrToFloat(BaseOperator):
	signature = 'float(string)'
	muted_exceptions = [ValueError]
	nopython = False
	def forward(x):
		return float(x)

class FloatToStr(BaseOperator):
	signature = 'string(float)'
	muted_exceptions = [ValueError]
	nopython = False
	def forward(x):
		# if(int(x) == x):
		# 	return str(int(x))
		return str(x)




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
				# print(self.u_vs)
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

	def how_search(self,ops,goal,search_depth=1,max_solutions=1):
		return how_search(self,ops,goal,search_depth=search_depth,max_solutions=max_solutions)

	def unify_op(self,op,goal):
		return unify_op(self,op,goal)

	def check_produce_goal(self,ops,goal):
		return check_produce_goal(self,ops,goal)

	def forward(self,ops):
		forward(self,ops)

	
		



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

def broadcast_forward_op_comp(kb,op_comp):
	if(op_comp.out_type == None): raise ValueError("Only typed outputs work with this function.")

	arg_sets = [kb.u_vs.get(t,[]) for t in op_comp.arg_types]
	lengths = tuple([len(x) for x in arg_sets])
	out = np.empty(lengths,dtype=np.int64)
	d = Dict.empty(numba_type_map[op_comp.out_type],i8)
	arg_ind_combinations = itertools.product(*[np.arange(l) for l in lengths])
	uid = 1
	for arg_inds in arg_ind_combinations:
		try:
			v = op_comp(*[arg_set[i] for i,arg_set in zip(arg_inds,arg_sets)])
			if(v not in d):
				d[v] = uid; uid += 1;
			out[tuple(arg_inds)] = d[v]
		except ValueError:
			out[tuple(arg_inds)] = 0
	return out, d





# Add.broadcast_forward = Add_forward
# Subtract.broadcast_forward = Subtract_forward
# Concatenate.broadcast_forward = cat_forward
def forward(kb,ops):
	kb._assert_declared_values()

	output_types = set()
	# output_types = set([op.out_type for op in ops])
	new_records = {typ:[] for typ in output_types}
	depth = kb.curr_infer_depth = kb.curr_infer_depth+1
	
	for op in ops:
		if(not all([t in kb.u_vs for t in op.arg_types])): continue
		typ = op.out_type
		if(isinstance(op,BaseOperatorMeta)):
			args = [kb.u_vs[t] for t in op.u_arg_types]
			btsr, vmap = op.broadcast_forward(*args)
		elif(isinstance(op,OperatorComposition)):
			btsr, vmap = broadcast_forward_op_comp(kb,op)

		records = insert_record(kb, depth, op, btsr, vmap)
		new_records[typ] = records
		output_types.add(op.out_type)
		
	for typ in output_types:
		if(typ in new_records):
			vmaps = List([rec[4] for rec in new_records[typ]])
			kb.u_vds[typ] = join_new_vals(kb.u_vds[typ],vmaps,depth)

			if(typ == TYPE_ALIASES['float']):
				kb.u_vs[typ] = array_from_dict(kb.u_vds[typ])
			else:
				kb.u_vs[typ] = list_from_dict(kb.u_vds[typ])
	# print("F_end")

# HE_deffered = deferred_type()
# @jitclass([('op_uid', i8),
# 		   ('args', i8[:])])
# class HistElm(object):
# 	def __init__(self,op_uid,args):
# 		self.op_uid = op_uid
# 		self.args = args
# 	# def __repr__(self):
# 	# 	return str(self.op_uid) + ":" + str(self.args) 
# HE = HistElm.class_type.instance_type
# HE_deffered.define(HE)


HistElm = namedtuple("HistElm",["op_uid","args"])
HE = NamedTuple([i8,i8[::1]],HistElm)


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

def _retrace_goal_history(kb,ops,goal,g_typ, max_solutions):
	u_vds = kb.u_vds[g_typ]
	records = kb.hists[g_typ]

	goals = List.empty_list(kb.registered_types[g_typ])
	goals.append(goal)

	hist_elems = HistElmListList()#List.empty_list(ListType(HE))#new_HE_list(1)#List([List.empty_list(HE)],listtype=ListType(HE))
	arg_inds = retrace_back_one(goals,records,u_vds,hist_elems,kb.curr_infer_depth,max_solutions)
	# print("arg_inds", arg_inds, hist_elems)
	out = [{g_typ: hist_elems}]
	i = 1
	while(True):
		nxt = {}
		new_arg_inds = None
		for typ in arg_inds:
			records,u_vds = kb.hists[typ], kb.u_vds[typ]
			hist_elems = HistElmListList()#List.empty_list(ListType(HE))#new_HE_list(len(goals))#List([List.empty_list(HE) for i in range(len(goals))])
			
			goals = select_from_collection(kb.u_vs[typ],arg_inds[typ])
			# print("goals")
			# print(goals)
			typ_new_inds = retrace_back_one(goals,records,u_vds,hist_elems,kb.curr_infer_depth-i,max_solutions)
			# print("typ_new_inds")
			# print(typ_new_inds)
			if(new_arg_inds is None):
				new_arg_inds = typ_new_inds
			else:
				for typ,inds in typ_new_inds.items():
					if(typ not in new_arg_inds):
						new_arg_inds[typ] = inds
					else:
						new_arg_inds[typ] = np.append(new_arg_inds[typ],inds)

			nxt[typ] = hist_elems
		out.append(nxt)
		if(new_arg_inds is None or len(new_arg_inds) == 0):
			break
		assert i <= kb.curr_infer_depth, "Retrace has persisted past current infer depth."
		i += 1
		arg_inds = new_arg_inds
	return list(reversed(out))

def retrace_solutions(kb,ops,goal,g_typ,max_solutions=1):
	goal_history = _retrace_goal_history(kb,ops,goal,g_typ,max_solutions)

	tups = []
	for depth in range(len(goal_history)):
		tups_depth = {}
		for typ in goal_history[depth].keys():
			tups_depth_typ = tups_depth[typ] = []
			for j,l in enumerate(goal_history[depth][typ]):
				tups_depth_j = []
				for he in l:
					if(he.op_uid == 0):
						tups_depth_j.append(Var(binding=kb.dec_u_vs[typ][he.args[0].item()],type=typ))
					else:
						op = BaseOperator.operators_by_uid[he.op_uid]
						prod_lists = [[op]] +[tups[depth-1][a_typ][he.args[k]] for k,a_typ in enumerate(op.arg_types)]
						for t in itertools.product(*prod_lists):
							# if(isinstance(t[0],OperatorComposition)):
							# 	op_comp = t[0]
							# 	t = OperatorCompositiondeepcopy(op_comp.tup)
							# 	t.bind(*op_comp.args)

							# 	raise ValueError("POOP")

							tups_depth_j.append(t)
				tups_depth_typ.append(tups_depth_j)
		tups.append(tups_depth)

	out = [OperatorComposition(t) for t in tups[-1][g_typ][0][:max_solutions]]

	return out



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
@njit(nogil=True,fastmath=True,cache=True, locals={"arg_ind":i8[::1],"arg_uids":i8[::1]}) 
def retrace_back_one(goals, records, u_vds, hist_elems, max_depth, max_solutions=1):
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
		# print("SHALLOW MAX",shallowest_depth,max_depth)
		for depth in range(shallowest_depth,max_depth+1):
			# print("depth:",depth)
		# depth = shallowest_depth

		#If the goal was declared (i.e. it is present at depth 0) then
		#	 make a no-op history element for it
			if(depth == 0):
				_,_,_,_, vmap = records[0][0]
				if(goal in vmap):
					arg_ind = np.array([vmap[goal]],dtype=np.int64)
					hist_elems_k.append(HistElm(0,arg_ind))

			#Otherwise the goal was infered from the declared values
			else:
				#For every record (i.e. history of an inference with a particular op) 
				if(depth >= len(records)): continue
				_records = records[depth]
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
							
							#Redundant Arguments not allowed 
							# if(len(np.unique(arg_uids)) == len(arg_uids)):
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



def _infer_goal_type(goal):
	if(isinstance(goal, (int,float))):
		return TYPE_ALIASES['float']
	elif(isinstance(goal, (str))):
		return TYPE_ALIASES['string']
	else:
		raise NotImplemented("Object goals not implemented yet")



def unify_op(kb,op,goal):
	g_typ = _infer_goal_type(goal)
	kb._assert_declared_values()
	#Handle Copy/No-op right up front
	if(isinstance(op,OperatorComposition) and op.depth == 0):
		if(g_typ in kb.u_vs and goal in kb.u_vs[g_typ]):
			return [[goal]]
		else:
			return []
	if(op.out_type != g_typ): return []
	
	if(not all([t in kb.u_vs for t in op.arg_types])):return []

	if(isinstance(op,BaseOperatorMeta)):
		args = [kb.u_vs[t] for t in op.u_arg_types]
		_hist, vmap = op.broadcast_forward(*args)
	elif(isinstance(op,OperatorComposition)):
		_hist, vmap = broadcast_forward_op_comp(kb,op)
		
	arg_sets = [kb.u_vs.get(t,[]) for t in op.arg_types]
	if(goal in vmap):
		inds = np.stack(np.where(_hist == vmap[goal])).T

		return [[arg_sets[i][j] for i,j in enumerate(ind)] for ind in inds]
	return []


def how_search(kb,ops,goal,search_depth=1,max_solutions=10,min_stop_depth=-1):
	if(min_stop_depth == -1): min_stop_depth = search_depth
	kb._assert_declared_values()
	g_typ = _infer_goal_type(goal)
	# print(g_typ)
	for depth in range(1,search_depth+1):
		print("depth:",depth, "/", search_depth,kb.curr_infer_depth)
		if(depth < kb.curr_infer_depth): continue
	# while():
		
		if((g_typ in kb.u_vds) and (goal in kb.u_vds[g_typ]) and kb.curr_infer_depth > min_stop_depth):
			break
		
		forward(kb,ops)
		# print(kb.u_vds[g_typ])
		# print(kb.hists)



	if((g_typ in kb.u_vds) and (goal in kb.u_vds[g_typ])):
		# print("RETRACE")
		return retrace_solutions(kb,ops,goal,g_typ,max_solutions=max_solutions)
	return []



if __name__ == "__main__":
	# t2 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
	# print("Init all %.4f ms" % (t2-t1))

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
