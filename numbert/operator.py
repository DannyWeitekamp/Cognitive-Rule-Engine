from numba import types, njit, jit
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple 
from numba.cpython.unicode import  _set_code_point
from numbert.utils import cache_safe_exec
from numbert.core import TYPE_ALIASES ,py_type_map, numba_type_map, numpy_type_map
from numbert.gensource import gen_source_broadcast_forward
from collections import namedtuple, deque
import math
import numpy as np
import timeit
import time
import inspect
import warnings
import re
import types as pytypes
import sys
import __main__
N = 10

print("START")

WARN_START = ("-"*26) + "WARNING" + ("-"*26) + "\n"
WARN_END = ("-"*24) + "END WARNING" + ("-"*24) + "\n"

def compile_forward(op):
	
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
				condition_func = op.condition
				nopython= False
				warnings.warn("\n"+ WARN_START + str(e) + ("\nWarning: Operator %s failed to compile condition() " +
					"in nopython mode. To remove this message add nopython=False in the class definition.\n" +
					WARN_END)%op.__name__)			
	else:
		forward_func = op.forward
		condition_func = op.condition if(hasattr(op,'condition')) else None

	time1 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
	source = gen_source_broadcast_forward(op,condition_func, nopython)
	time2 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)/float(1e6)
	print("%s: Gen Source Time %.4f ms" % (op.__name__, time2-time1))
	f_name = op.__name__+"_forward"
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
