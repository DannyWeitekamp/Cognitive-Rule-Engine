import numpy as np
import sys,os
import hashlib 
import timeit
from types import FunctionType

import os
import time
from pathlib import Path
import shutil
# from cre.caching import cache_dir

#Snatch AppDirs from numba numba and find the cache dir
from numba.misc.appdirs import AppDirs
from base64 import b64encode

#Resolve the location of the cache_dir
appdirs = AppDirs(appname="cre", appauthor=False)
cache_dir = os.path.join(appdirs.user_cache_dir,"cre_cache")
os.environ['NUMBA_CACHE_DIR'] = os.path.join(os.path.split(cache_dir)[0], "numba_cache")
print("CRE Cache Lives Here: ", cache_dir)

#Make the cache_dir if it doesn't exist
if not os.path.exists(cache_dir):
	os.makedirs(cache_dir)
# Path(os.path.join(cache_dir,"__init__.py")).touch(exist_ok=True)

#Add cre_cache to path
sys.path.insert(0, appdirs.user_cache_dir)

def clear_cache():
	#Clear the cache folder determined by AppDirs
	shutil.rmtree(os.path.abspath(os.path.join(cache_dir,"../")),ignore_errors=True)

	#Clear __pycache__ in /cre
	shutil.rmtree(os.path.abspath(os.path.join(os.path.dirname(__file__),"../cre/__pycache__")), ignore_errors=True)
	print("Cache Cleared!")

import linecache
def cache_safe_exec(source,lcs=None,gbls=None,cache_name='cache-safe'):
    fp = "<ipython-%s>" %cache_name
    lines = [line + '\n' for line in source.splitlines()]
    linecache.cache[fp] = (len(source), None, lines, fp)
    code = compile(source,fp,'exec')
    l = lcs if lcs is not None else {}
    g = gbls if gbls is not None else globals()
    exec(code,g,l)
    return l,g


class _UniqueHashable():
	def get_hashable(self):
		raise NotImplemented()

	def get_hashcode(self):
		return unique_hash(self.get_hashable())
	


def update_unique_hash(m,obj):
	if(isinstance(obj,str)):
		m.update(obj.encode('utf-8'))
	elif(isinstance(obj,(tuple,list, np.ndarray))):
		for i,x in enumerate(obj):
			update_unique_hash(m,i)
			update_unique_hash(m,x)
	elif(isinstance(obj,dict)):
		for k,v in obj.items():
			update_unique_hash(m,k)
			update_unique_hash(m,v)
	elif(isinstance(obj,FunctionType)):
		m.update(obj.__code__.co_code)
	elif(isinstance(obj,_UniqueHashable)):
		update_unique_hash(m,obj.get_hashable())
	elif(isinstance(obj,bytes)):
		m.update(obj)
	else:
		m.update(str(obj).encode('utf-8'))


def unique_hash(stuff, hash_func='sha256'):
	'''Returns a 64-bit encoded hashstring of some 'stuff' '''
	m = hashlib.new(hash_func)
	update_unique_hash(m,stuff)	

	# Encode in base64 map the usual altchars '/' and "+' to 'A' and 'B".
	s = b64encode(m.digest(),altchars=b'AB').decode('utf-8')
	# Strip the trailing '='.
	s = s[:-1]
	return s

def unique_hash_v(stuff, hash_func='sha256'):
	'''Same as unique_hash but adds in cre.__version__'''
	from cre import __version__
	m = hashlib.new(hash_func)
	update_unique_hash(m,__version__)	
	update_unique_hash(m,stuff)	

	# Encode in base64 map the usual altchars '/' and "+' to 'A' and 'B".
	s = b64encode(m.digest(),altchars=b'AB').decode('utf-8')
	# Strip the trailing '='.
	s = s[:-1]
	return s


def get_cache_path(name,hsh=None,suffix=".py"):
	if(hsh is None):
		return os.path.join(cache_dir,name+suffix)
	else:
		return os.path.join(cache_dir,name,"_" + str(hsh) +suffix)

def source_in_cache(name,hsh=None):
	path = get_cache_path(name,hsh)
	return path if os.path.exists(path) else None


def source_from_cache(name,hsh):
	path = get_cache_path(name,hsh)
	with open(path,mode='r') as f:
		out = f.read()
	return out

def aot_compile(name,hsh):
	out = import_from_cached(name,hsh,['cc'])	
	out['cc'].compile()

def source_to_cache(name,hsh,source,is_aot=False):
	path = get_cache_path(name,hsh)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path,mode='w') as f:
		f.write(source)
		f.flush()
	if(is_aot): aot_compile(name,hsh)


def gen_import_str(name,hsh,targets,aot_module=None):
	aot_module = aot_module if aot_module else ''
	if(isinstance(targets,dict)):
		targets_str = ", ".join([f'{k} as {v}' for k,v in targets.items()])
	else:
		targets_str = ", ".join(targets)
	return f"from cre_cache.{name}.{aot_module}_{hsh} import {targets_str}"


def _import_cached(name,hsh,aot_module=None):
	if(not aot_module): aot_module = ''
	mod_str = f'cre_cache.{name}.{aot_module}_{hsh}'
	try:
		return importlib.import_module(mod_str)
	except (ModuleNotFoundError) as e:
		#Invalidates any the finder caches in case the module has was newly created
		importlib.invalidate_caches()
		return importlib.import_module(mod_str)

import importlib
def import_from_cached(name,hsh,targets,aot_module=None):
	if(aot_module):
		try:
			#Try to import the AOT Module
			mod = _import_cached(name,hsh,aot_module)
		except (ImportError,AttributeError) as e:
			try:
				#If couldn't import try to compile the AOT Module
				aot_compile(name,hsh)
				mod = _import_cached(name,hsh,aot_module)
			except Exception as e:
				#If still couldn't import or recompile, import as cached
				mod = _import_cached(name,hsh)
	else:
		mod = _import_cached(name,hsh)
		
	return {x:getattr(mod,x) for x in targets}
	# l = {x:getattr(mod,x) for x in targets}
	# return {k:l[k] for k in targets}
	# path = get_cache_path(name,hsh)




if(__name__ == "__main__"):
	from numba import njit, jit
	from cre.operator import BaseOperator
	from cre.gensource import *
	import numpy as np
	import importlib

	class PoopyMeta(type, _UniqueHashable):
		def get_hashable(cls):
			d = {k: v for k,v in vars(cls).items() if not k in ['get_hashable','__weakref__','__doc__','__dict__']}
			print("WHEE",d)
			return d

	class POOPY(metaclass=PoopyMeta):
		crabs = 1
		def shloopy():
			return 1+2

		# @classmethod
		



	print(POOPY.get_hashable())
	# print(Add.get_hashable())

	def myfunc():
		pass

	print(myfunc.__code__.co_code)
	# myfunc.__code__.__text_signature__



	N=1000
	def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))


	#Basic tests
	obj = {"A":1,"B":2,"C":3}
	obj2 = ["A",1,"B",2,"C",3]
	obj3 = ["A",myfunc,"B",2,"C",3]
	obj4 = ["A",POOPY.shloopy,"B",2,"C",3]
	obj4 = ["A",POOPY,"B",2,"C",3]
	# obj4 = ["A",Add,"B",2,"C",3]


	print(unique_hash(obj))
	print(unique_hash(obj2))
	print(unique_hash(obj3))
	print(unique_hash(obj4))

	source = '@njit'
	print(hash(source))

	hash_code = POOPY.get_hashcode()

	print(hash_code)

	print(get_cache_path('hello',hash_code))
	source_to_cache("hello", hash_code,source)

	print(source_in_cache("hello", hash_code))
	print(source_from_cache("hello", hash_code))

	#Start op test
	class Floop(BaseOperator):
		signature = 'float(float,float)'
		def condition(x, y):
			return x > y and y > 0.0
		def forward(x, y):
			return x + y

	hash_code = Floop.get_hashcode()
	source = gen_source_broadcast_forward(Floop,True)

	print(get_cache_path('Floop',hash_code))
	source_to_cache("Floop", hash_code,source)

	print(source_in_cache("Floop", hash_code))
	print(source_from_cache("Floop", hash_code))


	print(hash_code)	
	print(source)

	# f = njit(Floop.forward)
	# c = njit(Floop.condition)
	# g = {'f': f, "c" : c, "jit" : jit}
	# l = {}

	# cache_path = get_cache_path('Floop',hash_code)
	# spec = importlib.util.spec_from_file_location("cre.cache.type1", cache_path)
	# my_mod = importlib.util.module_from_spec(spec)
	# spec.loader.exec_module(my_mod)


	# from cre_cache.Floop._36e47400043b53379e884eddd923dff7aa3f52cc5ab38f3a3f42c5eaa01d942e import Floop_forward
	# exec("from cre_cache.Floop._36e47400043b53379e884eddd923dff7aa3f52cc5ab38f3a3f42c5eaa01d942e import Floop_forward", g, l)

	# l['Floop_forward'](np.array([1.0,2.0]))

	spec = {
		"x" : "number",
		"y" : "number",
	}
	source = gen_source_standard_imports()
	source += gen_source_tuple_defs("Point",spec)
	source += gen_source_get_enumerized("Point",spec)
	source += gen_source_enumerize_nb_objs("Point",spec)
	source += gen_source_pack_from_numpy("Point",spec)
	# print(source)
	
	hash_code = unique_hash(["Point",spec])
	source_to_cache("Point",hash_code,source)
	# print(hash_code)
	print(source_from_cache("Point", hash_code))
	d = import_from_cached("Point",hash_code,["Point"])
	# exec("from cre_cache.Point._b666f2d9dfe40302fff737e562dc999fb89e3cd4be6f2f0200875eadf7286613 import Point", {}, l)
	print(d['Point'])







