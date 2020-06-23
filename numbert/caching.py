
import sys,os
import hashlib 
import timeit
from types import FunctionType
#Snatch AppDirs from numba numba and find the cache dir
from numba.misc.appdirs import AppDirs

# from numbert.core import Add
appdirs = AppDirs(appname="numbert", appauthor=False)
cache_dir = appdirs.user_cache_dir


class _UniqueHashable():
	@classmethod
	def get_hashable(cls):
		raise NotImplemented()


def update_unique_hash(m,obj):
	if(isinstance(obj,str)):
		m.update(obj.encode('utf-8'))
	elif(isinstance(obj,(tuple,list))):
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


def unique_hash(stuff,hash_func='sha256'):
	m = hashlib.new(hash_func)
	update_unique_hash(m,stuff)	
	return m.hexdigest()

def get_cache_path(name,hsh):
	return os.path.join(cache_dir,name,str(hsh) +".py")

def source_in_cache(name,hsh):
	path = get_cache_path(name,hsh)
	return path if os.path.exists(path) else None


def source_from_cache(name,hsh):
	path = get_cache_path(name,hsh)
	with open(path,mode='r') as f:
		out = f.read()
	return out

def source_to_cache(name,hsh,source):
	path = get_cache_path(name,hsh)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path,mode='w') as f:
		f.write(source)


# def hash_sha1(stuff):
# 	m = sha1()
# 	m.update(stuff.encode('utf-8'))
# 	return m.hexdigest()

class POOPY(_UniqueHashable):
	crabs = 1
	def shloopy():
		return 1+2

	@classmethod
	def get_hashable(cls):
		d = {k: v for k,v in vars(cls).items() if not k in ['__weakref__','__doc__','__dict__']}
		print("WHEE")
		return d



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

print(get_cache_path('hello',hash(source)))
source_to_cache("hello", hash(source),source)

print(source_in_cache("hello", hash(source)))
print(source_from_cache("hello", hash(source)))

