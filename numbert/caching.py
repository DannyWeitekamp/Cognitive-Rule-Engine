
import sys,os
#Snatch AppDirs from numba numba and find the cache dir
from numba.misc.appdirs import AppDirs
appdirs = AppDirs(appname="numbert", appauthor=False)
cache_dir = appdirs.user_cache_dir


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




source = 'print("Hello WORLD")'
print(hash(source))
print(get_cache_path('hello',hash(source)))
source_to_cache("hello", hash(source),source)

print(source_in_cache("hello", hash(source)))
print(source_from_cache("hello", hash(source)))

