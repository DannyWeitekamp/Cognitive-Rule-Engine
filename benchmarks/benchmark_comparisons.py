import numpy as np
from numba.types import unicode_type
from numba.typed import List, Dict
from numba import njit, f8, i8
import timeit

N=1000
def time_ms(f):
	f() #warm start
	return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))

str_list = List.empty_list(unicode_type)
int_list = List.empty_list(i8)
for i in range(100):
	str_list.append("medium_str{}".format(i))
	int_list.append(i)

int_arr = np.arange(100,dtype=np.int64)


@njit
def cross_eq(X):
	m = np.empty((len(X),len(X)),dtype=np.uint8)
	for i, x in enumerate(X):
		for j, y in enumerate(X):
			m[i,j] = (x == y)
	return m

@njit
def enumerize(X,d):
	m = np.empty((len(X),len(X)),dtype=np.int64)
	for i, x in enumerate(X):
		if(x not in d):
			d[x] = len(d)
		m[i] = d[x]
	return m



def enumerize_str_list():
	d = Dict.empty(unicode_type,i8)
	enumerize(str_list,d)

def cross_str_list():
	cross_eq(str_list)

def cross_int_list():
	cross_eq(int_list)

def cross_int_arr():
	cross_eq(int_arr)


print("enumerize_str_list: ", time_ms(enumerize_str_list))
print("cross_str_list: ", time_ms(cross_str_list))
print("cross_int_list: ", time_ms(cross_int_list))
print("cross_int_arr: ", time_ms(cross_int_arr))







