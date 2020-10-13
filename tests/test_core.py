from numbert import core

import numpy as np
from numba import njit
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import ListType, DictType, unicode_type, Array, Tuple
from pprint import pprint


from numbert.knowledgebase import NBRT_KnowledgeBase, how_search, forward
from example_ops import * #Grumbo_forward1, Grumbo_forward2, BaseOperator,NBRT_KnowledgeBase,\
	 #Add, Subtract, Concatenate, StrToFloat, \
	  #how_search, forward






def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))



# kb = NBRT_KnowledgeBase()
# kb.declare(1)
# kb.declare("1")


X = np.array(np.arange(10),np.float32)
y = 17
# forward(X,y,[Add,Multiply])
# print("OUT",Add_forward1(X,X))
# print(Add_forward2(X,X))

@njit(nogil=True,fastmath=True,cache=True) 
def make_S(offset):
	S = List.empty_list(unicode_type)
	for x in range(48+offset,48+offset+5):
	# for x in range(97+offset,97+offset+5):
		S.append("1"+chr(x))
	S.append("a")
	return S#np.array(S)
S1 = make_S(0)
S2 = make_S(5)


@njit(cache=True)
def newVmap(X,typ):
	d = Dict.empty(typ,i8)
	for x in X:
		d[x] = 0
	return d

def buildKB():
	kb = NBRT_KnowledgeBase()

	for s in S1:
		kb.declare(s)

	for n in X:
		kb.declare(n.item())



	# kb.u_vs['unicode_type'] = S1
	# kb.u_vs['f8'] = X

	# kb.u_vds['unicode_type'] = newVmap(S1,unicode_type)
	# kb.u_vds['f8'] = newVmap(X,f8)

	# for s in S1:
	# 	kb.u_vds['unicode_type'][s] = 0
	# for x in X:
	# 	kb.u_vds['f8'][x] = 0

	return kb

print("MID")
kb = buildKB()
print("MID2")
# print("STAAAAART")
# for op in [Add,Subtract,Concatenate]:
# 	print(op, op.uid)
# for op in [Add,Subtract,Concatenate]:
# 	print(op, op.uid)
# print(BaseOperator.operators_by_uid)
# forward(kb,[Add,Subtract,Concatenate])
# forward(kb,[Add,Subtract,Concatenate])
import time

kb = buildKB()
start = time.time()
print("HOW RESULTS")
results = how_search(kb,[Add,Subtract],37,search_depth=3)
end = time.time()
# pprint(results)
print("Time elapsed: ",end - start)


kb = buildKB()
start = time.time()
print("HOW RESULTS")
results = how_search(kb,[Add,Subtract],37,search_depth=3)
end = time.time()
pprint(results)
print("Time elapsed: ",end - start)

# print(Add.broadcast_forward.stats.cache_hits)


kb = buildKB()
start = time.time()
how_search(kb,[Concatenate],"abab",1)
end = time.time()
print("Time elapsed: ",end - start)


kb = buildKB()
start = time.time()
print("MOOSE PIMPLE")
forward(kb,[SquaresOfPrimes])
print(kb.u_vs)
# how_search(kb,[HalfOfEven],100,1)
end = time.time()
print("Time elapsed: ",end - start)

# kb = buildKB()
# start = time.time()
# how_search(kb,[Add,Subtract,Concatenate],"abab",2)
# end = time.time()
# print("Time elapsed: ",end - start)

# forward(kb,[Add,Subtract,Concatenate])
# for typ in kb.registered_types:
# 	print(typ)
# 	print(kb.u_vs[typ])

# print(cat_forward1(S))

@njit(nogil=True,fastmath=True,cache=True) 
def g1():
	# uid = 1; d = Dict.empty(f8,i8)
	# Add_forward1(uid,d, X,X)	
	Grumbo_forward1(X,X)	

@njit(nogil=True,fastmath=True,cache=True) 
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

# d  =Dict()
# d['unicode_type'] = unicode_type
# d['f8'] = f8
# print("g1", time_ms(g1))
# print("g2", time_ms(g2))
# print("s1", time_ms(s1))
# print("forward-depth-5", time_ms(f1))
# print("search-21-depth-2", time_ms(h1))
