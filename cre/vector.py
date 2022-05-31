import operator
from numba import types, njit, u1,u2,u4,u8, i8, carray
from numba.types import Tuple
from numba.typed import List
from numba.extending import overload_method, overload
from numba.experimental.structref import new
from cre.structref import define_structref_template
from llvmlite import ir
import numpy as np



vector_fields = [
    ("head", i8),
    ("data", i8[::1])
]

Vector, VectorTypeTemplate = define_structref_template("Vector",vector_fields)
VectorType = VectorTypeTemplate(fields=vector_fields)

@njit(cache=True)
def new_vector(size):
    '''
    Makes a new vector of size.
    '''
    st = new(VectorType)
    st.head = 0
    st.data = np.zeros(size,dtype=np.int64)
    return st

@njit(cache=True)
def _expand_to(self,size):
    if(len(self.data) < size):
        new_data = np.empty(size, dtype=np.int64)
        new_data[:len(self.data)] = self.data
        new_data[len(self.data):-1] = 0
        self.data = new_data

@overload_method(VectorTypeTemplate, "assert_size")
def assert_size(self,size):
    def impl(self,size):
        _expand_to(self,size)
    return impl

@njit(inline='never')
def _expand(self):
    '''
    Doubles the size of the vector. For whatever reason it runs faster when this
    is a separate function. Probably prevents some lookahead execution gone wrong.
    '''
    new_data = np.zeros(len(self.data)*2, dtype=np.int64)
    new_data[:len(self.data)] = self.data
    new_data[len(self.data):-1] = 0
    self.data = new_data

@overload_method(VectorTypeTemplate, "expand")
def assert_size(self):
    def impl(self):
        _expand(self)
    return impl

@overload_method(VectorTypeTemplate, "set_item_safe")
def vector_set_item_safe(self, i, x):
    ''' 
    Set's slot i to x. Ensures that data is large enough.
    '''
    def impl(self, i, x):
        if(i >= self.head): self.head = i+1
        _expand_to(self,self.head)
        self.data[i] = x
    return impl

@overload_method(VectorTypeTemplate, "add")
def vector_add(self, x):
    ''' 
    Adds an item to the vector and increments the head.
    '''
    def impl(self, x):
        if(self.head >= len(self.data)): _expand(self)
        self.data[self.head] = x
        self.head += 1
    return impl

@overload_method(VectorTypeTemplate, "pop")
def vector_pop(self):
    ''' 
    Returns the last item from the end of the vector and decrements the head.
    Returns zero if the vector is empty (does not raise an error).
    '''
    def impl(self):
        if(not self.head): return 0
        self.head -= 1
        return self.data[self.head]
    return impl



@overload_method(VectorTypeTemplate, "clear")
def vector_clear(self):
    ''' 
    Moves the head to zero, essentially clearing the vector.
    '''
    def impl(self):
        self.head = 0
    return impl

@overload_method(VectorTypeTemplate, "copy")
def vector_copy(self):
    ''' 
    Moves the head to zero, essentially clearing the vector.
    '''
    def impl(self):
        st = new(VectorType)
        st.head = self.head
        st.data = self.data.copy()
    return impl

@overload(operator.getitem)
def impl_getitem(self, i):
    if not isinstance(self, VectorTypeTemplate):
        return
    def impl(self,i):
        return self.data[i]
    return impl

@overload(operator.setitem)
def impl_setitem(self, i, x):
    if not isinstance(self, VectorTypeTemplate):
        return
    def impl(self,i,x):
        self.data[i] = x
    return impl

@overload(len)
def impl_len(self):
    if not isinstance(self, VectorTypeTemplate):
        return
    def impl(self):
        return self.head
    return impl






import timeit
N=100
def time_ms(f):
    f() #warm start
    return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))



@njit
def test_vector():
    v = new_vector(10)
    for i in range(100):
        v.add(i)
        v[0] = i
        print(v[0],v[i],v.head,len(v.data))
    for i in range(50):
        print(v.pop())
    v.clear()
    print("H",v.head)
    print("L",len(v))

# test_vector()

@njit
def add_pop():
    v = new_vector(10000)
    for i in range(10000):
        v.add(i)
    for i in range(10000):
        v.pop()

@njit
def expand(self):
    # new_size = self.size*2
    new_data = np.empty(len(self.data)*2, dtype=np.int64)
    new_data[:len(self.data)] = self.data
    self.data = new_data
    # self.size = new_size

@njit
def dd(self, x):
    if(self.head >= len(self.data)): expand(self)
    self.data[self.head] = x
    self.head += 1

@njit
def add_pop_dd():
    v = new_vector(10000)
    for i in range(10000):
        dd(v,i)
    for i in range(10000):
        v.pop()


@njit
def add_pop_manual():
    v = new_vector(10000)
    for i in range(10000):
        v.data[i] = i
        v.head += i
        # v.add(i)
    for i in range(10000):
        v.pop()


@njit
def add_pop_inline():
    v = np.empty((10000,),dtype=np.int64)
    h = 0
    d = 0
    for i in range(10000):
        v[i] = i*i
        h += 1
    for i in range(10000):
        h -= 1
        d = v[h]





@njit
def v_new(size):
    v = np.empty((size+1,),dtype=np.int64)
    v[0] = 1
    return v

@njit
def v_expand(v):
    n_v = np.empty(len(v)*2, dtype=np.int64)
    n_v[:len(v)] = v
    return n_v

@njit
def v_add(v,x):
    if(v[0] > len(v)): v = v_expand(v)
    v[v[0]] = x 
    v[0] += 1
    return v

@njit
def v_pop(v):
    if(v[0] == 1): return 0
    v[0] -= 1
    return v[v[0]]


@njit
def add_pop_just_arr():
    v = v_new(10000)
    d = 0
    for i in range(10000):
        v_add(v,i)
    for i in range(10000):
        d = v_pop(v)

def add_pop_python_list():
    v = []
    for i in range(10000):
        v.append(i)

def add_pop_numpy_list():
    v = np.empty(10000, dtype=np.int64)
    for i in range(10000):
        v[i] = i

    # for i in range(10000):
    #     d = v_pop(v)
# test_vector()
# print(time_ms(add_pop))
# print(time_ms(add_pop_dd))
# print(time_ms(add_pop_manual))
# print(time_ms(add_pop_inline))
# print(time_ms(add_pop_just_arr))
# print(time_ms(add_pop_python_list))
# print(time_ms(add_pop_numpy_list))




# test_vector()




