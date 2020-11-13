import numpy as np

from numba import njit
from numba.core import types
from numba.experimental import structref

from numba.tests.support import skip_unless_scipy
from numba.pycc import CC
from numba import i8, f8
from numba.core.types import DictType, unicode_type
from numba.typed import List, Dict
from numba.types import unicode_type
cc = CC('my_module')





# Define a StructRef.
# `structref.register` associates the type with the default data model.
# This will also install getters and setters to the fields of
# the StructRef.
@structref.register
class MyStructType(types.StructRef):
	def preprocess_fields(self, fields):
		# This method is called by the type constructor for additional
		# preprocessing on the fields.
		# Here, we don't want the struct to take Literal types.
		return tuple((name, types.unliteral(typ)) for name, typ in fields)


mst = MyStructType( [
		('name', unicode_type),
		('vector', f8[:])
	])

print(mst)
print(mst.get_data_type())

@cc.export("MyStruct_get_name", unicode_type(mst,))
@njit
def MyStruct_get_name(self):
    # In jit-code, the StructRef's attribute is exposed via
    # structref.register
    return self.name

@cc.export("MyStruct_get_vector", f8[:](mst,))
@njit
def MyStruct_get_vector(self):
    return self.vector


# alice = mst("Alice", np.zeros(3))


# cc.compile()

from my_module import MyStruct_get_name, MyStruct_get_vector



# Define a Python type that can be use as a proxy to the StructRef
# allocated inside Numba. Users can construct the StructRef via
# the constructor for this type in python code and jit-code.
class MyStruct(structref.StructRefProxy):
    def __new__(cls, name, vector):
        # Overriding the __new__ method is optional, doing so
        # allows Python code to use keyword arguments,
        # or add other customized behavior.
        # The default __new__ takes `*args`.
        # IMPORTANT: Users should not override __init__.
        return structref.StructRefProxy.__new__(cls, name, vector)

    # By default, the proxy type does not reflect the attributes or
    # methods to the Python side. It is up to users to define
    # these. (This may be automated in the future.)

    @property
    def name(self):
        # To access a field, we can define a function that simply
        # return the field in jit-code.
        # The definition of MyStruct_get_name is shown later.
        return MyStruct_get_name(self)

    @property
    def vector(self):
        # The definition of MyStruct_get_vector is shown later.
        return MyStruct_get_vector(self)

# @njit
# def MyStruct_get_name(self):
#     return self.name

# # @cc.export("MyStruct_get_vector", f8[:](mst.get_data_type(),))
# @njit
# def MyStruct_get_vector(self):
#     return self.vector



# This associates the proxy with MyStructType for the given set of
# fields. Notice how we are not contraining the type of each field.
# Field types remain generic.
structref.define_proxy(MyStruct, MyStructType, ["name", "vector"])


# Let's test our new StructRef.

# Define one in Python
alice = mst("Alice", np.zeros(3))

# Define one in jit-code
@njit
def make_bob():
    bob = mst("unnamed", np.zeros(3))
    # Mutate the attributes
    bob.name = "Bob"
    bob.vector = np.random.random(3)
    bob.vector /= np.sqrt(np.sum(bob.vector**2))
    # print(type(bob))
    return bob

bob = make_bob()

# Out: Alice: [0.5488135  0.71518937 0.60276338]
print(f"{alice.name}: {alice.vector}")
# Out: Bob: [0.88325739 0.73527629 0.87746707]
print(f"{bob.name}: {bob.vector}")

# Define a jit function to operate on the structs.
@njit
def distance(a, b):
    return np.linalg.norm(a.vector - b.vector)

# Out: 0.4332647200356598
print(distance(alice, bob))
