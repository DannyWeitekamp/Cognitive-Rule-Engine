from numba import types, njit, jit, prange
from numbert.operator import BaseOperator
import numpy as np
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




