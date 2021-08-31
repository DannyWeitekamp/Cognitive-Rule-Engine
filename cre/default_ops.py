from cre.op import Op


@Op(shorthand = '({0} == {1})', commutes=True)
def Equals(a, b):
    return a == b

@Op(shorthand = '({0} < {1})')
def LessThan(a, b):
    return a < b

@Op(shorthand = '({0} <= {1})')
def LessThanEq(a, b):
    return a <= b


@Op(shorthand = '({0} < {1})')
def GreaterThan(a, b):
    return a < b

@Op(shorthand = '({0} <= {1})')
def GreaterThanEq(a, b):
    return a <= b


@Op(shorthand = '({0} + {1})', commutes=True)
def Add(a, b):
    return a + b

@Op(shorthand = '({0} - {1})')
def Subtract(a, b):
    return a - b

@Op(shorthand = '({0} * {1})', commutes=True)
def Multiply(a, b):
    return a * b

def denom_not_zero(a,b):
    return b != 0

@Op(shorthand = '({0} / {1})', check=denom_not_zero)
def Divide(a, b):
    return a / b




