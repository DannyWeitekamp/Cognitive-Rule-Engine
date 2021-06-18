from numba import f8, njit
from numba.extending import intrinsic
from cre.op import Op
from cre.fact import define_fact
import time

BOOP, BOOPType = define_fact("BOOP", {"A" : f8, "B": f8})

print(f8(f8,f8))
class Add(Op):
    signature = f8(f8,f8)
    def check(a,b):
        return a > 0
    def call(a,b):
        return a + b

print(f8(f8,f8))
class SumBOOPs(Op):
    signature = BOOPType(BOOPType,BOOPType)
    def check(a,b):
        return a.A > 0
    def call(a,b):
        return BOOP(a.A + b.A, a.B + b.B)


print(Add.call)
print(Add.call(1,2))
print(Add.check(1,2))
print(Add(1,2))

print("----")
print(SumBOOPs.call(BOOP(1,2),BOOP(3,4)))
print(SumBOOPs.check(BOOP(1,2),BOOP(3,4)))
print(SumBOOPs(BOOP(1,2),BOOP(3,4)))


time1 = time.time_ns()/float(1e6)
    
@njit(f8(f8,f8),cache=True)
def add(a,b):
    return a + b

@intrinsic
def intr_add(typingctx, a, b):
    sig = f8(f8,f8)
    def codegen(context, builder, sig, args):
        [a0,a1] = args
        fndesc = add.overloads[(f8,f8)].fndesc
        ret = context.call_internal(builder, fndesc, sig, (a0,a1))
        return ret
    return sig, codegen


@njit(cache=True)
def from_intr_add(a,b):
    return intr_add(a,b)

print(from_intr_add(1,2))
time2 = time.time_ns()/float(1e6)
print("%s: %.4f ms" % ("from_intr_add:", time2-time1))

time2 = time.time_ns()/float(1e6)
@intrinsic
def intr_add_one(typingctx, a):
    sig = f8(f8,)
    def codegen(context, builder, sig, args):
        [a0,] = args
        fndesc = add.overloads[(f8,f8)].fndesc
        one = context.get_constant_generic(builder,f8,1.0)
        ret = context.call_internal(builder, fndesc, f8(f8,f8), (a0,one))
        return ret
    return sig, codegen


@njit(cache=True)
def from_intr_add_one(a):
    return intr_add_one(a)

time3 = time.time_ns()/float(1e6)
print("%s: %.4f ms" % ("from_intr_add_one:", time3-time2))

print(from_intr_add_one(4))

