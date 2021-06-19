from numba import f8, njit
from numba.extending import intrinsic
from cre.op import Op, OpComp
from cre.fact import define_fact
from cre.var import Var
from cre.caching import cache_safe_exec
import time


BOOP, BOOPType = define_fact("BOOP", {"A" : f8, "B": f8})

time1 = time.time_ns()/float(1e6)
print(f8(f8,f8))
class Add(Op):
    signature = f8(f8,f8)
    def check(a,b):
        return a > 0
    def call(a,b):
        return a + b
time2 = time.time_ns()/float(1e6)
print("%s: %.4f ms" % ("def Add:", time2-time1))

v1 = Var(float,'v1')
v2 = Var(float,'v2')
a = Add(v1,v2)
# print(a(1))


time1 = time.time_ns()/float(1e6)
class SumBOOPs(Op):
    signature = BOOPType(BOOPType,BOOPType)
    def check(a,b):
        return a.A > 0
    def call(a,b):
        return BOOP(a.A + b.A, a.B + b.B)
time2 = time.time_ns()/float(1e6)
print("%s: %.4f ms" % ("def SumBOOPs:", time2-time1))

print(Add.call)
print(Add.call(1,2))
print(Add.check(1,2))
print(Add(1,2))

print("----")
print(SumBOOPs.call(BOOP(1,2),BOOP(3,4)))
print(SumBOOPs.check(BOOP(1,2),BOOP(3,4)))
print(SumBOOPs(BOOP(1,2),BOOP(3,4)))


print(Add(Var(float),Add(Var(float),2)))
# print()

oc = Add(Var(float),Add(Var(float),2))


instrs = list(oc.instructions)
sig = f8(f8,f8)

from numba import float64, int64, njit
from numba.types import unicode_type
from numba.extending import intrinsic
import pickle

time1 = time.time_ns()/float(1e6)
fndesc0 = pickle.loads(pickle.dumps(instrs[0].op.call_fndesc))
time2 = time.time_ns()/float(1e6)
print("%s: %.4f ms" % ("unplickle:", time2-time1))

def codegen(context, builder, sig, args):
    [v0,v1] = args
    c0 = context.get_constant_generic(builder, float64, 2)
    o0 = instrs[0].op
    i0 = context.call_internal(builder, fndesc0, o0.signature, (v1, c0))
    o1 = instrs[1].op
    i1 = context.call_internal(builder, o1.call_fndesc, o1.signature, (v0, i0))

    return i1

@intrinsic
def call_intr(ctx, v0,v1):
    return sig, codegen

time1 = time.time_ns()/float(1e6)

def gen_call():
    @njit(sig,cache=True)
    def call(v0,v1):
        return call_intr(v0,v1)
    return call
call = gen_call()
print(call(1,2))
time2 = time.time_ns()/float(1e6)
print("%s: %.4f ms" % ("poop:", time2-time1))

# g = {"instrs": list(oc.instructions),"sig":f8(f8,f8)}
# l = {}
# source = oc.gen_intrinsic_source()
# print(source)
# exec(source,g,l)
# call = l['call']




# print(oc.gen_intrinsic_source())

# print(OpComp(Add,1,2))
# print(OpComp(Add,Var(float),2))
# print(OpComp(Add,Var(float),OpComp(Add,Var(float),2)))


# time1 = time.time_ns()/float(1e6)
    
# @njit(f8(f8,f8),cache=True)
# def add(a,b):
#     return a + b

# @intrinsic
# def intr_add(typingctx, a, b):
#     sig = f8(f8,f8)
#     def codegen(context, builder, sig, args):
#         [a0,a1] = args
#         fndesc = add.overloads[(f8,f8)].fndesc
#         ret = context.call_internal(builder, fndesc, sig, (a0,a1))
#         return ret
#     return sig, codegen


# @njit(cache=True)
# def from_intr_add(a,b):
#     return intr_add(a,b)

# print(from_intr_add(1,2))
# time2 = time.time_ns()/float(1e6)
# print("%s: %.4f ms" % ("from_intr_add:", time2-time1))

# time2 = time.time_ns()/float(1e6)
# @intrinsic
# def intr_add_one(typingctx, a):
#     sig = f8(f8,)
#     def codegen(context, builder, sig, args):
#         [a0,] = args
#         fndesc = add.overloads[(f8,f8)].fndesc
#         one = context.get_constant_generic(builder,f8,1.0)
#         ret = context.call_internal(builder, fndesc, f8(f8,f8), (a0,one))
#         return ret
#     return sig, codegen


# @njit(cache=True)
# def from_intr_add_one(a):
#     return intr_add_one(a)

# time3 = time.time_ns()/float(1e6)
# print("%s: %.4f ms" % ("from_intr_add_one:", time3-time2))

# print(from_intr_add_one(4))
# time4 = time.time_ns()/float(1e6)
# print("%s: %.4f ms" % ("call from_intr_add_one:", time4-time3))



# class AddToBOOP(Op):
#     signature = BOOPType(BOOPType,f8)
#     def call(a,b):
#         return BOOP(a.A + b, a.B + b)

# print("TTTTT", isinstance(Var(BOOPType),Var))
# time1 = time.time_ns()/float(1e6)
# ab = AddToBOOP(Var(BOOPType), 1)
# time2 = time.time_ns()/float(1e6)
# print(ab(BOOP(1,2)))
# time3 = time.time_ns()/float(1e6)
# print(ab(BOOP(1,2)))
# print(ab(BOOP(1,2)))
# print(ab(BOOP(1,2)))
# print(ab(BOOP(1,2)))
# time4 = time.time_ns()/float(1e6)

# print("%s: %.4f ms" % ("instantiate:", time2-time1))
# print("%s: %.4f ms" % ("call 1st:", time3-time2))
# print("%s: %.4f ms" % ("call 2nd:", time4-time3))





# time1 = time.time_ns()/float(1e6)
# inc_boop_intr = AddToBOOP.gen_intrinsic((Var(BOOPType), 1),'call')
# time2 = time.time_ns()/float(1e6)
# print("%s: %.4f ms" % ("gen_intrinsic:", time2-time1))
# # print()

# time1 = time.time_ns()/float(1e6)
# @njit(cache=False)
# def inc_boop(a):
#     return inc_boop_intr(a)

# print(inc_boop(BOOP(1,2)))
# time2 = time.time_ns()/float(1e6)
# print("%s: %.4f ms" % ("call_gened_intr:", time2-time1))


# time1 = time.time_ns()/float(1e6)
# inc_boop(BOOP(1,2))
# time2 = time.time_ns()/float(1e6)
# print("%s: %.4f ms" % ("call_gened_intr:", time2-time1))
# @intrinsic
# def multi_arg_intr(typingctx, *args_typs):
#     sig = f8(*args_typs)
#     def codegen(context, builder, sig, args):
#         return context.get_constant_generic(builder,f8,1.0)
#     return sig, codegen
# @njit(cache=True)
# def mult_arg():
#     return multi_arg_intr(1,2,3,"4")

# print(mult_arg())
