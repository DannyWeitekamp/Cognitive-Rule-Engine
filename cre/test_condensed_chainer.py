from numba import f8
from cre.op import Op
from cre.condensed_chainer import gen_apply_multi_source, apply_multi, c_chainer_ctor

class Add(Op):
    signature = f8(f8,f8)        
    short_hand = '({0}+{1})'
    def check(a, b):
        return a > 0
    def call(a, b):
        return a + b

class Multiply(Op):
    signature = f8(f8,f8)
    short_hand = '({0}*{1})'
    def check(a, b):
        return b != 0
    def call(a, b):
        return a * b  

planner = c_chainer_ctor()

print(gen_apply_multi_source(Add))

apply_multi(Add,planner, 0)
