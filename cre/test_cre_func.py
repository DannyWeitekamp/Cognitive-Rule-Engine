from numba import generated_jit, njit, i8
from numba.types import unicode_type, FunctionType
from cre.cre_func import CREFunc
from cre.cre_object import _get_chr_mbrs_infos_from_attrs, _iter_mbr_infos
from cre.fact import define_fact
from cre.var import Var
from cre.utils import PrintElapse

from numba.core.runtime.nrt import rtsys
def used_bytes():
    stats = rtsys.get_allocation_stats()
    return stats.alloc-stats.free

# BOOP = define_fact("BOOP", {"A" :i8, "B" :i8})


# @njit
# def foo():
#     b = BOOP(0,1)
#     return _get_chr_mbrs_infos_from_attrs(b, ("A",))

# print(foo())


if __name__ == "__main__":
    import faulthandler; faulthandler.enable()

    # @CREFunc(signature=i8(i8,i8,i8,i8))
    # def Add(a, b, c, d):
    #     return a + b + c +d

    # a = Var(i8,'a')
    # b = Var(i8,'b')
    # c = Var(i8,'c')

    # z = Add(a,b,c,c)
    # assert z(1,2,3) == 9
    # q = z(c,c,c)
    # assert(q(7)==28)

    # print("?", 1+2+3+1+1+3+3)
    # print("------------")
    # print(Add(a,Add(a,b,c,a),c,c)(1,2,3))
    # print("------------")
    # print(Add(Add(Add(2,1,1,a),2,1,Add(2,1,1,b)),1,1,c)(1,2,3))
    # z = Add(Add(Add(2,1,1,a),2,1,Add(2,1,1,b)),1,1,c)
    # print(z(1,2,3))


    # @CREFunc(signature=unicode_type(unicode_type,unicode_type),
    #         shorthand='{0}+{1}')
    # def Concat(a, b):
    #     return a + b
    
    # a = Var(unicode_type,'a')
    # b = Var(unicode_type,'b')
    # c = Var(unicode_type,'c')

    # for i in range(2):
    #     z = Concat(a, Concat(Concat(b,c),a ))
    #     print(z)
    #     print(z("|","X","Y"))
    #     if(i == 0):
    #         init_bytes = used_bytes()
    #     else:
    #         assert used_bytes() == init_bytes


    # BOOP = define_fact("BOOP", {"A" :unicode_type, "B" :i8})

    # @CREFunc(signature=BOOP(BOOP,BOOP),
    #         shorthand='{0}+{1}')
    # def Smerpify(a, b):
    #     return BOOP(a.A + b.A, a.B + b.B)

    # a = Var(BOOP,'a')
    # b = Var(BOOP,'b')
    # c = Var(BOOP,'c')

    # z = Smerpify(a,b)
    # print(z)
    # # print(z(BOOP("A",1),BOOP("B",2)))

    # print("<------------------->")

    # print()

    # z = Concat(a.A, b.A)
    # ba, bb = BOOP("A",1), BOOP("B",2)

    # with PrintElapse("Z"):
    #     z(ba,bb)
    # with PrintElapse("Z"):
    #     z(ba,bb)


    @CREFunc(signature=i8(i8,i8))
    def Add(a, b):
        return a + b

    from cre.cre_func import (set_base_arg_val_impl, _func_from_address,
        cre_func_call_self, get_str_return_val_impl, call_self_f_type)

    @njit(cache=True)
    def baz(op):
        z = 0
        for i in range(1000):
            for j in range(1000):
                z += op(i, j)
        return z

    a = Var(i8,'a')
    b = Var(i8,'b')
    comp = Add(Add(0,b),a)

    print(Add._type.name)
    print(comp._type.name)

    baz(Add)
    baz(comp)

    with PrintElapse("baz"):
        print(baz(Add))

    with PrintElapse("baz comp"):
        print(baz(comp))

    # _func_from_address(call_self_f_type, self.call_self_addr)(self)

    i8_ft = FunctionType(i8(i8,i8))

    set_base = set_base_arg_val_impl(0)
    ret_impl = get_str_return_val_impl(i8)


    @njit(cache=True)
    def foo(op):
        z = 0
        for i in range(1000):
            for j in range(1000):
                set_base(op, 0, i)
                set_base(op, 1, j)
                cre_func_call_self(op)
                z += ret_impl(op)
        return z

    @njit(cache=True)
    def foo_fast(op):
        z = 0
        # f = _func_from_address(i8_ft, op.call_heads_addr)
        for i in range(1000):
            for j in range(1000):
                f = _func_from_address(i8_ft,op.call_heads_addr)
                z += f(i,j)
                
        return z

    @njit(i8(i8,i8), cache=True)
    def njit_add(a,b):
        return a + b

    @njit(i8(FunctionType(i8(i8,i8))), cache=True)
    def bar(op):
        z = 0
        for i in range(1000):
            for j in range(1000):
                z += op(i, j)
        return z

    @njit(i8(), cache=True)
    def inline():
        z = 0
        for i in range(1000):
            for j in range(1000):
                z += njit_add(i, j)
        return z

    foo(Add)
    foo_fast(Add)
    bar(njit_add)
    inline()

    with PrintElapse("foo"):
        foo(Add)

    with PrintElapse("foo fast"):
        foo_fast(Add)

    with PrintElapse("foo python"):
        foo.py_func(Add)
        

    with PrintElapse("bar"):
        bar(njit_add)

    with PrintElapse("bar  python"):
        bar.py_func(njit_add)


    with PrintElapse("inline"):
        inline()




    








