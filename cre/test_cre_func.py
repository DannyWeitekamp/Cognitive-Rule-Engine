from numba import generated_jit, njit, i8
from cre.cre_func import CREFunc
from cre.cre_object import _get_chr_mbrs_infos_from_attrs, _iter_mbr_infos
from cre.fact import define_fact
from cre.var import Var

# BOOP = define_fact("BOOP", {"A" :i8, "B" :i8})


# @njit
# def foo():
#     b = BOOP(0,1)
#     return _get_chr_mbrs_infos_from_attrs(b, ("A",))

# print(foo())


if __name__ == "__main__":
    # @global_func("Add_poop", i8(i8,i8))

    @CREFunc(signature=i8(i8,i8,i8,i8))
    def Add(a, b, c, d):
        return a + b + c +d
 
    @generated_jit
    def foo(cf):
        print(cf)
        def impl(cf):
            print(cf.h0, cf.h1)
            # for tup in _iter_mbr_infos(cf):
            #     print(tup)
        return impl

    print(Add)
    a = Var(i8,'a')
    b = Var(i8,'b')
    c = Var(i8,'c')

    z = Add(a,b,c,c)
    print("OUT", z(1,2,3))

    q = z(c,c,c)
    print("OUT", q(7))    
    foo(Add)


