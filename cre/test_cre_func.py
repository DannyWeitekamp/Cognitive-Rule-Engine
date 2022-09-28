from numba import generated_jit, njit, i8
from cre.cre_func import CREFunc
from cre.cre_object import _get_chr_mbrs_infos_from_attrs, _iter_mbr_infos
from cre.fact import define_fact

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
        #hi
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

    print("OUT", Add(1,2))
    foo(Add)


