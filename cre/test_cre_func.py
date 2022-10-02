from numba import generated_jit, njit, i8
from numba.types import unicode_type
from cre.cre_func import CREFunc
from cre.cre_object import _get_chr_mbrs_infos_from_attrs, _iter_mbr_infos
from cre.fact import define_fact
from cre.var import Var

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

    @CREFunc(signature=i8(i8,i8,i8,i8))
    def Add(a, b, c, d):
        return a + b + c +d
    a = Var(i8,'a')
    b = Var(i8,'b')
    c = Var(i8,'c')

    z = Add(a,b,c,c)
    assert z(1,2,3) == 9
    q = z(c,c,c)
    assert(q(7)==28)

    print("?", 1+2+3+1+1+3+3)
    print("------------")
    print(Add(a,Add(a,b,c,a),c,c)(1,2,3))
    print("------------")
    print(Add(Add(Add(2,1,1,a),2,1,Add(2,1,1,b)),1,1,c)(1,2,3))
    z = Add(Add(Add(2,1,1,a),2,1,Add(2,1,1,b)),1,1,c)
    print(z(1,2,3))


    @CREFunc(signature=unicode_type(unicode_type,unicode_type),
            shorthand='{0}+{1}')
    def Concat(a, b):
        return a + b
    
    a = Var(unicode_type,'a')
    b = Var(unicode_type,'b')
    c = Var(unicode_type,'c')

    for i in range(10):
        z = Concat(a, Concat(Concat(b,c),a ))
        print(z)
        print(z("|","X","Y"))
        if(i == 0):
            init_bytes = used_bytes()
        else:
            print("<<", used_bytes()-init_bytes)





