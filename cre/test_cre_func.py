from numba import njit, i8
from cre.cre_func import CREFunc


if __name__ == "__main__":
    # @global_func("Add_poop", i8(i8,i8))

    @CREFunc(signature=i8(i8,i8))
    def Add(a,b):
        return a + b

    print(Add)
