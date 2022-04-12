from numba import njit, generated_jit, types
from numba.types import unicode_type, i8, intp, Tuple, f8
from cre.fact import define_fact, UntypedFact, call_untyped_fact, BaseFact
from cre.tuple_fact import TupleFact, TF
from cre.context import cre_context
from cre.default_ops import Add, Subtract, Divide
from cre.var import Var
from cre.op import GenericOpType
from cre.utils import _func_from_address
Grounding = define_fact("Grounding")

@generated_jit(cache=True, nopython=True)
def new_op_grounding(op, return_type, *args):
    '''Creates a new Grounding Fact for an op and a set of arguments'''

    # Fix annoying numba bug where StarArgs becomes a tuple() of cre.Tuples
    if(isinstance(args,tuple) and isinstance(args[0],types.BaseTuple)):
        args = args[0]
    
    # Because of numba issue: https://github.com/numba/numba/issues/7973
    #  inlining Grounding(head=???, value=??) w/ arguments doesn't work
    #  so just grab it's ctor and use that directly
    head_type = TF(GenericOpType,TF(*args))
    grounding_type = Grounding(head=head_type, value=return_type.instance_type)
    ctor = grounding_type._ctor[0]
    
    # Find the types for the op's check and call
    call_sig = return_type.instance_type(*args)
    check_sig = types.bool_(*args)
    call_f_type = types.FunctionType(call_sig)
    check_f_type = types.FunctionType(check_sig)

    def impl(op, return_type, *args):
        head = TF(op, TF(*args))        
        if(op.check_addr != 0):
            check = _func_from_address(check_f_type, op.check_addr)
            check_ok = check(*args)
            if(not check(*args)): raise ValueError("Grounding attempt failed check().")

        call = _func_from_address(call_f_type, op.call_addr)
        value = call(*args)
        grounding = ctor(head=head, value=value)
        return grounding
    return impl


def ground_op(op,*args):
    return_type = op.signature.return_type
    print(op, return_type)
    return new_op_grounding(op, return_type, *args)

if __name__ == "__main__":
    import faulthandler; faulthandler.enable()
    g = ground_op(Add(Var(f8),Var(f8)), 1., 7.)
    print("<<g", g)
    g = ground_op(Add(Var(unicode_type), Var(unicode_type)),  "1", "7")
    print("<<g", g)
