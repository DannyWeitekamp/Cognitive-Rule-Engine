import numpy as np
from numba import njit, generated_jit, types, literal_unroll, u8, i8, f8, u1
from numba.types import unicode_type,  intp, Tuple,  Tuple, DictType, ListType
from numba.typed import Dict, List
from numba.experimental.structref import new
from cre.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.cre_object import CREObjType
from cre.fact import define_fact, UntypedFact, call_untyped_fact, BaseFact
from cre.fact_intrinsics import fact_lower_getattr, resolve_fact_getattr_type
from cre.tuple_fact import TupleFact, TF
from cre.context import cre_context
from cre.default_ops import Add, Subtract, Divide
from cre.var import Var, GenericVarType
from cre.op import GenericOpType
from cre.utils import _func_from_address, _cast_structref, _obj_cast_codegen, _func_from_address, _incref_structref
from cre.structref import define_structref
from cre.incr_processor import incr_processor_fields, IncrProcessorType, init_incr_processor
from cre.memory import Memory, MemoryType
from numba.extending import overload_method, overload, lower_cast, SentryLiteralArgs
from numba.experimental.function_type import _get_wrapper_address
import cloudpickle




enumerizer_fields = {
    **incr_processor_fields,
    # "ops" 
    "head_to_inds" : types.Any,
    "float_to_enum" : DictType(f8,i8),
    "string_to_enum" : DictType(f8,i8),
    "obj_to_enum" : DictType(CREObjType,i8),
    "num_noms" : i8
}

Enumerizer, EnumerizerType, EnumerizerTypeClass  = define_structref("Enumerizer", enumerizer_fields, return_type_class=True) 

# @njit(cache=True)    
# def flattener_ctor():
#     st = new(EnumerizerType)
#     return st


@generated_jit(cache=True, nopython=True)
@overload_method(EnumerizerTypeClass, "add_grounding")
def new_op_grounding(self, op, return_type, *args):
    '''Creates a new gval Fact for an op and a set of arguments'''

    # Fix annoying numba bug where StarArgs becomes a tuple() of cre.Tuples
    if(isinstance(args,tuple) and isinstance(args[0],types.BaseTuple)):
        args = args[0]
    
    # Because of numba issue: https://github.com/numba/numba/issues/7973
    #  inlining gval(head=???, value=??) w/ arguments doesn't work
    #  so just grab it's ctor and use that directly
    # head_type = TF(GenericOpType,TF(*args))
    head_type = Tuple((GenericOpType,args))
    # print(head_type)
    grounding_type = gval(head=head_type, value=return_type.instance_type, flt_val=f8, nom_val=i8)
    ctor = grounding_type._ctor[0]
    
    # Find the types for the op's check and call
    call_sig = return_type.instance_type(*args)
    check_sig = types.bool_(*args)
    call_f_type = types.FunctionType(call_sig)
    check_f_type = types.FunctionType(check_sig)

    def impl(self, op, return_type, *args):
        # head = TF(op, TF(*args))        
        head = (op, args)       
        if(op.check_addr != 0):
            check = _func_from_address(check_f_type, op.check_addr)
            check_ok = check(*args)
            if(not check(*args)): raise ValueError("gval attempt failed check().")

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
