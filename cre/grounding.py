from numba import njit, generated_jit, types, literal_unroll
from numba.types import unicode_type, i8, intp, Tuple, f8, Tuple, DictType, ListType
from numba.typed import Dict, List
from numba.experimental.structref import new
from cre.cre_object import CREObjType
from cre.fact import define_fact, UntypedFact, call_untyped_fact, BaseFact
from cre.fact_intrinsics import fact_lower_getattr, resolve_fact_getattr_type
from cre.tuple_fact import TupleFact, TF
from cre.context import cre_context
from cre.default_ops import Add, Subtract, Divide
from cre.var import Var, GenericVarType
from cre.op import GenericOpType
from cre.utils import _func_from_address
from cre.structref import define_structref
from cre.incr_processor import incr_processor_fields
from cre.memory import Memory, MemoryType
from numba.extending import overload_method, overload, lower_cast, SentryLiteralArgs
# from collections import OrderedSet




Grounding = define_fact("Grounding")


flattener_fields = {
    **incr_processor_fields,    
    "fact_visible_attr_pairs" : types.Any,
    "in_mem" : MemoryType,
    "out_mem" : MemoryType,
}

GenericFlattener, GenericFlattenerType, Flattener  = define_structref("Flattener", flattener_fields, return_template=True) 


# with cre_context("test_inheritence") as context:
spec1 = {"A" : {"type" : "string", "is_semantic_visible" : True}, 
         "B" : {"type" : "number", "is_semantic_visible" : False}}
BOOP1 = define_fact("BOOP1", spec1)
spec2 = {"inherit_from" : BOOP1,
         "C" : {"type" : "number", "is_semantic_visible" : True}}
BOOP2 = define_fact("BOOP2", spec2)
spec3 = {"inherit_from" : BOOP2,
         "D" : {"type" : "number", "is_semantic_visible" : True}}
BOOP3 = define_fact("BOOP3", spec3)



@generated_jit(cache=True, nopython=True)    
def flattener_update_for_attr(fact_type, attr, in_mem, out_mem):#, context_name="cre_default"):
    print("flattener_update_for_attr:: ")
    grounding_types = []
    # for fact_ref, attr in fact_visible_attr_pairs:
        # print(fact_ref, attr)
    # head_type = Tuple((GenericVarType,fact_type.instance_type))
    head_type = GenericVarType
    # head_type = fact_type.instance_type
    val_type = resolve_fact_getattr_type(fact_type.instance_type, attr.literal_value)        
    grounding_type = Grounding(head=head_type, value=val_type)#, flt_val=f8, nom_val=i8)
    # print(grounding_type)
    ctor = grounding_type._ctor[0]

    def impl(fact_type, attr, in_mem, out_mem):
        for fact in in_mem.get_facts(fact_type):
            v = fact_lower_getattr(fact, attr)
            # g = ctor(head=(Var(fact_type,fact.A),fact), value=v)
            g = ctor(head=Var(fact_type,fact.A), value=v)

            out_mem.declare(g)
            # g = ctor(head=fact, value=v)
            print(g)
    return impl




@generated_jit(cache=True, nopython=True)    
def flattener_update(fact_visible_attr_pairs, in_mem, out_mem):#, context_name="cre_default"):
    print("flattener_update:: ")

    # grounding_types = []
    # for fact_ref, attr in fact_visible_attr_pairs:
    #     # print(fact_ref, attr)
    #     val_type = resolve_fact_getattr_type(fact_ref.instance_type, attr.literal_value)        
    #     grounding_type = Grounding(head=head_type, value=val_type)#, flt_val=f8, nom_val=i8)
    #     grounding_types.append(grounding_type)
    # grounding_types = tuple(grounding_types)
    # ctor = grounding_type._ctor[0]
    # print(fact_visible_attr_pairs)
    # if(not context): context
    def impl(fact_visible_attr_pairs, in_mem, out_mem):
        i = 0
        for tup in literal_unroll(fact_visible_attr_pairs):
            fact_type, attr = tup
            flattener_update_for_attr(fact_type, attr, in_mem, out_mem)
            # # print(fact_type, attr)
            # # print("--", fact_type, "--")
            # print("--", i, ":", attr, "--")
            # for fact in in_mem.get_facts(fact_type):
            #     v = fact_lower_getattr(fact, attr)
            #     g = Grounding(head=(Var(fact_type,fact.A),fact), val=v)
            #     print(g)
            i += 1
            # print()

        # return st
    return impl


@generated_jit(cache=True, nopython=True)    
def flattener_ctor(fact_visible_attr_pairs, in_mem, out_mem):
    FlattenerType = 






def get_semantic_visibile_fact_attrs(fact_types):
    context = cre_context()

    sem_vis_fact_attrs = {}
    for ft in fact_types:
        ft = ft_ref.instance_type if (isinstance(ft, types.TypeRef)) else ft
        parents = context.parents_of.get(ft._fact_name,[])
        # print()
        # print(ft.spec)
        for attr, d in ft.spec.items():
            if(d.get("is_semantic_visible", False)):
                is_new = True
                for p in parents:
                    if((p,attr) in sem_vis_fact_attrs):
                        is_new = False
                        break
                if(is_new): sem_vis_fact_attrs[(ft,attr)] = True

    return tuple([(fact_type,types.literal(attr)) for fact_type, attr in sem_vis_fact_attrs.keys()])
                # print([(ft._fact_name, attr) for ft, attr in visible_semantic_attrs])
    # print([(ft._fact_name, attr) for ft, attr in visible_semantic_attrs])
                # print(ft._fact_name, attr)


svfa = get_semantic_visibile_fact_attrs((BOOP1, BOOP2, BOOP3))
print([(ft._fact_name, attr) for ft, attr in svfa])

mem = Memory()
mem.declare(BOOP1("A", 1))
mem.declare(BOOP1("B", 2))
mem.declare(BOOP2("C", 3, 13))
mem.declare(BOOP2("D", 4, 14))
mem.declare(BOOP3("E", 5, 15, 105))
mem.declare(BOOP3("F", 6, 16, 106))


flattener_update(svfa, mem, Memory())
# flattener_ctor((BOOP1, BOOP2, BOOP3))


raise ValueError()


@njit(cache=True)
def flattener_update(self):
    for change_event in self.get_changes():
        # print("CHANGE",change_event)
        comp = spp.mem.get_fact(change_event.idrec, ComponentType)
        sources.append(comp)




# @lower_cast(ShortestPathProcessorType, IncrProcessorType)
# def upcast(context, builder, fromty, toty, val):
#     return _obj_cast_codegen(context, builder, val, fromty, toty, incref=False)


# define_structref()
# raise ValueError()


enumerizer_fields = {
    **incr_processor_fields,
    # "ops" 
    "head_to_inds" : types.Any,
    "float_to_enum" : DictType(f8,i8),
    "string_to_enum" : DictType(f8,i8),
    "obj_to_enum" : DictType(CREObjType,i8),
    "num_noms" : i8
}

Enumerizer, EnumerizerType, EnumerizerTypeClass  = define_structref("Enumerizer", enumerizer_fields, return_template=True) 

@njit(cache=True)    
def flattener_ctor():
    st = new(EnumerizerType)
    return st


@generated_jit(cache=True, nopython=True)
@overload_method(EnumerizerTypeClass, "add_grounding")
def new_op_grounding(self, op, return_type, *args):
    '''Creates a new Grounding Fact for an op and a set of arguments'''

    # Fix annoying numba bug where StarArgs becomes a tuple() of cre.Tuples
    if(isinstance(args,tuple) and isinstance(args[0],types.BaseTuple)):
        args = args[0]
    
    # Because of numba issue: https://github.com/numba/numba/issues/7973
    #  inlining Grounding(head=???, value=??) w/ arguments doesn't work
    #  so just grab it's ctor and use that directly
    # head_type = TF(GenericOpType,TF(*args))
    head_type = Tuple((GenericOpType,args))
    # print(head_type)
    grounding_type = Grounding(head=head_type, value=return_type.instance_type, flt_val=f8, nom_val=i8)
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
