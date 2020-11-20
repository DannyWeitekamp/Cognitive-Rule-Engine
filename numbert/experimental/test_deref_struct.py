from numba import i8
from numba.types import unicode_type, void, Type
# from numbert.experimental.struct_gen import gen_struct_code
from numba.extending import intrinsic, box, NativeValue



# if(not source_in_cache("KnowledgeBaseData",'KnowledgeBaseData')):
# source = gen_struct_code("MyStruct",data_fields)
# print(source)


from numba.experimental import structref
from numba.experimental.structref import _Utils
from numba.core import types, cgutils
from numba import njit

######## Struct Definition ########
data_fields = [
    ("A" , i8),
    ("B" , unicode_type)
]

@njit
def MyStruct_get_A(self):
    return self.A

@njit
def MyStruct_get_B(self):
    return self.B

@structref.register
class MyStructTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class MyStruct(structref.StructRefProxy):
    def __new__(cls, d, v):
        return structref.StructRefProxy.__new__(cls, d, v)

    @property
    def A(self):
        return MyStruct_get_A(self)
    
    @property
    def B(self):
        return MyStruct_get_B(self)
    

structref.define_proxy(MyStruct, MyStructTypeTemplate, ['A','B'])
MyStructType = MyStructTypeTemplate(fields=data_fields)

#Struct Definition

print(MyStruct._numba_box_)

class UntypedStructRef(Type):
    """
    Pointer to a Numba "meminfo" (i.e. the information for a managed
    piece of memory).
    """
    mutable = True

    def __init__(self, dtype, meminfo):
        self.meminfo = meminfo
        self.dtype = dtype
        name = "UntypedStructRef[%s]" % dtype
        super(UntypedStructRef, self).__init__(name)

    @property
    def key(self):
        return self.dtype

@box(UntypedStructRef)
def box_untyped_structref(typ, val, c):
    print("HERE",typ)
    meminfo = val.meminfo
    typ = val.dtype
    # utils = _Utils(c.context, c.builder, typ)
    # struct_ref = utils.get_struct_ref(val)
    # meminfo = struct_ref.meminfo

    mip_type = types.MemInfoPointer(types.voidptr)
    boxed_meminfo = c.box(mip_type, meminfo)

    ctor_pyfunc = c.pyapi.unserialize(c.pyapi.serialize_object(obj_ctor))
    ty_pyobj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))

    res = c.pyapi.call_function_objargs(
        ctor_pyfunc, [ty_pyobj, boxed_meminfo],
    )
    c.pyapi.decref(ctor_pyfunc)
    c.pyapi.decref(ty_pyobj)
    c.pyapi.decref(boxed_meminfo)
    return res


    # toty = toty.get_precise()

    # pyapi = context.get_python_api(builder)
    # sfunc = cgutils.create_struct_proxy(toty)(context, builder)

    # gil_state = pyapi.gil_ensure()
    # addr = lower_get_wrapper_address(
    #     context, builder, val, toty.signature,
    #     failure_mode='return_exc')
    # sfunc.addr = pyapi.long_as_voidptr(addr)
    # pyapi.decref(addr)
    # pyapi.gil_release(gil_state)

    # llty = context.get_value_type(types.voidptr)
    # sfunc.pyaddr = builder.ptrtoint(val, llty)
    # return sfunc._getvalue()


import sys, os
# os.environ['NUMBA_DUMP_IR'] = "1"


#Assuming an typed structref MyStruct w/ A: i8 and B: unicode_type
from numba.extending import intrinsic
from numba.core import types, cgutils
from numba import njit

@intrinsic
def _struct_from_meminfo(typingctx, struct_type, meminfo):
    inst_type = struct_type.instance_type

    def codegen(context, builder, signature, args):
        _, meminfo = args

        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo

        return st._getvalue()

    sig = inst_type(struct_type, types.MemInfoPointer(types.voidptr))
    return sig, codegen


from numba.typed import Dict

@njit
def foo(d):
    meminfo = d[0]
    struct = _struct_from_meminfo(MyStructType,meminfo)
    print(struct.A, struct.B)
    struct.A += 1

s = MyStruct(1,"IT EXISTS")

d = Dict.empty(i8,types.MemInfoPointer(types.voidptr))
d[0] = s._meminfo

print(s.A)
foo(d)
print(s.A, s.B)
foo(d)

