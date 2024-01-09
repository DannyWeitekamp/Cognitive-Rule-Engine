from numba.typed import Dict, List
from numba.types import DictType, ListType
from numba import i8, types
from cre.caching import unique_hash_v, source_in_cache, import_from_cached, source_to_cache
from numba.experimental import structref
from numba.core.typeconv import Conversion
from numba.extending import intrinsic
from llvmlite import ir as llvmir
from numba.core import cgutils, errors, imputils, types, utils

def _gen_getter_jit(typ,attr):
    return f'''@njit(cache=True)
def {typ}_get_{attr}(self):
    return self.{attr}
'''

def _gen_getter(typ,attr):
    return f'''    @property
    def {attr}(self):
        return {typ}_get_{attr}(self)
    '''
    
def gen_structref_code(typ,fields,
    extra_imports="from numba.experimental import structref",
    register_decorator="@structref.register",
    define_constructor=True,
    define_boxing=True

    ):
    attrs = [x[0] if isinstance(x,tuple) else x for x in fields]
    getters = "\n".join([_gen_getter(typ,attr) for attr in attrs])
    getter_jits = "\n".join([_gen_getter_jit(typ,attr) for attr in attrs])
    attr_list = ",".join(["'%s'"%attr for attr in attrs])
    code = \
f'''
from numba.core import types
from numba import njit
from cre.structref import CastFriendlyStructref, define_boxing
{extra_imports}
{getter_jits}
{register_decorator}
class {typ}TypeTemplate(CastFriendlyStructref):
    pass
    # def preprocess_fields(self, fields):
    #     return tuple((name, types.unliteral(typ)) for name, typ in fields)

class {typ}(structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)

{getters}

{f'structref.define_constructor({typ}, {typ}TypeTemplate, [{attr_list}])' if(define_constructor) else ''}
{f'define_boxing({typ}TypeTemplate, {typ})' if(define_boxing) else ''}


'''
    return code

# def gen_struct(typ,fields):
#     code = gen_structref_code(typ,fields)
#     print(code)
#     l = {}
#     exec(code,{},l)
#     print(l[f"{typ}TypeTemplate"])
#     return l[f"{typ}TypeTemplate"](fields=fields)

def define_structref_template(name, fields, define_constructor=True,define_boxing=True):
    if(isinstance(fields,dict)): fields = [(k,v) for k,v in fields.items()]
    hash_code = unique_hash_v([name,fields, define_constructor, define_boxing])
    # if(name == "ExplanationTreeEntry"): print(name, fields, hash_code)
    
    if(not source_in_cache(name,hash_code)):
        source = gen_structref_code(name, fields, define_constructor=define_constructor,
             define_boxing=define_boxing)
        source_to_cache(name,hash_code,source)
        
    ctor, type_class = import_from_cached(name,hash_code,[name,f"{name}TypeTemplate"]).values()
    ctor._hash_code = hash_code
    type_class.__str__ = type_class.__repr__ = lambda self : name + "Type"
    return ctor, type_class

def define_structref(name, fields, define_constructor=True, define_boxing=True, return_type_class=False):
    if(isinstance(fields,dict)): fields = [(k,v) for k,v in fields.items()]
    ctor, type_class = define_structref_template(name,fields, define_constructor=define_constructor,define_boxing=define_boxing)
    struct_type = type_class(fields=fields)
    struct_type._hash_code = ctor._hash_code
    if(return_type_class):
        return ctor, struct_type, type_class
    else:
        return ctor, struct_type


def define_boxing(struct_type, obj_class):
    '''Same as in numba.experimental.structref but give the type a reference to the proxy'''
    struct_type._proxy_class = obj_class
    structref.define_boxing(struct_type, obj_class)

class CastFriendlyStructref(types.StructRef):
    def can_convert_to(self, typingctx, other):
        """
        Convert this Record to the *other*.
        This method only implements width subtyping for records.
        """
        from numba.core.errors import NumbaExperimentalFeatureWarning
        if isinstance(other, CastFriendlyStructref):
            if len(other._fields) > len(self._fields):
                return
            for other_fd, self_fd in zip(other._fields,
                                         self._fields):
                if not other_fd == self_fd and other_fd[1] != types.Any and self_fd[1] != types.Any:
                    return
            # warnings.warn(f"{self} has been considered a subtype of {other} "
            #               f" This is an experimental feature.",
            #               category=NumbaExperimentalFeatureWarning)
            return Conversion.safe

    def can_convert_from(self, typingctx, other):
        return other.can_convert_to(typingctx, other)

    def unify(self, typingctx, other):
        if(self.can_convert_to(typingctx, other)):
            return other

    def hrepr(self):
        '''A human-readable repr for cre objects'''
        return f"{self._typename}({', '.join([f'{k}={str(v)}' for k,v in self._fields])})" 
# print(DictType(i8,i8))
# print(fields)
# print(gen_structref_code("BOOP",fields))


# fields = [('dict', DictType(i8,i8)),
#     ('v', i8)]

# gen_struct("BOOP", fields)


from numba.core.runtime.nrtdynmod import _meminfo_struct_type

def imp_dtor(context, module, instance_type, user_dtor=None):
    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = llvmir.FunctionType(llvmir.VoidType(),
                                     [llvoidptr, llsize, llvoidptr])

    
    # print(user_dtor)


    fname = "_Dtor.{0}".format(instance_type.name)
    dtor_fn = cgutils.get_or_insert_function(module, dtor_ftype, fname)
    if dtor_fn.is_declaration:
        # Define
        builder = llvmir.IRBuilder(dtor_fn.append_basic_block())

        alloc_fe_type = instance_type.get_data_type()
        alloc_type = context.get_value_type(alloc_fe_type)
        ptr = builder.bitcast(dtor_fn.args[0], alloc_type.as_pointer())

        if(True):
            
            # print(user_dtor_res.fndesc.__dict__)
            # print(user_dtor)

            alloc_size = context.get_abi_sizeof(alloc_type)
            mi_size = context.get_abi_sizeof(_meminfo_struct_type)
            # print(mi_size, type(mi_size))

            raw_data_ptr = builder.ptrtoint(dtor_fn.args[0], cgutils.intp_t)
            # A bit of a hack get meminfo from data pointer by abusing fact that it is allocated in the same block as its data 
            # raw_mi_ptr = builder.sub(raw_data_ptr, context.get_constant(types.intp, mi_size))
            # meminfo = builder.inttoptr(raw_mi_ptr, cgutils.voidptr_t)
            # raw_mi_ptr = context.cast(builder, raw_mi_ptr, types.intp, types.intp)
            # print(raw_mi_ptr.__dict__)

            # inst_struct_type = cgutils.create_struct_proxy(instance_type)
            # inst_struct = context.make_helper(builder, instance_type)
            # inst_struct.meminfo = meminfo

            # context.compile_internal(builder, user_dtor_py_func, types.void(types.intp,), (raw_mi_ptr,))

            # builder.call(user_dtor, [inst_struct._getvalue()])
            # builder.call(user_dtor, [inst_struct._getvalue()])
            builder.call(user_dtor, [raw_data_ptr])

        
        data = context.make_helper(builder, alloc_fe_type, ref=ptr)

        # print(context.nrt.get_meminfos(builder, alloc_fe_type, data._getvalue()))

        
        context.nrt.decref(builder, alloc_fe_type, data._getvalue())


        builder.ret_void()

    return dtor_fn



@intrinsic
def new(typingctx, struct_type, user_dtor_type=None):
    """new(struct_type)

    A jit-code only intrinsic. Used to allocate an **empty** mutable struct.
    The fields are zero-initialized and must be set manually after calling
    the function.

    Example:

        instance = new(MyStruct)
        instance.field = field_value
    """
    # from numba.experimental.jitclass.base import imp_dtor
    # print("<<", user_dtor.dispatcher.py_func)
    
    inst_type = struct_type.instance_type
    user_dtor_res = user_dtor_type.dispatcher.get_compile_result(types.void(i8,))
    # user_dtor_py_func = user_dtor_type.dispatcher.py_func

    # user_dtor = user_dtor_type.dispatcher.get_compile_result(types.void(inst_type)).entry_point
    # print(user_dtor_res)

    def codegen(context, builder, signature, args):
        # FIXME: mostly the same as jitclass ctor_impl()
        model = context.data_model_manager[inst_type.get_data_type()]
        alloc_type = model.get_value_type()
        alloc_size = context.get_abi_sizeof(alloc_type)

        # print("args", args)
        # user_dtor = args[1] if len(args) == 2 else None

        
        user_dtor = context.declare_external_function(builder.module, user_dtor_res.fndesc)
        meminfo = context.nrt.meminfo_alloc_dtor(
            builder,
            context.get_constant(types.uintp, alloc_size),
            imp_dtor(context, builder.module, inst_type, user_dtor),
        )
        data_pointer = context.nrt.meminfo_data(builder, meminfo)
        data_pointer = builder.bitcast(data_pointer, alloc_type.as_pointer())

        # user_dtor = context.declare_external_function(builder.module, user_dtor_res.fndesc)
        builder.call(user_dtor, [builder.ptrtoint(data_pointer, cgutils.intp_t)])


        # Nullify all data
        builder.store(cgutils.get_null_value(alloc_type), data_pointer)

        inst_struct = context.make_helper(builder, inst_type)
        inst_struct.meminfo = meminfo

        return inst_struct._getvalue()

    sig = inst_type(struct_type, user_dtor_type)
    return sig, codegen



_, StructRefType = define_structref("StructRef",[])
