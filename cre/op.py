import operator
import numpy as np
import numba
from numba.core.dispatcher import Dispatcher
from numba import types, njit, i8, u8, i4, u1, u2, literally, generated_jit, boolean
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type, void, Tuple
from numba.experimental import structref
from numba.experimental.structref import new, define_boxing, define_attributes, _Utils
from numba.extending import lower_cast, NativeValue, box, unbox, overload_method, intrinsic, overload_attribute, intrinsic, lower_getattr_generic, overload, infer_getattr, lower_setattr_generic
from numba.core.typing.templates import AttributeTemplate
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.caching import gen_import_str, unique_hash,import_from_cached, source_to_cache, source_in_cache, cache_safe_exec, get_cache_path
from cre.context import cre_context
from cre.structref import define_structref, define_structref_template
from cre.utils import (_struct_from_meminfo, _meminfo_from_struct, _cast_structref, cast_structref, decode_idrec, lower_getattr, _struct_from_ptr,  lower_setattr, lower_getattr,
                       _raw_ptr_from_struct, _raw_ptr_from_struct_incref, _decref_ptr, _incref_ptr, _incref_structref, _ptr_from_struct_incref, ptr_t)
from cre.utils import encode_idrec, assign_to_alias_in_parent_frame, as_typed_list
from cre.subscriber import base_subscriber_fields, BaseSubscriber, BaseSubscriberType, init_base_subscriber, link_downstream
from cre.vector import VectorType
from cre.fact import Fact, gen_fact_import_str, get_offsets_from_member_types
from cre.var import Var, var_memcopy, GenericVarType
from cre.cre_object import CREObjType, cre_obj_field_dict, CREObjTypeClass, CREObjProxy
from cre.core import T_ID_OP, register_global_default
from cre.make_source import make_source, gen_def_func, gen_assign, gen_if, gen_not, resolve_template, gen_def_class
from numba.core import imputils, cgutils
from numba.core.datamodel import default_manager, models, register_default
from numba.experimental.function_type import _get_wrapper_address


from operator import itemgetter
from copy import copy
from os import getenv
import inspect, cloudpickle, pickle
from textwrap import dedent, indent
from collections.abc import Iterable
import warnings




import time
class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.2f} ms')

def warn_cant_compile(func_name, op_name, e):
    s = f'''
########## WARNING CAN'T COMPILE {op_name}.{func_name}() ########### 
{indent(str(e),'  ')} 
########################################################
Numba was unable to compile {func_name}() for op {op_name}. 
Using objmode (i.e. native python) instead. To ignore warning set nopython=False.\n\
'''
    warnings.warn(s, NumbaPerformanceWarning)


def gen_op_source(cls):
    '''Generate source code for the relevant functions of a user defined Op.
       Note: The jitted call() and check() functions are pickled, the bytes dumped  
        into the source, and then reconstituted at runtime. This strategy seems to work
        even if globals are used in the function. 
    '''
    arg_types = cls.signature.args
    return_type = cls.signature.return_type
    # arg_imports = "\n".join([gen_fact_import_str(x) for x in arg_types if isinstance(x,Fact)])
    field_dict = {**op_fields_dict,**{f"arg{i}" : t for i,t in enumerate(arg_types)}}
    arg_names = ', '.join([f'a{i}' for i in range(len(arg_types))])
    # offsets = get_offsets_from_member_types(field_dict)
    # arg_offsets = offsets[len(op_fields_dict):]
    nl = "\n"
    source = \
f'''import numpy as np
from numba import njit, void, i8, boolean, objmode
from numba.experimental.function_type import _get_wrapper_address
from numba.core.errors import NumbaError, NumbaPerformanceWarning
from cre.utils import _func_from_address, _load_ptr
from cre.op import op_fields_dict, OpTypeClass, warn_cant_compile
import cloudpickle

nopython_call = {cls.nopython_call} 
nopython_check = {cls.nopython_check} 

method_addrs = np.zeros((7,),dtype=np.int64)

call_sig = cloudpickle.loads({cloudpickle.dumps(cls.call_sig)})
call_pyfunc = cloudpickle.loads({cls.call_bytes})
{"".join([f'h{i}_type, ' for i in range(len(arg_types))])} = call_sig.args

try:
    call = njit(call_sig,cache=True)(call_pyfunc)
except NumbaError as e:
    nopython_call=False
    warn_cant_compile('call',{cls.__name__!r}, e)

if(nopython_call==False):
    return_type = call_sig.return_type
    @njit(cache=True)
    def call({arg_names}):
        with objmode(_return=return_type):
            _return = call_pyfunc({arg_names})
        return _return

# call_addr = _get_wrapper_address(call, call_sig)
call_heads = call

@njit(call_sig.return_type(i8[::1],) ,cache=True)
def call_head_ptrs(ptrs):
{indent(nl.join([f'i{i} = _load_ptr(h{i}_type,ptrs[{i}])' for i in range(len(arg_types))]),prefix='    ')}
    return call_heads({",".join([f'i{i}' for i in range(len(arg_types))])})

'''
    if(hasattr(cls,'check')):
        source += f'''
check_sig = cloudpickle.loads({cloudpickle.dumps(cls.check_sig)})
check_pyfunc = cloudpickle.loads({cls.check_bytes})

try:
    check = njit(check_sig,cache=True)(check_pyfunc)
except NumbaError as e:
    nopython_check=False
    warn_cant_compile('check',{cls.__name__!r}, e)

if(nopython_check==False):
    @njit(cache=True)
    def check({arg_names}):
        with objmode(_return=boolean):
            _return = check_pyfunc({arg_names})
        return _return

# check_addr = _get_wrapper_address(check, check_sig)

method_addrs[6] = _get_wrapper_address(check, check_sig)
'''
# arg_offsets = {str(arg_offsets)}
    source += f'''
@njit(boolean(*call_sig.args), cache=True)
def match({arg_names}):
    {f"if(not check({arg_names})): return 0" if(hasattr(cls,'check')) else ""}
    return 1 if(call({arg_names})) else 0

match_heads = match



@njit(boolean(i8[::1],) ,cache=True)
def match_head_ptrs(ptrs):
{indent(nl.join([f'i{i} = _load_ptr(h{i}_type,ptrs[{i}])' for i in range(len(arg_types))]),prefix='    ')}
    return match_heads({",".join([f'i{i}' for i in range(len(arg_types))])})

method_addrs[0] = _get_wrapper_address(call_heads, call_sig)
method_addrs[1] = _get_wrapper_address(call_head_ptrs, call_sig.return_type(i8[::1]))
method_addrs[2] = method_addrs[0] # call is call_heads 
method_addrs[3] = _get_wrapper_address(match_heads, boolean(*call_sig.args))
method_addrs[4] = _get_wrapper_address(match_head_ptrs, boolean(i8[::1],))
method_addrs[5] = method_addrs[3] # match is match_heads

field_dict = {{**op_fields_dict,**{{f"arg{{i}}" : t for i,t in enumerate(call_sig.args)}}}}
{cls.__name__+'Type'} = OpTypeClass([(k,v) for k,v in field_dict.items()]) 
'''
    return source


from cre.var import GenericVarType

_head_range_type = np.dtype([('start', np.uint8), ('length', np.uint8)])
head_range_type = numba.from_dtype(_head_range_type)

op_fields_dict = {
    **cre_obj_field_dict, 
    "name" : unicode_type,
    "expr_template" : unicode_type,
    "shorthand_template" : unicode_type,

    # Mapping variable base_ptrs to aliases
    "base_var_map" : DictType(i8, i8),
    # Inverse of 'base_var_map'
    # "inv_base_var_map" : DictType(unicode_type, i8),
    # Pointers of op's head vars (i.e. x.nxt and x.nxt.nxt are both heads of x)

    "base_vars" : ListType(GenericVarType),
    "head_vars" : ListType(GenericVarType),
    "head_var_ptrs" : i8[::1],
    "head_ranges" : head_range_type[::1],

    "return_type_name" : unicode_type,
    
    "arg_type_names" : ListType(unicode_type),
    "call_addr" : i8,
    "call_heads_addr" : i8,
    "call_head_ptrs_addr" : i8,
    "match_addr" : i8,
    "match_heads_addr" : i8,
    "match_head_ptrs_addr" : i8,
    "check_addr" : i8,
    

    "return_t_id" : u2,
    "is_ptr_op" : types.boolean,
    
    # "arg_types" : types.Any,
    # "out_type" : types.Any,
    # "is_const" : i8[::1]
}

class OpTypeClass(CREObjTypeClass):
    t_id = T_ID_OP
    def preprocess_fields(self, fields):
        self.t_id = T_ID_OP
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

    def __str__(self):
        return f"cre.GenericOpType"

# lower_cast(OpTypeClass, CREObjType)(impl_cre_obj_upcast)


@register_default(OpTypeClass)
class OpModel(models.StructModel):
    """Model for cre.Op. A reference to the structref payload, and the Op class.
    """
    def __init__(self, dmm, fe_typ):
        dtype = fe_typ.get_data_type()
        members = [
            ("meminfo", types.MemInfoPointer(dtype)),
            ("py_class", types.pyobject),
        ]
        super().__init__(dmm, fe_typ, members)


# @structref.register

default_manager.register(OpTypeClass, OpModel)
define_attributes(OpTypeClass)


def op_define_boxing(struct_type, obj_class):
    """
        Like structref.define_boxing but adds in __class__ to the datamodel
    """
    if struct_type is types.StructRef:
        raise ValueError(f"cannot register {types.StructRef}")

    obj_ctor = obj_class._numba_box_

    @box(struct_type)
    def box_op(typ, val, c):
        ''' NRT -> Python '''
        struct_ref = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        meminfo = struct_ref.meminfo
        py_class = struct_ref.py_class

        mip_type = types.MemInfoPointer(types.voidptr)
        boxed_meminfo = c.box(mip_type, meminfo)
        boxed_py_class = c.box(types.pyobject, py_class)

        ctor_pyfunc = c.pyapi.unserialize(c.pyapi.serialize_object(obj_ctor))
        ty_pyobj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))

        res = c.pyapi.call_function_objargs(
            ctor_pyfunc, [ty_pyobj, boxed_meminfo, boxed_py_class],
        )
        c.pyapi.decref(ctor_pyfunc)
        c.pyapi.decref(ty_pyobj)
        c.pyapi.decref(boxed_meminfo)
        c.pyapi.decref(boxed_py_class)
        return res

    @unbox(struct_type)
    def unbox_op(typ, obj, c):
        ''' Python -> NRT '''
        mi_obj = c.pyapi.object_getattr_string(obj, "_meminfo")
        op_class_obj = c.pyapi.object_getattr_string(obj, "__class__")

        mip_type = types.MemInfoPointer(types.voidptr)
        mi = c.unbox(mip_type, mi_obj).value

        op_class = c.unbox(types.pyobject, op_class_obj).value

        struct_ref = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        struct_ref.meminfo = mi
        struct_ref.py_class = op_class

        out = struct_ref._getvalue()

        c.pyapi.decref(mi_obj)
        # c.pyapi.decref(op_class_obj) <- Don't decref this
        return NativeValue(out)




GenericOpType = OpTypeClass([(k,v) for k,v in op_fields_dict.items()])
register_global_default("Op", GenericOpType)


class UntypedOp():
    '''An Op that has not been given a signature yet'''
    def __init__(self,name,members):
        self.name = name
        self.members = members
        self.members['call'] = njit(cache=True)(self.members['call'])
        self._specialize_cache = {}
    @property
    def arg_names(self):
        if(not hasattr(self,"_arg_names")):
            f = self.members['call']
            py_func = f.py_func if isinstance(f, Dispatcher) else f
            self._arg_names = inspect.getfullargspec(py_func)[0]
        return self._arg_names

    def __repr__(self):
        return f'UntypedOp(name={self.name}, members={self.members})'

    def __str__(self):
        return f'{self.name}({", ".join(self.arg_names)})'


    def __call__(self,*py_args):
        arg_types = [] 
        all_const = True
        any_not_var = False
        any_deref_vars = False
        py_args = list(py_args)
        for i, x in enumerate(py_args):
            if(isinstance(x,types.Type)):
                arg_types.append(x)
                x = py_args[i] = Var(x)
            else:
                arg_types.append(resolve_return_type(x))

            if(isinstance(x,(OpMeta,OpComp, Op))):
                all_const = False
                any_not_var = True
            elif(isinstance(x,(Var))):
                if(len(x.deref_infos) > 0):
                    any_deref_vars = True
                all_const = False
            else:
                any_not_var = True

        call = self.members['call']
        if(all_const):
            return call(*py_args)

        arg_types = tuple(arg_types)
        # print(tuple(arg_types),tuple(arg_types) in call.overloads)
        # print("-- overloads",call.overloads.keys())
        cres = call.overloads.get(arg_types,None)
        if(cres is not None):
            sig = cres.signature
        else:
            # Note: get_call_template is an internal method of numba.Dispatcher 
            (template,*rest) = call.get_call_template(arg_types,{})
            # print("<<", "cases", template.cases)
            # print(rest)
            # print([x for x in template.cases if x.args==arg_types])
            sig = [x for x in template.cases if x.args==arg_types][0] #Note: this can be finicky might need to find the best one
        # self.members['signature'] = sig
        members = {**self.members, 'signature': sig}

        if(any_not_var or any_deref_vars):
            tup = (sig, str(py_args))#tuple([str(x) for x in py_args]))
            # print(tup)
            if(tup not in self._specialize_cache):
                # print("MISS")
                op = new_op(self.name, members)
                op_comp = OpComp(op, *py_args)
                op_cls = op_comp.flatten(return_class=True)
                self._specialize_cache[tup] = op_cls
                # print("_specialize_cache")
                # op_cls.__qualname__ = f"{cls.__module__}.{name}(signature={members['signature']})" 
                # op = op_cls.make_singleton_inst([x[0] for x in op_comp.head_vars.values()])
            # else:
                # print("HIT")

            # with PrintElapse("E"):
            op_cls = self._specialize_cache[tup]
            head_vars = {}
            base_to_arg_num = {}
            for x in py_args:
                # print("<<", x,type(x),isinstance(x,Var))
                # print
                
                if(isinstance(x,Var)):
                    b_ptr = x.base_ptr
                    if(b_ptr not in base_to_arg_num): base_to_arg_num[b_ptr] = len(base_to_arg_num)
                    head_vars[(b_ptr, x.get_derefs_str())] = (x,base_to_arg_num[b_ptr])

                if(isinstance(x, Op)):
                    for h_var in x.head_vars:
                        b_ptr = h_var.base_ptr
                        if(b_ptr not in base_to_arg_num): base_to_arg_num[b_ptr] = len(base_to_arg_num)
                        head_vars[(b_ptr, h_var.get_derefs_str())] = (h_var,base_to_arg_num[b_ptr])


            # print(len(head_vars),head_vars.keys())
            head_vars = [x[0] for x in sorted(head_vars.values(),key=lambda x : x[1])]

            # print()
            
            #NOTE need to extract vars from input ops
            # print("py_args", py_args)
            # print("head_vars", head_vars)
            op = op_cls.make_singleton_inst(head_vars=head_vars)
            return op

                # ???




        # elif(any_not_var):
        #     print("ANY NOT VAR")
        #     op = new_op(self.name, members)
        #     return OpComp(op, *py_args).flatten()
        else:
            var_ptrs = np.array([v.get_ptr() for v in py_args],dtype=np.int64)
            # Retreive from Cache/Make typed version 
            if(sig not in self._specialize_cache):
                # print("MISS", sig)
                op_cls = new_op(self.name, members,return_class=True)
                # op_cls.__reduce__ = lambda : (new_op, (self.name, ), {"return_class":True})
                self._specialize_cache[sig] = op_cls
                # op_cls.__qualname__ = self.__qualname__ + 
            # else:
                # print("HIT")
            
            op_cls = self._specialize_cache[sig]
            return op_cls.make_singleton_inst(head_vars=py_args)




def resolve_return_type(x):
    if(isinstance(x,Var)):
        return x.head_type
    elif(isinstance(x,Op)):
        return x.signature.return_type
    elif(isinstance(x, OpComp)):
        return x.op.signature.return_type
    elif(isinstance(x, UntypedOp)):
        raise ValueError(f"Cannot resolve return type of UntypedOp {x}")
    elif(isinstance(x, (int))):
        return types.int64
    elif(isinstance(x, (float))):
        return types.float64
    elif(isinstance(x, (str))):
        return types.unicode_type

        



# @njit(cache=True)
# def var_to_ptr(x):
#     return _raw_ptr_from_struct_incref(x)

def new_vars_from_types(types, names):
    '''Generate vars with 'types' and 'names'
        and output a ptr_array of their memory addresses.'''
    #TODO: Op should have deconstructor so these are decrefed.
    # ptrs = np.empty(len(types), dtype=np.int64)
    return [Var(t,names[i]) for i, t in enumerate(types)]

@njit(cache=True)
def gen_placeholder_aliases(var_ptrs):
    _vars = Dict.empty(i8, GenericVarType)
    temp_inv_base_var_map = Dict.empty(unicode_type, i8)
    
    n_auto_gen = 0
    for var_ptr in var_ptrs:
        head_var = _struct_from_ptr(GenericVarType, var_ptr)
        base_ptr = i8(head_var.base_ptr)
        if(base_ptr not in _vars):
            base_var = _struct_from_ptr(GenericVarType, base_ptr)
            _vars[i8(base_ptr)] = base_var
            
    for var in _vars.values():
        if(var.alias != ""):
            if(var.alias not in temp_inv_base_var_map):
                temp_inv_base_var_map[var.alias] = var_ptr
            else:
                raise NameError("Duplicate Var alias.")
        else:
            n_auto_gen += 1
        
    generated_aliases = List.empty_list(unicode_type)
    ascii_code = 97# i.e. 'a'
    bad_aliases = ['i','j','k','o','I','J','K','O']
    for i in range(n_auto_gen):
        alias = chr(ascii_code)
        while(alias in temp_inv_base_var_map or alias in bad_aliases):
            ascii_code += 1
            alias = chr(ascii_code)
        generated_aliases.append(alias)
        ascii_code += 1

    return generated_aliases


@njit(cache=True)
def make_head_ranges(n_args, head_var_ptrs):
    head_ranges = np.empty((n_args,),
                            dtype=head_range_type)
    start, length = 0,0
    prev_base_ptr = 0
    k = 0 
    for ptr in head_var_ptrs:
        head_var = _struct_from_ptr(GenericVarType, ptr)
        base_ptr = i8(head_var.base_ptr)
        if(prev_base_ptr == 0 or base_ptr == prev_base_ptr):
            length += 1
        else:
            head_ranges[k].start = start 
            head_ranges[k].length = length
            k += 1
            start += length; length = 1
            
        prev_base_ptr = base_ptr
    head_ranges[k][0] = start 
    head_ranges[k][1] = length
    return head_ranges


# @njit(cache=True)
# def op_reparametrize(op,head_var_ptrs):
#     st = new(GenericOpType)

#     st.idrec = op.idrec
#     st.name = op.name
#     st.return_type_name = op.return_type_name
#     st.return_t_id = op.return_t_id

#     st.arg_type_names = op.arg_type_names
#     st.expr_template = op.expr_template
#     st.shorthand_template = op.shorthand_template

#     st.call_heads_addr = op.call_heads_addr
#     st.call_head_ptrs_addr = op.call_head_ptrs_addr
#     st.call_addr = op.call_addr
#     st.match_heads_addr = op.match_heads_addr
#     st.match_head_ptrs_addr = op.match_head_ptrs_addr
#     st.match_addr = op.match_addr
#     st.check_addr = op.check_addr

#     st.is_ptr_op = op.is_ptr_op

#     st.base_var_map = Dict.empty(i8, unicode_type)
#     st.inv_base_var_map = Dict.empty(unicode_type, i8)
#     st.base_vars = List.empty_list(GenericVarType)
#     st.head_vars = List.empty_list(GenericVarType)
#     generated_aliases = gen_placeholder_aliases(head_var_ptrs)
#     j = 0
#     for head_ptr in head_var_ptrs:
#         head_var = _struct_from_ptr(GenericVarType, head_ptr)
#         st.head_vars.append(head_var)
#         base_ptr = i8(head_var.base_ptr)
#         if(base_ptr not in st.base_var_map):
#             base_var = _struct_from_ptr(GenericVarType, base_ptr)
#             st.base_vars.append(base_var)
#             alias = base_var.alias
#             if(alias == ""):
#                 alias = generated_aliases[j]; j+=1;
            
#             st.base_var_map[base_ptr] = alias
#             st.inv_base_var_map[alias] = base_ptr

#     st.head_var_ptrs = head_var_ptrs
#     st.head_ranges = make_head_ranges(len(st.base_var_map), head_var_ptrs)


@njit(cache=True)
def op_ctor(name, return_type_name, return_t_id, arg_type_names, head_var_ptrs, method_addrs,
            expr_template="MOOSE", shorthand_template="ROOF",is_ptr_op=False):
    '''The constructor for an Op instance that can be passed to the numba runtime'''
    st = new(GenericOpType)
    st.idrec = encode_idrec(T_ID_OP, 0, 0)
    st.name = name
    st.return_type_name = return_type_name
    st.return_t_id = return_t_id

    st.arg_type_names = arg_type_names
    st.base_var_map = Dict.empty(i8, i8)
    # st.inv_base_var_map = Dict.empty(unicode_type, i8)
    st.base_vars = List.empty_list(GenericVarType)
    st.head_vars = List.empty_list(GenericVarType)

    generated_aliases = gen_placeholder_aliases(head_var_ptrs)
    j = 0
    for head_ptr in head_var_ptrs:
        head_var = _struct_from_ptr(GenericVarType, head_ptr)
        st.head_vars.append(head_var)
        base_ptr = i8(head_var.base_ptr)
        if(base_ptr not in st.base_var_map):
            base_var = _struct_from_ptr(GenericVarType, base_ptr)
            st.base_vars.append(base_var)
            alias = base_var.alias
            if(alias == ""):
                alias = generated_aliases[j]; j+=1;
            
            st.base_var_map[base_ptr] = len(st.base_var_map)
            # st.inv_base_var_map[alias] = base_ptr

    st.head_var_ptrs = head_var_ptrs
    st.head_ranges = make_head_ranges(len(st.base_var_map), head_var_ptrs)

    st.expr_template = expr_template
    st.shorthand_template = shorthand_template

    st.call_heads_addr = method_addrs[0]
    st.call_head_ptrs_addr = method_addrs[1]
    st.call_addr = method_addrs[2]
    st.match_heads_addr = method_addrs[3]
    st.match_head_ptrs_addr = method_addrs[4]
    st.match_addr = method_addrs[5]
    st.check_addr = method_addrs[6]

    st.is_ptr_op = is_ptr_op

    return st


@intrinsic
def _copy_op(typingctx, inst_type):
    # from cre.utils import _meminfo_copy_unsafe
    from numba.experimental.jitclass.base import imp_dtor
    def codegen(context, builder, signature, args):
        inp_op = args[0]
        ctor = cgutils.create_struct_proxy(inst_type)
        inp_op_struct = ctor(context, builder, value=inp_op)
        # meminfo = dstruct.meminfo

        model = context.data_model_manager[inst_type.get_data_type()]
        alloc_type = model.get_value_type()
        alloc_size = context.get_abi_sizeof(alloc_type)

        meminfo = context.nrt.meminfo_alloc_dtor(
            builder,
            context.get_constant(types.uintp, alloc_size),
            imp_dtor(context, builder.module, inst_type),
        )
        data_pointer = context.nrt.meminfo_data(builder, meminfo)
        data_pointer = builder.bitcast(data_pointer, alloc_type.as_pointer())

        # Nullify all data
        builder.store(cgutils.get_null_value(alloc_type), data_pointer)

        inst_struct = context.make_helper(builder, inst_type)
        inst_struct.meminfo = meminfo
        inst_struct.py_class = inp_op_struct.py_class
        return inst_struct._getvalue()


    sig = inst_type(inst_type)
    return sig, codegen

@njit(cache=True)
def op_copy(op, new_base_vars=None):
    st = _copy_op(op)
    st.idrec = op.idrec#encode_idrec(T_ID_OP, 0, 0)
    st.name = op.name
    st.return_type_name = op.return_type_name
    st.arg_type_names = op.arg_type_names

    if(new_base_vars is None):
        st.base_var_map = op.base_var_map 
        # st.inv_base_var_map = op.inv_base_var_map 
        st.base_vars = op.base_vars
        st.head_vars = op.head_vars
        st.head_var_ptrs = op.head_var_ptrs
    else:
        assert(len(new_base_vars) == len(op.base_vars))

        st.head_var_ptrs = np.empty(len(op.head_vars),dtype=np.int64)
        st.base_var_map = Dict.empty(i8, i8)
        # st.inv_base_var_map = Dict.empty(unicode_type, i8)
        base_var_map = Dict.empty(i8,GenericVarType)
        for i, (o_v, n_v) in enumerate(zip(op.base_vars, new_base_vars)):
            st.head_var_ptrs[i] = n_v.base_ptr
            st.base_var_map[n_v.base_ptr] = i#n_v.alias
            # st.inv_base_var_map[n_v.alias] = n_v.base_ptr
            base_var_map[o_v.base_ptr] = n_v

        st.head_vars = List.empty_list(GenericVarType)
        for hv in op.head_vars:
            new_hv = new(GenericVarType)
            var_memcopy(hv, new_hv)
            new_base_var = base_var_map[hv.base_ptr]
            lower_setattr(new_hv, "base_ptr", new_base_var.base_ptr)
            if(i8(new_hv.base_ptr_ref) != 0):
                lower_setattr(new_hv, "base_ptr_ref", _ptr_from_struct_incref(new_base_var))

            st.head_vars.append(new_hv)

        st.base_vars = new_base_vars

    st.head_ranges = op.head_ranges 
    st.expr_template = op.expr_template
    st.shorthand_template = op.shorthand_template
    
    st.call_heads_addr = op.call_heads_addr
    st.call_head_ptrs_addr = op.call_head_ptrs_addr
    st.call_addr = op.call_addr
    st.match_heads_addr = op.match_heads_addr
    st.match_head_ptrs_addr = op.match_head_ptrs_addr
    st.match_addr = op.match_addr
    st.check_addr = op.check_addr

    st.is_ptr_op = op.is_ptr_op

    return st



@njit(cache=True)
def op_dtor(self):
    for head_ptr in self.head_var_ptrs:
        _decref_ptr(head_ptr)




def new_op(name, members, head_vars=None, return_class=False):
    '''Creates a new custom Op with 'name' and 'members' '''
    has_check = 'check' in members
    assert 'call' in members, "Op must have call() defined"
    assert hasattr(members['call'], '__call__'), "call() must be a function"
    # assert 'signature' in members, "Op must have signature"
    assert not (has_check and not hasattr(members['check'], '__call__')), "check() must be a function"

    if('signature' not in members):
        return UntypedOp(name,members)
    
    # with PrintElapse("\thandling"):
    # See if call/check etc. are raw python functions and need to be wrapped in @jit.
    # call_needs_jitting = not isinstance(members['call'], Dispatcher)
    # check_needs_jitting = has_check and (not isinstance(members['check'],Dispatcher))
    
    # 'cls' is the class of the user defined Op (i.e. Add(a,b)).
    cls = type.__new__(OpMeta, name, (Op,), members) # (cls, name, bases, dct)
    # print(">>", cls)
    cls.__reduce__ = lambda self : (OpMeta,(name,(Op,),members))
    # cls.__reduce__ = lambda self : (new_op,(name,members))
    # cls.__getstate__ = lambda self : (name, members)
    # cls.__setstate__ = lambda name, members : new_op(name, {**members}, return_class=True) 
    # cls.__qualname__ = f'{members["call"].__qualname__}.__class__' 
    
    cls.__module__ = members["call"].__module__
    # cls.__qualname__ = f"{cls.__module__}.{name}(signature={members['signature']})" 
    # cls.__call__ = call_op
    cls._handle_commutes()
    cls._handle_nopython()

    # with PrintElapse("\tprocess_method"):
    cls.process_method("call", cls.signature)
    
    if(has_check):
        cls.process_method("check", u1(*cls.signature.args))

    cls._handle_defaults()  
    # cls._handle_default_templates()
    
    # If either call or check needs to be jitted then generate source code
    #  to do the jitting and put it in the cache. Or retrieve if already
    #  defined.  This way jit(cache=True) can reliably re-retreive.
    # with PrintElapse("\tget_src"):
    # if(call_needs_jitting or check_needs_jitting):
    name = cls.__name__

    #Triggers getter
    long_hash = cls.long_hash 
    if(not source_in_cache(name, long_hash) or getattr(cls,'cache',True) == False):
        source = gen_op_source(cls)
        source_to_cache(name,long_hash,source)

    # to_import = ['call','call_addr'] if call_needs_jitting else []
    to_import = ['call', 'method_addrs']
    if(has_check): to_import += ['check']
    l = import_from_cached(name, long_hash, to_import)
    for key, value in l.items():
        setattr(cls, key, value)

    # with PrintElapse("\tgwap"):
    # Make static so that self isn't the first argument for call/check.
    cls.call = staticmethod(cls.call)
    if(has_check): cls.check = staticmethod(cls.check)

    # # Get store the addresses for call/check
    # if(not call_needs_jitting): 
    #     cls.call_sig = cls.signature
    #     cls.call_addr = _get_wrapper_address(cls.call,cls.call_sig)

    # if(not check_needs_jitting and has_check):
    #     # cls.check_sig = u1(*cls.signature.args)
    #     cls.check_addr = _get_wrapper_address(cls.check,cls.check_sig)

    # print(cls.match_head_ptrs.overloads.keys())
    # cls.match_head_ptrs_addr = _get_wrapper_address(cls.match_head_ptrs,boolean(i8[::1],))

    # Standardize shorthand definitions
    if(hasattr(cls,'shorthand') and 
        not isinstance(cls.shorthand,dict)):
        cls.shorthand = {'*' : cls.shorthand}

    # with PrintElapse("\tMakeInst"):
    if(cls.__name__ != "__GenerateOp__" and not return_class):
        op_inst = cls.make_singleton_inst(head_vars=head_vars)
        return op_inst

    return cls



# def call_op(self,*py_args):
#     if(all([not isinstance(x,(Var,OpMeta,OpComp, Op)) for x in py_args])):
#         # If all of the arguments are constants then just call the Op
#         return self.call(*py_args)
#     else:
#         # Otherwise build an OpComp and flatten it into a new Op
#         op_comp = OpComp(self,*py_args)
#         op = op_comp.flatten()
        
#         return op

        
class OpMeta(type):
    ''' A the metaclass for op. Useful for singleton generation.
    '''
    # def __repr__(cls):
    #     return cls.__name__ + f'({",".join(["?"] * len(cls.signature.args))})'

    # This is similar to defining __init_subclass__ in Op inside except we can
    #    return whatever we want. In this case we return a singleton instance.
    def __new__(meta_cls, *args):
        name, members = args[0], args[2]
        if(name == "Op"):
            ret = super().__new__(meta_cls,*args)
            # print(ret)
            return ret
        # print("<<<<", [str(x.__name__) for x in args[1]])
        # members = 
        return new_op(name, members)


    def __call__(self,*args, **kwargs):
        ''' A decorator function that builds a new Op'''
        if(len(args) > 1): raise ValueError("Op() takes at most one position argument 'signature'.")
        
        def wrapper(call_pyfunc):
            assert hasattr(call_pyfunc,"__call__")

            name = call_pyfunc.__name__
            members = kwargs
            members["call"] = call_pyfunc

            op = new_op(name, members)

            # Since decorator replaces, ensure original function is pickle accessible. 
            # op.py_func = call_func
            call_pyfunc.__qualname__  = call_pyfunc.__qualname__ + ".py_func"
            # print("<<call_func", type(call_func))
            return op

        if(len(args) == 1):
            if(isinstance(args[0],(str, numba.core.typing.templates.Signature))):
                kwargs['signature'] = args[0]
            elif(hasattr(args[0],'__call__')):
                return wrapper(args[0])
            else:
                raise ValueError(f"Unrecognized type {type(args[0])} for 'signature'")        

        return wrapper

    def _get_simple_sig_str(cls):
        sig = getattr(cls,'signature',None)
        if(sig is None): return "None"
        return f'{str(sig.return_type)}({",".join([str(x) for x in sig.args])})'

    def __str__(cls):
        return f"cre.{cls.__name__}(signature={cls._get_simple_sig_str()})"

    def __repr__(cls):
        return f"cre.op.OpMeta(name={cls.__name__!r}, signature={cls._get_simple_sig_str()})"

    # def __reduce__(cls):
    #     return (self.__class__,())

        

    def _handle_nopython(cls):
        if(not hasattr(cls,'nopython_call')):
            cls.nopython_call = getattr(cls,'nopython',None)
        if(not hasattr(cls,'nopython_check')):
            cls.nopython_check = getattr(cls,'nopython',None)

    def _handle_commutes(cls):
        if(not hasattr(cls,'commutes')):
            cls.commutes = []
            cls.right_commutes = {}
            return

        arg_types = cls.signature.args
        if(isinstance(cls.commutes,bool)):
            if(cls.commutes == True):
                cls.commutes, d = [], {}
                for i, typ in enumerate(arg_types): 
                    d[typ] = d.get(typ,[]) + [i] 
                for typ, inds in d.items():
                    cls.commutes += [inds]#list(combinations(inds,2))
            else:
                cls.commutes = []
        else:
            assert(isinstance(cls.commutes,Iterable))


        right_commutes = {}#Dict.empty(i8,i8[::1])
        for i in range(len(cls.commutes)):
            commuting_set =  cls.commutes[i]
            # print(commuting_set)
            for j in range(len(commuting_set)-1,0,-1):
                right_commutes[commuting_set[j]] = np.array(commuting_set[0:j],dtype=np.int64)
                for k in commuting_set[0:j]:
                    typ1, typ2 = arg_types[k], arg_types[commuting_set[j]]
                    assert typ1==typ2, \
            f"cre.Op {cls.__name__!r} has invalid 'commutes' member {cls.commutes}. Signature arguments {k} and {commuting_set[j]} have different types {typ1} and {typ2}."
        cls.right_commutes = right_commutes
        

    def _handle_defaults(cls):
        # By default use the variable names that the user defined in the call() fn.
        cls.default_arg_names = inspect.getfullargspec(cls.call_pyfunc)[0]
        cls.default_vars = new_vars_from_types(cls.call_sig.args, cls.default_arg_names)

        arg_type_names = [str(x) for x in cls.signature.args]
        cls.arg_type_names = as_typed_list(unicode_type, arg_type_names)

        if(not hasattr(cls,'_expr_template')):
            cls._expr_template = f"{cls.__name__}({', '.join([f'{{{i}}}' for i in range(len(arg_type_names))])})"  
        
        cls._shorthand_template = cls.shorthand if(hasattr(cls,'shorthand')) else cls._expr_template

        
        


    def process_method(cls,name, sig):
        if(not hasattr(cls, name+"_pyfunc")):
            func = getattr(cls,name)
            if(isinstance(func, Dispatcher)):
                py_func = func.py_func
            else:
                py_func = func

            setattr(cls, name+"_pyfunc", py_func)
            setattr(cls, name+"_sig", sig)
            
            # Speed up by caching cloudpickle.dumps(py_func) inside py_func object
            if(hasattr(py_func,'_cre_cloudpickle_bytes')):
                setattr(cls, name+'_bytes', py_func._cre_cloudpickle_bytes)
            else:
                # print("<<", py_func)
                # print("<<", py_func.__dict__)
                # print(py_func.__loader__)
                cloudpickle_bytes = cloudpickle.dumps(py_func)
                # print(cloudpickle_bytes)
                setattr(cls, name+'_bytes', cloudpickle_bytes)
                py_func._cre_cloudpickle_bytes = cloudpickle_bytes


    @property
    def long_hash(cls):
        if(not hasattr(cls,'_long_hash')):
            long_hash = unique_hash([cls.signature, cls.call_bytes,
                    cls.check_bytes if hasattr(cls,'check') else None])

            cls._long_hash = long_hash
        return cls._long_hash


    # def __getstate__(self) = lambda self : (name, members)
    # cls.__setstate__ = lambda name, members : new_op(name, {**members}, return_class=True) 
    # def __del__(cls):

        # # pass
        # print("OP META DTOR")
        # var_ptrs_dtor(self.head_var_ptrs)

@njit(cache=True)
def var_ptrs_dtor(var_ptrs):
    for ptr in var_ptrs:
        _decref_ptr(ptr)



class Op(CREObjProxy,metaclass=OpMeta):
    ''' Base class for an functional operation. Custom operations can be
        created by subclassing Op and defining a signature, call(), and 
        optional check() method. 
    '''
    # @classmethod
    t_id = T_ID_OP

    @classmethod
    def _numba_box_(cls, ty, mi, py_cls=None):
        # print(ty, mi, py_cls)
        # instance = structref.StructRefProxy._numba_box_(cls, ty, mi)
        instance = super(structref.StructRefProxy,cls).__new__(cls)
        instance._type = ty
        instance._meminfo = mi
        # print("<<", py_cls, ty)
        if(isinstance(py_cls,OpMeta)):# is not None):
            instance.__class__ = py_cls# is not None):



        # else:
        #     raise ValueError()

        # print("CLASS:", py_cls)
        # print("<< ", type(instance))
        return instance

    def get_var(self,i):
        return get_var(self, i)

    def as_op_comp(self):
        if(hasattr(self,'op_comp')):
            return self.op_comp            
        return OpComp(self,*[self.get_var(i) for i in range(len(self.signature.args))])

    
    #     pass
    # def __call__(self,*py_args):
        
    #     if(all([not isinstance(x,(Var,OpMeta,OpComp, Op)) for x in py_args])):
    #         # If all of the arguments are constants then just call the Op
    #         return self.call(*py_args)
    #     else:
    #         # Otherwise build an OpComp and flatten it into a new Op
    #         op_comp = OpComp(self,*py_args)
    #         op = op_comp.flatten()
    #         op.op_comp = op_comp
    #         set_expr_template(op, op.gen_expr())
    #         set_shorthand_template(op, op.gen_expr(use_shorthand=True))
    #         return op

    def __call__(self,*py_args):
        # print(py_args)
        if(all([not isinstance(x,(Var,OpMeta,OpComp, Op)) for x in py_args])):
            # If all of the arguments are constants then just call the Op
            return self.call(*py_args)
        else:
            # Otherwise build an OpComp and flatten it into a new Op
            op_comp = OpComp(self,*py_args)
            op = op_comp.flatten()
            
            return op

    def __str__(self):
        return op_str(self)
        # name = get_name(self)
        # arg_names = op_get_arg_names(self)
        # return self.shorthand_template % (arg_names)
        # if(name == "__GenerateOp__"):
        #     print("THIS!!!!!", self.expr_template)
        #     return self.expr_template
        # else:
        #     return op_str(self)

    def __repr__(self):
        return op_repr(self)
        # name = get_name(self)
        # arg_names = op_get_arg_names(self)
        # return self.expr_template % (arg_names)
        # if(name == "__GenerateOp__"):
        #     return self.expr_template
        # else:
        #     return op_repr(self)
    def __del__(self):
        pass
        # print("OP DTOR")
        # op_dtor(self)

    @classmethod
    def make_singleton_inst(cls, head_vars=None):
        '''Creates a singleton instance of the user defined subclass of Op.
            These instances unbox into the NRT as GenericOpType.
        '''        
        if(isinstance(head_vars,np.ndarray)):
            head_var_ptrs = head_vars
        else:
            if(head_vars is None): 
                head_vars = cls.default_vars
            head_var_ptrs = np.array([v.get_ptr() for v in head_vars], dtype=np.int64)

        # print("---------------")
        # print(cls.__name__,
        #     str(cls.signature.return_type),
        #     cls.arg_type_names,
        #     var_ptrs,
        #     cls._expr_template,
        #     cls._shorthand_template,
        #     cls.call_addr,
        #     cls.__dict__.get('check_addr',0),)
        # Make the op instance
        return_t_id = cre_context().get_t_id(_type=cls.signature.return_type)
        op_inst = op_ctor(
            cls.__name__,
            str(cls.signature.return_type),
            u2(return_t_id),
            cls.arg_type_names,
            head_var_ptrs,
            cls.method_addrs,
            str(cls._expr_template),
            str(cls._shorthand_template),
            
            # call_addr=cls.call_addr,
            # check_addr=cls.__dict__.get('check_addr',0),
            # match_head_ptrs_addr=cls.match_head_ptrs_addr,
            )
        # print("<<", op_inst, cls._expr_template)
        
        op_inst.__class__ = cls
        # set_expr_template(op_inst, cls._expr_template)
        # set_shorthand_template(op_inst, cls._shorthand_template)
        

        # In the NRT the instance will be a GenericOpType, but on
        #   the python side we need it to be its own custom class
        

        # template_placeholders = [f'{{{i}}}' for i in range(len(arg_names))]

        #Make and store repr and shorthand expressions
        # expr_template = op_inst.gen_expr(
        #     arg_names=template_placeholders,
        #     use_shorthand=False,
        # )
        # set_expr_template(op_inst, expr_template)
        # if(hasattr(op_inst,"shorthand")):
        #     set_shorthand_template(op_inst, op_inst.gen_expr(
        #         arg_names=template_placeholders,
        #         use_shorthand=True,
        #     ))
        # else:
        #     set_shorthand_template(op_inst, expr_template)

        return op_inst

    # def recover_singleton_inst(self,context=None):
    #     '''Sometimes numba needs to emit an op instance 
    #      that isn't the singleton instance. This recovers
    #      the singleton instance from the cre_context'''
    #     context = cre_context(context)
    #     return context.op_instances[self.name]



    @property
    def name(self):
        return get_name(self)

    @property
    def expr_template(self):
        return get_expr_template(self)

    @property
    def shorthand_template(self):
        return get_shorthand_template(self)

    @property
    def base_var_map(self):
        return get_base_var_map(self)

    @property
    def head_var_ptrs(self):
        return get_head_var_ptrs(self)

    @property
    def head_vars(self):
        return get_head_vars(self)

    @property
    def head_ranges(self):
        return get_head_ranges(self)

    @property
    def nargs(self):
        return len(get_head_ranges(self))

    @property
    def return_type_name(self):
        return get_return_type_name(self)


    @property
    def arg_names(self):
        if(not hasattr(self,'_arg_names')):
            self._arg_names = extract_arg_names(self)
        return self._arg_names

    @property
    def long_hash(self):
        return self.__class__.long_hash


    @property
    def unq(self):
        '''Returns a unique tuple for the the Op, useful for use
           in dictionary keys since the Op overloads __eq__ and thus
           cannot be a dictionary key '''
        if(not hasattr(self,'_unq')):
            self._unq = (self.name, self.long_hash)
        return self._unq


    def __hash__(self):
        return hash(self.long_hash)

    def __lt__(self, other): 
        from cre.default_ops import LessThan
        return LessThan(self, other)
    def __le__(self, other): 
        from cre.default_ops import LessThanEq
        return LessThanEq(self, other)
    def __gt__(self, other): 
        from cre.default_ops import GreaterThan
        return GreaterThan(self, other)
    def __ge__(self, other):
        from cre.default_ops import GreaterThanEq
        return GreaterThanEq(self, other)
    def __eq__(self, other): 
        from cre.default_ops import Equals
        return Equals(self, other)
    def __ne__(self, other): 
        from cre.default_ops import Equals
        return ~Equals(self, other)

    def __add__(self, other):
        from cre.default_ops import Add
        return Add(self, other)

    def __radd__(self, other):
        from cre.default_ops import Add
        return Add(other, self)

    def __sub__(self, other):
        from cre.default_ops import Subtract
        return Subtract(self, other)

    def __rsub__(self, other):
        from cre.default_ops import Subtract
        return Subtract(other, self)

    def __mul__(self, other):
        from cre.default_ops import Multiply
        return Multiply(self, other)

    def __rmul__(self, other):
        from cre.default_ops import Multiply
        return Multiply(other, self)

    def __truediv__(self, other):
        from cre.default_ops import Divide
        return Divide(self, other)

    def __rtruediv__(self, other):
        from cre.default_ops import Divide
        return Divide(other, self)

    def __floordiv__(self, other):
        from cre.default_ops import FloorDivide
        return FloorDivide(self, other)

    def __rfloordiv__(self, other):
        from cre.default_ops import FloorDivide
        return FloorDivide(other, self)

    def __pow__(self, other):
        from cre.default_ops import Power
        return Power(self, other)

    def __rpow__(self, other):
        from cre.default_ops import Power
        return Power(other, self)

    def __mod__(self, other):
        from cre.default_ops import Modulus
        return Modulus(other, self)

    def __and__(self,other):
        from cre.conditions import op_to_cond, conditions_and
        self = op_to_cond(self)
        if(isinstance(other, Op)): other = op_to_cond(other)
        # print("<<", type(self), type(other))
        return conditions_and(self, other)

    def __or__(self,other):
        from cre.conditions import op_to_cond, conditions_or
        self = op_to_cond(self)
        if(isinstance(other, Op)): other = op_to_cond(other)
        return conditions_or(self, other)

    def __invert__(self):
        from cre.conditions import literal_ctor, literal_to_cond, literal_not
        return literal_to_cond(literal_not(literal_ctor(self)))

    
             

    def gen_expr(self, lang='python',
         arg_names=None, use_shorthand=False, **kwargs):
        '''Generates a one line expression for this Op (which might have been built
            from a composition of ops) in terms of user defined Ops.
            E.g. Multiply(Add(x,y),2).gen_expr() -> "Multiply(Add(x,y),2)"
                 or if shorthands are defined -> "((x+y)*2)" '''
        if(arg_names is None):
            arg_names = get_arg_seq(self).split(", ") 
        if(hasattr(self,'op_comp')):
            return self.op_comp.gen_expr(
                lang=lang,
                arg_names=arg_names,
                use_shorthand=use_shorthand,
                **kwargs)
        else:
            if( use_shorthand and 
                hasattr(self,'shorthand')):
                template = resolve_template(lang,self.shorthand,'shorthand')
                return template.format(*arg_names)
            else:
                return f"{self.name}({', '.join(arg_names)})"

    # Make Source:
    @make_source('*')
    def mk_src(self, lang='python', ind='   ', **kwargs):
        body = ''
        call = self.make_source(lang,'call')
        body += indent(self.make_source(lang,'call'),prefix=ind)
        pc = self.make_source(lang,'parent_class')
        return gen_def_class(lang, get_name(self), body, parent_classes=pc, inst_self=True)

    # Make Source: parent_class
    @make_source('python','parent_class')
    def mk_src_py_parent_class(self, **kwargs):
        return "cre.Op"
    @make_source('js','parent_class')
    def mk_src_js_parent_class(self, **kwargs):
        return "Callable"
    @make_source('*','parent_class')
    def mk_src_any_parent_class(self, **kwargs):
        return ""

    # Make Source: call
    @make_source("python", 'call')
    def mk_src_py_call(self, **kwargs):
        return dedent(inspect.getsource(self.call.py_func))

    @make_source("*", 'call')
    def mk_src_any_call(self, lang='*', ind='    ', **kwargs):
        try:
            temp = resolve_template(lang, self.call_body, 'call_body')
            call_body = f'{temp.format(*self.arg_names,ind=ind)}\n'
        except Exception as e:
            temp = resolve_template(lang, self.shorthand, 'shorthand')
            call_body = f'{ind}return {temp.format(*self.arg_names)}\n'

        return gen_def_func(lang,'call',", ".join(self.arg_names), call_body)




@njit(cache=True)
def extract_arg_names(op):
    out = List.empty_list(unicode_type)
    for v_ptr in op.base_var_map:
        v =_struct_from_ptr(GenericVarType, v_ptr)
        out.append(v.alias)
    return out






### Various jitted getters ###

@njit(cache=True)
def get_name(self):
    return self.name

@njit(cache=True)
def get_expr_template(self):
    return self.expr_template

@njit(cache=True)
def get_shorthand_template(self):
    return self.shorthand_template

@njit(cache=True)
def get_base_var_map(self):
    return self.base_var_map

@njit(cache=True)
def get_head_var_ptrs(self):
    return self.head_var_ptrs

@njit(cache=True)
def get_head_vars(self):
    return self.head_vars

@njit(cache=True)
def get_head_ranges(self):
    return self.head_ranges

@njit(cache=True)
def get_var(self,i):
    v_ptr = List(self.base_var_map.keys())[i]
    return _struct_from_ptr(GenericVarType,v_ptr)


@njit(cache=True)
def get_return_type_name(self):
    return self.return_type_name

# @njit(cache=True)
# def op_str(self):
#     return self.name + "(" + get_arg_seq(self) + ")"

@njit(cache=True)
def op_str(self):
    s = self.shorthand_template
    if(self.is_ptr_op):
        arg_names = List.empty_list(unicode_type)
        for head_var in self.head_vars:
            arg_names.append(head_var.alias)
    else:
        arg_names = op_get_arg_names(self)
    # print("arg_names", arg_names)
    for i,arg_name in enumerate(arg_names):
        s =  s.replace(f'{{{i}}}',arg_name)
    return s

# @njit(cache=True)
# def op_repr(self):
#     return self.name + "(" + get_arg_seq(self,type_annotations=True) + ")"

@njit(cache=True)
def op_repr(self):
    s = self.expr_template
    arg_names = op_get_arg_names(self)
    # print("arg_names", arg_names)
    for i,arg_name in enumerate(arg_names):
        s =  s.replace(f'{{{i}}}',f'[{arg_name}:{self.arg_type_names[i]}]')

    #Add spaces after commas
    s = s.replace(", ", ",")
    s = s.replace(",", ", ")
    return s

from cre.var import get_base_type_name

@njit(cache=True)
def get_arg_seq(self,type_annotations=False):
    s = ""
    for i,v in enumerate(self.base_vars):
        s += v.alias
        if(type_annotations): 
            # v = _struct_from_ptr(GenericVarType, self.inv_base_var_map[alias])
            s += ":" + get_base_type_name(v)#.base_type_name

        if(i < len(self.base_vars)-1): 
            s += ", " + (" " if type_annotations else "")
    return s

@njit(cache=True)
def op_get_arg_names(self):
    arg_names = List.empty_list(unicode_type)
    for v in self.base_vars:
        arg_names.append(v.alias)
    return arg_names

#### Various Setters ####
@njit(cache=True)
def set_expr_template(self, expr):
    self.expr_template = expr

@njit(cache=True)
def set_shorthand_template(self, expr):
    self.shorthand_template = expr




op_define_boxing(OpTypeClass,Op)   

def str_deref_attrs(deref_attrs):
    s = ""
    for attr in deref_attrs:
        if(attr.isdigit()):
            s += f"[{attr}]"
        else:
            s += f".{attr}"
    return s



def g_nm(x,names):
    '''Helper function that helps with putting instances of Var
        as dictionary keys by replacing them with their NRT pointers.
        This must be done because __eq__ is not a comparator on Var.'''
    if(isinstance(x, Var)):
        return names.get(x.base_ptr,None)
    else:
        return names.get(x,None)

class DerefInstr():
    '''A placeholder instruction for dereferencing'''
    def __init__(self, var):
        self.var = var
        self.deref_attrs = var.deref_attrs

    @property
    def template(self):
        return f'{{}}{str_deref_attrs(self.deref_attrs)}'

    def _get_hashable(self):
        if(not hasattr(self,"_hashable")):
            self._hashable = (self.var.base_ptr, tuple(self.deref_attrs))
        return self._hashable

    def __eq__(self, other):
        return self._get_hashable() == other._get_hashable()

    def __hash__(self):
        return hash(self._get_hashable())


# class InstrProxy():
#     '''A proxy class to wrap an Op or OpComp that safely implements 
#         __eq__ and __hash__'''
#     def __init__(self,op_comp):
#         if(isinstance(op_comp, Op)):
#             op_comp = op.as_op_comp()
#         self.op_comp = op_comp
#         self._hash = hash(self.op_comp.used_ops)

#     def __str__(self):
#         return str(self.op_comp)

#     def __repr__(self):
#         return f"InstrProxy({self.op_comp})"

#     def __eq__(self):

# @njit(cache=True)
# def var_from_ptr(ptr):
#     return _struct_from_ptr(GenericVarType,ptr)

# def extract_head_ptr(x):
#     if(isinstance(x,DerefInstr)):
#         return x.var.get_ptr_incref()
#     return x


# def gen_ranges(head_vars):
#     start,n = 0,0
#     prev_arg_ind = 0
#     for (head_inst, arg_ind, head_typ) in head_vars.values():
#         if(arg_ind == tup):
#             n += 1
#         else:
#             start += n
#             n = 0
#             yield (start, n)

            



class OpComp():
    '''A helper class representing a composition of operations
        that can be flattened into a new Op definition.'''

    def _repr_arg_helper(self,x):
        if isinstance(x, Var):
            return f'{{{self.base_vars[x.base_ptr][1]}}}'
        elif(isinstance(x, OpComp)):
            arg_names = [f'{{{self.base_vars[v_p][1]}}}' for v_p in x.base_vars]
            # print("arg_names", arg_names)
            return x.gen_expr('python',arg_names=arg_names)
        elif(isinstance(x,DerefInstr)):
            deref_attrs = x.deref_attrs
            return f'{{{self.base_vars[x.var.base_ptr][1]}}}{str_deref_attrs(deref_attrs)}'
        else:
            return repr(x) 


    def __init__(self,op,*py_args):
        '''Constructs the the core pieces of an OpComp from a set of arguments.
            These pieces are the outermost 'op', and a set of 'constants',
            'vars', and 'instructions' (i.e. other op_comps)'''
        base_vars = {} #base_var_ptr -> (var_inst, arg_index, base_type)
        head_vars = {} #head_var_ptr or DerefInstr -> (var_inst, arg_index, head_type)

        # head_var_ptrs = []
        constants = {}
        instructions = {}
        args = []
        arg_types = []
        n_terms = 0
        n_ops = 1
        depth = 1
        
        for i, x in enumerate(py_args):
            if(isinstance(x, Op)): x = x.as_op_comp()
            if(isinstance(x, OpComp)):
                n_terms += x.n_terms
                n_ops += x.n_ops
                depth = max(x.depth+1, depth)
                for v_ptr,(v,_,t) in x.base_vars.items():
                    if(v_ptr not in base_vars):
                        base_vars[v_ptr] = (v,len(arg_types),t)
                        arg_types.append(t)

                for c,t in x.constants.items():
                    constants[c] = t
                for instr, sig in x.instructions.items():
                    instructions[instr] = sig
                for vptr_or_instr, (head_var,_,typ) in x.head_vars.items():
                    if(vptr_or_instr not in head_vars):
                        arg_ind = base_vars[head_var.base_ptr][1]
                        head_vars[vptr_or_instr] = (head_var, arg_ind, typ)
                        # head_var_ptrs.append(extract_head_ptr(vptr_or_instr))
            else:
                if(isinstance(x,Var)):
                    n_terms += 1
                    b_ptr = x.base_ptr
                    if(b_ptr not in base_vars):
                        t = x.base_type
                        base_vars[b_ptr] = (x,len(arg_types),t)
                        arg_types.append(t)
                    
                    if(len(x.deref_infos) > 0):
                        d_instr = DerefInstr(x)
                        if(d_instr not in head_vars): 
                            arg_ind = base_vars[b_ptr][1]
                            head_vars[d_instr] = (x, arg_ind, x.head_type)

                            # head_var_ptrs.append(x.get_ptr())
                        instructions[d_instr] = x.head_type(x.base_type,)
                        x = d_instr
                    else:
                        if(b_ptr not in head_vars):
                            arg_ind = base_vars[b_ptr][1]
                            head_vars[b_ptr] =  (x, arg_ind ,x.head_type)
                            # head_var_ptrs.append(x.get_ptr())
                else:
                    constants[x] = op.signature.args[i]
            args.append(x)


        
        head_vars = {k:v for i,(k,v) in enumerate(sorted(head_vars.items(), key=lambda x: x[1][1]))}

        # print("<<",_vars)
        self.n_terms = n_terms
        self.n_ops = n_ops
        self.depth = depth
        self.op = op
        self.base_vars = base_vars
        # print(self.base_vars)
        self.head_vars = head_vars
        self.args = args
        self.arg_types = arg_types
        # self.head_types = head_types
        # print([x[0].get_ptr() for x in head_vars.values()])
        # self.head_var_ptrs = np.array([x[0].get_ptr_incref() for x in head_vars.values()],dtype=np.int64)
        self.constants = constants
        self.signature = op.signature.return_type(*arg_types)
        

        self.expr_template = f"{op.name}({', '.join([self._repr_arg_helper(x) for x in self.args])})"  
        self.name = self.expr_template

        instructions[self] = op.signature

        # print("Num_terms", n_terms)
        
        self.instructions = instructions


    def flatten(self, return_class=False):
        ''' Flattens the OpComp into a single Op. Generates the source
             for the new Op as needed.'''
        if(not hasattr(self,'_generate_op')):
            # print([type(x) for x in self.used_ops])

            long_hash = unique_hash([self.expr_template,*[op.unq for _,op in self.used_ops.values()]])
            # print(long_hash)
            if(not source_in_cache('__GenerateOp__', long_hash)):
                source = self.gen_flattened_op_src(long_hash)
                source_to_cache('__GenerateOp__', long_hash, source)
            l = import_from_cached('__GenerateOp__', long_hash, ['__GenerateOp__'])
            op_cls = self._generate_op_cls = l['__GenerateOp__']
            place_holder_arg_names = [f'{{{i}}}' for i in range(len(self.base_vars))]
            op_cls._expr_template = self.gen_expr(arg_names=place_holder_arg_names)
            op_cls._shorthand_template = self.gen_expr(arg_names=place_holder_arg_names, use_shorthand=True)

            #Dynamically make subclass
            # op_cls = type("__GenerateOp__", op_cls.__bases__,dict(op_cls.__dict__))
            # op_cls.op_comp = self
            
            # print(op_cls._expr_template)
            
            if(return_class):
                return op_cls

            # print("<<",type(op_cls))
            # print(get_cache_path('__GenerateOp__',long_hash))

            # var_ptrs = np.empty(len(self.base_vars),dtype=np.int64)
            # for i,v_p in enumerate(self.base_vars):
            #     var_ptrs[i] = v_p#v.get_ptr()

            op = self._generate_op = op_cls.make_singleton_inst([x[0] for x in self.head_vars.values()])
            op.op_comp = self
            
            
        return self._generate_op

    @property
    def used_ops(self):
        ''' Returns a dictionary keyed by ops used in the OpComp expression.
            values are unique integers 0,1,2 etc.'''
        if(not hasattr(self,'_used_ops')):
            _used_ops = {}
            # Since __equal__() is overloaded it's best to use .unq as unqiue keys
            # used_unqs = set() 
            oc = 0
            for i,instr in enumerate(self.instructions):
                if(isinstance(instr, OpComp) and instr.op.unq not in _used_ops):
                    _used_ops[instr.op.unq] = (oc,instr.op)
                    oc += 1
            # print("**", _used_ops)
            self._used_ops = _used_ops
        return self._used_ops

    #### Code Generation Methods ####
    def gen_op_imports(self,lang='python'):
        ''' Generates import expressions for any 'used_ops' in the OpComp'''
        op_imports = ""
        for i,(_,op) in enumerate(self.used_ops.values()):
            to_import = {"call_sig" : f'call_sig{i}',
                "call" : f'call{i}'}
            if(hasattr(op,"check")):
                to_import.update({"check" : f'check{i}', "check_sig" : f'check_sig{i}'})

            op_imports += gen_import_str(op.name, op.long_hash, to_import) + "\n"
        return op_imports

        
    def _gen_constant_defs(self,lang,names={},ind='    '):
        constant_defs = ''
        for i,(c,t) in enumerate(self.constants.items()):
            names[c] = f'c{i}'
            constant_defs += f"{ind}{gen_assign(lang, f'c{i}', f'{c!r}')}\n"
        return constant_defs

    def _gen_arg_seq(self, lang, arg_names=None, names={}):
        if(arg_names is None):
            arg_names = [f'a{i}' for i in range(len(self.base_vars))]
        for i,(v_p,(v,_,t)) in enumerate(self.base_vars.items()):
            names[v_p] = arg_names[i]
            # print(i, v_p, arg_names[i], len(self.base_vars))
        return arg_names

        
    def _call_check_prereqs(self,lang, arg_names=None,
                            op_fnames=None,
                            op_call_fnames=None,
                            op_check_fnames=None,
                            skip_consts=False,
                            skip_deref_instrs=False,
                            **kwargs):
        ''' Helper function that fills 'names' which is a dictionary of 
            various objects to their equivalent string expressions, in 
            addition to 'arg_names' which is autofilled as needed, 
            and optionally 'const_defs'. '''
        names = {}
        
        if(not skip_deref_instrs):
            arg_names = self._gen_arg_seq(lang, arg_names, names)
            for i,instr in enumerate(self.instructions):
                names[instr] = f'i{i}'
        else:
            # arg_names = self._gen_arg_seq(lang, arg_names, names)
            arg_names = [f'a{i}' for i in range(len(self.head_vars))]
            # print(">>",self.head_types)
            for i, var_ptr_or_instr in enumerate(self.head_vars):
                # if(isinstance(var_or_instr, Var)):
                    # names[var_ptr_or_instr] = arg_names[i]
                names[var_ptr_or_instr] = arg_names[i]
                # else:
                    # names[var_or_instr] = arg_names[i]

            for i,instr in enumerate(self.instructions):
                if(not isinstance(instr, DerefInstr)):
                    names[instr] = f'i{i}'

            #     print("<<", type(instr))
            #     if(isinstance(instr, OpComp)):
            #         names[instr] = f'a{i}'
            #         # arg_names.append(f'a{i}')
            #     else:
            #         names[instr] = f'i{i}'


            # assert len(arg_names) == len(self.head_types)

        if(op_call_fnames is None):
            op_call_fnames = {op : f'{op.name}.call' for _,op in self.used_ops.values()}
        for op, n in op_call_fnames.items(): names[(op.unq,'call')] = n
            
        if(op_check_fnames is None):
            op_check_fnames = {op : f'{op.name}.check' for _,op in self.used_ops.values()}
        for op, n in op_check_fnames.items(): names[(op.unq,'check')] = n
            
        if(op_fnames is None):
            op_fnames = {op : op.name for _,op in self.used_ops.values()}
        for op, n in op_fnames.items(): names[op.unq] = n
            
        if(skip_consts):
            return names, arg_names
        else:
            const_defs = self._gen_constant_defs(lang,names)
            return names, arg_names, const_defs

    def gen_expr(self, lang='python', **kwargs):
        '''Generates a oneline expression for the OpComp'''        
        names, arg_names = self._call_check_prereqs(lang, skip_consts=True, **kwargs)
        for i,instr in enumerate(self.instructions):
            instr_reprs = []
            if(isinstance(instr, DerefInstr)):
                names[instr] = instr.template.format(g_nm(instr.var,names))
            else:
                for x in instr.args:
                    if(g_nm(x,names) is not None):
                        instr_reprs.append(g_nm(x,names))
                    elif(isinstance(x,(Op,OpComp))):
                        instr_reprs.append(x.gen_expr(lang,**kwargs))
                    else:
                        instr_reprs.append(repr(x))

                instr_kwargs = {**kwargs,'arg_names':instr_reprs}
                names[instr] = instr.op.gen_expr(lang=lang,**instr_kwargs) #f'{names[]}({instr_names})'
        return names[list(self.instructions.keys())[-1]]


    def _gen_instrs_helper(self, names, lang='python', on_check_fail='return 0', ind='    ', **kwargs):
        ''' Handles the source generation for instructions such as Ops and DerefInstrs.'''
        body = ""
        for i,instr in enumerate(self.instructions):
            if(isinstance(instr, DerefInstr)):
                if(kwargs.get('skip_deref_instrs',False)): continue

                # On a 'DerefInstr' assign 'i{i}' to the head of instr.var (e.g. `i0 = a0.value`)
                deref_str = f'{g_nm(instr.var,names)}{str_deref_attrs(instr.deref_attrs)}'
                body += f"{ind}{gen_assign(lang, f'i{i}', deref_str)}\n"
            else:
                if(kwargs.get('skip_op_instrs',False)): continue

                # Delimited input sequence for check/call e.g. `a0,i0` as in `call(a0,i0)`
                inp_seq = ", ".join([g_nm(x,names) for x in instr.args])

                if(hasattr(instr.op,'check') and on_check_fail != ""):
                    # Assign a variable for the result of applying check e.g. `k0 = check(a0,a1)`
                    check_fname = names[(instr.op.unq,'check')]
                    body += f"{ind}{gen_assign(lang, f'k{i}', f'{check_fname}({inp_seq}')})\n"

                    # If that variable is false do something e.g. `if(not k0): return 0`
                    body += f"{ind}{gen_if(lang, cond=gen_not(lang,f'k{i}'), rest=on_check_fail, newline='')}\n"

                # Assign a variable for the result of applying call e.g. `i0 = call0(a0,i0)`
                call_fname = names[(instr.op.unq,'call')]
                body += f"{ind}{gen_assign(lang, f'i{i}', f'{call_fname}({inp_seq})')}\n"
        return body

    
    def gen_base_func_from_head_func(self, fname, head_fname,  lang='python',  ind='    ', **kwargs):
        ''' Creates source for a function that applies the op's deref instructions (e.g. `i0 = a0.value`) 
            to get the head values then applies the function with `head_fname` on those. For instance:
            ```
            @njit(call_sig,cache=True)
            def call(a0, a1):
                i0 = a0.val
                return call_heads(i0, a0, a1) ```
        '''
        names, arg_names, const_defs = self._call_check_prereqs(lang, **kwargs)
        body = ""
        body = self._gen_instrs_helper(names, lang=lang, skip_op_instrs=True, ind=ind, **kwargs)
        inp_seq = ", ".join([g_nm(v_ptr_or_dref,names) for v_ptr_or_dref in self.head_vars])
        tail = f"{ind}return {head_fname}({inp_seq})\n"
        return gen_def_func(lang, fname, ", ".join(arg_names), body, tail)        

    def gen_head_ptr_wrapper(self, fname, head_fname, head_type_names=None, lang='python',  ind='    ', **kwargs):
        '''Creates source for a function that takes in an array of pointers, loads the pointers
            by the appropriate types and then calls the function with 'head_fname' on them: For instance:
            ```
            @njit(boolean(i8[::1],),cache=True)
            def match_head_ptrs(ptrs):
                i0 = _load_ptr(h0_type,ptrs[0])
                i1 = _load_ptr(h1_type,ptrs[1])
                i2 = _load_ptr(h2_type,ptrs[2])
                return match_heads(i0,i1,i2) ```
        '''
        names, arg_names, const_defs = self._call_check_prereqs(lang, skip_deref_instrs=True, **kwargs)
        n_heads = len(self.head_vars)

        # List of head_type names
        if(head_type_names is None): 
            head_type_names = [f"h{i}_type" for i in range(n_heads)] 

        # Make _load_ptr() instructions (e.g. `i0 = _load_ptr(h0_type,ptrs[0])`)
        body = ""
        for i in range(n_heads):
            load_str = f'_load_ptr({head_type_names[i]},ptrs[{i}])'
            body += f"{ind}{gen_assign(lang,f'i{i}',load_str)}\n"
        inps = ",".join([f'i{i}' for i in range(len(arg_names))])
        tail = f"{ind}return {head_fname}({inps})\n"
        return gen_def_func(lang, fname, 'ptrs', body, tail)

    def gen_call(self, lang='python', fname="call", ind='    ', **kwargs):
        '''Generates source for the equivalent call function for the OpComp'''
        names, arg_names, const_defs = self._call_check_prereqs(lang, **kwargs)
        body = const_defs
        body += self._gen_instrs_helper(names, lang=lang, on_check_fail='raise Exception()', ind=ind, **kwargs)

        tail = f'{ind}return i{len(self.instructions)-1}\n'
        return gen_def_func(lang, fname, ", ".join(arg_names), body, tail)


    def gen_match(self, lang='python', fname="match", ind='    ', **kwargs):
        '''Generates source for the equivalent match function for the OpComp'''
        names, arg_names, const_defs = self._call_check_prereqs(lang, **kwargs)
        # print(names)
        body = const_defs
        body += self._gen_instrs_helper(names, lang=lang,on_check_fail='return 0', ind=ind, **kwargs)

        tail = f"{ind}return True if i{len(self.instructions)-1} else False\n"
        return gen_def_func(lang, fname, ", ".join(arg_names), body, tail)


    def gen_check(self, lang='python', fname="check", ind='    ', **kwargs):
        '''Generates source for the equivalent check function for the OpComp'''
        names, arg_names, const_defs = self._call_check_prereqs(lang, **kwargs)
        body = const_defs
        body += self._gen_instrs_helper(names, lang=lang,on_check_fail='return 0', ind=ind, **kwargs)
        
        tail = f"{ind}return True\n" 
        return gen_def_func(lang, fname, ", ".join(arg_names), body, tail)

    
    def gen_flattened_op_src(self, long_hash, ind='    '):
        '''Generates python source for the equivalant flattened Op'''
        has_check = any([hasattr(op, 'check') for _,op in self.used_ops.values()])

        op_imports = self.gen_op_imports('python')
        op_call_fnames = {op : f'call{j}' for j,op in self.used_ops.values()}
        op_check_fnames = {op : f'check{j}' for j,op in self.used_ops.values()} if(has_check) else None
            

        call_heads_src = self.gen_call('python', fname='call_heads', skip_deref_instrs=True,
            op_call_fnames=op_call_fnames, op_check_fnames=op_check_fnames)
        call_head_ptrs_src = self.gen_head_ptr_wrapper("call_head_ptrs", "call_heads")
        call_src = self.gen_base_func_from_head_func("call", "call_heads")
        
        match_heads_src = self.gen_match('python', fname="match_heads",
             skip_deref_instrs=True, op_call_fnames=op_call_fnames, op_check_fnames=op_check_fnames)
        match_head_ptrs_src = self.gen_head_ptr_wrapper("match_head_ptrs", "match_heads")
        match_src = self.gen_base_func_from_head_func("match", "match_heads")

        if(has_check):
            check_src = self.gen_check('python', op_call_fnames=op_call_fnames, op_check_fnames=op_check_fnames)

        return_type = list(self.instructions.values())[-1].return_type
        call_sig = return_type(*self.arg_types)
        check_sig = u1(*self.arg_types)
        head_types = [x[2] for x in self.head_vars.values()]
        # print("-------")
        # print(head_types)
        # print(call_sig)
        # print(call_src)
        # print(call_sig)
# 
# boolean(*head_arg_types)
        source = f'''import numpy as np
from numba import i8, boolean, njit
from numba.types import unicode_type
from numba.extending import intrinsic
from numba.experimental.function_type import _get_wrapper_address
from numba.core import cgutils
from cre.op import Op
from cre.utils import _load_ptr
import cloudpickle
{op_imports}

call_sig = cloudpickle.loads({cloudpickle.dumps(call_sig)})
head_types = cloudpickle.loads({cloudpickle.dumps(head_types)})

{"".join([f'h{i}_type,' for i in range(len(head_types))])} = head_types

method_addrs = np.zeros((7,),dtype=np.int64)

@njit(call_sig.return_type(*head_types),cache=True)
{call_heads_src}

@njit(call_sig.return_type(i8[::1]), cache=True)
{call_head_ptrs_src}

@njit(call_sig,cache=True)
{call_src}

@njit(boolean(*head_types), cache=True)
{match_heads_src}

@njit(boolean(i8[::1],),cache=True)
{match_head_ptrs_src}

@njit(boolean(*call_sig.args),cache=True)
{match_src}

method_addrs[0] = _get_wrapper_address(call_heads, call_sig.return_type(*head_types))
method_addrs[1] = _get_wrapper_address(call_head_ptrs, call_sig.return_type(i8[::1]))
method_addrs[2] = _get_wrapper_address(call, call_sig)
method_addrs[3] = _get_wrapper_address(match_heads, boolean(*head_types))
method_addrs[4] = _get_wrapper_address(match_head_ptrs, boolean(i8[::1],))
method_addrs[5] = _get_wrapper_address(match, boolean(*call_sig.args))
    
'''
        if(has_check): source +=f'''
@njit(boolean(*call_sig.args),cache=True)
{check_src}

method_addrs[6] = _get_wrapper_address(check, boolean(*call_sig.args))
'''
        source += f'''

class __GenerateOp__(Op):
    signature = call_sig
    call = call
    {"check = check" if(has_check) else ""}
    method_addrs = method_addrs
    _long_hash = {long_hash!r}
    



'''
        return source

    
    def __hash__(self):
        return hash(self.expr_template)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.expr_template

    def __call__(self,*args):
        flattened_inst = self.flatten()
        # print("FALT")
        return flattened_inst(*args)

    def __del__(self):
        pass
        # print("OP COMP DTOR")
        # var_ptrs_dtor(self.head_var_ptrs)
# 





'''
@njit(call_sig,cache=True)
def call(a0):
    i0 = a0.B
    i1 = call0(i0, i0)
    return i1


@njit(call_sig.return_type(i8[::1]), cache=True)
def call_head_ptrs(ptrs):
    i0 = _load_ptr(h0_type,ptrs[0])
    return call0(i0)


@njit(boolean(*head_types), cache=True)
def match_heads(a0):
    i1 = call0(a0, a0)
    return True if i1 else False


@njit(boolean(*call_sig.args),cache=True)
def match(a0):
    i0 = a0.B
    return match_heads(i0)


@njit(boolean(i8[::1],),cache=True)
def match_head_ptrs(ptrs):
    i0 = _load_ptr(h0_type,ptrs[0])
    return match_heads(i0)

THINKING THINKING 

So for each thing call, check, apply, match we need:
-normal
-head_ptrs
-heads



call_heads
call_head_ptrs
call

check_heads
check_head_ptrs
check

match_heads
match_head_ptrs
match


def match

'''
