from numba import types, njit, i8, i4, i2, i1, u2, f8
from numba.types import unicode_type, boolean
from numba.extending import (models, register_model, type_callable,
        typeof_impl, models, register_model, make_attribute_wrapper,
        overload_attribute, lower_builtin, unbox, box, NativeValue,
        overload_method, overload, intrinsic)
from numba.core.datamodel.registry import register_default
from numba.core import cgutils
from numba.cpython.unicode import _malloc_string, _strncpy, _kind_to_byte_width
from llvmlite import ir
from llvmlite.ir import types as ll_types
from llvmlite.ir import Constant
from cre.core import DEFAULT_TYPE_T_IDS, DEFAULT_T_ID_TYPES, T_ID_CONDITIONS, T_ID_LITERAL, T_ID_FUNC, T_ID_FACT, T_ID_VAR, T_ID_UNDEFINED, T_ID_BOOL, T_ID_INT, T_ID_FLOAT, T_ID_STR, T_ID_TUPLE_FACT
from cre.utils import cast, PrintElapse
from cre.obj import CREObjType
from numba.core.datamodel import default_manager, models
from numba.experimental.structref import _Utils, define_attributes

from numba.core.imputils import (lower_builtin, lower_getattr,
                                 lower_getattr_generic,
                                 lower_setattr_generic,
                                 lower_cast, lower_constant,
                                 iternext_impl, impl_ret_borrowed,
                                 impl_ret_new_ref, impl_ret_untracked,
                                 RefType)
import inspect


T_ID_UNDEFINED = DEFAULT_TYPE_T_IDS['undefined']
T_ID_BOOL = DEFAULT_TYPE_T_IDS['bool']
T_ID_INT = DEFAULT_TYPE_T_IDS['int'] 
T_ID_FLOAT = DEFAULT_TYPE_T_IDS['float']
T_ID_STR = DEFAULT_TYPE_T_IDS['str']
T_ID_CRE_OBJ = DEFAULT_TYPE_T_IDS['CREObj']
T_ID_FACT = DEFAULT_TYPE_T_IDS['Fact']
T_ID_TUPLE_FACT = DEFAULT_TYPE_T_IDS['TupleFact']
T_ID_VAR = DEFAULT_TYPE_T_IDS['Var']
T_ID_FUNC = DEFAULT_TYPE_T_IDS['CREFunc']
T_ID_LITERAL = DEFAULT_TYPE_T_IDS['Literal']
T_ID_CONDITIONS = DEFAULT_TYPE_T_IDS['Conditions']
T_ID_RULE = DEFAULT_TYPE_T_IDS['Rule']


# ------------------------------
# : Struct Definitions for Member

# Stub Class For Member Constructor
class Member(object):
    pass

class MemberTypeClass(types.Type):
    def __init__(self, val_type=None):
        self.val_type = val_type
        name = f'Member[{val_type}]' if val_type is not None else 'Member'
        super(MemberTypeClass, self).__init__(name=name)


@register_model(MemberTypeClass)
class MemberModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # Unicode Layout
        if(fe_type.val_type == unicode_type):
            members = [
                ('val',      types.int64),
                ('t_id',     types.uint16),
                ('kind',     types.uint8), 
                ('is_ascii', types.uint8), 
                ('length',   types.uint32), 
            ]

        # Standard Layout
        else:
            members = [
                ('val',       types.int64),
                ('t_id',      types.uint16),
                ('pad1',      types.uint8), 
                ('pad2',      types.uint8), 
                ('t_id_infs', types.uint32), 
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)

MemberType = MemberTypeClass()
MemberTypeUnicode = MemberTypeClass(unicode_type)

make_attribute_wrapper(MemberTypeClass, 'val', 'val')
make_attribute_wrapper(MemberTypeClass, 't_id', 't_id')

# ------------------------------------
# : Member(...) constructor

# Set the return type for 
@type_callable(Member)
def type_member(context):
    def typer(val_type):
        # Return Empty if given TypeRef
        if(isinstance(val_type, types.TypeRef)):
            val_type = val_type.instance_type

        if(isinstance(val_type, types.DTypeSpec)):
            val_type = val_type.dtype

        # Promote Integers to 64-bit
        if(isinstance(val_type, types.Integer)):
            return MemberTypeClass(i8)
        # Promote Floats to 64-bit
        if(isinstance(val_type, types.Float)):
            return MemberTypeClass(f8)
        return MemberTypeClass(val_type)
    return typer

@lower_builtin(Member, types.TypeRef)
@lower_builtin(Member, types.DTypeSpec)
def impl_member(context, builder, sig, args):
    member = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    t_id = DEFAULT_TYPE_T_IDS[sig.return_type.val_type]
    member.t_id = ir.Constant(ir.IntType(16), t_id)
    return member._getvalue()

@lower_builtin(Member, types.Boolean)
@lower_builtin(Member, types.Integer)
def impl_member(context, builder, sig, args):
    member = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    val = context.cast(builder, args[0], sig.args[0], types.int64)
    member.val = builder.bitcast(val, cgutils.intp_t)
    t_id = T_ID_INT if isinstance(sig.args[0], types.Integer) else T_ID_BOOL
    member.t_id = ir.Constant(ir.IntType(16), t_id)
    return member._getvalue()

@lower_builtin(Member, types.Number)
def impl_member(context, builder, sig, args):
    member = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    val = context.cast(builder, args[0], sig.args[0], types.float64)
    member.val = builder.bitcast(args[0], cgutils.intp_t)
    member.t_id = ir.Constant(ir.IntType(16), T_ID_FLOAT)
    return member._getvalue()


meminfo_type = types.MemInfoPointer(types.voidptr)
@lower_builtin(Member, types.UnicodeType)
def impl_member(context, builder, sig, args):
    in_str = cgutils.create_struct_proxy(unicode_type)(context, builder, value=args[0])
    member = cgutils.create_struct_proxy(sig.return_type)(context, builder)

    meminfo = in_str.meminfo

    meminfo_ptr = builder.ptrtoint(meminfo, cgutils.intp_t)
    is_const_str = builder.icmp_unsigned('==', meminfo_ptr, meminfo_ptr.type(0))

    with builder.if_else(is_const_str) as (is_const, not_const):
        # If the input string is compile-time const then copy it.
        with is_const:
            def copy_str(s):
                char_width = _kind_to_byte_width(s._kind)
                tmp_str = _malloc_string(s._kind, char_width, s._length, s._is_ascii)
                _strncpy(tmp_str, 0, s, 0, s._length)
                return tmp_str

            tmp_str = context.compile_internal(builder, copy_str, unicode_type(unicode_type,), (args[0],))
            tmp_str = cgutils.create_struct_proxy(unicode_type)(context, builder, value=tmp_str)
            member.val = builder.ptrtoint(tmp_str.meminfo, cgutils.intp_t)
                        
        # Otherwise borrow a reference to it.
        with not_const:
            if context.enable_nrt:
                context.nrt.incref(builder, meminfo_type, meminfo)
            member.val = meminfo_ptr

    member.t_id = ir.Constant(ir.IntType(16), T_ID_STR)
    member.length = context.cast(builder, in_str.length, types.int64, types.int32)
    member.kind = context.cast(builder, in_str.kind, types.int32, types.int8)
    member.is_ascii = context.cast(builder, in_str.is_ascii, types.int32, types.int8)

    return member._getvalue()

# ------------------------------------
# : .get_val(...) 

@intrinsic
def str_mbr_to_unicode_type(typingctx, member):
    from numba.cpython.unicode import PY_UNICODE_1BYTE_KIND
    def codegen(context, builder, sig, args):
        in_mbr = cgutils.create_struct_proxy(MemberTypeUnicode)(context, builder, value=args[0])
        
        # Copy info into new unicode_type 
        out_str = cgutils.create_struct_proxy(unicode_type)(context, builder)
        out_str.meminfo = builder.inttoptr(in_mbr.val, cgutils.voidptr_t)

        is_null = builder.icmp_unsigned('==', in_mbr.val, in_mbr.val.type(0))        
        with builder.if_else(is_null) as (is_null, not_null):
            with is_null:
                out_str.data = context.insert_const_bytes(builder.module, bytes())
                out_str.kind = out_str.kind.type(PY_UNICODE_1BYTE_KIND)
            with not_null:
                out_str.data = context.nrt.meminfo_data(builder, out_str.meminfo)
                out_str.kind = context.cast(builder, in_mbr.kind, types.int8, types.int32)

        out_str.length = context.cast(builder, in_mbr.length, types.int32, types.int64)
        out_str.is_ascii = context.cast(builder, in_mbr.is_ascii, types.int8, types.int32)
        out_str.hash = out_str.hash.type(-1)
        out_str.parent = cgutils.get_null_value(out_str.parent.type)

        # Borrow reference
        if context.enable_nrt:
            context.nrt.incref(builder, meminfo_type, out_str.meminfo)

        return out_str._getvalue()
    return unicode_type(member,), codegen

@intrinsic
def _bitcast_float64(typingctx, val):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], ir.DoubleType())
    return f8(val,), codegen 

@overload_method(MemberTypeClass, 'is_none')
def get_val_overload(self, typ=None):
    def impl(self, typ=None):        
        return self.t_id > T_ID_STR and self.val == 0
    return impl

@overload_method(MemberTypeClass, 'get_val')
def get_val_overload(self, typ=None):
    val_type = self.val_type
    
    if(val_type is None and typ is None):
        raise ValueError("Must provide return type to get_val() for untyped Member.")

    if(typ is None or isinstance(typ, types.Omitted)):
        _typ = val_type
    else:
        _typ = typ.instance_type
        if(val_type is not None and typ is not None and val_type is not _typ):
            raise ValueError(f"Incompatable get_val() return type {_typ!r} for {self}.")

    typ_t_id = DEFAULT_TYPE_T_IDS[_typ]
    # print("WHAA", typ_t_id, _typ)

    @njit
    def check_t_id(self):
        if(self.t_id != typ_t_id):
            print("BEF", self.t_id, typ_t_id)
            raise Exception("Member t_id does not match return type.")

    @njit
    def check_not_none(self):   
        if(self.val == 0):
            raise Exception("get_val() failed for Member with None value.")

    #  Int/Bool cases
    if(typ_t_id in [T_ID_BOOL, T_ID_INT]):
        def impl(self, typ=None):
            check_t_id(self)
            return _typ(self.val)

    #  Float cases
    elif(typ_t_id == T_ID_FLOAT):
        def impl(self, typ=None):
            check_t_id(self)    
            return _bitcast_float64(self.val)

    # Str Case
    elif(typ_t_id == T_ID_STR):
        def impl(self, typ=None):
            check_t_id(self)
            s = str_mbr_to_unicode_type(self)
            return s
    #TODO: List, Dict case

    # Fact Case
    else:
        def impl(self, typ=None):
            check_not_none(self)
            return cast(self.val, _typ)

    return impl

if __name__ == "__main__":
    @njit()
    def foo(x):
        mbr = Member(-67)
        print(mbr.get_val(), mbr.get_val(i8), mbr.val, mbr.t_id)

        mbr = Member(i2(-67))
        print(mbr.get_val(), mbr.get_val(i8), mbr.val, mbr.t_id)
        
        mbr = Member(True)
        print(mbr.get_val(), mbr.get_val(boolean), mbr.val, mbr.t_id)

        mbr = Member(67.7)
        print(mbr.get_val(), mbr.get_val(f8), mbr.val, mbr.t_id)

        mbr = Member(-67.7)
        print(mbr.get_val(), mbr.get_val(f8), mbr.val, mbr.t_id)

        mbr = Member(67)
        print(mbr.get_val(), mbr.get_val(i8), mbr.val, mbr.t_id)

        mbr = Member("123")
        print(str_mbr_to_unicode_type(mbr))
        print(mbr.get_val(), mbr.get_val(unicode_type), mbr.val, mbr.t_id)

        mbr = Member(x)
        print(mbr.get_val(), mbr.get_val(unicode_type), mbr.val, mbr.t_id)

        # mbr = Member(boolean)
        mbr = Member(i8)
        mbr = Member(f8)
        mbr = Member(unicode_type)

        print(mbr.get_val())

    foo("123")

    from numba.typed import List
    lst = List.empty_list(unicode_type)
    lst.append("A")
    lst.append("B")
    lst.append("C")
    lst.append("D")

    @njit()
    def bleep(lst):
        tup = (lst[0],lst[1],lst[2],lst[3])
        for i in range(10000):
            s = tup[0] + tup[3]
        return s

    @njit()
    def blarg(lst):
        tup = (Member(lst[0]),Member(lst[1]),Member(lst[2]),Member(lst[3]))
        for i in range(10000):
            s = tup[0].get_val(unicode_type) + tup[3].get_val(unicode_type)
