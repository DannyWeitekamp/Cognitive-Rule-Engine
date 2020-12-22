from numba import types
from numba.experimental.structref import _Utils, imputils
from numba.extending import intrinsic
from numba.core import cgutils


meminfo_type = types.MemInfoPointer(types.voidptr)

@intrinsic
def lower_setattr(typingctx, inst_type, attr_type, val_type):
    if (isinstance(attr_type, types.Literal) and 
        isinstance(inst_type, types.StructRef)):
        
        attr = attr_type.literal_value
        def codegen(context, builder, sig, args):
            [instance, attr_v, val] = args

            utils = _Utils(context, builder, inst_type)
            dataval = utils.get_data_struct(instance)
            # cast val to the correct type
            field_type = inst_type.field_dict[attr]
            casted = context.cast(builder, val, val_type, field_type)

            # read old
            old_value = getattr(dataval, attr)
            # incref new value
            context.nrt.incref(builder, val_type, casted)
            # decref old value (must be last in case new value is old value)
            context.nrt.decref(builder, val_type, old_value)
            # write new
            setattr(dataval, attr, casted)
        sig = types.void(inst_type, types.literal(attr), val_type)
        return sig, codegen


@intrinsic
def _struct_from_meminfo(typingctx, struct_type, meminfo):
    inst_type = struct_type.instance_type

    def codegen(context, builder, sig, args):
        _, meminfo = args

        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo
        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        return st._getvalue()

    sig = inst_type(struct_type, types.MemInfoPointer(types.voidptr))
    return sig, codegen


@intrinsic
def _meminfo_from_struct(typingctx, val):
    def codegen(context, builder, sig, args):
        [td] = sig.args
        [d] = args

        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder, value=d)
        meminfo = dstruct.meminfo
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)
        # Returns the plain MemInfo
        return meminfo
        
    sig = meminfo_type(val,)
    return sig, codegen


@intrinsic
def _cast_structref(typingctx, cast_type_ref, inst_type):
    # inst_type = struct_type.instance_type
    cast_type = cast_type_ref.instance_type
    def codegen(context, builder, sig, args):
        # [td] = sig.args
        _,d = args

        ctor = cgutils.create_struct_proxy(inst_type)
        dstruct = ctor(context, builder, value=d)
        meminfo = dstruct.meminfo
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        st = cgutils.create_struct_proxy(cast_type)(context, builder)
        st.meminfo = meminfo
        #NOTE: Fixes sefault but not sure about it's lifecycle (i.e. watch out for memleaks)
        # context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)

        return st._getvalue()
    sig = cast_type(cast_type_ref, inst_type)
    return sig, codegen
