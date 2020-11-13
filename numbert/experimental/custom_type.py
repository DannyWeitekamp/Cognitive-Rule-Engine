from numba.typed import Dict
from numba import i8

class Interval(object):
    """
    A half-open interval on the real number line.
    """
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return 'Interval(%f, %f)' % (self.lo, self.hi)

    @property
    def width(self):
        return self.hi - self.lo


from numba import types

class IntervalType(types.Type):
    def __init__(self):
        super(IntervalType, self).__init__(name='Interval')

interval_type = IntervalType()

from numba.extending import typeof_impl

@typeof_impl.register(Interval)
def typeof_index(val, c):
    return interval_type


from numba.extending import type_callable

@type_callable(Interval)
def type_interval(context):
    def typer(lo, hi):
        if isinstance(lo, types.Float) and isinstance(hi, types.Float):
            return interval_type
    return typer


from numba.extending import models, register_model

@register_model(IntervalType)
class IntervalModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('lo', types.float64),
            ('hi', types.float64),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)


from numba.extending import make_attribute_wrapper

make_attribute_wrapper(IntervalType, 'lo', 'lo')
make_attribute_wrapper(IntervalType, 'hi', 'hi')


from numba.extending import overload_attribute

@overload_attribute(IntervalType, "width")
def get_width(interval):
    def getter(interval):
        return interval.hi - interval.lo
    return getter


from numba.extending import lower_builtin
from numba.core import cgutils

@lower_builtin(Interval, types.Float, types.Float)
def impl_interval(context, builder, sig, args):
    typ = sig.return_type
    lo, hi = args
    interval = cgutils.create_struct_proxy(typ)(context, builder)
    interval.lo = lo
    interval.hi = hi
    return interval._getvalue()


from numba.extending import unbox, NativeValue

@unbox(IntervalType)
def unbox_interval(typ, obj, c):
    """
    Convert a Interval object to a native interval structure.
    """
    lo_obj = c.pyapi.object_getattr_string(obj, "lo")
    hi_obj = c.pyapi.object_getattr_string(obj, "hi")
    interval = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    interval.lo = c.pyapi.float_as_double(lo_obj)
    interval.hi = c.pyapi.float_as_double(hi_obj)
    c.pyapi.decref(lo_obj)
    c.pyapi.decref(hi_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(interval._getvalue(), is_error=is_error)

from numba.extending import box

@box(IntervalType)
def box_interval(typ, val, c):
    """
    Convert a native interval structure to an Interval object.
    """
    interval = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    lo_obj = c.pyapi.float_from_double(interval.lo)
    hi_obj = c.pyapi.float_from_double(interval.hi)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Interval))
    res = c.pyapi.call_function_objargs(class_obj, (lo_obj, hi_obj))
    c.pyapi.decref(lo_obj)
    c.pyapi.decref(hi_obj)
    c.pyapi.decref(class_obj)
    return res


from numba import jit

@jit(nopython=True)
def inside_interval(interval, x):
    return interval.lo <= x < interval.hi

@jit(nopython=True)
def interval_width(interval):
    return interval.width

@jit(nopython=True)
def sum_intervals(i, j):
    return Interval(i.lo + j.lo, i.hi + j.hi)

@jit(nopython=True)
def add_to_dict(a):
    d = Dict.empty(Interval,i8)
    d[a] = 0
    return d


a = Interval(1,2)
b = Interval(1.5,1.7)

print(inside_interval(a,1.5))
print(interval_width(a))
print(sum_intervals(a,b))
print(add_to_dict(a))
