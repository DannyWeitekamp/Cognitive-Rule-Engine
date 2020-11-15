# from numba.typed import DictType
from numba.types import DictType
from numba import i8

def _gen_getter_jit(typ,attr):
    return f'''@njit
def {typ}_get_{attr}(self):
    return self.{attr}
'''

def _gen_getter(typ,attr):
    return f'''    @property
    def {attr}(self):
        return {typ}_get_{attr}(self)
    '''

def gen_struct_code(typ,fields):
    getters = "\n".join([_gen_getter(typ,attr) for attr,t in fields])
    getter_jits = "\n".join([_gen_getter_jit(typ,attr) for attr,t in fields])
    attr_list = ",".join(["'%s'"%attr for attr,t in fields])
    code = \
f'''
from numba.experimental import structref
from numba.core import types
from numba import njit
{getter_jits}
@structref.register
class {typ}TypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class {typ}(structref.StructRefProxy):
    def __new__(cls, d, v):
        return structref.StructRefProxy.__new__(cls, d, v)

{getters}

structref.define_proxy({typ}, {typ}TypeTemplate, [{attr_list}])

'''
    return code

def gen_struct(typ,fields):
    code = gen_struct_code(typ,fields)
    print(code)
    l = {}
    exec(code,{},l)
    print(l[f"{typ}TypeTemplate"])
    return l[f"{typ}TypeTemplate"](fields=fields)


# print(DictType(i8,i8))
# print(fields)
# print(gen_struct_code("BOOP",fields))


# fields = [('dict', DictType(i8,i8)),
#     ('v', i8)]

# gen_struct("BOOP", fields)


