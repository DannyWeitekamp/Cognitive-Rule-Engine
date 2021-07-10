# from numba.typed import DictType
from numba.types import DictType
from numba import i8, types
from cre.caching import unique_hash, source_in_cache, import_from_cached, source_to_cache
from numba.experimental import structref

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
from cre.structref import CastFriendlyStructref
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
{f'structref.define_boxing({typ}TypeTemplate, {typ})' if(define_boxing) else ''}


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
    if(isinstance(fields,dict)): [(k,v) for k,v in fields.items()]
    hash_code = unique_hash([name,fields])
    if(not source_in_cache(name,hash_code)):
        source = gen_structref_code(name, fields, define_constructor=define_constructor,
             define_boxing=define_boxing)
        source_to_cache(name,hash_code,source)
        
    ctor, template = import_from_cached(name,hash_code,[name,f"{name}TypeTemplate"]).values()
    ctor._hash_code = hash_code
    return ctor,template

def define_structref(name, fields, define_constructor=True,define_boxing=True):
    ctor, template = define_structref_template(name,fields, define_constructor=define_constructor,define_boxing=define_boxing)
    struct_type = template(fields=fields)
    struct_type._hash_code = ctor._hash_code
    return ctor, struct_type



class CastFriendlyStructref(types.StructRef):
    def can_convert_to(self, typingctx, other):
        """
        Convert this Record to the *other*.
        This method only implements width subtyping for records.
        """
        from numba.core.errors import NumbaExperimentalFeatureWarning
        # print(other.__dict__)
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
# print(DictType(i8,i8))
# print(fields)
# print(gen_structref_code("BOOP",fields))


# fields = [('dict', DictType(i8,i8)),
#     ('v', i8)]

# gen_struct("BOOP", fields)


