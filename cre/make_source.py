import inspect
from types import MethodType

src_backups = {"js" : "javascript", "nools" : "javascript", "cre" : "python"}
def set_backup_source_target(target, backup):
    src_backups[target] = backup


def resolve_make_source(self, lang, piece='', variant=''):
    print("resolve_make_source")
    src_reg = self._make_src_registry
    out = src_reg.get((lang, piece, variant), None)
    if(out is not None): return out.func(self)
    # if(piece != ""):
    #     out = src_reg.get((lang, piece, ""), None)
    #     if(out is not None): return out.func(self)
    # out = src_reg.get((lang, "", ""), None)
    # if(out is not None): return out.func(self)

    backup_lang = src_backups.get(lang,None)
    if(backup_lang):
        out = src_reg.get((backup_lang, piece, variant), None)
        if(out is not None): return out.func(self)
        # if(piece != ""):
        #     out = src_reg.get((backup_lang, piece, ""), None)
        #     if(out is not None): return out.func(self)
        # out = src_reg.get((backup_lang, "", ""), None)
        # if(out is not None): return out.func(self)

    return None

make_source_registry = {}

class MakeSourceInst():
    def __init__(self, func, lang, piece="",variant=""):
        self.func = func
        self.id_tup = (lang, piece, variant)

        make_source_registry[self.id_tup] = self

        # print(func, lang, piece, variant)

    def __set_name__(self, owner, name):
        # print("set_name",owner, name)
        self.parent_class = owner
        if(not hasattr(self.parent_class, '_make_src_registry')):

            self.parent_class._make_src_registry = {}
            setattr(self.parent_class, 'make_source' , resolve_make_source)
            # self.parent_class.make_source = MethodType(resolve_make_source, None, self.parent_class)
            print(self.parent_class.make_source)

        self.parent_class._make_src_registry[self.id_tup] = self

def make_source(lang,piece="",variant=""):
    def wrapper(func):
        return MakeSourceInst(func, lang, piece, variant)
    return wrapper


class Foo():

    @make_source("python")
    def python_call_src(self):
        return self.make_source("python","call", "long")


    @make_source("python","call", "long")
    def python_call_long_src(self):
        return \
'''
a = 7
b = 8
return a + b
'''

    @make_source("python","call", "short")
    def python_call_short_src(self):
        return \
'''
return 7 + 8
'''

# Foo().boop()
print("SOURCE", Foo().make_source("python","call", "long"))
print("SOURCE", Foo().make_source("python","call", "short"))
print("SOURCE", Foo().make_source("python"))

