import inspect
from types import MethodType

src_equivalents = {}
src_backups = {"*" : "python", "nools" : "javascript", "cre" : "python"}
def set_backup_source_target(target, backup):
    src_backups[target] = backup

def set_eq_source_targets(lang1, lang2):
    s1 = src_equivalents.get(lang1,set())
    s2 = src_equivalents.get(lang2,set())
    s1.add(lang2)
    s2.add(lang1)
    for l in s1: 
        if(s2 != l): s2.add(l)
    for l in s2: 
        if(s1 != l): s1.add(l)

    src_equivalents[lang1] = s1
    src_equivalents[lang2] = s2

set_eq_source_targets("js", "javascript")
set_eq_source_targets("py", "python")

def apply_make_source(self, lang="", piece="", **kwargs):
    # print(self, lang, piece, kwargs)
    # print("resolve_make_source")

    # if(not hasattr(self,parent_class)):
    #     raise RuntimeError("Unbound make_source() target.")
    piece_templates = resolve_template(lang,
        self._make_src_registry, accum_all=True)

    for piece_template in piece_templates:
        if(piece in piece_template):
            return piece_template[piece].func(self,{'lang':lang, 'piece':piece **kwargs})

    return None

# make_source_registry = {}

class MakeSourceInst():
    def __init__(self, func, lang, piece="",child_mk_src=None):
        self.func = func
        self.lang = lang
        self.piece = piece
        # print("CHILD", child_mk_src)
        self.child_mk_src = child_mk_src

    def _bind_parent(self, parent_class):
        # print("BIND", self.child_mk_src)
        self.parent_class = parent_class
        if(not hasattr(parent_class, '_make_src_registry')):
            # Define the _make_src_registry and make_source() method on the class.
            parent_class._make_src_registry = {}
            if(hasattr(parent_class, 'make_source')):
                pass # TODO throw error if make_source was directly defined 
                # if(?):raise NameError("Cannot ")
            else:
                setattr(parent_class, 'make_source',
                    MethodType(apply_make_source,parent_class))

        mk_sr_reg = parent_class._make_src_registry          
        lang_pieces = mk_sr_reg.get(self.lang,{})
        lang_pieces[self.piece] = self
        mk_sr_reg[self.lang] = lang_pieces

        if(self.child_mk_src is not None):
            self.child_mk_src._bind_parent(parent_class)

    # @property
    # def parent_class(self):
    #     print("GET parent")
    #     if(not hasattr('_parent_class') and
    #          self.child_mk_src is not None):
    #         if(self.child_mk_src.parent_class is not None):
    #             self._bind_parent(self.child_mk_src.parent_class)
    #     return self._parent_class

    def __set_name__(self, owner, name):
        ''' Defining this automatically binds method decorations 
            to the class they are applied in.'''
        self._bind_parent(owner)
        

def make_source(lang, piece="", owner=None):
    ''' A decorator function that marks a source generating method 
            with its lang and piece.'''
    if(isinstance(lang,MakeSourceInst)):
        # Allows multiple @make_source decorations to be applied
        #  to the same generating function.
        mk_inst = lang
        lang, piece = mk_inst.lang, mk_inst.piece
    def wrapper(func_or_mk_src):
        if(isinstance(func_or_mk_src, MakeSourceInst)): 
            child = func_or_mk_src
            mk_src_inst = MakeSourceInst(child.func,lang, piece, child)
        else:
            mk_src_inst = MakeSourceInst(func_or_mk_src, lang, piece)
        
        # This allows the end user to use make_source on unbound methods
        #  i.e. make_source('python', "", MyClass)(my_gen_func)
        if(owner is not None): mk_src_inst._bind_parent(owner)
        return mk_src_inst
    return wrapper


# class Foo():

#     @make_source("python")
#     def python_call_src(self):
#         return self.make_source("python","call", "long")


#     @make_source("python","call", "long")
#     def python_call_long_src(self):
#         return \
# '''
# a = 7
# b = 8
# return a + b
# '''

#     @make_source("python","call", "short")
#     def python_call_short_src(self):
#         return \
# '''
# return 7 + 8
# '''

# Foo().boop()
# print("SOURCE", Foo().make_source("python","call", "long"))
# print("SOURCE", Foo().make_source("python","call", "short"))
# print("SOURCE", Foo().make_source("python"))

def resolve_template(lang, templates,desc="",accum_all=False):
    ret = []
    if(lang in templates):
        ret.append(templates[lang])    
    if(not accum_all and len(ret) > 0): return ret[0]

    # Check for other names for the same language
    for l in src_equivalents.get(lang,[]):
        if(l in templates):
            ret.append(templates[l])
        if(not accum_all and len(ret) > 0): return ret[0]

    # Check for backups
    backup_lang = src_backups.get(lang,None)
    if(backup_lang in templates):
        ret.append(templates[backup_lang])
    if(not accum_all and len(ret) > 0): return ret[0]

    # See if wild card works
    if("*" in templates):
        ret.append(templates["*"])
    if(not accum_all and len(ret) > 0): return ret[0]

    if(len(ret) > 0):
        return ret
    else:
        raise LookupError(f"Missing {desc} template for language : {lang}")


assign_templates = {
    "javascript" : "let {alias} = {rest}",
    "python" : "{alias} = {rest}"
}
def gen_assign(lang,alias,rest):
    template = resolve_template(lang, assign_templates, 'assign')
    return template.format(alias=alias,rest=rest)
    


def_func_templates = {
    "javascript" : '''function {fname}({args}){{
{body}{tail}}}''',

    "python" : '''def {fname}({args}):
{body}{tail}'''

}

def gen_def_func(lang, fname, args, body, tail=''):
    template = resolve_template(lang, def_func_templates, 'def_func')
    return template.format(fname=fname,args=args, body=body, tail=tail)
    


