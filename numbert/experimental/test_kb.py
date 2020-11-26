from numbert.experimental.context import define_fact
from numbert.experimental.kb import KnowledgeBase, KnowledgeBaseType, decode_idrec, encode_idrec, next_empty_f_id, make_f_id_empty
from numba import njit
from numba.types import unicode_type
from numbert.experimental.struct_gen import gen_struct_code
import logging

id_rec = encode_idrec(7,8,9)
print(id_rec)
print(decode_idrec(id_rec))


main_logger = logging.getLogger('numba.core')
main_logger.setLevel(logging.DEBUG)


# TextField = define_fact("TextField",{
#     "value" : "string",
#     "above" : {"type" : "string", "flags" : ["reference"]},
#     "below" : {"type" : "string", "flags" : ["reference"]},
#     "to_left" : {"type" : "string", "flags" : ["reference"]},
#     "to_right" : {"type" : "string", "flags" : ["reference"]},
# })
tf_data_fields = [
    ("value" , unicode_type),
    ("above" , unicode_type),
    ("below" , unicode_type),
    ("to_left" , unicode_type),
    ("to_right" , unicode_type),
]
from numba.experimental import structref
from numba.core import types
from numba import njit
@njit
def TextField_get_value(self):
    return self.value

@njit
def TextField_get_above(self):
    return self.above

@njit
def TextField_get_below(self):
    return self.below

@njit
def TextField_get_to_left(self):
    return self.to_left

@njit
def TextField_get_to_right(self):
    return self.to_right

@structref.register
class TextFieldTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class TextField(structref.StructRefProxy):
    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)

    @property
    def value(self):
        return TextField_get_value(self)
    
    @property
    def above(self):
        return TextField_get_above(self)
    
    @property
    def below(self):
        return TextField_get_below(self)
    
    @property
    def to_left(self):
        return TextField_get_to_left(self)
    
    @property
    def to_right(self):
        return TextField_get_to_right(self)
    

structref.define_proxy(TextField, TextFieldTypeTemplate, ['value','above','below','to_left','to_right'])

# print(gen_struct_code("TextField",tf_data_fields))
# l = {}
# exec(gen_struct_code("TextField",tf_data_fields), {}, l)

# TextField, TextFieldTypeTemplate = l['TextField'], l['TextFieldTypeTemplate']
TextFieldType = TextFieldTypeTemplate(fields=tf_data_fields)
# if(not source_in_cache("KnowledgeBaseData",'KnowledgeBaseData')):
#     source = gen_struct_code("KnowledgeBaseData",kb_data_fields)
#     source_to_cache("KnowledgeBaseData",'KnowledgeBaseData',source)
    
# KnowledgeBaseData, KnowledgeBaseDataTypeTemplate = import_from_cached("KnowledgeBaseData",
#     "KnowledgeBaseData",["KnowledgeBaseData","KnowledgeBaseDataTypeTemplate"]).values()
# print(KnowledgeBaseData, KnowledgeBaseDataTypeTemplate)

# KnowledgeBaseDataType = KnowledgeBaseDataTypeTemplate(fields=kb_data_fields)


@njit
def foo(kb):
    kb.declare("A",TextField("A","B","C","D","E"))
    kb.declare("B",TextField("A","B","1","2","3"))
    kb.declare("C",TextField("A","B","C4","D5","E7"))


@njit
def bar(kb,v):
    
    kb.retract(v)



kb = KnowledgeBase()
print(kb.kb_data.empty_f_id_stacks)
print(kb.kb_data.empty_f_id_heads)

# make_f_id_empty(kb.kb_data,0,7)
# make_f_id_empty(kb.kb_data,0,8)
# make_f_id_empty(kb.kb_data,0,9)
# print(kb.kb_data.empty_f_id_stacks)
# print(kb.kb_data.empty_f_id_heads)
# print(next_empty_f_id(kb.kb_data,0))
# print(next_empty_f_id(kb.kb_data,0))
# print(next_empty_f_id(kb.kb_data,0))
foo(kb)
# print(kb.kb_data.fact_meminfos)
print(kb.kb_data.names_to_idrecs)

bar(kb,"C")
try:
    bar(kb,"C")
except KeyError as e:
    print("ERROR")


# print(kb.kb_data.fact_meminfos)
print(kb.kb_data.names_to_idrecs)
# ks = KnowledgeStore("TextField",kb)
# kb.declare("A",TextField("A","B","C","D","E"))
# ks = kb.stores['TextField']
# print("DONE")
# print(ks.store_data)
# print(type(kb))
# print(kb.__dict__)
# @njit(locals={"kb" : KnowledgeBaseType})
# def foo(kb):
#     kb.declare("Q",TextField("A","B","C","D","E"))

# foo(kb)
