from numbert.experimental.context import define_fact
from numbert.experimental.kb import KnowledgeBase, KnowledgeBaseType, decode_idrec, encode_idrec, next_empty_f_id, make_f_id_empty
from numba import njit
from numba.types import unicode_type
from numbert.experimental.struct_gen import gen_struct_code
import logging
import numpy as np
import pytest

# main_logger = logging.getLogger('numba.core')
# main_logger.setLevel(logging.DEBUG)


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
TextFieldType = TextFieldTypeTemplate(fields=tf_data_fields)

##### test_encode_decode #####

def test_encode_decode():
    id_rec = encode_idrec(7,8,9)
    assert isinstance(id_rec,int)
    assert decode_idrec(id_rec) == (7,8,9)


##### test_declare_retract #####

@njit
def declare_retract(kb):
    for i in range(100):
        i_s = "A" + str(i)
        kb.declare(i_s,TextField(i_s,i_s,i_s,i_s,i_s))

    for i in range(0,100,10):
        i_s = "A" + str(i)
        kb.retract(i_s)

    return kb.kb_data.empty_f_id_heads[0]

@njit
def declare_again(kb):
    for i in range(0,100,10):
        i_s = "B" + str(i)
        kb.declare(i_s,TextField(i_s,i_s,i_s,i_s,i_s))

    return kb.kb_data.empty_f_id_heads[0]

def test_declare_retract():
    #NRT version
    kb = KnowledgeBase()
    assert declare_retract(kb) == 10
    assert declare_again(kb) == 0

    #Python version
    kb = KnowledgeBase()
    assert declare_retract.py_func(kb) == 10
    assert declare_again.py_func(kb) == 0


##### test_retract_keyerror #####

@njit(cache=True)
def retract_keyerror(kb):
    kb.declare("A",TextField("A","B","C","D","E"))
    kb.retract("A")
    kb.retract("A")

def test_retract_keyerror():
    #NRT version
    kb = KnowledgeBase()
    with pytest.raises(KeyError):
        retract_keyerror(kb)

    #Python version
    kb = KnowledgeBase()
    with pytest.raises(KeyError):
        retract_keyerror.py_func(kb)

