from numbert.experimental.context import kb_context
from numbert.experimental.fact import define_fact
from numbert.experimental.kb import KnowledgeBase, KnowledgeBaseType, decode_idrec, encode_idrec, next_empty_f_id, make_f_id_empty
from numba import njit
from numba.types import unicode_type, NamedTuple
import logging
import numpy as np
import pytest
from collections import namedtuple


spec = {"value" : "string",
        "above" : "string",
        "below" : "string",
        "to_left" : "string",
        "to_right" : "string",
        }


# with kb_context("test_kb") as context:
TextField, TextFieldType = define_fact("TextField",spec)



# from numba.extending import overload_method
# @overload_method(TextFieldTypeTemplate, 'as_named_tuple')
# def TextField_as_named_tuple(self):
#     return TextField_NT(self.value,self.above,self.below,self.to_left,self.to_right)


##### test_encode_decode #####

def test_encode_decode():
    id_rec = encode_idrec(7,8,9)
    assert isinstance(id_rec,int)
    assert decode_idrec(id_rec) == (7,8,9)


##### test_declare_retract #####

@njit(cache=True)
def declare_retract(kb):
    for i in range(100):
        i_s = "A" + str(i)
        kb.declare(i_s,TextField(i_s,i_s,i_s,i_s,i_s))

    for i in range(0,100,10):
        i_s = "A" + str(i)
        kb.retract(i_s)

    # print(kb.kb_data.empty_f_id_heads)
    t_id = kb.context_data.fact_to_t_id["TextField"]
    return kb.kb_data.empty_f_id_heads[t_id]

@njit(cache=True)
def declare_again(kb):
    for i in range(0,100,10):
        i_s = "B" + str(i)
        kb.declare(i_s,TextField(i_s,i_s,i_s,i_s,i_s))


    t_id = kb.context_data.fact_to_t_id["TextField"]
    return kb.kb_data.empty_f_id_heads[t_id]

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

##### test_all_facts_of_type #####

@njit(cache=True)
def all_of_type(kb):
    kb.declare("A",TextField("A","B","C","D","E"))
    kb.retract("A")
    kb.retract("A")

def test_all_facts_of_type():
    #NRT version
    kb = KnowledgeBase()
    declare_retract(kb)
    all_tf = kb.all_facts_of_type(TextFieldType)
    print(type(all_tf[0]))
    assert isinstance(all_tf[0],TextField)
    assert len(all_tf) == 90
    





if __name__ == "__main__":
    test_encode_decode()
    test_declare_retract()
    test_retract_keyerror()
