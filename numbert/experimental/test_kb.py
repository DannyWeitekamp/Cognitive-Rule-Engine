from numbert.experimental.context import kb_context
from numbert.experimental.fact import define_fact
from numbert.experimental.kb import KnowledgeBase, KnowledgeBaseType, decode_idrec, encode_idrec, next_empty_f_id, make_f_id_empty
from numba import njit
from numba.types import unicode_type, NamedTuple
from numba.core.errors import TypingError
from numba.experimental.structref import new
import logging
import numpy as np
import pytest
from collections import namedtuple
from numbert.experimental.subscriber import BaseSubscriberType, init_base_subscriber
from numbert.experimental.utils import _struct_from_meminfo


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
        kb.declare(TextField(i_s,i_s,i_s,i_s,i_s),i_s)

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
        kb.declare(TextField(i_s,i_s,i_s,i_s,i_s),i_s)


    t_id = kb.context_data.fact_to_t_id["TextField"]
    return kb.kb_data.empty_f_id_heads[t_id]

@njit(cache=True)
def bad_declare_type(kb):
    kb.declare("Bad")

@njit(cache=True)
def bad_retract_type(kb):
    kb.retract(["A",1])

def test_declare_retract():
    #NRT version
    kb = KnowledgeBase()
    assert declare_retract(kb) == 10
    assert declare_again(kb) == 0

    #Python version
    kb = KnowledgeBase()
    assert declare_retract.py_func(kb) == 10
    assert declare_again.py_func(kb) == 0

    with pytest.raises(TypingError):
        bad_declare_type(kb)

    with pytest.raises(TypingError):
        bad_declare_type.py_func(kb)

    with pytest.raises(TypingError):
        bad_retract_type(kb)

    with pytest.raises(TypingError):
        bad_retract_type.py_func(kb)

##### test_modify #####
@njit(cache=True)
def modify_right(kb,fact,v):
    kb.modify(fact,"to_right",v)

@njit(cache=True)
def bad_modify_type(kb):
    kb.modify("???","to_right","???")
    

def test_modify():
    kb = KnowledgeBase()
    fact = TextField("A","B","C","D","E")

    modify_right(kb,fact,"nb")
    assert fact.to_right == "nb"

    modify_right.py_func(kb,fact,"py")
    assert fact.to_right == "py"

    with pytest.raises(TypingError):
        bad_modify_type(kb)

    with pytest.raises(TypingError):
        bad_modify_type.py_func(kb)
    


    

##### test_declare_overloading #####
@njit(cache=True)
def declare_unnamed(kb):
    return kb.declare(TextField("A","B","C","D","E"))

def test_declare_overloading():
    kb = KnowledgeBase()
    idrec1 = declare_unnamed(kb)
    idrec2 = declare_unnamed.py_func(kb)
    assert idrec1 != idrec2



##### test_retract_keyerror #####

@njit(cache=True)
def retract_keyerror(kb):
    kb.declare(TextField("A","B","C","D","E"),"A")
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
    return kb.all_facts_of_type(TextFieldType)

def test_all_facts_of_type():
    #NRT version
    kb = KnowledgeBase()
    declare_retract(kb)
    all_tf = all_of_type(kb)
    assert isinstance(all_tf[0],TextField)
    assert len(all_tf) == 90

    all_tf = all_of_type.py_func(kb)
    assert isinstance(all_tf[0],TextField)
    assert len(all_tf) == 90


##### test_subscriber #####



@njit(cache=True)
def dummy_subscriber_ctor(kb_meminfo):
    st = new(BaseSubscriberType)
    init_base_subscriber(st,_struct_from_meminfo(KnowledgeBaseType,kb_meminfo) )

    return st

def test_subscriber():
    #NRT version
    kb = KnowledgeBase()
    dummy_subscriber = dummy_subscriber_ctor(kb._meminfo) 
    kb.add_subscriber(dummy_subscriber)

    idrec = declare_unnamed(kb)

    assert kb.kb_data.subscribers[0].grow_queue[0] == idrec

    kb.retract(idrec)

    assert kb.kb_data.subscribers[0].change_queue[0] == idrec



    





if __name__ == "__main__":
    test_modify()
    test_encode_decode()
    test_declare_retract()
    test_retract_keyerror()
    test_subscriber()
