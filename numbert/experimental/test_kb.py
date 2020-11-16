from numbert.experimental.context import define_fact
from numbert.experimental.kb import KnowledgeBase, KnowledgeStore, KnowledgeBaseType
from numba import njit
import logging

main_logger = logging.getLogger('numba.core')
main_logger.setLevel(logging.DEBUG)


TextField = define_fact("TextField",{
    "value" : "string",
    "above" : {"type" : "string", "flags" : ["reference"]},
    "below" : {"type" : "string", "flags" : ["reference"]},
    "to_left" : {"type" : "string", "flags" : ["reference"]},
    "to_right" : {"type" : "string", "flags" : ["reference"]},
})


kb = KnowledgeBase()
# ks = KnowledgeStore("TextField",kb)
kb.declare("A",TextField("A","B","C","D","E"))
ks = kb.stores['TextField']
print("DONE")
print(ks.store_data)
print(type(kb))

# @njit(locals={"kb" : KnowledgeBaseType})
# def foo(kb):
#     kb.declare("Q",TextField("A","B","C","D","E"))

# foo(kb)
