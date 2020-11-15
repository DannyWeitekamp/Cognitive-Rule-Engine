from numbert.experimental.context import define_fact
from numbert.experimental.kb import KnowledgeBase, KnowledgeStore

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
