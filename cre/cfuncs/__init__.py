from llvmlite import binding as ll
import cre_cfuncs


for name, c_addr in cre_cfuncs.c_helpers.items():
    ll.add_symbol(name, c_addr)
