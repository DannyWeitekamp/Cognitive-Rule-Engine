from cre.memory import Memory 
from cre.grounding import new_flattener, get_semantic_visibile_fact_attrs, flattener_update
from cre.fact import define_fact
# with cre_context("test_inheritence") as context:
spec1 = {"A" : {"type" : "string", "is_semantic_visible" : True}, 
         "B" : {"type" : "number", "is_semantic_visible" : False}}
BOOP1 = define_fact("BOOP1", spec1)
spec2 = {"inherit_from" : BOOP1,
         "C" : {"type" : "number", "is_semantic_visible" : True}}
BOOP2 = define_fact("BOOP2", spec2)
spec3 = {"inherit_from" : BOOP2,
         "D" : {"type" : "number", "is_semantic_visible" : True}}
BOOP3 = define_fact("BOOP3", spec3)


mem = Memory()
mem.declare(BOOP1("A", 1))
mem.declare(BOOP1("B", 2))
mem.declare(BOOP2("C", 3, 13))
mem.declare(BOOP2("D", 4, 14))
mem.declare(BOOP3("E", 5, 15, 105))
mem.declare(BOOP3("F", 6, 16, 106))
out_mem = Memory()


fl = new_flattener((BOOP1, BOOP2, BOOP3), mem, out_mem)

# svfa = get_semantic_visibile_fact_attrs((BOOP1, BOOP2, BOOP3))
# print([(ft._fact_name, attr) for ft, attr in svfa])

flattener_update(fl)

print("-------")
print(out_mem)
print(out_mem)


# mem.declare(BOOP1("zA", 1))
# mem.declare(BOOP1("zB", 2))
# mem.declare(BOOP2("zC", 3, 13))
# mem.declare(BOOP2("zD", 4, 14))
# mem.declare(BOOP3("zE", 5, 15, 105))
# mem.declare(BOOP3("zF", 6, 16, 106))

# flattener_update(fl)

# print(mem,out_mem)
# flattener_ctor((BOOP1, BOOP2, BOOP3))


# raise ValueError()
