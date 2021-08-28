from cre.fact import _fact_from_spec, _standardize_spec, _merge_spec_inheritance, \
     define_fact, cast_fact, _cast_structref, BaseFact, BaseFactType, DeferredFactRefType
from cre.context import cre_context
from cre.memory import Memory
from numba import njit, u8
from numba.typed import List
import pytest

def test__standardize_spec():
    with cre_context("test__standardize_spec") as context:
        spec = {"A" : "string", "B" : "number"}
        spec = _standardize_spec(spec, context,"BOOP")
        print(spec)

        #Standardized specs should at least have 'type'
        assert str(spec['A']['type']) == 'unicode_type'
        assert str(spec['B']['type']) == 'float64'

        #Strings must always be treated as nominal
        assert 'nominal' in spec['A']['flags']
    

def test__merge_spec_inheritance():
    with cre_context("test__merge_spec_inheritance") as context:
        spec1 = {"A" : "string", "B" : "number"}
        BOOP, BOOPType = define_fact("BOOP", spec1, context="test__merge_spec_inheritance")

        #Should be able to inherit from ctor, type or type string
        spec2 = {"inherit_from" : BOOP, "C" : "number"}
        spec_out, inherit_from = _merge_spec_inheritance(spec2,context)
        assert inherit_from._fact_name == "BOOP"
        assert "inherit_from" not in spec_out

        spec2 = {"inherit_from" : BOOPType, "C" : "number"}
        spec_out, inherit_from = _merge_spec_inheritance(spec2,context)
        assert inherit_from._fact_name == "BOOP"
        assert "inherit_from" not in spec_out

        spec2 = {"inherit_from" : "BOOP", "C" : "number"}
        spec_out, inherit_from = _merge_spec_inheritance(spec2,context)
        assert inherit_from._fact_name == "BOOP"
        assert "inherit_from" not in spec_out

        assert "A" in spec_out
        assert "B" in spec_out

        #It is illegal to redefine an attribute to have a new type
        with pytest.raises(TypeError):
            spec2 = {"inherit_from" : "BOOP", "B": "string", "C" : "string"}
            spec2 = _standardize_spec(spec2, context, "BOOP2")
            spec_out, inherit_from = _merge_spec_inheritance(spec2, context)

        #But okay to to redefine an attribute if the types match
        spec2 = {"inherit_from" : "BOOP", "B": "number", "C" : "string"}
        spec2 = _standardize_spec(spec2, context, "BOOP2")
        spec_out, inherit_from = _merge_spec_inheritance(spec2, context)


def test_define_fact():
    spec = {"A" : "string", "B" : "number"}
    spec2 = {"A" : "string", "B" : "string"}
    with cre_context("test_define_fact") as context:
        
        ctor, typ1 = define_fact("BOOP", spec)
        #Redefinition illegal with new types
        with pytest.raises(AssertionError):
            define_fact("BOOP", spec2)

    with cre_context("test_define_fact2") as context:
        #But is okay if defined under a different context
        ctor, typ2 = define_fact("BOOP", spec2)


def test_inheritence():
    with cre_context("test_inheritence") as context:
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1, BOOP1Type = define_fact("BOOP1", spec1, context="test_inheritence")
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2, BOOP2Type = define_fact("BOOP2", spec2, context="test_inheritence")
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3, BOOP3Type = define_fact("BOOP3", spec3, context="test_inheritence")

        assert context.parents_of["BOOP3"] == ["BOOP1","BOOP2"]
        assert context.children_of["BOOP3"] == []
        assert context.parents_of["BOOP2"] == ["BOOP1"]
        assert context.children_of["BOOP2"] == ["BOOP3"]
        assert context.parents_of["BOOP1"] == []
        assert context.children_of["BOOP1"] == ["BOOP2","BOOP3"]

        b1 = BOOP1("A",7)
        @njit
        def check_has_base(b):
            return b.idrec

        assert check_has_base(b1) == u8(-1)
        assert check_has_base.py_func(b1) == u8(-1)



def test_cast_fact():
    with cre_context("test_cast_fact") as context:
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1, BOOP1Type = define_fact("BOOP1", spec1, context="test_cast_fact")
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2, BOOP2Type = define_fact("BOOP2", spec2, context="test_cast_fact")
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3, BOOP3Type = define_fact("BOOP3", spec3, context="test_cast_fact")


        b1 = BOOP1("A",7)
        b3 = BOOP3("A",1,2,3)
        bs = BaseFact()

        #Downcast
        @njit
        def down_cast(b):
            return cast_fact(BOOP1Type,b)    

        _b1 = down_cast(b3)
        assert type(b1) == type(_b1)
        _b1 = down_cast.py_func(b3)    
        assert type(b1) == type(_b1)

        #Upcast back
        @njit
        def up_cast(b):
            return cast_fact(BOOP3Type,b)    
        _b3 = up_cast(_b1)
        assert type(b3) == type(_b3)
        _b3 = up_cast.py_func(_b1)    
        assert type(b3) == type(_b3)    

        FLOOP, FLOOPType = define_fact("FLOOP", spec3, context="test_cast_fact")

        #Bad cast
        @njit
        def bad_cast(b):
            return cast_fact(FLOOPType,b) 

        with pytest.raises(TypeError):
            bad_cast(b3)

        with pytest.raises(TypeError):
            bad_cast.py_func(b3)

        #Always allow casting to and from BaseFact
        @njit
        def base_down_cast(b):
            return cast_fact(BaseFactType,b)    
        _bs = base_down_cast(_b1)
        assert type(bs) == type(_bs)
        _bs = base_down_cast.py_func(_b1)    
        assert type(bs) == type(_bs)    

        @njit
        def base_up_cast(b):
            return cast_fact(BOOP1Type,b)    
        _b1 = base_up_cast(_bs)
        assert type(b1) == type(_b1)
        _b1 = base_up_cast.py_func(_bs)    
        assert type(b1) == type(_b1)    




def test_protected_mutability():
    print("RUNTIME1.")
    with cre_context("test_protected_mutability") as context:
        print("RUNTIME1.2")
        spec = {"A" : "string", "B" : "number"}
        BOOP, BOOPType = define_fact("BOOP", spec,context="test_protected_mutability")
        print("RUNTIME1.3")
        mem = Memory(context="test_protected_mutability")
        print("RUNTIME1")
        b1 = BOOP("A",0)
        b2 = BOOP("B",0)
        print("RUNTIME1")
        @njit
        def edit_it(b):
            b.B += 1

        print("RUNTIME2")

        edit_it(b1)
        edit_it(b2)
        # edit_it.py_func(b2)
        print("RUNTIME3")
        @njit
        def declare_it(mem,b,name):
            mem.declare(b,name)

        print("RUNTIME3.1",b1.fact_num)
        declare_it(mem,b1,"b1")
        print("RUNTIME3.2")
        declare_it.py_func(mem,b2,"b2")

        print("RUNTIMEz")

        with pytest.raises(AttributeError):
            print("RUNTIME_PY")
            edit_it.py_func(b1)

        with pytest.raises(AttributeError):
            print("RUNTIME_PY")
            edit_it.py_func(b2)

        with pytest.raises(AttributeError):
            print("RUNTIME_NB", b1.B)
            edit_it(b1)

        with pytest.raises(AttributeError):
            print("RUNTIME_NB", b1.B)
            edit_it(b1)

from cre.utils import lower_setattr
from cre.fact_intrinsics import fact_lower_setattr
from numba import literally


# @njit
# def set_it(self,attr,other):
#     fact_lower_setattr(self,literally(attr),other)
    # self.other = other


def _test_list_type():
    with cre_context("test_list_type"):
        spec = {"A" : "string", "B" : "number"}
        BOOP, BOOPType = define_fact("BOOP", spec)

        spec = {"name":"string","items" : "ListType(BOOP)","other" : "BOOP"}
        BOOPList, BOOPListType = define_fact("BOOPList", spec)

        a = BOOP("A",0)
        b = BOOP("B",1)
        c = BOOP("C",2)

        bl = BOOPList("L",List([a,b]),c)
        # assert str(bl) == "BOOPList(items=List([BOOP(A='A', B=0.0), BOOP(A='B', B=1.0)]))"
        bl.other = a
        bl.name = "BOB"
        # set_it(bl,"other",a)
        # set_it(bl,"name","BOB")
        print(bl.idrec)
        print(bl.other)
        print(bl)
        print(bl.items)

        spec = {"items" : "ListType(SelfRefList)"}
        SelfRefList, SelfRefListType = define_fact("SelfRefList", spec)







        
#Work in progress...
def _test_reference_type():
    with cre_context("test_reference_type"):
        spec = {"A" : "string", "B" : "number"}
        BOOP, BOOPType = define_fact("BOOP", spec)

        spec = {"name" : "string", "other" : "BOOP"}
        TestRef, TestRefType = define_fact("TestRef", spec)

        a = BOOP("A", 1)
        b = TestRef("B", a)

        # assert str(a) == 
        print(a, b)
        print(a)

        print(TestRef("B"))


        spec = {"name" : "string", "next" : "TestLL"}
        TestLL, TestLLType = define_fact("TestLL", spec)

        print(TestLLType.name)

        t1 = TestLL("A")
        t2 = TestLL("B",next=t1)

        # print(t2.next)
        # print(t1,t2)



        







if __name__ == "__main__":
    # test_list_type()

    _test_list_type()
    _test_reference_type()
    # test__standardize_spec()
    # test__merge_spec_inheritance()
    # test_define_fact()
    # test_inheritence()
    # test_cast_fact()
    # test_protected_mutability()

    # _test_reference_type()





##### NOTES NOTES NOTES ####
'''
[] Need a way to organize tests -> need to have some standard types:
    -TObj = {"name" : "string", "value" : "number"}
    -TRef = {"name" : "string", "other" : "TestObj"}
    -TLL = {"name" : "string", "next" : "TLL"}
    -TList = {"name" : "string", "items" : "ListType("string")"}
    -TSelfList = {"name" : "string", "items" : "ListType("TSelfList")"}
    -TObjList = {"name" : "string", "items" : "ListType("TObj")"}

[] Should test alternative definition patterns
    -string vs type obj
    
[] For each type need to test:
    -getattr, setattr, str 
    -mem.modify
    -matching




'''
