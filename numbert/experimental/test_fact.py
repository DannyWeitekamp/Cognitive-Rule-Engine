from numbert.experimental.fact import _fact_from_spec, _standardize_spec, _merge_spec_inheritance, \
    define_fact, cast_fact, _cast_structref, BaseFact, BaseFactType
from numbert.experimental.context import kb_context
from numbert.experimental.kb import KnowledgeBase
from numba import njit, u8
import pytest

def test__standardize_spec():
    spec = {"A" : "string", "B" : "number"}
    spec = _standardize_spec(spec)

    #Standardized specs should at least have 'type'
    assert spec['A']['type'] == 'unicode_type'
    assert spec['B']['type'] == 'float64'

    #Strings must always be treated as nominal
    assert 'nominal' in spec['A']['flags']
    

def test__merge_spec_inheritance():
    context = kb_context("test__merge_spec_inheritance")
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
        spec2 = _standardize_spec({"inherit_from" : "BOOP", "B": "string", "C" : "string"})
        spec_out, inherit_from = _merge_spec_inheritance(spec2, context)

    #But okay to to redefine an attribute if the types match
    spec2 = _standardize_spec({"inherit_from" : "BOOP", "B": "number", "C" : "string"})
    spec_out, inherit_from = _merge_spec_inheritance(spec2, context)


def test_define_fact():
    spec = {"A" : "string", "B" : "number"}
    ctor, typ1 = define_fact("BOOP", spec, context="test__fact_from_spec")

    #Redefinition illegal
    with pytest.raises(AssertionError):
        define_fact("BOOP", spec, context="test__fact_from_spec")

    #But is okay if using a new context
    ctor, typ2 = define_fact("BOOP", spec, context="test__fact_from_spec2")
    # assert str(typ1.context) != str(typ2.context)


def test_inheritence():
    with kb_context("test_inheritence") as context:
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
    with kb_context("test_cast_fact") as context:
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
    with kb_context("test_protected_mutability") as context:
        print("RUNTIME1.2")
        spec = {"A" : "string", "B" : "number"}
        BOOP, BOOPType = define_fact("BOOP", spec,context="test_protected_mutability")
        print("RUNTIME1.3")
        kb = KnowledgeBase(context="test_protected_mutability")
        print("RUNTIME1")
        b1 = BOOP("A",0)
        b2 = BOOP("B",0)
        print("RUNTIME1")
        @njit
        def edit_it(b):
            b.B += 1

        edit_it(b1)
        edit_it(b2)
        # edit_it.py_func(b2)

        @njit
        def declare_it(kb,b,name):
            kb.declare(name,b)
        declare_it(kb,b1,"b1")
        declare_it.py_func(kb,b2,"b2")

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

        

        





if __name__ == "__main__":
    test__standardize_spec()
    test__merge_spec_inheritance()
    test_define_fact()
    test_inheritence()
    test_cast_fact()
    test_protected_mutability()