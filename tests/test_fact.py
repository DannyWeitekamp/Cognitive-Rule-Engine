import numpy as np
from cre.fact import (_fact_from_spec, _standardize_spec, _merge_spec_inheritance, 
     define_fact, cast_fact, _cast_structref, BaseFact, DeferredFactRefType, isa,
      uint_to_inheritance_bytes, get_inheritance_bytes_len_ptr)
from cre.context import cre_context
from cre.memory import Memory
from cre.cre_object import CREObjType
from numba import njit, u8
from numba.typed import List
from numba.types import ListType
import cre.dynamic_exec
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
        BOOP = define_fact("BOOP", spec1, context="test__merge_spec_inheritance")

        #Should be able to inherit from ctor, type or type string
        spec2 = {"inherit_from" : BOOP, "C" : "number"}
        spec_out, inherit_from = _merge_spec_inheritance(spec2,context)
        assert inherit_from._fact_name == "BOOP"
        assert "inherit_from" not in spec_out

        spec2 = {"inherit_from" : BOOP, "C" : "number"}
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
        
        typ1 = define_fact("BOOP", spec)
        #Redefinition illegal with new types
        with pytest.raises(AssertionError):
            define_fact("BOOP", spec2)

    with cre_context("test_define_fact2") as context:
        #But is okay if defined under a different context
        typ2 = define_fact("BOOP", spec2)


def test_untyped_fact():
    with cre_context("test_untyped_fact") as context:
        BOOP = define_fact("BOOP")

        @njit(cache=True)
        def make_boop():
            return BOOP(A="A", B=1.0)

        # Note: Can't do in jit context yet 
        # assert make_boop().A == "A"
        assert make_boop.py_func().A == "A"


def test_inheritence_bytes():
    assert np.array_equal(uint_to_inheritance_bytes(0xFF),[255])
    assert np.array_equal(uint_to_inheritance_bytes(0xFF00+1),[255,1])
    assert np.array_equal(uint_to_inheritance_bytes(0xFF00+0xF),[255,15])
    assert np.array_equal(uint_to_inheritance_bytes(0xFF00+0xFF),[255,255])
    assert np.array_equal(uint_to_inheritance_bytes(0xFF00+0xFF+1),[1,0,0])

    

def test_inheritence():
    with cre_context("test_inheritence") as context:
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1, context="test_inheritence")
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2 = define_fact("BOOP2", spec2, context="test_inheritence")
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3 = define_fact("BOOP3", spec3, context="test_inheritence")

        assert context.parents_of["BOOP3"] == ["BOOP1","BOOP2"]
        assert context.children_of["BOOP3"] == []
        assert context.parents_of["BOOP2"] == ["BOOP1"]
        assert context.children_of["BOOP2"] == ["BOOP3"]
        assert context.parents_of["BOOP1"] == []
        assert context.children_of["BOOP1"] == ["BOOP2","BOOP3"]

        # Context should keep track of parent and child t_ids
        cd = context.context_data
        b1_t_id = cd.fact_num_to_t_id[BOOP1._fact_num]
        b2_t_id = cd.fact_num_to_t_id[BOOP2._fact_num]
        b3_t_id = cd.fact_num_to_t_id[BOOP3._fact_num]

        assert np.array_equal(cd.parent_t_ids[b1_t_id],[])
        assert np.array_equal(cd.parent_t_ids[b2_t_id],[b1_t_id])
        assert np.array_equal(cd.parent_t_ids[b3_t_id],[b1_t_id,b2_t_id])
        assert np.array_equal(cd.child_t_ids[b1_t_id],[b1_t_id,b2_t_id,b3_t_id])
        assert np.array_equal(cd.child_t_ids[b2_t_id],[b2_t_id,b3_t_id])
        assert np.array_equal(cd.child_t_ids[b3_t_id],[b3_t_id])

        b1 = BOOP1("A",7)
        @njit(cache=True)
        def get_idrec(b):
            return b.idrec

        assert get_idrec(b1) == u8(-1)
        assert get_idrec.py_func(b1) == u8(-1)

        b2 = BOOP2("A",7, 6)
        b3 = BOOP3("A",7, 6)

        l,p = get_inheritance_bytes_len_ptr(b1)
        print(l,p)
        l,p = get_inheritance_bytes_len_ptr(b2)
        print(l,p)
        l,p = get_inheritance_bytes_len_ptr(b3)
        print(l,p)

        @njit(cache=True)
        def check_isa(b1,b2,b3):
            okay = np.empty((9,),dtype=np.uint)
            okay[0] = (b1.isa(BOOP1) == 1)
            okay[1] = (b2.isa(BOOP1) == 1)
            okay[2] = (b3.isa(BOOP1) == 1)

            okay[3] = (b1.isa(BOOP2) == 0)
            okay[4] = (b2.isa(BOOP2) == 1)
            okay[5] = (b3.isa(BOOP2) == 1)

            okay[6] = (b1.isa(BOOP3) == 0)
            okay[7] = (b2.isa(BOOP3) == 0)
            okay[8] = (b3.isa(BOOP3) == 1)
            return okay

        py_okay = check_isa.py_func(b1,b2,b3)
        assert all(py_okay), str(py_okay)

        nb_okay = check_isa(b1,b2,b3)
        assert all(nb_okay), str(nb_okay)
        
            

def test_cast_fact():
    with cre_context("test_cast_fact") as context:
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3 = define_fact("BOOP3", spec3)

        print(BOOP1, type(BOOP1))
        print(BOOP2, type(BOOP2))
        print(BOOP3, type(BOOP3))


        b1 = BOOP1("A",7)
        b3 = BOOP3("A",1,2,3)
        bs = BaseFact()

        print(b1, type(b1))

        #upcast
        @njit
        def down_cast(b):
            return cast_fact(BOOP1,b)    

        _b1 = down_cast(b3)
        assert type(b1) == type(_b1)
        _b1 = down_cast.py_func(b3)    
        assert type(b1) == type(_b1)

        #Upcast back
        @njit
        def up_cast(b):
            return cast_fact(BOOP3,b)    
        _b3 = up_cast(_b1)
        assert type(b3) == type(_b3)
        _b3 = up_cast.py_func(_b1)    
        assert type(b3) == type(_b3)    

        FLOOP = define_fact("FLOOP", spec3)

        #Bad cast
        @njit
        def bad_cast(b):
            return cast_fact(FLOOP,b) 

        with pytest.raises(TypeError):
            bad_cast(b3)

        with pytest.raises(TypeError):
            bad_cast.py_func(b3)

        #Always allow casting to and from BaseFact
        @njit
        def base_down_cast(b):
            return cast_fact(BaseFact,b)    
        _bs = base_down_cast(_b1)
        assert type(bs) == type(_bs)
        _bs = base_down_cast.py_func(_b1)    
        assert type(bs) == type(_bs)    

        @njit
        def base_up_cast(b):
            return cast_fact(BOOP1,b)    
        _b1 = base_up_cast(_bs)
        assert type(b1) == type(_b1)
        _b1 = base_up_cast.py_func(_bs)    
        assert type(b1) == type(_b1)     

def test_fact_eq():
    with cre_context("test_fact_eq") as context:
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1)

        b1 = BOOP1("A",7)
        b2 = BOOP1("A",7)
        b3 = BOOP1("B",8)

        @njit(cache=True)
        def do_eq(a,b):
            return a == b

        assert do_eq(b1,b2) == True
        assert do_eq(b1,b1) == True
        assert do_eq(b2,b3) == False
        assert do_eq.py_func(b1,b2) == True
        assert do_eq.py_func(b1,b1) == True
        assert do_eq.py_func(b2,b3) == False

        assert do_eq(b1,None) == False
        assert do_eq.py_func(b1,None) == False



def test_protected_mutability():
    print("RUNTIME1.")
    with cre_context("test_protected_mutability") as context:
        print("RUNTIME1.2")
        spec = {"A" : "string", "B" : "number"}
        BOOP = define_fact("BOOP", spec,context="test_protected_mutability")
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

from cre.utils import lower_setattr, PrintElapse
from cre.fact_intrinsics import fact_lower_setattr
from numba import literally


from cre.var import Var
def test_as_conditions():
    with cre_context("test_as_conditions"):
        spec = {"name" : "string", "prev" : "TestDLL", "next" : "TestDLL"}
        TestDLL = define_fact("TestDLL", spec)
        a = TestDLL(name="A")
        b = TestDLL(name="B")
        c = TestDLL(name="C")
        a.next = b
        b.prev = a
        b.next = c
        c.prev = b

        sel = Var(TestDLL, "sel")
        fact_ptr_to_var_map = {a.get_ptr() : sel, b.get_ptr(): sel.next}
        conds = a.as_conditions(fact_ptr_to_var_map)
        print(conds)

        assert str(conds) == "(sel.name == 'A') & (sel.prev == None) & (sel == sel.next.prev)"


        spec = {"name" : "string", "parent" : "TestContainer"}
        TestChild = define_fact("TestChild", spec)

        c1 = TestChild(name="c1")
        c2 = TestChild(name="c2")

        spec = {"name" : "string", "children" : "ListType(TestChild)"}
        TestContainer = define_fact("TestContainer", spec)

        c1 = TestChild(name="c1")
        c2 = TestChild(name="c2")

        C = TestContainer(name="C", children=List([c1,c2]))
        c1.parent = C
        c2.parent = C


        sel = Var(TestChild, "sel")
        fact_ptr_to_var_map = {c1.get_ptr() : sel, C.get_ptr(): sel.parent}

        conds = c1.as_conditions(fact_ptr_to_var_map)
        print(str(conds))
        conds = c1.as_conditions(fact_ptr_to_var_map)
        conds = c1.as_conditions(fact_ptr_to_var_map)
        conds = c1.as_conditions(fact_ptr_to_var_map)
        conds = c1.as_conditions(fact_ptr_to_var_map)
        conds = c1.as_conditions(fact_ptr_to_var_map)
        conds = c1.as_conditions(fact_ptr_to_var_map)
        conds = c1.as_conditions(fact_ptr_to_var_map)
        conds = c1.as_conditions(fact_ptr_to_var_map)
        print(str(conds))
        assert str(conds) == "(sel.name == 'c1') & (sel == sel.parent.children[0])"


# @njit
# def set_it(self,attr,other):
#     fact_lower_setattr(self,literally(attr),other)
    # self.other = other

from cre.utils import _list_base_from_ptr, _list_base, _cast_list


@njit(cache=True)
def get_base(x):
    return _list_base(x)

def _test_list_type():
    with cre_context("test_list_type"):
        spec = {"A" : "string", "B" : "number"}
        BOOP = define_fact("BOOP", spec)

        spec = {"name":"string","items" : "ListType(BOOP)","other" : "BOOP"}
        BOOPList = define_fact("BOOPList", spec)

        blst_t = ListType(BaseFact)

        @njit(cache=True)
        def get_items(srl):
            items = srl.items
            if(items is not None):
                print(_cast_list(blst_t, items))
                # print("base1",_list_base(items))    
                return srl.items
            return None

        @njit(cache=True)
        def len_items(srl):
            if(srl.items is not None):
                # no = srl.items
                # print("base2",_list_base(no))    
                return len(srl.items)
            return None

        @njit(cache=True)
        def iter_items(srl):
            if(srl.items is not None):
                # no = srl.items
                # print("base3",_list_base(no))    
                i = 0
                for p in srl.items:
                    i += 1
                return i
            return None


        a = BOOP("A",0)
        b = BOOP("B",1)
        c = BOOP("C",2)

        the_list = List([a,b])
        print("BASE!", get_base(the_list))
        bl = BOOPList("L",the_list,c)

        print("BASE!", get_base(bl.items))


        # assert str(bl) == "BOOPList(items=List([BOOP(A='A', B=0.0), BOOP(A='B', B=1.0)]))"
        # bl.other = a
        # bl.name = "BOB"
        # `
        # bl.name = "BOB"
        # set_it(bl,"other",a)
        # set_it(bl,"name","BOB")

        print(bl.idrec)
        print(bl.other)
        print(bl.items)
        print("BASE!", get_base(bl.items))

        print("START!")

        print(get_items(bl))
        print(len_items(bl))
        print(iter_items(bl))

        print("END")

        # raise ValueError()
        # print(bl.items)

        spec = {"items" : "ListType(SelfRefList)"}
        SelfRefList = define_fact("SelfRefList", spec)

        # i =0
        # for i in range(10000):
        #     str(i)
        #     i+=1
        a = SelfRefList()
        b = SelfRefList()
        c = SelfRefList()
        p1 = SelfRefList(List([a,b,c]))
        print(p1.items)

        p1 = SelfRefList()
        p1.items = List([a,b,c])
        print(p1.items)

        
        print(get_items(p1))
        print(len_items(p1))
        print(iter_items(p1))







        
#Work in progress...
def _test_reference_type():
    with cre_context("test_reference_type"):
        spec = {"A" : "string", "B" : "number"}
        BOOP = define_fact("BOOP", spec)

        spec = {"name" : "string", "other" : "BOOP"}
        TestRef = define_fact("TestRef", spec)

        a = BOOP("A", 1)
        b = TestRef("B", a)

        # assert str(a) == 
        # print(a, b)
        # print(a)

        # print(TestRef("B"))


        spec = {"name" : "string", "next" : "TestLL"}
        TestLL = define_fact("TestLL", spec)

        # print(TestLLType.name)

        t1 = TestLL("A")
        t2 = TestLL("B",next=t1)

        # print(t2.next)
        # print(t1,t2)



        

@njit(cache=True)
def as_cre_obj(x):
    return _cast_structref(CREObjType,x)


@njit(cache=True)
def eq(a,b):
    return a == b

def test_eq():
    with cre_context("test_list_type"):
        spec = {"A" : "string", "B" : "number"}
        BOOP = define_fact("BOOP", spec)

        a1 = BOOP("HI",2)
        a2 = BOOP("HI",2)
        b1 = BOOP("HI",3)
        b2 = BOOP("HO",2)

    # print(eq(a1, a2))
        assert eq(a1, a2)
        assert not eq(a1, b1)
        assert not eq(a1, b2)

        a1 = as_cre_obj(BOOP("HI",2))
        a2 = as_cre_obj(BOOP("HI",2))
        b1 = as_cre_obj(BOOP("HI",3))
        b2 = as_cre_obj(BOOP("HO",2))

    # print(eq(a1, a2))
        assert eq(a1, a2)
        assert not eq(a1, b1)
        assert not eq(a1, b2)

        

@njit(cache=True)
def hsh(x):
    return hash(x)


def test_hash():
    with cre_context("test_hash"):
        spec = {"A" : "string", "B" : "number"}
        BOOP = define_fact("BOOP", spec)

        a1 = BOOP("HI",2)
        a2 = BOOP("HI",2)
        b1 = BOOP("HI",3)
        b2 = BOOP("HO",2)

        assert hsh(a1) == hsh(a2)
        assert hsh(a1) != hsh(b1)
        assert hsh(a1) != hsh(b2)

        a3_boop = BOOP("HI",2)
        a3 = as_cre_obj(a3_boop)
        assert hsh(a3) == hsh(a1)

        # check that mutation causes rehash
        a3_boop.B = 7
        assert hsh(a3) != hsh(a1)


        a1 = as_cre_obj(BOOP("HI",2))
        a2 = as_cre_obj(BOOP("HI",2))
        b1 = as_cre_obj(BOOP("HI",3))
        b2 = as_cre_obj(BOOP("HO",2))

        assert hsh(a1) == hsh(a2)
        assert hsh(a1) != hsh(b1)
        assert hsh(a1) != hsh(b2)

        a3_boop = BOOP("HI",2)
        a3 = as_cre_obj(a3_boop)
        assert hsh(a3) == hsh(a1)

        # check that mutation causes rehash
        a3_boop.B = 7
        assert hsh(a3) != hsh(a1)

with cre_context("_b_boop_ctor_1000"):
    # def _define_boop():
    #     with cre_context("_b_boop_ctor_1000"):
    spec = {"A" : "string", "B" : "number"}
    BOOP = define_fact("BOOP", spec)
            # return (BOOP,), {}

    @njit(cache=True)
    def _b_boop_ctor_10000():
        for i in range(10000):
            b = BOOP("HI",i)

def test_b_boop_ctor_10000(benchmark):
    benchmark.pedantic(_b_boop_ctor_10000, warmup_rounds=1, rounds=10)


# @njit(cache=True)
def _b_py_dict_boop_10000():
    for i in range(10000):
        b = {"A" : "HI", "B" : i}

def test_b_py_dict_boop_10000(benchmark):
    benchmark.pedantic(_b_py_dict_boop_10000, warmup_rounds=1, rounds=10)





if __name__ == "__main__":
    import faulthandler; faulthandler.enable()
    # test_list_type()
    
    # _test_list_type()
    # _test_reference_type()
    # test__standardize_spec()
    # test__merge_spec_inheritance()
    # test_define_fact()
    # test_inheritence()
    # test_cast_fact()
    # test_protected_mutability()
    # test_fact_eq()

    # test_untyped_fact()
    # test_fact_eq()

    # test_as_conditions()

    # _test_reference_type()
    # test_hash()
    # test_eq()
    # test_inheritence()
    # test_inheritence_bytes()
    # _test_list_type()





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
