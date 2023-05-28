from cre.context import cre_context
from cre.fact import define_fact
from cre.tuple_fact import TF, TupleFact
from cre.memset import MemSet, MemSetType, decode_idrec, encode_idrec, next_empty_f_id, make_f_id_empty, retracted_f_ids_for_t_id, get_facts
from numba import njit
from numba.types import unicode_type, NamedTuple
from numba.typed import List
from numba.core.errors import TypingError
from numba.experimental.structref import new
import logging
import numpy as np
import pytest
from collections import namedtuple
from cre.utils import _struct_from_meminfo, PrintElapse, used_bytes, NRTStatsEnabled
import gc
from numba.core.runtime.nrt import rtsys
from weakref import WeakKeyDictionary
import pytest
from copy import copy


tf_spec = {"value" : "string",
        "above" : "string",
        "below" : "string",
        "to_left" : "string",
        "to_right" : "string",
        }

##### test_declare_retract #####

with cre_context("test_declare_retract"):
    TextField = define_fact("TextField",tf_spec)

@njit(cache=True)
def declare_retract(ms, t_id):
    for i in range(100):
        i_s = "A" + str(i)
        ms.declare(TextField(i_s,i_s,i_s,i_s,i_s),i_s)

    for i in range(0,100,10):
        i_s = "A" + str(i)
        ms.retract(i_s)

    # print(ms.empty_f_id_heads)
    # t_id = ms.context_data.fact_to_t_id["TextField"]
    return retracted_f_ids_for_t_id(ms,t_id).head

@njit(cache=True)
def declare_again(ms,t_id):
    for i in range(0,100,10):
        i_s = "B" + str(i)
        ms.declare(TextField(i_s,i_s,i_s,i_s,i_s),i_s)


    # t_id = ms.context_data.fact_to_t_id["TextField"]
    return retracted_f_ids_for_t_id(ms,t_id).head#ms.empty_f_id_heads[t_id]





def test_declare_retract():
    with cre_context("test_declare_retract"):
        #NRT version
        ms = MemSet()
        assert declare_retract(ms,TextField.t_id) == 10
        assert declare_again(ms,TextField.t_id) == 0
        print("A")
        #Python version
        ms = MemSet()
        assert declare_retract.py_func(ms,TextField.t_id) == 10
        assert declare_again.py_func(ms,TextField.t_id) == 0

def test_declare_retract_tuple_fact():
    with cre_context("test_declare_retract_tuple_fact"):
        #NRT version
        ms = MemSet()
        idrec1 = ms.declare(("A",1))
        idrec2 = ms.declare(TF("A",1))
        print(decode_idrec(idrec1))
        print(decode_idrec(idrec2))

##### test_modify #####
@njit(cache=True)
def modify_right(ms,fact,v):
    ms.modify(fact,"to_right",v)

@njit(cache=True)
def bad_modify_type(ms):
    ms.modify("???","to_right","???")
    

def test_modify():
    with cre_context("test_modify"):
        TextField = define_fact("TextField",tf_spec)
        ms = MemSet()
        fact = TextField("A","B","C","D","E")

        modify_right(ms,fact,"nb")
        assert fact.to_right == "nb"

        modify_right.py_func(ms,fact,"py")
        assert fact.to_right == "py"

        with pytest.raises(TypingError):
            bad_modify_type(ms)

        with pytest.raises(TypingError):
            bad_modify_type.py_func(ms)
        

##### test_declare_overloading #####

with cre_context("test_declare_overloading"):
    TextField = define_fact("TextField",tf_spec)

@njit(cache=True)
def declare_unnamed(ms):
    return ms.declare(TextField("A","B","C","D","E"))

def test_declare_overloading():
    with cre_context("test_declare_overloading"):
        ms = MemSet()
        idrec1 = declare_unnamed(ms)
        idrec2 = declare_unnamed.py_func(ms)
        assert idrec1 != idrec2


##### test_retract_keyerror #####
with cre_context("test_retract_keyerror"):
    TextField = define_fact("TextField",tf_spec)

@njit(cache=True)
def retract_keyerror(ms):
    ms.declare(TextField("A","B","C","D","E"),"A")
    ms.retract("A")
    ms.retract("A")

def test_retract_keyerror():
    with cre_context("test_retract_keyerror"):
        #NRT version
        ms = MemSet()
        with pytest.raises(KeyError):
            retract_keyerror(ms)

        #Python version
        ms = MemSet()
        with pytest.raises(KeyError):
            retract_keyerror.py_func(ms)

##### test_get_facts #####

# @njit(cache=True)
# def all_of_type(ms):
#     return ms.all_facts_of_type(TextField)
from itertools import product
def test_get_facts():
    with cre_context("test_get_facts"):
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3 = define_fact("BOOP3", spec3)

        ms = MemSet()
        ms.declare(BOOP1("A",1))
        ms.declare(BOOP1("B",2))
        ms.declare(BOOP1("C",3))

        @njit(cache=True)
        def iter_b1(ms):
            l = List()
            for x in ms.get_facts(BOOP1):
                l.append(x)
            return l

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            assert isinstance(all_tf[0], BOOP1._fact_proxy)
            assert len(all_tf) == 3
        
        # ms = MemSet()
        ms.declare(BOOP2("D",4))
        ms.declare(BOOP2("E",5))
        ms.declare(BOOP2("F",6))        

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            print(all_tf)
            assert isinstance(all_tf[0], BOOP1._fact_proxy)
            assert len(all_tf) == 6


        # ms = MemSet()
        ms.declare(BOOP3("G",7))
        ms.declare(BOOP3("H",8))
        ms.declare(BOOP3("I",9))  

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            print(all_tf)
            assert isinstance(all_tf[0], BOOP1._fact_proxy)
            assert len(all_tf) == 9


def test_retroactive_register():
    with cre_context("test_context_retroactive_register") as context:
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3 = define_fact("BOOP3", spec3)
    # Check that retroactive registration works fine for declare()
    with cre_context("other_context") as context:
        with pytest.raises(ValueError):
            context.get_t_id(name="BOOP1")

        ms = MemSet()
        ms.declare(BOOP1("A",1))
        ms.declare(BOOP1("A",2))
        ms.declare(BOOP2("B",2, 3))
        ms.declare(BOOP2("B",3, 3))
        ms.declare(BOOP3("C",3, 4, 5))
        ms.declare(BOOP3("C",4, 4, 5))

        b1_t_id = context.get_t_id(_type=BOOP1)
        b2_t_id = context.get_t_id(_type=BOOP2)
        b3_t_id = context.get_t_id(_type=BOOP3)

        assert b1_t_id != b2_t_id and b2_t_id != b3_t_id

        # c = context
        # assert np.array_equal(c.get_parent_t_ids(t_id=b3_t_id),[b1_t_id,b2_t_id])
        # assert np.array_equal(c.get_parent_t_ids(t_id=b2_t_id),[b1_t_id])
        # assert np.array_equal(c.get_parent_t_ids(t_id=b1_t_id),[])
        
        # assert np.array_equal(c.get_child_t_ids(t_id=b3_t_id),[b3_t_id])
        # assert np.array_equal(c.get_child_t_ids(t_id=b2_t_id),[b2_t_id,b3_t_id])
        # assert np.array_equal(c.get_child_t_ids(t_id=b1_t_id),[b1_t_id,b2_t_id,b3_t_id])


from cre.memset import Indexer, indexer_update, indexer_get_facts, indexer_get_fact
def test_indexer():
    with cre_context("test_indexer"):
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3 = define_fact("BOOP3", spec3)        

        ms = MemSet()
        ms.declare(BOOP1("a", 1))
        ms.declare(BOOP2("a", 2))
        ms.declare(BOOP3("a", 3))
        ms.declare(BOOP1("b", 1))
        ms.declare(BOOP2("b", 2))
        ms.declare(BOOP3("b", 3))

        indexer = Indexer("A")
        indexer_update(indexer, ms)
        facts = indexer_get_facts(indexer, ms,"a")

        assert len(facts) == 3
        assert all([x.A == 'a' for x in facts])

        facts = ms.get_facts(A='b')
        assert len(facts) == 3
        assert all([x.A == 'b' for x in facts])

        x = ms.get_fact(B=1)
        assert x.B == 1

        with pytest.raises(KeyError):
            x = ms.get_fact(B=4)



# from itertools import product

# NOTE: Something funny going on here, getting errors like:
#    "Invalid use of getiter with parameters (cre.FactIterator[BOOP1])"
def _test_iter_facts():
    with cre_context("test_iter_facts"):
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3 = define_fact("BOOP3", spec3)

        ms = MemSet()
        ms.declare(BOOP1("A",1))
        ms.declare(BOOP1("B",2))
        ms.declare(BOOP1("C",3))

        @njit(cache=True)
        def iter_b1(ms):
            l = List()
            for x in ms.iter_facts(BOOP1):
                l.append(x)
            return l

        iter_b1(ms)
        raise ValueError()

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            assert isinstance(all_tf[0], BOOP1)
            assert len(all_tf) == 3
        
        # ms = MemSet()
        ms.declare(BOOP2("D",4))
        ms.declare(BOOP2("E",5))
        ms.declare(BOOP2("F",6))        

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            print(all_tf)
            assert isinstance(all_tf[0], BOOP1)
            assert len(all_tf) == 6


        # ms = MemSet()
        ms.declare(BOOP3("G",7))
        ms.declare(BOOP3("H",8))
        ms.declare(BOOP3("I",9))  

        for t,i in product(['py','nb'],[0,1]):
            all_tf = iter_b1.py_func(ms) if t else iter_b1(ms)
            print(all_tf)
            assert isinstance(all_tf[0], BOOP1)
            assert len(all_tf) == 9


with cre_context("test_mem_leaks"):
    TextField = define_fact("TextField",tf_spec)
    BOOP = define_fact("BOOP",{"A": "string", "B" : "number"})


def test_mem_leaks():

    ''' Test for MemSet leaks in mem. This test might fail if other tests fail
        even if there is nothing wrong '''
    with NRTStatsEnabled:
        with cre_context("test_mem_leaks"):
            init_used = used_bytes()

            # Empty Mem
            ms = MemSet()
            ms = None; gc.collect()
            print(used_bytes()-init_used)
            assert used_bytes()-init_used <= 0

            # Declare a bunch of stuff
            ms = MemSet()
            for i in range(100):
                tf = TextField()
                ms.declare(tf, str(i))
            tf, ms = None, None; gc.collect()
            assert used_bytes()-init_used <= 0

            # Declare More than one kind of stuff
            ms = MemSet()
            for i in range(100):
                tf = TextField(value=str(i))
                b = BOOP(A=str(i), B=i)
                ms.declare(tf, str(i))
                ms.declare(b, "B"+str(i))
            tf, ms, b = None, None, None; gc.collect()
            assert used_bytes()-init_used <= 0

            # Declare More than one kind of stuff and retract some
            ms = MemSet()
            for i in range(100):
                tf = TextField(value=str(i))
                b = BOOP(A=str(i), B=i)
                ms.declare(tf, str(i))
                ms.declare(b, "B"+str(i))
            for i in range(0,100,10):
                ms.retract(str(i))
                ms.retract("B"+str(i))
            tf, ms, b = None, None, None; gc.collect()
            # print(used_bytes()-init_used)
            assert used_bytes()-init_used <= 0

def test_free_refs():
    with NRTStatsEnabled:
        with cre_context("test_free_refs"):
            BOOP = define_fact("BOOP", {"name" : unicode_type, "nxt" : "TestLL"})
            TestLL = define_fact("TestLL", {"name" : unicode_type, "nxt" : "TestLL"})
            init_used = used_bytes(False)

            for i in range(2):
                a = TestLL("a")
                # print("a_refs", a._meminfo.refcount)
                b = TestLL("b",a)
                c = TestLL("c",a)
                # print('0: ---')
                # print("a_refs", a._meminfo.refcount)
                # print("b_refs", b._meminfo.refcount)
                a.nxt = b
                # print('1: ---')
                # print("a_refs", a._meminfo.refcount)
                # print("b_refs", b._meminfo.refcount)
                a.nxt = c
                # print('2: ---')
                # print("a_refs", a._meminfo.refcount)
                # print("b_refs", b._meminfo.refcount)
                ms = MemSet(auto_clear_refs=i==1)
                ms.declare(a)
                ms.declare(b)
                ms.declare(c)
                # print('3: ---')
                # print("a_refs", a._meminfo.refcount)
                # print("b_refs", b._meminfo.refcount)
                # if(i==0): ms.clear_refs()
                ms = None

                # print("BYTES", used_bytes()-init_used)
                # print("a_refs", a._meminfo.refcount)
                # print("b_refs", b._meminfo.refcount)
                # print("c_refs", c._meminfo.refcount)
                a,b,c = None,None, None
                # print("BYTES", used_bytes()-init_used)
                assert used_bytes(False) == init_used




def test_long_hash():
    print("START test_long_hash")
    with cre_context("test_long_hash"):
        spec1 = {"A" : "string", "B" : "number"}
        BOOP1 = define_fact("BOOP1", spec1)
        spec2 = {"inherit_from" : BOOP1, "C" : "number"}
        BOOP2 = define_fact("BOOP2", spec2)
        spec3 = {"inherit_from" : BOOP2, "D" : "number"}
        BOOP3 = define_fact("BOOP3", spec3)

        ms10 = MemSet()
        ms10.declare(BOOP1("A",1))
        ms10.declare(BOOP1("B",2))
        ms10.declare(BOOP1("C",3))

        print("--MS10--")
        hsh10 = ms10.long_hash()

        ms11 = MemSet()
        ms11.declare(BOOP1("C",3))
        ms11.declare(BOOP1("A",1))
        ms11.declare(BOOP1("B",2))

        print("--MS11--")
        hsh11 = ms11.long_hash()

        assert hsh10 == hsh11

        ms12 = copy(ms10)
        hsh12 = ms12.long_hash()

        assert hsh10 == hsh12

        ms20 = MemSet()
        ms20.declare(BOOP2("C",3))
        ms20.declare(BOOP2("A",1))
        ms20.declare(BOOP2("B",2))

        hsh20 = ms20.long_hash()
        
        
        
        assert hsh10 != hsh20

        # Ensure that hashing doesn't comute within facts 
        TextField = define_fact("TextField", {"id" : str, "value" : str})
        ms1 = MemSet()
        ms1.declare(TextField('A','4'))
        ms1.declare(TextField('B','4'))
        ms1.declare(TextField('C',''))

        ms2 = MemSet()
        ms2.declare(TextField('A',''))
        ms2.declare(TextField('B','4'))
        ms2.declare(TextField('C','4'))

        hsh1 = ms1.long_hash()
        hsh2 = ms2.long_hash()
        print(hsh1,hsh2)
        assert hsh1 != hsh2



def _test_modify_from_deref_infos():
    from cre.var import Var
    from cre.memset import memset_modify_w_deref_infos
    with cre_context("test_modify_from_deref_infos"):
        BOOP = define_fact("BOOP",{"A": "string", "B" : "number", "C" : "BOOP",  "D" : "List(str)"})
        c = BOOP("C",7)
        d = BOOP("D",8)
        b = BOOP("A", 1, c)
        print(b)

        ms = MemSet()
        ms.declare(b)

        di_ba = Var(BOOP).A.deref_infos
        di_bb = Var(BOOP).B.deref_infos
        di_bc = Var(BOOP).C.deref_infos
        di_bd = Var(BOOP).D.deref_infos

        print(b.A)
        memset_modify_w_deref_infos(ms, b, di_ba, "B")
        print(b.A)
        assert b.A == "B"

        memset_modify_w_deref_infos(ms, b, di_bb, 2.0)
        assert b.B == 2.0

        memset_modify_w_deref_infos(ms, b, di_bc, d)
        assert b.C == d

        memset_modify_w_deref_infos(ms, b, di_bc, c)
        assert b.C == c

        memset_modify_w_deref_infos(ms, b, di_bc, None)
        assert b.C == None

        # TODO: Empty list 
        # memset_modify_w_deref_infos(ms, b, di_bd, None)
        # print(b.D)
        # assert len(b.D) == 0






###################### BENCHMARKS ########################




#### helper funcs #####

with cre_context("test_memset"):
    BOOP = define_fact("BOOP",{"A": {"type" : str, "unique_id" : True}, "B" : "number"})


def _benchmark_setup():
    with cre_context("test_memset"):
        mem = MemSet()
    return (mem,), {}

#### declare_10000 ####



@njit(cache=True)
def _delcare_10000(ms):
    out = np.empty((10000,),dtype=np.uint64)
    for i in range(10000):
        out[i] = ms.declare(BOOP(str(i),i))
    return out

@pytest.mark.benchmark(group="memset")
def test_b_declare_10000(benchmark):
    benchmark.pedantic(_delcare_10000,setup=_benchmark_setup, warmup_rounds=1)

#### retract_10000 ####

def _retract_setup():
    (ms,),_ = _benchmark_setup()
    idrecs = _delcare_10000(ms)
    return (ms,idrecs), {}

@njit(cache=True)
def _retract_10000(ms,idrecs):
    for idrec in idrecs:
        ms.retract(idrec)

@pytest.mark.benchmark(group="memset")
def test_b_retract_10000(benchmark):
    benchmark.pedantic(_retract_10000,setup=_retract_setup, warmup_rounds=1)


#### get_facts_10000 ####

def get_facts_setup():
    with cre_context("test_memset"):
        ms = MemSet()
        _delcare_10000(ms)
    return (ms,), {}

@njit(cache=True)
def _get_facts_10000(ms):
    for x in ms.get_facts(BOOP):
        pass

@pytest.mark.benchmark(group="memset")
def test_b_get_facts_10000(benchmark):
    benchmark.pedantic(_get_facts_10000,setup=get_facts_setup, warmup_rounds=1)

def _get_facts_by_id_10000(ms):
    for i in range(10000):
        ms.get_fact(A=str(i))

@pytest.mark.benchmark(group="memset")
def test_b_get_facts_by_id_10000(benchmark):
    with cre_context("test_memset"):
        benchmark.pedantic(_get_facts_by_id_10000, setup=get_facts_setup, warmup_rounds=1)



if __name__ == "__main__":
    import faulthandler; faulthandler.enable()
    test_long_hash()
    # test_indexer()
    # test_declare_retract()
    # test_retroactive_register()
    # test_declare_retract_tuple_fact()
    # test_declare_overloading()
    # test_modify()
    
    # test_retract_keyerror()
    # test_subscriber()
    # test_get_facts()
    # test_mem_leaks()
    # test_get_facts()
    # _test_iter_facts()

    # _delcare_10000(MemSet())

    # _test_modify_from_deref_infos()
    # test_free_refs()
    # with PrintElapse("EEP"):
    #     with cre_context("test_memset"):
    #         (ms,),_ = get_facts_setup()

    #         d = ms.as_dict()
    #         print(d)
    #         a = ms.get_fact(A=str('0'))
    #         print(a.as_dict())
            
                
            # print(ms.get_fact(A=str(i)))
