import numpy as np
from numba import njit, f8, i8, u2, generated_jit
from numba.typed import List, Dict
from numba.types import DictType, ListType, unicode_type, Tuple
# from cre.op import Op
from cre.sc_planner import (gen_apply_multi_source, search_for_explanations,
                     apply_multi, SetChainingPlanner, insert_record,
                     join_records_of_type, forward_chain_one, extract_rec_entry,
                     retrace_goals_back_one, expl_tree_ctor, planner_declare,
                    build_explanation_tree, ExplanationTreeType, SC_Record, SC_RecordType)
from cre.utils import (_ptr_from_struct_incref, _list_from_ptr, _dict_from_ptr, _struct_from_ptr,
                        used_bytes, NRTStatsEnabled)
from cre.var import Var
from cre.context import cre_context
from cre.fact import define_fact
from cre.core import T_ID_FLOAT
from cre.func import CREFunc
# from cre.default_funcs import CastFloat, CastStr
from cre.default_funcs import CastFloat, CastStr
from numba.core.runtime.nrt import rtsys
import gc

import time
class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.2f} ms')


def get_base_funcs():
    from cre.default_funcs import Add, Multiply, Concatenate
    Add_f8 = Add(Var(f8),Var(f8))
    Multiply_f8 = Multiply(Var(f8),Var(f8))
    print(Add_f8, Multiply_f8)
    return Add_f8, Multiply_f8, Concatenate
    


i8_2x_tuple = Tuple((i8,i8))
def setup_float(planner=None,n=5):
    if(planner is None):
        planner = SetChainingPlanner()

    for x in range(5):
        planner.declare(float(x))

    return planner

def setup_str(planner=None,n=5):
    if(planner is None):
        planner = SetChainingPlanner()

    for x in range(65,n+65):
        planner.declare(chr(x))

    return planner

def test_apply_multi():
    Add, Multiply, Concatenate = get_base_funcs()
    planner = setup_float()
    rec = apply_multi(Add, planner, 0)
    d_typ = DictType(f8,i8_2x_tuple)
    f8_t_id = cre_context().get_t_id(f8)
    @njit(cache=True)
    def summary_vals_map(planner,target=6.0):
        d = _dict_from_ptr(d_typ, planner.val_map_ptr_dict[u2(f8_t_id)])
        return len(d), min(d), max(d)

    @njit(cache=True)
    def args_for(planner,target=6.0):
        d = _dict_from_ptr(d_typ, planner.val_map_ptr_dict[u2(f8_t_id)])
        l = List()
        re_ptr = d[target][1]
        # re_rec, re_next_re_ptr, re_args = extract_rec_entry(re_ptr)
        while(re_ptr != 0):
            re_rec, re_ptr, re_args = extract_rec_entry(re_ptr)
            l.append(re_args)
            # re_ptr = re_next_re_ptr
        return l
    assert summary_vals_map(planner) == (9,0.0,8.0)
    # print(np.array(args_for(planner,6)))
    assert np.array_equal(np.array(args_for(planner,6)),
                 np.array([[4, 2],[3, 3]]))
    

def test_insert_record():
    Add, Multiply, Concatenate = get_base_funcs()
    planner = setup_float()
    rec = apply_multi(Add, planner, 0)

    float64_t_id = cre_context().get_t_id(f8)
    insert_record(planner, rec, u2(float64_t_id), 1)
    @njit(cache=True)
    def len_f_recs(planner, t_id, depth):
        return len(planner.forward_records[depth][u2(t_id)])
    
    assert len_f_recs(planner,float64_t_id,1) == 1

@generated_jit(cache=True, nopython=True)
def summarize_depth_vals(planner, typ, depth):
    ''' Returns a summary of unique values of type 'typ' at 'depth':
        (len(flat_vals), min(flat_vals), max(flat_vals),
            len(val_map), min(val_map), max(val_map))
     '''
    from cre.fact import Fact
    _typ = typ.instance_type
    typ_name = str(_typ)
    typ_t_id = cre_context().get_t_id(_typ)
    l_typ = ListType(_typ)
    d_typ = DictType(_typ, Tuple((i8,i8)))
    if(isinstance(_typ, Fact)):
        def impl(planner, typ, depth): 
            print("----",typ_name, depth,"-----")
            tup = (u2(typ_t_id),depth)
            if(tup not in planner.flat_vals_ptr_dict): return 0,None, None, 0,None, None

            l = _list_from_ptr(l_typ, planner.flat_vals_ptr_dict[tup])
            d = _dict_from_ptr(d_typ, planner.val_map_ptr_dict[u2(typ_t_id)])
            print(l,d)
            first_l, last_l = None, None
            for i,x in enumerate(l):
                if(i == 0):
                    first_l = x
                if(i == len(l)-1):
                    last_l = x

            first_d, last_d = None, None
            for i,x in enumerate(d):
                if(i == 0):
                    first_d = x
                if(i == len(d)-1):
                    last_d = x


            return len(l), first_l, last_l, len(d), first_d, last_d
    else:
        def impl(planner, typ, depth): 

            print("----",typ_name, depth,"-----")
            tup = (u2(typ_t_id),depth)
            if(tup not in planner.flat_vals_ptr_dict): return 0,None, None, 0,None, None

            l = _list_from_ptr(l_typ, planner.flat_vals_ptr_dict[tup])
            d = _dict_from_ptr(d_typ, planner.val_map_ptr_dict[u2(typ_t_id)])
            # print(l,d)
            print(l)
            # print(d)
            return len(l), min(l),max(l),len(d), min(d),max(d)
    return impl

def test_join_records_of_type():
    Add, Multiply, Concatenate = get_base_funcs()
    planner = setup_float()
    rec = apply_multi(Add, planner, 0)
    insert_record(planner, rec, T_ID_FLOAT, 1)
    rec = apply_multi(Multiply, planner, 0)
    insert_record(planner, rec, T_ID_FLOAT, 1)

    # d_typ = DictType(f8, i8)
    # l_typ = ListType(f8)
    join_records_of_type(planner,1,f8)

    # @njit(cache=True)
    # def summarize_depth_vals(planner, typ_name, depth):
    #     l = _list_from_ptr(l_typ, planner.flat_vals_ptr_dict[(typ_name,depth)])
    #     d = _dict_from_ptr(d_typ, planner.val_map_ptr_dict[typ_name])
    #     return len(l), min(l),max(l),len(d), min(d),max(d)

    assert summarize_depth_vals(planner,f8, 1) == (12, 0.0, 16.0, 12, 0.0, 16.0)




def test_forward_chain_one():
    Add, Multiply, Concatenate = get_base_funcs()
    # fd_typ = DictType(f8, i8)
    # fl_typ = ListType(f8)
    # sd_typ = DictType(unicode_type, i8)
    # sl_typ = ListType(unicode_type)
    
    planner = setup_float()
    planner = setup_str(planner)
    forward_chain_one(planner, [Add,Multiply,Concatenate])

    assert summarize_depth_vals(planner,f8,1) == \
        (12, 0.0, 16.0, 12, 0.0, 16.0)

    assert summarize_depth_vals(planner,unicode_type,1) == \
        (30, 'A', 'EE', 30, 'A', 'EE')


    print("<<", planner.num_forward_inferences)

    forward_chain_one(planner, [Add,Multiply,Concatenate])


    print(summarize_depth_vals(planner,f8,1))
    assert summarize_depth_vals(planner,f8,2) == \
        (53, 0.0, 256.0, 53, 0.0, 256.0)

    print(summarize_depth_vals(planner,unicode_type,1))
    assert summarize_depth_vals(planner,unicode_type,2) == \
        (780, 'A', 'EEEE', 780, 'A', 'EEEE')


def setup_retrace(n=5):
    Add, Multiply, Concatenate = get_base_funcs()
    print(repr(Add), repr(Multiply), repr(Concatenate))
    planner = setup_float(n=n)
    planner = setup_str(planner,n=n)
    forward_chain_one(planner, [Add,Multiply,Concatenate])
    forward_chain_one(planner, [Add,Multiply,Concatenate])
    return planner


@njit(unicode_type(ExplanationTreeType,i8), cache=False)
def tree_str(root,ind=0):
    # print("START STR TREE")
    # if(len(root.children) == 0): return "?"
    s_ind = ' '*ind
    s = ''
    for entry in root.entries:
        # print("child.is_func", child.is_func)
        if(entry.is_func):
            func, child_arg_ptrs = entry.func, entry.child_arg_ptrs
        #     # for i in range(ind): s += " "
                
            s += f"\n{s_ind}{func}("
        #     # print(child_arg_ptrs)
            for i, ptr in enumerate(child_arg_ptrs):
                
                ch_expl = _struct_from_ptr(ExplanationTreeType, ptr)
                # print(ch_expl)
                # tree_str(ch_expl, ind+1)
        #         # print("str",tree_str(ch_expl, ind+1))
                s += f'{tree_str(ch_expl, ind+1)}'
                if (i != len(child_arg_ptrs) -1): s += ","
                    
                # s += ","
            s += ")"
        else:
            s += "?"
    return s        

def test_build_explanation_tree():
    planner = setup_retrace()
    print("BEF EX")
    root = build_explanation_tree(planner, f8, 36.0)
    print("BEF STR")
    for func in root:
        print(func)

def test_search_for_explanations(n=5):
    funcs = get_base_funcs()
    # print(repr(Add), repr(Multiply), repr(Concatenate))
    planner = setup_float(n=n)
    # planner = setup_str(planner,n=n)

    expl_tree = search_for_explanations(planner, 36.0, funcs=funcs, search_depth=2)
    # print(tree_str(expl_tree))
    for func in expl_tree:
        print(func)



#NOTE: Need to fix this seems to leak declared objects. 
def test_mem_leaks(n=5):
    with NRTStatsEnabled:
        with cre_context("test_mem_leaks") as context:
            funcs = get_base_funcs()
            init_used = used_bytes()

            # for i in range(5):
            #     planner = setup_float(n=n)
            #     expl_tree = search_for_explanations(planner, 36.0,
            #         funcs=funcs, search_depth=2, context=context)
            #     expl_tree_iter = iter(expl_tree)
            #     for f_comp,binding in expl_tree_iter:
            #         pass

            #     planner = None
            #     expl_tree = None
            #     expl_tree_iter = None
            #     if(i == 0): 
            #         init_used = used_bytes()
            #     else:
            #         print(used_bytes() - init_used)

            # assert used_bytes() == init_used


            BOOP = define_fact("BOOP", {
                "A" : "string",
                "B" : {"type": "number", "visible":  True, "semantic" : True}
            })
            
            def declare_em(planner,s="A"):
                for i in range(n):
                    b = BOOP(s,i)
                    print("BEF", b._meminfo.refcount)
                    planner.declare(b)
                    print("AFT", b._meminfo.refcount)
                return b

            print("-----")
            for i in range(5):
                planner = SetChainingPlanner([BOOP])
                b = declare_em(planner,"A")
                # expl_tree = search_for_explanations(planner, 36.0,
                #     funcs=funcs, search_depth=2, context=context)
                # expl_tree_iter = iter(expl_tree)
                # for f_comp,binding in expl_tree_iter:
                #     pass
                print("<<", b._meminfo.refcount)

                planner = None
                expl_tree = None
                expl_tree_iter = None
                b = None
                if(i == 0): 
                    init_used = used_bytes()
                else:
                    print(used_bytes() - init_used)

            assert used_bytes() == init_used




def test_declare_fact():
    with cre_context("test_declare_fact"):
        # print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        BOOP = define_fact("BOOP", {
            "A" : "string",
            "B" : {"type": "number", "visible":  True, "semantic" : True}
        })
        
        def declare_em(planner,s="A"):
            for i in range(5):
                b = BOOP(s,i)
                planner.declare(b)

        planner = SetChainingPlanner([BOOP])
        declare_em(planner,"A")

        # print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        # print(summarize_depth_vals(planner, BOOP, 0))
        # print(summarize_depth_vals(planner, unicode_type, 0))
        # print(summarize_depth_vals(planner, f8, 0))
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        assert summarize_depth_vals(planner, BOOP, 0)[0] == 5
        # assert summarize_depth_vals(planner, unicode_type, 0)[0] == 5
        assert summarize_depth_vals(planner, f8, 0)[0] == 5


        expls = planner.search_for_explanations(36.0, funcs=get_base_funcs(), search_depth=2)
        A_f_comp_binding_pairs = list(iter(expls))

        planner = SetChainingPlanner([BOOP])
        declare_em(planner,"A")
        declare_em(planner,"B")
        
        assert summarize_depth_vals(planner, BOOP, 0)[0] == 10
        # assert summarize_depth_vals(planner, unicode_type, 0)[0] == 5
        assert summarize_depth_vals(planner, f8, 0)[0] == 5

        expls = planner.search_for_explanations(36.0, funcs=get_base_funcs(), search_depth=2)
        AB_f_comp_binding_pairs = list(iter(expls))

        assert len(AB_f_comp_binding_pairs) >= 4 * len(A_f_comp_binding_pairs)

        # print("(()()))")
        for i, (func, binding) in enumerate(A_f_comp_binding_pairs):
            print("<<", func, binding)


            assert(func(*binding)==36.0)

            # Flattening takes a while so stop after 3 
            if(i >= 2): break
            

def test_declare_fact_w_conversions():
    with cre_context("test_declare_fact_w_conversions"):
        BOOP = define_fact("BOOP", {
            "A" : str,
            "B" : {"type": str, "visible":  True,
                 "semantic" : True, 'conversions' : {float : CastFloat}}
        })
        
        def declare_em(planner,s="A"):
            for i in range(5):
                b = BOOP(s,str(i))
                planner.declare(b)

            # Declare one that can't be converted
            planner.declare(BOOP(s,s))

        planner = SetChainingPlanner([BOOP])
        declare_em(planner,"A")
        print("---------------------------------------------------------")

        assert summarize_depth_vals(planner, BOOP, 0)[0] == 6
        assert summarize_depth_vals(planner, unicode_type, 0)[0] == 6
        assert summarize_depth_vals(planner, f8, 0)[0] == 5

        expls = planner.search_for_explanations(36.0, funcs=get_base_funcs(), search_depth=2)
        A_f_comp_binding_pairs = list(iter(expls))

        assert len(A_f_comp_binding_pairs) > 0

        print("---------------------------------------------------------")

        planner = SetChainingPlanner([BOOP])
        declare_em(planner,"A")
        declare_em(planner,"B")
        
        assert summarize_depth_vals(planner, BOOP, 0)[0] == 12
        assert summarize_depth_vals(planner, unicode_type, 0)[0] == 7
        assert summarize_depth_vals(planner, f8, 0)[0] == 5

        expls = planner.search_for_explanations(36.0, funcs=get_base_funcs(), search_depth=2)
        AB_f_comp_binding_pairs = list(iter(expls))

        assert len(AB_f_comp_binding_pairs) >= 4 * len(A_f_comp_binding_pairs)

        for i, (func, binding) in enumerate(A_f_comp_binding_pairs):
            print("<<", func, binding)

            assert(func(*binding)==36.0)

            # Flattening takes a while so stop after 3 
            if(i >= 2): break

        print("---------------------------------------------------------")
        # Check for key error bug when don't have funcs for all decalared types.
        from cre.default_funcs import Add, Multiply
        Add_f8 = Add(f8, f8)
        Multiply_f8 = Multiply(f8, f8)
        funcs = [Add_f8, Multiply_f8]
        planner = SetChainingPlanner([BOOP])
        
        for i in range(5):
            b = BOOP("A",str(i))
            planner.declare(b)

        expls = planner.search_for_explanations(36.0, funcs=funcs, search_depth=2)
        AB_f_comp_binding_pairs = list(iter(expls))

        assert len(AB_f_comp_binding_pairs) > 0

        for i, (func, binding) in enumerate(AB_f_comp_binding_pairs):
            print("<<", func, binding)

            assert(func(*binding)==36.0)

            # Flattening takes a while so stop after 3 
            if(i >= 2): break


        # Special Example
        # TODO: WRITE ASSERTIONS FOR THIS
        @CREFunc(signature=f8(f8,f8),
            shorthand = '{0} + {1}',
            commutes=True)
        def Add(a, b):
            return a + b

        @CREFunc(signature=f8(f8,f8,f8),
            shorthand = '{0} + {1} + {2}',
            commutes=True)
        def Add3(a, b, c):
            return a + b + c

        @CREFunc(signature=f8(f8), shorthand = 'TensDigit({0})')
        def TensDigit(a):
            return (a // 10) % 10

        print("THIS", TensDigit(12.0), TensDigit(7))

        @CREFunc(signature=f8(f8), shorthand = 'OnesDigit({0})')
        def OnesDigit(a):
            return a % 10

        funcs = [OnesDigit, TensDigit, Add, Add3]
        planner = SetChainingPlanner([BOOP])
        planner.declare(BOOP("A","7"))
        planner.declare(BOOP("B","7"))
        planner.declare(BOOP("C","1"))
        expls = planner.search_for_explanations(1.0, funcs=funcs, search_depth=2, min_stop_depth=2, min_solution_depth=1)
        if(expls is None):
            print("NO EXPLANATIONS")
        else:
            for f in expls:
                print(f)
        print(summarize_depth_vals(planner, f8, 0))
        print(summarize_depth_vals(planner, f8, 1))
        print(summarize_depth_vals(planner, f8, 2))


        funcs = [OnesDigit, TensDigit, Add3]
        planner = SetChainingPlanner([BOOP])
        planner.declare(BOOP("A","7"))
        planner.declare(BOOP("B","7"))
        planner.declare(BOOP("C","1"))
        expls = planner.search_for_explanations(5.0, funcs=funcs, search_depth=2, min_stop_depth=1)
        if(expls is None):
            print("NO EXPLANATIONS")
        else:
            for f in expls:
                print(f)
        print(summarize_depth_vals(planner, f8, 0))
        print(summarize_depth_vals(planner, f8, 1))
        print(summarize_depth_vals(planner, f8, 2))


        # AB_f_comp_binding_pairs = list(iter(expls))




def test_min_stop_depth():
    with cre_context("test_min_stop_depth"):
        BOOP = define_fact("BOOP", {
            "A" : str,
            "B" : {"type": str, "visible":  True,
                 "semantic" : True, 'conversions' : {float : CastFloat}}
        })
        from cre.default_funcs import Add, Multiply
        Add_f8 = Add(f8, f8)
        Multiply_f8 = Multiply(f8, f8)

        planner = SetChainingPlanner([BOOP])

        for i in range(5):
            b = BOOP("A",str(i))
            planner.declare(b)

        expls = planner.search_for_explanations(36.0, funcs=[Add_f8, Multiply_f8], search_depth=2)
        new_func, match = list(expls)[0]
        print(new_func, match)
        # new_func = f_comp.flatten()

        planner = SetChainingPlanner([BOOP])

        for i in range(5):
            b = BOOP("A",str(i))
            planner.declare(b)
        planner.declare(BOOP("Q","36"))
        print("----------------------------------------")

        expls = planner.search_for_explanations(36.0, funcs=[new_func], 
            search_depth=1, min_stop_depth=1)
        assert len(list(expls)) > 0

        func, match = list(expls)[0]

        print(func, match)

        planner = SetChainingPlanner([BOOP])
        for i in range(2,5):
            b = BOOP("A",str(i))
            planner.declare(b)

        expls = planner.search_for_explanations(3.0, funcs=[new_func], 
            search_depth=1, min_stop_depth=1, min_solution_depth=1)

        assert expls == None

        print("----------------------------------------")
        planner = SetChainingPlanner([BOOP])
        for i in range(5):
            b = BOOP("A",str(i))
            planner.declare(b)
        # planner.declare(BOOP("Q","36"))

        expls = planner.search_for_explanations(36.0, funcs=[Add_f8, Multiply_f8],
                    min_stop_depth=1, search_depth=1)

        print(summarize_depth_vals(planner, BOOP, 1))
        print(summarize_depth_vals(planner, unicode_type, 1))
        print(summarize_depth_vals(planner, f8, 1))
        # for i, (f_comp, binding) in enumerate(expls):
        #     print(f_comp, binding)
        # raise ValueError()

        print("END")

        expls = planner.search_for_explanations(36.0, funcs=[Add_f8, Multiply_f8],
                    min_stop_depth=1, search_depth=2)

        print(summarize_depth_vals(planner, BOOP, 2))
        print(summarize_depth_vals(planner, unicode_type, 2))
        print(summarize_depth_vals(planner, f8, 1))
        print(summarize_depth_vals(planner, f8, 2))

        all_expls =  list(iter(expls))
        print(len(all_expls))

        # raise ValueError()

        for i, (func, match) in enumerate(expls):
            print(func, match)
            print(func(*match))

        print("END")



def test_non_numerical_vals():
    with cre_context("test_min_stop_depth"):
        BOOP = define_fact("BOOP", {
            "A" : str,
            "B" : {"type": str, "visible":  True,
                 "semantic" : True, 'conversions' : {float : CastFloat}}
        })
        from cre.default_funcs import Add, Multiply
        Add_f8 = Add(f8, f8)

        planner = SetChainingPlanner([BOOP])
        planner.declare(BOOP("A",'1'))
        planner.declare(BOOP("B",'+'))
        planner.declare(BOOP("C",'1'))

        print(summarize_depth_vals(planner,BOOP, 0))
        print(summarize_depth_vals(planner,unicode_type, 0))
        print(summarize_depth_vals(planner,f8, 0))

        expls = planner.search_for_explanations(2.0, funcs=[Add_f8], 
            search_depth=1)

        for i, (func, binding) in enumerate(expls):
            print(func, binding)

def test_const_funcs():
    with cre_context("test_const_funcs"):
        BOOP = define_fact("BOOP", {
            "A" : str,
            "B" : {"type": str, "visible":  True,
                 "semantic" : True, 'conversions' : {float : CastFloat}}
        })
        from cre.default_funcs import Add, Multiply
        Add_f8 = Add(f8, f8)

        planner = SetChainingPlanner([BOOP])
        planner.declare(BOOP("A",'1'))
        planner.declare(BOOP("B",'+'))
        planner.declare(BOOP("C",'1'))

        @CREFunc(signature=f8(), no_raise=True, shorthand='10')
        def Ten():
            return 10

        print(summarize_depth_vals(planner,BOOP, 0))
        print(summarize_depth_vals(planner,unicode_type, 0))
        print(summarize_depth_vals(planner,f8, 0))

        expls = planner.search_for_explanations(11.0, funcs=[Add_f8, Ten], 
            search_depth=1)

        for i, (func, match) in enumerate(expls):
            print(func, match)
            print(func(*match))
            assert func(*match) == 11.0
            


def test_policy_search(n=5):
    [Add_f8, Multiply_f8, Concatenate] = funcs = get_base_funcs()
        
    # No Policy
    planner = setup_float(n=n)
    expl_tree = search_for_explanations(planner, 36.0, funcs=funcs, search_depth=2)
    no_policy_expls = list(expl_tree)    

    # Policy
    policy = [[Add_f8],[Multiply_f8]]
    planner = setup_float(n=n)
    expl_tree = search_for_explanations(planner, 36.0, policy=policy, search_depth=2)
    policy_expls = list(expl_tree)    

    for expl in no_policy_expls:
        print(expl)
    print("----------------", len(no_policy_expls))
    for i, expl in enumerate(set([str(x) for x in no_policy_expls])):
        print(i, expl)        
    print("----------------")
    for expl in policy_expls:
        print(expl)

    # Policy w/ Args
    policy = [[(Add_f8, [4.0,2.0])],[(Multiply_f8, [])]]
    planner = setup_float(n=n)
    expl_tree = search_for_explanations(planner, 36.0, policy=policy, search_depth=2)
    policy_expls = list(expl_tree)    
    
    assert len(policy_expls) < len(no_policy_expls)

def test_divide():
    from cre.default_funcs import Divide, Multiply
    Divide_f8 = Divide(Var(f8),Var(f8))
    Multiply_f8 = Multiply(Var(f8),Var(f8))


    planner = SetChainingPlanner()
    for x in [360, 135, 6]:
        planner.declare(float(x))

    expls = planner.search_for_explanations(13.5, funcs=[Divide_f8, Multiply_f8], 
            search_depth=3, min_stop_depth=1)
    for f, m in expls:
        print(f.depth, f, m)


    planner = SetChainingPlanner()
    for x in [2, 12, 3]:
        planner.declare(float(x))

    expls = planner.search_for_explanations(8.0, funcs=[Divide_f8, Multiply_f8], 
            search_depth=3, min_stop_depth=1)

    print(summarize_depth_vals(planner, f8, 0))
    print(summarize_depth_vals(planner, f8, 1))
    print(summarize_depth_vals(planner, f8, 2))

    for f, m in expls:
        print(f.depth, f, m)


def test_const_declarations():
    from cre.default_funcs import Divide, Multiply
    Divide_f8 = Divide(Var(f8),Var(f8))
    Multiply_f8 = Multiply(Var(f8),Var(f8))


    planner = SetChainingPlanner()
    planner.declare(float(360), is_const=True)
    for x in [135, 6]:
        planner.declare(float(x))

    # Test No Policy
    expls = planner.search_for_explanations(13.5, funcs=[Divide_f8, Multiply_f8], 
            search_depth=3, min_stop_depth=1)
    for f, m in expls:
        print(f.depth, f, m)


    planner = SetChainingPlanner()
    planner.declare(float(360), is_const=True)
    for x in [135, 6]:
        planner.declare(float(x))

    # Test Policy
    policy = [[(Divide_f8, [135.0, 360.0]), (Multiply_f8, [6.0, 6.0])], [(Multiply_f8, [])]]
    expls = planner.search_for_explanations(13.5, policy=policy, 
            search_depth=3, min_stop_depth=1)
    for f, m in expls:
        print(f.depth, f, m)

    # Another test
    planner = SetChainingPlanner()
    for x in [2, 12, 3]:
        planner.declare(float(x))

    expls = planner.search_for_explanations(8.0, funcs=[Divide_f8, Multiply_f8], 
            search_depth=3, min_stop_depth=1)

    print(summarize_depth_vals(planner, f8, 0))
    print(summarize_depth_vals(planner, f8, 1))
    print(summarize_depth_vals(planner, f8, 2))

    for f, m in expls:
        print(f.depth, f, m)





        # planner = SetChainingPlanner([BOOP])
        # planner.declare(7.0)
        # planner.declare(1.0)
        # planner.declare(6.0)
        # planner.declare(9.0)
        # planner.declare(5.0)
        # planner.declare(4.0)
        # expls = planner.search_for_explanations(36.0, funcs=funcs, search_depth=2)







        # gen_src_declare_fact(BOOP, ["A","B"])
        # gen_src_declare_fact(BOOP, ["A","B"])

        # planner = SetChainingPlanner()
        # b = BOOP("A",1.0)
        # planner_declare_fact(planner,b,[("B","unicode_type"), ("B", 'float')])


    # goals = Dict.empty(f8,ExplanationTreeType)
    # goals[36.0] = expl_tree_ctor()
    # retrace_goals_back_one(planner, goals)
    # goals = List([36.0])
    # e_trees = List([expl_tree_ctor()])
    # print(retrace_back_one(planner, DictType(f8,i8),'float64', goals, e_trees))

    # goals = List(["AABC"])
    # e_trees = List([expl_tree_ctor()])
    # print(retrace_back_one(planner, DictType(unicode_type,i8),'unicode_type', goals, e_trees))

    # build_explanation_tree(planner,36.0, f8)


def benchmark_apply_multi():
    Add, Multiply, Concatenate = get_base_funcs()
    planner = setup_float(n=1000)

    apply_multi(Add, planner, 0)
    with PrintElapse("benchmark_apply_multi"):
        for i in range(10):
            apply_multi(Add, planner, 0)

def benchmark_retrace_goals_back_one():
    Add, Multiply, Concatenate = get_base_funcs()
    planner = setup_retrace()
    goals = List([36.0])

    apply_multi(Add, planner, 0)
    with PrintElapse("benchmark_retrace_back_one"):
        for i in range(10):
            retrace_goals_back_one(planner, DictType(f8,i8),'float64', goals)



# @njit(cache=False)
# def foo_gen():
#     for i in range(10):
#         yield i

# def product_of_generators(generators):
#     iters = []
#     out = []
    
#     while(True):
#         #Create any iterators that need to be created
#         while(len(iters) < len(generators)):
#             it = generators[len(iters)]()
#             iters.append(it)
        
#         iter_did_end = False
#         while(len(out) < len(iters)):
#             #Try to fill in any missing part of out
#             try:
#                 nxt = next(iters[len(out)])
#                 out.append(nxt)
#             #If any of the iterators failed pop up an iterator
#             except StopIteration as e:
#                 # Stop yielding when 0th iter fails
#                 if(len(iters) == 1):
#                     return
#                 out = out[:-1]
#                 iters = iters[:-1]
#                 iter_did_end = True

#         if(iter_did_end): continue

#         yield out
#         out = out[:-1]

# with PrintElapse("gen_iters"):
#     l = [x for x in product_of_generators([foo_gen,foo_gen,foo_gen, foo_gen])]
#     print(len(l))
#     print()





if __name__ == "__main__":
    # Makes it easier to track down segfaults
    import faulthandler; faulthandler.enable()
    # test_non_numerical_vals()
    # with PrintElapse("test_build_explanation_tree"):
    #     test_build_explanation_tree()
    # with PrintElapse("test_build_explanation_tree"):
    #     test_build_explanation_tree()
    # with PrintElapse("test_search_for_explanations"):
    #     test_search_for_explanations()
    # with PrintElapse("test_search_for_explanations"):
    #     test_search_for_explanations()
# 

    # pass
    # test_apply_multi()
    # test_insert_record()
    # test_join_records_of_type()
    # test_forward_chain_one()
    # test_build_explanation_tree()
    # test_search_for_explanations()
    # test_divide()
    # test_declare_fact()
    # test_mem_leaks(n=10)
    # benchmark_apply_multi()
    # benchmark_retrace_back_one()
        # test_apply_multi()
    # gen = foo_gen()
    # for i in gen:
    #     print(i)
    # test_declare_fact()
    # test_declare_fact()
    # test_declare_fact_w_conversions()
    test_min_stop_depth()
    # test_const_funcs()
    # test_const_declarations()

    # test_policy_search()
# from numba import njit, i8
# from numba.typed import Dict
# from numba.types import ListType
# import numpy as np
# import dill
# from cre.utils import _struct_from_ptr, _ptr_from_struct_incref
# from cre.condensed_chainer import CondensedRecord
# from cre_cache.Add._5e24697b8e500d3d837dca80591bde623483d2322c5204e56fd36c79ddc2ed7d import call, check

# typ0, = dill.loads(b'\x80\x04\x95\xd5\x00\x00\x00\x00\x00\x00\x00\x8c\x19numba.core.types.abstract\x94\x8c\x13_type_reconstructor\x94\x93\x94\x8c\x07copyreg\x94\x8c\x0e_reconstructor\x94\x93\x94\x8c\x18numba.core.types.scalars\x94\x8c\x05Float\x94\x93\x94\x8c\ndill._dill\x94\x8c\n_load_type\x94\x93\x94\x8c\x06object\x94\x85\x94R\x94N\x87\x94}\x94(\x8c\x04name\x94\x8c\x07float64\x94\x8c\x08bitwidth\x94K@\x8c\x05_code\x94K\x17u\x87\x94R\x94\x85\x94.')
# ret_typ = dill.loads(b'\x80\x04\x95\xd3\x00\x00\x00\x00\x00\x00\x00\x8c\x19numba.core.types.abstract\x94\x8c\x13_type_reconstructor\x94\x93\x94\x8c\x07copyreg\x94\x8c\x0e_reconstructor\x94\x93\x94\x8c\x18numba.core.types.scalars\x94\x8c\x05Float\x94\x93\x94\x8c\ndill._dill\x94\x8c\n_load_type\x94\x93\x94\x8c\x06object\x94\x85\x94R\x94N\x87\x94}\x94(\x8c\x04name\x94\x8c\x07float64\x94\x8c\x08bitwidth\x94K@\x8c\x05_code\x94K\x17u\x87\x94R\x94.')

# l_typ0 = ListType(typ0)
# @njit(cache=True)
# def Add_apply_multi(planner, depth, start0=0, start1=0):
#     tup0 = ('float64',depth)
#     print("START")
#     if(tup0 in planner.flat_vals_ptr_dict):
#         iter_ptr0 = planner.flat_vals_ptr_dict[tup0]
#         iter0 = _list_from_ptr(l_typ0, iter_ptr0)
#         print(iter0)
#     else:
#         return None
#     print("End")
#     for x in iter0:
#         print(x)
#     print("?",len(iter0))
#     l0, l1 = len(iter0)-start0, len(iter0)-start1
#     hist_shape = (l0, l1)
#     print(hist_shape )
#     hist = np.zeros(hist_shape, dtype=np.uint64)
#     vals_to_uid = Dict.empty(ret_typ, i8)
#     print(hist)
#     uid=1
#     for i0 in range(start0,len(iter0)):
#         for i1 in range(start1,len(iter0)):
#             print(i0, i1)
#             a0,a1 = iter0[i0],iter0[i1]
#             if(not check(a0,a1)): continue
#             v = call(a0,a1)
#             if(v in vals_to_uid):
#                 hist[i0,i1] = vals_to_uid[v]
#             else:
#                 hist[i0,i1] = uid
#                 vals_to_uid[v] = uid; uid+=1
#     vals_to_uid_ptr = _ptr_from_struct_incref(vals_to_uid)
#     return CondensedRecord(hist.flatten(), hist_shape, vals_to_uid_ptr)
    

# print(apply_multi(Add,planner, 0))
# print(


