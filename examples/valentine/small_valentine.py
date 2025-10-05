import numpy as np
from numba import njit, f8
from numba.typed import List
from cre.conditions import *
from cre.memset import MemSet
from cre.context import cre_context
from cre.utils import used_bytes, NRTStatsEnabled
from cre.utils import PrintElapse, _struct_from_ptr, _list_base,_list_base_from_ptr,_load_ptr, _incref_structref, _raw_ptr_from_struct
from numba.core.runtime.nrt import rtsys
import gc
from cre.matching import repr_match_iter_dependencies
import pytest




with cre_context("remove valentine"):
    Department = define_fact("Department", 
     {"city": "string", "num" : "int"})
    Employee = define_fact("Employee",
     {"num": "int", "home_city" : "string", "dept_num" : "int"})
    Project = define_fact("Project",
     {"proj_num" : "int", "emp_num" : "int"})

    for i in range(2):

        ms = MemSet()
        ms.declare(Employee(num=1, home_city="Seattle", dept_num=1))
        ms.declare(Employee(num=2, home_city="Orlando", dept_num=1))
        ms.declare(Employee(num=3, home_city="LA", dept_num=6))
        ms.declare(Employee(num=4, home_city="New York",dept_num=6))
        ms.declare(Employee(num=5, home_city="LA", dept_num=7))
        ms.declare(Employee(num=6, home_city="Houston", dept_num=7))
        ms.declare(Employee(num=7, home_city="Orlando", dept_num=8))
        ms.declare(Employee(num=8,  home_city="Houston", dept_num=8))
        ms.declare(Project(proj_num=10780, emp_num=8))
        ms.declare(Project(proj_num=10781, emp_num=6))
        ms.declare(Project(proj_num=10781, emp_num=5))
        ms.declare(Project(proj_num=10782, emp_num=7))
        ms.declare(Department(city="LA", num=1))
        ms.declare(Department(city="New York", num=6))
        ms.declare(Department(city="Houston", num=7))
        ms.declare(Department(city="Houston", num=8))
        
        working_memory = ms

        AND(D:=Var(Department, 'D'), D.city == "Houston",
            E:=Var(Employee,"E"), E.home_city != D.city, E.num == P.emp_num,
            P:=Var(Project,"P"), E.num == P.emp_num,
            V1:=Var(Employee,"V1"), V1.home_city != E.home_city, V1.num > E.num
        )

        D, E, P = Var(Department, 'D'), Var(Employee,"E"), Var(Project,"P")
        conds = (D.city == "Houston") & (E.home_city != D.city) & (E.num == P.emp_num)

        # N=1 Valentines
        V1 = Var(Employee,"V1")
        conds &= (V1.home_city != E.home_city) & (V1.num != E.num) & (V1.num > E.num)

        # N=2 Valentines
        # V2 = Var(Employee,"V2")
        # conds &= (V2.home_city != E.home_city) & (V2.num != E.num) & (V2.num != V1.num)

        # N=3 Valentines
        # V3 = Var(Employee,"V3")
        # conds &= (V3.home_city != E.home_city) & (V3.num != E.num) & (V3.num != V1.num) & (V3.num != V2.num)

        # N=4 Valentines
        # V4 = Var(Employee,"V4")
        # conds &= (V4.home_city != E.home_city) & (V4.num != E.num) & (V4.num != V1.num) & (V4.num != V2.num) & (V4.num != V3.num)

        if(i == 0):
            itr = conds.get_matches(ms)
            for i in range(100):
                (_, E,_,*rest) = next(itr)
                print(E.num, *[x.num for x in rest])
        else:
            with PrintElapse("match"):
                next(conds.get_matches(working_memory))

        # with PrintElapse("match"):
        #     for match in c.get_matches(ms):
        #         print(match)


    
