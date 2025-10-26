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

with cre_context("valentine"):
    Department = define_fact("Department", 
        {"city": "string", "num" : "int"})
    Employee = define_fact("Employee",
        {"num": "int", "home_city" : "string", "dept_num" : "int"})
    Project = define_fact("Project",
        {"proj_num" : "int", "emp_num" : "int"})

def make_valentines_data(reps=1):
    ms = MemSet()
    for i in range(reps):
        o = i*100
        ms.declare(Employee(num=1+o, home_city="Seattle", dept_num=1+o))
        ms.declare(Employee(num=10+o, home_city="Orlando", dept_num=1+o))
        ms.declare(Employee(num=3+o, home_city="Orlando", dept_num=1+o))
        ms.declare(Employee(num=17+o, home_city="Orlando", dept_num=2+o))
        ms.declare(Employee(num=5+o, home_city="Seattle", dept_num=2+o))
        ms.declare(Employee(num=19+o, home_city="LA", dept_num=2+o))
        ms.declare(Employee(num=7+o, home_city="Seattle", dept_num=2+o))
        ms.declare(Employee(num=8+o, home_city="Dallas", dept_num=3+o))
        ms.declare(Employee(num=24+o, home_city="LA", dept_num=4+o))
        ms.declare(Employee(num=2+o, home_city="Dallas", dept_num=4+o))
        ms.declare(Employee(num=11+o, home_city="Dallas", dept_num=5+o))
        ms.declare(Employee(num=26+o, home_city="Dallas", dept_num=5+o))
        ms.declare(Employee(num=13+o, home_city="LA", dept_num=6+o))
        ms.declare(Employee(num=23+o, home_city="New York",dept_num=6+o))
        ms.declare(Employee(num=15+o, home_city="LA", dept_num=7+o))
        ms.declare(Employee(num=16+o, home_city="LA", dept_num=7+o))
        ms.declare(Employee(num=4+o, home_city="Chicago", dept_num=7+o))
        ms.declare(Employee(num=18+o, home_city="LA", dept_num=8+o))
        ms.declare(Employee(num=6+o, home_city="LA", dept_num=8+o))
        ms.declare(Employee(num=20+o, home_city="LA", dept_num=8+o))
        ms.declare(Employee(num=21+o, home_city="LA", dept_num=9+o))
        ms.declare(Employee(num=22+o, home_city="LA", dept_num=9+o))
        ms.declare(Employee(num=14+o, home_city="LA", dept_num=9+o))
        ms.declare(Employee(num=9+o, home_city="New York",dept_num=9+o))
        ms.declare(Employee(num=25+o, home_city="New York",dept_num=10+o))
        ms.declare(Employee(num=12+o, home_city="LA", dept_num=10+o))
        ms.declare(Project(proj_num=10780+o, emp_num=7+o))
        ms.declare(Project(proj_num=10781+o, emp_num=8+o))
        ms.declare(Project(proj_num=10781+o, emp_num=9+o))
        ms.declare(Project(proj_num=10781+o, emp_num=1+o))
        ms.declare(Project(proj_num=10782+o, emp_num=2+o))
        ms.declare(Project(proj_num=10782+o, emp_num=3+o))
        ms.declare(Project(proj_num=10783+o, emp_num=4+o))
        ms.declare(Project(proj_num=10784+o, emp_num=5+o))
        ms.declare(Project(proj_num=10785+o, emp_num=6+o))
        ms.declare(Project(proj_num=10785+o, emp_num=10+o))
        ms.declare(Project(proj_num=10785+o, emp_num=11+o))
        ms.declare(Project(proj_num=10786+o, emp_num=12+o))
        ms.declare(Department(city="LA", num=1+o))
        ms.declare(Department(city="LA", num=2+o))
        ms.declare(Department(city="LA", num=3+o))
        ms.declare(Department(city="New York", num=4+o))
        ms.declare(Department(city="New York", num=5+o))
        ms.declare(Department(city="New York", num=6+o))
        ms.declare(Department(city="Houston", num=7+o))
        ms.declare(Department(city="Houston", num=8+o))
        ms.declare(Department(city="Houston", num=9+o))
        ms.declare(Department(city="Chicago", num=10+o))
        ms.declare(Department(city="Chicago", num=11+o))
        ms.declare(Department(city="Phoenix", num=12+o))

    print("WM SIZE=", len(ms))
    print(ms)
    return ms


def main(n_valentines, reps=1):
    with cre_context("valentine"):
        for i in range(2):
            working_memory = make_valentines_data(reps)

            D, E, P = Var(Department, 'D'), Var(Employee,"E"), Var(Project,"P")
            conds = (D.city == "Houston") & (E.home_city != D.city) & (E.num == P.emp_num)

            # N=1 Valentines
            if(n_valentines >= 1):
                V1 = Var(Employee,"V1")
                conds &= (V1.home_city != E.home_city) & (V1.num != E.num) & (V1.num > E.num)

            # N=2 Valentines
            if(n_valentines >= 2):
                V2 = Var(Employee,"V2")
                conds &= (V2.home_city != E.home_city) & (V2.num != E.num) & (V2.num != V1.num)

            # N=3 Valentines
            if(n_valentines >= 3):
                V3 = Var(Employee,"V3")
                conds &= (V3.home_city != E.home_city) & (V3.num != E.num) & (V3.num != V1.num) & (V3.num != V2.num)

            # N=4 Valentines
            if(n_valentines >= 4):
                V4 = Var(Employee,"V4")
                conds &= (V4.home_city != E.home_city) & (V4.num != E.num) & (V4.num != V1.num) & (V4.num != V2.num) & (V4.num != V3.num)

            # N=4 Valentines
            if(n_valentines >= 5):
                V5 = Var(Employee,"V5")
                conds &= (V5.home_city != E.home_city) & (V5.num != E.num) & (V5.num != V1.num) & (V5.num != V2.num) & (V5.num != V3.num) & (V5.num != V4.num)

            if(i == 0):
                # NOTE: Because CRE currently depends on Numba (a JIT compiler for Python)
                #  we need to run this once without timing it to eat any initial compilation
                #  or loading costs of the relevant functions that use numba. 
                itr = conds.get_matches(working_memory)
                for i in range(100):
                    (_, E,_,*rest) = next(itr)
                    print(E.num, *[x.num for x in rest])
            else:
                with PrintElapse("match"):
                    next(conds.get_matches(working_memory))

                # INSERT_YOUR_CODE

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--valentines', type=int, default=3, help='Number of valentines')
    parser.add_argument('-r', '--reps', type=int, default=1, help='Number of repeats of the data')
    args, unknown = parser.parse_known_args()
    n_valentines = args.valentines
    main(args.valentines, args.reps)


