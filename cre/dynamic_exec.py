import numpy as np
from numba import f8, u1, u2, i8, u8, types, njit
from numba.types import FunctionType, unicode_type, ListType, boolean
from numba.typed import List
from numba.extending import  overload, overload_method
from cre.utils import _load_ptr, _struct_from_ptr, _cast_structref, _raw_ptr_from_struct, _raw_ptr_from_struct_incref, CastFriendlyMixin, decode_idrec, _func_from_address, _incref_structref, _struct_get_data_ptr, _sizeof_type, _decref_structref
from cre.structref import define_structref
from cre.cre_object import CREObjTypeClass, CREObjType, member_info_type, cre_obj_iter_t_id_item_ptrs
from numba.core.datamodel import default_manager, models
from numba.experimental.structref import define_attributes, StructRefProxy, new, define_boxing
import operator
from cre.core import T_ID_CONDITIONS, T_ID_LITERAL, T_ID_OP, T_ID_FACT, T_ID_VAR, T_ID_UNDEFINED, T_ID_BOOL, T_ID_INT, T_ID_FLOAT, T_ID_STR, T_ID_TUPLE_FACT
# from cre.primitive import BooleanPrimitiveType, IntegerPrimitiveType, FloatPrimitiveType, StringPrimitiveType
from cre.tuple_fact import TupleFact
from cre.var import GenericVarType
from cre.op import GenericOpType
from cre.fact import BaseFact
from cre.conditions import LiteralType, ConditionsType
from numba.cpython.hashing import _Py_hash_t, _Py_uhash_t, _PyHASH_XXROTATE, _PyHASH_XXPRIME_1, _PyHASH_XXPRIME_2, _PyHASH_XXPRIME_5, process_return
from cre.hashing import accum_item_hash

cast = _cast_structref


    



### __eq___ ###


@njit(boolean(u2,i8,i8), cache=True)
def eq_from_t_id_ptr(t_id, data_ptr_a, data_ptr_b):
    if(t_id == T_ID_BOOL):
        return _load_ptr(boolean, data_ptr_a) == _load_ptr(boolean, data_ptr_b)
    elif(t_id == T_ID_INT):
        return _load_ptr(i8, data_ptr_a) == _load_ptr(i8, data_ptr_b)
    elif(t_id == T_ID_FLOAT):
        return _load_ptr(f8, data_ptr_a) == _load_ptr(f8, data_ptr_b)
    elif(t_id == T_ID_STR):
        return _load_ptr(unicode_type, data_ptr_a) == _load_ptr(unicode_type, data_ptr_b)
    return False





@njit(boolean(CREObjType, CREObjType),cache=True)
def var_eq(a, b):
    t_id_b,_, _ = decode_idrec(b.idrec)
    if(t_id_b != T_ID_VAR): return False

    va = _cast_structref(GenericVarType, a)
    vb = _cast_structref(GenericVarType, b)

    if(va.base_ptr != vb.base_ptr): return False

    if(len(va.deref_infos) != len(vb.deref_infos)): return False
    for deref_info_a, deref_info_b in zip(va.deref_infos, vb.deref_infos):
        if(deref_info_a.offset != deref_info_b.offset): return False

    
    return True

@njit(boolean(CREObjType, CREObjType),cache=True)
def op_eq(a, b):
    t_id_b,_, _ = decode_idrec(b.idrec)
    if(t_id_b != T_ID_OP): return False

    oa = _cast_structref(GenericOpType, a)
    ob = _cast_structref(GenericOpType, b)

    if(oa.call_addr != ob.call_addr): return False
    if(len(oa.head_vars) != len(ob.head_vars)): return False

    for oa_var, ob_var in zip(oa.head_vars, ob.head_vars):
        if(not var_eq(oa_var, ob_var)): return False

    return True

@njit(boolean(CREObjType, CREObjType),cache=True)
def literal_eq(a, b):
    t_id_b,_, _ = decode_idrec(b.idrec)
    if(t_id_b != T_ID_LITERAL): return False

    la = _cast_structref(LiteralType, a)
    lb = _cast_structref(LiteralType, b)

    if(not op_eq(la.op, lb.op)): return False
    if(la.negated != lb.negated): return False
    
    return True

@njit(boolean(CREObjType, CREObjType),cache=True)
def conds_eq(a,b):
    t_id_b,_, _ = decode_idrec(b.idrec)
    if(t_id_b != T_ID_CONDITIONS): return False
    xa = _cast_structref(ConditionsType, a)
    xb = _cast_structref(ConditionsType, b)

    if(len(xa.dnf) != len(xb.dnf)): return False

    for conjuct_a, conjuct_b in zip(xa.dnf, xb.dnf):
        if(len(conjuct_a) != len(conjuct_b)): return False
        for lit_a, lit_b in zip(conjuct_a, conjuct_b):
            if(not literal_eq(lit_a, lit_b)): return False
    return True

@njit(boolean(CREObjType, CREObjType),cache=True)
def fact_eq(a, b):
    ''' based roughly on _tuple_hash from numba.cpython.hashing'''
    # while(not is_done):
    tla = a.num_chr_mbrs
    tlb = b.num_chr_mbrs

    if(_raw_ptr_from_struct(a) == _raw_ptr_from_struct(b)):
        return True

    if(tla != tlb): return False

    for (info_a), (info_b) in zip(cre_obj_iter_t_id_item_ptrs(a),cre_obj_iter_t_id_item_ptrs(b)):
        t_id_a, m_id_a, data_ptr_a = info_a
        t_id_b, m_id_b, data_ptr_b = info_b
        if(t_id_a == T_ID_UNDEFINED): 
            t_id_a,_, _ = decode_idrec(_struct_from_ptr(CREObjType, _load_ptr(i8, data_ptr_a) ).idrec)
        if(t_id_b == T_ID_UNDEFINED): 
            t_id_b,_, _ = decode_idrec(_struct_from_ptr(CREObjType, _load_ptr(i8, data_ptr_b) ).idrec)

        if(t_id_a != t_id_b): return False

        if(t_id_a == T_ID_TUPLE_FACT):
            raise Exception()
        elif(t_id_a == T_ID_FACT):
            if(not _load_ptr(i8, data_ptr_a) == _load_ptr(i8, data_ptr_b)): return False
        else:
            if(not eq_from_t_id_ptr(t_id_a, data_ptr_a, data_ptr_b)): return False

    
    return True


@njit(boolean(CREObjType, CREObjType),cache=True)
def tuple_fact_eq(a, b):
    ''' based roughly on _tuple_hash from numba.cpython.hashing'''
    t_id_b,_, _ = decode_idrec(b.idrec)
    if(t_id_b != T_ID_TUPLE_FACT): return False

    stack_buffer = None
    tf_a = _cast_structref(TupleFact, a)
    tf_b = _cast_structref(TupleFact, b)
    stack_head = -1
    is_done = False

    # acc = _PyHASH_XXPRIME_5
    while(not is_done):
        tla = tf_a.num_chr_mbrs
        tlb = tf_b.num_chr_mbrs

        if(tla != tlb): return False

        for (info_a), (info_b) in zip(cre_obj_iter_t_id_item_ptrs(tf_a),cre_obj_iter_t_id_item_ptrs(tf_b)):
            t_id_a, m_id_a, data_ptr_a = info_a
            t_id_b, m_id_b, data_ptr_b = info_b

            if(t_id_a == T_ID_UNDEFINED): 
                t_id_a,_, _ = decode_idrec(_struct_from_ptr(CREObjType, _load_ptr(i8, data_ptr_a) ).idrec)
            if(t_id_b == T_ID_UNDEFINED): 
                t_id_b,_, _ = decode_idrec(_struct_from_ptr(CREObjType, _load_ptr(i8, data_ptr_b) ).idrec)


            if(t_id_a != t_id_b): return False

            if(t_id_a == T_ID_TUPLE_FACT):
                # Use an inline stack instead of recrusion because numba breaks on recursion
                stack_head += 1
                if(stack_buffer is None):
                    stack_buffer = np.empty((2,2), dtype=np.int64)
                if(stack_head >= len(stack_buffer)):
                    new_stack_buffer  = np.empty(((stack_head+1)*2,2), dtype=np.int64)
                    new_stack_buffer[:len(stack_buffer)] = stack_buffer
                    stack_buffer = new_stack_buffer
                stack_buffer[stack_head,0] = _load_ptr(i8, data_ptr_a)
                stack_buffer[stack_head,1] = _load_ptr(i8, data_ptr_b)

            elif(t_id_a <= T_ID_STR):
                    if(not eq_from_t_id_ptr(t_id_a, data_ptr_a, data_ptr_b)): return False
                    
            else:

                # Kind of a heavy handed way to do this... but lack
                #  of recrusion makes it the best option                    
                mbr_a = _struct_from_ptr(CREObjType,_load_ptr(i8,data_ptr_a))
                mbr_b = _struct_from_ptr(CREObjType,_load_ptr(i8,data_ptr_b))
                if(t_id_a==T_ID_VAR):
                    if(not var_eq(mbr_a,mbr_b)): return False
                elif(t_id_a==T_ID_OP):
                    if(not op_eq(mbr_a,mbr_b)): return False
                elif(t_id_a==T_ID_LITERAL):
                    if(not literal_eq(mbr_a,mbr_b)): return False
                elif(t_id_a==T_ID_CONDITIONS):
                    if(not conds_eq(mbr_a,mbr_b)): return False
                else:
                    if(not fact_eq(mbr_a,mbr_b)): return False

                

        if(stack_head > -1):
            tf_a = _struct_from_ptr(TupleFact, stack_buffer[stack_head,0]);
            tf_b = _struct_from_ptr(TupleFact, stack_buffer[stack_head,1]);
            stack_head -=1;
        else:
            is_done = True

    return True



@overload(operator.eq)
def _cre_obj_eq(a,b):
    # print(a,b, isinstance(a,CREObjTypeClass) and isinstance(b,CREObjTypeClass))
    if(isinstance(a,CREObjTypeClass) and isinstance(b,CREObjTypeClass)):
        def impl(a, b):
            t_id,_, _ = decode_idrec(a.idrec)
            if(t_id==T_ID_TUPLE_FACT):
                return tuple_fact_eq(a,b)
            elif(t_id==T_ID_VAR):
                return var_eq(a,b)
            elif(t_id==T_ID_OP):
                return op_eq(a,b)
            elif(t_id==T_ID_LITERAL):
                return literal_eq(a,b)
            elif(t_id==T_ID_CONDITIONS):
                return conds_eq(a,b)
            else:
                return fact_eq(a,b)
            
        return impl


### __hash___ ###


# print(_PyHASH_XXPRIME_1, _PyHASH_XXPRIME_2, _PyHASH_XXPRIME_5)

@njit(_Py_uhash_t(u2,i8))
def hash_from_t_id_ptr(t_id, data_ptr):
    if(t_id == T_ID_BOOL):
        return hash(_load_ptr(boolean, data_ptr))
    elif(t_id == T_ID_INT):
        return hash(_load_ptr(i8, data_ptr))
    elif(t_id == T_ID_FLOAT):
        return hash(_load_ptr(f8, data_ptr))
    elif(t_id == T_ID_STR):
        return hash(_load_ptr(unicode_type, data_ptr))
    return u8(-1)













@njit(_Py_hash_t(CREObjType),cache=True)
def var_hash(x):
    ''' based roughly on _tuple_hash from numba.cpython.hashing'''
    # while(not is_done):
    if(x.hash_val == 0):
        vx = _cast_structref(GenericVarType, x)

        acc = _PyHASH_XXPRIME_5
        acc = accum_item_hash(acc,  vx.base_ptr) 
        acc = accum_item_hash(acc, len(vx.deref_infos))
        
        for deref_info in vx.deref_infos:
            acc = accum_item_hash(acc, deref_info.offset) 
        x.hash_val = _Py_hash_t(acc)

    
    return x.hash_val


@njit(_Py_hash_t(CREObjType),cache=True)
def op_hash(x):
    if(x.hash_val == 0):
        ox = _cast_structref(GenericOpType, x)

        acc = _PyHASH_XXPRIME_5
        
        acc = accum_item_hash(acc, ox.call_addr) 
        acc = accum_item_hash(acc, len(ox.head_vars))
        # print("??", ox.head_ranges, ox.call_addr)
        for ox_var in ox.head_vars:
            # print("<<:", var_hash(ox_var), ox_var.alias)
            acc = accum_item_hash(acc, var_hash(ox_var)) 
        x.hash_val = acc

    return x.hash_val



@njit(_Py_hash_t(CREObjType),cache=True)
def literal_hash(x):
    if(x.hash_val == 0):
        lx = _cast_structref(LiteralType, x)

        acc = op_hash(lx.op)
        acc = accum_item_hash(acc, lx.negated) 

        x.hash_val = acc

    return x.hash_val    


@njit(_Py_hash_t(CREObjType),cache=True)
def conds_hash(x):
    if(x.hash_val == 0):
        cx = _cast_structref(ConditionsType, x)

        acc = _PyHASH_XXPRIME_5
        for conjuct in cx.dnf:
            acc = accum_item_hash(acc, _PyHASH_XXPRIME_2)            
            for lit in conjuct:
                acc = accum_item_hash(acc, literal_hash(lit))
        x.hash_val = acc

    return x.hash_val    

@njit(_Py_hash_t(CREObjType),cache=True)
def fact_hash(x):
    ''' based roughly on _tuple_hash from numba.cpython.hashing'''
    if(x.hash_val == 0):
        acc = _PyHASH_XXPRIME_5
        tl = x.num_chr_mbrs
        for t_id, m_id, data_ptr in cre_obj_iter_t_id_item_ptrs(x):
            if(t_id == T_ID_UNDEFINED): 
                t_id,_, _ = decode_idrec(_struct_from_ptr(CREObjType,_load_ptr(i8, data_ptr)).idrec)
            if(t_id == T_ID_TUPLE_FACT):
                raise Exception()
                # acc = accum_item_hash(acc, tuple_fact_hash(x))
            elif(t_id == T_ID_FACT):
                # hash on the pointer 
                ptr = _load_ptr(i8, data_ptr)
                acc = accum_item_hash(acc, _PyHASH_XXPRIME_5 if(ptr == 0) else ptr)
            else:
                # hash primitives
                acc = accum_item_hash(acc, hash_from_t_id_ptr(t_id, data_ptr))

        acc = accum_item_hash(acc,tl)
        x.hash_val = acc


    return x.hash_val

@njit(_Py_hash_t(CREObjType),cache=True)
def tuple_fact_hash(x):
    ''' based roughly on _tuple_hash from numba.cpython.hashing'''
    if(x.hash_val == 0):
        stack_buffer = None
        p = _cast_structref(TupleFact, x)
        stack_head = -1
        is_done = False

        acc = _PyHASH_XXPRIME_5
        while(not is_done):
            tl = p.num_chr_mbrs
            for t_id, m_id, data_ptr in cre_obj_iter_t_id_item_ptrs(p):
                # print(":: t_id", t_id, data_ptr)
                if(t_id == T_ID_UNDEFINED): 
                    t_id,_, _ = decode_idrec(_struct_from_ptr(CREObjType,_load_ptr(i8, data_ptr)).idrec)
                if(t_id == T_ID_TUPLE_FACT):
                    # print("IS PRED")
                    # Use an inline stack instead of recrusion because numba can't cache recursion
                    stack_head += 1
                    if(stack_buffer is None):
                        stack_buffer = np.empty(2, dtype=np.int64)
                    if(stack_head >= len(stack_buffer)):
                        new_stack_buffer  = np.empty((stack_head+1)*2, dtype=np.int64)
                        new_stack_buffer[:len(stack_buffer)] = stack_buffer
                        stack_buffer = new_stack_buffer
                    stack_buffer[stack_head] = _load_ptr(i8, data_ptr)

                elif(t_id <= T_ID_STR):
                    acc = accum_item_hash(acc, hash_from_t_id_ptr(t_id, data_ptr))
                    
                else:
                    # Kind of a heavy handed way to do this... but lack
                    #  of recrusion makes it the best option                    
                    mbr = _struct_from_ptr(CREObjType,_load_ptr(i8,data_ptr))
                    # print("::  " ,decode_idrec(x.idrec), x)
                    if(t_id==T_ID_VAR):
                        # print("----- VAR -----")
                        mbr_hash = var_hash(mbr)
                    elif(t_id==T_ID_OP):
                        # print("----- OP -----")
                        mbr_hash = op_hash(mbr)
                    elif(t_id==T_ID_LITERAL):
                        mbr_hash = literal_hash(mbr)
                    elif(t_id==T_ID_CONDITIONS):
                        mbr_hash = conds_hash(mbr)
                    else:
                        mbr_hash = fact_hash(mbr)
                    acc = accum_item_hash(acc,  mbr_hash )
                    # print(t_id, "<<", acc,hash_from_t_id_ptr(t_id, data_ptr))

            acc = accum_item_hash(acc,tl)

            if(stack_head > -1):
                p = _struct_from_ptr(TupleFact, stack_buffer[stack_head]);
                stack_head -=1;
            else:
                is_done = True
        x.hash_val = acc

    return x.hash_val

# @njit(_Py_hash_t(CREObjType))
# def cre_obj_hash(x):
#     t_id,_,_ = decode_idrec(x.idrec)

#     # if(t_id <= T_ID_STR):
#     #     hsh = non_tuple_fact_hash(x)
#     # elif(t_id == T_ID_TUPLE_FACT):
#     hsh = tuple_fact_hash(x)

#     # return i8()
#     if hsh == _Py_uhash_t(-1):
#         return process_return(1546275796)

#     return process_return(hsh)

@overload(hash)
@overload_method(CREObjTypeClass, '__hash__')
def _cre_obj_hash(x):
    if(isinstance(x,CREObjTypeClass)):
        def impl(x):
            t_id,_, _ = decode_idrec(x.idrec)
            if(t_id==T_ID_TUPLE_FACT):
                return tuple_fact_hash(x) 
            elif(t_id==T_ID_VAR):
                return var_hash(x) 
            elif(t_id==T_ID_OP):
                return op_hash(x) 
            elif(t_id==T_ID_LITERAL):
                return literal_hash(x)
            elif(t_id==T_ID_CONDITIONS):
                return conds_hash(x)
            else:
                return fact_hash(x)            
        return impl




#### __str__ ### 


# def __cre_obj_hash(x):
#     if(isinstance(x,CREObjTypeClass)):
#         def impl(x):
#             return cre_obj_hash(x)
#         return impl


