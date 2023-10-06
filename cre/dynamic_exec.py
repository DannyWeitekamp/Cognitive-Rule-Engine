import numpy as np
from numba import f8, u1, u2, i8, u8, types, njit, generated_jit
from numba.types import FunctionType, unicode_type, ListType, boolean
from numba.typed import List
from numba.extending import  overload, overload_method
from cre.utils import cast, _load_ptr, _raw_ptr_from_struct, _raw_ptr_from_struct_incref, CastFriendlyMixin, decode_idrec, _func_from_address, _incref_structref, _struct_get_data_ptr, _sizeof_type, _decref_structref
from cre.structref import define_structref
from cre.obj import CREObjTypeClass, CREObjType, member_info_type, _iter_mbr_infos, OBJECT_MBR_ID, PRIMITIVE_MBR_ID
from numba.core.datamodel import default_manager, models
from numba.experimental.structref import define_attributes, StructRefProxy, new, define_boxing
import operator
from cre.core import DEFAULT_T_ID_TYPES, T_ID_CONDITIONS, T_ID_LITERAL, T_ID_FUNC, T_ID_FACT, T_ID_VAR, T_ID_UNDEFINED, T_ID_BOOL, T_ID_INT, T_ID_FLOAT, T_ID_STR, T_ID_TUPLE_FACT, T_ID_CRE_OBJ
# from cre.primitive import BooleanPrimitiveType, IntegerPrimitiveType, FloatPrimitiveType, StringPrimitiveType
from cre.tuple_fact import TupleFact
from cre.var import VarType
# from cre.op import GenericOpType
from cre.func import CREFuncType, ARGINFO_VAR, ARGINFO_FUNC, ARGINFO_CONST
from cre.fact import BaseFact
from cre.conditions import LiteralType, ConditionsType
from numba.cpython.hashing import _Py_hash_t, _Py_uhash_t, _PyHASH_XXROTATE, _PyHASH_XXPRIME_1, _PyHASH_XXPRIME_2, _PyHASH_XXPRIME_5, process_return
from cre.hashing import accum_item_hash, unicode_hash_noseed



    
N_DEFAULT_TYPES = len(DEFAULT_T_ID_TYPES)


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

    va = cast(a, VarType)
    vb = cast(b, VarType)

    if(va.base_ptr != vb.base_ptr): return False

    if(len(va.deref_infos) != len(vb.deref_infos)): return False
    for deref_info_a, deref_info_b in zip(va.deref_infos, vb.deref_infos):
        if(deref_info_a.offset != deref_info_b.offset): return False

    
    return True

@njit(boolean(CREObjType, CREObjType),cache=True)
def cre_func_eq(a, b):
    t_id_b,_, _ = decode_idrec(b.idrec)
    if(t_id_b != T_ID_FUNC): return False

    oa = cast(a, CREFuncType)
    ob = cast(b, CREFuncType)

    stack = List()
    stack.append((oa,ob))

    while(len(stack) > 0):
        oa,ob = stack.pop()
        if(oa.call_heads_addr != ob.call_heads_addr): return False
        if(oa.n_args != ob.n_args): return False
        if(len(oa.head_infos) != len(ob.head_infos)): return False

        infs_a = oa.root_arg_infos
        infs_b = ob.root_arg_infos

        for inf_a, inf_b in zip(infs_a, infs_b):

            if(inf_a.type != inf_b.type): return False
            if(inf_a.t_id != inf_b.t_id): return False
            if(inf_a.type == ARGINFO_VAR):
                va = cast(inf_a.ptr, VarType)
                vb = cast(inf_b.ptr, VarType)
                if(not var_eq(va,vb)): return False
            elif(inf_a.type == ARGINFO_CONST):
                is_eq = eq_from_t_id_ptr(inf_a.t_id, 
                    inf_a.ptr, inf_b.ptr)
                if(not is_eq): return False
            elif(inf_a.type == ARGINFO_FUNC):
                _oa = cast(inf_a.ptr, CREFuncType)
                _ob = cast(inf_b.ptr, CREFuncType)
                stack.append((_oa,_ob))

    return True

@njit(boolean(CREObjType, CREObjType),cache=True)
def literal_eq(a, b):
    t_id_b,_, _ = decode_idrec(b.idrec)
    if(t_id_b != T_ID_LITERAL): return False

    la = cast(a, LiteralType)
    lb = cast(b, LiteralType)

    if(not cre_func_eq(la.op, lb.op)): return False
    if(la.negated != lb.negated): return False
    
    return True

@njit(boolean(CREObjType, CREObjType),cache=True)
def conds_eq(a,b):
    t_id_b,_, _ = decode_idrec(b.idrec)
    if(t_id_b != T_ID_CONDITIONS): return False
    xa = cast(a, ConditionsType)
    xb = cast(b, ConditionsType)

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

    if(cast(a, i8) == cast(b, i8)):
        return True

    if(tla != tlb): return False

    for (info_a), (info_b) in zip(_iter_mbr_infos(a),_iter_mbr_infos(b)):
        t_id_a, m_id_a, data_ptr_a = info_a
        t_id_b, m_id_b, data_ptr_b = info_b
        if(t_id_a == T_ID_UNDEFINED or t_id_a == T_ID_CRE_OBJ): 
            t_id_a,_, _ = decode_idrec(cast(_load_ptr(i8, data_ptr_a), CREObjType).idrec)
        if(t_id_b == T_ID_UNDEFINED or t_id_b == T_ID_CRE_OBJ): 
            t_id_b,_, _ = decode_idrec(cast(_load_ptr(i8, data_ptr_b), CREObjType).idrec)

        if(t_id_a != t_id_b): return False

        if(t_id_a == T_ID_TUPLE_FACT):
            raise Exception("TupleFact Members of Fact Not Implemented.")
        elif(t_id_a == T_ID_FACT or t_id_a >= N_DEFAULT_TYPES):
            # Don't check fact members 
            pass
            # if(not _load_ptr(i8, data_ptr_a) == _load_ptr(i8, data_ptr_b)): return False
        else:
            mbr_a = cast(_load_ptr(i8,data_ptr_a), CREObjType)
            mbr_b = cast(_load_ptr(i8,data_ptr_b), CREObjType)
            if(t_id_a==T_ID_VAR):
                if(not var_eq(mbr_a,mbr_b)): return False
            elif(t_id_a==T_ID_FUNC):
                if(not cre_func_eq(mbr_a,mbr_b)): return False
            elif(t_id_a==T_ID_LITERAL):
                if(not literal_eq(mbr_a,mbr_b)): return False
            elif(t_id_a==T_ID_CONDITIONS):
                if(not conds_eq(mbr_a,mbr_b)): return False
            elif(m_id_a == PRIMITIVE_MBR_ID):
                if(not eq_from_t_id_ptr(t_id_a, data_ptr_a, data_ptr_b)): return False
            else:
                # Skip any fact types
                pass
                # print("FACT EQ", mbr_a, mbr_b, fact_eq(mbr_a, mbr_b))
                # if(not fact_eq(mbr_a, mbr_b)): return False
    
    return True


@njit(boolean(CREObjType, CREObjType),cache=True)
def tuple_fact_eq(a, b):
    ''' based roughly on _tuple_hash from numba.cpython.hashing'''
    t_id_b,_, _ = decode_idrec(b.idrec)
    if(t_id_b != T_ID_TUPLE_FACT): return False

    stack_buffer = None
    tf_a = cast(a, TupleFact)
    tf_b = cast(b, TupleFact)
    stack_head = -1
    is_done = False

    # acc = _PyHASH_XXPRIME_5
    while(not is_done):
        tla = tf_a.num_chr_mbrs
        tlb = tf_b.num_chr_mbrs

        if(tla != tlb): return False

        for (info_a), (info_b) in zip(_iter_mbr_infos(tf_a),_iter_mbr_infos(tf_b)):
            t_id_a, m_id_a, data_ptr_a = info_a
            t_id_b, m_id_b, data_ptr_b = info_b

            if(t_id_a == T_ID_UNDEFINED): 
                t_id_a,_, _ = decode_idrec(cast(_load_ptr(i8, data_ptr_a), CREObjType).idrec)
            if(t_id_b == T_ID_UNDEFINED): 
                t_id_b,_, _ = decode_idrec(cast(_load_ptr(i8, data_ptr_b), CREObjType).idrec)


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
                if(not eq_from_t_id_ptr(t_id_a, data_ptr_a, data_ptr_b)):
                    return False
            else:
                # Kind of a heavy handed way to do this... but lack
                #  of recrusion makes it the best option                    
                mbr_a = cast(_load_ptr(i8,data_ptr_a), CREObjType)
                mbr_b = cast(_load_ptr(i8,data_ptr_b), CREObjType)
                if(t_id_a==T_ID_VAR):
                    if(not var_eq(mbr_a,mbr_b)): return False
                elif(t_id_a==T_ID_FUNC):
                    if(not cre_func_eq(mbr_a,mbr_b)): return False
                elif(t_id_a==T_ID_LITERAL):
                    if(not literal_eq(mbr_a,mbr_b)): return False
                elif(t_id_a==T_ID_CONDITIONS):
                    if(not conds_eq(mbr_a,mbr_b)): return False
                else:
                    # print("FACT EQ", mbr_a, mbr_b, fact_eq(mbr_a, mbr_b))
                    if(not fact_eq(mbr_a, mbr_b)): return False

                

        if(stack_head > -1):
            tf_a = cast(stack_buffer[stack_head,0], TupleFact);
            tf_b = cast(stack_buffer[stack_head,1], TupleFact);
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
            elif(t_id==T_ID_FUNC):
                return cre_func_eq(a,b)
            elif(t_id==T_ID_LITERAL):
                return literal_eq(a,b)
            elif(t_id==T_ID_CONDITIONS):
                return conds_eq(a,b)
            else:
                return fact_eq(a,b)
            
        return impl

@njit(boolean(CREObjType,CREObjType), cache=True)
def cre_obj_eq(a,b):
    return a == b


### __hash___ ###


# print(_PyHASH_XXPRIME_1, _PyHASH_XXPRIME_2, _PyHASH_XXPRIME_5)

@njit(_Py_uhash_t(u2,i8), cache=True)
def hash_from_t_id_ptr(t_id, data_ptr):
    if(t_id == T_ID_BOOL):
        return hash(_load_ptr(boolean, data_ptr))
    elif(t_id == T_ID_INT):
        return hash(_load_ptr(i8, data_ptr))
    elif(t_id == T_ID_FLOAT):
        return hash(_load_ptr(f8, data_ptr))
    elif(t_id == T_ID_STR):
        return unicode_hash_noseed(_load_ptr(unicode_type, data_ptr))
    # print(t_id, "BAD T_ID")
    return u8(0)













@njit(_Py_hash_t(CREObjType),cache=True)
def var_hash(x):
    ''' based roughly on _tuple_hash from numba.cpython.hashing'''
    # while(not is_done):
    if(x.hash_val == 0):
        vx = cast(x, VarType)

        acc = _PyHASH_XXPRIME_5
        acc = accum_item_hash(acc,  vx.base_ptr) 
        acc = accum_item_hash(acc, len(vx.deref_infos))
        
        for deref_info in vx.deref_infos:
            acc = accum_item_hash(acc, deref_info.offset) 
        x.hash_val = _Py_hash_t(acc)

    
    return x.hash_val


@njit(_Py_hash_t(CREObjType),cache=True)
def cre_func_hash(x):
    if(x.hash_val == 0):
        ox = cast(x, CREFuncType)
        stack = List()
        stack.append(ox)

        acc = _PyHASH_XXPRIME_5

        while(len(stack) > 0):
            ox = stack.pop()
            acc = accum_item_hash(acc, ox.call_heads_addr) 
            acc = accum_item_hash(acc, ox.n_args)
            
            for inf in ox.root_arg_infos:
                if(inf.type == ARGINFO_VAR):
                    _var = cast(inf.ptr, VarType)
                    acc = accum_item_hash(acc, var_hash(_var)) 
                elif(inf.type == ARGINFO_CONST):
                    const_hash = hash_from_t_id_ptr(inf.t_id, inf.ptr)
                    acc = accum_item_hash(acc, const_hash) 
                elif(inf.type == ARGINFO_FUNC):
                    _ox = cast(inf.ptr, CREFuncType)
                    stack.append(_ox)
        x.hash_val = acc
    return x.hash_val



@njit(_Py_hash_t(CREObjType),cache=True)
def literal_hash(x):
    if(x.hash_val == 0):
        lx = cast(x, LiteralType)

        acc = cre_func_hash(lx.op)
        acc = accum_item_hash(acc, lx.negated) 

        x.hash_val = acc

    return x.hash_val    


@njit(_Py_hash_t(CREObjType),cache=True)
def conds_hash(x):
    if(x.hash_val == 0):
        cx = cast(x, ConditionsType)

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
        for t_id, m_id, data_ptr in _iter_mbr_infos(x):
            if(t_id == T_ID_UNDEFINED or t_id == T_ID_CRE_OBJ): 
                t_id,_, _ = decode_idrec(cast(_load_ptr(i8, data_ptr), CREObjType).idrec)

            if(t_id == T_ID_TUPLE_FACT):
                raise Exception()
                # acc = accum_item_hash(acc, tuple_fact_hash(x))
            elif(t_id == T_ID_VAR):
                # hash on the pointer 
                ptr = _load_ptr(i8, data_ptr)
                if(ptr != 0):
                    mbr_hash = var_hash(cast(ptr, VarType)) 
                else:
                    mbr_hash = _Py_hash_t(_PyHASH_XXPRIME_5)
                acc = accum_item_hash(acc, mbr_hash)
            elif(m_id == PRIMITIVE_MBR_ID):
                # hash primitives
                acc = accum_item_hash(acc, hash_from_t_id_ptr(t_id, data_ptr))
            else:
                # hash on the pointer 
                # ptr = _load_ptr(i8, data_ptr)
                # acc = accum_item_hash(acc, _PyHASH_XXPRIME_5 if(ptr == 0) else ptr)
                pass

        acc = accum_item_hash(acc,tl)
        x.hash_val = acc


    return x.hash_val

@njit(_Py_hash_t(CREObjType),cache=True)
def tuple_fact_hash(x):
    ''' based roughly on _tuple_hash from numba.cpython.hashing'''
    if(x.hash_val == 0):
        stack_buffer = None
        p = cast(x, TupleFact)
        stack_head = -1
        is_done = False

        acc = _PyHASH_XXPRIME_5
        while(not is_done):
            tl = p.num_chr_mbrs
            for t_id, m_id, data_ptr in _iter_mbr_infos(p):
                # print(":: t_id", t_id, data_ptr)
                if(t_id == T_ID_UNDEFINED): 
                    t_id,_, _ = decode_idrec(cast(_load_ptr(i8, data_ptr), CREObjType).idrec)
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
                    mbr = cast(_load_ptr(i8,data_ptr), CREObjType)
                    # print("::  " ,decode_idrec(x.idrec), x)
                    if(t_id==T_ID_VAR):
                        # print("----- VAR -----")
                        mbr_hash = var_hash(mbr)
                    elif(t_id==T_ID_FUNC):
                        # print("----- OP -----")
                        mbr_hash = cre_func_hash(mbr)
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
                p = cast(stack_buffer[stack_head], TupleFact);
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

@njit(_Py_hash_t(CREObjType), locals={"hsh" : _Py_hash_t}, cache=True)
def cre_obj_hash(x):
    t_id,_, _ = decode_idrec(x.idrec)
    hsh = 0

    if(t_id==T_ID_TUPLE_FACT):
        hsh = tuple_fact_hash(x) 
    elif(t_id==T_ID_VAR):
        hsh = var_hash(x) 
    elif(t_id==T_ID_FUNC):
        hsh = cre_func_hash(x) 
    elif(t_id==T_ID_LITERAL):
        hsh = literal_hash(x)
    elif(t_id==T_ID_CONDITIONS):
        hsh = conds_hash(x)
    else:
        hsh = fact_hash(x)
    return hsh



@overload(hash)
@overload_method(CREObjTypeClass, '__hash__')
@overload_method(CREObjType, '__hash__') # Not sure why but is necessary
def _cre_obj_hash_overload(x):
    print("------IMPL-----", x, isinstance(x, CREObjTypeClass))
    if(isinstance(x,CREObjTypeClass)):
        def impl(x):
            return cre_obj_hash(x)
        return impl


#### __str__ ### 


# def __cre_obj_hash(x):
#     if(isinstance(x,CREObjTypeClass)):
#         def impl(x):
#             return cre_obj_hash(x)
#         return impl


