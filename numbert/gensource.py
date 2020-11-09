
from numbert.core import numba_type_map
from numbert.caching import source_in_cache, source_to_cache

LOOPLIFT_UNJITABLES = True
UID_START = 1


def gen_source_standard_imports():
    imports = "import numpy as np\n"
    imports += "from collections import namedtuple\n"
    imports += "from numba import jit,njit\n"
    imports += "from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16\n"
    imports += "from numba.typed import List, Dict\n"
    imports += "from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple, Tuple\n"
    imports += "from numbert.numbalizer import _assert_map\n"
    return imports

def gen_source_get_enumerized(name,spec,ind='   '):
    arg_str = ind*3 + "string_enums, number_enums,\n"
    arg_str += ind*3 + "string_backmap,number_backmap,\n"
    arg_str += ind*3 + "enum_counter"

    # attr_str = ind*3 + ", ".join(spec.keys()) 

    header = "@njit(cache=True,fastmath=True,nogil=True)\n"
    # header += "def {}_get_enumerized(\n{},\n{},assert_maps=True):\n".format(name,attr_str,arg_str)
    header += "def {}_get_enumerized(x,\n{},assert_maps=True):\n".format(name,arg_str)

    strings = ", ".join(["(x.{},{})".format(k,i) for i,k in enumerate(spec.keys()) if spec[k] == 'string'])
    numbers = ", ".join(["(x.{},{})".format(k,i) for i,k in enumerate(spec.keys()) if spec[k] == 'number'])

    body = ind + "enumerized = np.empty(({},),np.int64)\n".format(len(spec.keys()))
    # if(strings != ""):
    #   body += ind +"if(assert_maps):\n"
    #   for i,k in enumerate(spec.keys()):
    #       if spec[k] == 'string':
    #           body += ind*2 +"_assert_map({}, string_enums, string_backmap, enum_counter)\n".format(k)
    #           # body += ind +"enumerized[{}] = string_enums[{}]\n".format(i,k)
    if(strings != ""):
    #   for i,k in enumerate(spec.keys()):
    #       if spec[k] == 'string':
    #           body += ind +"enumerized[{}] = string_enums[{}]\n".format(i,k)
        body += ind + "for v,i in [{strings}]:\n".format(strings=strings)
        body += ind*2 + "if(assert_maps): _assert_map(v, string_enums, string_backmap, enum_counter)\n"
        body += ind*2 + "enumerized[i] = string_enums[v]\n"
    if(numbers != ""):  
        body += ind + "for v,i in [{numbers}]:\n".format(numbers=numbers)
        body += ind*2 + "if(assert_maps): _assert_map(v, number_enums, number_backmap, enum_counter)\n"
        body += ind*2 + "enumerized[i] = number_enums[v]\n"
    body += ind + "return enumerized\n\n"

    

    source = header + body#  +c
    return source

def gen_source_enumerize_nb_objs(name,spec,ind='   '):

    arg_str = "string_enums, number_enums,\n"
    arg_str += ind*3 + "string_backmap,number_backmap,\n"
    arg_str += ind*3 + "enum_counter"

    header = "@njit(cache=True,fastmath=True,nogil=True)\n"
    # header += "def {}_get_enumerized(\n{},\n{},assert_maps=True):\n".format(name,attr_str,arg_str)
    header += "def {}_enumerize_nb_objs(inp,out,{}):\n".format(name,arg_str)
    body = ind + 'for k,v in inp.items():\n'
    body += ind*2 + 'out[k] = {}_get_enumerized(v,{})\n\n'.format(name,arg_str)
    source = header + body#+("\n"*10)
    return source


def gen_source_tuple_defs(name,spec,ind='   '):
    
    tuple_defs = "{} = namedtuple('{}', {}, module=__name__)\n".format(name,name,["%s"%k for k in spec.keys() if k != 'type'])
    sv = list(spec.values())
    if(len(set(sv))==1):
        tuple_defs += "NB_{} = NamedUniTuple({},{},{})\n".format(name,str(numba_type_map[sv[0]]),len(sv),name)
    else:
        typ_str = ", ".join([str(numba_type_map[x]) for x in spec.values()])
        tuple_defs += "NB_{} = NamedTuple(({}),{})\n".format(name,typ_str,name)
    # tuple_defs += "{} = NB_{}_NamedTuple.instance_class\n".format(name,name)
    return tuple_defs + "\n"


def gen_source_pack_from_numpy(name,spec,ind='   '):
    header = "@njit(cache=True,fastmath=True,nogil=True)\n"
    header += "def {}_pack_from_numpy(inp,mlens):\n".format(name)

    cast_map = {"string":"str(x.{})", 'number': 'float(x.{})'}

    body = ind + "out = Dict.empty(unicode_type,NB_{})\n".format(name)
    # body = ""
    body += ind + "for i in range(inp.shape[0]):\n"
    body += ind*2 + "x = inp[i]\n"
    body += ind*2 + "__name__ = str(x.__name__)\n"
    for i,(attr, typ) in enumerate(spec.items()):
        body += ind*2 + ("_{} = " + cast_map[typ]+ "\n").format(attr,attr)
    body += ind*2 +"out[__name__] = {}({})\n".format(name,", ".join(["_%s"%x for x in spec.keys()]))
    body += ind + "return out\n"

    source = header + body #+("\n"*10)
    return source




def gen_source_broadcast_forward(op, nopython):
    has_condition = hasattr(op,'condition')
    _ = "    "
    f_name = op.__name__+"_forward"
    if(nopython):
        header = '@jit(nogil=True, fastmath=True, cache=True) \n'
    elif(LOOPLIFT_UNJITABLES and len(op.muted_exceptions) == 0):
        header = '@jit(fastmath=True, looplift=True, forceobj=True) \n'
    else:
        header = ""
    func_def =  'def {}({}): \n' #+ \

    func_def = func_def.format(f_name,
         ",".join(["x%i"%i for i in range(len(op.u_arg_types))]) )

    defs = _+", ".join(["L%i"%i for i in range(len(op.u_arg_types))]) + " = " + \
              ", ".join(["len(x%i)"%i for i in range(len(op.u_arg_types))]) + "\n"

    defs += _+"out = np.empty((%s),dtype=np.int64)\n"%",".join(["L%s"%x for x in op.u_arg_inds])
    defs += _+"d = Dict.empty({},i8)\n".format(op.out_type)
    defs += _+"uid = {}\n".format(UID_START)
                
    loops = ""
    curr_indent = 1
    for i in range(len(op.arg_types)):
        curr_indent = i+1
        l = _*curr_indent + "for i{} in range(L{}):\n"
        l = l.format(i,op.u_arg_inds[i])
        loops += l
    

    all_indicies = ["i%s"%i for i in range(len(op.arg_types))]
    arg_terms = ["x{}[i{}]".format(op.u_arg_inds[i],i) for i in range(len(op.arg_types))]
    cond_expr = "{}\n"
    if(len(op.right_commutes) > 0 or has_condition):
        curr_indent += 1
        conds = []

        if(len(op.right_commutes) > 0):
            for i_a, i_bs in op.right_commutes.items():
                conds.append("i{} >= i{}".format(i_a,i_bs[-1]))
        if(has_condition):
            conds.append("c({})".format(",".join(arg_terms)))

        cond_expr =  _*curr_indent     + "if({}):\n".format(" and ".join(conds))
        cond_expr += "{}\n"#_*(curr_indent+1) + "{}\n"
        cond_expr += _*(curr_indent)   + "else:\n"
        cond_expr += _*(curr_indent+1) + "out[{}] =  0\n".format(",".join(all_indicies))
        # print("COMMUTES", op.right_commutes)

    # use_try = False
    try_expr = "{}"
    if(len(op.muted_exceptions) > 0):
        # use_try = True
        try_expr = _*(curr_indent+1) + "try:\n"
        try_expr += "{}\n"
        try_expr += _*(curr_indent+1) + "except ({}):\n".format(",".join([x.__name__ for x in op.muted_exceptions]))
        try_expr += _*(curr_indent+2) + "out[{}] =  0\n".format(",".join(all_indicies))
        curr_indent += 1

    
    exec_code =  _*(curr_indent+1) +"v = f({})\n".format(",".join(arg_terms))
    exec_code += _*(curr_indent+1) +"if(v not in d):\n"
    exec_code += _*(curr_indent+2) +"d[v] = uid; uid +=1;\n"
    exec_code += _*(curr_indent+1) +"out[{}] = d[v]".format(",".join(all_indicies))


    exec_code = try_expr.format(exec_code)

    cond_expr = cond_expr.format(exec_code)
    ret_expr = _+"return out, d\n"
    source = header + func_def + defs +  loops + cond_expr+ret_expr

    return source   

def gen_source_inf_hist_types(typ,hsh,custom_type=False,ind='   '):
    if(custom_type): typ = "NB_"+typ
    # s = "from ._{} import NB_{}_NamedTuple as {}\n\n".format(d_hsh,typ,typ) if d_hsh else ''
    s = "from numba.pycc import CC\n\n"
    s += "cc = CC('InfHistory_{}')\n\n".format(hsh)
    s += "record_type = Tuple([i8, i8[::1], i8[::1],\n"
    s += ind + "ListType(unicode_type),\n"
    s += ind +  "DictType({},i8)])\n".format(typ)
    s += "record_list_type = ListType(record_type)\n"
    # s += "nt = {}_InfHistory = namedtuple('{}_InfHistory',\n".format(typ,typ)
    # s += ind + "['u_vds','u_vs','dec_u_vs','records'])\n".format(typ)
    s += "hist_type  = Tuple((\n"
    s += ind + "DictType({},i8),\n".format(typ)
    if(typ == 'f8'):
        s += ind + "f8[::1],\n"
        s += ind + "f8[::1],\n"
    else:
        s += ind + "ListType({}),\n".format(typ)
        s += ind + "ListType({}),\n".format(typ)
    s += ind + "DictType(i8,record_list_type)\n"
    # s += "],nt)\n"
    s += "))\n\n"

    s += "from numbert.aot_template_funcs import declare\n"
    s += "declare = cc.export('declare',(hist_type,{}))(declare)\n\n".format(typ)

    s += "from numbert.aot_template_funcs import declare_nb_objects\n"
    s += "declare_nb_objects = cc.export('declare_nb_objects',(hist_type,DictType(unicode_type,{})))(declare_nb_objects)\n\n".format(typ)

    if(typ == 'f8'):
        s += "from numbert.aot_template_funcs import make_contiguous_f8\n"
        s += "make_contiguous = cc.export('make_contiguous',hist_type(hist_type,i8))(make_contiguous_f8)\n\n".format(typ)


    else:
        s += "from numbert.aot_template_funcs import make_contiguous\n"
        s += "make_contiguous = cc.export('make_contiguous',hist_type(hist_type,i8))(make_contiguous)\n\n".format(typ)

    # s += "from numbert.aot_template_funcs import insert_record\n"
    # s += "insert_record = cc.export('insert_record',(hist_type,i8,i8,ListType(unicode_type),i8[::1],i8[::1],DictType({},i8)))(insert_record)\n\n".format(typ)

    s += "from numbert.aot_template_funcs import backtrace_goals, HE\n"
    s += "backtrace_goals = cc.export('backtrace_goals',DictType(unicode_type,i8[:])(ListType({}),hist_type,ListType(ListType(HE)),i8,i8))(backtrace_goals)\n\n".format(typ)

    if(typ == 'f8'):
        s += "from numbert.aot_template_funcs import backtrace_selection_f8\n"
        s +=  "backtrace_selection = cc.export('backtrace_selection',DictType(unicode_type,i8[:])(i8[:],hist_type,ListType(ListType(HE)),i8,i8))(backtrace_selection_f8)\n"
    else:
        s += "from numbert.aot_template_funcs import backtrace_selection\n"
        s +=  "backtrace_selection = cc.export('backtrace_selection',DictType(unicode_type,i8[:])(i8[:],hist_type,ListType(ListType(HE)),i8,i8))(backtrace_selection)\n"

    return s


# def gen_source_backtrace_selection(typ, ind='   '):
#     s =  "@cc.export('backtrace_selection',DictType(unicode_type,i8[:])(i8[:],hist_type,ListType(ListType(HE)),i8,i8))\n"
#     s += "def backtrace_selection(sel,history,hist_elems,max_depth, max_solutions=1):\n"    
#     s += ind + "_,u_vs,_,_ = history\n"
#     if(typ == 'f8'):
#         s += ind + "goals = u_vs[sel]\n"
#     else:
#         s += ind + "goals = List()\n"
#         s += ind + "for s in sel:\n"
#         s += ind*2 + "goals.append(u_vs[s])\n"
#     s += ind + "return backtrace_goals(goals,history,hist_elems,max_depth,max_solutions=max_solutions)\n\n"
#     return s


# def backtrace_selection(sel,history,hist_elems,max_depth, max_solutions=1):
#     '''Same as backtrace_goals except takes a set of indicies into u_vs'''
#     _,u_vs,_,_ = history
#     #f8
#     goals = u_vs[sel]
#     #other
#     goals = List()
#     for s in sel:
#         goals.append(u_vs[s])
#     return backtrace_goals(goals,history,hist_elems,max_depth,max_solutions=max_solutions)


def gen_source_empty_inf_history(typ, custom_type=False, ind='   '):
    typ_v = "NB_"+typ if custom_type else typ
    header = "@cc.export('empty_inf_history',hist_type())\n"
    header += "@njit(cache=True,fastmath=True,nogil=True)\n"
    header += "def empty_inf_history():\n"

    body =  ind + "records = Dict.empty(i8,record_list_type)\n"
    body += ind + "records[0] = List.empty_list(record_type)\n"
    body += ind + "tl = List.empty_list(unicode_type);tl.append('{}');\n".format(typ)
    body += ind + "vmap = Dict.empty({},i8)\n".format(typ_v)
    #Type : (0 (i.e. no-op), _hist, shape, arg_types, vmap)
    body += ind + "records[0].append(\n" 
    body += ind*2 + "(0, np.empty((0,),dtype=np.int64),\n"
    body += ind*2 + "np.empty((0,),dtype=np.int64),\n"
    body += ind*2 + "tl,vmap))\n"
        
    body += ind + "u_vds = Dict.empty({},i8)\n".format(typ_v)
    if(typ == 'f8'):
        body += ind + "u_vs = np.empty(0)\n"
        body += ind + "dec_u_vs = np.empty(0)\n"
    else:
        body += ind + "u_vs = List.empty_list({})\n".format(typ_v)
        body += ind + "dec_u_vs = List.empty_list({})\n".format(typ_v)
    body += ind + "return (u_vds,u_vs,dec_u_vs,records)\n\n"
    return header + body

def gen_source_insert_record(typ, custom_type=False,ind='   '):
    typ_v = "NB_"+typ if custom_type else typ
    t_types = "i8, i8[::1], i8[::1], ListType(unicode_type), DictType({},i8)".format(typ_v)

    header = "@cc.export('insert_record',(hist_type,i8,{}))\n".format(t_types)
    header += "@jit(nogil=True, fastmath=True, cache=True)\n"
    header += 'def insert_record(history, depth, op_uid, btsr_flat, btsr_shape, arg_types, vmap):\n'
    body = ind + '_,_,_,records = history\n'
    # body += ind + 'btsr_flat = btsr.reshape(-1)\n'
    # body += ind + 'btsr_shape = np.array(btsr.shape,np.int64)\n'
    # body += ind + 'arg_types = List([{}])\n'.format(["'{}'".format(x) for x in op.arg_types])
    body += ind + 'r_d = records.get(depth, List.empty_list(record_type))\n'
    body += ind + 'r_d.append((op_uid, btsr_flat, btsr_shape, arg_types, vmap))\n'
    body += ind + 'records[depth] = r_d\n\n'
    return header + body


def assert_gen_source(typ, hash_code, spec=None, custom_type=False):
    if(not source_in_cache(typ,hash_code)):
        source = gen_source_standard_imports()
        if(custom_type): source += gen_source_tuple_defs(typ,spec)
        source += gen_source_inf_hist_types(typ,hash_code,custom_type=custom_type)
        # source += gen_source_backtrace_selection(typ)
        source += gen_source_empty_inf_history(typ,custom_type=custom_type)
        source += gen_source_insert_record(typ,custom_type=custom_type)
        if(custom_type):
            source += gen_source_get_enumerized(typ,spec)
            source += gen_source_enumerize_nb_objs(typ,spec)
            source += gen_source_pack_from_numpy(typ,spec)
        source_to_cache(typ,hash_code,source,True)
