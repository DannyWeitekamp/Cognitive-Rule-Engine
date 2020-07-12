
from numbert.core import numba_type_map

LOOPLIFT_UNJITABLES = True
UID_START = 1


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
	# 	body += ind +"if(assert_maps):\n"
	# 	for i,k in enumerate(spec.keys()):
	# 		if spec[k] == 'string':
	# 			body += ind*2 +"_assert_map({}, string_enums, string_backmap, enum_counter)\n".format(k)
	# 			# body += ind +"enumerized[{}] = string_enums[{}]\n".format(i,k)
	if(strings != ""):
	# 	for i,k in enumerate(spec.keys()):
	# 		if spec[k] == 'string':
	# 			body += ind +"enumerized[{}] = string_enums[{}]\n".format(i,k)
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
	source = header + body+("\n"*10)
	return source


def gen_source_tuple_defs(name,spec,ind='   '):
	
	tuple_defs = "{} = namedtuple('{}', {}, module=__name__)\n".format(name,name,["%s"%k for k in spec.keys() if k != 'type'])
	sv = list(spec.values())
	if(len(set(sv))==1):
		tuple_defs += "NB_{}_NamedTuple = NamedUniTuple({},{},{})\n".format(name,str(numba_type_map[sv[0]]),len(sv),name)
	else:
		typ_str = ", ".join([str(numba_type_map[x]) for x in spec.values()])
		tuple_defs += "NB_{}_NamedTuple = NamedTuple(({}),{})\n".format(name,typ_str,name)
	# tuple_defs += "{} = NB_{}_NamedTuple.instance_class\n".format(name,name)
	return tuple_defs


def gen_source_pack_from_numpy(name,spec,ind='   '):
	header = "@njit(cache=True,fastmath=True,nogil=True)\n"
	header += "def {}_pack_from_numpy(inp,mlens):\n".format(name)

	cast_map = {"string":"charseq_to_str(x.{},mlens[{lens}])", 'number': 'float(x.{})'}

	body = ind + "out = Dict.empty(unicode_type,NB_{}_NamedTuple)\n".format(name)
	# body = ""
	body += ind + "for i in range(inp.shape[0]):\n"
	body += ind*2 + "x = inp[i]\n"
	body += ind*2 + "__name__ = charseq_to_str(x.__name__,mlens[0])\n"
	for i,(attr, typ) in enumerate(spec.items()):
		body += ind*2 + ("_{} = " + cast_map[typ]+ "\n").format(attr,attr,lens=i+1)
	body += ind*2 +"out[__name__] = {}({})\n".format(name,", ".join(["_%s"%x for x in spec.keys()]))
	body += ind + "return out\n"

	source = header + body #+("\n"*10)
	return source



def gen_source_broadcast_forward(op, condition_func, nopython):
	_ = "    "
	f_name = op.__name__+"_forward"
	if(nopython):
		header = '@jit(nogil=True, fastmath=True, cache=True) \n'
	elif(LOOPLIFT_UNJITABLES and len(op.muted_exceptions) == 0):
		header = '@jit(fastmath=True, looplift=True, forceobj=True) \n'
	else:
		header = ""
	func_def =	'def {}({}): \n' #+ \

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
	if(len(op.right_commutes) > 0 or condition_func != None):
		curr_indent += 1
		conds = []

		if(len(op.right_commutes) > 0):
			for i_a, i_bs in op.right_commutes.items():
				conds.append("i{} >= i{}".format(i_a,i_bs[-1]))
		if(condition_func != None):
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