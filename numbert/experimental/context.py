from numba import types, njit, guvectorize,vectorize,prange
from numba.experimental import jitclass
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple 
from numba.cpython.unicode import  _set_code_point
from numbert.utils import cache_safe_exec
from numbert.core import TYPE_ALIASES, REGISTERED_TYPES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map
from numbert.gensource import assert_gen_source
from numbert.caching import unique_hash, source_to_cache, import_from_cached, source_in_cache
from collections import namedtuple
import numpy as np
import timeit
import itertools
import types as pytypes
import sys
import __main__
import os


# numba_type_ids = {k:i  for i,k in enumerate(numba_type_map)}


Dict_Unicode_to_Enums = DictType(unicode_type,i8[:])
Dict_Unicode_to_i8 = DictType(unicode_type,i8)
Dict_Unicode_to_Flags = DictType(unicode_type,u1[:])

@njit(cache=True)
def new_kb_context():
	string_enums = Dict.empty(unicode_type,i8)
	number_enums = Dict.empty(f8,i8)
	string_backmap = Dict.empty(i8,unicode_type)
	number_backmap = Dict.empty(i8,f8)
	enum_counter = np.array(0)
	attr_inds_by_type = Dict.empty(unicode_type,Dict_Unicode_to_i8)
	# nominal_maps = Dict.empty(unicode_type,u1[:])
	spec_flags = Dict.empty(unicode_type,Dict_Unicode_to_Flags)
	return (string_enums, number_enums, string_backmap, number_backmap,
		enum_counter, attr_inds_by_type, spec_flags)


class KnowledgeBaseContext(object):
	_contexts = {}
	
	@classproperty
	def default_context():
		df_c = os.environ.get("NUMBERT_DEFAULT_CONTEXT")
		return df_c if df_c else "numbert"

	@classmethod
	def init(cls, name):
		if(not name in cls._contexts):
			cls._contexts[name] = cls()

	@classmethod
	def get_context(cls, name=None):
		if(name is None): return cls.default_context
		if(isinstance(name,KnowledgeBaseContext)): return name
		if(name not in cls._contexts): cls.init(name)
		return cls._contexts[name]

	@classmethod
	def set_default(cls, name):
		_default_context = cls.get_context(name)

	def __init__(self):
		self.registered_specs = {}
		self.jitstructs = {}
		self.nb_data = new_kb_context()

		# for x in ["<#ANY>",'','?sel']:
		# 	self.enumerize_value(x)

	def register_specification(self, name, spec):
		spec = self._standardize_spec(spec)
		if(name in self.registered_specs):
			assert self.registered_specs[name] == spec, \
			"Specification redefinition not permitted. Attempted on %r" % name
		else:
			self.registered_specs[name] = spec
			self._assert_flags(name,spec)
			self._update_attr_inds(name,spec)
			jitstruct = self.jitstruct_from_spec(name,spec)
			self.jitstructs[name] = jitstruct

			REGISTERED_TYPES[name] = jitstruct.numba_type
			TYPE_ALIASES[name] = jitstruct.__name__
			JITSTRUCTS[name] = jitstruct
	def jitstruct_from_spec(self,name,spec,ind="   "):
		
		#For the purposes of autogenerating code we need a clean alphanumeric name 
		name = "".join(x for x in name if x.isalnum())

		#Unstandardize to use types only. Probably don't need tags for source gen.
		spec = {attr:x['type'] for attr,x in spec.items()}

		hash_code = unique_hash([name,spec])
		assert_gen_source(name, hash_code, spec=spec, custom_type=True)

		out = import_from_cached(name,hash_code,[
			'{}_get_enumerized'.format(name),
			'{}_pack_from_numpy'.format(name),
			'{}'.format(name),
			'NB_{}'.format(name),
			'{}_enumerize_nb_objs'.format(name)
			]).values()

		get_enumerized, pack_from_numpy, nt, nb_nt, enumerize_nb_objs = tuple(out)

		def py_get_enumerized(_self,assert_maps=True):
			return get_enumerized(_self,
								   string_enums=self.string_enums,
								   number_enums=self.number_enums,
								   string_backmap=self.string_backmap,
								   number_backmap=self.number_backmap,
								   enum_counter=self.enum_counter,
								   assert_maps=assert_maps)
		nt.get_enumerized = py_get_enumerized#pytypes.MethodType(_get_enumerized, self) 
		nt._get_enumerized = get_enumerized#pytypes.MethodType(_get_enumerized, self) 
		nt.pack_from_numpy = pack_from_numpy
		nt.enumerize_nb_objs = enumerize_nb_objs
		nt.numba_type = nb_nt
		nt.hash = hash_code

		return nt

	def _standardize_spec(self,spec):
		out = {}
		# print("prestandardize")
		# print(spec)
		for attr,v in spec.items():
			if(isinstance(v,str)):
				typ, flags = v.lower(), []
			elif(isinstance(v,dict)):
				assert "type" in v, "Attribute specifications must have 'type' property, got %s." % v
				typ = v['type'].lower()
				flags = [x.lower() for x in v.get('flags',[])]
			else:
				raise ValueError("Spec attribute %r = %r is not valid type with type %s." % (attr,v,type(v)))

			#Strings are always nominal
			if(typ == 'string' and ('nominal' not in flags)): flags.append('nominal')

			out[attr] = {"type": typ, "flags" : flags}
		# print("poaststandardize")
		# print(out)
		return out

	def _register_flag(self,flag):
		d = self.spec_flags[flag] = Dict.empty(unicode_type,u1[:])
		for name,spec in self.registered_specs.items():
			d[name] = np.array([flag in x['flags'] for attr,x in spec.items()], dtype=np.uint8)


	def _assert_flags(self,name,spec):
		for flag in itertools.chain(*[x['flags'] for atrr,x in spec.items()]):
			if flag not in self.spec_flags:
				self._register_flag(flag)
		for flag, d in self.spec_flags.items():
			d[name] = np.array([flag in x['flags'] for attr,x in spec.items()], dtype=np.uint8)

	def _update_attr_inds(self,name,spec):
		d = Dict.empty(unicode_type,i8)
		for i,attr in enumerate(spec.keys()):
			d[attr] = i
		self.attr_inds_by_type[name] = d

KnowledgeBaseContext._default_context = KnowledgeBaseContext.get_context('numbert')

def register_specification(name, spec, context=None):
	KnowledgeBaseContext.get_context(context).register_specification(name,spec)

def register_specifications(self, specs, context=None):
	for name, spec in specs.items():
		register_specification(name,spec,context=context)


class _BaseContextful(object):
	def __init__(self, context):

		#Context stuff
		self.context = KnowledgeBaseContext.get_context(context)
		self.string_enums, self.number_enums, self.string_backmap, self.number_backmap, \
		self.enum_counter, self.attr_inds_by_type, self.spec_flags = self.context.nb_data
