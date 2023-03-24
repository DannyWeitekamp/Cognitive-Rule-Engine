from numba import types, njit, guvectorize, vectorize, prange, generated_jit
from numba.experimental import jitclass, structref
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.core.types import (DictType, ListType, unicode_type, float64, NamedTuple, NamedUniTuple, UniTuple, Array, Tuple)
from numba.core.extending import (
    infer_getattr,
    lower_getattr_generic,
    lower_setattr_generic,
    overload_method,
    intrinsic,
    overload,
    box,
    unbox,
    NativeValue
)
from numba.core.datamodel import default_manager, models
from numba.core import types, cgutils
from numba.types import ListType
from numba.core.typing import signature

# from numba.core.extending import overload

from cre.core import TYPE_ALIASES, JITSTRUCTS, py_type_map, numba_type_map, numpy_type_map, register_global_default, lines_in_type_registry, add_to_type_registry, add_type_pickle
from cre.caching import unique_hash_v, source_to_cache, import_from_cached, source_in_cache, get_cache_path
from cre.structref import gen_structref_code, define_structref
# from cre.context import cre_context
from cre.utils import (cast, struct_get_attr_offset, _obj_cast_codegen,
                       _ptr_from_struct_codegen, CastFriendlyMixin, _obj_cast_codegen,
                        PrintElapse, _struct_get_data_ptr, _ptr_from_struct_incref)
from cre.obj import CREObjTypeClass, cre_obj_field_dict, CREObjModel, CREObjType, member_info_type, CREObjProxy

from numba.core.typeconv import Conversion
import operator
from numba.core.imputils import (lower_cast)
import cloudpickle
import numpy as np


SPECIAL_SPEC_ATTRIBUTES = ["inherit_from"]

class Fact(CREObjTypeClass):
    def __init__(self, name, fields, hash_code=None):
        if(hash_code is None):
            hash_code = unique_hash_v([name,fields])
        super(Fact, self).__init__(fields)
    
        # Sets the numba type name. Using hash_code is much shorter
        #  and thus more readable than using the fields dict.        
        self.name = f'{name}_{hash_code}'

    def __str__(self):
        if(hasattr(self, '_fact_name')):
            return f'{self._fact_name}_{self._hash_code[:10]}'
        elif(hasattr(self,"_specialization_name")):
            return self._specialization_name
        else:
            return "Fact"

    __repr__ = __str__

    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

    @property
    def field_dict_keys(self):
        if(not hasattr(self, "_field_dict_keys")):
            self._field_dict_keys = list(self.field_dict.keys())
        return self._field_dict_keys

    def get_attr_deref_info(self, attr):
        from cre.context import cre_context
        from cre.utils import deref_info_type, DEREF_TYPE_ATTR, DEREF_TYPE_LIST
        priv_var_name = f"_{attr}_deref_info"
        if(not hasattr(self, priv_var_name)):
            a_id = self.get_attr_a_id(attr)
            offset = self.get_attr_offset(attr)
            head_type = self.get_attr_type(attr)
            
            head_t_id = cre_context().get_t_id(_type=head_type)
            # deref_type = DEREF_TYPE_LIST if isinstance(head_type,types.ListType) else DEREF_TYPE_ATTR

            deref_info = np.zeros((),dtype=deref_info_type)
            deref_info['type'] = DEREF_TYPE_ATTR
            deref_info['a_id'] = a_id
            deref_info['t_id'] = head_t_id
            deref_info['offset'] = offset

            setattr(self, priv_var_name, deref_info)
        return getattr(self, priv_var_name)
        
    def get_attr_type(self, attr):
        attr_type = self.spec[attr]['type']
        if(isinstance(attr_type, ListType)):
            if(isinstance(attr_type.item_type,DeferredFactRefType)):
                attr_type = types.ListType(attr_type.item_type.get())
        if(isinstance(attr_type, DeferredFactRefType)):
            attr_type = attr_type.get()
        return attr_type

    def get_attr_offset(self,attr):
        return self._attr_offsets[self.get_attr_a_id(attr)]

    def get_attr_a_id(self, attr):
        return u1(self.field_dict_keys.index(attr))

    def get_attr_from_a_id(self, a_id):
        # print(self.name,a_id, self.field_dict_keys)
        return self.field_dict_keys[a_id]

    def __call__(self, *args, **kwargs):
        ''' If a fact_type is called with types return a signature 
            otherwise use it's ctor to return a new instance'''
        if len(args) > 0 and isinstance(args[0], types.Type):
            return signature(self, *args)

        # Ensure that any lists are converted to typed lists
        args = [List(x) if(isinstance(x,list)) else x for x in args]
        kwargs = {k: (List(v) if(isinstance(v,list)) else v) for k,v in kwargs.items()}
        return self._ctor[0](*args, **kwargs)

    def filter_spec(self, flags):
        ''' Returns a filtered version of this Fact's spec which picks
            out only the subset of attributes in the spec that satisfy
            a set of flags like [('visible', 'few_valued'), ('unique_id')]
            where the list of lists or tuples of flags represents a disjunctive 
            normal statement i.e. ('visible' and 'few_valued') or 'unique_id'.
        '''

        # Ensure that flags is list of lists
        if(not isinstance(flags,(list,tuple))): flags = [flags]
        if(len(flags) == 0): return self.clean_spec
        flags = list(flags)
        if(not isinstance(flags[0], (list, tuple))): flags = [flags]

        from cre.attr_filter import attr_filter_registry
        filtered_spec = {}
        for and_flags in flags:
            flag_0 = and_flags[0]
            negated_0 = flag_0[0] == "~"
            flag_0 = flag_0[1:] if(negated_0) else flag_0
            if(flag_0 not in attr_filter_registry):
                raise ValueError(f"No filter registered under flag {flag_0!r}")
            conj_filtered_spec = {k:v for k,v in attr_filter_registry[flag_0].get_attrs(self, negated_0)}
            for i in range(1,len(and_flags)):
                flag = and_flags[i]
                negated = flag[0] == "~"
                flag = flag[1:] if negated else flag
                if(flag not in attr_filter_registry):
                    raise ValueError(f"No filter registered under flag {flag!r}")
                _fs = attr_filter_registry[flag].get_attrs(self, negated)
                conj_filtered_spec = {k : v for k,v in _fs if k in conj_filtered_spec}
            filtered_spec = {**filtered_spec, **conj_filtered_spec}
        return filtered_spec

    @property
    def clean_spec(self):
        if(not hasattr(self,'_clean_spec')):
            self._clean_spec = _clean_spec(self.spec)
        return self._clean_spec

    def __getstate__(self):
        d = self.__dict__.copy()
        if('_clean_spec' in d): del d['_clean_spec']

        # NOTE: While ops still use py_classes we need clean out 'conversions' from the spec
        if('spec' in d):

            d_spec = d['spec'].copy()
            
            for attr, attr_spec in d_spec.items():
                if('conversions' in attr_spec):
                    attr_spec_copy = attr_spec.copy()
                    attr_spec_copy['conversions'] = tuple([(attr_spec['type'], attr, typ) for typ in attr_spec_copy['conversions']])
                    d_spec[attr] = attr_spec_copy
            d['spec'] = d_spec
            # print(d)
        if('_code' in d): del d['_code']
                    
        return d

    def __setstate__(self, d):
        self.__dict__ = d


@generated_jit(cache=True)
def call_untyped_fact(self, **kwargs):

    typs = tuple(kwargs.values())
    return_sig = False
    if(isinstance(typs[0],types.TypeRef)):
        typs = tuple(x.instance_type for x in typs)
        return_sig = True    
    fact_type = self.specialize(**{k:v for k,v in zip(kwargs,typs)})

    if(return_sig):
        def impl(self,**kwargs):
            return fact_type
    else:
        def impl(self,**kwargs):
            return fact_type(**kwargs)

    return impl

from numba.core.typing.typeof import typeof
class UntypedFact(Fact):
    specializations = {}
    def __init__(self, name, fields, hash_code=None):
        super(UntypedFact, self).__init__(name, fields, hash_code)
    def __call__(self, *args, **kwargs):
        if(len(kwargs) == 0):
            return signature(self,*args)
        else:

            if(isinstance(list(kwargs.values())[0],types.Type)):
                fact_type = self.specialize(**kwargs)
                return fact_type#call_untyped_fact(self, **kwargs)
            else:
                fact_type = self.specialize(**{k:typeof(v) for k,v in kwargs.items()})
                ctor = fact_type._ctor[0]
                return ctor(**kwargs)


    def specialize(self, **kwargs):
        typs = tuple(kwargs.values())
        self_field_dict = {k:v for (k,v) in zip(kwargs.keys(), typs)}
        spec = {**self_field_dict, "inherit_from" : self}
        fact_type = define_fact(self._fact_name, spec)
        return fact_type



    # @property
    # def key(self):
    #     """
    #     A property used for __eq__, __ne__ and __hash__.  Can be overridden
    #     in subclasses.
    #     """
    #     # if(hasattr(self,'name'))
    #     return self.name
    #     # return (self._fact_name, getattr(self,"_hash_code",None), getattr())

    

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     if(hasattr(state,'spec')): del state['spec']
    #     if(hasattr(state,'fact_ctor')): del state['fact_ctor']
    #     return state

    # def __setstate__(self,state):
    #     self.__dict__.update(state)
    #     state = self.__dict__.copy()
        
    #     return state







###### Fact Specification Preprocessing #######
class DeferredFactRefType():
    '''A placeholder type for when referencing a fact type that
        is not defined yet. Note: Sort of mimics deferred_type but
         doesn't subclass because would use special numba pickling,
         which I haven't quite figured out.
    '''
    def __init__(self,typ):
        self._fact_name = typ._fact_name if isinstance(typ, types.StructRef) else typ 
        super(DeferredFactRefType,self).__init__()
    def __eq__(self,other):
        return isinstance(other,DeferredFactRefType) \
               and self._fact_name == other._fact_name
    def __str__(self):
        return f"DeferredFactRefType[{self._fact_name}]"

    def __repr__(self):
        return f"DeferredFactRefType({self._fact_name!r})"

    @property
    def instance_type(self):
        return getattr(self,'_definition', None)

    def define(self,x):
        self._definition = x

    def get(self):
        if(not hasattr(self,"_definition")): raise TypeError(f"Attempt to use {str(self)} without definition")
        return self._definition

    def __hash__(self):
        return hash(self._fact_name)

    def __setstate__(self,state):
        self._fact_name = state[0]

    def __getstate__(self):
        return (self._fact_name,)#({'_fact_name' : self._fact_name})



def _standardize_type(typ, context, name='', attr=''):
    '''Takes in a string or type and returns the standardized type'''
    if(isinstance(typ, type)):
        typ = typ.__name__
    if(isinstance(typ,str)):
        typ_str = typ
        is_list = typ_str.lower().startswith("list")
        if(is_list): typ_str = typ_str.split("(")[1][:-1]


        is_deferred = False
        if(typ_str.lower() in TYPE_ALIASES): 
            typ = numba_type_map[TYPE_ALIASES[typ_str.lower()]]
        # elif(typ_str == name):
        #     typ = context.get_deferred_type(name)# DeferredFactRefType(name)
        elif(typ_str in context.name_to_type):
            typ = context.name_to_type[typ_str]
        else:
            typ = context.get_deferred_type(typ_str)
            is_deferred = True
            # raise TypeError(f"Attribute type {typ_str!r} not recognized in spec" + 
            #     f" for attribute definition {attr!r}." if attr else ".")

        if(is_list): typ = ListType(typ)

    if(hasattr(typ, "_fact_type")): typ = typ._fact_type
    return typ


def _merge_spec_inheritance(spec : dict, context):
    '''Expands a spec with attributes from its 'inherit_from' type'''
    if("inherit_from" not in spec): return spec, None
    inherit_from = spec["inherit_from"]

    unified_attrs = []
    if(isinstance(inherit_from, dict)):
        unified_attrs = inherit_from.get('unified_attrs')
        inherit_from = inherit_from['type']

    if(isinstance(inherit_from, str)):
        temp = inherit_from
        inherit_from = context.name_to_type[inherit_from]
    if(not isinstance(inherit_from,types.StructRef)):
        inherit_from = context.name_to_type[inherit_from._fact_name]
        
        
    if(not hasattr(inherit_from, 'spec')):
        raise ValueError(f"Invalid inherit_from : {inherit_from}")

    inherit_spec = inherit_from.spec

    _intersect = set(inherit_spec.keys()).intersection(set(spec.keys()))
    for k in _intersect:
        if(k in unified_attrs): continue
        if(spec[k]['type'] != inherit_spec[k]['type']): 
            raise TypeError(f"Attribute type {k}:{spec[k]['type']} does not" +
                            f"match inherited attribute {k}:{inherit_spec[k]['type']}")
    del spec['inherit_from']
    return {**inherit_spec, **spec}, inherit_from

def _standardize_conversions(conversions, attr_type, context):
    from cre.func import UntypedCREFunc
    assert isinstance(conversions, dict), f"'conversions' expecting dict : type -> conversion_op, not {type(conversions)}."
    stand_conv = {}
    for conv_type, conv_op in conversions.items():
        conv_type = _standardize_type(conv_type, context)
        # print(attr_type)
        if(isinstance(conv_op, UntypedCREFunc)): conv_op = conv_op(attr_type)
        assert conv_op.return_type == conv_type, f"{conv_op} does not return conversion type {conv_type}."
        stand_conv[conv_type] = conv_op
    return stand_conv

    # attr_spec['conversions'] = {_standardize_type(k, context):v for }

def _standardize_spec(spec : dict, context, name=''):
    '''Takes in a spec and puts it in standard form'''

    out = {}
    for attr, attr_spec in spec.items():
        if(attr in SPECIAL_SPEC_ATTRIBUTES): out[attr] = attr_spec; continue;

        if(isinstance(attr_spec, dict) and not "type" in attr_spec):
            raise AttributeError("Attribute specifications must have 'type' property, got %s." % v)
        
        typ, attr_spec = (attr_spec['type'], attr_spec) if isinstance(attr_spec, dict) else (attr_spec, {})
        typ = _standardize_type(typ, context, name, attr)

        if('conversions' in attr_spec): 
            attr_spec['conversions'] = _standardize_conversions(attr_spec['conversions'], typ, context)

        out[attr] = {"type": typ,**{k:v for k,v in attr_spec.items() if k != "type"}}
    return out

def _clean_spec(spec : dict):
    '''Replaces any defferred types in a spec with their definitions'''
    new_spec = {}
    for attr, attr_spec in spec.items():
        attr_t = attr_spec['type']
        attr_t = attr_t.instance_type if (isinstance(attr_t, (types.TypeRef,DeferredFactRefType))) else attr_t

        # Handle List case
        if(isinstance(attr_t, types.ListType)):
            item_t = attr_t.item_type
            item_t = item_t.instance_type if (isinstance(item_t, (types.TypeRef,DeferredFactRefType))) else item_t
            attr_t = types.ListType(item_t)
        new_spec[attr] = {**attr_spec, 'type': attr_t}
    return new_spec








###### Fact Definition #######
class FactProxy(CREObjProxy):
    # '''Essentially the same as numba.experimental.structref.StructRefProxy 0.51.2
    #     except that __new__ is not defined to statically define the constructor.
    # '''
    # __slots__ = ('_fact_type', '_meminfo')

    @classmethod
    def _numba_box_(cls, mi):
        """Called by boxing logic, the conversion of Numba internal
        representation into a PyObject.

        Parameters
        ----------
        mi :
            a wrapped MemInfoPointer.

        Returns
        -------
        instance :
             a FactProxy instance.
        """
        inst = super(FactProxy,cls)._numba_box_(BaseFact,mi)
        return inst

    @classmethod
    def _numba_box_no_recover_(cls, ty, mi):
        '''Same as StructRefProxy'''
        instance = ty.__new__(cls)
        instance._type = ty
        instance._meminfo = mi
        return instance


    @property
    def _numba_type_(self):
        """Returns the Numba type instance for this structref instance.

        Subclasses should NOT override.
        """
        return self._fact_type

    # def __eq__(self, other):
    #     from cre.dynamic_exec import fact_eq
    #     if(isinstance(other, FactProxy)):
    #         return fact_eq(self,other)
    #     return False

    def restore(self,context=None):
        context = cre_context(context)
        # if(context.tt_id):

    # def __hash__(self):
    #     from cre.dynamic_exec import fact_hash
    #     return fact_hash(self)

    def get_ptr(self):
        return fact_to_ptr(self)

    def get_ptr_incref(self):
        return fact_to_ptr_incref(self)

    def _gen_val_var_possibilities(self, self_var):
        for attr, config in self._fact_type.clean_spec.items():
            typ = config['type']
            # with PrintElapse("getattr_var"):
            val = getattr(self,attr)
            # with PrintElapse("getattr_var"):
            attr_var = getattr(self_var, attr)
            if(isinstance(val, List)):
                for i in range(len(val)):
                    item_var = attr_var[i]
                    item_val = val[i]
                    yield (item_val, item_var)
            else:
                yield (val, attr_var)
            # else:
                # Primitive case
                # one_lit_conds.append(attr_var==val)

    def as_conditions(self, fact_ptr_to_var_map=None, keep_null=True, add_implicit_neighbor_self_refs=True, neigh_count = 0):
        from cre.default_funcs import Equals
        from cre.utils import as_typed_list
        from cre.dynamic_exec import var_eq
        from cre.var import Var



        self_ptr = self.get_ptr()

        if(fact_ptr_to_var_map is None):
             fact_ptr_to_var_map = {self_ptr : Var(self._fact_type, "X")}
            

        self_var = fact_ptr_to_var_map[self_ptr]
        one_lit_conds = []

        

        
        # with PrintElapse("CONSTRUCTS"):       
        # for attr, config in self._fact_type.spec.items():
        for attr_val, attr_var in self._gen_val_var_possibilities(self_var):
            if(isinstance(attr_val, FactProxy)):
                # Fact case
                attr_val_fact_ptr = attr_val.get_ptr()
                if(attr_val_fact_ptr not in fact_ptr_to_var_map):
                    if(add_implicit_neighbor_self_refs):
                        nbr_var = Var(attr_val._fact_type, f"Nbr{neigh_count}")
                        fact_ptr_to_var_map[attr_val_fact_ptr] = nbr_var
                        neigh_count += 1
                    else:
                        continue


                val_var = fact_ptr_to_var_map[attr_val_fact_ptr]
                #   FIXME: use cre_obj.__eq__()
                
                    # str(attr_var) == str(val_var)
                # print("<<", str(attr_var), str(val_var), var_eq(attr_var, val_var))
                if(add_implicit_neighbor_self_refs and str(attr_var) == str(val_var)):
                    # for case like x.next == x.next, try make conditions like x == x.next.prev
                    # with PrintElapse("LOOP"):
                        #     list(attr_val._gen_val_var_possibilities(attr_var))
                    for attr_val2, attr_var2 in attr_val._gen_val_var_possibilities(attr_var):

                        if(isinstance(attr_val2, FactProxy) and 
                            attr_val2.get_ptr() == self_ptr):
                            one_lit_conds.append(self_var==attr_var2)

                else:
                    # with PrintElapse("NEW LIT"):
                    one_lit_conds.append(attr_var==fact_ptr_to_var_map[attr_val_fact_ptr])
                
            else:
                # Note: Making literals with primitives is slow w/ current Op 
                #  implmenentation since it compiles compositions. 

                # Primitive case
                if(not keep_null and attr_val is None): continue
                one_lit_conds.append(attr_var==attr_val)

        # with PrintElapse("ANDS"):    
        # print(fact_ptr_to_var_map)
        _vars = list({v.get_ptr():v for v in fact_ptr_to_var_map.values()}.values())
        # print(_vars)   
        conds = _vars[0]
        for i in range(1, len(_vars)):
            conds = conds & _vars[i]

        for c in one_lit_conds:
            conds = conds & c

        return conds

    def isa(self, typ):
        return isa(self,typ)

    def asa(self, typ):
        if(not isa(self,typ)):
            raise TypeError(f"Cannot cast fact '{str(self)}' to '{str(typ)}.'")
        return super().asa(typ)
    # #     instance = typ.__new__(CREObjProxy)
    # #     instance._type = typ
    # #     instance._meminfo = self._meminfo
    # #     return instance
        # return asa(self,typ)

    def __repr__(self):
        return str(self)

    # def __eq__(self, other):
    #     from cre.dynamic_exec import fact_eq
    #     return fact_eq(self.asa(CREObjType), other.asa(CREObjType))

    # def __hash__(self):
    #     from cre.dynamic_exec import fact_hash
    #     return fact_hash(self.asa(CREObjType))


def gen_fact_import_str(t):
    return f"from cre_cache.{t._fact_name}._{t._hash_code} import {t._fact_name}"

def gen_inherit_import_str(t):
    return f"from cre_cache.{t._fact_name}._{t._hash_code} import {t._fact_name} as parent_type, inheritance_bytes as parent_inh_bytes"

def _gen_getter_jit(f_typ,typ,attr):
    if(isinstance(typ,(Fact,DeferredFactRefType))):
        return \
f'''@njit(cache=True)
def get_{attr}_as_ptr(self):
    return get_fact_attr_ptr(self, '{attr}')

@njit(cache=True)
def get_{attr}(self):
    return self.{attr}
'''
    else:
        return \
f'''@njit(cache=True)
def get_{attr}(self):
    return self.{attr}
'''

def _gen_setter_jit(f_typ, attr, a_id):
    return f'''@njit(types.void({f_typ},field_list[{a_id}][1]), cache=True)
def set_{attr}(self, val):
    fact_mutability_protected_setattr(self,'{attr}',val)
'''

def _gen_props(attr):
    return f'''    {attr} = property(get_{attr}, set_{attr})'''

def get_type_default(t):
    if(isinstance(t,(bool,types.Boolean))):
        return False
    elif(isinstance(t,(str,types.UnicodeType))):
        return ""
    elif(isinstance(t,(float,types.Float))):
        return 0.0
    elif(isinstance(t,(int,types.Integer))):
        return 0
    else:
        return None



fact_types = (Fact, DeferredFactRefType)

#### Resolving Byte Offsets of Struct Members ####

def get_offsets_from_member_types(fields):
    # from cre.fact import fact_types, FactModel, BaseFact
    if(isinstance(fields, dict)): fields = [(k,v) for k,v in fields.items()]
    #Replace fact references with BaseFact
    # fact_types = (types.StructRef, DeferredFactRefType)
    fields = [(a,BaseFact if isinstance(t,fact_types) else t) for a,t in fields]

    class TempTypeTemplate(types.StructRef):
        pass

    default_manager.register(TempTypeTemplate, CREObjModel)

    TempType = TempTypeTemplate(fields)

    return [struct_get_attr_offset(TempType,attr) for attr, _ in fields]


def repr_fact_attr(inst):
    if(inst is None): return 'None'

    inst_type = type(inst)
    # cre_context().get
    # print(isinstance(inst, Fact), str(inst_type))
    if(hasattr(inst_type,"_fact_type") and
        hasattr(inst_type._fact_type, "_specialization_name")):
        return str(inst)

    ptr = inst.get_ptr()
    if(ptr != 0):
        return f"<{inst_type._fact_name} at {hex(ptr)}>"
    else:
        return 'None'

def repr_list_attr(val,dtype_name=None):
    # if(isinstance(val,Fact)):
    # ptr = get_ptr(val)
    if(val is not None):
        if(dtype_name is not None):
            return f'List([{", ".join([f"<{dtype_name} at {hex(fact_to_ptr(x))}>" for x in val])}])'
        else:
            return f'List([{", ".join([repr(x) for x in val])}])'
    else:
        return 'None'



def gen_repr_attr_code(a,t):
    '''Helper function for generating code for the repr/str of the fact'''
    if(isinstance(t,fact_types)):
        return f'{a}={{repr_fact_attr(self.{a})}}'
        # return f'{a}=<{t._fact_name} at {{hex({typ_name}_get_{a}_as_ptr(self))}}>'
    elif(isinstance(t,ListType)):
        # TODO : might want to print lists like reference where just the address is printed
        # if():
        s = ", " + f'"{t.dtype._fact_name}"' if isinstance(t.dtype,fact_types) else ""
        return f'{a}={{repr_list_attr(self.{a}{s})}}'
            # s = f'f"<{t.dtype._fact_name} at {{hex(fact_to_ptr(x))}}>"'
            # return f'{a}={{"List([" + ", ".join([{s} for x in self.{a}]) + "])" if self.{a} is not None else "None"}}'
        # else:
            # return f'{a}=List([{{", ".join([repr(x) for x in self.{a}])}}])'
    else:
        return f'{a}={{repr(self.{a})}}' 

# def gen_assign_str(a,t):
#     # if(isinstance(t,fact_types)):
#         # s = f"_ptr_from_struct_incref({a}) if ({a} is not None) else 0"
#     # elif(isinstance(t,ListType) and isinstance(t.dtype,fact_types)):
#     #     # s = f"{a}_c = _cast_list(base_list_type,{a})\n    "
#     #     # s += f"st.{a} = {a}_c" 
#     #     # s = f"st.{a} = _cast_list(base_list_type,{a})" 
#     #     s = f"st.{a} = {a}"
#     # else:
#     s = f"{a}"

#     return f"st.{a} = " + s

@njit(u1[::1](u8),cache=True)
def uint_to_inheritance_bytes(n):
    buffer = np.empty((8,), dtype=np.uint8)
    i = 0
    while(n != 0):
        buffer[8-(i+1)] = n & 0xFF
        i += 1
        n = n >> 8
    return buffer[-i:]

from cre.obj import member_info_type
from cre.utils import _sizeof_type, _load_ptr



def _prep_field(attr, t, imports_set):
    from cre.tuple_fact import TupleFactClass
    # elif(isinstance(t,types.ListType) and isinstance(t.dtype,types.StructRef)):
    #     imports_set.add(f"{gen_fact_import_str(t.dtype)}")

    # upcast any facts to BaseFact since references to undefined fact types not supported
    if(isinstance(t,fact_types)):
        if(isinstance(t,Fact) and not isinstance(t,TupleFactClass)):
            imports_set.add(f"{gen_fact_import_str(t)}")
        return attr, BaseFact
    elif(isinstance(t,ListType)):
        if(isinstance(t.dtype,fact_types)):
            _, dtype = _prep_field(attr, t.dtype, imports_set)
            # if(isinstance(dtype, types.Optional)): dtype = dtype.type

            return (attr, ListType(dtype))
        return attr, t
    else:
        return attr, t


def _prep_fields_populate_imports(fields, inherit_from=None):
    imports_set = set()
    if(inherit_from is not None):
        imports_set.add(f"{gen_inherit_import_str(inherit_from)}")
    fields = [_prep_field(attr,t,imports_set) for attr, t in fields]
        
    return fields, "\n".join(list(imports_set))

def gen_fact_src(typ, fields, t_id, inherit_from=None, specialization_name=None, is_untyped=False, hash_code="", ind='    '):
    '''Generate the source code for a new fact '''
    fields, fact_imports = _prep_fields_populate_imports(fields, inherit_from)

    _base_fact_field_dict = {**base_fact_field_dict}
    all_fields = [(k,v) for k,v in _base_fact_field_dict.items()] + fields    # all_fields = [(k,v) for (k,v) in all_fields]

    # all_fields = base_fact_fields+fields
    properties = "\n".join([_gen_props(attr) for attr,t in all_fields])
    getter_jits = "\n".join([_gen_getter_jit(typ,t,attr) for attr,t in all_fields])
    setter_jits = "\n".join([_gen_setter_jit(typ,attr,a_id) for a_id, (attr,t) in enumerate(all_fields)])
    # field_list = ",".join(["'%s'"%attr for attr,t in fields])

    param_defaults_seq = ",".join([f"{attr}={get_type_default(t)!r}" for attr,t in fields])
    param_seq = ",".join([f"{attr}" for attr,t in fields])
    attr_tup = tuple([attr for attr,t in fields])

    # base_list = ",".join([f"'{k}'" for k,v in _base_fact_fields])
    # base_type_list = ",".join([str(v) for k,v in _base_fact_fields])

    # fact_types = (types.StructRef, DeferredFactRefType)
    
    # field_type_list = ",".join([repr_type(v) for k,v in fields])


    # assign_str = lambda a,t: f"st.{a} = " + (f"_ptr_from_struct_incref({a}) if ({a} is not None) else 0" \
    #                         if isinstance(t,fact_types) else f"{a}")
    init_fields = f'\n{ind}'.join([f"fact_lower_setattr(st,'{k}',{k})" for k,v in fields])

    str_temp = ", ".join([gen_repr_attr_code(k,v) for k,v in fields])

    #TODO get rid of this

    attr_offsets = get_offsets_from_member_types(all_fields)

# The source code template for a user defined fact. Written to the
#  system cache so it can be its own module. Doing so helps njit(cache=True)
#  work when using user defined facts.
    code = \
f'''
import numpy as np
from numba.core import types
from numba import njit, literally, literal_unroll
from numba.core.types import *
from numba.core.types import unicode_type, ListType, UniTuple, Tuple
from numba.experimental import structref
from numba.experimental.structref import new#, define_boxing
from numba.core.extending import overload, lower_cast, type_callable
from numba.core.imputils import numba_typeref_ctor
from cre.fact_intrinsics import define_boxing, get_fact_attr_ptr, _register_fact_structref, fact_mutability_protected_setattr, fact_lower_setattr, _fact_get_chr_mbrs_infos
from cre.fact import repr_list_attr, repr_fact_attr, FactProxy, Fact, UntypedFact{", BaseFact, base_list_type, fact_to_ptr, get_inheritance_bytes_len_ptr" if typ != "BaseFact" else ""}, uint_to_inheritance_bytes
from cre.utils import cast, ptr_t, _get_member_offset,  _load_ptr, _obj_cast_codegen, encode_idrec
import cloudpickle
from cre.obj import member_info_type, set_chr_mbrs
{fact_imports}



attr_offsets = np.array({attr_offsets!r},dtype=np.int16)
inheritance_bytes = tuple({"list(parent_inh_bytes) + [u1(0)] + " if inherit_from else ""}list(uint_to_inheritance_bytes({t_id}))) 
num_inh_bytes = len(inheritance_bytes)
hash_code = '{hash_code}'

@_register_fact_structref
class {typ}Class({"UntypedFact" if is_untyped else "Fact"}):
    def __init__(self, fields, hash_code=None):
        super().__init__('{typ}', fields, hash_code)
        self._fact_name = '{typ}'
        self.t_id = {t_id}
        self._attr_offsets = attr_offsets
        self._hash_code = '{hash_code}'
        {f'self._specialization_name = "{specialization_name}"' if(specialization_name is not None) else ''}

    

field_list = cloudpickle.loads({cloudpickle.dumps(all_fields)})
{typ} = fact_type = {typ}Class(field_list, hash_code)
{typ}_w_mbr_infos = {typ}Class(field_list+
[("chr_mbrs_infos", UniTuple(member_info_type,{len(fields)})),
 ("num_inh_bytes", u1),
 ("inh_bytes", UniTuple(u1, num_inh_bytes))])


{(f"""{typ}.parent_type = parent_type
pt = parent_type
while(pt is not None):
    @lower_cast({typ}, pt)
    def upcast(context, builder, fromty, toty, val):
        return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)                        
    pt = getattr(pt, 'parent_type', None)
""") if inherit_from is not None else ""
}

@njit(cache=True)
def get_chr_mbrs_infos():
    st = new({typ})
    return _fact_get_chr_mbrs_infos(st)

chr_mbrs_infos = get_chr_mbrs_infos()
{typ}Class._chr_mbrs_infos = chr_mbrs_infos

{(f"""#locals={{'inheritance_bytes':Tuple((u1,num_inh_bytes))}}
@njit(u1(BaseFact), cache=True)
def isa_{typ}(fact):
    l, p = get_inheritance_bytes_len_ptr(fact)
    if(l >= num_inh_bytes):
        for i,b in enumerate(literal_unroll(inheritance_bytes)):
            f_b = _load_ptr(u1, p+i)
            if(b != f_b):
                return False
        return True
    else:
        return False
""") if typ != "BaseFact" else (
f"""@njit(u1(BaseFact), cache=True)
def isa_{typ}(fact):
    return True
"""
)
}
{typ}Class._isa = isa_{typ}

@njit(cache=True)
def ctor({param_defaults_seq}):
    st = new({typ}_w_mbr_infos)
    fact_lower_setattr(st,'idrec',encode_idrec({t_id},0,u1(-1)))
    fact_lower_setattr(st,'hash_val',0)
    set_chr_mbrs(st, {attr_tup!r})
    fact_lower_setattr(st,'num_inh_bytes', num_inh_bytes)
    fact_lower_setattr(st,'inh_bytes', inheritance_bytes)
    {init_fields}
    return cast(st, {typ})

# Put in a tuple so it doesn't get wrapped in a method
{typ}Class._ctor = (ctor,)

{getter_jits}

{setter_jits}
        
class {typ}Proxy(FactProxy):
    __numba_ctor = ctor
    _fact_type = {typ}
    _fact_type_class = {typ}Class
    _fact_name = '{typ}'
    {f'_specialization_name = "{specialization_name}"' if(specialization_name is not None) else ''}
    t_id = {t_id}
    _attr_offsets = attr_offsets
    _chr_mbrs_infos = chr_mbrs_infos
    _isa = isa_{typ}
    _code = {typ}._code
    _hash_code = '{hash_code}'

    def __new__(cls, *args,**kwargs):
        return ctor(*args,**kwargs)

    def __repr__(self):
        return f'{typ}({str_temp})'

    def __str__(self):
        return self.__repr__()


{properties}

# Overload '{typ}' as its own constructor
@type_callable({typ})
def ssp_call(context):
    {"raise NotImplementedError('NotImplementedError: Cannot initialize UntypedFact in jitted context.')" if is_untyped else ""}
    # Note to self this requires *args, see https://github.com/numba/numba/issues/7973
    def typer({param_seq}):    
        {f'return {typ}.specialize({param_seq})' if is_untyped else f'return {typ}'}
    return typer

@overload(numba_typeref_ctor)
def overload_{typ}(self, {param_defaults_seq}):
    if(self.instance_type is not {typ}): return
    def impl(self, {param_defaults_seq}):
        return ctor({param_seq})
    return impl

{typ}Class._fact_type = {typ}
{typ}Class._fact_proxy = {typ}Proxy
{typ}Class._proxy_class = {typ}Proxy

{typ}._fact_type_class = {typ}Class
{typ}._fact_proxy = {typ}Proxy
{typ}._proxy_class = {typ}Proxy


define_boxing({typ}Class,{typ}Proxy)



{(f"""from cre.var import VarType, var_ctor
# @njit(VarType(unicode_type), cache=True)
# def as_var(alias):
#     return var_ctor({typ}, {t_id}, alias)
# {typ}._as_var = as_var
""") if typ != "BaseFact" else ""
}


'''
    return code



def _fact_from_fields(name, fields, inherit_from=None, specialization_name=None, is_untyped=False, return_proxy=False, return_type_class=False):
    # context = cre_context(context)
    

    hash_code = unique_hash_v([name,fields])
    if(not source_in_cache(name,hash_code)):
        # Possible for other types to be defined while running the Fact source
        #  so preregister the t_id then add the pickle later.
        t_id = add_to_type_registry(name, hash_code)
        source = gen_fact_src(name, fields, t_id, inherit_from, specialization_name, is_untyped, hash_code)
        source_to_cache(name, hash_code, source)
        
        fact_type = tuple(import_from_cached(name, hash_code, [name]).values())[0]
        add_type_pickle(fact_type, t_id)
        
    to_get = [name]
    if(return_proxy): to_get.append(name+"Proxy")
    if(return_type_class): to_get.append(name+"Class")
        
    out = tuple(import_from_cached(name, hash_code, to_get).values())
    for x in out: x._hash_code = hash_code
        
    return tuple(out) if len(out) > 1 else out[0]

def _fact_from_spec(name, spec, inherit_from=None, specialization_name=None, return_proxy=False, return_type_class=False):
    # assert parent_fact_type
    is_untyped = bool(spec is None)
    fields = [(k,v['type']) for k, v in spec.items()] if spec else {}
    return _fact_from_fields(name, fields,
             inherit_from=inherit_from, specialization_name=specialization_name,
            is_untyped=is_untyped, return_proxy=return_proxy,
            return_type_class=return_type_class)

def _spec_eq(spec_a, spec_b):
    # print(list(spec_a.keys()), list(spec_b.keys()))
    for attr_a, attr_b in zip(spec_a, spec_b):
        if(attr_a != attr_b): 
            # print(attr_a, "!=", attr_b)
            return False
        typ_a, typ_b = spec_a[attr_a]['type'], spec_b[attr_a]['type']

        typ_strs = []
        for typ in [typ_a, typ_b]:
            if(isinstance(typ, ListType)):
                if(isinstance(typ.item_type,(DeferredFactRefType, Fact))):
                    item_str = typ.item_type._fact_name
                else:
                    item_str = str(typ.item_type)
                typ_strs.append(f"List({item_str})")

            elif(isinstance(typ, (DeferredFactRefType, Fact))):
                typ_strs.append(typ._fact_name)
            else:
                typ_strs.append(str(typ))

        if(typ_strs[0] != typ_strs[1]):
            # print(typ_strs[0], "!=", typ_strs[1])
            return False
    return True


def define_fact(name : str, spec : dict = None, context=None, return_proxy=False, return_type_class=False, allow_redef=False):
    '''Defines a new fact.'''

    from cre.context import cre_context
    context = cre_context(context)
    # print("DEFINE", name, context.name)
    specialization_name = name
    if(spec is not None):
        spec = _standardize_spec(spec,context,name)
        spec, inherit_from = _merge_spec_inheritance(spec, context)
    
        if(inherit_from is not None and 
            inherit_from._fact_name == name):
            # Specialize UntypedFact case
            fact_type = context.name_to_type[name]
            typ_assigments = ", ".join([f"{k}={str(v['type'])}" for k,v in spec.items() if k not in base_fact_field_dict])
            specialization_name = f"{name}({typ_assigments})"
    else:
        inherit_from = None


    if(specialization_name in context.name_to_type):
        assert _spec_eq(context.name_to_type[specialization_name].spec, spec), \
        f"Redefinition of fact '{specialization_name}' in context '{context.name}' not permitted"
        # print("SPECIALIZATION NAME:", specialization_name)
        fact_type = context.name_to_type[specialization_name]
    else:

        fact_type = _fact_from_spec(name, spec, inherit_from=inherit_from, 
            specialization_name= (specialization_name if specialization_name != name else None),
            return_proxy=False, return_type_class=False)
        dt = context.get_deferred_type(name)
        dt.define(fact_type)
        # context._assert_flags(name, spec)
        context._register_fact_type(specialization_name, fact_type, inherit_from=inherit_from)
    _spec = spec if(spec is not None) else {}
    # _spec = _undeffer_spec(_spec)
    # print({_id : str(config['type']) for _id, config in _spec.items()})
    fact_type.spec = _spec
    fact_type._fact_proxy.spec = _spec
    fact_type._fact_type_class._spec = _spec

    # Needs to be done because different definitions can share a 
    #  fact_type object
    if(hasattr(fact_type,'_clean_spec')): del fact_type._clean_spec

    out = [fact_type]
    if(return_proxy): out.append(fact_type._fact_proxy)
    if(return_type_class): out.append(fact_type._fact_type_class)
    return tuple(out) if len(out) > 1 else out[0]
    # return fact_ctor, fact_type


def define_facts(specs, #: list[dict[str,dict]],
                 context=None):
    '''Defines several facts at once.'''
    for name, spec in specs.items():
        define_fact(name,spec,context=context)

###### Base #####

base_fact_field_dict = {
    **cre_obj_field_dict,
}

base_fact_fields  = [(k,v) for k,v in base_fact_field_dict.items()]

BaseFact = _fact_from_fields("BaseFact", [])
register_global_default("Fact", BaseFact)

base_list_type = ListType(BaseFact)

# @lower_cast(Fact, CREObjType)
@lower_cast(Fact, BaseFact)
def upcast(context, builder, fromty, toty, val):
    return _obj_cast_codegen(context, builder, val, fromty, toty,incref=False)


@njit(i8(CREObjType), cache=True)
def fact_to_ptr(fact):
    return cast(fact, i8)

@njit(cache=True)
def fact_to_basefact(fact):
    return cast(fact, BaseFact)

@njit(cache=True)
def fact_to_ptr_incref(fact):
    return _ptr_from_struct_incref(fact)


###### Fact Casting #######
@generated_jit
def cast_fact(typ, val):
    '''Casts a fact to a new type of fact if possible'''
    from cre.context import cre_context
    context = cre_context()    
    inst_type = typ.instance_type

    #Check if the fact_type can be casted 
    if(inst_type._fact_name != "BaseFact" and val._fact_name != "BaseFact" and
       inst_type._fact_name not in context.children_of[val._fact_name] and 
       inst_type._fact_name not in context.parents_of[val._fact_name]
       
    ):
        error_message = f"Cannot cast fact of type '{val._fact_name}' to '{inst_type._fact_name}.'"
        #If it shouldn't be possible then throw an error
        def error(typ,val):
            raise TypeError(error_message)
        return error
    
    def impl(typ,val):
        return cast(val, inst_type)

    return impl


@njit(Tuple((u1,i8))(BaseFact), cache=True)
def get_inheritance_bytes_len_ptr(st):
    ptr = _struct_get_data_ptr(st) + st.chr_mbrs_infos_offset + \
             (st.num_chr_mbrs * _sizeof_type(member_info_type))
    num_inh_bytes = _load_ptr(u1, ptr)
    return num_inh_bytes, ptr+1


@njit(ListType(i8)(BaseFact), cache=True)
def get_inheritance_t_ids(st):
    nbytes, ptr = get_inheritance_bytes_len_ptr(st)
    t_ids = List.empty_list(i8)
    prev_byte = u1(-1)
    val = i8(0)
    prev_val = i8(0)
    for i in range(nbytes):
        byte = _load_ptr(u1,ptr+i)
        val = (val << 8) | byte
        if(byte != 0 and prev_byte == 0):
            t_ids.append(prev_val>>8)    
            val = i8(byte)
            
        prev_byte = byte
        prev_val = val
    t_ids.append(val)    
    return t_ids


# @njit()
# def fact_setitem(fact, index):





@generated_jit(cache=True,nopython=True)
@overload_method(Fact, "isa")
def isa(self, typ):
    typ = typ.instance_type


    if(typ is BaseFact or typ is CREObjType):
        def impl(self, typ):
            return True

    # If the type has defined it's own _isa then use that
    elif(hasattr(typ, '_isa')):
        _isa = typ._isa
        def impl(self, typ):
            return _isa(self)

    # Otherwise just check if the first t_id in inh_bytes is t_id of typ   
    #   This case handles cases like TupleFact that can only have one 
    #   level of inheritance.
    elif(hasattr(typ, "t_id")):
        t_id = typ.t_id
        inh_bytes = uint_to_inheritance_bytes(t_id)
        def impl(self, typ):
            l, p = get_inheritance_bytes_len_ptr(self)
            for i,b in enumerate(literal_unroll(inh_bytes)):
                f_b = _load_ptr(u1, p+i)
                if(b != f_b):
                    return False
            return True
    return impl

@generated_jit(cache=True,nopython=True)
@overload_method(Fact, "asa")
def asa(self, typ):
    # from numba.extending import SentryLiteralArgs
    # SentryLiteralArgs(['unsafe']).for_function(asa).bind(self, typ, unsafe) 

    _typ = typ.instance_type
    use_unsafe_cast = (_typ is CREObjType) or (_typ is BaseFact)

    # print("TRY CAST:", f'{str(self)} to {str(_typ)}')
    if(not use_unsafe_cast):
        error_message = f"Cannot cast fact of type '{str(self)}' to '{str(_typ)}.'"

        # from cre.context import cre_context
        # context = cre_context()    
        
        # print(context.children_of[self], _typ, _typ in context.children_of[self])
        # print(context.parents_of[self], _typ, _typ in context.parents_of[self])
        # #Check if the fact_type can be casted 
        # if(self in context.children_of and 
        #    _typ not in context.children_of[self] and 
        #    self in context.parents_of and
        #    _typ not in context.parents_of[self]
        # ):

        #     # error_message = f"Cannot cast fact of type '{val._fact_name}' to '{inst_type._fact_name}.'"
        #     #If it shouldn't be possible then throw an error
        #     def error(self, typ):
        #         raise TypeError(error_message)
        #     return error

        # fn1, fn2 = self._fact_name, _typ._fact_name
        _isa = typ.instance_type._isa
        def impl(self, typ):
            # print(self, _isa(self))
            if(not _isa(self)): raise TypeError(error_message)
            return cast(self, typ)
    else:
        def impl(self, typ):
            return cast(self, typ)
    return impl




#### Hashing ####

# @njit(cache=True)
# def fact_hash():


