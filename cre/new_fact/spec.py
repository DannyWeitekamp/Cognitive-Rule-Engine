
SPECIAL_SPEC_ATTRIBUTES = ["inherit_from"]



def merge_spec_inheritance(spec : dict, context):
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
        conv_type = context.standardize_type(conv_type)
        if(isinstance(conv_op, UntypedCREFunc)): conv_op = conv_op(attr_type)
        assert conv_op.return_type == conv_type, f"{conv_op} does not return conversion type {conv_type}."
        stand_conv[conv_type] = conv_op
    return stand_conv

    # attr_spec['conversions'] = {_standardize_type(k, context):v for }

def standardize_spec(spec : dict, context, name=''):
    '''Takes in a spec and puts it in standard form'''
    out = {}
    for attr, attr_spec in spec.items():
        if(attr in SPECIAL_SPEC_ATTRIBUTES): out[attr] = attr_spec; continue;

        if(isinstance(attr_spec, dict) and not "type" in attr_spec):
            raise AttributeError("Attribute specifications must have 'type' property, got %s." % v)
        
        typ, attr_spec = (attr_spec['type'], attr_spec) if isinstance(attr_spec, dict) else (attr_spec, {})
        typ = context.standardize_type(typ, name, attr)

        if('conversions' in attr_spec): 
            attr_spec['conversions'] = _standardize_conversions(attr_spec['conversions'], typ, context)

        out[attr] = {"type": typ,**{k:v for k,v in attr_spec.items() if k != "type"}}
    return out

def clean_spec(spec : dict):
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



def filter_spec(spec, flags):
    ''' Returns a filtered version of this spec containing only
        the subset of attributes in the satisfying a set of flags
        like [('visible', 'few_valued'), ('unique_id')] where the list
        of lists or tuples of flags represents a disjunctive normal 
        statement i.e. ('visible' and 'few_valued') or 'unique_id'.
    '''

    # Ensure that flags is list of lists
    if(not isinstance(flags,(list,tuple))): flags = [flags]
    if(len(flags) == 0): return spec

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
        conj_filtered_spec = {k:v for k,v in attr_filter_registry[flag_0].get_attrs(spec, negated_0)}
        for i in range(1,len(and_flags)):
            flag = and_flags[i]
            negated = flag[0] == "~"
            flag = flag[1:] if negated else flag
            if(flag not in attr_filter_registry):
                raise ValueError(f"No filter registered under flag {flag!r}")
            _fs = attr_filter_registry[flag].get_attrs(spec, negated)
            conj_filtered_spec = {k : v for k,v in _fs if k in conj_filtered_spec}
        filtered_spec = {**filtered_spec, **conj_filtered_spec}
    return filtered_spec

def spec_eq(spec_a, spec_b):
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
