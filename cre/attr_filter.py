from numba import types

attr_filter_registry = {}

class AttrFilter():
    def __init__(self, flag, filter_func=None):
        self.flag = flag
        if(filter_func is None): 
            filter_func = lambda attr, config, _ : False
        self.filter_func = filter_func
        attr_filter_registry[self.flag] = self

    def get_attrs(self, fact_type, negated=False):
        clean_spec = fact_type.clean_spec

        for attr, config in clean_spec.items():
            flag = config.get(self.flag, None)
            if(flag == True ^ negated): yield attr, config
            if(flag == False ^ negated): continue
            if(self.filter_func(attr, config, clean_spec) ^ negated):
                yield attr, config

def should_be_relational(attr, config, clean_spec):
    from cre.fact import Fact
    attr_t = config['type']
    if(isinstance(attr_t, Fact)): return True
    if(isinstance(attr_t, types.ListType)):
        if(isinstance(attr_t.item_type, Fact)): return True
    return False

def should_be_spatial(attr, config, clean_spec):
    return (
        (attr_t == "x" and isinstance(config['type'], types.number_domain)) or 
        (attr_t == "y" and isinstance(config['type'], types.number_domain)) or 
        (attr_t == "width" and isinstance(config['type'], types.number_domain)) or 
        (attr_t == "height" and isinstance(config['type'], types.number_domain))
    )

def should_be_few_valued(attr, config, clean_spec):
    return isinstance(config['type'], types.Boolean)

def should_be_parent(attr, config, clean_spec):
    return "parent" in attr 

def should_be_unique_id(attr, config, clean_spec):
    if("id" in clean_spec):
        return attr == "id"
    elif("name" in clean_spec):
        return attr == "name"
    return False

visible = AttrFilter("visible")
relational = AttrFilter("relational", should_be_relational)
semantic = AttrFilter("semantic")
spatial = AttrFilter("spatial", should_be_spatial)
few_valued = AttrFilter("few_valued", should_be_few_valued)
parent = AttrFilter("parent", should_be_few_valued)
unique_id = AttrFilter("unique_id", should_be_unique_id)
