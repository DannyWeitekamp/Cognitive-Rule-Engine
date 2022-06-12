from numba import types

attr_filter_registry = {}

class AttrFilter():
    def __init__(self, flag, filter_func=None):
        self.flag = flag
        if(filter_func is None): 
            filter_func = lambda attr, config : False
        self.filter_func = filter_func
        attr_filter_registry[self.flag] = self

    def get(self, fact_type, negated=False):
        for attr, config in fact_type.clean_spec.items():
            flag = config.get(self.flag, None)
            if(flag == True ^ negated): yield attr, config
            if(flag == False ^ negated): continue
            if(self.filter_func(attr, config) ^ negated):
                yield attr, config

def should_be_relational(attr, config):
    from cre.fact import Fact
    attr_t = config['type']
    if(isinstance(attr_t, Fact)): return True
    if(isinstance(attr_t, types.ListType)):
        if(isinstance(attr_t.item_type, Fact)): return True
    return False

def should_be_spatial(attr, config):
    return (
        (attr_t == "x" and isinstance(config['type'], types.number_domain)) or 
        (attr_t == "y" and isinstance(config['type'], types.number_domain)) or 
        (attr_t == "width" and isinstance(config['type'], types.number_domain)) or 
        (attr_t == "height" and isinstance(config['type'], types.number_domain))
    )

visible = AttrFilter("visible")
relational = AttrFilter("relational", should_be_relational)
semantic = AttrFilter("semantic")
spatial = AttrFilter("spatial", should_be_spatial)
