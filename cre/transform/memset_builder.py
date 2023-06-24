from cre.context import cre_context
from cre.memset import MemSet

class MemSetBuilder():
    def __init__(self, out_memsets=None, context=None):
        self.context = cre_context(context)
        self.non_relational_attrs = {}
        self.relational_attrs = {}
        self.out_memsets = out_memsets

    def get_non_relational(self, type_name):
        if(type_name not in self.non_relational_attrs):
            fact_type = self.context.get_type(name=type_name)
            attrs = list(fact_type.filter_spec("~relational").keys())
            self.non_relational_attrs[type_name] = attrs
        return self.non_relational_attrs[type_name]

    def get_relational(self, type_name):
        if(type_name not in self.relational_attrs):
            fact_type = self.context.get_type(name=type_name)
            attrs = list(fact_type.filter_spec("relational").keys())
            self.relational_attrs[type_name] = attrs
        return self.relational_attrs[type_name]

    def tranform(self, state_dict, out_memset=None, return_map=False):
        '''Converts a dictionary of dictionaries each representing a fact into a MemSet'''
        # Make each fact instance, but skip setting any relational members.
        fact_instances = {}
        for _id, config in state_dict.items():
            fact_type = self.context.get_type(name=config['type'])
            non_rel_attrs = self.get_non_relational(config['type'])
            kwargs = {attr:config[attr] for attr in non_rel_attrs if attr in config} 
            fact = fact_type(**kwargs)
            fact_instances[_id] = fact

        # Now that all facts exist set any relational attributes
        for _id, fact in fact_instances.items():
            config = state_dict[_id]
            rel_attrs = self.get_relational(config['type'])

            for attr in rel_attrs:
                if(attr not in config): continue
                val_name = config[attr]
                if(not val_name): continue # i.e. skip if None or empty str
                if(val_name not in fact_instances):
                    raise ValueError(f"Reference to unspecified fact {val_name}.")
                setattr(fact, attr, fact_instances[val_name])

        # Declare each fact to a new MemSet
        if(out_memset is None):
            out_memset = self.out_memsets if(self.out_memsets) else MemSet(auto_clear_refs=True)
        for _id, fact in fact_instances.items():
            out_memset.declare(fact)

        if(return_map):
            return out_memset, fact_instances
        else:
            return out_memset

    def __call__(self, state_dict, *args, **kwargs):
        return self.tranform(state_dict, *args, **kwargs)
