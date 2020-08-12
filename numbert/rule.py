


class BaseRule():
    def _init_condition():
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        self.cond_data = to_flat_instructions(cls.condition(), cls.pattern.keys())






class Add(BaseRule):
    pattern = {
        "sel": {
            "type" : "TextField",
            "pos_pattern" : {
                "above": "arg1",
            }
        },
        "arg0": {
            "type" : "TextField",
            "pos_pattern" : {
                "below": "arg1",
            }
        },
        "arg1": {
            "type" : "TextField",
            "pos_pattern" : {
                "above": "arg0",
                "below": "sel",
            }
        }
    }
    def condition(sel,arg0,arg1):
        return sel.to_right.above.value == "moose"

    def forward(kb,sel,arg0,arg1):
        kb.halt()



