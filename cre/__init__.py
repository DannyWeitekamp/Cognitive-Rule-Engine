import contextvars


skip = contextvars.ContextVar("cre_skip_package_level_imports",default=False)
# print("<<", skip.get())
if(not skip.get()):
    import cre.cfuncs

    # Instantiate the default context at the package level
    from cre.version import __version__
    import cre.context as context_module

    cre_context_ctxvar = contextvars.ContextVar("cre_context",
            default=context_module.CREContext.get_default_context()) 
    context_module.cre_context_ctxvar = cre_context_ctxvar

    # Necessary to import these at package level to ensure that they are defined
    #  before a context is created
    import cre.obj
    from cre.fact import define_fact, BaseFact, Fact, FactProxy
    from cre.tuple_fact import TupleFact, TF
    from cre.var import Var
    from cre.func import CREFunc, UntypedCREFunc
    from cre.conditions import Literal, Conditions

    # Other helpful things exposed at the package level
    from cre.context import cre_context
    from cre.memset import MemSet

    import cre.dynamic_exec

    # Import this to monkey-patch numba 
    import cre.type_conv

    
