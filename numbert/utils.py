import linecache
def cache_safe_exec(source,lcs=None,gbls=None,cache_name='cache-safe'):
    fp = "<ipython-%s>" %cache_name
    lines = [line + '\n' for line in source.splitlines()]
    linecache.cache[fp] = (len(source), None, lines, fp)
    code = compile(source,fp,'exec')
    l = lcs if lcs is not None else {}
    g = gbls if gbls is not None else globals()
    exec(code,g,l)
    return l,g