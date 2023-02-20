

if __name__ == "__main__":
    from cre import define_fact, Var
    from cre.rule import Rule
    from cre.caching import unique_hash
    import cloudpickle

    BOOP = define_fact("BOOP", {"A" : str, "B" : float})
    l1 = Var(BOOP, 'l1')
    l2 = Var(BOOP, 'l2')

    c = (l1.B < 7) & (l2.B == 5) & (l1.B < l2.B)

    @Rule(c)
    def foo(wm, _l1, _l2):
        b = BOOP(_l1.A + "_" + _l2.A, _l1.B + _l2.B)
        wm.declare(b)

    @Rule(c)
    def bar(wm, _l1, _l2):
        b = BOOP(_l1.A + "_" + _l2.A, _l1.B + _l2.B)
        wm.declare(b)

    # print(cloudpickle.dumps(foo))
    # print(unique_hash(cloudpickle.dumps(foo)))

    # r = Rule(c, foo)
    # print(r)

    print(foo)
    print(bar)
        
