from numbert.experimental.fact import _fact_from_spec, _standardize_spec, _merge_spec_inheritance, \
    define_fact, cast_fact


def test__standardize_spec():
    spec = {"A" : "string", "B" : "number"}
    spec = _standardize_spec(spec)
    print(spec)
    assert spec['A']['type'] == 'unicode_type'
    assert 'nominal' in spec['A']['flags']
    assert spec['B']['type'] == 'float64'

def test__merge_spec_inheritance():
    spec1 = {"A" : "string", "B" : "number"}
    spec2 = {"inherit_from" : spec1, "C" : "number"}

    spec = _merge_spec_inheritance(spec2)
    assert "A" in spec
    assert "B" in spec
    assert "inherit_from" not in spec
    print(spec)

def test_define_fact():
    spec = {"A" : "string", "B" : "number"}
    ctor, typ = define_fact("BOOP", spec, context="test__fact_from_spec")
    assert True
    print(ctor, typ)





if __name__ == "__main__":
    test__standardize_spec()
    test__merge_spec_inheritance()
    test_define_fact()
