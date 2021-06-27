from cre.make_source import make_source



def test_all():
    class Foo():
        @make_source("python")
        def python_call_src(self):
            return 'just python'

        @make_source("python","special_piece")
        def python_call_long_src(self, b="key word",**kwargs):
            return 'special ' + b

        @make_source("js")
        @make_source("python","piece")
        @make_source("python","other_piece")
        def boop(self, b="key word", **kwargs):
            return 'boop ' + b

    assert Foo.make_source('python') == 'just python'
    assert Foo.make_source('python',"special_piece") == 'special key word'
    assert Foo.make_source('python',"special_piece",b="thing") == 'special thing'

    assert Foo.make_source('js',b="thing") == 'boop thing'
    assert Foo.make_source('python',"piece",b="thing") == 'boop thing'
    assert Foo.make_source('python',"other_piece",b="thing") == 'boop thing'




if __name__ == "__main__":
    test_all()


