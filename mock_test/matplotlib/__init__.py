import sys
class DummyMock:
    def __getattr__(self, name): return DummyMock()
    def __call__(self, *args, **kwargs): return DummyMock()
    def __getitem__(self, key): return DummyMock()
    def __setitem__(self, key, val): pass
    def __iter__(self): return iter([DummyMock(), DummyMock()])
sys.modules[__name__] = DummyMock()
