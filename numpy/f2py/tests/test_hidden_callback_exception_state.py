"""Hidden callbacks preserve exception state across thread-local cleanup."""

import pytest

from . import util

_ROUTINES = [('primary', 1), ('secondary', 3), ('tertiary', 5)]


class TestHiddenCallbackPendingExceptions(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'callback', 'gh18385.f90')]
    options = ['-DF2PY_USE_PYTHON_TLS']

    @pytest.mark.parametrize('routine, first', _ROUTINES)
    def test_exception_identity_and_traceback(self, monkeypatch, routine, first):
        expected = ValueError('hidden callback failure')
        calls = []

        def callback(a, b):
            calls.append((a, b))
            raise expected

        monkeypatch.setattr(self.module, 'callback', callback, raising=False)
        with pytest.raises(ValueError) as error:
            getattr(self.module, routine)()
        assert error.value is expected
        assert calls == [(first, first + 1)]
        traceback = error.value.__traceback__
        while traceback.tb_next is not None:
            traceback = traceback.tb_next
        assert traceback.tb_frame.f_code is callback.__code__

    @pytest.mark.parametrize('routine, first', _ROUTINES)
    def test_successful_callback_after_exception(self, monkeypatch, routine, first):
        def callback(a, b):
            raise ValueError(f'from {a}')

        monkeypatch.setattr(self.module, 'callback', callback, raising=False)
        with pytest.raises(ValueError, match=f'from {first}'):
            getattr(self.module, routine)()

        calls = []
        monkeypatch.setattr(self.module, 'callback',
                            lambda a, b: calls.append((a, b)), raising=False)
        self.module.primary()
        self.module.secondary()
        self.module.tertiary()
        assert calls == [(1, 2), (3, 4), (5, 6)]
