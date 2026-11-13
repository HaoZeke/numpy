import numpy.f2py.crackfortran as crackfortran
import pytest

from . import util


def test_cf2py_intent_is_merged(tmp_path, capsys):
    # gh-15934: cracking the callback example used to report
    # appenddecl: "intent" not implemented.
    src = tmp_path / "callback.f"
    src.write_text(
        "      subroutine foo(fun, r)\n"
        "      external fun\n"
        "      real*8 r, fun\n"
        "Cf2py intent(out) r\n"
        "      r = fun(2d0)\n"
        "      end\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    captured = capsys.readouterr().out
    assert "not implemented" not in captured
    foo = next(block for block in blocks if block.get("name") == "foo")
    assert "out" in foo["vars"]["r"]["intent"]


@pytest.mark.slow
class TestCallbackIntentOut(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "intent_out.f")]

    def test_callback_return(self):
        def fun(i):
            return i * i

        assert self.module.foo(fun) == 110.0
