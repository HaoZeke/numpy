import pytest

from numpy.f2py import crackfortran

from . import util


def _routine(src, tmp_path, name):
    path = tmp_path / name
    path.write_text(src)
    crackfortran.reset_global_f2py_vars()
    crackfortran.verbose = 0
    return crackfortran.crackfortran([str(path)])[0]


def test_trailing_directive_keeps_integer_intent(tmp_path):
    mod = _routine(
        "subroutine foo(x)\n"
        "  integer :: x !f2py intent(out) x\n"
        "  x = 7\n"
        "end subroutine\n",
        tmp_path,
        "trail.f90",
    )
    assert mod["name"] == "foo"
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["vars"]["x"]["intent"] == ["out"]


def test_directive_on_subroutine_line(tmp_path):
    mod = _routine(
        "subroutine foo(x) !f2py intent(out) x\n"
        "  integer :: x\n"
        "  x = 7\n"
        "end subroutine\n",
        tmp_path,
        "subline.f90",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["x"]
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["vars"]["x"]["intent"] == ["out"]


def test_trailing_directive_in_continued_decl(tmp_path):
    mod = _routine(
        "subroutine foo(x, y)\n"
        "  integer :: &\n"
        "    x & !f2py intent(out) x\n"
        "    , y\n"
        "end subroutine\n",
        tmp_path,
        "contmid.f90",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["x", "y"]
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["vars"]["y"]["typespec"] == "integer"
    assert mod["vars"]["x"]["intent"] == ["out"]


def test_trailing_directive_before_continued_name(tmp_path):
    mod = _routine(
        "subroutine foo(x)\n"
        "  integer :: & !f2py intent(out) x\n"
        "    x\n"
        "  x = 7\n"
        "end subroutine\n",
        tmp_path,
        "contopen.f90",
    )
    assert mod["name"] == "foo"
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["vars"]["x"]["intent"] == ["out"]


def test_fixed_form_trailing_directive(tmp_path):
    mod = _routine(
        "      subroutine foo(x)\n"
        "      integer x !f2py intent(out) x\n"
        "      x = 4\n"
        "      end\n",
        tmp_path,
        "trail.f",
    )
    assert mod["name"] == "foo"
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["vars"]["x"]["intent"] == ["out"]


def test_fixed_form_continuation_after_trailing_directive(tmp_path):
    mod = _routine(
        "      subroutine foo(x, y)\n"
        "      integer x !f2py intent(out) x\n"
        "     & , y\n"
        "      end\n",
        tmp_path,
        "fixcont.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["x", "y"]
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["vars"]["y"]["typespec"] == "integer"
    assert mod["vars"]["x"]["intent"] == ["out"]
    assert "intent" not in mod["vars"]["y"]


def test_fixed_form_skip_before_column6_continuation(tmp_path):
    gaps = (
        "C comment\n",
        "c comment\n",
        "! comment\n",
        "# comment\n",
        "* comment\n",
        "\n",
        "      ! note\n",
        "C comment\n\n",
    )
    for gap in gaps:
        mod = _routine(
            "      subroutine foo(x, y)\n"
            "      integer x !f2py intent(out) x\n"
            + gap
            + "     & , y\n"
            "      end\n",
            tmp_path,
            "fixskip.f",
        )
        assert mod["vars"]["x"]["typespec"] == "integer"
        assert mod["vars"]["y"]["typespec"] == "integer"
        assert mod["vars"]["x"]["intent"] == ["out"]
        assert "intent" not in mod["vars"]["y"]

    mod = _routine(
        "      subroutine foo(x, y, z)\n"
        "      integer x,\n"
        "     & y !f2py intent(out) x\n"
        "C comment\n"
        "     & ,z\n"
        "      end\n",
        tmp_path,
        "fixmid.f",
    )
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["vars"]["y"]["typespec"] == "integer"
    assert mod["vars"]["z"]["typespec"] == "integer"
    assert mod["vars"]["x"]["intent"] == ["out"]
    assert "intent" not in mod["vars"]["y"]
    assert "intent" not in mod["vars"]["z"]

    mod = _routine(
        "      subroutine foo(a, b, c)\n"
        "      integer a,\n"
        "     & b, !f2py intent(out) a\n"
        "\n"
        "     1 c\n"
        "      end\n",
        tmp_path,
        "fixdigit.f",
    )
    assert mod["vars"]["a"]["typespec"] == "integer"
    assert mod["vars"]["b"]["typespec"] == "integer"
    assert mod["vars"]["c"]["typespec"] == "integer"
    assert mod["vars"]["a"]["intent"] == ["out"]
    assert "intent" not in mod["vars"]["b"]
    assert "intent" not in mod["vars"]["c"]


def test_free_form_trailing_directive_at_eof(tmp_path):
    mod = _routine(
        "subroutine foo(x)\n"
        "  integer :: x & !f2py intent(out) x\n",
        tmp_path,
        "eofcont.f90",
    )
    assert mod["name"] == "foo"
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["vars"]["x"]["intent"] == ["out"]


def test_fixed_form_trailing_directive_on_continued_line(tmp_path):
    mod = _routine(
        "      subroutine foo(x, y, z)\n"
        "      integer x,\n"
        "     & y !f2py intent(out) x\n"
        "     & ,z\n"
        "      end\n",
        tmp_path,
        "fixmid.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["x", "y", "z"]
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["vars"]["y"]["typespec"] == "integer"
    assert mod["vars"]["z"]["typespec"] == "integer"
    assert mod["vars"]["x"]["intent"] == ["out"]
    assert "intent" not in mod["vars"]["y"]
    assert "intent" not in mod["vars"]["z"]


def test_fixed_form_digit_column_continuation(tmp_path):
    mod = _routine(
        "      subroutine foo(a, b, c)\n"
        "      integer a,\n"
        "     & b, !f2py intent(out) a\n"
        "     1 c\n"
        "      end\n",
        tmp_path,
        "fixdigit.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a", "b", "c"]
    assert mod["vars"]["a"]["typespec"] == "integer"
    assert mod["vars"]["b"]["typespec"] == "integer"
    assert mod["vars"]["c"]["typespec"] == "integer"
    assert mod["vars"]["a"]["intent"] == ["out"]
    assert "intent" not in mod["vars"]["b"]
    assert "intent" not in mod["vars"]["c"]


def test_fixed_form_comment_with_f2py_text(tmp_path):
    comments = (
        "C!f2py intent(out) x\n",
        "c!f2py intent(out) x\n",
        "C !f2py intent(out) x\n",
        "CC !f2py intent(out) x\n",
        "* !f2py intent(out) x\n",
        "! note !f2py intent(out) x\n",
        "# note !f2py intent(out) x\n",
        "C!F2PY intent(out) x\n",
    )
    for comment in comments:
        mod = _routine(
            "      subroutine foo(x)\n"
            "      integer x\n"
            + comment
            + "      end\n",
            tmp_path,
            "fixcmt.f",
        )
        assert mod["name"] == "foo"
        assert mod["args"] == ["x"]
        assert mod["vars"]["x"]["typespec"] == "integer"
        assert "intent" not in mod["vars"]["x"]

    gaps = (
        "C note !f2py intent(in) y\n",
        "c note !f2py intent(in) y\n",
        "C!f2py intent(in) y\n",
        "CC !f2py intent(in) y\n",
        "* !f2py intent(in) y\n",
        "! note !f2py intent(in) y\n",
        "# note !F2PY intent(in) y\n",
    )
    for gap in gaps:
        mod = _routine(
            "      subroutine foo(x, y)\n"
            "      integer x !f2py intent(out) x\n"
            + gap
            + "     & , y\n"
            "      end\n",
            tmp_path,
            "fixcmtcont.f",
        )
        assert mod["name"] == "foo"
        assert mod["args"] == ["x", "y"]
        assert mod["vars"]["x"]["typespec"] == "integer"
        assert mod["vars"]["y"]["typespec"] == "integer"
        assert mod["vars"]["x"]["intent"] == ["out"]
        assert "intent" not in mod["vars"]["y"]

    for directive in (
        "Cf2py intent(out) x\n",
        "cF2py intent(out) x\n",
        "*f2py intent(out) x\n",
        "#f2py intent(out) x\n",
        "!f2py intent(out) x\n",
    ):
        mod = _routine(
            "      subroutine foo(x)\n"
            "      integer x\n"
            + directive
            + "      end\n",
            tmp_path,
            "fixdirect.f",
        )
        assert mod["vars"]["x"]["typespec"] == "integer"
        assert mod["vars"]["x"]["intent"] == ["out"]


@pytest.mark.slow
class TestTrailingDirective(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine foo(x)
  integer :: x !f2py intent(out) x
  x = 7
end subroutine
"""

    def test_returned_value(self):
        assert self.module.foo() == 7
