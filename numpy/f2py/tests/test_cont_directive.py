import pytest

from numpy.f2py import crackfortran

from . import util


def _routine(src, tmp_path, name):
    path = tmp_path / name
    path.write_text(src)
    crackfortran.reset_global_f2py_vars()
    crackfortran.verbose = 0
    return crackfortran.crackfortran([str(path)])[0]


def test_free_directive_between_continuations(tmp_path):
    mod = _routine(
        "subroutine foo(n)\n"
        "  integer :: &\n"
        "  !f2py intent(out) n\n"
        "    n\n"
        "  n = 5\n"
        "end subroutine\n",
        tmp_path,
        "cont.f90",
    )
    assert mod["name"] == "foo"
    assert mod["vars"]["n"]["typespec"] == "integer"
    assert mod["vars"]["n"]["intent"] == ["out"]


def test_free_continued_directive_inside_declaration(tmp_path):
    mod = _routine(
        "subroutine foo(n)\n"
        "  integer :: &\n"
        "  !f2py intent(out) &\n"
        "  !f2py n\n"
        "    n\n"
        "  n = 5\n"
        "end subroutine\n",
        tmp_path,
        "cont_dir.f90",
    )
    assert mod["name"] == "foo"
    assert mod["vars"]["n"]["typespec"] == "integer"
    assert mod["vars"]["n"]["intent"] == ["out"]
    assert "intent" not in mod["vars"]


def test_free_dangling_intent(tmp_path):
    mod = _routine(
        "subroutine foo(n, m)\n"
        "  integer :: &\n"
        "  !f2py intent(out) &\n"
        "    n\n"
        "  integer :: m\n"
        "end subroutine\n",
        tmp_path,
        "dangling.f90",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["n", "m"]
    assert mod["vars"]["n"]["typespec"] == "integer"
    assert mod["vars"]["m"]["typespec"] == "integer"
    assert "intent" not in mod["vars"]["n"]
    assert "intent" not in mod["vars"]["m"]
    assert "intent" not in mod["vars"]


def test_fixed_directive_before_column6_continuation(tmp_path):
    mod = _routine(
        "      subroutine foo(a\n"
        "!f2py intent(out) a\n"
        "     &, b)\n"
        "      integer a, b\n"
        "      end\n",
        tmp_path,
        "cont_col6.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a", "b"]
    assert mod["vars"]["a"]["intent"] == ["out"]
    assert mod["vars"]["b"]["typespec"] == "integer"


def test_fixed_directive_on_its_own_line(tmp_path):
    mod = _routine(
        "      subroutine foo(a)\n"
        "!f2py intent(out) a\n"
        "      integer a\n"
        "      end\n",
        tmp_path,
        "own.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a"]
    assert mod["vars"]["a"]["intent"] == ["out"]


def test_free_inline_directive_on_continued_line(tmp_path):
    mod = _routine(
        "subroutine foo(n)\n"
        "  integer :: &\n"
        "  n !f2py intent(out) n\n"
        "  n = 5\n"
        "end subroutine\n",
        tmp_path,
        "inline.f90",
    )
    assert mod["name"] == "foo"
    assert mod["vars"]["n"]["typespec"] == "integer"
    assert mod["vars"]["n"]["intent"] == ["out"]
    assert "=" not in mod["vars"]["n"]


def test_fixed_directive_between_continuations(tmp_path):
    mod = _routine(
        "      subroutine foo(a,\n"
        "!f2py intent(out) a\n"
        "     & b)\n"
        "      integer a, b\n"
        "      end\n",
        tmp_path,
        "cont.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a", "b"]
    assert mod["vars"]["a"]["intent"] == ["out"]
    assert mod["vars"]["b"]["typespec"] == "integer"


def test_free_continued_callstatement(tmp_path):
    mod = _routine(
        "subroutine foo(x)\n"
        "  integer :: x\n"
        "!f2py callstatement (*f2py_func)(&\n"
        "!f2py &X);\n"
        "  x = 1\n"
        "end subroutine\n",
        tmp_path,
        "call_cont.f90",
    )
    assert mod["name"] == "foo"
    assert mod["f2pyenhancements"]["callstatement"] == "(*f2py_func)(x);"
    assert "=" not in mod["vars"]["x"]


def test_free_inline_callstatement_before_directive(tmp_path):
    mod = _routine(
        "subroutine foo(x)\n"
        "  integer :: &\n"
        "    x !f2py callstatement Py_INCREF(cap); (*f2py_func)(&X);\n"
        "  !f2py callprotoargument int*\n"
        "end subroutine\n",
        tmp_path,
        "inline_call.f90",
    )
    assert mod["name"] == "foo"
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["f2pyenhancements"]["callstatement"] == (
        "Py_INCREF(cap); (*f2py_func)(&X);"
    )
    assert mod["f2pyenhancements"]["callprotoargument"] == "int*"


def test_free_deferred_callstatement(tmp_path):
    mod = _routine(
        "subroutine foo(x)\n"
        "  integer :: &\n"
        "  !f2py callstatement (*f2py_func)(X);\n"
        "  x\n"
        "end subroutine\n",
        tmp_path,
        "defer_call.f90",
    )
    assert mod["name"] == "foo"
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["f2pyenhancements"]["callstatement"] == "(*f2py_func)(x);"


def test_free_deferred_amp_callstatement(tmp_path):
    mod = _routine(
        "subroutine foo(x)\n"
        "  integer :: &\n"
        "  !f2py callstatement (*f2py_func)(&X);\n"
        "    x\n"
        "end subroutine\n",
        tmp_path,
        "defer_amp.f90",
    )
    assert mod["name"] == "foo"
    assert mod["vars"]["x"]["typespec"] == "integer"
    assert mod["f2pyenhancements"]["callstatement"] == "(*f2py_func)(&x);"


def test_fixed_closed_directive_column6(tmp_path):
    mod = _routine(
        "      subroutine foo(a)\n"
        "!f2py intent(out)\n"
        "     & a\n"
        "      integer a\n"
        "      end\n",
        tmp_path,
        "closed_col6.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a"]
    assert mod["vars"]["a"]["intent"] == ["out"]


def test_fixed_inline_directive_keeps_argument(tmp_path):
    mod = _routine(
        "      subroutine foo(a,\n"
        "     & b !f2py intent(in) b\n"
        "     & ,c)\n"
        "      integer a, b, c\n"
        "      end\n",
        tmp_path,
        "inline_col6.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a", "b", "c"]
    assert mod["vars"]["b"]["intent"] == ["in"]
    assert mod["vars"]["a"]["typespec"] == "integer"
    assert mod["vars"]["c"]["typespec"] == "integer"


def test_free_callstatement_continued_directive(tmp_path):
    mod = _routine(
        "subroutine foo\n"
        "!f2py callstatement (*f2py_func)(& &\n"
        "!f2py &X);\n"
        "end subroutine\n",
        tmp_path,
        "amp.f90",
    )
    assert mod["name"] == "foo"
    assert mod["f2pyenhancements"]["callstatement"] == "(*f2py_func)(& x);"


@pytest.mark.parametrize("comment", ["C keep going", "! keep going"])
def test_fixed_comment_between_directive_and_column6(tmp_path, comment):
    src = (
        "      subroutine foo(a\n"
        "!f2py intent(out) a\n"
        + comment + "\n"
        "     &, b)\n"
        "      integer a, b\n"
        "      end\n"
    )
    mod = _routine(src, tmp_path, "comment_col6.f")
    assert mod["name"] == "foo"
    assert mod["args"] == ["a", "b"]
    assert mod["vars"]["a"]["intent"] == ["out"]
    assert mod["vars"]["b"]["typespec"] == "integer"


def test_fixed_directive_comma_before_column6(tmp_path):
    mod = _routine(
        "      subroutine foo(a\n"
        "!f2py intent(out),\n"
        "     &, b)\n"
        "      integer a, b\n"
        "      end\n",
        tmp_path,
        "comma_col6.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a", "b"]
    assert mod["vars"]["a"]["typespec"] == "integer"
    assert mod["vars"]["b"]["typespec"] == "integer"


def test_fixed_data_slash_keeps_directive(tmp_path):
    mod = _routine(
        "      subroutine foo(a, b)\n"
        "      data a /1/\n"
        "!f2py intent(out)\n"
        "     & b\n"
        "      integer a, b\n"
        "      end\n",
        tmp_path,
        "data_slash.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a", "b"]
    assert mod["vars"]["a"]["="] == "1"
    assert mod["vars"]["a"]["typespec"] == "integer"
    assert "intent" not in mod["vars"]["a"]
    assert mod["vars"]["b"]["intent"] == ["out"]
    assert mod["vars"]["b"]["typespec"] == "integer"


def test_fixed_common_slash_keeps_directive(tmp_path):
    mod = _routine(
        "      subroutine foo(a)\n"
        "      common /blk/\n"
        "!f2py intent(in)\n"
        "     & a\n"
        "      integer a\n"
        "      end\n",
        tmp_path,
        "common_slash.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a"]
    assert mod["common"]["blk"] == []
    assert mod["commonvars"] == []
    assert mod["vars"]["a"]["intent"] == ["in"]
    assert mod["vars"]["a"]["typespec"] == "integer"


def test_fixed_data_nospace_slash_keeps_directive(tmp_path):
    mod = _routine(
        "      subroutine foo(a, b)\n"
        "      data x/1/\n"
        "!f2py intent(out)\n"
        "     & a\n"
        "      integer a, b, x\n"
        "      end\n",
        tmp_path,
        "data_nospace.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a", "b"]
    assert mod["vars"]["a"]["intent"] == ["out"]
    assert mod["vars"]["a"]["typespec"] == "integer"
    assert "intent" not in mod["vars"]["b"]
    assert mod["vars"]["b"]["typespec"] == "integer"


def test_fixed_quoted_data_slash_keeps_directive(tmp_path):
    mod = _routine(
        "      subroutine foo(a, b)\n"
        "      data a /'/'/\n"
        "!f2py intent(out)\n"
        "     & b\n"
        "      character a\n"
        "      integer b\n"
        "      end\n",
        tmp_path,
        "data_quoted.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a", "b"]
    assert mod["vars"]["a"]["="] == "'/'"
    assert mod["vars"]["a"]["typespec"] == "character"
    assert "intent" not in mod["vars"]["a"]
    assert mod["vars"]["b"]["intent"] == ["out"]


def test_fixed_open_data_slash_takes_value(tmp_path):
    mod = _routine(
        "      subroutine foo(a, b)\n"
        "      data a /\n"
        "!f2py intent(out) a\n"
        "     & 1/\n"
        "      integer a, b\n"
        "      end\n",
        tmp_path,
        "data_open.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a", "b"]
    assert mod["vars"]["a"]["="] == "1"
    assert mod["vars"]["a"]["intent"] == ["out"]
    assert "intent" not in mod["vars"]["b"]


def test_fixed_concat_slash_stays_open(tmp_path):
    mod = _routine(
        "      subroutine foo(a)\n"
        "      a = 'x' //\n"
        "!f2py intent(in)\n"
        "     & 'y'\n"
        "      character*(*) a\n"
        "      end\n",
        tmp_path,
        "concat_slash.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a"]
    assert mod["vars"]["a"]["typespec"] == "character"
    assert mod["vars"]["a"]["intent"] == ["in"]


def test_fixed_division_slash_stays_open(tmp_path):
    samples = (
        (
            "div_ab.f",
            "      subroutine foo(x)\n"
            "      x = a/b/\n"
            "!f2py intent(inout) x\n"
            "     & c\n"
            "      integer x\n"
            "      end\n",
            "x = a/b/ c",
            "intent(inout) x",
        ),
        (
            "div_num.f",
            "      subroutine foo(a, b)\n"
            "      a = 1/2/\n"
            "!f2py intent(in) b\n"
            "     & 3\n"
            "      integer a, b\n"
            "      end\n",
            "a = 1/2/ 3",
            "intent(in) b",
        ),
    )
    for name, src, assignment, directive in samples:
        path = tmp_path / name
        path.write_text(src)
        found = []

        def collect(line, reset=0, found=found):
            if reset == 0 and line.strip():
                found.append(" ".join(line.split()))

        crackfortran.reset_global_f2py_vars()
        crackfortran.verbose = 0
        crackfortran.readfortrancode([str(path)], dowithline=collect)
        assert assignment in found
        assert directive in found


def test_fixed_blank_common_keeps_directive(tmp_path):
    mod = _routine(
        "      subroutine foo(a)\n"
        "      common //\n"
        "!f2py intent(in)\n"
        "     & a\n"
        "      integer a\n"
        "      end\n",
        tmp_path,
        "blank_common.f",
    )
    assert mod["name"] == "foo"
    assert mod["args"] == ["a"]
    assert mod["common"]["_BLNK_"] == []
    assert mod["commonvars"] == []
    assert mod["vars"]["a"]["intent"] == ["in"]
    assert mod["vars"]["a"]["typespec"] == "integer"


@pytest.mark.slow
class TestFreeContDirective(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine foo(n)
  integer :: &
  !f2py intent(out) n
    n
  n = 5
end subroutine
"""

    def test_returned_value(self):
        assert self.module.foo() == 5


@pytest.mark.slow
class TestFixedContDirective(util.F2PyTest):
    suffix = ".f"
    code = """
      subroutine foo(a,
!f2py intent(out) a
     & b)
      integer a, b
      a = 1
      b = 2
      end
"""

    def test_returned_value(self):
        assert self.module.foo(2) == 1
