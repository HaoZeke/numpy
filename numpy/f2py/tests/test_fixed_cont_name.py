import pytest

from numpy.f2py import crackfortran

from . import util


@pytest.mark.slow
@pytest.mark.parametrize("source", ["contname_amp.f", "contname_star.f"])
def test_fixed_continuation_joins_name(source):
    mod = util.build_module(
        [util.getpath("tests", "src", "crackfortran", source)])
    assert mod.foobar(3) == 10


@pytest.mark.slow
def test_fixed_continuation_keeps_joined_keyword():
    mod = util.build_module(
        [util.getpath("tests", "src", "crackfortran", "contname_keyword.f")])
    assert mod.dps() == 5.5
    assert mod.dcs() == 1 + 2j
    assert mod.dpf(1.5) == 3.0


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "value"),
    [("realx", 4), ("integerval", 5), ("doublex", 6),
     ("endval", 7), ("todo", 8), ("vec3d", 9), ("fres", 10), ("realent", 11)],
)
def test_fixed_continuation_joins_keyword_prefix_name(name, value):
    mod = util.build_module(
        [util.getpath(
            "tests", "src", "crackfortran", "contname_keyword_prefix.f")])
    assert getattr(mod, name)(3) == value


@pytest.mark.slow
def test_fixed_continuation_joins_keyword_prefix_args():
    mod = util.build_module(
        [util.getpath(
            "tests", "src", "crackfortran", "contname_keyword_prefix.f")])
    assert mod.farg(3, 4) == 7


def test_fixed_continuation_keeps_directive_and_virtual(tmp_path):
    src = tmp_path / "directives.f"
    src.write_text(
        "      subroutine foo(a, x)\n"
        "cf2py fortranname\n"
        "cf2py&bar\n"
        "cf2py integer, optional :: x\n"
        "cf2py required\n"
        "cf2py&x\n"
        "      integer a, x\n"
        "      virtual\n"
        "     &a(2)\n"
        "      end\n"
    )
    crackfortran.reset_global_f2py_vars()
    crackfortran.quiet = 1
    crackfortran.verbose = 0
    pyf = crackfortran.crack2fortran(crackfortran.crackfortran([str(src)]))
    assert "fortranname bar" in pyf
    assert "required" in pyf
    assert "dimension(2)" in pyf


def _read_fixed(tmp_path, text):
    src = tmp_path / "stmt.f"
    src.write_text(text)
    crackfortran.reset_global_f2py_vars()
    crackfortran.quiet = 1
    crackfortran.verbose = 0
    lines = []
    crackfortran.readfortrancode([str(src)], lambda line, n=0: lines.append(line))
    return [line.strip() for line in lines if line.strip()]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("      do\n     &i = 1, n\n", "do i = 1, n"),
        ("      do\n     &10 i = 1, n\n", "do 10 i = 1, n"),
        ("      save\n     &a, b\n", "save a, b"),
        ("      else\n     &if (a .lt. b) then\n", "else if (a .lt. b) then"),
        ("      go\n     &to 10\n", "go to 10"),
        ("      go to\n     &10\n", "go to 10"),
        ("      end\n     &do\n", "end do"),
        ("      x = 1; do\n     &i = 1, n\n", "x = 1; do i = 1, n"),
        ("      note(some\n     &text)\n", "note(some text)"),
        ("      real, note(a b) :: fo\n     &o\n", "real, note(a b) :: foo"),
    ],
)
def test_fixed_continuation_keeps_blank(tmp_path, text, expected):
    assert _read_fixed(tmp_path, text) == [expected]


def test_fixed_continuation_keeps_callback_result(tmp_path):
    src = tmp_path / "callback.f"
    src.write_text(
        "      subroutine s(f, n, r)\n"
        "      external f\n"
        "      real f\n"
        "      integer n\n"
        "      real r\n"
        "      integer i\n"
        "      r = 0\n"
        "      do\n"
        "     &i = 1, f(n)\n"
        "      r = r + 1\n"
        "      enddo\n"
        "      end\n"
    )
    crackfortran.reset_global_f2py_vars()
    crackfortran.quiet = 1
    crackfortran.verbose = 0
    pyf = crackfortran.crack2fortran(crackfortran.crackfortran([str(src)]))
    assert "function f(n) ! in" in pyf
    assert "result" not in pyf
