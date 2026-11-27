import contextlib
import io
import re

import pytest

import numpy as np
from numpy.f2py import crackfortran
from numpy.f2py.capi_maps import getstrlength
from numpy.testing import assert_array_equal, assert_equal

from . import util


@pytest.mark.slow
class TestStarParamLen(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine fill(c, m)
        integer, parameter :: n = 4
        integer, intent(in) :: m
        character*(n), intent(out) :: c(m)
        c = 'abcd'
      end subroutine fill

      subroutine kindlen(c)
        character*(4_4), intent(out) :: c
        c = 'wxyz'
      end subroutine kindlen

      module hoststar
        integer, parameter :: n = 4
      contains
        function word() result(s)
          character*(n) :: s
          s = 'wxyz'
        end function word

        subroutine g(n, k, c)
          integer, intent(in) :: n, k
          character*(n), intent(out) :: c(k)
          c = 'abcdef'
        end subroutine g

        subroutine gs(n, s)
          integer, intent(in) :: n
          character*(n), intent(out) :: s
          s = 'abcdef'
        end subroutine gs
      end module hoststar
    """

    @pytest.mark.parametrize("case", ["fill", "word"])
    def test_parameter_length(self, case):
        if case == "fill":
            out = self.module.fill(3)
            assert out.shape == (3,)
            assert out.dtype == np.dtype("S4")
            assert_array_equal(out, np.array([b"abcd", b"abcd", b"abcd"]))
        else:
            got = self.module.hoststar.word()
            assert_equal(got, b"wxyz")
            assert np.asarray(got).dtype == np.dtype("S4")

    def test_local_name_hides_module_length(self):
        out = self.module.hoststar.g(6, 2)
        assert out.shape == (2,)
        assert out.dtype == np.dtype("S6")
        assert_array_equal(out, np.array([b"abcdef", b"abcdef"]))

    def test_scalar_local_name_hides_module_length(self):
        got = self.module.hoststar.gs(6)
        assert_equal(got, b"abcdef")
        assert "string(len=(n))" in self.module.hoststar.gs.__doc__

    def test_parenthesized_kind_length(self):
        got = self.module.kindlen()
        assert_equal(got, b"wxyz")
        doc = self.module.kindlen.__doc__
        assert re.search(r"string\(len=4\)(?!\d)", doc)


@pytest.mark.slow
class TestLenSelectorParam(util.F2PyTest):
    suffix = ".f90"
    code = """
      module hostlen
        integer, parameter :: n = 4
      contains
        subroutine glen(c, m)
          integer, intent(in) :: m
          character(len=n), intent(out) :: c(m)
          c = 'abcd'
        end subroutine glen
      end module hostlen
    """

    def test_module_len_selector(self):
        out = self.module.hostlen.glen(3)
        assert out.shape == (3,)
        assert out.dtype == np.dtype("S4")
        assert_array_equal(out, np.array([b"abcd", b"abcd", b"abcd"]))


def _blocks_by_name(blocks, found=None):
    if found is None:
        found = {}
    for block in blocks:
        if isinstance(block, dict):
            name = block.get("name")
            if name:
                found[name] = block
            _blocks_by_name(block.get("body") or [], found)
    return found


class TestKindExprFold:
    src = """
      subroutine star_sum(c)
        character*(4_4+1), intent(out) :: c
      end subroutine star_sum

      subroutine len_sum(c)
        character(len=(4_4+1)), intent(out) :: c
      end subroutine len_sum

      subroutine star_mul(c)
        character*(4_8*2), intent(out) :: c
      end subroutine star_mul

      subroutine len_paren(c)
        character(len=(4_4)+1), intent(out) :: c
      end subroutine len_paren

      subroutine bare_kind(c)
        character*(4_4), intent(out) :: c
      end subroutine bare_kind

      subroutine paren_kind(c)
        character(len=(4_4)), intent(out) :: c
      end subroutine paren_kind

      subroutine init_sum(k)
        integer :: k = (4_4)+1
      end subroutine init_sum

      subroutine init_mul(k)
        integer :: k = (8_4)*2
      end subroutine init_mul

      subroutine hide_len(n, s)
        integer, intent(in) :: n
        character*(n), intent(out) :: s
      end subroutine hide_len

      subroutine star_named(c)
        character*(4_wp+1), intent(out) :: c
      end subroutine star_named

      subroutine len_named(c)
        character(len=(4_wp)+1), intent(out) :: c
      end subroutine len_named

      subroutine star_div(c)
        character*(9_4/2), intent(out) :: c
      end subroutine star_div

      subroutine star_half(c)
        character*(8_4/2), intent(out) :: c
      end subroutine star_half

      subroutine init_named(k)
        integer :: k = (4_wp)+1
      end subroutine init_named

      subroutine init_div(k)
        integer :: k = 9_4/2
      end subroutine init_div

      subroutine init_call(k)
        integer :: k = (1_selected_int_kind(8))+1
      end subroutine init_call

      subroutine init_inside(k)
        integer :: k = (4_wp+1)
      end subroutine init_inside

      subroutine init_wpone(k)
        integer :: k = 1_wp+1
      end subroutine init_wpone
    """

    def _crack(self, tmp_path):
        path = tmp_path / "kind_expr.f90"
        path.write_text(self.src)
        with contextlib.redirect_stdout(io.StringIO()):
            blocks = crackfortran.crackfortran([str(path)])
        return _blocks_by_name(blocks)

    def test_kind_expr_character_length(self, tmp_path):
        got = self._crack(tmp_path)
        assert got["star_sum"]["vars"]["c"]["charselector"]["*"] == "5"
        assert got["len_sum"]["vars"]["c"]["charselector"]["*"] == "5"
        assert got["star_mul"]["vars"]["c"]["charselector"]["*"] == "8"

    def test_parenthesized_kind_expr(self, tmp_path):
        got = self._crack(tmp_path)
        assert got["len_paren"]["vars"]["c"]["charselector"]["*"] == "5"
        assert got["init_sum"]["vars"]["k"]["="] == "5"
        assert got["init_mul"]["vars"]["k"]["="] == "16"
        assert got["bare_kind"]["vars"]["c"]["charselector"]["*"] == "4"
        assert got["paren_kind"]["vars"]["c"]["charselector"]["*"] == "4"
        assert got["hide_len"]["vars"]["s"]["charselector"]["*"] == "(n)"
        ev = crackfortran._eval_scalar
        assert ev("(4_4)+1", {}) == "5"
        assert ev("(8_4)*2", {}) == "16"
        assert ev("(4_4)", {}) == "4"
        assert ev("((4_4))", {}) == "4"
        assert ev("(2_8*3)", {}) == "6"
        assert ev("(4_8 + 1)", {}) == "5"
        assert ev("4_4", {}) == "4"
        assert ev("1.0_8", {}) == "1.0"
        assert ev("3.14_dp", {}) == "3.14"
        assert ev("(n)", {}) == "(n)"
        assert ev('"4_4"', {}) == "'4_4'"

    def test_whole_kind_suffix(self, tmp_path):
        got = self._crack(tmp_path)
        assert got["star_named"]["vars"]["c"]["charselector"]["*"] == "5"
        assert got["len_named"]["vars"]["c"]["charselector"]["*"] == "5"
        assert got["star_div"]["vars"]["c"]["charselector"]["*"] == "4"
        assert got["star_half"]["vars"]["c"]["charselector"]["*"] == "4"
        assert got["init_named"]["vars"]["k"]["="] == "5"
        assert got["init_div"]["vars"]["k"]["="] == "4"
        assert got["init_call"]["vars"]["k"]["="] == "2"
        assert got["init_inside"]["vars"]["k"]["="] == "5"
        assert got["init_wpone"]["vars"]["k"]["="] == "2"
        ev = crackfortran._eval_scalar
        assert ev("(4_wp)+1", {}) == "5"
        assert ev("(8_wp)*2", {}) == "16"
        assert ev("4_wp", {}) == "4"
        assert ev("1_wp+1", {}) == "2"
        assert ev("9_4/2", {}) == "4"
        assert ev("8_4/2", {}) == "4"
        assert ev("1.0_8", {}) == "1.0"
        assert ev("2.5_8", {}) == "2.5"
        assert ev("1.0e2_8", {}) == "100.0"
        assert ev("3.14_dp", {}) == "3.14"
        assert ev("1_selected_int_kind(8)", {}) == "1"
        assert ev("(1_selected_int_kind(8))+1", {}) == "2"
        assert ev("(1_selected_int_kind(8)+1)", {}) == "2"

    def test_integer_division(self, tmp_path):
        src = """
      subroutine negdiv(k)
        integer :: k = -9_4/2
      end subroutine negdiv

      subroutine pardiv(k)
        integer :: k = (9_4)/(2_4)
      end subroutine pardiv

      subroutine sumdiv(k)
        integer :: k = (4_4+1)/2
      end subroutine sumdiv

      subroutine leaddiv(k)
        integer :: k = 09/2
      end subroutine leaddiv

      subroutine parenneg(k)
        integer :: k = (-9_4)/2
      end subroutine parenneg

      subroutine negright(k)
        integer :: k = 8_4/(-2)
      end subroutine negright

      subroutine negfour(k)
        integer :: k = (-4_4)/2
      end subroutine negfour

      subroutine mulreal(x)
        real :: x = 1.5*9/2
      end subroutine mulreal

      subroutine onedot(x)
        real :: x = 1.0*9/2
      end subroutine onedot

      subroutine kindmul(x)
        real :: x = 1.0*9_4/2
      end subroutine kindmul

      subroutine powdiv(x)
        real :: x = 2.0**2/3
      end subroutine powdiv
    """
        path = tmp_path / "intdiv.f90"
        path.write_text(src)
        with contextlib.redirect_stdout(io.StringIO()):
            blocks = crackfortran.crackfortran([str(path)])
        got = _blocks_by_name(blocks)
        assert got["negdiv"]["vars"]["k"]["="] == "-4"
        assert got["pardiv"]["vars"]["k"]["="] == "4"
        assert got["sumdiv"]["vars"]["k"]["="] == "2"
        assert got["leaddiv"]["vars"]["k"]["="] == "4"
        assert got["parenneg"]["vars"]["k"]["="] == "-4"
        assert got["negright"]["vars"]["k"]["="] == "-4"
        assert got["negfour"]["vars"]["k"]["="] == "-2"
        assert got["mulreal"]["vars"]["x"]["="] == "6.75"
        assert got["onedot"]["vars"]["x"]["="] == "4.5"
        assert got["kindmul"]["vars"]["x"]["="] == "4.5"
        assert got["powdiv"]["vars"]["x"]["="] == "1.3333333333333333"
        ev = crackfortran._eval_scalar
        assert ev("-9_4/2", {}) == "-4"
        assert ev("(9_4)/(2_4)", {}) == "4"
        assert ev("(4_4+1)/2", {}) == "2"
        assert ev("09/2", {}) == "4"
        assert ev("(-9_4)/2", {}) == "-4"
        assert ev("8_4/(-2)", {}) == "-4"
        assert ev("(-4_4)/2", {}) == "-2"
        assert ev("9_4/(-2_4)", {}) == "-4"
        assert ev("1.5*9/2", {}) == "6.75"
        assert ev("1.0*9/2", {}) == "4.5"
        assert ev("1.0*9_4/2", {}) == "4.5"
        assert ev("1.0_8*9_4/2", {}) == "4.5"
        assert ev("2.0**2/3", {}) == "1.3333333333333333"
        assert ev("1.0/2", {}) == "0.5"
        assert ev("9/2*1.5", {}) == "6.0"
        assert ev("1.5*(9/2)", {}) == "6.0"
        assert ev("(1.5*9)/2", {}) == "6.75"
        assert ev("1.0+9/2", {}) == "5.0"
        assert ev("9_4/2", {}) == "4"
        assert ev("a/2", {"a": 9}) == "4"
        assert ev("a/2", {"a": 9.0}) == "4.5"
        assert ev("'9/2'", {}) == "'9/2'"


class TestParamValueDiv:
    def _crack(self, tmp_path, src):
        path = tmp_path / "paramdiv.f90"
        path.write_text(src)
        with contextlib.redirect_stdout(io.StringIO()):
            blocks = crackfortran.crackfortran([str(path)])
        return _blocks_by_name(blocks)

    def test_parameter_kind_and_shape(self, tmp_path):
        src = """
      subroutine kindlen(c)
        integer, parameter :: n = 9_4/2
        character*(n), intent(out) :: c
      end subroutine kindlen

      subroutine parlen(c)
        integer, parameter :: n = (9_4)/(2_4)
        character*(n), intent(out) :: c
      end subroutine parlen

      subroutine dimen(a)
        integer, parameter :: n = 9/2
        integer :: a(n)
      end subroutine dimen
    """
        got = self._crack(tmp_path, src)
        assert got["kindlen"]["vars"]["c"]["charselector"]["*"] == "4"
        assert got["parlen"]["vars"]["c"]["charselector"]["*"] == "4"
        assert got["dimen"]["vars"]["a"]["dimension"] == ["4"]

    def test_same_block_parameter_lengths(self, tmp_path):
        src = """
      subroutine s(c, d, e)
        integer, parameter :: n = 9/2
        integer, parameter :: m = -9_4/2
        integer, parameter :: k = 09/2
        character*(n), intent(out) :: c
        character*(m), intent(out) :: d
        character*(k), intent(out) :: e
      end subroutine s
    """
        got = self._crack(tmp_path, src)
        assert got["s"]["vars"]["c"]["charselector"]["*"] == "4"
        assert got["s"]["vars"]["d"]["charselector"]["*"] == "-4"
        assert got["s"]["vars"]["e"]["charselector"]["*"] == "4"

    def test_signed_folded_length(self, tmp_path):
        src = """
      function fneg() result(c)
        integer, parameter :: n = -9/2
        character*(n) :: c
      end function fneg

      function fkind() result(s)
        integer, parameter :: m = -9_4/2
        character*(m) :: s
      end function fkind

      function fres() result(s)
        integer, parameter :: m = -9_4/2
        character(len=m) :: s
      end function fres

      function fpos() result(c)
        integer, parameter :: n = 9/2
        character*(n) :: c
      end function fpos

      function fzero() result(c)
        integer, parameter :: n = 09/2
        character*(n) :: c
      end function fzero

      module hostneg
        integer, parameter :: m = -9/2
      contains
        subroutine inner(c)
          character*(m), intent(out) :: c
        end subroutine inner
      end module hostneg

      module hostkind
        integer, parameter :: m = -9_4/2
      contains
        subroutine innerk(c)
          character*(m), intent(out) :: c
        end subroutine innerk
      end module hostkind

      module hostpos
        integer, parameter :: m = 9/2
      contains
        subroutine innerp(c)
          character*(m), intent(out) :: c
        end subroutine innerp
      end module hostpos
    """
        got = self._crack(tmp_path, src)
        assert got["fneg"]["vars"]["c"]["charselector"]["*"] == "-4"
        assert getstrlength(got["fneg"]) == "-4"
        assert got["fkind"]["vars"]["s"]["charselector"]["*"] == "-4"
        assert getstrlength(got["fkind"]["vars"]["s"]) == "-4"
        assert got["fres"]["vars"]["s"]["charselector"]["len"] == "-4"
        assert getstrlength(got["fres"]) == "-4"
        assert got["fpos"]["vars"]["c"]["charselector"]["*"] == "4"
        assert got["fzero"]["vars"]["c"]["charselector"]["*"] == "4"
        assert got["inner"]["vars"]["c"]["charselector"]["*"] == "-4"
        assert getstrlength(got["inner"]["vars"]["c"]) == "-4"
        assert got["innerk"]["vars"]["c"]["charselector"]["*"] == "-4"
        assert got["innerp"]["vars"]["c"]["charselector"]["*"] == "4"

    def test_real_parameter_division(self, tmp_path):
        src = """
      subroutine pmix(x, y, z, w)
        real, parameter :: a = 9/2
        real, intent(in) :: x = a
        real, parameter :: b = 9/2*1.5
        real, intent(in) :: y = b
        real, parameter :: c = 1.5*9/2
        real, intent(in) :: z = c
        integer, parameter :: n = 9/2*1.5
        integer, intent(in) :: w = n
      end subroutine pmix
    """
        got = self._crack(tmp_path, src)
        assert got["pmix"]["vars"]["x"]["="] == "4.0"
        assert got["pmix"]["vars"]["y"]["="] == "6.0"
        assert got["pmix"]["vars"]["z"]["="] == "6.75"
        assert got["pmix"]["vars"]["w"]["="] == "6"

    def test_real_parameter_chained_division(self, tmp_path):
        src = """
      subroutine pdiv(y)
        real, parameter :: a = 9/2
        real, parameter :: b = a/3
        real, intent(in) :: y = b
      end subroutine pdiv
    """
        got = self._crack(tmp_path, src)
        assert got["pdiv"]["vars"]["y"]["="] == "1.3333333333333333"

    def test_real_parameter_name_quotient(self, tmp_path):
        src = """
      subroutine peight(x)
        real, parameter :: a = 8/2
        real, parameter :: b = a/3
        real, intent(in) :: x = b
      end subroutine peight

      subroutine precip(x)
        real, parameter :: a = 9/2
        real, parameter :: b = 1/a
        real, intent(in) :: x = b
      end subroutine precip
    """
        got = self._crack(tmp_path, src)
        assert got["peight"]["vars"]["x"]["="] == "1.3333333333333333"
        assert got["precip"]["vars"]["x"]["="] == "0.25"

    def test_integer_parameter_truncation(self, tmp_path):
        src = """
      subroutine trunc(c, a, im, ip, iq, k)
        integer, parameter :: n = 1.5*9/2
        character*(n), intent(out) :: c
        integer :: a(n)
        integer, parameter :: m = 5/2*1.1
        integer, intent(in) :: im = m
        integer, parameter :: p = 5.0/2
        integer, intent(in) :: ip = p
        integer, parameter :: q = -5.0/2
        integer, intent(in) :: iq = q
        integer :: k = -5.0/2
      end subroutine trunc
    """
        got = self._crack(tmp_path, src)
        assert got["trunc"]["vars"]["c"]["charselector"]["*"] == "6"
        assert got["trunc"]["vars"]["a"]["dimension"] == ["6"]
        assert got["trunc"]["vars"]["im"]["="] == "2"
        assert got["trunc"]["vars"]["ip"]["="] == "2"
        assert got["trunc"]["vars"]["iq"]["="] == "-2"
        assert got["trunc"]["vars"]["k"]["="] == "-2"

    def test_host_integer_parameter_length(self, tmp_path):
        src = """
      module hostsix
        integer, parameter :: n = 9/2*1.5
      contains
        subroutine innersix(c)
          character*(n), intent(out) :: c
        end subroutine innersix
      end module hostsix

      subroutine samesix(c)
        integer, parameter :: n = 9/2*1.5
        character*(n), intent(out) :: c
      end subroutine samesix
    """
        got = self._crack(tmp_path, src)
        assert got["hostsix"]["vars"]["n"]["="] == "6"
        assert got["innersix"]["vars"]["c"]["charselector"]["*"] == "6"
        assert getstrlength(got["innersix"]["vars"]["c"]) == "6"
        assert got["samesix"]["vars"]["c"]["charselector"]["*"] == "6"

    def test_real_parameter_int_value(self, tmp_path):
        src = """
      subroutine p4(y, z)
        real, parameter :: a = 4
        real, parameter :: b = a/3
        real, intent(in) :: y = b
        real, intent(in) :: z = a/3
      end subroutine p4

      subroutine pprod(y)
        real, parameter :: a = 2*2
        real, intent(in) :: y = a/3
      end subroutine pprod

      subroutine pdot(y)
        real, parameter :: a = 4.0
        real, intent(in) :: y = a/3
      end subroutine pdot

      subroutine pdbl(y)
        double precision, parameter :: a = 4
        double precision, intent(in) :: y = a/3
      end subroutine pdbl

      subroutine phalf(y)
        real, parameter :: a = 4
        real, intent(in) :: y = a/2
      end subroutine phalf

      subroutine pint(y, z)
        integer, parameter :: n = 4
        real, intent(in) :: y = n/3
        integer, parameter :: m = 5
        real, intent(in) :: z = m/2
      end subroutine pint

      subroutine pfive(y, z, x)
        real, parameter :: a = 5
        real, parameter :: b = a/2
        real, intent(in) :: y = b
        real, intent(in) :: z = 1/a
        integer :: x(a)
      end subroutine pfive

      subroutine pdbl5(y)
        double precision, parameter :: a = 5
        double precision, intent(in) :: y = a/2
      end subroutine pdbl5

      subroutine pkind(x, a)
        real, parameter :: r8 = 1.0d0
        real, parameter :: r4 = 4.0
        real(kind=r8) :: x
        integer :: a(r4)
      end subroutine pkind
    """
        got = self._crack(tmp_path, src)
        assert got["p4"]["vars"]["y"]["="] == "1.3333333333333333"
        assert got["p4"]["vars"]["z"]["="] == "1.3333333333333333"
        assert got["pprod"]["vars"]["y"]["="] == "1.3333333333333333"
        assert got["pdot"]["vars"]["y"]["="] == "1.3333333333333333"
        assert got["pdbl"]["vars"]["y"]["="] == "1.3333333333333333"
        assert got["phalf"]["vars"]["y"]["="] == "2.0"
        assert got["pint"]["vars"]["y"]["="] == "1"
        assert got["pint"]["vars"]["z"]["="] == "2"
        assert got["pfive"]["vars"]["y"]["="] == "2.5"
        assert got["pfive"]["vars"]["z"]["="] == "0.2"
        assert got["pfive"]["vars"]["x"]["dimension"] == ["5"]
        assert got["pdbl5"]["vars"]["y"]["="] == "2.5"
        assert got["pkind"]["vars"]["x"]["kindselector"]["kind"] == "8"
        assert got["pkind"]["vars"]["a"]["dimension"] == ["4"]

    def test_complex_parameter_division(self, tmp_path):
        src = """
      subroutine cdiv(y)
        complex, parameter :: a = 4
        complex, intent(in) :: y = a/3
      end subroutine cdiv

      subroutine zrec(z)
        double complex, parameter :: b = 5
        double complex, intent(in) :: z = 1/b
      end subroutine zrec

      subroutine zdiv(y)
        double complex, parameter :: a = 5
        double complex, intent(in) :: y = a/2
      end subroutine zdiv
    """
        got = self._crack(tmp_path, src)
        assert got["cdiv"]["vars"]["y"]["="] == "1.3333333333333333"
        assert got["zrec"]["vars"]["z"]["="] == "0.2"
        assert got["zdiv"]["vars"]["y"]["="] == "2.5"

    def test_integer_parameter_from_real(self, tmp_path):
        src = """
      module hostfold
        real, parameter :: a = 9/2*1.5
        integer, parameter :: n = a
        character*(n) :: c
      contains
        subroutine innerfold(s)
          character*(n), intent(out) :: s
        end subroutine innerfold
      end module hostfold

      subroutine samefold(c)
        real, parameter :: a = 9/2*1.5
        integer, parameter :: n = a
        character*(n), intent(out) :: c
      end subroutine samefold
    """
        got = self._crack(tmp_path, src)
        assert got["hostfold"]["vars"]["a"]["="] == "6.0"
        assert got["hostfold"]["vars"]["n"]["="] == "6"
        assert got["hostfold"]["vars"]["c"]["charselector"]["*"] == "6"
        assert got["innerfold"]["vars"]["s"]["charselector"]["*"] == "6"
        assert getstrlength(got["innerfold"]["vars"]["s"]) == "6"
        assert got["samefold"]["vars"]["c"]["charselector"]["*"] == "6"
        assert getstrlength(got["samefold"]["vars"]["c"]) == "6"
