import pytest

import numpy as np
from numpy.f2py import crackfortran
from numpy.testing import assert_array_equal

from . import util


def test_selected_real_kind_tracks_compiler_and_flags(monkeypatch):
    queries = []

    def query(compiler, flags, p, r, radix):
        queries.append((compiler, flags, p, r, radix))
        return 2 if compiler == ("flang-23",) else 4

    monkeypatch.setattr(crackfortran, "_selected_real_kind_from_compiler", query)
    monkeypatch.setenv("FC", "flang-23")
    monkeypatch.setenv("FFLAGS", "-fdefault-real-8 -Dname='two words'")
    result = crackfortran._selected_real_kind_func(0)
    assert result == 2
    assert isinstance(result, crackfortran._Integer)
    assert result.kind == crackfortran._default_kinds["integer"]
    monkeypatch.setenv("FC", "gfortran")
    monkeypatch.setenv("FFLAGS", "-fdefault-real-8")
    assert crackfortran._selected_real_kind_func(6, 400, 2) == 4
    assert queries == [
        (("flang-23",), ("-fdefault-real-8", "-Dname=two words"), 0, None, None),
        (("gfortran",), ("-fdefault-real-8",), 6, 400, 2),
    ]


def test_selected_real_kind_keeps_optional_argument_presence(monkeypatch):
    queries = []

    def query(compiler, flags, p, r, radix):
        queries.append((p, r, radix))
        return -5 if radix == 10 else 10

    monkeypatch.setattr(crackfortran, "_selected_real_kind_from_compiler", query)
    monkeypatch.setenv("FC", "flang-23")
    assert crackfortran._selected_real_kind_func(r=400) == 10
    assert crackfortran._selected_real_kind_func(radix=10) == -5
    assert queries == [(None, 400, None), (None, None, 10)]


def test_selected_real_kind_does_not_guess_after_probe_failure(monkeypatch):
    def query(*args):
        raise ValueError("compiler probe failed")

    monkeypatch.setattr(crackfortran, "_selected_real_kind_from_compiler", query)
    monkeypatch.setenv("FC", "flang-23")
    with pytest.raises(ValueError, match="compiler probe failed"):
        crackfortran._selected_real_kind_func(0)


@pytest.mark.slow
class TestSelectedRealKindCompiler(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine kind_query(p, r, radix, result)
      implicit none
      integer, intent(in) :: p, r, radix
      integer, intent(out) :: result
      result = selected_real_kind(p, r, radix)
      end subroutine kind_query
    """

    @pytest.mark.parametrize("precision, exponent_range, radix", [
        (0, 0, 2),
        (6, 400, 2),
        (15, 400, 2),
        (33, 4931, 2),
        (40, 5000, 2),
        (0, 0, 10),
    ])
    def test_exact_compiler_intrinsic(self, precision, exponent_range, radix):
        expected = self.module.kind_query(precision, exponent_range, radix)
        actual = crackfortran._selected_real_kind_func(
            precision, exponent_range, radix)
        assert actual == expected


@pytest.mark.slow
class TestCompilerSelectedRealABI(util.F2PyTest):
    suffix = ".f90"
    code = """
      module compiler_selected_real
      implicit none
      integer, parameter :: rk = selected_real_kind(0)
      real(rk) :: stored = 1.5_rk
      contains
      subroutine twice(values)
        real(rk), intent(inout) :: values(2)
        values = values * 2
      end subroutine twice
      function half_value(value) result(result)
        real(rk), intent(in) :: value
        real(rk) :: result
        result = value / 2
      end function half_value
      end module compiler_selected_real
    """

    def test_selected_kind_matches_wrapper_storage(self):
        module = self.module.compiler_selected_real
        kind = int(module.rk)
        assert kind == crackfortran._selected_real_kind_func(0)
        expected_dtype = {2: np.dtype(np.float16), 4: np.dtype(np.float32)}[kind]
        assert module.stored.dtype == expected_dtype
        assert module.stored.itemsize == expected_dtype.itemsize
        assert module.stored == expected_dtype.type(1.5)

    def test_selected_kind_scalar_and_array_conversion(self):
        module = self.module.compiler_selected_real
        kind = int(module.rk)
        dtype = {2: np.dtype(np.float16), 4: np.dtype(np.float32)}[kind]
        values = np.array([1, 2], dtype=dtype)
        module.twice(values)
        assert_array_equal(values, [2, 4])
        assert values.tobytes() == np.array([2, 4], dtype=dtype).tobytes()
        result = module.half_value(dtype.type(3))
        assert type(result) is {2: np.float16, 4: float}[kind]
        assert result == dtype.type(1.5)
