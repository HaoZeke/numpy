import textwrap

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from . import util


@pytest.mark.slow
class TestISOC(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "isocintrin", "isoCtests.f90"),
    ]

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.clongdouble])
    def test_bindc_complex_float(self, dtype):
        a = np.array([1j, 2j, 3j], dtype=dtype)
        b = np.array([4j, 5j, 6j], dtype=dtype)
        out = {
            np.complex64: self.module.coddity.add_cfloat_arr,
            np.complex128: self.module.coddity.add_cdouble_arr,
            np.clongdouble: self.module.coddity.add_clong_double_arr
        }[dtype](a, b)
        exp = a + b
        assert_allclose(out, exp)
        assert out.dtype == exp.dtype

    # gh-30352: c_float, c_double and c_long_double are complex kinds too
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.clongdouble])
    def test_complex_real_kinds(self, dtype):
        a = np.array([1 + 1j, 2 - 2j, 3j], dtype=dtype)
        b = np.array([4j, 5, -6j], dtype=dtype)
        out = {
            np.complex64: self.module.coddity.add_cfloat_rkind_arr,
            np.complex128: self.module.coddity.add_cdouble_rkind_arr,
            np.clongdouble: self.module.coddity.add_clong_double_rkind_arr
        }[dtype](a, b)
        exp = a + b
        assert_allclose(out, exp)
        assert out.dtype == exp.dtype

    # gh-24553
    def test_c_double(self):
        out = self.module.coddity.c_add(1, 2)
        exp_out = 3
        assert out == exp_out

    # gh-9693
    def test_bindc_function(self):
        out = self.module.coddity.wat(1, 20)
        exp_out = 8
        assert out == exp_out

    # gh-25207
    def test_bindc_kinds(self):
        out = self.module.coddity.c_add_int64(1, 20)
        exp_out = 21
        assert out == exp_out

    # gh-25207 (int64/add_arr)
    @pytest.mark.parametrize("dtype", [np.int8, np.long, np.int64])
    def test_bindc_add_int_arr(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([4, 5, 6], dtype=dtype)
        out = {
            np.int8: self.module.coddity.add_int8_arr,
            np.long: self.module.coddity.add_clong_arr,
            np.int64: self.module.coddity.add_arr
        }[dtype](a, b)
        exp_out = a + b
        assert_allclose(out, exp_out)
        assert out.dtype == exp_out.dtype


@pytest.mark.parametrize("kind, ctype", [
    ("c_float", "complex_float"),
    ("c_double", "complex_double"),
    ("c_long_double", "complex_long_double"),
])
def test_complex_real_kind_ctype(kind, ctype):
    from numpy.f2py.capi_maps import getctype

    assert getctype({"typespec": "complex", "kindselector": {"kind": kind}}) == ctype
@pytest.mark.slow
class TestISOCSizedKinds(util.F2PyTest):
    # gh-25229: kinds whose C type has an implementation-defined width
    suffix = ".f90"
    kinds = ["c_size_t", "c_ptrdiff_t", "c_intmax_t", "c_intptr_t",
             "c_int_fast8_t", "c_int_fast16_t", "c_int_fast32_t",
             "c_int_fast64_t"]

    code = "module isoc_sized\n  use iso_c_binding\n  implicit none\ncontains\n"
    for kind in [*kinds, "c_bool"]:
        # C_<NAME> interoperates with the C type <NAME> (F2018 Table 18.2)
        typespec = "logical" if kind == "c_bool" else "integer"
        code += textwrap.indent(textwrap.dedent(f"""
        function fsize_{kind}() result(n)
          integer(c_int) :: n
          {typespec}({kind}) :: x
          n = int(c_sizeof(x), c_int)
        end function fsize_{kind}
        subroutine csize_{kind}(n)
          integer(c_int), intent(out) :: n
          !f2py callstatement n = (int)sizeof({kind[2:]})
          !f2py callprotoargument int*
          n = 0
        end subroutine csize_{kind}
        """), "  ")
    for kind in kinds:
        code += textwrap.indent(textwrap.dedent(f"""
        subroutine add_{kind}(a, b, c)
          integer({kind}), intent(in) :: a
          integer({kind}), intent(inout) :: b
          integer({kind}), intent(out) :: c
          c = a + b
          b = b + 1_{kind}
        end subroutine add_{kind}
        subroutine add_{kind}_arr(a, c)
          integer({kind}), intent(in) :: a(3)
          integer({kind}), intent(out) :: c(3)
          c = a + 1_{kind}
        end subroutine add_{kind}_arr
        """), "  ")
    code += textwrap.indent(textwrap.dedent("""
    subroutine not_c_bool(a, b, c)
      logical(c_bool), intent(in) :: a
      logical(c_bool), intent(inout) :: b
      logical(c_bool), intent(out) :: c
      c = .not. a
      b = .not. b
    end subroutine not_c_bool
    subroutine not_c_bool_arr(a, c)
      logical(c_bool), intent(in) :: a(3)
      logical(c_bool), intent(out) :: c(3)
      c = .not. a
    end subroutine not_c_bool_arr
    """), "  ")
    code += "end module isoc_sized\n"
    for kind in [*kinds, "c_bool"]:
        typespec = "logical" if kind == "c_bool" else "integer"
        code += textwrap.dedent(f"""
        subroutine call_{kind}(f, a, c)
          use iso_c_binding, only: {kind}
          external f
          {typespec}({kind}) :: f
          {typespec}({kind}), intent(in) :: a
          {typespec}({kind}), intent(out) :: c
          c = f(a)
        end subroutine call_{kind}
        """)

    def c_size(self, kind):
        m = self.module.isoc_sized
        fsize = getattr(m, f"fsize_{kind}")()
        csize = getattr(m, f"csize_{kind}")()
        if fsize != csize:
            # LLVM Flang's C_INT_FAST16_T and C_INT_FAST32_T do not match
            # the C companion types. Open issue
            # "[Flang] Incorrect value of C_INT_FAST*_T",
            # https://github.com/llvm/llvm-project/issues/208361
            pytest.skip(f"the Fortran compiler's {kind} has {fsize} bytes, "
                        f"the C {kind[2:]} has {csize}")
        return csize

    def dtype(self, kind):
        return np.dtype(f"{'u' if kind == 'c_size_t' else 'i'}"
                        f"{self.c_size(kind)}")

    @pytest.mark.parametrize("kind", kinds)
    def test_array(self, kind):
        dtype = self.dtype(kind)
        a = np.arange(3)
        c = getattr(self.module.isoc_sized, f"add_{kind}_arr")(a)
        assert c.dtype == dtype
        assert_array_equal(c, a + 1)

    @pytest.mark.parametrize("kind", kinds)
    def test_scalar(self, kind):
        add = getattr(self.module.isoc_sized, f"add_{kind}")
        hi = np.iinfo(f"i{self.c_size(kind)}").max
        assert add(hi - 1, 1) == hi
        if kind == "c_size_t":
            with pytest.raises(OverflowError):
                add(-1, 1)
        else:
            assert add(-hi, -1) == -hi - 1
            with pytest.raises(OverflowError):
                add(hi + 1, 0)

    @pytest.mark.parametrize("kind", kinds)
    @pytest.mark.parametrize("obj", [False, True])
    def test_inout(self, kind, obj):
        dtype = self.dtype(kind)
        b = np.array(41, dtype=object if obj else dtype)
        assert getattr(self.module.isoc_sized, f"add_{kind}")(1, b) == 42
        assert b == 42

    def test_c_bool(self):
        size = self.c_size("c_bool")
        m = self.module.isoc_sized
        b = np.array(True)
        assert m.not_c_bool(True, b) == 0
        assert not b
        c = m.not_c_bool_arr([True, False, True])
        if size == np.dtype(np.bool).itemsize:
            assert c.dtype == np.bool
        assert_array_equal(c, [False, True, False])

    @pytest.mark.parametrize("kind", [*kinds, "c_bool"])
    def test_callback(self, kind):
        size = self.c_size(kind)
        call = getattr(self.module, f"call_{kind}")
        if kind == "c_bool":
            assert call(lambda x: not x, True) == 0
        else:
            hi = np.iinfo(f"i{size}").max
            assert call(lambda x: x - 1, hi) == hi - 1


def test_process_f2cmap_dict():
    from numpy.f2py.auxfuncs import process_f2cmap_dict

    f2cmap_all = {"integer": {"8": "rubbish_type"}}
    new_map = {"INTEGER": {"4": "int"}}
    c2py_map = {"int": "int", "rubbish_type": "long"}

    exp_map, exp_maptyp = ({"integer": {"8": "rubbish_type", "4": "int"}}, ["int"])

    # Call the function
    res_map, res_maptyp = process_f2cmap_dict(f2cmap_all, new_map, c2py_map)

    # Assert the result is as expected
    assert res_map == exp_map
    assert res_maptyp == exp_maptyp
