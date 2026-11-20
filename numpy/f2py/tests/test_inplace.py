
import pytest

import numpy as np
from numpy.f2py.tests import util
from numpy.testing import assert_array_equal


@pytest.mark.slow
class TestInplace(util.F2PyTest):
    sources = [util.getpath("tests", "src", "inplace", "foo.f")]

    @pytest.mark.parametrize("func", ["inplace", "inplace_out"])
    @pytest.mark.parametrize("writeable", ["writeable", "readonly"])
    @pytest.mark.parametrize("view", [
        None, (), (slice(None, 2, None), slice(None, None, 2))])
    @pytest.mark.parametrize("dtype", ["f4", "f8"])
    def test_inplace(self, dtype, view, writeable, func):
        # Test inplace modifications of an input array.
        a = np.arange(12.0, dtype=dtype).reshape((3, 4)).copy()
        a.flags.writeable = writeable == "writeable"
        k = a if view is None else a[view]

        ffunc = getattr(self.module, func)
        if not a.flags.writeable:
            with pytest.raises(ValueError, match="WRITEBACKIFCOPY base is read-only"):
                ffunc(k)
            return

        ref_k = k
        exp_copy = k.copy()
        exp_k = k ** 2
        exp_a = a.copy()
        exp_a[view or ()] = exp_k
        if func == "inplace_out":
            kout, copy = ffunc(k)
            assert kout is k
        else:
            copy = ffunc(k)
        assert_array_equal(copy, exp_copy)
        assert k is ref_k
        assert np.allclose(k, exp_k)
        assert np.allclose(a, exp_a)

    @pytest.mark.parametrize("func", ["inplace", "inplace_out"])
    def test_inplace_error(self, func):
        ffunc = getattr(self.module, func)
        with pytest.raises(ValueError, match="input.*not compatible"):
            ffunc(np.array([1 + 1j]))


@pytest.mark.slow
class TestInOutReadOnly(util.F2PyTest):
    suffix = ".f90"
    code = """
    subroutine bump_f(a, m, n)
      integer, intent(in) :: m, n
      real(8) :: a(m, n)
    !f2py intent(in,out) :: a
      a = a + 1.0d0
    end subroutine bump_f

    subroutine bump_c(a, m, n)
      integer, intent(in) :: m, n
      real(8) :: a(m, n)
    !f2py intent(in,out,c) :: a
      a = a + 1.0d0
    end subroutine bump_c
    """

    @pytest.mark.parametrize(("func", "order"), [
        ("bump_f", "F"),
        ("bump_c", "C"),
    ])
    def test_readonly_in_out_copies(self, func, order):
        orig = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order=order)
        arr = orig.copy(order=order)
        arr.flags.writeable = False
        out = getattr(self.module, func)(arr)
        assert_array_equal(arr, orig)
        assert_array_equal(out, orig + 1.0)
        assert out is not arr
        assert out.flags.writeable
