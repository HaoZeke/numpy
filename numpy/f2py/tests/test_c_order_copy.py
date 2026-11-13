import numpy as np
import pytest

from . import util


class TestCOrderCopy(util.F2PyTest):
    sources = [util.getpath("tests", "src", "c_order_copy", "echo.f90")]

    def _views(self, arr):
        # C-order, a non-contiguous slice, and a negative-stride view.
        return (
            np.array(arr, order="C", copy=True),
            arr[::2, ::2] if arr.ndim == 2 else arr[::2, ::2, ::2],
            arr[::-1, ::-1] if arr.ndim == 2 else arr[::-1, ::-1, ::-1],
        )

    def _assert_echoed(self, echo, view):
        out = echo(view)
        assert out.shape == view.shape
        assert out.flags.f_contiguous
        assert np.array_equal(out, view)

    def test_copy_in_rank2(self):
        # Small enough to stay on NumPy's copy.
        small = np.arange(24, dtype=np.float64).reshape(4, 6)
        for view in self._views(small):
            self._assert_echoed(self.module.echo2, view)
        # Above 8 MiB, so the tiled transpose runs.
        large = np.arange(1200 * 1024, dtype=np.float64).reshape(1200, 1024)
        assert large.nbytes > 8 << 20
        self._assert_echoed(self.module.echo2, large)
        self._assert_echoed(self.module.echo2, large[::-1, ::-1])
        parent = np.arange(2400 * 1024, dtype=np.float64).reshape(2400, 1024)
        sliced = parent[::2, :]
        assert sliced.nbytes > 8 << 20 and not sliced.flags.c_contiguous
        self._assert_echoed(self.module.echo2, sliced)

    def test_copy_in_rank3(self):
        small = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
        for view in self._views(small):
            self._assert_echoed(self.module.echo3, view)
        large = np.arange(200 * 200 * 30, dtype=np.float64).reshape(200, 200, 30)
        assert large.nbytes > 8 << 20
        self._assert_echoed(self.module.echo3, large)
        self._assert_echoed(self.module.echo3, large[::-1, ::-1, ::-1])
        parent = np.arange(400 * 200 * 30, dtype=np.float64).reshape(400, 200, 30)
        sliced = parent[::2]
        assert sliced.nbytes > 8 << 20 and not sliced.flags.c_contiguous
        self._assert_echoed(self.module.echo3, sliced)

    def test_inout_writeback_fortran_order(self):
        rank2 = np.arange(12, dtype=np.float64).reshape(3, 4, order="F")
        self.module.bump2(rank2)
        assert np.array_equal(
            rank2, np.arange(12, dtype=np.float64).reshape(3, 4, order="F") + 1
        )
        rank3 = np.arange(24, dtype=np.float64).reshape(2, 3, 4, order="F")
        self.module.bump3(rank3)
        assert np.array_equal(
            rank3,
            np.arange(24, dtype=np.float64).reshape(2, 3, 4, order="F") + 1,
        )

    def test_inout_rejects_c_order_and_noncontiguous(self):
        c_order = np.arange(12, dtype=np.float64).reshape(3, 4)
        with pytest.raises(ValueError, match="not fortran contiguous"):
            self.module.bump2(c_order)
        sliced = np.asfortranarray(c_order)[::2, :]
        assert not sliced.flags.f_contiguous
        with pytest.raises(ValueError, match="not fortran contiguous"):
            self.module.bump2(sliced)
