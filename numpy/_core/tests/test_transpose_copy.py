"""C-order to Fortran-order copies above the tiled-transpose threshold."""
import numpy as np
from numpy.testing import assert_array_equal
import pytest

# raw_array_assign_array tiles a transpose once the array is past 8 MiB.
SMALL = ((4, 6), (3, 4, 5))


def _large_shape(dtype, ndim):
    item = np.dtype(dtype).itemsize
    count = (8 << 20) // item + 64
    if ndim == 2:
        n = int(np.sqrt(count))
        return (n, count // n)
    n = int(round(count ** (1 / 3)))
    n = max(n, 2)
    m = n
    p = max(count // (n * m), 2)
    return (n, m, p)


def _arrays(shape, dtype, order):
    arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape, order=order)
    return arr


def _views(arr):
    yield arr
    yield arr[::-1, ::-1] if arr.ndim == 2 else arr[::-1, ::-1, ::-1]
    yield arr[::2, :] if arr.ndim == 2 else arr[::2]


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float64,
                                    np.complex128])
def test_asfortranarray_matches_values(ndim, dtype):
    for shape in (SMALL[ndim - 2], _large_shape(dtype, ndim)):
        src = _arrays(shape, dtype, "C")
        if np.prod(shape) * np.dtype(dtype).itemsize > 8 << 20:
            assert src.nbytes > 8 << 20
        got = np.asfortranarray(src)
        assert got.flags.f_contiguous
        assert_array_equal(got, src)


@pytest.mark.parametrize("ndim", [2, 3])
def test_copy_order_f_and_transpose_view(ndim):
    for shape in (SMALL[ndim - 2], _large_shape(np.float64, ndim)):
        src = _arrays(shape, np.float64, "C")
        assert_array_equal(src.copy(order="F"), src)
        # A Fortran array's transpose is C-ordered; copying it back is the
        # other direction of the same transpose.
        fort = np.asfortranarray(src)
        assert_array_equal(fort.T.copy(), fort.T)


@pytest.mark.parametrize("ndim", [2, 3])
def test_noncontiguous_and_negative_stride(ndim):
    shape = _large_shape(np.float64, ndim)
    src = _arrays(shape, np.float64, "C")
    for view in _views(src):
        got = np.asfortranarray(view)
        assert_array_equal(got, view)
        assert got.flags.f_contiguous


def test_overlap_uses_a_snapshot():
    src = np.arange(1200 * 1024, dtype=np.float64).reshape(1200, 1024)
    snapshot = src.copy()
    src[::-1, :] = src
    assert_array_equal(src, snapshot[::-1, :])


def test_cast_and_object_and_unaligned_stay_on_the_generic_path():
    src = np.arange(1200 * 1024, dtype=np.float32).reshape(1200, 1024)
    dst = np.empty(src.shape, dtype=np.float64, order="F")
    dst[...] = src
    assert_array_equal(dst, src.astype(np.float64))

    objects = np.empty((2, 3), dtype=object)
    objects[...] = [[1, "a", None], [2, "b", (3,)]]
    copied = np.empty((2, 3), dtype=object, order="F")
    copied[...] = objects
    assert copied[0, 1] is objects[0, 1]
    assert copied[1, 2] == (3,)

    buf = np.zeros(src.size * 8 + 1, dtype=np.uint8)
    unaligned = np.ndarray(src.shape, dtype=np.float64, buffer=buf, offset=1)
    unaligned[...] = np.arange(unaligned.size, dtype=np.float64).reshape(src.shape)
    assert_array_equal(np.asfortranarray(unaligned), unaligned)


def test_small_slice_and_zero_size():
    src = np.arange(24, dtype=np.float64).reshape(4, 6)
    assert_array_equal(np.asfortranarray(src[::2, ::3]), src[::2, ::3])
    empty = np.zeros((0, 4), dtype=np.float64)
    got = np.asfortranarray(empty)
    assert got.shape == (0, 4)
    assert got.flags.f_contiguous
