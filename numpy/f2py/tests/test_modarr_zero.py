import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestZeroSizeModuleArray(util.F2PyTest):
    suffix = ".f90"
    code = """
module zarr
  implicit none
  real(8), allocatable :: r(:)
  complex(8), allocatable :: z(:,:)
contains
  subroutine seed()
    if (allocated(r)) deallocate(r)
    allocate(r(1))
    r = 1.0d0
    if (allocated(z)) deallocate(z)
    allocate(z(1, 1))
    z = (1.0d0, 2.0d0)
  end subroutine seed

  subroutine empty()
    if (allocated(r)) deallocate(r)
    allocate(r(0))
    if (allocated(z)) deallocate(z)
    allocate(z(0, 3))
  end subroutine empty

  subroutine is_allocated(ra, za)
    logical, intent(out) :: ra, za
    ra = allocated(r)
    za = allocated(z)
  end subroutine is_allocated
end module zarr
"""

    @pytest.mark.parametrize("name,value,nonempty,allocated_index", [
        (
            "r",
            np.zeros((0,), dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
            0,
        ),
        (
            "z",
            np.zeros((0, 3), dtype=np.complex128),
            np.array([[1.0 + 2.0j, 3.0 + 4.0j]], dtype=np.complex128),
            1,
        ),
    ])
    def test_assign_zero_size(self, name, value, nonempty, allocated_index):
        mod = self.module.zarr
        mod.seed()
        setattr(mod, name, value)
        got = getattr(mod, name)
        assert got.shape == value.shape
        assert got.dtype == value.dtype

        setattr(mod, name, nonempty)
        np.testing.assert_array_equal(getattr(mod, name), nonempty)

        mod.empty()
        setattr(mod, name, None)
        assert getattr(mod, name) is None
        assert mod.is_allocated()[allocated_index] == 0
