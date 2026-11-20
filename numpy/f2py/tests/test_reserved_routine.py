import pytest

import numpy as np

from . import util

CODE = """
subroutine int(n)
  integer, intent(inout) :: n
  n = n + 1
end subroutine

integer function len(n)
  integer, intent(in) :: n
  len = n + 1
end function

integer function max(callback, n)
  external callback
  integer callback
  integer, intent(in) :: n
  max = callback(n)
end function

module reserved_module
contains
  subroutine min(n)
    integer, intent(inout) :: n
    n = n + 1
  end subroutine

  integer function size(n)
    integer, intent(in) :: n
    size = n + 1
  end function
end module
"""

BINDC_CODE = """
subroutine bar(n)
  integer, intent(inout) :: n
  n = n + 1
end subroutine

subroutine type(n) bind(c)
  integer, intent(inout) :: n
  n = n + 1
end subroutine
"""


@pytest.mark.slow
def test_reserved_routines_call_fortran_symbols():
    mod = util.build_code(CODE, suffix=".f90", module_name="reserved_routines")

    arr = np.array(4, dtype=np.int32)
    mod.int_bn(arr)
    assert int(arr) == 5
    assert mod.len_bn(4) == 5
    assert mod.max_bn(lambda value: value + 1, 4) == 5

    arr = np.array(4, dtype=np.int32)
    mod.reserved_module.min_bn(arr)
    assert int(arr) == 5
    assert mod.reserved_module.size_bn(4) == 5


@pytest.mark.slow
def test_bindc_reserved_routine_import():
    mod = util.build_code(
        BINDC_CODE, suffix=".f90", module_name="reserved_bindc_routines"
    )

    arr = np.array(4, dtype=np.int32)
    mod.bar(arr)
    assert int(arr) == 5
