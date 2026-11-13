import numpy as np
import pytest

from numpy.f2py import capi_maps, crackfortran
from numpy.f2py.auxfuncs import getargs2

from . import util


def _routine(src, name):
    blocks = crackfortran.crackfortran([str(src)])
    found = []

    def walk(block):
        if isinstance(block, list):
            for item in block:
                walk(item)
            return
        if not isinstance(block, dict):
            return
        if block.get("name") == name and block.get("block") == "subroutine":
            found.append(block)
        for child in block.get("body") or []:
            walk(child)

    walk(blocks)
    assert found
    return found[0]


def test_lower_bound_only(tmp_path):
    # dimension(0:) has no upper bound. It must stay dimension(:),
    # and the wrapper must not emit array_Dims[0]=1 + ;
    src = tmp_path / "lb.f90"
    src.write_text(
        "subroutine isort(array)\n"
        "  integer, intent(inout) :: array(0:)\n"
        "end subroutine isort\n"
        "subroutine bounded(n, array)\n"
        "  integer, intent(in) :: n\n"
        "  integer, intent(out) :: array(2:n)\n"
        "end subroutine bounded\n"
    )
    rout = _routine(src, "isort")
    assert rout["vars"]["array"]["dimension"] == [":"]
    _, depargs = getargs2(rout)
    capi_maps.depargs = depargs
    dims = capi_maps.getarrdims("array", rout["vars"]["array"])
    assert dims["setdims"] == ""

    bounded = _routine(src, "bounded")
    assert bounded["vars"]["array"]["dimension"] != [":"]
    assert "n" in bounded["vars"]["array"]["dimension"][0]


@pytest.mark.slow
class TestLowerBoundColon(util.F2PyTest):
    suffix = ".f90"
    code = """
module sorter
contains
  subroutine isort(array)
    integer, intent(inout) :: array(0:)
    integer :: i, j, tmp
    do i = 0, size(array) - 2
      do j = i + 1, size(array) - 1
        if (array(j) < array(i)) then
          tmp = array(i)
          array(i) = array(j)
          array(j) = tmp
        end if
      end do
    end do
  end subroutine isort
end module sorter
"""

    def test_sort_in_place(self):
        a = np.array([3, 1, 4, 1, 5], dtype=np.int32)
        self.module.sorter.isort(a)
        assert a.tolist() == [1, 1, 3, 4, 5]
