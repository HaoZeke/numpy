import numpy as np
import pytest

from numpy.f2py import crackfortran

from . import util


def test_kind_rename_chain(tmp_path):
    src = tmp_path / "kinds.f90"
    src.write_text(
        "module kinds\n"
        "  use, intrinsic :: iso_fortran_env, only: rk => real64\n"
        "end module kinds\n"
        "module m\n"
        "  use kinds, only: dp => rk\n"
        "contains\n"
        "  subroutine scale(x)\n"
        "    complex(dp), intent(inout) :: x\n"
        "    x = x * 2\n"
        "  end subroutine scale\n"
        "end module m\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    scale = None
    for block in blocks:
        for routine in block.get("body") or []:
            if isinstance(routine, dict) and routine.get("name") == "scale":
                scale = routine
    assert scale is not None
    assert scale["vars"]["x"]["kindselector"]["kind"] == "real64"


@pytest.mark.slow
class TestGenericOnlyName(util.F2PyTest):
    # The specific procedure is private. The public name is the generic.
    suffix = ".f90"
    code = """
module parent
  implicit none
  private
  public :: transform
  interface transform
    module procedure work
  end interface
contains
  subroutine work(n)
    integer, intent(out) :: n
    n = 4
  end subroutine work
end module parent
"""

    def test_call_through_generic(self):
        assert self.module.parent.work() == 4


@pytest.mark.slow
class TestRenamedKindChain(util.F2PyTest):
    suffix = ".f90"
    code = """
module kinds
  use, intrinsic :: iso_fortran_env, only: rk => real64
end module kinds
module m
  use kinds, only: dp => rk
contains
  subroutine scale(x)
    complex(dp), intent(inout) :: x(*)
    x(1) = x(1) * 2
  end subroutine scale
end module m
"""

    def test_complex128(self):
        x = np.array([1 + 0j], dtype=np.complex128)
        self.module.m.scale(x)
        assert x.dtype == np.dtype(np.complex128)
        assert x[0] == np.complex128(2 + 0j)


@pytest.mark.slow
def test_kind_module_read_later(tmp_path):
    # The using module is read first. The rename lives in a later file,
    # which is how fftpack lays out ``dp => rk => real64``.
    user = tmp_path / "user.f90"
    kinds = tmp_path / "kinds.f90"
    user.write_text(
        "module m\n"
        "  use kinds, only: dp => rk\n"
        "contains\n"
        "  subroutine scale(x)\n"
        "    complex(dp), intent(inout) :: x(*)\n"
        "    x(1) = x(1) * 2\n"
        "  end subroutine scale\n"
        "end module m\n"
    )
    kinds.write_text(
        "module kinds\n"
        "  use, intrinsic :: iso_fortran_env, only: rk => real64\n"
        "end module kinds\n"
    )
    mod = util.build_module(
        [str(user), str(kinds)], module_name="latekind")
    x = np.array([1 + 0j], dtype=np.complex128)
    mod.m.scale(x)
    assert x.dtype == np.dtype(np.complex128)
    assert x[0] == np.complex128(2 + 0j)


@pytest.mark.slow
def test_public_interface_complex_kind(tmp_path):
    # The callable signature is the unnamed module interface, which says
    # complex. The external body is a separate file. The kind module is
    # read last.
    iface = tmp_path / "iface.f90"
    kinds = tmp_path / "kinds.f90"
    body = tmp_path / "body.f90"
    iface.write_text(
        "module api\n"
        "  use kinds\n"
        "  implicit none\n"
        "  interface\n"
        "    subroutine scale(x)\n"
        "      import rk\n"
        "      complex(kind=rk), intent(inout) :: x(*)\n"
        "    end subroutine scale\n"
        "  end interface\n"
        "end module api\n"
    )
    kinds.write_text(
        "module kinds\n"
        "  use, intrinsic :: iso_fortran_env, only: rk => real64\n"
        "end module kinds\n"
    )
    body.write_text(
        "subroutine scale(x)\n"
        "  use kinds, only: dp => rk\n"
        "  complex(dp), intent(inout) :: x(*)\n"
        "  x(1) = x(1) * 2\n"
        "end subroutine scale\n"
    )
    mod = util.build_module(
        [str(iface), str(kinds), str(body)], module_name="ifacekind")
    x = np.array([1 + 0j], dtype=np.complex128)
    mod.api.scale(x)
    assert x[0] == np.complex128(2 + 0j)
