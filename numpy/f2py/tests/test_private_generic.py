import pytest

from numpy.f2py import crackfortran

from . import util


def _routines(blocks):
    found = []

    def walk(block):
        if isinstance(block, list):
            for item in block:
                walk(item)
            return
        if not isinstance(block, dict):
            return
        if block.get("block") in ("function", "subroutine"):
            found.append(block)
        for child in block.get("body") or []:
            walk(child)

    walk(blocks)
    return found


def test_private_generic_is_skipped(tmp_path):
    src = tmp_path / "priv.f90"
    src.write_text(
        "module parent\n"
        "  implicit none\n"
        "  private\n"
        "  public :: transform\n"
        "  interface transform\n"
        "    module procedure work\n"
        "  end interface\n"
        "  interface secret\n"
        "    module procedure hidden\n"
        "  end interface\n"
        "contains\n"
        "  subroutine work(n)\n"
        "    integer, intent(out) :: n\n"
        "    n = 4\n"
        "  end subroutine work\n"
        "  subroutine hidden(n)\n"
        "    integer, intent(out) :: n\n"
        "    n = 9\n"
        "  end subroutine hidden\n"
        "end module parent\n"
    )
    routines = _routines(crackfortran.crackfortran([str(src)]))
    names = {rout["name"]: rout for rout in routines}
    assert "hidden" not in names
    assert names["work"]["f2pyenhancements"]["fortranname"] == "transform"


@pytest.mark.slow
class TestPrivateGeneric(util.F2PyTest):
    suffix = ".f90"
    code = """
module parent
  implicit none
  private
  public :: transform
  interface transform
    module procedure work
  end interface
  interface secret
    module procedure hidden
  end interface
contains
  subroutine work(n)
    integer, intent(out) :: n
    n = 4
  end subroutine work
  subroutine hidden(n)
    integer, intent(out) :: n
    n = 9
  end subroutine hidden
end module parent
"""

    def test_public_generic(self):
        assert self.module.parent.work() == 4
        assert not hasattr(self.module.parent, "hidden")
