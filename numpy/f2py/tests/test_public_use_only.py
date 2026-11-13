from numpy.f2py import crackfortran
from numpy.f2py.f90mod_rules import buildhooks


def test_private_parameters_are_not_imported(tmp_path):
    # The init wrapper needs `work`. It must not import the
    # parameters `prime` and `offset_basis`.
    src = tmp_path / "parent.f90"
    src.write_text(
        "module parent\n"
        "  implicit none\n"
        "  public :: work\n"
        "  interface\n"
        "    module subroutine work(n)\n"
        "      integer, intent(out) :: n\n"
        "    end subroutine work\n"
        "  end interface\n"
        "end module parent\n"
        "submodule (parent) child\n"
        "  integer, parameter :: prime = 3\n"
        "  integer, parameter :: offset_basis = 1\n"
        "contains\n"
        "  subroutine work(n)\n"
        "    integer, intent(out) :: n\n"
        "    n = prime + offset_basis\n"
        "  end subroutine work\n"
        "end submodule child\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    hooks = buildhooks(
        {"block": "python module", "body": blocks, "name": "t"})[1]
    assert "only : work" in hooks
    assert "only : prime" not in hooks
    assert "only : offset_basis" not in hooks
