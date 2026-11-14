from numpy.f2py import crackfortran


def test_operator_star_name_does_not_abort_analysis(tmp_path):
    # `operator(*)` is a legal generic. The `*` used to be compiled as
    # a regex quantifier, which aborted analysis of the whole module.
    src = tmp_path / "opstar.f90"
    src.write_text("""
module m
  type :: pair
    integer :: n
  end type
  public :: operator(*)
  interface operator(*)
    module procedure mul
  end interface
contains
  function mul(a, b) result(c)
    type(pair), intent(in) :: a, b
    type(pair) :: c
    c%n = a%n * b%n
  end function
  integer function plain()
    plain = 3
  end function
end module
""")
    blocks = crackfortran.crackfortran([str(src)])
    names = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if isinstance(node, dict):
            if node.get("block") in ("function", "subroutine"):
                names.append(node.get("name"))
            walk(node.get("body") or [])

    walk(blocks)
    assert "plain" in names
    assert "mul" in names
