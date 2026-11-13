from numpy.f2py import crackfortran


def _kind(src, routine, arg):
    blocks = crackfortran.crackfortran([str(src)])
    found = []

    def walk(block):
        if isinstance(block, list):
            for item in block:
                walk(item)
            return
        if not isinstance(block, dict):
            return
        if block.get("name") == routine and block.get("block") == "function":
            found.append(block)
        for child in block.get("body") or []:
            walk(child)

    walk(blocks)
    assert found
    return found[0]["vars"][arg]["kindselector"]["kind"]


def test_imported_kind_parameter(tmp_path):
    # dp = real64 is 8. An interface that imports dp must not keep
    # the name in real(kind=dp).
    src = tmp_path / "quad.f90"
    src.write_text(
        "module kinds\n"
        "  use, intrinsic :: iso_fortran_env, only: real64\n"
        "  integer, parameter :: dp = real64\n"
        "end module kinds\n"
        "module quad\n"
        "  use kinds, only: dp\n"
        "  interface\n"
        "    function trap(x) result(y)\n"
        "      import :: dp\n"
        "      real(dp), intent(in) :: x\n"
        "      real(dp) :: y\n"
        "    end function trap\n"
        "  end interface\n"
        "end module quad\n"
    )
    assert _kind(src, "trap", "x") == "8"
    assert _kind(src, "trap", "y") == "8"
