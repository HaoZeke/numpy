from numpy.f2py import crackfortran


def test_two_modules_in_one_crack(tmp_path):
    # `end block` closed the procedure. The following `end` then ran
    # with `groupcounter` already 0, so a second module in the same
    # crack aborted.
    first = tmp_path / "first.f90"
    second = tmp_path / "second.f90"
    first.write_text(
        "module first\n"
        "contains\n"
        "  integer function fa(n)\n"
        "    integer, intent(in) :: n\n"
        "    fa = n + 1\n"
        "  end function fa\n"
        "end module first\n"
    )
    second.write_text(
        "module second\n"
        "contains\n"
        "  integer function fb(n)\n"
        "    integer, intent(in) :: n\n"
        "    block\n"
        "      integer :: t\n"
        "      t = n + 2\n"
        "      fb = t\n"
        "    end block\n"
        "  end function fb\n"
        "end module second\n"
    )
    crackfortran.verbose = 0
    post = crackfortran.crackfortran([str(first), str(second)])
    names = [block.get("name") for block in post if isinstance(block, dict)]
    assert "first" in names
    assert "second" in names
