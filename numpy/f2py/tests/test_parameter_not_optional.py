from numpy.f2py import crackfortran


def test_parameter_initializer_is_not_optional(tmp_path):
    # Module PARAMETER names are constants. An initializer must not mark
    # them optional the way a defaulted module variable is.
    src = tmp_path / "param.f90"
    src.write_text(
        "module m\n"
        "  integer, parameter :: n = 3\n"
        "  integer :: k = 4\n"
        "contains\n"
        "  subroutine s(a)\n"
        "    integer, intent(in) :: a\n"
        "  end subroutine s\n"
        "end module m\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    mod = blocks[0]
    assert "parameter" in mod["vars"]["n"]["attrspec"]
    assert "optional" not in mod["vars"]["n"]["attrspec"]
    assert "optional" in mod["vars"]["k"]["attrspec"]
    pyf = crackfortran.crack2fortran(blocks)
    assert "integer, parameter :: n=3" in pyf
    assert "integer, parameter,optional :: n=3" not in pyf
    assert "integer, optional :: k=4" in pyf
