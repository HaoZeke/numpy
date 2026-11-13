from numpy.f2py import crackfortran


def test_end_procedure_does_not_close_the_module(tmp_path):
    # `end procedure` names the separate module procedure. It must
    # not be read as a bare `end` that closes the module.
    src = tmp_path / "m.f90"
    src.write_text(
        "module m\n"
        "  integer :: keep\n"
        "contains\n"
        "  module procedure work\n"
        "    integer :: n\n"
        "    n = 1\n"
        "  end procedure work\n"
        "end module m\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    assert blocks[0]["block"] == "module"
    assert blocks[0]["name"] == "m"
    assert "n" not in blocks[0]["vars"]
    assert "keep" in blocks[0]["vars"]
