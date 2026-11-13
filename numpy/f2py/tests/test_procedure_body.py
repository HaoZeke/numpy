from numpy.f2py import crackfortran


def test_module_procedure_body_is_not_module_data(tmp_path):
    # Locals of `module procedure work` belong to that body.
    # They used to be module variables, so the generated C named `n`
    # and `use`d the submodule.
    src = tmp_path / "m.f90"
    src.write_text(
        "module m\n"
        "  integer :: keep\n"
        "contains\n"
        "  module procedure work\n"
        "    integer :: n\n"
        "    real :: buf(max(n, 0))\n"
        "    n = 1\n"
        "    buf(1) = n\n"
        "  end procedure work\n"
        "  module procedure other\n"
        "    integer :: m\n"
        "    m = 2\n"
        "  end procedure other\n"
        "end module m\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    assert blocks[0]["block"] == "module"
    assert blocks[0]["name"] == "m"
    assert "keep" in blocks[0]["vars"]
    assert "n" not in blocks[0]["vars"]
    assert "buf" not in blocks[0]["vars"]
    assert "m" not in blocks[0]["vars"]
