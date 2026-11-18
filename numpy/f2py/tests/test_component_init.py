from numpy.f2py import crackfortran


def test_component_initializer_is_not_optional(tmp_path):
    src = tmp_path / "rec.f90"
    src.write_text(
        "module m\n"
        "  integer :: module_default = 4\n"
        "  type :: rec\n"
        "    integer :: count = 2\n"
        "  end type rec\n"
        "end module m\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    module = blocks[0]
    assert "optional" in module["vars"]["module_default"]["attrspec"]
    rec = next(b for b in module["body"] if b["block"] == "type")
    assert rec["vars"]["count"]["="] == "2"
    assert "optional" not in rec["vars"]["count"].get("attrspec", [])
    pyf = crackfortran.crack2fortran(blocks)
    assert "integer :: count=2" in pyf
    assert "optional :: count" not in pyf
    assert "optional :: module_default" in pyf
