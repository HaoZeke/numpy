from numpy.f2py import crackfortran


def test_protected_is_not_optional(tmp_path):
    src = tmp_path / "prot.f90"
    src.write_text(
        "module origin\n"
        "  integer, protected :: initialized = 3\n"
        "  integer, protected :: uninitialized\n"
        "  integer :: defaulted = 4\n"
        "end module origin\n"
        "module consumer\n"
        "  use origin, only: initialized, uninitialized\n"
        "end module consumer\n"
    )
    blocks = crackfortran.crackfortran([str(src)])

    def check(parsed):
        origin = next(block for block in parsed if block["name"] == "origin")
        consumer = next(block for block in parsed if block["name"] == "consumer")
        assert origin["vars"]["initialized"]["attrspec"] == ["protected"]
        assert origin["vars"]["uninitialized"]["attrspec"] == ["protected"]
        assert "optional" in origin["vars"]["defaulted"]["attrspec"]
        use = consumer["use"]["origin"]
        assert use["only"] == 1
        assert set(use["map"]) == {"initialized", "uninitialized"}

    check(blocks)
    pyf = tmp_path / "prot.pyf"
    pyf.write_text(crackfortran.crack2fortran(blocks))
    check(crackfortran.crackfortran([str(pyf)]))
