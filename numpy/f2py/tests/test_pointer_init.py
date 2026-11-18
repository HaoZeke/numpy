from numpy.f2py import crackfortran


def test_pointer_component_init_is_not_a_fake_component(tmp_path):
    src = tmp_path / "node.f90"
    src.write_text(
        "module m\n"
        "  type :: node\n"
        "    type(node), pointer :: next => null()\n"
        "  end type node\n"
        "end module m\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    node = blocks[0]["body"][0]
    assert node["vars"]["next"]["="] == "null()"
    pyf = crackfortran.crack2fortran(blocks)
    assert "next=>null()" in pyf
    assert ":: null" not in pyf
