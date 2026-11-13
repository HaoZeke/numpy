from numpy.f2py import crackfortran


def test_submodule_result_uses_parent_parameter(tmp_path):
    # `length` is a parameter of the parent. The submodule result
    # must be dimension 4, not the bare name, or the C wrapper
    # mentions an undeclared `length`.
    src = tmp_path / "math.f90"
    src.write_text(
        "module math\n"
        "  integer, parameter :: length = 4\n"
        "  interface\n"
        "    module function ramp(start) result(y)\n"
        "      real, intent(in) :: start\n"
        "      real :: y(length)\n"
        "    end function ramp\n"
        "  end interface\n"
        "end module math\n"
        "submodule (math) simpl\n"
        "contains\n"
        "  module function ramp(start) result(y)\n"
        "    real, intent(in) :: start\n"
        "    real :: y(length)\n"
        "    y = start\n"
        "  end function ramp\n"
        "end submodule simpl\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    dims = [
        (body.get("vars") or {}).get("y", {}).get("dimension")
        for block in blocks
        if block.get("name") == "simpl"
        for body in block.get("body") or []
        if body.get("name") == "ramp"
    ]
    assert dims == [["4"]]
