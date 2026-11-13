from numpy.f2py import crackfortran


def test_result_named_string_stays_character(tmp_path):
    # `string` is renamed to `string_bn`. The result must use that
    # name, or implicit typing makes it a real and the wrapper assigns
    # a character to a real.
    src = tmp_path / "to_string.f90"
    src.write_text(
        "function to_string(value) result(string)\n"
        "  integer, intent(in) :: value\n"
        "  character(len=:), allocatable :: string\n"
        "  string = 'x'\n"
        "end function\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    fn = blocks[0]
    assert fn["result"] == "string_bn"
    assert fn["vars"]["string_bn"]["typespec"] == "character"
    assert "string" not in fn["vars"]
