from numpy.f2py import crackfortran


def test_logical_prefix_types_the_result_name(tmp_path):
    # `result(close)` is the value. `close` starts with `c`, so
    # leaving it implicit makes the wrapper result a real.
    src = tmp_path / "is_ok.f90"
    src.write_text(
        "elemental logical function is_ok(x) result(close)\n"
        "  real, intent(in) :: x\n"
        "  close = x > 0.0\n"
        "end function is_ok\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    assert blocks[0]["vars"]["close"]["typespec"] == "logical"
