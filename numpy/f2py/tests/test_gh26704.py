from numpy.f2py import crackfortran, f2py2e


_FIXED = """\
      REAL FUNCTION ADD(X,
     *Y)
      REAL X, Y
      ADD = X + Y
      END
"""


def test_f77_flag_keeps_fixed_form_for_an_f90_suffix(tmp_path):
    # gh-26704. A fixed-form continuation is column 6. In free form the
    # function name is lost. -f77 selects the form; the suffix must not
    # put it back.
    src = tmp_path / "cont.f90"
    src.write_text(_FIXED)
    saved = crackfortran.form_override
    try:
        crackfortran.form_override = None
        lost = crackfortran.crackfortran([str(src)])
        assert lost[0]["name"] == "unknown_function"

        crackfortran.reset_global_f2py_vars()
        f2py2e.scaninputline(["-f77", str(src)])
        found = crackfortran.crackfortran([str(src)])
        assert found[0]["name"] == "add"
        assert found[0]["args"] == ["x", "y"]
    finally:
        crackfortran.form_override = saved
