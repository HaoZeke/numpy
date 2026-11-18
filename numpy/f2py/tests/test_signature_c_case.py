from numpy.f2py import crackfortran


def test_c_statements_keep_case_in_generated_signature(tmp_path, monkeypatch):
    source = tmp_path / "case.pyf"
    source.write_text(
        """\
python module CaseModule
  usercode '''
  static PyObject *MixedHelper(void) { return PyLong_FromLong(1); }
  '''
  interface
    subroutine MixedRoutine()
      callstatement MixedHelper()
      callprotoargument PyObject *MixedValue
    end subroutine MixedRoutine
  end interface
  pymethoddef '''
  {"MixedMethod", MixedHelper, METH_NOARGS, NULL},
  '''
end python module CaseModule
"""
    )
    monkeypatch.setattr(crackfortran, "dolowercase", 1)

    first = crackfortran.crack2fortran(crackfortran.crackfortran([str(source)]))
    generated = tmp_path / "generated.pyf"
    generated.write_text(first)
    second = crackfortran.crack2fortran(
        crackfortran.crackfortran([str(generated)])
    )

    for text in (first, second):
        assert "PyObject *MixedValue" in text
        assert "MixedHelper" in text
        assert '"MixedMethod"' in text
        assert "subroutine mixedroutine" in text
