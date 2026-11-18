from numpy.f2py import crackfortran


def test_extended_intents_survive_generated_signature(tmp_path, monkeypatch):
    source = tmp_path / "intents.f90"
    source.write_text(
        """\
subroutine intents(a,b)
  integer, intent(out) :: a
  !f2py intent(out=result) a
  integer :: b
  !f2py intent(outin,c) b
end subroutine intents
"""
    )
    monkeypatch.setattr(crackfortran, "dolowercase", 0)

    first = crackfortran.crack2fortran(
        crackfortran.crackfortran([str(source)])
    )
    generated = tmp_path / "generated.pyf"
    generated.write_text(first)
    second = crackfortran.crack2fortran(
        crackfortran.crackfortran([str(generated)])
    )

    for text in (first, second):
        assert "out=result" in text
        assert "outin" in text
    assert crackfortran.true_intent_list(
        {"intent": ["outin", "hide"]}
    ) == ["hide"]
    assert crackfortran.true_intent_list(
        {"intent": ["outin", "inplace", "c"]}
    ) == ["inplace", "c"]
