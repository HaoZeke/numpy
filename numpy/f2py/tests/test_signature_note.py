from numpy.f2py import crackfortran


def test_notes_survive_generated_signature_roundtrip(tmp_path, monkeypatch):
    source = tmp_path / "notes.f90"
    source.write_text(
        r"""subroutine noted(n)
  integer, intent(in) :: n
  !f2py note("Routine, quote: 'x' and path C:\\Temp\n\nSecond") noted
  !f2py note('Argument "n", path C:\\data') n
  !f2py note("line one\n line two") n
end subroutine noted
"""
    )
    monkeypatch.setattr(crackfortran, "dolowercase", 0)

    original = crackfortran.crackfortran([str(source)])
    first = crackfortran.crack2fortran(original)
    generated = tmp_path / "generated.pyf"
    generated.write_text(first)
    roundtrip = crackfortran.crackfortran([str(generated)])
    second = crackfortran.crack2fortran(roundtrip)

    assert roundtrip[0]["note"] == original[0]["note"]
    assert roundtrip[0]["vars"]["n"]["note"] == original[0]["vars"]["n"]["note"]
    assert [line.strip() for line in first.splitlines() if "note(" in line] == [
        line.strip() for line in second.splitlines() if "note(" in line
    ]
