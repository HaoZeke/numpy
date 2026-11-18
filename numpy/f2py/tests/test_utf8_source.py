import numpy.f2py.crackfortran as crackfortran


def test_utf8_source_without_bom(tmp_path, monkeypatch):
    # LAPACK comments use UTF-8 (no BOM). Without charset_normalizer the
    # reader used to open those files as ASCII and abort.
    monkeypatch.setattr(crackfortran, "charset_normalizer", None)
    src = tmp_path / "leq.f90"
    src.write_bytes(
        "subroutine s(n)\n"
        "  ! reflector order is n ≤ 10\n"
        "  integer, intent(in) :: n\n"
        "end subroutine s\n".encode("utf-8")
    )
    blocks = crackfortran.crackfortran([str(src)])
    assert blocks[0]["name"] == "s"
    assert blocks[0]["args"] == ["n"]
