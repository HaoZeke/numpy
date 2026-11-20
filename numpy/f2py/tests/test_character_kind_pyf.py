import textwrap

import pytest

from numpy.f2py import crackfortran


@pytest.mark.parametrize("length,kind", [("5", "4"), ("*", "4")])
def test_signature_keeps_character_kind(tmp_path, length, kind):
    src = textwrap.dedent(
        f"""\
        subroutine s(c)
          character(len={length}, kind={kind}), intent(in) :: c
        end subroutine s
        """
    )
    path = tmp_path / "c.f90"
    path.write_text(src)
    pyf = crackfortran.crack2fortran(crackfortran.crackfortran([str(path)]))
    assert f"character(len={length},kind={kind})" in pyf
