import textwrap

import pytest

from numpy.f2py import crackfortran
@pytest.mark.parametrize("declaration, expected", [
    (
        "integer, Intent(Out) :: a",
        ["out"],
    ),
    (
        """
        integer :: a
        INTENT(IN OUT) :: a
        """,
        ["inout"],
    ),
])
def test_intent_spec_case(tmp_path, monkeypatch, declaration, expected):
    fpath = tmp_path / "intent.f90"
    fpath.write_text(textwrap.dedent(f"""
        subroutine foo(a)
          {declaration}
        end subroutine foo
    """))
    monkeypatch.setattr(crackfortran, "dolowercase", 0)
    monkeypatch.setattr(crackfortran, "verbose", 0)
    post = crackfortran.crackfortran([str(fpath)])
    assert post[0]["vars"]["a"]["intent"] == expected
