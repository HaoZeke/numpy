import textwrap

import pytest

from numpy.f2py import crackfortran


@pytest.mark.parametrize("source, name, key", [
    (
        """
        subroutine foo(a)
          real :: a
          !f2py check() a
        end subroutine foo
        """,
        "a",
        "check",
    ),
    (
        """
        subroutine foo(n)
          integer :: n
          !f2py depend() n
        end subroutine foo
        """,
        "n",
        "depend",
    ),
])
def test_empty_clause(tmp_path, monkeypatch, source, name, key):
    monkeypatch.setattr(crackfortran, "verbose", 0)
    path = tmp_path / "empty.f90"
    path.write_text(textwrap.dedent(source))
    mod = crackfortran.crackfortran([str(path)])
    assert mod[0]["vars"][name][key] == []
