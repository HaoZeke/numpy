import textwrap

import pytest

from numpy.f2py import crackfortran


@pytest.mark.parametrize("source, expect", [
    (
        """
        subroutine foo(n, m)
          integer :: n
          integer :: m
          !f2py intent(hide), depend(m) :: n = m
        end subroutine foo
        """,
        {"intent": ["hide"], "depend": ["m"], "=": "m"},
    ),
    (
        """
        subroutine foo(n)
          integer :: n
          !f2py check(n > 0) :: n = 2
        end subroutine foo
        """,
        {"check": ["n > 0"], "=": "2"},
    ),
])
def test_directive_clauses(tmp_path, monkeypatch, source, expect):
    monkeypatch.setattr(crackfortran, "verbose", 0)
    path = tmp_path / "directive.f90"
    path.write_text(textwrap.dedent(source))
    mod = crackfortran.crackfortran([str(path)])
    var = mod[0]["vars"]["n"]
    for key, value in expect.items():
        assert var[key] == value
