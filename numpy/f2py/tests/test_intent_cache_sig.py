import pytest

from numpy.f2py import crackfortran


@pytest.mark.parametrize(
    ("intent", "expect"),
    [
        ("cache", "intent(cache)"),
        ("in,cache", "intent(in,cache)"),
    ],
)
def test_signature_keeps_intent_cache(tmp_path, intent, expect):
    source = (
        "subroutine foo(a, n)\n"
        "  real :: a(n)\n"
        "  integer :: n\n"
        f"  !f2py intent({intent}) a\n"
        "end subroutine foo\n"
    )
    path = tmp_path / "cache.f90"
    path.write_text(source)
    first = crackfortran.crack2fortran(
        crackfortran.crackfortran([str(path)])
    )
    generated = tmp_path / "generated.pyf"
    generated.write_text(first)
    second = crackfortran.crack2fortran(
        crackfortran.crackfortran([str(generated)])
    )
    assert expect in first
    assert expect in second
