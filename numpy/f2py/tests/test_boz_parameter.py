import textwrap

import pytest

from numpy.f2py import crackfortran


@pytest.mark.parametrize("expr,size", [
    ("int(z'1e') + int(o'10') + int(b'10')", "40"),
    ('int(Z"1E") + int(O"10") + int(B"10")', "40"),
])
def test_boz_parameter_sizes_intent_out(tmp_path, expr, size):
    source = tmp_path / "boz.f90"
    source.write_text(textwrap.dedent(f"""\
        subroutine s(x)
          implicit none
          integer, parameter :: n = {expr}
          real, intent(out) :: x(n)
          x = 1.0
        end subroutine s
        """))
    mod = crackfortran.crackfortran([str(source)])
    assert mod[0]["vars"]["x"]["dimension"] == [size]
