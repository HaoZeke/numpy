import pytest

from . import util


@pytest.mark.slow
@pytest.mark.parametrize(("name", "decl", "default", "expected"), [
    ("optdef_real", "real(8)", "-1.5d0", -1.5),
    ("optdef_complex", "complex(8)", "(1d-1, 2.5D0)", 0.1 + 2.5j),
])
def test_optional_default_d_exponent(name, decl, default, expected):
    code = f"""
subroutine get(x, y)
  {decl}, optional :: x
  {decl}, intent(out) :: y
!f2py {decl} :: x = {default}
  y = x
end subroutine
"""
    mod = util.build_code(code, suffix=".f90", module_name=name)
    assert mod.get() == expected
