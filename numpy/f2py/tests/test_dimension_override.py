import pytest

from . import util

ARG = """
subroutine get(a, b)
  real, dimension(3), intent(out) :: a, b(2)
  a = 1
  b = 2
end subroutine
"""

MOD = """
module dimovr_mod
  integer, dimension(3) :: a, b(2)
end module
"""


@pytest.mark.slow
@pytest.mark.parametrize(("name", "code", "shape_of_b"), [
    ("dimovr_arg", ARG, lambda mod: mod.get()[1].shape),
    ("dimovr_mod", MOD, lambda mod: mod.dimovr_mod.b.shape),
], ids=["argument", "module"])
def test_entity_array_spec_overrides_dimension(name, code, shape_of_b):
    mod = util.build_code(code, suffix=".f90", module_name=name)
    assert shape_of_b(mod) == (2,)
