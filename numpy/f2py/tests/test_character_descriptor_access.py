import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestCharacterDescriptorAccess(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine read_first_character(value, code)
  implicit none
  character, intent(in) :: value
  integer, intent(out) :: code
  code = iachar(value)
end subroutine read_first_character
"""

    @pytest.mark.parametrize("dtype", ["<U1", ">U1", "<U2", ">U2"])
    def test_first_character_uses_dtype_byteorder(self, dtype):
        value = np.array("AZ", dtype=dtype)
        assert self.module.read_first_character(value) == ord("A")

    @pytest.mark.parametrize("dtype", ["<U2", ">U2"])
    def test_later_non_ascii_character_does_not_change_first(self, dtype):
        value = np.array("A\u00e9", dtype=dtype)
        assert self.module.read_first_character(value) == ord("A")
