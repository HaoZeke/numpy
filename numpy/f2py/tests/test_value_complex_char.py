import numpy as np
import pytest
from numpy.testing import assert_allclose

from . import util


@pytest.mark.slow
class TestValueComplexChar(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "value_complex_char", "value_complex_char.f90")
    ]

    def test_complex_value(self):
        z = np.complex128(1.5 + 2.25j)
        assert_allclose(self.module.pass_complex8(z), 6.0)
        assert_allclose(self.module.value_mod.mod_complex_bindc(z), 6.0)

    def test_character_value(self):
        assert self.module.pass_character("A") == ord("A")
        assert self.module.pass_character_len1("Q") == ord("Q")
        assert self.module.value_mod.mod_character_bindc(b"M") == ord("M")

    def test_character_string(self):
        assert self.module.pass_string("ABC") == ord("A") + 3
