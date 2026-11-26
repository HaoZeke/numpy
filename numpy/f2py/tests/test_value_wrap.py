import numpy as np
import pytest
from numpy.testing import assert_allclose

from . import util


@pytest.mark.slow
class TestValueWrap(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "value_wrap", "value_wrap.f90"),
    ]

    @pytest.mark.parametrize("name,dtype,n", [
        ("addi", np.int32, np.int32(10)),
        ("addd", np.float64, np.float64(0.5)),
    ])
    def test_value_scalar(self, name, dtype, n):
        x = np.array([1, 2, 3], dtype=dtype, order="F")
        getattr(self.module, name)(n, x)
        assert_allclose(x, np.array([1, 2, 3], dtype=dtype) + n)
