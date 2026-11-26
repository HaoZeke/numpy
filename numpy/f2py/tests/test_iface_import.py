import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestIfaceImport(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "iface_import", "apply.f90"),
        util.getpath("tests", "src", "iface_import", "csum.c"),
    ]

    @pytest.mark.parametrize("name", ["apply_renamed", "apply_named"])
    def test_c_sum(self, name):
        x = np.array([1, 2, 3, 4], dtype=np.int32)
        assert getattr(self.module, name)(x) == 10
