import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestCommonBlock(util.F2PyTest):
    sources = [util.getpath("tests", "src", "common", "block.f")]

    def test_common_block(self):
        self.module.initcb()
        assert self.module.block.long_bn == np.array(1.0, dtype=np.float64)
        assert self.module.block.string_bn == np.array("2", dtype="|S1")
        assert self.module.block.ok == np.array(3, dtype=np.int32)


@pytest.mark.slow
class TestCommonBlockAllHidden(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "common", "gh20053.f"),
        util.getpath("tests", "src", "common", "gh20053.pyf"),
    ]
    module_name = "gh20053_common"

    def test_gh20053(self):
        # Importing used to fail with SystemError on the empty table
        self.module.set_hidden(3.0)
        assert self.module.get_hidden() == 3.0
        assert not hasattr(self.module, "hidden")


@pytest.mark.slow
class TestCommonWithUse(util.F2PyTest):
    sources = [util.getpath("tests", "src", "common", "gh19161.f90")]

    def test_common_gh19161(self):
        assert self.module.data.x == 0
