import os
import pytest
import numpy as np

from . import util


class TestData(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_stmts.f90")]

    # For gh-23276
    def test_data_stmts(self):
        assert self.module.cmplxdat.i == 2
        assert self.module.cmplxdat.j == 3
        assert self.module.cmplxdat.x == 1.5
        assert self.module.cmplxdat.y == 2.0
        assert self.module.cmplxdat.medium_ref_index == np.array(1.+0.j)
