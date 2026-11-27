import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestParamCharLen(util.F2PyTest):
    suffix = ".f90"
    code = """
module m
  integer, parameter :: n(2) = (/3, 5/)
  character(len=n(1)) :: w3
  character(len=n(2)) :: w5(2)
end module
"""

    @pytest.mark.parametrize("name,length,rank", [("w3", 3, 0), ("w5", 5, 1)])
    def test_length(self, name, length, rank):
        blob = b"abcdef"[:length]
        mod = self.module.m
        if rank == 0:
            setattr(mod, name, blob)
        else:
            setattr(mod, name, np.array([blob, blob]))
        got = np.asarray(getattr(mod, name))
        assert got.dtype.itemsize == length
        assert got.reshape(-1)[0] == blob
