"""Error text from f2py wrappers outside scalar check().

NumPy scalars are not PyArrayObjects.
Describing one by reading array fields faults when a character argument
rejects the value.
"""

import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestScalarDescribe(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "regression", "gh31584_errors.f90"),
    ]

    def test_numpy_scalar_is_described(self):
        mod = self.module.__name__
        with pytest.raises(TypeError) as caught:
            self.module.need_char(np.float64(1.0))
        assert str(caught.value) == (
            f"{mod}.need_char() 1st argument (x) can't be converted to "
            "character -- expected str|bytes|sequence-of-str-or-bytes, "
            "got f8-numpy.float64-scalar"
        )
