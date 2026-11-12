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


@pytest.mark.slow
class TestRestoredStrings(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "regression", "gh31584_errors.f90"),
    ]

    def test_character_inout_copy_back(self):
        with pytest.raises(ValueError) as caught:
            self.module.io_char(b"Z")
        assert str(caught.value) == (
            "try_pyarr_from_character failed -- expected bytes "
            "array-scalar|array, got 1-bytes"
        )

    def test_unexpected_size_keeps_bracket_space(self):
        arr = np.zeros((0, 4), dtype=np.float32)
        with pytest.raises(ValueError) as caught:
            self.module.dim_vec(arr)
        assert str(caught.value) == (
            "unexpected array size: size=4, arr_size=0, rank=1, "
            "effrank=1, arr.nd=2, dims=[ 4 ], arr.dims=[ 0 4 ]\n"
        )


@pytest.mark.slow
class TestNullCapsule(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "regression", "gh31584_errors.f90"),
    ]

    def test_unallocated_allocatable_cpointer(self):
        # Reaching this assertion means the read did not segfault.
        with pytest.raises(ValueError) as caught:
            self.module.m._cpointer
        assert type(caught.value) is ValueError
        assert str(caught.value) == "array 'a' is not allocated"


@pytest.mark.slow
class TestDummyRoutineCPointer(util.F2PyTest):
    suffix = ".pyf"
    module_name = "gh31584_dummy"

    code = """
python module gh31584_dummy
    interface
        subroutine dummy()
            fortranname
            callstatement {}
        end subroutine dummy
    end interface
end python module gh31584_dummy
"""

    def test_empty_fortranname_cpointer(self):
        self.module.dummy()
        with pytest.raises(ValueError, match="array 'dummy' is not allocated"):
            self.module.dummy._cpointer
