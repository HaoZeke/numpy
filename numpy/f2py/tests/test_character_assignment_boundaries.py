import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestCharacterAssignmentBoundaries(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "common", "gh9370.f"),
        util.getpath("tests", "src", "common", "gh9370_zero_length.f90"),
    ]

    @pytest.mark.parametrize("value", [b"", "", np.array(b"")],
                             ids=["bytes", "str", "array"])
    def test_empty_input_preserves_common_guard(self, value):
        self.module.initblock()
        self.module.block.block_string = value
        assert self.module.block.block_string.dtype == np.dtype("S3")
        assert self.module.block.block_string.tobytes() == b"\x00" * 3
        assert self.module.block.guard.tobytes() == b"xyz"

    @pytest.mark.parametrize("value", [b"XY", "XY", np.array(b"XY"), b""],
                             ids=["bytes", "str", "array", "empty"])
    def test_zero_length_scalar_preserves_common_guard(self, value):
        self.module.init_zero_common()
        block = self.module.zero_scalar
        view = block.scalar_value
        block.scalar_value = value
        assert block.scalar_value is view
        assert view.dtype == np.dtype("S0")
        assert view.tobytes() == b""
        assert block.scalar_guard.tobytes() == b"abcdefghijklmnop"

    @pytest.mark.parametrize("value", [
        [b"AB", b"CD", b"EF"],
        np.array([b"AB", b"CD", b"EF"]),
        [b"", b"", b""],
    ], ids=["list", "array", "empty"])
    def test_zero_length_array_preserves_common_guard(self, value):
        self.module.init_zero_common()
        block = self.module.zero_array
        view = block.array_values
        block.array_values = value
        assert block.array_values is view
        assert view.dtype == np.dtype("S0")
        assert view.shape == (3,)
        assert view.tobytes() == b""
        assert block.array_guard.tobytes() == b"abcdefghijklmnop"

    def test_zero_length_array_validates_shape(self):
        self.module.init_zero_common()
        with pytest.raises(ValueError):
            self.module.zero_array.array_values = [b"AB", b"CD"]
        assert self.module.zero_array.array_guard.tobytes() == b"abcdefghijklmnop"

    @pytest.mark.parametrize("value", [
        [], [b"AB", b"CD"], np.array([b"AB", b"CD"]), [b"", b""],
    ], ids=["zero-extent", "list", "array", "empty"])
    def test_zero_length_allocatable_preserves_shape(self, value):
        module = self.module.zero_char_mod
        module.values = value
        assert module.values.dtype == np.dtype("S0")
        assert module.values.shape == (len(value),)
        assert module.values.tobytes() == b""
        assert module.storage_shape() == (len(value), 0)
        module.values = None
        assert module.values is None
