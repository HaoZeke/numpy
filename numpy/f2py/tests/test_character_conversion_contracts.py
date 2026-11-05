import pytest

import numpy as np

from . import util


class CannotConvertCharacter:
    def __str__(self):
        raise ValueError("character conversion failed")


@pytest.mark.slow
class TestCharacterConversionContracts(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "common", "gh9370.f"),
        util.getpath("tests", "src", "common", "gh9370_zero_length.f90"),
        util.getpath("tests", "src", "common", "gh9370_dynamic_lengths.f90"),
    ]

    @pytest.mark.parametrize("shape, common, field, guard", [
        ((), "zero_scalar", "scalar_value", "scalar_guard"),
        ((3,), "zero_array", "array_values", "array_guard"),
    ], ids=["scalar", "array"])
    def test_raw_zero_item_descriptor(self, shape, common, field, guard):
        self.module.init_zero_common()
        value = np.ndarray(shape, dtype="S0", buffer=bytearray(1))
        assert value.itemsize == 0
        assert value.dtype == np.dtype("S0")
        block = getattr(self.module, common)
        view = getattr(block, field)
        setattr(block, field, value)
        assert getattr(block, field) is view
        assert view.shape == shape
        assert view.dtype == np.dtype("S0")
        assert view.tobytes() == b""
        assert getattr(block, guard).tobytes() == b"abcdefghijklmnop"

    @pytest.mark.parametrize("count", [2, 4])
    def test_raw_zero_item_descriptor_rejects_wrong_shape(self, count):
        self.module.init_zero_common()
        value = np.ndarray((count,), dtype="S0", buffer=bytearray(1))
        assert value.itemsize == 0
        with pytest.raises(ValueError):
            self.module.zero_array.array_values = value
        assert self.module.zero_array.array_guard.tobytes() == b"abcdefghijklmnop"

    @pytest.mark.parametrize("target", ["fixed", "scalar", "array", "allocatable"])
    def test_conversion_error_preserves_storage(self, target):
        self.module.initblock()
        self.module.init_zero_common()
        self.module.zero_char_mod.values = [b"A", b"B"]
        bad = CannotConvertCharacter()
        with pytest.raises(ValueError, match="character conversion failed"):
            if target == "fixed":
                self.module.block.block_string = bad
            elif target == "scalar":
                self.module.zero_scalar.scalar_value = bad
            elif target == "array":
                self.module.zero_array.array_values = [bad, bad, bad]
            else:
                self.module.zero_char_mod.values = [bad]
        assert self.module.block.block_string.tobytes() == b"abc"
        assert self.module.block.guard.tobytes() == b"xyz"
        assert self.module.zero_scalar.scalar_guard.tobytes() == b"abcdefghijklmnop"
        assert self.module.zero_array.array_guard.tobytes() == b"abcdefghijklmnop"
        assert self.module.zero_allocation_state() == (1, 2, 0)
        self.module.zero_char_mod.values = None
        assert self.module.zero_char_mod.values is None

    @pytest.mark.parametrize("target", ["fixed", "zero-width"])
    def test_encoding_error_precedes_zero_width_return(self, target):
        self.module.initblock()
        self.module.init_zero_common()
        with pytest.raises(UnicodeEncodeError):
            if target == "fixed":
                self.module.block.block_string = "\u00e9"
            else:
                self.module.zero_scalar.scalar_value = "\u00e9"
        assert self.module.block.block_string.tobytes() == b"abc"
        assert self.module.block.guard.tobytes() == b"xyz"
        assert self.module.zero_scalar.scalar_guard.tobytes() == b"abcdefghijklmnop"

    @pytest.mark.parametrize("values, width, expected", [
        ([b"ab", b"C"], 2, b"abC\x00"),
        ([b"ABCD", b""], 4, b"ABCD\x00\x00\x00\x00"),
        (np.array([b"ab", b"C"], dtype="S7"), 7,
         b"ab" + b"\x00" * 5 + b"C" + b"\x00" * 6),
    ], ids=["inferred-two", "inferred-four", "declared-seven"])
    def test_runtime_string_width_is_retained(self, values, width, expected):
        result, actual_width = self.module.echo_runtime_strings(values)
        assert actual_width == width
        assert result.shape == (2,)
        assert result.dtype == np.dtype(f"S{width}")
        assert result.tobytes() == expected

    def test_zero_extent_assignment_and_none_are_distinct(self):
        module = self.module.zero_char_mod
        module.values = [b"A", b"B"]
        assert self.module.zero_allocation_state() == (1, 2, 0)
        module.values = []
        assert module.values.shape == (0,)
        assert module.values.dtype == np.dtype("S0")
        assert self.module.zero_allocation_state() == (1, 0, 0)
        module.values = None
        assert module.values is None
        assert self.module.zero_allocation_state() == (0, -1, -1)
        module.values = []
        assert module.values.shape == (0,)
        assert module.values.dtype == np.dtype("S0")
        assert self.module.zero_allocation_state() == (1, 0, 0)
        module.values = None
        assert module.values is None

    @pytest.mark.parametrize("count", [0, 2])
    def test_raw_zero_item_allocatable(self, count):
        value = np.ndarray((count,), dtype="S0", buffer=bytearray(1))
        assert value.itemsize == 0
        self.module.zero_char_mod.values = value
        assert self.module.zero_char_mod.values.shape == (count,)
        assert self.module.zero_char_mod.values.dtype == np.dtype("S0")
        assert self.module.zero_char_mod.values.tobytes() == b""
        assert self.module.zero_allocation_state() == (1, count, 0)
        self.module.zero_char_mod.values = None
        assert self.module.zero_char_mod.values is None
        assert self.module.zero_allocation_state() == (0, -1, -1)
