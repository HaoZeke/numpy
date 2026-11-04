"""Numeric protocol failures must not select a sequence's first element."""

import gc
import sys
import warnings
import weakref

import numpy as np
import pytest

from . import util


@pytest.fixture(scope="module")
def scalar_modules(tmp_path_factory):
    source = util.getpath("tests", "src", "regression", "gh24394.f90")
    native = util.build_module([source])
    size = np.dtype("l").itemsize
    mapping = tmp_path_factory.mktemp("scalar-conversion") / "f2cmap"
    mapping.write_text(repr({"integer": {str(size): "long"}}), encoding="ascii")
    name = f"integer{size}"
    c_long = util.build_module(
        [source], options=["--f2cmap", str(mapping)], only=[name]
    )
    return native, c_long, name


@pytest.fixture(params=["integer4", "integer8", "real4", "real8", "c-long"])
def converter(request, scalar_modules):
    native, c_long, long_name = scalar_modules
    if request.param == "c-long":
        return getattr(c_long, long_name)
    return getattr(native, request.param)


class Sequence:
    def __init__(self, item=2):
        self.item = item
        self.reads = 0

    def __getitem__(self, index):
        self.reads += 1
        if index != 0:
            raise IndexError(index)
        return self.item


class ProtocolSequence(Sequence):
    def __init__(self, error, item=2):
        super().__init__(item)
        self.error = error
        self.conversions = 0

    def __float__(self):
        self.conversions += 1
        raise self.error

    def __int__(self):
        self.conversions += 1
        raise self.error


class WarningSequence(Sequence):
    def __init__(self, category):
        super().__init__()
        self.category = category

    def __float__(self):
        warnings.warn("numeric conversion warning", self.category, stacklevel=1)
        return 2.0

    def __int__(self):
        warnings.warn("numeric conversion warning", self.category, stacklevel=1)
        return 2


class ConversionWarning(DeprecationWarning):
    pass


class ConversionTypeWarning(DeprecationWarning, TypeError):
    pass


@pytest.mark.parametrize(
    "value", [2, 2.0, np.int32(2), np.int64(2), np.float32(2), np.float64(2),
              np.array(2.0), "2"]
)
def test_supported_scalar_inputs(converter, value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert converter(value) == 2


@pytest.mark.parametrize("value", [[2, 9], (2, 9), [[2, 9], 8], 2 + 3j])
def test_nonnumeric_fallback_compatibility(converter, value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert converter(value) == 2


def test_sequence_without_numeric_protocol(converter):
    value = Sequence()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert converter(value) == 2
    assert value.reads == 1


@pytest.mark.parametrize("value", [bytearray((2, 9)), memoryview(bytes((2, 9)))])
def test_nonnumeric_buffer_fallback_compatibility(converter, value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert converter(value) == 2


@pytest.mark.parametrize("value", [bytearray(b"2"), memoryview(b"2")])
def test_successful_buffer_numeric_conversion(converter, value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert converter(value) == 2


def test_absent_integer_protocol_releases_type_dictionaries(converter):
    class BaseSequence:
        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return 2

    class ChildSequence(BaseSequence):
        pass

    value = ChildSequence()
    references = [weakref.ref(BaseSequence), weakref.ref(ChildSequence)]
    for _ in range(20):
        assert converter(value) == 2
    del value, ChildSequence, BaseSequence
    gc.collect()
    assert all(reference() is None for reference in references)


def test_truncation_protocol_releases_type_dictionary(converter):
    class TruncatedSequence:
        def __trunc__(self):
            return 2

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return 2

    reference = weakref.ref(TruncatedSequence)
    value = TruncatedSequence()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for _ in range(20):
            assert converter(value) == 2
    del value, TruncatedSequence
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("category", [ConversionWarning, ConversionTypeWarning])
def test_numeric_warning_as_error_is_preserved(converter, category):
    value = WarningSequence(category)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(category, match="numeric conversion warning"):
            converter(value)
    assert value.reads == 0


@pytest.mark.parametrize("category", [TypeError, ValueError, RuntimeError,
                                     OverflowError, MemoryError])
def test_numeric_protocol_exception_is_preserved(converter, category):
    error = category("original numeric failure")
    value = ProtocolSequence(error)
    with pytest.raises(category, match="original numeric failure") as caught:
        converter(value)
    assert caught.value is error
    assert value.conversions == 1
    assert value.reads == 0


def test_inherited_index_failure_is_preserved(converter):
    error = TypeError("original index failure")

    class IndexedSequence(Sequence):
        def __index__(self):
            raise error

    class Child(IndexedSequence):
        pass

    value = Child()
    with pytest.raises(TypeError, match="original index failure") as caught:
        converter(value)
    assert caught.value is error
    assert value.reads == 0


def test_invalid_numeric_protocol_result_is_preserved(converter):
    class InvalidSequence(Sequence):
        def __float__(self):
            return object()

        def __int__(self):
            return object()

    value = InvalidSequence()
    with pytest.raises(TypeError):
        converter(value)
    assert value.reads == 0


def test_complex_subclass_protocol_failure_is_preserved(converter):
    error = TypeError("complex numeric failure")

    class ComplexValue(complex):
        def __float__(self):
            raise error

        def __int__(self):
            raise error

    with pytest.raises(TypeError, match="complex numeric failure") as caught:
        converter(ComplexValue(2, 3))
    assert caught.value is error


def test_nested_protocol_failure_is_preserved(converter):
    error = TypeError("nested numeric failure")
    value = ProtocolSequence(error)
    with pytest.raises(TypeError, match="nested numeric failure") as caught:
        converter([[value]])
    assert caught.value is error
    assert value.reads == 0


def test_numpy_numeric_conversion_error_is_not_bypassed(converter):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError):
            converter(np.array([2, 9]))


def test_numpy_conversion_warning_as_error_is_not_bypassed(converter):
    class WarningArray(np.ndarray):
        def __float__(self):
            warnings.warn("array scalar conversion", ConversionWarning,
                          stacklevel=1)
            return 2.0

        def __int__(self):
            warnings.warn("array scalar conversion", ConversionWarning,
                          stacklevel=1)
            return 2

    value = np.array([2]).view(WarningArray)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ConversionWarning, match="array scalar conversion"):
            converter(value)


def test_warning_handler_failure_is_preserved(converter, monkeypatch):
    error = RuntimeError("warning handler failed")

    def showwarning(*args, **kwargs):
        raise error

    value = WarningSequence(ConversionWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("always", ConversionWarning)
        monkeypatch.setattr(warnings, "showwarning", showwarning)
        with pytest.raises(RuntimeError, match="warning handler failed") as caught:
            converter(value)
    assert caught.value is error
    assert value.reads == 0


def test_sequence_item_exception_is_preserved(converter):
    error = LookupError("sequence item failure")

    class BrokenSequence(Sequence):
        def __getitem__(self, index):
            raise error

    with pytest.raises(LookupError, match="sequence item failure") as caught:
        converter(BrokenSequence())
    assert caught.value is error


def test_complex_real_attribute_exception_is_preserved(converter):
    error = RuntimeError("complex real attribute failure")

    class ComplexValue(complex):
        @property
        def real(self):
            raise error

    with pytest.raises(RuntimeError, match="complex real attribute failure") as caught:
        converter(ComplexValue(2, 3))
    assert caught.value is error


def test_fallback_reference_ownership(converter):
    class Scalar:
        def __float__(self):
            return 2.0

        def __int__(self):
            return 2

    item = Scalar()
    value = Sequence(item)
    reference = weakref.ref(item)
    for _ in range(50):
        assert converter(value) == 2
    del item, value
    gc.collect()
    assert reference() is None


def test_failed_fallback_reference_ownership(converter):
    class Scalar:
        def __float__(self):
            raise RuntimeError("item conversion failed")

        def __int__(self):
            raise RuntimeError("item conversion failed")

    def convert(value):
        with pytest.raises(RuntimeError, match="item conversion failed"):
            converter(value)

    item = Scalar()
    value = Sequence(item)
    reference = weakref.ref(item)
    for _ in range(50):
        convert(value)
    del item, value
    gc.collect()
    assert reference() is None


def test_cyclic_sequence_raises_recursion_error(converter):
    value = []
    value.append(value)
    try:
        with pytest.raises(RecursionError):
            converter(value)
        assert converter(2) == 2
    finally:
        value.clear()


def test_integer_truncation_protocol_error(scalar_modules):
    native, c_long, long_name = scalar_modules
    error = TypeError("truncation protocol failed")

    class TruncatedSequence(Sequence):
        def __trunc__(self):
            raise error

    for converter in (native.integer4, native.integer8, getattr(c_long, long_name)):
        value = TruncatedSequence()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            if sys.version_info < (3, 14):
                with pytest.raises(
                    TypeError, match="truncation protocol failed"
                ) as caught:
                    converter(value)
                assert caught.value is error
                assert value.reads == 0
            else:
                assert converter(value) == 2
                assert value.reads == 1
