import gc
import importlib.util
import pickle
import re
import subprocess
import sys
import weakref
from importlib.metadata import metadata
from pathlib import Path

import pytest

import numpy as np

from . import util


def test_api_floor_matches_numpy_minimum_python():
    requires_python = metadata("numpy")["Requires-Python"]
    minimum = re.search(r">=(\d+)\.(\d+)", requires_python)
    assert minimum is not None
    major, minor = map(int, minimum.groups())
    floor = f"0x{major:02X}{minor:02X}0000"
    header = util.getpath("src", "fortranobject.h").read_text()
    template = util.getpath("_backends", "meson.build.template").read_text()
    assert f"#define Py_LIMITED_API {floor}" in header
    assert f"f2py_limited_api = '{major}.{minor}'" in template


@pytest.fixture(scope="module")
def limited_api_probe():
    source = util.getpath("tests", "src", "limited_api", "probe.c")
    return util.build_meson([source], module_name="test_f2py_limited_api")


@pytest.fixture(scope="module", params=[False, True])
def limited_api_callback(request):
    source = util.getpath("tests", "src", "callback", "foo.f")
    suffix = "_python_tls" if request.param else ""
    options = ["-DF2PY_USE_PYTHON_TLS"] if request.param else []
    return util.build_module(
        [source], options=options,
        module_name="test_f2py_limited_callback" + suffix,
    )


def test_private_type_layout_is_rejected(capfd):
    source = util.getpath("tests", "src", "limited_api", "probe.c")
    with pytest.raises(subprocess.CalledProcessError) as error:
        util.build_meson(
            [source], module_name="test_f2py_limited_api_reject",
            define_macros=[("F2PY_TEST_PRIVATE_API", None)],
        )
    assert error.value.cmd[1] == "compile"
    output = capfd.readouterr()
    assert "tp_name" in output.out + output.err


def test_stable_abi_filename(limited_api_probe):
    if sys.platform != "win32":
        filename = Path(limited_api_probe.__file__).name
        assert ".abi3." in filename or ".abi3t." in filename


def test_type_behavior(limited_api_probe):
    obj = limited_api_probe.new_fortran()
    cls = type(obj)
    assert cls.__name__ == "fortran"
    assert cls.__module__ == "builtins"
    assert repr(cls) == "<class 'fortran'>"
    assert obj() is None
    assert repr(obj) == "<fortran function identity>"
    assert "__name__" in dir(obj)
    obj.annotation = 7
    assert obj.__dict__["annotation"] == 7
    del obj.annotation
    with pytest.raises(AttributeError):
        obj.annotation
    with pytest.raises(TypeError):
        cls()
    with pytest.raises(TypeError):
        object.__new__(cls)
    with pytest.raises(TypeError):
        pickle.dumps(obj)
    with pytest.raises(TypeError):
        cls.annotation = 1
    with pytest.raises(TypeError):
        type("FortranSubclass", (cls,), {})
    assert gc.is_tracked(obj)


def test_instance_dictionary_cycles_are_collected(limited_api_probe):
    class Token:
        pass

    obj = limited_api_probe.new_fortran()
    token = Token()
    reference = weakref.ref(token)
    token.owner = obj
    obj.token = token
    del token, obj
    gc.collect()
    assert reference() is None


def test_reimport_owns_type_and_releases_module(limited_api_probe):
    spec = importlib.util.spec_from_file_location(
        limited_api_probe.__name__, limited_api_probe.__file__,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module_reference = weakref.ref(module)
    obj = module.new_fortran()
    type_reference = weakref.ref(type(obj))
    del module
    gc.collect()
    assert module_reference() is not None
    assert obj() is None
    del obj
    gc.collect()
    assert module_reference() is None
    assert type_reference() is None
    assert limited_api_probe.new_fortran()() is None


def test_failed_array_constructor_preserves_descriptor(limited_api_probe):
    dtype = np.dtype(np.float64)
    references = sys.getrefcount(dtype)
    for _ in range(100):
        with pytest.raises(ValueError, match="negative dimensions"):
            limited_api_probe.new_invalid_fortran()
    assert sys.getrefcount(dtype) == references
    assert np.empty(1, dtype=dtype).dtype == dtype


@pytest.mark.parametrize("value, expected", [
    (np.int64(4), "i8-int64-scalar"),
    (np.float64(4), "f8-float64-scalar"),
    (np.array(4, dtype=np.int64), "i8-ndarray-scalar"),
    (np.array([1, 2], dtype=np.int64), "(2,)-i8-ndarray"),
])
def test_scalar_and_array_description(limited_api_probe, value, expected):
    assert limited_api_probe.describe(value) == expected


def test_callback_attribute_ownership(limited_api_callback):
    references = []

    class Code:
        pass

    class Callback:
        @property
        def __code__(self):
            code = Code()
            references.append(weakref.ref(code))
            return code

        def __call__(self):
            return 4

    callback = Callback()
    for _ in range(100):
        assert limited_api_callback.t(callback) == 4
    assert len(references) == 100
    assert all(reference() is None for reference in references)


def test_callback_attribute_error_propagates(limited_api_callback):
    class Callback:
        @property
        def __code__(self):
            raise RuntimeError("callback code lookup")

        def __call__(self):
            return 4

    with pytest.raises(RuntimeError, match="callback code lookup"):
        limited_api_callback.t(Callback())


def test_callback_extra_argument_ownership(limited_api_callback):
    token = object()
    extra_args = (token,)

    def callback(arg):
        assert arg is token
        return 4

    callback_refs = sys.getrefcount(callback)
    argument_refs = sys.getrefcount(token)
    for _ in range(100):
        assert limited_api_callback.t(callback, fun_extra_args=extra_args) == 4
    assert sys.getrefcount(callback) == callback_refs
    assert sys.getrefcount(token) == argument_refs


def test_reimport_preserves_callback_module_and_error(limited_api_callback):
    original = limited_api_callback
    original_error = getattr(original, f"_{original.__name__}_error")
    spec = importlib.util.spec_from_file_location(
        original.__name__, original.__file__,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert original_error is not getattr(module, f"_{module.__name__}_error")
    module_reference = weakref.ref(module)
    original.global_f = lambda x: x + 10
    module.global_f = lambda x: x + 20
    try:
        assert original.hidden_callback(2) == 12
        assert module.hidden_callback(2) == 22
        with pytest.raises(original_error):
            original.t(42)
        del module
        gc.collect()
        assert module_reference() is None
        assert original.hidden_callback(2) == 12
        with pytest.raises(original_error):
            original.t(42)
    finally:
        del original.global_f
