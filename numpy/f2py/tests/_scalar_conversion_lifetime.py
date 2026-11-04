"""Execute class ownership checks without test-runner thread history."""

import gc
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import sysconfig
import warnings
import weakref
from pathlib import Path


def _assert_runtime(metadata):
    assert list(sys.version_info[:2]) == metadata["python_version"]
    assert bool(sysconfig.get_config_var("Py_GIL_DISABLED")) is metadata["free_threaded"]
    assert getattr(sys, "_is_gil_enabled", lambda: True)() is metadata["gil_enabled"]
    assert "PYTHON_GIL" not in os.environ
    assert sys.flags.isolated and sys.flags.optimize == 0


def _assert_types_are_collectable():
    class Collectable:
        pass

    reference = weakref.ref(Collectable)
    del Collectable
    gc.collect()
    assert reference() is None, "type lifetime control is affected by interpreter thread history"


def _check_absent_integer_protocol_releases_type_dictionaries(converter):
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


def _check_truncation_protocol_releases_type_dictionary(converter):
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


def main():
    metadata = json.loads(sys.argv[1])
    checks = {
        "test_absent_integer_protocol_releases_type_dictionaries":
            _check_absent_integer_protocol_releases_type_dictionaries,
        "test_truncation_protocol_releases_type_dictionary":
            _check_truncation_protocol_releases_type_dictionary,
    }
    check = checks[metadata["case"]]
    _assert_runtime(metadata)
    assert "pytest" not in sys.modules
    assert "numpy" not in sys.modules
    numpy_origin = Path(metadata["numpy_origin"]).resolve(strict=True)
    assert hashlib.sha256(numpy_origin.read_bytes()).hexdigest() == metadata["numpy_sha256"]
    sys.path.insert(0, str(numpy_origin.parent.parent))
    import numpy

    assert Path(numpy.__file__).resolve(strict=True) == numpy_origin
    assert hashlib.sha256(numpy_origin.read_bytes()).hexdigest() == metadata["numpy_sha256"]
    core_origin = Path(metadata["numpy_core_origin"]).resolve(strict=True)
    core = sys.modules["numpy._core._multiarray_umath"]
    assert Path(core.__file__).resolve(strict=True) == core_origin
    assert hashlib.sha256(core_origin.read_bytes()).hexdigest() == metadata["numpy_core_sha256"]
    origin = Path(metadata["origin"]).resolve(strict=True)
    assert hashlib.sha256(origin.read_bytes()).hexdigest() == metadata["sha256"]
    specification = importlib.util.spec_from_file_location(metadata["module"], origin)
    assert specification is not None
    assert isinstance(specification.loader, importlib.machinery.ExtensionFileLoader)
    assert Path(specification.origin).resolve(strict=True) == origin
    module = importlib.util.module_from_spec(specification)
    sys.modules[metadata["module"]] = module
    specification.loader.exec_module(module)
    assert Path(module.__file__).resolve(strict=True) == origin
    assert module.__name__ == metadata["module"]
    assert hashlib.sha256(origin.read_bytes()).hexdigest() == metadata["sha256"]
    _assert_runtime(metadata)
    _assert_types_are_collectable()
    converter = getattr(module, metadata["function"])
    check(converter)
    _assert_runtime(metadata)
    print(json.dumps({"case": metadata["case"], "module_origin": str(origin),
                      "module_sha256": metadata["sha256"], "function": metadata["function"],
                      "numpy_core_origin": str(core_origin),
                      "numpy_core_sha256": metadata["numpy_core_sha256"],
                      "gil_enabled": metadata["gil_enabled"], "pid": os.getpid(),
                      "converter_calls": 20, "lifetime_assertions_executed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
