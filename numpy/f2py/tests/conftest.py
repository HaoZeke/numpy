"""LFortran 0.69 skips that do not belong on any one f2py feature branch.

The tests below live on several already-pushed branches. Skipping them
here keeps this branch linear: collection marks the node, whichever
version of the test file the merge produced.
"""
import os

import pytest


def _lfortran():
    return "lfortran" in os.environ.get("FC", "")


# node id suffix -> reason. A file path skips every test in that file.
_SKIPS = (
    (
        "test_character.py",
        "LFortran 0.69 segfaults on character array arguments",
    ),
    (
        "test_modules.py::TestAllocatableCharacterArray",
        "LFortran 0.69 segfaults while reading an allocatable character array",
    ),
    (
        "test_modules.py::TestAllocatableOnlyModule",
        "LFortran 0.69 segfaults on a module that only contains an allocatable",
    ),
    (
        "test_modules.py::TestStrictSetattr",
        "LFortran 0.69 segfaults when a wrapped module name is assigned",
    ),
    (
        "test_optional_absent.py::TestOptionalAbsent",
        "LFortran 0.69 segfaults in the optional-argument wrapper",
    ),
    (
        "test_optional_absent.py::TestPyfAbsent",
        "LFortran 0.69 segfaults in a signature-file optional wrapper",
    ),
    (
        "test_optional_absent.py::TestKeepPreviousCall",
        "LFortran 0.69 segfaults when an optional character argument is omitted",
    ),
    (
        "test_optional_absent.py::TestOptionalCallbackBuilds",
        "LFortran 0.69 segfaults when an optional callback runs",
    ),
    (
        "test_language_support.py::TestF90Allocatable",
        "LFortran 0.69 segfaults when Python allocates a module array",
    ),
    (
        "test_language_support.py::TestF2003Features",
        "LFortran 0.69 defines the module variable twice",
    ),
    (
        "test_allocatable_args.py::TestAllocatableArguments",
        "LFortran 0.69 does not apply an ALLOCATABLE statement to a deferred-shape local",
    ),
    (
        "test_check_format.py::TestCheckFormat",
        "LFortran 0.69 sets c_long_double_complex to -1, so this source does not compile",
    ),
    (
        "test_docs.py::TestDocAdvanced",
        "LFortran 0.69 appends an underscore to bind(c) f2py_check_signals",
    ),
    (
        "test_entry.py::TestEntryInResultFunction",
        "LFortran 0.69 returns 0 from an ENTRY instead of the entry's value",
    ),
    (
        "test_isoc.py::TestISOC",
        "LFortran 0.69 sets c_long_double_complex to -1, so this module does not compile",
    ),
    (
        "test_regression.py::test_gh30804_opt_openmp_flag",
        "LFortran 0.69 does not accept -fopenmp",
    ),
    (
        "test_regression.py::test_gh25784",
        "LFortran 0.69 rejects -ffixed-form",
    ),
    (
        "test_regression.py::test_gh13356_entry_return",
        "LFortran 0.69 returns 0 from an ENTRY instead of the entry's value",
    ),
    (
        "test_return_character.py::TestFReturnCharacterNoWrap",
        "LFortran 0.69 segfaults on a direct CHARACTER function return",
    ),
    (
        "test_signals.py::TestCheckSignals",
        "LFortran 0.69 appends an underscore to a bind(c) name "
        "when --mangle-underscore-external is set",
    ),
    (
        "test_size.py::TestSizeSumExample",
        "LFortran 0.69 requires a dummy to be declared before it is used in a dimension",
    ),
    (
        "test_string.py::TestLargeString::test_string_array_len",
        "LFortran 0.69 segfaults on a very long character array",
    ),
)

_XFAILS = (
    (
        "test_kind.py::TestKind::test_int",
        "LFortran has no integer kind for 19 decimal digits, so "
        "selected_int_kind(19) is -1. f2py's static kind table uses "
        "gfortran's 16.",
    ),
    (
        "test_kind.py::TestKind::test_real",
        "LFortran reports selected_real_kind(16) as 16, IEEE quad. "
        "gfortran's is 10, 80-bit extended precision. f2py's static "
        "kind table follows gfortran.",
    ),
)


def _matches(nodeid, key):
    at = nodeid.find(key.split("::", 1)[0])
    if at < 0:
        return False
    rest = nodeid[at:]
    return rest == key or rest.startswith(key + "::") or rest.startswith(key + "[")


def pytest_collection_modifyitems(config, items):
    if not _lfortran():
        return
    for item in items:
        for key, reason in _XFAILS:
            if _matches(item.nodeid, key):
                item.add_marker(pytest.mark.xfail(reason=reason))
        for key, reason in _SKIPS:
            if _matches(item.nodeid, key):
                item.add_marker(pytest.mark.skip(reason=reason))
