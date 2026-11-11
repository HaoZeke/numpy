"""Checks for the cmocka runner's dependency probe and TAP parsing.

These tests do not build ``fortranobject.c``. The 25 cmocka cases stay in
``test_fortranobject.py``.
"""
import pytest

from numpy.f2py.tests.test_fortranobject import (
    classify_meson_setup,
    meson_setup_status,
    parse_tap,
)


@pytest.mark.parametrize(
    ("line", "name", "passed", "skipped"),
    [
        ("ok 1 - test_alpha", "test_alpha", True, False),
        ("not ok 2 - test_beta", "test_beta", False, False),
        ("ok 3 - test_gamma # SKIP", "test_gamma", False, True),
        (
            "ok 4 - test_delta # SKIP needs a feature",
            "test_delta",
            False,
            True,
        ),
        ("ok 5 # SKIP test_epsilon", "test_epsilon", False, True),
        (
            "not ok 6 - test_zeta # SKIP not built",
            "test_zeta",
            False,
            True,
        ),
    ],
)
def test_tap_supported_lines(line, name, passed, skipped):
    results = parse_tap(line + "\n# detail\n")
    assert set(results) == {name}
    got_passed, got_skipped, diagnostics = results[name]
    assert got_passed is passed
    assert got_skipped is skipped
    assert diagnostics == ["# detail"]


@pytest.mark.parametrize(
    "line",
    [
        "ok 1 - test_name # skip lower case",
        "ok 1 - test_name leftover",
        "ok 1 -",
        "ok",
        "OK 1 - test_name",
        "ok 1 - test_name # TODO later",
    ],
)
def test_tap_malformed_is_not_a_pass(line):
    assert parse_tap(line) == {}


def test_classify_missing_dependency_is_not_a_setup_error():
    output = (
        'meson.build:2:0: ERROR: Dependency "cmocka" not found '
        "(tried pkg-config and cmake)\n"
    )
    assert classify_meson_setup("cmocka", 1, output) == (
        "missing",
        "cmocka is not available",
    )


def test_classify_other_missing_dependency_is_not_cmocka():
    output = 'ERROR: Dependency "zlib" not found\n'
    status, _detail = classify_meson_setup("cmocka", 1, output)
    assert status == "config"


def test_classify_meson_missing_is_toolchain():
    status, detail = classify_meson_setup(
        "cmocka", None, "", FileNotFoundError()
    )
    assert status == "toolchain"
    assert detail == "meson is not available"


def test_classify_oserror_is_config():
    status, detail = classify_meson_setup(
        "cmocka", None, "", OSError(1, "Permission denied")
    )
    assert status == "config"
    assert detail != "cmocka is not available"
    assert "Permission denied" in detail


def test_classify_compiler_style_error_is_not_missing_cmocka():
    output = "ERROR: Unknown compiler(s): [['gcc']]\n"
    status, detail = classify_meson_setup("cmocka", 1, output)
    assert status == "config"
    assert "cmocka is not available" not in detail


def test_real_meson_finds_installed_cmocka():
    status, detail = meson_setup_status("cmocka")
    assert status == "available"
    assert detail == ""


def test_real_meson_reports_missing_optional_name():
    status, detail = meson_setup_status("cmocka-not-installed")
    assert status == "missing"
    assert detail == "cmocka is not available"


def test_real_meson_syntax_error_is_config():
    status, detail = meson_setup_status(
        dep_name="cmocka",
        build_text="project('broken', 'c')\nthis is not meson\n",
    )
    assert status == "config"
    assert detail != "cmocka is not available"
