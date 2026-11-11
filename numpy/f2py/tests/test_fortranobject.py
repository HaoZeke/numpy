"""
C unit tests of the f2py runtime, ``numpy/f2py/src/fortranobject.c``.

The tests are written with cmocka in ``src/fortranobject/test_fortranobject.c``.
Each test listed in ``run()`` is one pytest item. The items are skipped when
meson reports that the optional ``cmocka`` dependency is absent. A missing
meson program is a separate skip. Any other setup failure is a test error.
The cmocka group runs in a subprocess so a runtime crash fails that C test
instead of stopping the session.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

from . import util

pytestmark = pytest.mark.slow

SOURCE = util.getpath("tests", "src", "fortranobject", "test_fortranobject.c")
MODULE_NAME = "_test_fortranobject_ext"
_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
# cmocka 1.1.7 prints "ok N - name" and "not ok N - name".
# Its skip line is "ok N # SKIP name".
# Conventional TAP puts the directive after the name:
# "ok N - name # SKIP reason".
_TAP_RESULT = re.compile(rf"^(?P<status>not ok|ok) (?P<num>[0-9]+) (?P<rest>.*)$")
_TAP_SKIP_NAME_ONLY = re.compile(rf"^# SKIP (?P<name>{_NAME})(?: (?P<reason>.*))?$")
_TAP_NAMED = re.compile(
    rf"^- (?P<name>{_NAME})(?: # (?P<directive>SKIP)\b(?P<reason>.*)?)?$"
)
_DEP_NOT_FOUND = re.compile(r"Dependency ['\"]([^'\"]+)['\"] not found")


def c_test_names():
    return re.findall(r"cmocka_unit_test\w*\(\s*(\w+)", SOURCE.read_text())


def classify_meson_setup(dep_name, returncode, output, exc=None):
    """Classify a ``meson setup`` attempt for ``dependency(dep_name)``.

    Returns ``(status, detail)``. ``status`` is ``available``, ``missing``,
    ``toolchain``, or ``config``. Only ``missing`` means that this dependency
    was not found. ``toolchain`` is a missing meson program. ``config`` is any
    other setup failure.
    """
    if isinstance(exc, FileNotFoundError):
        return "toolchain", "meson is not available"
    if isinstance(exc, OSError):
        detail = exc.strerror or str(exc)
        return "config", f"meson setup failed: {detail}"
    if returncode == 0:
        return "available", ""
    missing = _DEP_NOT_FOUND.findall(output)
    if dep_name in missing:
        return "missing", "cmocka is not available"
    error_line = ""
    for line in output.splitlines():
        if "ERROR:" in line:
            error_line = line.strip()
            break
    return "config", error_line or "meson setup failed"


def meson_setup_status(dep_name=None, build_text=None):
    """Run ``meson setup`` and classify it.

    With ``dep_name``, the build file is ``dependency(dep_name)``.
    ``build_text`` replaces that file when a caller needs a different setup.
    """
    if build_text is None:
        build_text = (
            "project('check_dependency', 'c')\n"
            f"dependency('{dep_name}')\n"
        )
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "meson.build"), "w") as handle:
            handle.write(build_text)
        try:
            runmeson = subprocess.run(
                ["meson", "setup", "btmp"],
                check=False,
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            return classify_meson_setup(dep_name or "", None, "", exc)
        output = (runmeson.stdout or "") + (runmeson.stderr or "")
        return classify_meson_setup(
            dep_name or "", runmeson.returncode, output
        )
    finally:
        shutil.rmtree(tmpdir)


def parse_tap(output):
    """Map test names to ``(passed, skipped, diagnostics)``.

    A line that is not a supported result does not count as a pass.
    """
    results = {}
    diagnostics = None
    for raw in output.splitlines():
        line = raw.strip()
        parsed = _parse_tap_result(line)
        if parsed is None:
            if diagnostics is not None and line:
                diagnostics.append(line)
            continue
        name, passed, skipped = parsed
        diagnostics = []
        results[name] = (passed, skipped, diagnostics)
    return results


def _parse_tap_result(line):
    match = _TAP_RESULT.match(line)
    if match is None:
        return None
    rest = match.group("rest")
    failed = match.group("status") == "not ok"
    skip_only = _TAP_SKIP_NAME_ONLY.fullmatch(rest)
    if skip_only is not None:
        return skip_only.group("name"), False, True
    named = _TAP_NAMED.fullmatch(rest)
    if named is None:
        return None
    if named.group("directive") == "SKIP":
        return named.group("name"), False, True
    if failed:
        return named.group("name"), False, False
    return named.group("name"), True, False


@pytest.fixture(scope="module")
def c_test_results():
    status, detail = meson_setup_status("cmocka")
    if status == "missing":
        pytest.skip("cmocka is not available")
    if status == "toolchain":
        pytest.skip(detail)
    if status == "config":
        pytest.fail(detail)
    util.build_meson(
        [SOURCE],
        module_name=MODULE_NAME,
        extra_dat={"dependencies": ["cmocka"]},
    )
    code = (
        f"import sys; sys.path = {sys.path!r}; import {MODULE_NAME}; "
        f"sys.exit({MODULE_NAME}.run())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "CMOCKA_MESSAGE_OUTPUT": "TAP"},
        check=False,
    )
    return parse_tap(proc.stdout), proc


@pytest.mark.parametrize("name", c_test_names())
def test_fortranobject(c_test_results, name):
    results, proc = c_test_results
    if name not in results:
        pytest.fail(
            f"{name} did not run (exit status {proc.returncode})\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    passed, skipped, diagnostics = results[name]
    if skipped:
        pytest.skip(f"{name} was skipped")
    assert passed, "\n".join(diagnostics + [proc.stderr])
