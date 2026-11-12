"""
C unit tests of the f2py runtime, ``numpy/f2py/src/fortranobject.c``.

The tests are written with cmocka in ``src/fortranobject/test_fortranobject.c``
and skipped when meson cannot find cmocka.  They run in a subprocess, so that a
crash in the runtime fails the C test instead of stopping the test session.
"""
import os
import re
import subprocess
import sys

import pytest

from . import util

pytestmark = pytest.mark.slow

SOURCE = util.getpath("tests", "src", "fortranobject", "test_fortranobject.c")
MODULE_NAME = "_test_fortranobject_ext"
TAP_TEST_LINE = re.compile(r"(not )?ok \d+ (?:- )?(# SKIP )?(\w+)")


def c_test_names():
    return re.findall(r"cmocka_unit_test\w*\(\s*(\w+)", SOURCE.read_text())


def parse_tap(output):
    """Map test names to (passed, skipped, diagnostics) from cmocka TAP output.
    """
    results = {}
    diagnostics = None
    for line in output.splitlines():
        match = TAP_TEST_LINE.match(line)
        if match:
            diagnostics = []
            results[match.group(3)] = (match.group(1) is None,
                                       match.group(2) is not None,
                                       diagnostics)
        elif diagnostics is not None:
            diagnostics.append(line)
    return results


@pytest.fixture(scope="module")
def c_test_results():
    if not util.has_meson_dependency("cmocka"):
        pytest.skip("cmocka is not available")
    util.build_meson([SOURCE], module_name=MODULE_NAME,
                     extra_dat={"dependencies": ["cmocka"]})
    code = (f"import sys; sys.path = {sys.path!r}; import {MODULE_NAME}; "
            f"sys.exit({MODULE_NAME}.run())")
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
        pytest.fail(f"{name} did not run (exit status {proc.returncode})\n"
                    f"{proc.stdout}\n{proc.stderr}")
    passed, skipped, diagnostics = results[name]
    if skipped:
        pytest.skip(f"{name} was skipped")
    assert passed, "\n".join(diagnostics + [proc.stderr])
