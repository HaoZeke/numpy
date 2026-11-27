import shutil
import sys

import pytest

from . import util


@pytest.fixture(scope="session")
def check_compilers():
    """Return ``util.checker`` after probing for compilers once per session.

    The skip runs in this fixture, on the main thread. ``check_language``
    probes from worker threads and only returns a boolean.
    """
    if sys.platform == "win32":
        pytest.skip("No Fortran tests on Windows (Issue #25134)")
    if shutil.which("meson") is None:
        pytest.skip("meson not present, skipping compiler dependent test")
    if not util.IS_WASM and not util.IS_IOS:
        util.checker.check_compilers()
    return util.checker
