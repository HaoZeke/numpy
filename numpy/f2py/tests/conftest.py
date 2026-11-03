import pytest

from . import util


@pytest.fixture(scope="session")
def check_compilers():
    """Return ``util.checker`` after probing for compilers once per session.

    Skips the requesting test where compiled f2py tests cannot run, i.e. on
    Windows (gh-25134) or without meson.
    """
    util.checker.check_compilers()
    return util.checker
