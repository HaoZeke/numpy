import sys
import types

import pytest

from . import util


def test_build_module_rejects_imported_name(monkeypatch):
    # Building under an imported name used to return the earlier module
    name = "_test_util_imported_name"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    src = util.getpath("tests", "src", "regression", "inout.f90")
    with pytest.raises(RuntimeError, match="already in use"):
        util.build_module([src], module_name=name)


@pytest.mark.slow
def test_check_language_rejects_a_broken_snippet():
    # A compiler that exists must still reject a snippet that does not compile.
    if not util.check_language("fortran"):
        pytest.skip("No Fortran compiler available")
    broken = "this is not fortran\n"
    assert util.check_language("fortran", broken) is False
    good = "program p\nend program p\n"
    assert util.check_language("fortran", good) is True
