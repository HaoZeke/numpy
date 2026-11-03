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
