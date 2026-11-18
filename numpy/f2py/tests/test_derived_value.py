import os

import pytest

from numpy.f2py import f2py2e

from . import util

SKIPPED = (
    "module_value",
    "module_bindc_value",
    "free_value",
)


def _source():
    return util.getpath("tests", "src", "derived_value", "value_derived.f90")


def test_derived_value_skip_message(tmp_path, capfd):
    # Generating the wrapper used to raise KeyError: 'void'.
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        f2py2e.run_main(["-m", "modtest", str(_source())])
    finally:
        os.chdir(cwd)
    out, err = capfd.readouterr()
    for name in SKIPPED:
        assert f"Skipping {name} since dummy argument" in out
    assert "is a derived type with the value attribute" in out
    assert 'Constructing wrapper function "mvalue.add_one"' in out
    assert "assuming void" not in err


@pytest.mark.slow
class TestDerivedValueDummy(util.F2PyTest):
    sources = [util.getpath("tests", "src", "derived_value", "value_derived.f90")]

    def test_unsupported_routines_are_skipped(self):
        assert self.module.mvalue.add_one(3) == 4
        assert not hasattr(self.module.mvalue, "module_value")
        assert not hasattr(self.module.mvalue, "module_bindc_value")
        assert not hasattr(self.module, "free_value")
