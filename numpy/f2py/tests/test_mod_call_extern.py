from pathlib import Path

from numpy.f2py.f2py2e import run_main

from . import util


def test_intent_c_module_extern(tmp_path):
    pyf = util.getpath("tests", "src", "mod_call_extern", "sety.pyf")
    run_main(["--quiet", "--build-dir", str(tmp_path), str(pyf)])
    generated = Path(tmp_path, "mod_call_externmodule.c").read_text()
    assert "extern void sety_c (int*,int );" in generated.splitlines()
