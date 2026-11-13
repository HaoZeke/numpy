import subprocess

from numpy.f2py.tests import util


def test_compiler_probe_timeout_means_no_compiler(monkeypatch):
    # gh-9342: a Fortran compiler that never exits must not hold the suite.
    def stuck(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(util.subprocess, "run", stuck)
    assert util.check_language("fortran") is False
