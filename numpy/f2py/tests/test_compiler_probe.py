import subprocess

from numpy.f2py.tests import util


def test_windows_uses_the_compiler_probe(monkeypatch):
    # gh-25134: a Windows run with a working compiler must not be skipped.
    monkeypatch.setattr(util.sys, "platform", "win32")

    def succeed(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(util.subprocess, "run", succeed)
    assert util.check_language("fortran") is True


def test_windows_skips_when_the_probe_fails(monkeypatch):
    monkeypatch.setattr(util.sys, "platform", "win32")

    def fail(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(util.subprocess, "run", fail)
    assert util.check_language("fortran") is False
