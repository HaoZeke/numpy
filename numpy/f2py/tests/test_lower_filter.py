import sys
from pathlib import Path

from numpy.f2py.f2py2e import main as f2pycli

SRC = """
      subroutine HELLO(n)
      integer n
      n = 3
      end
      subroutine KEEP(n)
      integer n
      n = 4
      end
"""


def _generate(tmp_path, monkeypatch, *filter_args):
    src = tmp_path / "up.f"
    src.write_text(SRC)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["f2py", "--lower", "-m", "upmod", "up.f", *filter_args],
    )
    f2pycli()
    return Path("upmodmodule.c").read_text()


def test_only_follows_lower(tmp_path, monkeypatch):
    text = _generate(tmp_path, monkeypatch, "only:", "KEEP")
    assert "f2py_rout_upmod_keep" in text
    assert "f2py_rout_upmod_hello" not in text


def test_skip_follows_lower(tmp_path, monkeypatch):
    text = _generate(tmp_path, monkeypatch, "skip:", "HELLO")
    assert "f2py_rout_upmod_hello" not in text
    assert "f2py_rout_upmod_keep" in text
