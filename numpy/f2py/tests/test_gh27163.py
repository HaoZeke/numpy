import os
import shutil
import sys
from pathlib import Path

import pytest

from numpy.f2py._backends._meson import MesonTemplate

from . import util


def _meson(flags):
    template = MesonTemplate(
        "m",
        [Path("a.f90")],
        [],
        [],
        [],
        [],
        [],
        [],
        flags,
        "release",
        sys.executable,
    )
    return template.generate_meson_build()


def test_openmp_flag_is_a_link_argument():
    # gh-27163
    text = _meson(["-qopenmp"])
    assert "link_args: ['-qopenmp']" in text
    # The visibility keyword is a dict entry, so the quote sits before
    # the colon. Collapse both before matching.
    collapsed = text.replace("'", "").replace(" ", "")
    assert "gnu_symbol_visibility:default" in collapsed
    assert "link_args:" not in _meson([])


def _gfortran_openmp():
    fc = os.environ.get("FC", "gfortran")
    return "flang" not in fc and "ifx" not in fc and "lfortran" not in fc


@pytest.mark.slow
@pytest.mark.skipif(
    not _gfortran_openmp(),
    reason="this check uses gfortran -fopenmp; flang's -fopenmp link line needs libomp",
)
class TestGfortranOpenmpLink(util.F2PyTest):
    # -fopenmp has to be on the link line. -lgomp is not passed.
    sources = [util.getpath("tests", "src", "regression", "gh27163.f90")]
    options = ["--f90flags=-fopenmp"]

    def test_set_num_threads(self):
        self.module.setn(2)


def test_ifx_openmp_without_liomp5(monkeypatch):
    # ifx needs -qopenmp at link time. -liomp5 alone is not enough when
    # the reference is hidden. Skip when ifx itself is absent.
    if shutil.which("ifx") is None:
        pytest.skip("ifx is not installed")
    ifx = Path(shutil.which("ifx")).resolve()
    monkeypatch.setenv("FC", "ifx")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(ifx.parents[1] / "lib"))
    module = util.build_module(
        [util.getpath("tests", "src", "regression", "gh27163.f90")],
        options=["--f90flags=-qopenmp"],
        module_name="gh27163_ifx",
    )
    module.setn(2)
