import functools
import os
import subprocess
import tempfile

import pytest

from . import util


@functools.cache
def meson_finds_openmp():
    # Same lookup as the dependency('openmp') that --dep openmp generates
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "meson.build"), "w") as f:
            f.write("project('check_openmp', ['c', 'fortran'])\n")
            f.write("dependency('openmp')\n")
        try:
            runmeson = subprocess.run(
                ["meson", "setup", "btmp"],
                check=False,
                cwd=tmpdir,
                capture_output=True,
            )
        except OSError:
            return False
        return runmeson.returncode == 0


@pytest.mark.slow
class TestOpenMP(util.F2PyTest):
    # gh-30804
    sources = [util.getpath("tests", "src", "openmp", "omp_team_size.f90")]
    options = ["--dep", "openmp"]

    def setup_method(self):
        if not meson_finds_openmp():
            pytest.skip("meson does not find OpenMP for these compilers")
        super().setup_method()

    def test_parallel_region(self):
        assert self.module.omp_team_size(2) == 2
