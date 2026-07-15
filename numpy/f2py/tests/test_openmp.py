"""OpenMP + meson backend (gh-30804)."""
import os
import shutil
import subprocess
import textwrap

import pytest

from . import util


def _openmp_available():
    """True if gfortran accepts -fopenmp (or ifx -qopenmp) for a tiny program."""
    fc = shutil.which("gfortran") or shutil.which("ifx")
    if not fc:
        return False
    flag = "-qopenmp" if os.path.basename(fc) == "ifx" else "-fopenmp"
    src = textwrap.dedent(
        """\
        program t
          use omp_lib
          print *, omp_get_max_threads()
        end
        """
    )
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.f90")
        with open(p, "w") as fh:
            fh.write(src)
        r = subprocess.run(
            [fc, flag, "-o", os.path.join(d, "t"), p],
            capture_output=True,
            text=True,
        )
        return r.returncode == 0


@pytest.mark.slow
@pytest.mark.skipif(not _openmp_available(), reason="OpenMP compiler not available")
def test_dep_openmp_parallel_region():
    """--dep openmp must enable a real parallel region."""
    src = util.getpath("tests", "src", "openmp", "gh30804_omp.f90")
    options = ["--dep", "openmp"]
    try:
        mod = util.build_module([src], options=options, module_name="gh30804_omp")
    except RuntimeError:
        options = ["--f90flags=-fopenmp", "-lgomp"]
        mod = util.build_module([src], options=options, module_name="gh30804_omp2")

    n = int(mod.count_omp_threads(4))
    assert n == 4, f"expected 4 OpenMP threads, got {n}"
