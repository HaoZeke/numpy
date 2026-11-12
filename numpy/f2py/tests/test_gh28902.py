import sys
from pathlib import Path

import pytest

from numpy.f2py._backends._meson import MesonTemplate

from . import util


def test_plain_mpi_dep_requests_fortran():
    # gh-28902
    template = MesonTemplate(
        "m",
        [Path("a.f90")],
        ["mpi", "lapack"],
        [],
        [],
        [],
        [],
        [],
        [],
        "release",
        sys.executable,
    )
    text = template.generate_meson_build()
    assert "dependency('mpi', language: 'fortran')" in text
    assert "dependency('lapack')" in text


def test_fortran_mpi_imports():
    # gfortran can build this. The failure is an undefined
    # mpi_comm_rank_ at import, which needs an MPI library.
    try:
        module = util.build_module(
            [util.getpath("tests", "src", "regression", "gh28902.f90")],
            options=["--dep", "mpi"],
            module_name="gh28902_mpi",
        )
    except RuntimeError as exc:
        message = str(exc)
        # OpenMPI's mpif.h is generated for gfortran. flang rejects its
        # REAL128 kinds, and a missing MPI install cannot build this.
        if (
            "Dependency" in message and "mpi" in message
            or "not a supported type" in message
            or "mpif.h" in message
        ):
            pytest.skip("Fortran MPI headers are not usable with this compiler")
        raise
    assert hasattr(module, "rank_of")
