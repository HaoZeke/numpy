"""Call-backs reached from a Fortran library built before the extension."""

import importlib
import shutil
import subprocess
import sys
import textwrap

import pytest

from . import util

SOLVE_F90 = util.getpath("tests", "src", "callback", "gh31248_solve.f90")
HOST_F90 = util.getpath("tests", "src", "callback", "gh31248_host.f90")
STEPS = [(1, 0.5), (2, 0.25), (3, 0.125)]


def _need_tools():
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler available")
    if shutil.which("gfortran") is None or shutil.which("ar") is None:
        pytest.skip("gfortran and ar are required")


def _archive(source, directory, object_name):
    subprocess.check_call(
        ["gfortran", "-c", "-fPIC", str(source), "-o", f"{object_name}.o"],
        cwd=directory,
    )
    archive = directory / f"lib{object_name}.a"
    subprocess.check_call(
        ["ar", "rcs", str(archive), f"{object_name}.o"],
        cwd=directory,
    )
    return archive


def _argument_pyf(module):
    return textwrap.dedent(
        f"""
        python module {module}__user__routines
            interface
                subroutine monitor(step, x)
                    integer intent(in) :: step
                    double precision intent(in) :: x
                end subroutine monitor
            end interface
        end python module {module}__user__routines

        python module {module}
            interface
                subroutine solve(n, monitor)
                    use {module}__user__routines
                    integer intent(in) :: n
                    external monitor
                end subroutine solve
            end interface
        end python module {module}
        """
    )


def _host_pyf(module):
    return textwrap.dedent(
        f"""
        python module {module}__user__routines
            interface
                subroutine on_step(step, x)
                    intent(callback) on_step
                    integer intent(in) :: step
                    double precision intent(in) :: x
                end subroutine on_step
            end interface
        end python module {module}__user__routines

        python module {module}
            interface
                subroutine run(n)
                    use {module}__user__routines
                    integer intent(in) :: n
                    intent(callback, hide) on_step
                    external on_step
                end subroutine run
            end interface
        end python module {module}
        """
    )


def _build(workdir, module, pyf_text, link_args):
    pyf = workdir / f"{module}.pyf"
    pyf.write_text(pyf_text)
    cmd = [sys.executable, "-m", "numpy.f2py", "-c", pyf.name, *link_args]
    subprocess.check_call(cmd, cwd=workdir)
    sys.modules.pop(module, None)
    sys.path.insert(0, str(workdir))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(workdir))


def _seen_from_argument(module):
    seen = []
    module.solve(3, lambda step, x: seen.append((int(step), float(x))))
    return seen


@pytest.mark.slow
def test_argument_callback_from_static_archive(tmp_path):
    _need_tools()
    lib = tmp_path / "lib"
    work = tmp_path / "work"
    lib.mkdir()
    work.mkdir()
    archive = _archive(SOLVE_F90, lib, "solve")
    module = _build(work, "cbarg_a", _argument_pyf("cbarg_a"), [str(archive)])
    assert _seen_from_argument(module) == STEPS


@pytest.mark.slow
def test_argument_callback_from_relative_library_dir(tmp_path):
    _need_tools()
    lib = tmp_path / "lib"
    work = tmp_path / "work"
    lib.mkdir()
    work.mkdir()
    _archive(SOLVE_F90, lib, "solve")
    module = _build(
        work, "cbarg_l", _argument_pyf("cbarg_l"), ["-L../lib", "-lsolve"]
    )
    assert _seen_from_argument(module) == STEPS


@pytest.mark.slow
def test_external_callback_from_static_archive(tmp_path):
    _need_tools()
    lib = tmp_path / "lib"
    work = tmp_path / "work"
    lib.mkdir()
    work.mkdir()
    archive = _archive(HOST_F90, lib, "host")
    module = _build(work, "cbext_a", _host_pyf("cbext_a"), [str(archive)])
    seen = []
    module.on_step = lambda step, x: seen.append((int(step), float(x)))
    try:
        module.run(3)
    finally:
        del module.on_step
    assert seen == STEPS
