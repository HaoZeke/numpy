import subprocess
import sys

import pytest

from numpy.f2py.tests import util

SRC = """\
subroutine add1(n, m)
  integer, intent(in) :: n
  integer, intent(out) :: m
  m = n + 1
end subroutine add1
"""


@pytest.mark.slow
@pytest.mark.parametrize("order", ["h-first", "h-last"])
def test_compile_writes_signature(tmp_path, order):
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler found")
    src = tmp_path / "add1.f90"
    src.write_text(SRC)
    sig = tmp_path / "add1.pyf"
    bdir = tmp_path / "build"
    common = ["-c", "-m", "addsig", "--build-dir", str(bdir)]
    signature = ["-h", str(sig)]
    args = signature + common if order == "h-first" else common + signature
    cmd = [sys.executable, "-m", "numpy.f2py", *args, str(src)]
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = sig.read_text()
    assert "python module addsig" in text
    assert "m = n + 1" not in text
    imported = subprocess.run(
        [sys.executable, "-c", "import addsig; print(addsig.add1(3))"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == "4"


@pytest.mark.parametrize("order", ["h-first", "h-last"])
def test_compile_refuses_existing_signature(tmp_path, order):
    src = tmp_path / "add1.f90"
    src.write_text(SRC)
    sig = tmp_path / "sig.pyf"
    original = "sentinel signature\n"
    sig.write_text(original)
    common = ["-c", "-m", "addsig"]
    signature = ["-h", sig.name]
    args = signature + common if order == "h-first" else common + signature
    cmd = [sys.executable, "-m", "numpy.f2py", *args, src.name]
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert "Use --overwrite-signature to overwrite" in proc.stderr
    assert sig.read_text() == original
    assert list(tmp_path.glob("addsig*.so")) == []


@pytest.mark.slow
def test_compile_overwrites_signature(tmp_path):
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler found")
    src = tmp_path / "add1.f90"
    src.write_text(SRC)
    sig = tmp_path / "sig.pyf"
    original = "sentinel signature\n"
    sig.write_text(original)
    cmd = [
        sys.executable, "-m", "numpy.f2py",
        "-c", "-m", "addsig", "-h", sig.name,
        "--overwrite-signature", src.name,
    ]
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = sig.read_text()
    assert text != original
    assert "python module addsig" in text
    imported = subprocess.run(
        [sys.executable, "-c", "import addsig; print(addsig.add1(3))"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == "4"
