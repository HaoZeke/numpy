import subprocess
import sys

import pytest

SRC = """\
module m
contains
  subroutine alpha(n)
    integer, intent(in) :: n
  end subroutine alpha
  subroutine beta(n)
    integer, intent(in) :: n
  end subroutine beta
end module m
"""


@pytest.mark.parametrize("kind,selector", [
    ("only", "m::alpha"),
    ("skip", "m.beta"),
])
def test_qualified_only_skip(tmp_path, kind, selector):
    src = tmp_path / "mod.f90"
    src.write_text(SRC)
    out = tmp_path / "out.pyf"
    if kind == "only":
        files = [str(src)]
    else:
        full = tmp_path / "full.pyf"
        made = subprocess.run(
            [sys.executable, "-m", "numpy.f2py", "-h", str(full), str(src)],
            cwd=tmp_path, capture_output=True, text=True)
        assert made.returncode == 0, made.stderr
        files = [str(full)]
    proc = subprocess.run(
        [sys.executable, "-m", "numpy.f2py", "-h", str(out), *files,
         f"{kind}:", selector],
        cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = out.read_text()
    assert "subroutine alpha" in text
    assert "subroutine beta" not in text
