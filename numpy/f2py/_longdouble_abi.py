"""Detect if Fortran ``c_long_double`` matches C ``long double``.

ifx's ``c_long_double`` is IEEE binary128. The C compiler f2py links
with (clang or gcc on x86_64) uses 80-bit x87 in the same 16 bytes.
Passing one representation as the other yields a NaN or a garbage
value. Padding bits in the x87 16-byte slot differ even when the
value matches, so the probe compares the numeric value, not the raw
bytes.
"""
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

_CACHE = {}

_FORTRAN = """\
subroutine f2py_ld(buf) bind(c, name="f2py_ld")
  use iso_c_binding
  real(c_long_double), intent(out) :: buf
  buf = 3.5_c_long_double
end subroutine
"""

_C = """\
#include <string.h>
void f2py_ld(void *);
int main(void) {
  unsigned char raw[64] = {0};
  long double expect = 3.5L, got;
  f2py_ld(raw);
  memcpy(&got, raw, sizeof got);
  return got == expect ? 0 : 1;
}
"""


def fortran_c_long_double_status(fc=None, cc=None):
    """Return ``('match'|'mismatch'|'unknown', detail)``.

    ``detail`` is empty on match. On mismatch it is a sentence suitable
    as a pytest skip reason and as a build warning.
    """
    fc_argv = shlex.split(fc or os.environ.get('FC', '').strip() or 'gfortran')
    cc_argv = shlex.split(cc or os.environ.get('CC', '').strip() or 'cc')
    key = (tuple(fc_argv), tuple(cc_argv))
    if key not in _CACHE:
        _CACHE[key] = _probe(fc_argv, cc_argv)
    return _CACHE[key]


def _probe(fc_argv, cc_argv):
    try:
        with tempfile.TemporaryDirectory(prefix='f2py-ld-') as tmp:
            directory = Path(tmp)
            fort = directory / 'ld.f90'
            csrc = directory / 'ld.c'
            fort.write_text(_FORTRAN)
            csrc.write_text(_C)
            fo = directory / 'ld.o'
            co = directory / 'ldc.o'
            exe = directory / 'ld.out'
            if _run([*fc_argv, '-c', str(fort), '-o', str(fo)], directory).returncode != 0:
                return 'unknown', f'could not compile a c_long_double probe with {fc_argv[0]}'
            if _run([*cc_argv, '-c', str(csrc), '-o', str(co)], directory).returncode != 0:
                return 'unknown', f'could not compile the C long double probe with {cc_argv[0]}'
            base = Path(fc_argv[0]).name
            extra = ['-nofor-main'] if base in ('ifx', 'ifort') else []
            linked = _run([*fc_argv, *extra, '-o', str(exe), str(fo), str(co)], directory)
            if linked.returncode != 0 or not exe.is_file():
                return 'unknown', f'could not link the c_long_double probe with {fc_argv[0]}'
            ran = _run([str(exe)], directory)
    except (OSError, subprocess.SubprocessError) as exc:
        return 'unknown', f'c_long_double probe failed: {exc}'
    if ran.returncode == 0:
        return 'match', ''
    if ran.returncode == 1:
        return 'mismatch', (
            f"Fortran iso_c_binding c_long_double from {fc_argv[0]} does not "
            f"match C long double from {cc_argv[0]}. The layouts differ "
            "(ifx uses IEEE binary128, while clang/gcc long double on x86_64 "
            "is 80-bit x87 stored in 16 bytes), so a wrapper returns the wrong "
            "bits, often a NaN, instead of the Fortran value."
        )
    return 'unknown', f'c_long_double probe exited {ran.returncode}'


def _run(cmd, cwd):
    return subprocess.run(
        cmd, cwd=cwd, check=False, capture_output=True, text=True, timeout=60,
    )
