"""Compiler-aware selected_int_kind / selected_real_kind.

Kind numbers are processor-dependent. The tables below are gfortran's
numbering and are used only when ``FC`` is unset or the probe cannot
run. Otherwise the active compiler is asked, and a kind is mapped to a
C type by storage size (and, for reals, decimal precision), not by the
kind integer itself. flang's ``selected_real_kind(0) == 2`` is half
precision and is left unmapped: f2py has no 16-bit real C type.
"""
import os
import platform
import shlex
import subprocess
import tempfile
from pathlib import Path

_CACHE = {}
_OVERRIDE = None

_REAL_CTYPE = {32: 'float', 64: 'double', 128: 'long_double'}
_INT_CTYPE = {8: 'signed_char', 16: 'short', 32: 'int', 64: 'long_long'}


def selected_int_kind_fallback(r):
    # gfortran kind numbers. Used when the compiler probe is unavailable.
    m = 10 ** r
    if m <= 2 ** 8:
        return 1
    if m <= 2 ** 16:
        return 2
    if m <= 2 ** 32:
        return 4
    if m <= 2 ** 63:
        return 8
    if m <= 2 ** 128:
        return 16
    return -1


def selected_real_kind_fallback(p, r=0, radix=0):
    # gfortran-like kind numbers, including 80-bit extended as kind 10
    # on x86. Ignored when a compiler probe is available.
    if p < 7:
        return 4
    if p < 16:
        return 8
    machine = platform.machine().lower()
    if machine.startswith((
            'aarch64', 'alpha', 'arm64', 'loongarch', 'mips',
            'power', 'ppc', 'riscv', 's390x', 'sparc')):
        if p <= 33:
            return 16
    elif p < 19:
        return 10
    elif p <= 33:
        return 16
    return -1


def clear_kind_cache():
    _CACHE.clear()


def override_kind_table(table):
    """Install a fake probe result. ``None`` restores normal lookup."""
    global _OVERRIDE
    _OVERRIDE = table


def _fortran_compiler():
    fc = os.environ.get('FC', '').strip()
    if not fc:
        return None
    return shlex.split(fc)


def active_kind_table():
    if _OVERRIDE is not None:
        return _OVERRIDE
    argv = _fortran_compiler()
    if not argv:
        return None
    key = tuple(argv)
    if key in _CACHE:
        return _CACHE[key]
    table = _probe_compiler(argv)
    _CACHE[key] = table
    return table


def _run(cmd, cwd):
    return subprocess.run(
        cmd, cwd=cwd, check=False, capture_output=True, text=True, timeout=60,
    )


def _probe_compiler(argv):
    try:
        with tempfile.TemporaryDirectory(prefix='f2py-kind-') as tmp:
            directory = Path(tmp)
            kinds = _probe_selected(argv, directory)
            if kinds is None:
                return None
            traits = _probe_traits(argv, directory, kinds)
            if traits is None:
                return None
            return {'int': kinds['int'], 'real': kinds['real'], 'traits': traits}
    except (OSError, subprocess.SubprocessError):
        return None


def _probe_selected(argv, directory):
    source = directory / 'selected.f90'
    source.write_text(
        "program f2py_kind_selected\n"
        "  implicit none\n"
        "  integer :: p\n"
        "  do p = 0, 40\n"
        "    write(*,'(A,1X,I2,1X,I5)') 'I', p, selected_int_kind(p)\n"
        "    write(*,'(A,1X,I2,1X,I5)') 'R', p, selected_real_kind(p, 0)\n"
        "  end do\n"
        "end program\n"
    )
    exe = directory / 'selected'
    built = _run([*argv, str(source), '-o', str(exe)], directory)
    if built.returncode != 0 or not exe.is_file():
        return None
    ran = _run([str(exe)], directory)
    if ran.returncode != 0:
        return None
    ints, reals = {}, {}
    for line in ran.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[0] not in ('I', 'R'):
            continue
        try:
            p, k = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        (ints if parts[0] == 'I' else reals)[p] = k
    if len(ints) < 41 or len(reals) < 41:
        return None
    return {'int': ints, 'real': reals}


def _probe_traits(argv, directory, kinds):
    int_kinds = sorted({k for k in kinds['int'].values() if k > 0})
    real_kinds = sorted({k for k in kinds['real'].values() if k > 0})
    lines = [
        "program f2py_kind_traits",
        "  implicit none",
        "  integer :: k",
    ]
    for kind in int_kinds:
        lines.append(f"  k = {kind}")
        lines.append(
            "  write(*,'(A,1X,A,1X,I4,1X,I5,1X,I6)') 'S', 'I', k, "
            f"storage_size(0_{kind}), range(0_{kind})"
        )
    for kind in real_kinds:
        lines.append(f"  k = {kind}")
        lines.append(
            "  write(*,'(A,1X,A,1X,I4,1X,I5,1X,I5,1X,I6)') 'S', 'R', k, "
            f"storage_size(0.0_{kind}), precision(0.0_{kind}), range(0.0_{kind})"
        )
    lines.append("end program\n")
    source = directory / 'traits.f90'
    source.write_text("\n".join(lines))
    exe = directory / 'traits'
    built = _run([*argv, str(source), '-o', str(exe)], directory)
    if built.returncode != 0 or not exe.is_file():
        return None
    ran = _run([str(exe)], directory)
    if ran.returncode != 0:
        return None
    traits = {'integer': {}, 'real': {}}
    for line in ran.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != 'S' or parts[1] not in ('I', 'R'):
            continue
        try:
            kind = int(parts[2])
            storage = int(parts[3])
        except ValueError:
            continue
        typespec = 'integer' if parts[1] == 'I' else 'real'
        info = {'storage': storage}
        if typespec == 'integer':
            info['range'] = int(parts[4])
        else:
            if len(parts) < 6:
                continue
            info['precision'] = int(parts[4])
            info['range'] = int(parts[5])
        traits[typespec][kind] = info
    return traits


def selected_int_kind(r):
    try:
        index = operator_index(r)
    except TypeError:
        return selected_int_kind_fallback(r)
    table = active_kind_table()
    if table is not None and index in table['int']:
        return table['int'][index]
    return selected_int_kind_fallback(index)


def selected_real_kind(p, r=0, radix=0):
    if r or radix:
        return selected_real_kind_fallback(p, r, radix)
    try:
        index = operator_index(p)
    except TypeError:
        return selected_real_kind_fallback(p, r, radix)
    table = active_kind_table()
    if table is not None and index in table['real']:
        return table['real'][index]
    return selected_real_kind_fallback(index, r, radix)


def operator_index(value):
    return value.__index__()


def kind_traits(typespec, kind):
    table = active_kind_table()
    if not table:
        return None
    try:
        kind = int(str(kind).strip())
    except (TypeError, ValueError):
        return None
    return table['traits'].get(typespec, {}).get(kind)


def ctype_from_kind(typespec, kind):
    """C type for a compiler kind, chosen by storage size.

    Returns ``None`` for widths f2py does not wrap (IEEE half, 128-bit
    integers). Callers keep their previous default in that case.
    """
    traits = kind_traits(typespec, kind)
    if not traits:
        return None
    bits = traits['storage']
    if typespec == 'real':
        return _REAL_CTYPE.get(bits)
    if typespec == 'integer':
        return _INT_CTYPE.get(bits)
    return None
