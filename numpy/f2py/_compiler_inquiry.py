"""Run compiler-context REAL kind inquiries without changing parser values."""

import functools
import operator
import os
import re
import subprocess
import tempfile
import weakref
from pathlib import Path

from . import _compiler_context

_PROBE_COMPILE_TIMEOUT = 120
_PROBE_RUN_TIMEOUT = 30


def _compiler_key():
    """``FC`` when it is set, else the empty string that means gfortran."""
    return os.environ.get('FC') or ''


def _compiler_context_key():
    return _compiler_context.current(_compiler_key())


class _ProbeDirectory:
    """Keep a cached probe alive and clean its directory explicitly."""

    def __init__(self):
        directory = tempfile.TemporaryDirectory(prefix='f2py-kind-')
        self.name = directory.name
        self._cleanup = weakref.finalize(self, directory.cleanup)

    def cleanup(self):
        self._cleanup()


@functools.lru_cache(maxsize=8)
def _selected_real_kind_program(context, kinds):
    """Build a query with the argument kinds and target execution context."""
    context.require_execution()
    if not context.command:
        raise _compiler_context.CompilerExecutionError(
            'FC must identify a Fortran compiler')
    owner = _ProbeDirectory()
    directory = Path(owner.name)
    source = directory / 'probe.f90'
    object_file = directory / 'probe.o'
    executable = directory / 'probe'
    declarations = ''
    for name, kind in zip(('p', 'r', 'radix'), kinds):
        selector = '' if kind is None else f'(kind={kind})'
        declarations += f'      integer{selector} :: {name}\n'
    program = """      program f2py_kind_probe
      implicit none
      integer :: mask, result, status
@DECLARATIONS@      read(*, *, iostat=status) mask, p, r, radix
      if (status /= 0) stop 2
      select case (mask)
      case (1)
        result = selected_real_kind(p=p)
      case (2)
        result = selected_real_kind(r=r)
      case (3)
        result = selected_real_kind(p=p, r=r)
      case (4)
        result = selected_real_kind(radix=radix)
      case (5)
        result = selected_real_kind(p=p, radix=radix)
      case (6)
        result = selected_real_kind(r=r, radix=radix)
      case (7)
        result = selected_real_kind(p=p, r=r, radix=radix)
      case default
        stop 2
      end select
      write(*, '(i0)') result
      end program f2py_kind_probe
"""
    try:
        source.write_text(program.replace('@DECLARATIONS@', declarations),
                          encoding='ascii')
        commands = (
            ('compile', [*context.command, *context.compile_args,
                         '-c', str(source), '-o', str(object_file)]),
            ('link', [*context.command, str(object_file),
                      *context.link_args, '-o', str(executable)]),
        )
        for phase, command in commands:
            completed = subprocess.run(
                command, cwd=context.working_directory or None,
                env=dict(os.environ, PATH=context.search_path),
                capture_output=True, timeout=_PROBE_COMPILE_TIMEOUT, check=False)
            if completed.returncode:
                raise _compiler_context.CompilerExecutionError(
                    f'selected_real_kind probe {phase} failed: '
                    + completed.stderr.decode(errors='replace')[-1024:])
    except (OSError, subprocess.SubprocessError) as error:
        owner.cleanup()
        raise _compiler_context.CompilerExecutionError(
            'Cannot compile or link the configured Fortran kind probe') from error
    except BaseException:
        owner.cleanup()
        raise
    return owner, executable


def _kind_argument_kind(value):
    if value is None:
        return None
    kind = getattr(value, 'kind', None)
    if kind is not None:
        kind = operator.index(kind)
        if kind <= 0:
            raise ValueError('An INTEGER argument kind must be positive')
        return kind
    number = operator.index(value)
    if -(1 << 31) <= number < (1 << 31):
        return None
    for kind in (8, 16):
        if -(1 << (8 * kind - 1)) <= number < (1 << (8 * kind - 1)):
            return kind
    raise ValueError('Intrinsic argument exceeds supported INTEGER widths')


def _execute_selected_real_kind(context, p, r, radix):
    context.require_execution()
    values = (p, r, radix)
    kinds = tuple(_kind_argument_kind(value) for value in values)
    mask = sum(1 << index for index, value in enumerate(values)
               if value is not None)
    arguments = [0 if value is None else operator.index(value)
                 for value in values]
    owner, executable = _selected_real_kind_program(context, kinds)
    query = ' '.join(str(value) for value in (mask, *arguments)) + '\n'
    command = [*context.runner, str(executable)]
    try:
        completed = subprocess.run(
            command, input=query.encode('ascii'), cwd=owner.name,
            env=dict(os.environ, PATH=context.search_path),
            capture_output=True, timeout=_PROBE_RUN_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise _compiler_context.CompilerExecutionError(
            'Cannot execute the configured Fortran kind probe') from error
    try:
        output = completed.stdout.decode('ascii', errors='strict').strip()
    except UnicodeDecodeError as error:
        raise _compiler_context.CompilerExecutionError(
            'Fortran kind probe output must be an ASCII integer') from error
    if completed.returncode or re.fullmatch(r'-?[0-9]+', output) is None:
        raise _compiler_context.CompilerExecutionError(
            'Configured Fortran kind probe execution failed: '
            + completed.stderr.decode(errors='replace')[-1024:])
    return int(output)


def _selected_real_kind_from_compiler(compiler, flags, p, r, radix):
    context = _compiler_context.coerce(compiler)
    context = _compiler_context.replace(
        context, compile_args=tuple(flags), selected=True)
    return _execute_selected_real_kind(context, p, r, radix)


def selected_real_kind(p, r, radix):
    """Return the configured compiler's kind, or None without a selection."""
    compiler = _compiler_context_key()
    if not compiler:
        return None
    if not compiler.command:
        raise _compiler_context.CompilerExecutionError(
            'FC must identify a Fortran compiler')
    return _selected_real_kind_from_compiler(
        compiler.command, compiler.compile_args, p, r, radix)


@functools.lru_cache(maxsize=128)
def integer_specification(context, program):
    """Evaluate a scalar INTEGER specification with the selected processor."""
    context.require_execution()
    if not context.command:
        raise _compiler_context.CompilerExecutionError(
            'A configured Fortran compiler is required for this specification')
    owner = _ProbeDirectory()
    directory = Path(owner.name)
    source = directory / 'specification.f90'
    object_file = directory / 'specification.o'
    executable = directory / 'specification'
    environment = dict(os.environ, PATH=context.search_path)
    try:
        source.write_text(program, encoding='utf-8')
        commands = (
            ('compile', [*context.command, *context.compile_args,
                         '-c', str(source), '-o', str(object_file)]),
            ('link', [*context.command, str(object_file),
                      *context.link_args, '-o', str(executable)]),
        )
        for phase, command in commands:
            completed = subprocess.run(
                command, cwd=context.working_directory or None, env=environment,
                capture_output=True, timeout=_PROBE_COMPILE_TIMEOUT, check=False)
            if completed.returncode:
                raise _compiler_context.CompilerExecutionError(
                    f'Fortran specification {phase} failed: '
                    + completed.stderr.decode(errors='replace')[-4096:])
        completed = subprocess.run(
            [*context.runner, str(executable)], cwd=owner.name, env=environment,
            capture_output=True, timeout=_PROBE_RUN_TIMEOUT, check=False)
        output = completed.stdout.decode('ascii', errors='strict').strip()
        if completed.returncode or re.fullmatch(r'-?[0-9]+', output) is None:
            raise _compiler_context.CompilerExecutionError(
                'Fortran specification must produce one scalar INTEGER: '
                + completed.stderr.decode(errors='replace')[-4096:])
        return int(output)
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise _compiler_context.CompilerExecutionError(
            'Cannot compile or execute the configured Fortran specification') from error
    finally:
        owner.cleanup()
