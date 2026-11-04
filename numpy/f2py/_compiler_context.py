"""Compiler and execution settings for native and Meson machine-file probes."""

import contextvars
import os
import shlex
import shutil
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

_active_context = contextvars.ContextVar('f2py_compiler_context', default=None)


class CompilerExecutionError(RuntimeError):
    """The configured target execution route cannot run a compiler probe."""


def _split_environment(value):
    if not value:
        return ()
    try:
        if os.name == 'nt':
            from mesonbuild.mesonlib import split_args
            return tuple(split_args(value))
        return tuple(shlex.split(value))
    except ValueError as error:
        raise CompilerExecutionError(
            'Invalid compiler or target-runner command syntax') from error


def _binary(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if (isinstance(value, (list, tuple))
            and all(isinstance(item, str) for item in value)):
        return tuple(value)
    raise ValueError(
        'Compiler and execution commands must be strings or argv arrays')


def _arguments(value):
    return _split_environment(value) if isinstance(value, str) else _binary(value)


def _relative_program(command, directory):
    if command and not os.path.isabs(command[0]) and os.path.dirname(command[0]):
        return (str((Path(directory) / command[0]).resolve()), *command[1:])
    return command


@dataclass(frozen=True)
class CompilerContext:
    command: tuple = ()
    compile_args: tuple = ()
    link_args: tuple = ()
    runner: tuple = ()
    needs_runner: bool = False
    selected: bool = False
    cross: bool = False
    working_directory: str = ''
    search_path: str = ''

    def __bool__(self):
        return self.selected

    def require_execution(self):
        if self.needs_runner and not self.runner:
            raise CompilerExecutionError(
                'Fortran kind probes need a target runner; configure Meson '
                'exe_wrapper or MESON_EXE_WRAPPER')


class _DeferredCompilerContext:
    """Resolve machine files only when generation needs compiler settings."""

    def __init__(self, cross_files, native_files, source_dir, runner, fortran_args):
        self._cross_files = cross_files
        self._native_files = native_files
        self._source_dir = source_dir
        self._runner = runner
        self._fortran_args = tuple(fortran_args)
        self._environment = from_environment()
        self._resolved = None
        self._failure = None
        self._lock = threading.RLock()

    def resolve(self):
        with self._lock:
            if self._failure is not None:
                raise self._failure
            if self._resolved is None:
                try:
                    cross = _materialize_machine_files(
                        self._cross_files, 'cross', self._source_dir,
                        working_directory=self._environment.working_directory)
                    native = _materialize_machine_files(
                        self._native_files, 'native', self._source_dir,
                        working_directory=self._environment.working_directory)
                    selected = from_machine_files(
                        cross, native, source_dir=self._source_dir,
                        runner=self._runner, environment=self._environment)
                    if self._fortran_args:
                        selected = replace(selected, compile_args=(
                            *selected.compile_args, *self._fortran_args))
                    self._resolved = selected
                except CompilerExecutionError as error:
                    self._failure = error
                    raise
                except Exception as error:
                    self._failure = CompilerExecutionError(
                        'Cannot resolve configured Meson compiler context: '
                        + str(error)[:1024])
                    raise self._failure from error
            return self._resolved

    def __getattr__(self, name):
        if name not in CompilerContext.__dataclass_fields__:
            raise AttributeError(name)
        return getattr(self.resolve(), name)

    def __bool__(self):
        return bool(self.resolve())

    def __hash__(self):
        return hash(self.resolve())

    def __eq__(self, other):
        if isinstance(other, _DeferredCompilerContext):
            other = other.resolve()
        if not isinstance(other, CompilerContext):
            return NotImplemented
        return self.resolve() == other

    def require_execution(self):
        self.resolve().require_execution()

def from_environment(compiler=None):
    compiler = os.environ.get('FC', '') if compiler is None else compiler
    directory = str(Path.cwd())
    command = ((compiler,) if compiler and Path(compiler).is_file()
               else _split_environment(compiler))
    flags = _split_environment(os.environ.get('FFLAGS', ''))
    linker = (*flags, *_split_environment(os.environ.get('LDFLAGS', '')))
    runner = _split_environment(os.environ.get('MESON_EXE_WRAPPER', ''))
    return CompilerContext(
        command=_relative_program(command, directory), compile_args=flags,
        link_args=linker, runner=_relative_program(runner, directory),
        selected=bool(compiler), working_directory=directory,
        search_path=os.environ.get('PATH', ''))


def current(compiler=None):
    context = _active_context.get()
    return from_environment(compiler) if context is None else context


def coerce(compiler):
    if isinstance(compiler, _DeferredCompilerContext):
        return compiler.resolve()
    if isinstance(compiler, CompilerContext):
        return compiler
    if isinstance(compiler, str):
        return from_environment(compiler)
    context = current()
    if isinstance(context, _DeferredCompilerContext):
        context = context.resolve()
    return replace(context, command=tuple(compiler), selected=bool(compiler))


def _machine_paths(filenames, kind, working_directory=None):
    """Locate machine files in the caller directory or Meson's data paths."""
    directory = Path(working_directory or Path.cwd())
    result = []
    for filename in filenames:
        expanded = os.path.expanduser(os.path.expandvars(os.fspath(filename)))
        candidate = Path(expanded)
        candidates = [candidate if candidate.is_absolute()
                      else directory / candidate]
        if sys.platform != 'win32':
            data_home = os.environ.get(
                'XDG_DATA_HOME', os.path.expanduser('~/.local/share'))
            data_dirs = os.environ.get(
                'XDG_DATA_DIRS', '/usr/local/share:/usr/share').split(':')
            for data_dir in [data_home, *data_dirs]:
                candidate = Path(data_dir) / 'meson' / kind / expanded
                candidates.append(candidate if candidate.is_absolute()
                                  else directory / candidate)
        for candidate in candidates:
            if candidate.is_file():
                result.append(str(candidate.absolute()))
                break
        else:
            # Meson also accepts file descriptors and named pipes directly.
            # Their contents need one shared copy for generation and setup.
            path = candidates[0]
            if path.exists() and not path.is_dir():
                result.append(str(path.absolute()))
            else:
                raise CompilerExecutionError(
                    f'Cannot find specified Meson {kind} file: {filename}')
    return result


def _materialize_machine_files(filenames, kind, source_dir,
                               working_directory=None):
    base = Path(working_directory or Path.cwd())
    paths = _machine_paths(filenames, kind, working_directory=base)
    for index, filename in enumerate(paths):
        if Path(filename).is_file():
            continue
        directory = Path(source_dir or base)
        if not directory.is_absolute():
            directory = base / directory
        directory.mkdir(parents=True, exist_ok=True)
        with open(filename, 'rb') as source:
            with tempfile.NamedTemporaryFile(
                prefix='f2py-machine-', suffix='.ini', dir=directory,
                delete=False,
            ) as destination:
                shutil.copyfileobj(source, destination)
                paths[index] = destination.name
    if isinstance(filenames, list):
        filenames[:] = paths
    return paths

def from_machine_files(cross_files=(), native_files=(), source_dir=None,
                       runner=None, environment=None):
    from mesonbuild.envconfig import MachineInfo, machine_info_can_run
    from mesonbuild.machinefile import parse_machine_files

    context = from_environment() if environment is None else environment
    paths = cross_files or native_files
    source_path = Path(source_dir or context.working_directory)
    if not source_path.is_absolute():
        source_path = Path(context.working_directory) / source_path
    source_dir = str(source_path.resolve())
    paths = _machine_paths(paths, 'cross' if cross_files else 'native',
                           working_directory=context.working_directory)
    data = parse_machine_files(paths, source_dir)
    binaries = data.get('binaries', {})
    properties = data.get('properties', {})
    options = data.get('built-in options', {})
    command = (_binary(binaries['fortran'])
               if 'fortran' in binaries else context.command)
    compile_args = _arguments(options.get(
        'fortran_args', properties.get('fortran_args', context.compile_args)))
    link_args = _arguments(options.get(
        'fortran_link_args', properties.get('fortran_link_args', context.link_args)))
    cross = bool(cross_files)
    needed = properties.get('needs_exe_wrapper')
    if needed is None:
        needed = (cross and 'host_machine' in data
                  and not machine_info_can_run(
                      MachineInfo.from_literal(data['host_machine'])))
    if type(needed) is not bool:
        raise ValueError('Meson needs_exe_wrapper must be boolean')
    if runner is not None:
        wrapper = _binary(runner)
    elif needed and 'exe_wrapper' in binaries:
        wrapper = _binary(binaries['exe_wrapper'])
    else:
        wrapper = context.runner
    return CompilerContext(
        command=_relative_program(command, context.working_directory),
        compile_args=compile_args, link_args=link_args,
        runner=_relative_program(wrapper, context.working_directory),
        needs_runner=needed, selected=bool(command) or (cross and needed),
        cross=cross, working_directory=context.working_directory,
        search_path=context.search_path)


@contextmanager
def generation_context(cross_files=(), native_files=(), source_dir=None,
                       runner=None, fortran_args=()):
    if cross_files or native_files:
        context = _DeferredCompilerContext(
            cross_files, native_files, source_dir, runner, fortran_args)
    else:
        context = from_environment()
        if runner is not None:
            context = replace(
                context, runner=_relative_program(
                    _binary(runner), context.working_directory))
        if fortran_args:
            context = replace(
                context, compile_args=(*context.compile_args, *fortran_args))
    token = _active_context.set(context)
    try:
        yield context
    finally:
        _active_context.reset(token)
