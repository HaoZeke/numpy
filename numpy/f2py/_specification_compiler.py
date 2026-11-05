"""Resolve an automatically selected Meson compiler only when it is needed."""

import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from . import _compiler_context


class AutoCompilerContext(_compiler_context._DeferredCompilerContext):
    def __init__(self, environment, fortran_args):
        self._environment = environment
        self._fortran_args = tuple(fortran_args)
        self._variables = dict(os.environ, PATH=environment.search_path)
        self._resolved = None
        self._failure = None
        self._lock = _compiler_context.threading.RLock()

    def resolve(self):
        with self._lock:
            if self._failure is not None:
                raise self._failure
            if self._resolved is not None:
                return self._resolved
            try:
                with tempfile.TemporaryDirectory(prefix="f2py-compiler-") as directory:
                    source = Path(directory) / "source"
                    build = Path(directory) / "build"
                    source.mkdir()
                    (source / "meson.build").write_text(
                        "project('f2py-specification-compiler', 'fortran')\n",
                        encoding="ascii",
                    )
                    commands = (
                        ["meson", "setup", str(build), str(source), "--backend=ninja"],
                        ["meson", "introspect", "--compilers", str(build)],
                    )
                    for command in commands:
                        completed = subprocess.run(
                            command,
                            cwd=self._environment.working_directory or None,
                            env=self._variables,
                            capture_output=True,
                            check=False,
                            timeout=120,
                        )
                        if completed.returncode:
                            raise _compiler_context.CompilerExecutionError(
                                "Cannot resolve the Meson Fortran compiler: "
                                + completed.stderr.decode(errors="replace")[-4096:]
                            )
                    compilers = json.loads(completed.stdout)
                    command = compilers["host"]["fortran"]["exelist"]
                    if (
                        not isinstance(command, list)
                        or not command
                        or any(not isinstance(arg, str) or not arg for arg in command)
                    ):
                        raise ValueError(
                            "Meson Fortran exelist must be a nonempty string list"
                        )
                    self._resolved = replace(
                        self._environment,
                        command=_compiler_context._relative_program(
                            tuple(command), self._environment.working_directory
                        ),
                        compile_args=(
                            *self._environment.compile_args,
                            *self._fortran_args,
                        ),
                        selected=True,
                    )
            except _compiler_context.CompilerExecutionError as error:
                self._failure = error
                raise
            except (
                OSError,
                subprocess.SubprocessError,
                ValueError,
                KeyError,
                TypeError,
            ) as error:
                self._failure = _compiler_context.CompilerExecutionError(
                    "Cannot read the Meson Fortran compiler selection: "
                    + str(error)[:1024]
                )
                raise self._failure from error
            return self._resolved
