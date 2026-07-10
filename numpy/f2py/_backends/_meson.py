import errno
import os
import re
import shutil
import subprocess
import sys
from itertools import chain
from pathlib import Path
from string import Template

from ._backend import Backend


# Match key=value or key: value pairs inside --dep name[...] brackets.
# Values may be bare words, true/false, quoted strings, or simple lists.
_DEP_KWARG_RE = re.compile(
    r"""
    (\w+)                  # key
    \s*[:=]\s*             # = or :
    (
        true|false|        # booleans
        '(?:[^'\\]|\\.)*'| # single-quoted string
        "(?:[^"\\]|\\.)*"| # double-quoted string
        \[[^\]]*\]|        # simple list literal
        [^\s,\]]+          # bare token
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _meson_format_kwarg_value(raw: str) -> str:
    """Format a CLI dependency kwarg value as Meson syntax."""
    val = raw.strip()
    low = val.lower()
    if low in ("true", "false"):
        return low
    if (val.startswith("'") and val.endswith("'")) or (
        val.startswith('"') and val.endswith('"')
    ):
        # normalize to single quotes
        return f"'{val[1:-1]}'"
    if val.startswith("[") and val.endswith("]"):
        # list of strings/bools/numbers: re-quote string elements lightly
        inner = val[1:-1].strip()
        if not inner:
            return "[]"
        parts = []
        for item in re.split(r"\s*,\s*", inner):
            if not item:
                continue
            parts.append(_meson_format_kwarg_value(item))
        return "[" + ", ".join(parts) + "]"
    # bare word / number → string if not a plain integer/float
    if re.fullmatch(r"-?\d+(\.\d+)?", val):
        return val
    return f"'{val}'"


def format_meson_dependency(dep: str) -> str:
    """Turn a ``--dep`` token into a Meson ``dependency(...)`` call.

    Bare names::

        lapack  ->  dependency('lapack')

    Optional kwargs (do not hardcode language for all deps; gh-28902)::

        mpi[language=fortran]           -> dependency('mpi', language: 'fortran')
        mpi[language: 'fortran']        -> dependency('mpi', language: 'fortran')
        foo[static=true, method=pkg-config]
            -> dependency('foo', static: true, method: 'pkg-config')
    """
    dep = dep.strip()
    if not dep:
        raise ValueError("empty --dep value")
    if "[" not in dep:
        return f"dependency('{dep}')"
    if not dep.endswith("]"):
        raise ValueError(
            f"invalid --dep syntax {dep!r}: expected name[key=value, ...]"
        )
    name, rest = dep.split("[", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"invalid --dep syntax {dep!r}: missing dependency name")
    kwargs_body = rest[:-1].strip()
    if not kwargs_body:
        return f"dependency('{name}')"
    kwargs = []
    pos = 0
    for m in _DEP_KWARG_RE.finditer(kwargs_body):
        # allow commas/whitespace between pairs
        gap = kwargs_body[pos:m.start()].strip().strip(",")
        if gap:
            raise ValueError(
                f"invalid --dep syntax {dep!r}: unexpected {gap!r}"
            )
        key = m.group(1)
        val = _meson_format_kwarg_value(m.group(2))
        kwargs.append(f"{key}: {val}")
        pos = m.end()
    trailing = kwargs_body[pos:].strip().strip(",")
    if trailing:
        raise ValueError(
            f"invalid --dep syntax {dep!r}: unexpected {trailing!r}"
        )
    if not kwargs:
        raise ValueError(
            f"invalid --dep syntax {dep!r}: expected key=value pairs in brackets"
        )
    return f"dependency('{name}', {', '.join(kwargs)})"


def _quote_meson_arg(arg: str) -> str:
    """Wrap a compile arg in single quotes for meson if not already quoted."""
    if arg.startswith("'") and arg.endswith("'"):
        return arg
    return f"'{arg}'"


def _macros_to_flags(define_macros: list) -> list[str]:
    """Convert (name, value|None) define_macros to -D compiler flags."""
    flags = []
    for item in define_macros:
        if isinstance(item, tuple):
            name, value = item
            if value is None:
                flags.append(f"-D{name}")
            else:
                flags.append(f"-D{name}={value}")
        else:
            # bare name or already-formed -D...
            flags.append(item if str(item).startswith("-D") else f"-D{item}")
    return flags


class MesonTemplate:
    """Template meson build file generation class."""

    def __init__(
        self,
        modulename: str,
        sources: list[Path],
        deps: list[str],
        libraries: list[str],
        library_dirs: list[Path],
        include_dirs: list[Path],
        object_files: list[Path],
        linker_args: list[str],
        fortran_args: list[str],
        build_type: str,
        python_exe: str,
        c_args: list[str] | None = None,
    ):
        self.modulename = modulename
        self.build_template_path = (
            Path(__file__).parent.absolute() / "meson.build.template"
        )
        self.sources = sources
        self.deps = deps
        self.libraries = libraries
        self.library_dirs = library_dirs
        if include_dirs is not None:
            self.include_dirs = include_dirs
        else:
            self.include_dirs = []
        self.substitutions = {}
        self.objects = object_files
        # Convert args to '' wrapped variant for meson
        self.fortran_args = [_quote_meson_arg(x) for x in fortran_args]
        self.c_args = [_quote_meson_arg(x) for x in (c_args or [])]
        self.pipeline = [
            self.initialize_template,
            self.sources_substitution,
            self.objects_substitution,
            self.deps_substitution,
            self.include_substitution,
            self.libraries_substitution,
            self.c_args_substitution,
            self.fortran_args_substitution,
        ]
        self.build_type = build_type
        self.python_exe = python_exe
        self.indent = " " * 21

    def meson_build_template(self) -> str:
        if not self.build_template_path.is_file():
            raise FileNotFoundError(
                errno.ENOENT,
                "Meson build template"
                f" {self.build_template_path.absolute()}"
                " does not exist.",
            )
        return self.build_template_path.read_text()

    def initialize_template(self) -> None:
        self.substitutions["modulename"] = self.modulename
        self.substitutions["buildtype"] = self.build_type
        self.substitutions["python"] = self.python_exe

    def sources_substitution(self) -> None:
        self.substitutions["source_list"] = ",\n".join(
            [f"{self.indent}'''{source}'''," for source in self.sources]
        )

    def objects_substitution(self) -> None:
        self.substitutions["obj_list"] = ",\n".join(
            [f"{self.indent}'''{obj}'''," for obj in self.objects]
        )

    def deps_substitution(self) -> None:
        # gh-28902: allow --dep "mpi[language=fortran]" etc. A bare
        # ``openmp`` expands to both languages with a Fortran link since
        # dependency('openmp') defaults to the C compiler only (gh-27163,
        # gh-30804); explicit kwargs on openmp take precedence.
        deps_list = []
        has_openmp = False
        for dep in self.deps:
            if dep.lower() == "openmp":
                has_openmp = True
                deps_list.append(
                    f"{self.indent}dependency('openmp', language: 'c'),"
                )
                deps_list.append(
                    f"{self.indent}dependency('openmp', language: 'fortran'),"
                )
            else:
                deps_list.append(
                    f"{self.indent}{format_meson_dependency(dep)},"
                )
        # Items already carry self.indent; join with ",\n" only (same pattern as
        # sources_substitution) so multi-dep lines stay single-indented.
        self.substitutions["dep_list"] = ",\n".join(deps_list)
        if has_openmp:
            self.substitutions["link_language"] = (
                f"{self.indent}link_language: 'fortran',"
            )
        else:
            self.substitutions["link_language"] = ""

    def libraries_substitution(self) -> None:
        self.substitutions["lib_dir_declarations"] = "\n".join(
            [
                f"lib_dir_{i} = declare_dependency(link_args : ['''-L{lib_dir}'''])"
                for i, lib_dir in enumerate(self.library_dirs)
            ]
        )

        self.substitutions["lib_declarations"] = "\n".join(
            [
                f"{lib.replace('.', '_')} = declare_dependency(link_args : ['-l{lib}'])"
                for lib in self.libraries
            ]
        )

        self.substitutions["lib_list"] = f"\n{self.indent}".join(
            [f"{self.indent}{lib.replace('.', '_')}," for lib in self.libraries]
        )
        self.substitutions["lib_dir_list"] = f"\n{self.indent}".join(
            [f"{self.indent}lib_dir_{i}," for i in range(len(self.library_dirs))]
        )

    def include_substitution(self) -> None:
        self.substitutions["inc_list"] = f",\n{self.indent}".join(
            [f"{self.indent}'''{inc}'''," for inc in self.include_dirs]
        )

    def c_args_substitution(self) -> None:
        if self.c_args:
            self.substitutions["c_args"] = (
                f"{self.indent}c_args: [{', '.join(list(self.c_args))}],"
            )
        else:
            self.substitutions["c_args"] = ""

    def fortran_args_substitution(self) -> None:
        if self.fortran_args:
            self.substitutions["fortran_args"] = (
                f"{self.indent}fortran_args: [{', '.join(list(self.fortran_args))}],"
            )
        else:
            self.substitutions["fortran_args"] = ""

    def generate_meson_build(self):
        for node in self.pipeline:
            node()
        template = Template(self.meson_build_template())
        meson_build = template.substitute(self.substitutions)
        meson_build = meson_build.replace(",,", ",")
        return meson_build


class MesonBackend(Backend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dependencies = self.extra_dat.get("dependencies", [])
        self.cross_files = self.extra_dat.get("cross_files", [])
        self.meson_build_dir = "bbdir"
        self.build_type = (
            "debug" if any("debug" in flag for flag in self.fc_flags) else "release"
        )
        self.fc_flags = _get_flags(self.fc_flags)
        # -D macros: forward into both C and Fortran compile args (gh-28648)
        self.macro_flags = _macros_to_flags(self.define_macros)

    def _move_exec_to_root(self, build_dir: Path):
        walk_dir = Path(build_dir) / self.meson_build_dir
        path_objects = chain(
            walk_dir.glob(f"{self.modulename}*.so"),
            walk_dir.glob(f"{self.modulename}*.pyd"),
            walk_dir.glob(f"{self.modulename}*.dll"),
        )
        # Same behavior as distutils
        # https://github.com/numpy/numpy/issues/24874#issuecomment-1835632293
        for path_object in path_objects:
            dest_path = Path.cwd() / path_object.name
            if dest_path.exists():
                dest_path.unlink()
            shutil.copy2(path_object, dest_path)
            os.remove(path_object)

    def write_meson_build(self, build_dir: Path) -> None:
        """Writes the meson build file at specified location"""
        # Merge -D flags into fortran_args; also emit c_args for C wrappers
        fortran_args = list(self.fc_flags) + list(self.macro_flags)
        meson_template = MesonTemplate(
            self.modulename,
            self.sources,
            self.dependencies,
            self.libraries,
            self.library_dirs,
            self.include_dirs,
            self.extra_objects,
            self.flib_flags,
            fortran_args,
            self.build_type,
            sys.executable,
            c_args=list(self.macro_flags),
        )
        src = meson_template.generate_meson_build()
        Path(build_dir).mkdir(parents=True, exist_ok=True)
        meson_build_file = Path(build_dir) / "meson.build"
        meson_build_file.write_text(src)
        return meson_build_file

    def _run_subprocess_command(self, command, cwd):
        subprocess.run(command, cwd=cwd, check=True)

    def run_meson(self, build_dir: Path):
        setup_command = ["meson", "setup", self.meson_build_dir]
        # gh-28352: first-class --cross-file only (no MESON_ARGS env passthrough)
        for cross_file in self.cross_files:
            setup_command.extend(["--cross-file", cross_file])
        self._run_subprocess_command(setup_command, build_dir)
        compile_command = ["meson", "compile", "-C", self.meson_build_dir]
        self._run_subprocess_command(compile_command, build_dir)

    def compile(self) -> None:
        self.sources = _prepare_sources(self.modulename, self.sources, self.build_dir)
        _prepare_objects(self.modulename, self.extra_objects, self.build_dir)
        self.write_meson_build(self.build_dir)
        self.run_meson(self.build_dir)
        self._move_exec_to_root(self.build_dir)


def _copy_to_build_dir(source, bdir):
    """Copy ``source`` into ``bdir``; a no-op when ``source`` already
    lives there (the build dir may be the source dir, gh-29762).

    Returns True when a copy was made.
    """
    src = Path(source).resolve()
    dest = (Path(bdir) / Path(source).name).resolve()
    if src == dest:
        return False
    shutil.copy(source, dest)
    return True

def _prepare_sources(mname, sources, bdir):
    extended_sources = sources.copy()
    Path(bdir).mkdir(parents=True, exist_ok=True)
    # Copy sources
    for source in sources:
        if Path(source).exists() and Path(source).is_file():
            _copy_to_build_dir(source, bdir)
    generated_sources = [
        Path(f"{mname}module.c"),
        Path(f"{mname}-f2pywrappers2.f90"),
        Path(f"{mname}-f2pywrappers.f"),
    ]
    bdir = Path(bdir)
    for generated_source in generated_sources:
        if generated_source.exists():
            # only remove the original when it was copied elsewhere
            if _copy_to_build_dir(generated_source, bdir):
                generated_source.unlink()
            extended_sources.append(generated_source.name)
    extended_sources = [
        Path(source).name
        for source in extended_sources
        if not Path(source).suffix == ".pyf"
    ]
    return extended_sources

def _prepare_objects(mname, objects, bdir):
    Path(bdir).mkdir(parents=True, exist_ok=True)
    # Copy objects
    for obj in objects:
        if Path(obj).exists() and Path(obj).is_file():
            _copy_to_build_dir(obj, bdir)

def _get_flags(fc_flags):
    flag_values = []
    flag_pattern = re.compile(r"--f(77|90)flags=(.*)")
    for flag in fc_flags:
        match_result = flag_pattern.match(flag)
        if match_result:
            values = match_result.group(2).strip().split()
            values = [val.strip("'\"") for val in values]
            flag_values.extend(values)
    # Hacky way to preserve order of flags
    unique_flags = list(dict.fromkeys(flag_values))
    return unique_flags
