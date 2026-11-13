"""f2py wrapper overhead and kernel speed across Fortran compilers.

Run from outside the source tree so the installed numpy is imported::

    python3 benchmarks/benchmarks/bench_f2py_compilers.py

The ``time_*`` methods are airspeed-velocity benchmarks. ``setup`` builds
one extension per compiler. The ``__main__`` path is the measurement used
for the report: pyperf, in process, pinned to one core, median and
interquartile range.

Compiler paths default to this machine and can be overridden:

- ``F2PY_BENCH_GFORTRAN``
- ``F2PY_BENCH_FLANG``
- ``F2PY_BENCH_IFX``
- ``F2PY_BENCH_LFORTRAN``
- ``F2PY_BENCH_LFORTRAN_GAPS``
- ``F2PY_BENCH_MESON_DIR`` directory prepended to ``PATH`` (LFortran needs
  a Meson that knows the LFortran compiler class)

``F2PY_BENCH_SMOKE=1`` builds every compiler, calls each routine once, and
does not time them.
"""
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

try:
    from .common import Benchmark
except ImportError:  # running the file as a script
    class Benchmark:
        pass


# Rank-2 shapes. A 1-d array is both C- and Fortran-contiguous, so it
# cannot show f2py's copy-in.
SHAPES = {
    "1e3": (25, 40),
    "1e5": (250, 400),
    "1e7": (2500, 4000),
}
KERNEL_N = 10_000_000
CALLBACK_N = 100_000

OVERHEAD_F90 = textwrap.dedent("""\
    subroutine noop(a, b)
      implicit none
      real(8), intent(in) :: a
      integer, intent(in) :: b
    end subroutine noop

    subroutine noop1(a)
      implicit none
      real(8), intent(in) :: a(1)
    end subroutine noop1

    subroutine touch_in(a, m, n)
      implicit none
      integer, intent(in) :: m, n
      real(8), intent(in) :: a(m, n)
      real(8), save :: sink
      sink = a(1, 1)
    end subroutine touch_in

    subroutine touch_inout(a, m, n)
      implicit none
      integer, intent(in) :: m, n
      real(8), intent(inout) :: a(m, n)
      a(1, 1) = a(1, 1)
    end subroutine touch_inout

    subroutine daxpy(n, alpha, x, y)
      implicit none
      integer, intent(in) :: n
      real(8), intent(in) :: alpha
      real(8), intent(in) :: x(n)
      real(8), intent(inout) :: y(n)
      integer :: i
      do i = 1, n
        y(i) = alpha * x(i) + y(i)
      end do
    end subroutine daxpy

    subroutine stencil(n, x, y)
      implicit none
      integer, intent(in) :: n
      real(8), intent(in) :: x(n)
      real(8), intent(inout) :: y(n)
      integer :: i
      y(1) = x(1)
      y(n) = x(n)
      do i = 2, n - 1
        y(i) = x(i - 1) - 2.0d0 * x(i) + x(i + 1)
      end do
    end subroutine stencil
    """)

CALLBACK_F90 = textwrap.dedent("""\
    subroutine drive(fun, n, acc)
      implicit none
      integer, intent(in) :: n
      real(8), intent(out) :: acc
      real(8) :: fun
      external fun
      integer :: i
      real(8) :: y
      acc = 0.0d0
      do i = 1, n
        y = fun(i)
        acc = acc + y
      end do
    end subroutine drive
    """)

# LLVM flang 20 rejects Meson's clang sanity-check flags (-Werror=name).
# The wrapper drops only those. -O3 and -module-dir still reach flang.
FLANG_WRAP = textwrap.dedent("""\
    #!/usr/bin/env python3
    import os, sys
    real = os.environ.get("F2PY_BENCH_FLANG_REAL", {real!r})
    args = [a for a in sys.argv[1:] if not a.startswith("-Werror=")]
    os.execv(real, [real, *args])
    """)


def _compilers():
    return (
        {
            "key": "gfortran",
            "label": "gfortran 13.3.0",
            "fc": os.environ.get("F2PY_BENCH_GFORTRAN", "/usr/bin/gfortran"),
        },
        {
            "key": "flang",
            "label": "LLVM flang 20.1.2",
            "fc": os.environ.get("F2PY_BENCH_FLANG", "/usr/bin/flang-20"),
            "wrap": True,
        },
        {
            "key": "ifx",
            "label": "ifx 2025.3.3",
            "fc": os.environ.get(
                "F2PY_BENCH_IFX",
                "/opt/intel/oneapi/compiler/2025.3/bin/ifx",
            ),
            "lib": os.environ.get(
                "F2PY_BENCH_IFX_LIB",
                "/opt/intel/oneapi/compiler/2025.3/lib",
            ),
        },
        {
            "key": "lfortran",
            "label": "LFortran 0.69.0",
            "fc": os.environ.get(
                "F2PY_BENCH_LFORTRAN",
                str(Path.home() / "opt/lfortran/bin/lfortran"),
            ),
        },
        {
            "key": "lfortran-gaps",
            "label": "LFortran 3525343f+600c4214",
            "fc": os.environ.get(
                "F2PY_BENCH_LFORTRAN_GAPS",
                "/tmp/lfortran-gaps/build16/src/bin/lfortran",
            ),
        },
    )


def _meson_env(extra=None):
    env = os.environ.copy()
    meson_dir = os.environ.get("F2PY_BENCH_MESON_DIR", "/tmp/bin")
    if meson_dir and Path(meson_dir, "meson").is_file():
        env["PATH"] = meson_dir + os.pathsep + env.get("PATH", "")
    if extra:
        for key, value in extra.items():
            if not value:
                continue
            if key == "LD_LIBRARY_PATH":
                old = env.get("LD_LIBRARY_PATH", "")
                env[key] = value + (os.pathsep + old if old else "")
            else:
                env[key] = value
    return env


def _load_module(name, directory):
    matches = list(Path(directory).glob(name + "*.so"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} extension in {directory}, found {matches}")
    spec = importlib.util.spec_from_file_location(name, matches[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _opt_flag(build_dir):
    path = Path(build_dir) / "bbdir" / "compile_commands.json"
    if not path.is_file():
        return "unknown"
    commands = json.loads(path.read_text())
    for entry in commands:
        source = entry.get("file", "")
        if source.endswith(".f90"):
            flags = [tok for tok in entry["command"].split() if tok.startswith("-O")]
            return " ".join(flags) if flags else "none"
    return "unknown"


def _compiler_version(fc):
    try:
        out = subprocess.run(
            [fc, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as exc:
        return str(exc)
    text = (out.stdout or out.stderr or "").strip().splitlines()
    return text[0] if text else f"exit {out.returncode}"


def _build_one(compiler, source_text, suffix, work):
    """Compile one f2py module. Return (module, opt_flag) or raise."""
    src = Path(work) / f"{suffix}.f90"
    src.write_text(source_text)
    module_name = f"f2py_bench_{compiler['key'].replace('-', '_')}_{suffix}"
    build = Path(work) / suffix
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    fc = compiler["fc"]
    env = _meson_env()
    if compiler.get("lib"):
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = compiler["lib"] + (os.pathsep + old if old else "")
    if compiler.get("wrap"):
        wrap = Path(work) / "flang-wrap"
        wrap.write_text(FLANG_WRAP.format(real=fc))
        wrap.chmod(0o755)
        env["F2PY_BENCH_FLANG_REAL"] = fc
        fc = str(wrap)
    env["FC"] = fc
    cmd = [
        sys.executable, "-m", "numpy.f2py",
        "-c", str(src),
        "-m", module_name,
        "--backend", "meson",
        "--build-dir", str(build),
    ]
    proc = subprocess.run(
        cmd,
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        tail = (proc.stdout + "\n" + proc.stderr)[-2000:]
        raise RuntimeError(tail)
    module = _load_module(module_name, work)
    return module, _opt_flag(build)


class F2PyCompilerBench(Benchmark):
    """Wrapper overhead and kernel speed for one Fortran compiler.

    Sizes are element counts. Copy-in cases are rank-2 arrays.
    ``in_c`` is C-order (f2py copies into Fortran order). ``in_f`` and
    ``inout_f`` are Fortran-order. Kernels are length 1e7. The callback
    case is one Fortran loop of 1e5 Python calls.
    """

    params = [item["key"] for item in _compilers()]
    param_names = ["compiler"]
    timeout = 300

    def setup(self, compiler):
        spec = next(item for item in _compilers() if item["key"] == compiler)
        if not Path(spec["fc"]).is_file():
            raise NotImplementedError(f"{spec['fc']} is not installed")
        work = tempfile.mkdtemp(prefix=f"f2py-bench-{compiler}-")
        self._work = work
        try:
            self.mod, self.opt = _build_one(spec, OVERHEAD_F90, "overhead", work)
        except RuntimeError as exc:
            raise NotImplementedError(str(exc)) from exc
        try:
            self.cb, _ = _build_one(spec, CALLBACK_F90, "callback", work)
        except RuntimeError:
            self.cb = None
        self.one = np.zeros(1, dtype=np.float64)
        self.c_arrays = {
            name: np.zeros(shape, dtype=np.float64, order="C")
            for name, shape in SHAPES.items()
        }
        self.f_arrays = {
            name: np.zeros(shape, dtype=np.float64, order="F")
            for name, shape in SHAPES.items()
        }
        self.x = np.ones(KERNEL_N, dtype=np.float64)
        self.y = np.ones(KERNEL_N, dtype=np.float64)
        self.z = np.empty(KERNEL_N, dtype=np.float64)

    def teardown(self, compiler):
        work = getattr(self, "_work", None)
        if work:
            shutil.rmtree(work, ignore_errors=True)

    def time_noop_scalar(self, compiler):
        self.mod.noop(1.0, 2)

    def time_noop_array1(self, compiler):
        self.mod.noop1(self.one)

    def time_in_c_1e3(self, compiler):
        self.mod.touch_in(self.c_arrays["1e3"])

    def time_in_c_1e5(self, compiler):
        self.mod.touch_in(self.c_arrays["1e5"])

    def time_in_c_1e7(self, compiler):
        self.mod.touch_in(self.c_arrays["1e7"])

    def time_in_f_1e3(self, compiler):
        self.mod.touch_in(self.f_arrays["1e3"])

    def time_in_f_1e5(self, compiler):
        self.mod.touch_in(self.f_arrays["1e5"])

    def time_in_f_1e7(self, compiler):
        self.mod.touch_in(self.f_arrays["1e7"])

    def time_inout_f_1e3(self, compiler):
        self.mod.touch_inout(self.f_arrays["1e3"])

    def time_inout_f_1e5(self, compiler):
        self.mod.touch_inout(self.f_arrays["1e5"])

    def time_inout_f_1e7(self, compiler):
        self.mod.touch_inout(self.f_arrays["1e7"])

    def time_callback_1e5(self, compiler):
        if self.cb is None:
            raise NotImplementedError("callback extension did not build")
        self.cb.drive(_callback, CALLBACK_N)

    def time_daxpy_1e7(self, compiler):
        self.mod.daxpy(1.5, self.x, self.y)

    def time_stencil_1e7(self, compiler):
        self.mod.stencil(self.x, self.z)


def _callback(index):
    return 1.0


def _cpu_record():
    info = {}
    block = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"model name", "cpu family", "model", "stepping", "cpu MHz", "cpu cores"}:
            info[key] = value.strip()
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    info["governor"] = gov.read_text().strip() if gov.is_file() else "unavailable (no cpufreq sysfs)"
    info["affinity"] = sorted(os.sched_getaffinity(0))
    return info


def _iqr(values):
    # n=4 cut points are the three quartiles.
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def _summarize(bench):
    values = list(bench.get_values())
    median = statistics.median(values)
    iqr = _iqr(values)
    ratio = (iqr / median) if median else float("inf")
    return {
        "median_s": median,
        "iqr_s": iqr,
        "iqr_over_median": ratio,
        "n": len(values),
    }


CASES = (
    ("noop_scalar", "time_noop_scalar"),
    ("noop_array1", "time_noop_array1"),
    ("in_c_1e3", "time_in_c_1e3"),
    ("in_c_1e5", "time_in_c_1e5"),
    ("in_c_1e7", "time_in_c_1e7"),
    ("in_f_1e3", "time_in_f_1e3"),
    ("in_f_1e5", "time_in_f_1e5"),
    ("in_f_1e7", "time_in_f_1e7"),
    ("inout_f_1e3", "time_inout_f_1e3"),
    ("inout_f_1e5", "time_inout_f_1e5"),
    ("inout_f_1e7", "time_inout_f_1e7"),
    ("callback_1e5", "time_callback_1e5"),
    ("daxpy_1e7", "time_daxpy_1e7"),
    ("stencil_1e7", "time_stencil_1e7"),
)


def _pin():
    os.sched_setaffinity(0, {0})


def _smoke():
    _pin()
    print(json.dumps(_cpu_record(), indent=2))
    for spec in _compilers():
        print("====", spec["label"], _compiler_version(spec["fc"]), flush=True)
        if not Path(spec["fc"]).is_file():
            print("MISSING", spec["fc"])
            continue
        bench = F2PyCompilerBench()
        try:
            bench.setup(spec["key"])
        except NotImplementedError as exc:
            print("BUILD FAILED", str(exc)[-500:])
            continue
        print("opt", bench.opt)
        bench.time_noop_scalar(spec["key"])
        bench.time_noop_array1(spec["key"])
        bench.time_in_c_1e3(spec["key"])
        bench.time_in_f_1e3(spec["key"])
        bench.time_inout_f_1e3(spec["key"])
        bench.time_daxpy_1e7(spec["key"])
        bench.time_stencil_1e7(spec["key"])
        if bench.cb is None:
            print("callback BUILD FAILED")
        else:
            acc = bench.cb.drive(_callback, 10)
            print("callback acc", acc)
        bench.teardown(spec["key"])
        print("smoke ok", spec["key"])


def _measure():
    import pyperf

    _pin()
    # One Runner per process. In-process so a worker does not rebuild.
    sys.argv = [
        sys.argv[0],
        "--in-process",
        "--affinity", "0",
        "--processes", "1",
        "--values", "21",
        "--warmups", "3",
        "--min-time", "0.1",
    ]
    runner = pyperf.Runner(values=21, processes=1, warmups=3, min_time=0.1)
    report = {
        "numpy": np.__version__,
        "numpy_file": np.__file__,
        "cpu": _cpu_record(),
        "compilers": [],
    }
    for spec in _compilers():
        entry = {
            "key": spec["key"],
            "label": spec["label"],
            "fc": spec["fc"],
            "version": _compiler_version(spec["fc"]) if Path(spec["fc"]).is_file() else "missing",
            "cases": {},
        }
        report["compilers"].append(entry)
        if not Path(spec["fc"]).is_file():
            entry["error"] = "compiler missing"
            continue
        print(f"building {spec['label']}", flush=True)
        bench = F2PyCompilerBench()
        try:
            bench.setup(spec["key"])
        except NotImplementedError as exc:
            entry["error"] = str(exc)[-1500:]
            print(entry["error"], flush=True)
            continue
        entry["opt"] = bench.opt
        for case, method_name in CASES:
            method = getattr(bench, method_name)
            print(f"  {spec['key']} {case}", flush=True)
            try:
                method(spec["key"])
            except NotImplementedError as exc:
                entry["cases"][case] = {"error": str(exc)}
                continue
            except Exception as exc:  # a bad signature should be visible, not fatal
                entry["cases"][case] = {"error": f"{type(exc).__name__}: {exc}"}
                print("   ", entry["cases"][case]["error"], flush=True)
                continue
            timed = runner.bench_func(
                f"{spec['key']}:{case}",
                lambda method=method, key=spec["key"]: method(key),
            )
            entry["cases"][case] = _summarize(timed)
            ratio = entry["cases"][case]["iqr_over_median"]
            print(
                f"    median {entry['cases'][case]['median_s']:.6e} "
                f"iqr/med {ratio:.3%} n {entry['cases'][case]['n']}",
                flush=True,
            )
        bench.teardown(spec["key"])
    return report


def main():
    if os.environ.get("F2PY_BENCH_SMOKE"):
        _smoke()
        return
    report = _measure()
    out = os.environ.get("F2PY_BENCH_JSON")
    text = json.dumps(report, indent=2)
    if out:
        Path(out).write_text(text)
    print("F2PY_BENCH_JSON_BEGIN")
    print(text)
    print("F2PY_BENCH_JSON_END")


if __name__ == "__main__":
    main()
