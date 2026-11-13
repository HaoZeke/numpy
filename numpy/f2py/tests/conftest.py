"""Skip an f2py test under LFortran only when that compiler still has the gap.

The probes compile a tiny reproducer once per session. A test is skipped
only when its probe fails, and the skip reason is the gap name. A compiler
that can build and run the reproducer runs the test.

f2py passes LFortran flags the driver does not turn on by itself. A probe
that omits those flags can pass while the wrapped test still fails, so the
probes that stand in for an f2py build use the same flags.
"""
import os
import shlex
import subprocess
import tempfile

import pytest


def _lfortran():
    return "lfortran" in os.environ.get("FC", "")


def _compiler():
    fc = os.environ.get("FC", "lfortran")
    parts = shlex.split(fc)
    return parts[0] if parts else "lfortran"


# Flags numpy/f2py/_backends/meson.build.template adds for LFortran.
# Meson also injects the four that are not implicit typing or the
# external underscore; passing the whole set matches the compile line.
_DRIVER = (
    "--implicit-typing",
    "--implicit-interface",
    "--mangle-underscore-external",
    "--fixed-form-infer",
    "--separate-compilation",
    "--print-leading-space",
)


# node id suffix -> gap. A file path covers every test in that file.
_SKIPS = (
    ("test_character.py", "character-array"),
    ("test_modules.py::TestAllocatableCharacterArray", "allocatable-character"),
    ("test_modules.py::TestAllocatableOnlyModule", "allocatable-only-module"),
    ("test_modules.py::TestStrictSetattr", "bindc-module-variable"),
    ("test_optional_absent.py::TestOptionalAbsent", "absent-optional"),
    ("test_optional_absent.py::TestPyfAbsent", "absent-optional"),
    ("test_optional_absent.py::TestKeepPreviousCall", "absent-optional"),
    ("test_optional_absent.py::TestOptionalCallbackBuilds", "absent-optional"),
    ("test_language_support.py::TestF90Allocatable", "allocatable-only-module"),
    ("test_language_support.py::TestF2003Features", "bindc-module-variable"),
    ("test_allocatable_args.py::TestAllocatableArguments", "allocatable-statement"),
    ("test_check_format.py::TestCheckFormat", "c-long-double-complex"),
    ("test_docs.py::TestDocAdvanced", "bindc-underscore"),
    ("test_docs.py::test_gh26973_meson_example", "implicit-do"),
    ("test_docs.py::test_gh26973_meson_python_example", "free-subroutine-underscore"),
    ("test_entry.py::TestEntryInResultFunction", "entry"),
    ("test_isoc.py::TestISOC", "c-long-double-complex"),
    ("test_regression.py::test_gh30804_opt_openmp_flag", "fopenmp"),
    ("test_regression.py::test_gh25784", "fixed-form"),
    ("test_regression.py::test_gh31249", "free-subroutine-underscore"),
    ("test_regression.py::test_gh13356_entry_return", "entry"),
    ("test_regression.py::TestDataOnlyMultiModule", "module-parameter"),
    ("test_regression.py::TestAssignmentOnlyModules", "module-parameter"),
    ("test_return_character.py::TestFReturnCharacterNoWrap", "character-return"),
    ("test_signals.py::TestCheckSignals", "bindc-underscore"),
    ("test_size.py::TestSizeSumExample", "dimension-before-declaration"),
    ("test_string.py::TestLargeString::test_string_array_len", "character-array"),
    ("test_crackfortran.py::TestDimSpec::test_inv_array_size", "empty-array-bounds"),
    ("test_callback.py::TestF90Callback::test_gh17797", "duplicate-callback"),
)

_XFAILS = (
    (
        "test_kind.py::TestKind::test_int",
        "LFortran has no integer kind for 19 decimal digits, so "
        "selected_int_kind(19) is -1. f2py's static kind table uses "
        "gfortran's 16.",
    ),
    (
        "test_kind.py::TestKind::test_real",
        "LFortran reports selected_real_kind(16) as 16, IEEE quad. "
        "gfortran's is 10, 80-bit extended precision. f2py's static "
        "kind table follows gfortran.",
    ),
)

# gap -> (sources, extra compiler args, mode)
# "run" compiles the Fortran in one invocation and runs it.
# "pieces" compiles each Fortran file on its own, then links. That is
# what f2py does, and a module procedure passed across that boundary
# is a different bug from the same source in one file.
# "compile" only compiles. "symbol:NAME" requires that nm symbol.
_PROBES = {
    "character-array": (
        {"p.f90": (
            "subroutine take(c, o, m, n)\n"
            "character(len=1), intent(out) :: c(m)\n"
            "integer, intent(in) :: m, n\n"
            "integer(1), intent(in) :: o(m, n)\n"
            "integer :: i\n"
            "do i = 1, m\n"
            "c(i) = transfer(o(i, :), c(i))\n"
            "end do\n"
            "end subroutine\n"
            "program p\n"
            "implicit none\n"
            "character(len=1) :: c(2)\n"
            "integer(1) :: o(2, 1)\n"
            "o(1, 1) = iachar('a')\n"
            "o(2, 1) = iachar('b')\n"
            "call take(c, o, 2, 1)\n"
            "if (c(1) /= 'a' .or. c(2) /= 'b') error stop 1\n"
            "end program\n"
        )},
        _DRIVER,
        "run",
    ),
    "allocatable-character": (
        {
            "m.f90": (
                "module alloc_char_mod\n"
                "implicit none\n"
                "real*8, dimension(:,:), allocatable :: charge\n"
                "character(len=10), allocatable, dimension(:) :: names\n"
                "contains\n"
                "subroutine setup_data(n)\n"
                "integer, intent(in) :: n\n"
                "integer :: ierr\n"
                "if (allocated(charge)) deallocate(charge)\n"
                "if (allocated(names)) deallocate(names)\n"
                "allocate(charge(n, n), stat=ierr)\n"
                "allocate(names(n), stat=ierr)\n"
                "charge = 0.0d0\n"
                "names = 'test'\n"
                "end subroutine\n"
                "end module\n"
            ),
            "w.f90": (
                "subroutine f2pyinitalloc_char_mod(cb)\n"
                "use alloc_char_mod, only: setup_data\n"
                "external cb\n"
                "call cb(setup_data)\n"
                "end subroutine\n"
            ),
            "main.c": (
                "void f2pyinitalloc_char_mod_(void (*cb)(void (*)(int *)));\n"
                "static void go(void (*setup)(int *)) {\n"
                "int n = 3;\n"
                "setup(&n);\n"
                "}\n"
                "int main(void) {\n"
                "f2pyinitalloc_char_mod_(go);\n"
                "return 0;\n"
                "}\n"
            ),
        },
        _DRIVER,
        "pieces",
    ),
    "allocatable-statement": (
        {"p.f90": (
            "program p\n"
            "implicit none\n"
            "call s\n"
            "contains\n"
            "subroutine s\n"
            "integer :: a(:)\n"
            "allocatable a\n"
            "allocate(a(2))\n"
            "a = 1\n"
            "if (a(1) /= 1) error stop 1\n"
            "end subroutine\n"
            "end program\n"
        )},
        _DRIVER,
        "run",
    ),
    "allocatable-only-module": (
        {"p.f90": (
            "module m\n"
            "real, allocatable :: a(:)\n"
            "end module\n"
            "program p\n"
            "use m\n"
            "allocate(a(1))\n"
            "a(1) = 1.0\n"
            "if (a(1) /= 1.0) error stop 1\n"
            "end program\n"
        )},
        _DRIVER,
        "run",
    ),
    "absent-optional": (
        {"p.f90": (
            "module m\n"
            "contains\n"
            "subroutine s(flag, text)\n"
            "integer, intent(out) :: flag\n"
            "character(len=*), optional, intent(in) :: text\n"
            "if (present(text)) then\n"
            "flag = len(text)\n"
            "else\n"
            "flag = 0\n"
            "end if\n"
            "end subroutine\n"
            "end module\n"
            "program p\n"
            "use m\n"
            "implicit none\n"
            "integer :: flag\n"
            "call s(flag)\n"
            "if (flag /= 0) error stop 1\n"
            "call s(flag, 'abcd')\n"
            "if (flag /= 4) error stop 2\n"
            "end program\n"
        )},
        _DRIVER,
        "run",
    ),
    "entry": (
        {"p.f90": (
            "function f(x) result(r)\n"
            "double precision, intent(in) :: x\n"
            "double precision :: r, h\n"
            "integer :: g\n"
            "r = 2 * x\n"
            "return\n"
            "entry g(x)\n"
            "g = int(3 * x)\n"
            "return\n"
            "entry h(x)\n"
            "h = 4 * x\n"
            "end function\n"
            "program p\n"
            "implicit none\n"
            "interface\n"
            "double precision function h(x)\n"
            "double precision, intent(in) :: x\n"
            "end function\n"
            "end interface\n"
            "if (abs(h(1.5d0) - 6.0d0) > 1d-12) error stop 1\n"
            "end program\n"
        )},
        _DRIVER,
        "run",
    ),
    "dimension-before-declaration": (
        {"p.f90": (
            "subroutine foo(a, n, m, b)\n"
            "implicit none\n"
            "real, intent(in) :: a(n, m)\n"
            "integer, intent(in) :: n, m\n"
            "real, intent(out) :: b(size(a, 1))\n"
            "b(1) = a(1, 1)\n"
            "end subroutine\n"
        )},
        _DRIVER,
        "compile",
    ),
    "c-long-double-complex": (
        {"p.f90": (
            "module coddity\n"
            "use iso_c_binding, only: c_long_double_complex\n"
            "implicit none\n"
            "contains\n"
            "subroutine add_clong_double_arr(a, b, c)\n"
            "complex(c_long_double_complex), intent(in) :: a, b\n"
            "complex(c_long_double_complex), intent(out) :: c\n"
            "c = a + b\n"
            "end subroutine\n"
            "end module\n"
            "subroutine f2pyinitcoddity()\n"
            "end subroutine\n"
        )},
        _DRIVER,
        "symbol:f2pyinitcoddity_",
    ),
    "bindc-underscore": (
        {"p.f90": "subroutine foo() bind(c, name='foo')\nend subroutine\n"},
        _DRIVER,
        "symbol:foo",
    ),
    "bindc-module-variable": (
        {
            "m.f90": (
                "module m\n"
                "use iso_c_binding, only: c_int\n"
                "implicit none\n"
                "integer(c_int), bind(c) :: shared_count = 5\n"
                "end module\n"
            ),
            "p.f90": (
                "program p\n"
                "use m, only: shared_count\n"
                "implicit none\n"
                "if (shared_count /= 5) error stop 1\n"
                "shared_count = 6\n"
                "if (shared_count /= 6) error stop 2\n"
                "end program\n"
            ),
        },
        _DRIVER,
        "run",
    ),
    "implicit-unallocated": (
        {
            "p.f90": (
                "program p\n"
                "implicit none\n"
                "real, allocatable :: a(:)\n"
                "external show\n"
                "call show(a, allocated(a))\n"
                "end program\n"
            ),
            "show.c": "void show_(void) {}\n",
        },
        _DRIVER,
        "run",
    ),
    "fixed-form": (
        {"p.f": "      program p\n      end\n"},
        ("-ffixed-form",),
        "compile",
    ),
    "fopenmp": (
        {"p.f90": "program p\nend program\n"},
        ("-fopenmp",),
        "compile",
    ),
    "free-subroutine-underscore": (
        {"p.f90": "subroutine free_sub()\nend subroutine\n"},
        (),
        "symbol:free_sub_",
    ),
    "implicit-do": (
        {"p.f90": "program p\ndo 10 i = 1, 2\n10 continue\nend program\n"},
        (),
        "compile",
    ),
    "character-return": (
        {"p.f90": (
            "function f() result(c)\n"
            "character(len=3) :: c\n"
            "c = 'abc'\n"
            "end function\n"
            "program p\n"
            "implicit none\n"
            "character(len=3) :: c\n"
            "c = f()\n"
            "if (c /= 'abc') error stop 1\n"
            "end program\n"
        )},
        _DRIVER,
        "run",
    ),
    "module-parameter": (
        {
            "p.f90": (
                "module datonly\n"
                "implicit none\n"
                "integer, parameter :: max_value = 100\n"
                "end module\n"
                "subroutine init(cb)\n"
                "use datonly, only: max_value\n"
                "external cb\n"
                "call cb(max_value)\n"
                "end subroutine\n"
                "subroutine touch()\n"
                "integer, allocatable :: a(:)\n"
                "allocate(a(4000))\n"
                "a = 7\n"
                "end subroutine\n"
            ),
            "main.c": (
                "#include <unistd.h>\n"
                "void init_(void (*cb)(int *));\n"
                "void touch_(void);\n"
                "static int *held;\n"
                "static void cb(int *p) { held = p; }\n"
                "int main(void) {\n"
                "init_(cb);\n"
                "touch_();\n"
                "if (held == 0 || *held != 100) _exit(1);\n"
                "return 0;\n"
                "}\n"
            ),
        },
        _DRIVER,
        "run",
    ),
    "empty-array-bounds": (
        {"p.f90": (
            "subroutine s(a, n)\n"
            "integer :: n\n"
            "integer, dimension(2:n), intent(in) :: a\n"
            "if (a(2) > 0) then\n"
            "end if\n"
            "end subroutine\n"
            "program p\n"
            "integer :: a(0)\n"
            "call s(a, 1)\n"
            "end program\n"
        )},
        _DRIVER,
        "run",
    ),
    "duplicate-callback": (
        {"p.f90": (
            "subroutine s\n"
            "contains\n"
            "function f(e) result(r)\n"
            "integer :: e, r\n"
            "r = e\n"
            "end function\n"
            "function f(e) result(r)\n"
            "integer :: e, r\n"
            "r = e\n"
            "end function\n"
            "end subroutine\n"
        )},
        _DRIVER,
        "compile",
    ),
}

_GAP_OK = {}


def _matches(nodeid, key):
    at = nodeid.find(key.split("::", 1)[0])
    if at < 0:
        return False
    rest = nodeid[at:]
    return rest == key or rest.startswith(key + "::") or rest.startswith(key + "[")


def _ok(result):
    return result.returncode == 0


def _run(cmd, cwd, timeout=30):
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(cmd, 1, "", "")


def _probe_one(compiler, sources, extra, mode):
    with tempfile.TemporaryDirectory(prefix="f2py-lfortran-") as tmp:
        for name, text in sources.items():
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as handle:
                handle.write(text)
        if mode.startswith("symbol:"):
            symbol = mode.split(":", 1)[1]
            src = next(iter(sources))
            obj = os.path.join(tmp, "p.o")
            compiled = _run([compiler, "-c", *extra, src, "-o", obj], tmp)
            if not _ok(compiled):
                return False
            listed = _run(["nm", obj], tmp)
            if not _ok(listed):
                return False
            names = [line.split()[-1] for line in listed.stdout.splitlines() if line.split()]
            if symbol.endswith("_"):
                return symbol in names
            return symbol in names and symbol + "_" not in names
        if mode == "compile":
            src = next(iter(sources))
            return _ok(_run([compiler, "-c", *extra, src, "-o", "p.o"], tmp))
        objects = []
        for name in sources:
            if not name.endswith(".c"):
                continue
            obj = os.path.splitext(name)[0] + ".o"
            if not _ok(_run(["cc", "-c", name, "-o", obj], tmp)):
                return False
            objects.append(obj)
        fortran = [name for name in sources if not name.endswith(".c")]
        exe = os.path.join(tmp, "p.exe")
        if mode == "pieces":
            for name in fortran:
                obj = os.path.splitext(name)[0] + ".o"
                if not _ok(_run([compiler, "-c", *extra, name, "-o", obj], tmp)):
                    return False
                objects.append(obj)
            if not _ok(_run([compiler, *objects, "-o", exe], tmp)):
                return False
        elif not _ok(_run([compiler, *extra, *fortran, *objects, "-o", exe], tmp)):
            return False
        return _ok(_run([exe], tmp))


def _probe_gaps():
    compiler = _compiler()
    found = {}
    for name, (sources, extra, mode) in _PROBES.items():
        found[name] = _probe_one(compiler, sources, extra, mode)
    return found


def pytest_sessionstart(session):
    if _lfortran():
        _GAP_OK.update(_probe_gaps())


def pytest_collection_modifyitems(config, items):
    if not _lfortran():
        return
    if not _GAP_OK:
        _GAP_OK.update(_probe_gaps())
    for item in items:
        for key, reason in _XFAILS:
            if _matches(item.nodeid, key):
                item.add_marker(pytest.mark.xfail(reason=reason))
        for key, gap in _SKIPS:
            if _matches(item.nodeid, key) and not _GAP_OK.get(gap, False):
                item.add_marker(pytest.mark.skip(reason=gap))
