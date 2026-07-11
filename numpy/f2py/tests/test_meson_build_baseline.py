"""Meson build-time baseline for the f2py backend.

gh-25415 asked to compare the meson backend's build time against the
old distutils backend; distutils is gone, so that comparison is no
longer answerable. What the issue was actually chasing is a number to
judge future meson-side regressions against, so this establishes one:
build a representative multi-file f2py module and record how long the
meson backend takes, failing only on gross regression (a stalled or
badly slower build), not on ordinary variance.
"""
import textwrap
import time

import pytest

from . import util


@pytest.mark.slow
class TestMesonBuildBaseline(util.F2PyTest):
    # Three plain Fortran files (no .pyf -- a .pyf hardcodes its own
    # module name and would silently override module_name below) with
    # a callback dummy, a free function, and a CONTAINS block: enough
    # surface for meson to do real dependency scanning, unlike a
    # single-routine smoke build.
    sources = [
        util.getpath("tests", "src", "regression", "gh8288.f"),
        util.getpath("tests", "src", "regression", "gh19767.f90"),
        util.getpath("tests", "src", "regression", "gh20103.f90"),
    ]
    module_name = "meson_build_baseline"

    def test_build_completes_and_is_recorded(self):
        # The module built during class setup already paid the build
        # cost; here we do a second, timed, from-scratch build of an
        # equivalent single-file module through the same build_module
        # path (fresh subprocess, fresh meson build dir) to get a clean
        # number uncontaminated by the class-level build cache.
        source = textwrap.dedent("""
            function mysqrt2(x) result(r)
              implicit none
              real(kind=8), intent(in) :: x
              real(kind=8) :: r
              r = sqrt(x)
            end function mysqrt2
        """)

        t0 = time.monotonic()
        mod = util.build_code(source, suffix=".f90",
                              module_name="meson_build_baseline_probe")
        elapsed = time.monotonic() - t0

        # Generous ceiling: this is a regression guard against a stalled
        # or badly broken build, not a performance benchmark assertion.
        # Record the baseline number for humans comparing meson versions.
        print(f"\nmeson_build_baseline: single-file f2py build took "
              f"{elapsed:.2f}s")
        assert elapsed < 300, (
            f"meson build took {elapsed:.2f}s, over the 300s regression "
            "ceiling -- investigate before treating this as normal"
        )

        assert mod.mysqrt2(4.0) == 2.0
        assert self.module.first(lambda x: x + 1.0, 1.0) == 2.0
        assert self.module.mysqrt(4.0) == 2.0
        assert self.module.outer20103(3.0) == 7.0
