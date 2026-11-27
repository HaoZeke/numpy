import pytest

from numpy.f2py import crackfortran

from . import util


def _lines(lines):
    return "\n".join(lines) + "\n"


def _var_x(path):
    crackfortran.reset_global_f2py_vars()
    blocks = crackfortran.crackfortran([str(path)])
    return blocks[0]["vars"]["x"]


def _assert_real_out(var):
    assert var["typespec"] == "real"
    assert "out" in var["intent"]
    selector = var["kindselector"]
    assert "8" in (selector.get("kind"), selector.get("*"))


@pytest.mark.parametrize("suffix", [".f90", ".f"])
def test_include_continues_declaration(tmp_path, suffix):
    # The name and its kind sit on opposite sides of an include line.
    if suffix == ".f90":
        parent = _lines([
            "subroutine s(x)",
            "  real(kind=8), intent(out) :: &",
            "  include 'kind.inc'",
            "  x = 1.0d0 + 1.0d-10",
            "end subroutine s",
        ])
        tail = "  & x\n"
    else:
        parent = _lines([
            "      subroutine s(x)",
            "      real*8, intent(out) ::",
            "      include 'kind.inc'",
            "      x = 1.0d0 + 1.0d-10",
            "      end",
        ])
        tail = "     * x\n"
    src = tmp_path / f"s{suffix}"
    src.write_text(parent)
    (tmp_path / "kind.inc").write_text(tail)
    _assert_real_out(_var_x(src))


def test_nested_free_include_continues_declaration(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "main.f90").write_text(_lines([
        "subroutine s(x)",
        "  real(kind=8), intent(out) :: &",
        "  include 'sub/a2.inc'",
        "  x = 1.0d0 + 1.0d-10",
        "end subroutine s",
    ]))
    (sub / "a2.inc").write_text("  include 'n.inc'\n")
    (tmp_path / "n.inc").write_text("  & x\n")
    _assert_real_out(_var_x(tmp_path / "main.f90"))


def test_fixed_parent_reads_free_include(tmp_path):
    src = tmp_path / "s.f"
    src.write_text(_lines([
        "      subroutine s(x)",
        "      include 'd.inc'",
        "      end",
    ]))
    (tmp_path / "d.inc").write_text("integer, intent(inout) :: x\n")
    var = _var_x(src)
    assert var["typespec"] == "integer"
    assert "inout" in var["intent"]


def test_nested_include_prefers_top_source_directory(tmp_path, monkeypatch):
    # gfortran 13.3 reads n.inc from the primary source directory.
    cwd = tmp_path / "cwd"
    sub = cwd / "sub"
    sub.mkdir(parents=True)
    (cwd / "n.inc").write_text("      double precision x\n")
    (sub / "n.inc").write_text("      integer x\n")
    (sub / "a.inc").write_text("      include 'n.inc'\n")
    src = cwd / "main.f"
    src.write_text(_lines([
        "      subroutine s(x)",
        "      include 'sub/a.inc'",
        "      end",
    ]))
    monkeypatch.chdir(cwd)
    var = _var_x(src)
    assert var["typespec"] == "double precision"


def test_include_path_before_nested_directory(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    sub = cwd / "sub"
    idir = tmp_path / "idir"
    sub.mkdir(parents=True)
    idir.mkdir()
    (sub / "n.inc").write_text("      integer x\n")
    (idir / "n.inc").write_text("      complex x\n")
    (sub / "a.inc").write_text("      include 'n.inc'\n")
    src = cwd / "main.f"
    src.write_text(_lines([
        "      subroutine s(x)",
        "      include 'sub/a.inc'",
        "      end",
    ]))
    monkeypatch.chdir(cwd)
    crackfortran.reset_global_f2py_vars()
    crackfortran.include_paths[:] = [str(idir)]
    blocks = crackfortran.crackfortran([str(src)])
    assert blocks[0]["vars"]["x"]["typespec"] == "complex"


def test_nested_include_skips_included_directory(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    sub = cwd / "sub"
    sub.mkdir(parents=True)
    (sub / "n.inc").write_text("      integer x\n")
    (sub / "a.inc").write_text("      include 'n.inc'\n")
    src = cwd / "main.f"
    src.write_text(_lines([
        "      subroutine s(x)",
        "      include 'sub/a.inc'",
        "      end",
    ]))
    monkeypatch.chdir(cwd)
    var = _var_x(src)
    assert var["typespec"] == "real"


def test_each_source_file_has_its_include_directory(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "n.inc").write_text("      integer x\n")
    (two / "n.inc").write_text("      complex x\n")
    (one / "a.f").write_text(_lines([
        "      subroutine sa(x)",
        "      include 'n.inc'",
        "      end",
    ]))
    (two / "b.f").write_text(_lines([
        "      subroutine sb(x)",
        "      include 'n.inc'",
        "      end",
    ]))
    crackfortran.reset_global_f2py_vars()
    blocks = crackfortran.crackfortran([str(one / "a.f"), str(two / "b.f")])
    kinds = {block["name"]: block["vars"]["x"]["typespec"] for block in blocks}
    assert kinds == {"sa": "integer", "sb": "complex"}


def test_include_path_before_cwd(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    srcdir = tmp_path / "src"
    idir = tmp_path / "idir"
    cwd.mkdir()
    srcdir.mkdir()
    idir.mkdir()
    (cwd / "n.inc").write_text("      real x\n")
    (idir / "n.inc").write_text("      integer x\n")
    src = srcdir / "main.f"
    src.write_text(_lines([
        "      subroutine s(x)",
        "      include 'n.inc'",
        "      end",
    ]))
    monkeypatch.chdir(cwd)
    crackfortran.reset_global_f2py_vars()
    crackfortran.include_paths[:] = [str(idir)]
    blocks = crackfortran.crackfortran([str(src)])
    assert blocks[0]["vars"]["x"]["typespec"] == "integer"


def test_fixed_include_zero_column_starts_statement(tmp_path):
    src = tmp_path / "z.f"
    src.write_text(_lines([
        "      subroutine zed(a, b)",
        "      integer a",
        "      include 'z.inc'",
        "      end",
    ]))
    (tmp_path / "z.inc").write_text("     0real b\n")
    crackfortran.reset_global_f2py_vars()
    block = crackfortran.crackfortran([str(src)])[0]
    assert block["vars"]["a"]["typespec"] == "integer"
    assert block["vars"]["b"]["typespec"] == "real"


def test_fixed_include_tab_digit_continues(tmp_path):
    src = tmp_path / "t.f"
    src.write_text(_lines([
        "      subroutine s(x)",
        "      real*8, intent(out) ::",
        "      include 't.inc'",
        "      end",
    ]))
    (tmp_path / "t.inc").write_text("\t1 x\n")
    _assert_real_out(_var_x(src))


def test_fixed_include_bang_column_continues(tmp_path):
    src = tmp_path / "b.f"
    src.write_text(_lines([
        "      subroutine s(x)",
        "      real*8, intent(out) ::",
        "      include 'b.inc'",
        "      end",
    ]))
    (tmp_path / "b.inc").write_text("     ! x\n")
    _assert_real_out(_var_x(src))


def test_fixed_include_indented_bang_continues(tmp_path):
    # An indented ! comment does not end the continuation scan.
    src = tmp_path / "s.f"
    src.write_text(_lines([
        "      subroutine s(x)",
        "      real*8, intent(out) ::",
        "      include 't.inc'",
        "      end",
    ]))
    (tmp_path / "t.inc").write_text(_lines([
        "      ! name",
        "     * x",
    ]))
    _assert_real_out(_var_x(src))


def test_fixed_include_zero_column_splits_statements(tmp_path):
    src = tmp_path / "z.f"
    src.write_text(_lines([
        "      subroutine zed(a, x)",
        "      include 'z.inc'",
        "      end",
    ]))
    (tmp_path / "z.inc").write_text(_lines([
        "      integer a",
        "     0double precision x",
    ]))
    crackfortran.reset_global_f2py_vars()
    block = crackfortran.crackfortran([str(src)])[0]
    assert block["vars"]["a"]["typespec"] == "integer"
    assert block["vars"]["x"]["typespec"] == "double precision"


def test_fixed_include_bang_column_keeps_names(tmp_path):
    src = tmp_path / "b.f"
    src.write_text(_lines([
        "      subroutine s(x, y)",
        "      include 'b.inc'",
        "      end",
    ]))
    (tmp_path / "b.inc").write_text(_lines([
        "      real*8 ::",
        "     ! x, y",
    ]))
    crackfortran.reset_global_f2py_vars()
    block = crackfortran.crackfortran([str(src)])[0]
    for name in ("x", "y"):
        var = block["vars"][name]
        assert var["typespec"] == "real"
        selector = var["kindselector"]
        assert "8" in (selector.get("kind"), selector.get("*"))


def test_nested_fixed_include_continues_declaration(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "main.f").write_text(_lines([
        "      subroutine s(x)",
        "      real*8, intent(out) ::",
        "      include 'sub/a2.inc'",
        "      x = 1.0d0 + 1.0d-10",
        "      end",
    ]))
    (sub / "a2.inc").write_text("      include 'n.inc'\n")
    (tmp_path / "n.inc").write_text("     * x\n")
    _assert_real_out(_var_x(tmp_path / "main.f"))


@pytest.mark.slow
def test_include_continues_declaration_compiled(tmp_path):
    src = tmp_path / "s.f90"
    src.write_text(_lines([
        "subroutine s(x)",
        "  real(kind=8), intent(out) :: &",
        "  include 'kind.inc'",
        "  x = 1.0d0 + 1.0d-10",
        "end subroutine s",
    ]))
    (tmp_path / "kind.inc").write_text("  & x\n")
    mod = util.build_module(
        [str(src)],
        options=[f"--include-paths {tmp_path}"],
        module_name="inccontf90",
    )
    assert mod.s() == 1.0 + 1.0e-10
