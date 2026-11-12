import contextlib
import importlib
import io
import subprocess
import sys
import textwrap
import time

import pytest

import numpy as np
from numpy.f2py import capi_maps, crackfortran
from numpy.f2py.auxfuncs import getargs2
from numpy.f2py.crackfortran import markinnerspaces, nameargspattern

from . import util


@pytest.mark.slow
class TestNoSpace(util.F2PyTest):
    # issue gh-15035: add handling for endsubroutine, endfunction with no space
    # between "end" and the block name
    sources = [util.getpath("tests", "src", "crackfortran", "gh15035.f")]

    def test_module(self):
        k = np.array([1, 2, 3], dtype=np.float64)
        w = np.array([1, 2, 3], dtype=np.float64)
        self.module.subb(k)
        assert np.allclose(k, w + 1)
        self.module.subc([w, k])
        assert np.allclose(k, w + 1)
        assert self.module.t0("23") == b"2"


class TestPublicPrivate:
    def test_defaultPrivate(self):
        fpath = util.getpath("tests", "src", "crackfortran", "privatemod.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        mod = mod[0]
        assert "private" in mod["vars"]["a"]["attrspec"]
        assert "public" not in mod["vars"]["a"]["attrspec"]
        assert "private" in mod["vars"]["b"]["attrspec"]
        assert "public" not in mod["vars"]["b"]["attrspec"]
        assert "private" not in mod["vars"]["seta"]["attrspec"]
        assert "public" in mod["vars"]["seta"]["attrspec"]

    def test_defaultPublic(self, tmp_path):
        fpath = util.getpath("tests", "src", "crackfortran", "publicmod.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        mod = mod[0]
        assert "private" in mod["vars"]["a"]["attrspec"]
        assert "public" not in mod["vars"]["a"]["attrspec"]
        assert "private" not in mod["vars"]["seta"]["attrspec"]
        assert "public" in mod["vars"]["seta"]["attrspec"]

    def test_access_type(self, tmp_path):
        fpath = util.getpath("tests", "src", "crackfortran", "accesstype.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        tt = mod[0]['vars']
        assert set(tt['a']['attrspec']) == {'private', 'bind(c)'}
        assert set(tt['b_']['attrspec']) == {'public', 'bind(c)'}
        assert set(tt['c']['attrspec']) == {'public'}

    def test_access_type_not_variable(self):
        # gh-21160
        fpath = util.getpath("tests", "src", "crackfortran", "accesstype.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        tt = mod[0]['vars']
        assert not any('typespec' in tt[n] for n in ('a', 'b_', 'c'))
        pyf = crackfortran.crack2fortran(mod)
        declared = {line.split('::')[-1].strip()
                    for line in pyf.splitlines() if '::' in line}
        assert declared.isdisjoint({'a', 'b_', 'c'})

    def test_nowrap_private_procedures(self, tmp_path):
        fpath = util.getpath("tests", "src", "crackfortran", "gh23879.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        pyf = crackfortran.crack2fortran(mod)
        assert 'bar1337baz' not in pyf

class TestModuleProcedure:
    def test_moduleOperators(self, tmp_path):
        fpath = util.getpath("tests", "src", "crackfortran", "operators.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        mod = mod[0]
        assert "body" in mod and len(mod["body"]) == 9
        assert mod["body"][1]["name"] == "operator(.item.)"
        assert "implementedby" in mod["body"][1]
        assert mod["body"][1]["implementedby"] == \
            ["item_int", "item_real"]
        assert mod["body"][2]["name"] == "operator(==)"
        assert "implementedby" in mod["body"][2]
        assert mod["body"][2]["implementedby"] == ["items_are_equal"]
        assert mod["body"][3]["name"] == "assignment(=)"
        assert "implementedby" in mod["body"][3]
        assert mod["body"][3]["implementedby"] == \
            ["get_int", "get_real"]

    def test_procedure_statements(self):
        # gh-19896: every procedure statement of a generic interface counts
        fpath = util.getpath("tests", "src", "crackfortran", "gh19896.f90")
        mod = crackfortran.crackfortran([str(fpath)])[0]
        procs = {b["name"]: b["implementedby"] for b in mod["body"]
                 if b["block"] == "interface"}
        assert procs == {
            "operator(.dot.)": ["dot_real", "dot_int"],
            "operator(+)": ["union_bn"],
            "assignment(=)": ["int_from_char", "real_from_char"],
        }
        routines = {b["name"] for b in mod["body"]
                    if b["block"] in ("function", "subroutine")}
        assert set().union(*procs.values()) <= routines

    def test_generic_spec_blanks(self):
        # gh-19896: blanks around the operator do not change the name
        fpath = util.getpath("tests", "src", "crackfortran", "gh19896.f90")
        mod = crackfortran.crackfortran([str(fpath)])[0]
        names = [b["name"] for b in mod["body"] if b["block"] == "interface"]
        assert names == ["operator(.dot.)", "operator(+)", "assignment(=)"]
        for name in names:
            assert mod["vars"][name]["attrspec"] == ["public"]

    def test_signature_file_procedures(self, tmp_path):
        # gh-19896: a signature file keeps the procedure statements
        def procedures(mod):
            return {b["name"]: b["implementedby"] for b in mod[0]["body"]
                    if b["block"] == "interface"}

        fpath = util.getpath("tests", "src", "crackfortran", "gh19896.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        pyf = tmp_path / "gh19896.pyf"
        pyf.write_text(crackfortran.crack2fortran(mod))
        assert "module procedure dot_real, dot_int" in pyf.read_text()
        assert procedures(crackfortran.crackfortran([str(pyf)])) == \
            procedures(mod)

    def test_notPublicPrivate(self, tmp_path):
        fpath = util.getpath("tests", "src", "crackfortran", "pubprivmod.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        mod = mod[0]
        assert mod['vars']['a']['attrspec'] == ['private', ]
        assert mod['vars']['b']['attrspec'] == ['public', ]
        assert mod['vars']['seta']['attrspec'] == ['public', ]


@pytest.mark.slow
class TestExternal(util.F2PyTest):
    # issue gh-17859: add external attribute support
    sources = [util.getpath("tests", "src", "crackfortran", "gh17859.f")]

    def test_external_as_statement(self):
        def incr(x):
            return x + 123

        r = self.module.external_as_statement(incr)
        assert r == 123

    def test_external_as_attribute(self):
        def incr(x):
            return x + 123

        r = self.module.external_as_attribute(incr)
        assert r == 123


class TestExternalStatementAndType:
    # gh-28605: ``external fcn`` followed by a type statement for ``fcn``
    # recorded the dummy once per statement
    def test_externals_once(self):
        fpath = util.getpath("tests", "src", "crackfortran", "gh17859.f")
        mod = crackfortran.crackfortran([str(fpath)])
        rout = next(b for b in mod if b["name"] == "external_as_statement")
        assert rout["externals"] == ["fcn"]


@pytest.mark.slow
def test_gh28605_typed_external_without_call(tmp_path):
    # `external p` plus `real*8 p` and no call used to emit invalid C.
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler available")
    src = util.getpath("tests", "src", "crackfortran", "gh28605.f90")
    (tmp_path / "gh28605.f90").write_text(src.read_text())
    code = """
import sys
sys.path.insert(0, ".")
sys.argv = ["f2py", "-c", "-m", "gh28605", "gh28605.f90",
            "--backend", "meson", "--freethreading-compatible"]
from numpy.f2py.f2py2e import main
main()
import gh28605
def unused():
    return 0.0
got = gh28605.tcb(unused)
if abs(got - 1.0) > 1e-12:
    raise SystemExit(f"tcb={got!r}")
"""
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr


@pytest.mark.slow
class TestCrackFortran(util.F2PyTest):
    # gh-2848: commented lines between parameters in subroutine parameter lists
    sources = [util.getpath("tests", "src", "crackfortran", "gh2848.f90"),
               util.getpath("tests", "src", "crackfortran", "common_with_division.f")
              ]

    def test_gh2848(self):
        r = self.module.gh2848(1, 2)
        assert r == (1, 2)

    def test_common_with_division(self):
        assert len(self.module.mortmp.ctmp) == 11

class TestMarkinnerspaces:
    # gh-14118: markinnerspaces does not handle multiple quotations

    def test_do_not_touch_normal_spaces(self):
        test_list = ["a ", " a", "a b c", "'abcdefghij'"]
        for i in test_list:
            assert markinnerspaces(i) == i

    def test_one_relevant_space(self):
        assert markinnerspaces("a 'b c' \\' \\'") == "a 'b@_@c' \\' \\'"
        assert markinnerspaces(r'a "b c" \" \"') == r'a "b@_@c" \" \"'

    def test_ignore_inner_quotes(self):
        assert markinnerspaces("a 'b c\" \" d' e") == "a 'b@_@c\"@_@\"@_@d' e"
        assert markinnerspaces("a \"b c' ' d\" e") == "a \"b@_@c'@_@'@_@d\" e"

    def test_multiple_relevant_spaces(self):
        assert markinnerspaces("a 'b c' 'd e'") == "a 'b@_@c' 'd@_@e'"
        assert markinnerspaces(r'a "b c" "d e"') == r'a "b@_@c" "d@_@e"'


@pytest.mark.slow
class TestDimSpec(util.F2PyTest):
    """This test suite tests various expressions that are used as dimension
    specifications.

    There exists two usage cases where analyzing dimensions
    specifications are important.

    In the first case, the size of output arrays must be defined based
    on the inputs to a Fortran function. Because Fortran supports
    arbitrary bases for indexing, for instance, `arr(lower:upper)`,
    f2py has to evaluate an expression `upper - lower + 1` where
    `lower` and `upper` are arbitrary expressions of input parameters.
    The evaluation is performed in C, so f2py has to translate Fortran
    expressions to valid C expressions (an alternative approach is
    that a developer specifies the corresponding C expressions in a
    .pyf file).

    In the second case, when user provides an input array with a given
    size but some hidden parameters used in dimensions specifications
    need to be determined based on the input array size. This is a
    harder problem because f2py has to solve the inverse problem: find
    a parameter `p` such that `upper(p) - lower(p) + 1` equals to the
    size of input array. In the case when this equation cannot be
    solved (e.g. because the input array size is wrong), raise an
    error before calling the Fortran function (that otherwise would
    likely crash Python process when the size of input arrays is
    wrong). f2py currently supports this case only when the equation
    is linear with respect to unknown parameter.

    """

    suffix = ".f90"

    code_template = textwrap.dedent("""
      function get_arr_size_{count}(a, n) result (length)
        integer, intent(in) :: n
        integer, dimension({dimspec}), intent(out) :: a
        integer length
        length = size(a)
      end function

      subroutine get_inv_arr_size_{count}(a, n)
        integer :: n
        ! the value of n is computed in f2py wrapper
        !f2py intent(out) n
        integer, dimension({dimspec}), intent(in) :: a
        ! a is empty for some n (e.g. 2:n with n=1), so it must not be indexed
      end subroutine
    """)

    linear_dimspecs = [
        "n", "2*n", "2:n", "n/2", "5 - n/2", "3*n:20", "n*(n+1):n*(n+5)",
        "2*n, n"
    ]
    nonlinear_dimspecs = ["2*n:3*n*n+2*n"]
    all_dimspecs = linear_dimspecs + nonlinear_dimspecs

    code = ""
    for count, dimspec in enumerate(all_dimspecs):
        code += code_template.format(count=count, dimspec=dimspec)

    @pytest.mark.parametrize("dimspec", all_dimspecs)
    def test_array_size(self, dimspec):

        count = self.all_dimspecs.index(dimspec)
        get_arr_size = getattr(self.module, f"get_arr_size_{count}")

        for n in [1, 2, 3, 4, 5]:
            sz, a = get_arr_size(n)
            assert a.size == sz

    @pytest.mark.parametrize("dimspec", all_dimspecs)
    def test_inv_array_size(self, dimspec):

        count = self.all_dimspecs.index(dimspec)
        get_arr_size = getattr(self.module, f"get_arr_size_{count}")
        get_inv_arr_size = getattr(self.module, f"get_inv_arr_size_{count}")

        for n in [1, 2, 3, 4, 5]:
            sz, a = get_arr_size(n)
            if dimspec in self.nonlinear_dimspecs:
                # one must specify n as input, the call we'll ensure
                # that a and n are compatible:
                n1 = get_inv_arr_size(a, n)
            else:
                # in case of linear dependence, n can be determined
                # from the shape of a:
                n1 = get_inv_arr_size(a)
            # n1 may be different from n (for instance, when `a` size
            # is a function of some `n` fraction) but it must produce
            # the same sized array
            sz1, _ = get_arr_size(n1)
            assert sz == sz1, (n, n1, sz, sz1)


class TestAssumedShapeLowerBound:
    # gh-25421: a lower bound alone, as in a(0:), declares an assumed shape
    @pytest.mark.parametrize("dimspec, expected", [
        ("0:", [":"]),
        ("1:", [":"]),
        ("-1:, 2:", [":", ":"]),
        ("0:, :", [":", ":"]),
        ("0:*", ["*"]),
        ("0:n", ["1 + n"]),
    ])
    def test_dimension(self, tmp_path, dimspec, expected):
        fpath = tmp_path / "gh25421.f90"
        fpath.write_text(textwrap.dedent(f"""
            subroutine sub(a, n)
              integer, intent(in) :: n
              real, intent(in) :: a({dimspec})
            end subroutine sub
            """))
        mod = crackfortran.crackfortran([str(fpath)])
        assert mod[0]["vars"]["a"]["dimension"] == expected


class TestModuleDeclaration:
    def test_dependencies(self, tmp_path):
        fpath = util.getpath("tests", "src", "crackfortran", "foo_deps.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        assert mod[0]["vars"]["abar"]["="] == "bar('abar')"


class TestSignatureInterface:
    # gh-13553: the Fortran wrapper interface of a routine read from a
    # signature file must not contain f2py-only syntax
    def test_saved_interface(self):
        fpath = util.getpath("tests", "src", "assumed_shape", "gh13553.pyf")
        mod = crackfortran.crackfortran([str(fpath)])
        routines = mod[0]["body"][0]["body"]
        lines = {
            r["name"]: [
                line.strip() for line in r["saved_interface"].splitlines()
                if line.strip()
            ]
            for r in routines
        }
        assert "real*8, dimension(size(x)),intent(in) :: y" in lines["trapz"]
        assert "integer, intent(in) :: n" in lines["scale_by"]
        assert "real(kind=8), dimension(:),intent(in) :: a" in lines["fill"]
        assert "integer :: n" in lines["fill"]
        assert lines["dsum"][0].startswith("function dsum_impl")
        assert lines["dsum"][0].endswith("result (dsum)")
        assert "real(kind=8) :: dsum" in lines["dsum"]
        interfaces = "".join(r["saved_interface"] for r in routines)
        for word in ("required", "optional", "check", "depend", "hide",
                     "threadsafe", "fortranname"):
            assert word not in interfaces


class TestUseStatement:
    # gh-17563: USE statements with a module-nature (F2018 R1409)
    def test_module_nature(self, tmp_path):
        src = tmp_path / "use_nature.f90"
        src.write_text(textwrap.dedent("""\
            module use_nature
              use, intrinsic :: iso_fortran_env, only: wp => real64
              use, non_intrinsic :: helper
              use :: other, only: x
            end module use_nature
            """))
        mod = crackfortran.crackfortran([str(src)])
        assert mod[0]["use"] == {
            "iso_fortran_env": {"only": 1, "map": {"wp": "real64"}},
            "helper": {},
            "other": {"only": 1, "map": {"x": "x"}},
        }

    def test_repeated_use_merges(self, tmp_path):
        # gh-17563: a second USE of a module must not discard the first
        src = tmp_path / "use_repeat.f90"
        src.write_text(textwrap.dedent("""\
            subroutine use_repeat(i, x)
              use iso_fortran_env, only: sp => real32, dp => real64
              use iso_fortran_env, only: int32
              use helper, only: a
              use helper
              integer(int32), intent(in) :: i
              real(dp), intent(out) :: x
            end subroutine use_repeat
            """))
        mod = crackfortran.crackfortran([str(src)])
        assert mod[0]["use"] == {
            "iso_fortran_env": {
                "only": 1,
                "map": {"sp": "real32", "dp": "real64", "int32": "int32"}},
            "helper": {"only": 0, "map": {"a": "a"}},
        }
        pyf = crackfortran.crack2fortran(mod)
        assert "use iso_fortran_env, only: sp=>real32,dp=>real64,int32" in pyf
        assert "use helper, a=>a" in pyf

    def test_host_use_merges(self, tmp_path):
        src = tmp_path / "use_host.f90"
        src.write_text(textwrap.dedent("""\
            module use_host
              use iso_fortran_env, only: wp => real64
            contains
              subroutine s(n)
                use iso_fortran_env, only: int32
                integer(int32), intent(in) :: n
              end subroutine s
            end module use_host
            """))
        mod = crackfortran.crackfortran([str(src)])
        s = mod[0]["body"][0]
        assert crackfortran.get_usedict(s)["iso_fortran_env"] == {
            "only": 1, "map": {"wp": "real64", "int32": "int32"}}

    def test_renamed_intrinsic_kinds(self, tmp_path):
        # gh-17563, gh-30352: renamed kinds resolve to the intrinsic names
        src = tmp_path / "use_rename.f90"
        src.write_text(textwrap.dedent("""\
            module use_rename
              use, intrinsic :: iso_fortran_env, only: wp => real64
              use, intrinsic :: iso_c_binding, only: cdp => c_double
            contains
              subroutine s(x, z, n)
                use iso_fortran_env, only: ik => int64
                real(wp), intent(inout) :: x
                complex(kind=cdp), intent(inout) :: z
                integer(ik), intent(in) :: n
              end subroutine s
            end module use_rename
            """))
        mod = crackfortran.crackfortran([str(src)])
        vs = mod[0]["body"][0]["vars"]
        assert [vs[n]["kindselector"]["kind"] for n in "xzn"] == [
            "real64", "c_double", "int64"]
        assert [capi_maps.getctype(vs[n]) for n in "xzn"] == [
            "double", "complex_double", "long_long"]

    def test_kind_names_ignore_case(self, tmp_path, monkeypatch):
        # signature files keep their case unless --lower is given
        monkeypatch.setattr(crackfortran, "dolowercase", 0)
        src = tmp_path / "use_case.pyf"
        src.write_text(textwrap.dedent("""\
            python module use_case
                interface
                    subroutine s(x, z, n)
                        use iso_fortran_env
                        use iso_c_binding, only: DP => C_DOUBLE
                        REAL(KIND=REAL64), intent(inout) :: x
                        complex(dp), intent(inout) :: z
                        integer(Int64), intent(in) :: n
                    end subroutine s
                end interface
            end python module use_case
            """))
        mod = crackfortran.crackfortran([str(src)])
        vs = mod[0]["body"][0]["body"][0]["vars"]
        assert [capi_maps.getctype(vs[n]) for n in "xzn"] == [
            "double", "complex_double", "long_long"]


class TestGH30352Cases:
    """Cases from gh-30352 that the gh-17563 tests do not already build."""

    @pytest.mark.parametrize("stmt", [
        "use, intrinsic :: iso_fortran_env, only: dp => real64",
        "use,intrinsic::iso_fortran_env, dp => real64",
        "use :: iso_fortran_env, only: dp => real64",
    ])
    def test_intrinsic_spellings(self, tmp_path, capsys, stmt):
        fpath = tmp_path / "nature.f90"
        fpath.write_text(
            "module m\n"
            f"  {stmt}\n"
            "contains\n"
            "  subroutine s(x)\n"
            "    complex(kind=dp), intent(inout) :: x\n"
            "  end subroutine s\n"
            "end module m\n"
        )
        mod = crackfortran.crackfortran([str(fpath)])
        assert "Could not crack the use statement" not in capsys.readouterr().out
        var = mod[0]["body"][0]["vars"]["x"]
        assert var["kindselector"] == {"kind": "real64"}
        assert capi_maps.getctype(var) == "complex_double"

    def test_non_intrinsic_only(self, tmp_path):
        fpath = tmp_path / "nature.f90"
        fpath.write_text(textwrap.dedent("""\
            module kinds
              integer, parameter :: dp = 8
            end module kinds
            module m
              use, non_intrinsic :: kinds, only: wp => dp
            end module m
            """))
        mod = crackfortran.crackfortran([str(fpath)])
        assert mod[1]["use"] == {"kinds": {"only": 1, "map": {"wp": "dp"}}}

    def test_rename_without_lowercasing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crackfortran, "dolowercase", 0)
        fpath = tmp_path / "upper.f90"
        fpath.write_text(
            "MODULE M\n"
            "  USE ISO_FORTRAN_ENV, ONLY: DP => REAL64\n"
            "CONTAINS\n"
            "  SUBROUTINE S(X)\n"
            "    COMPLEX(KIND=DP), INTENT(INOUT) :: X\n"
            "  END SUBROUTINE S\n"
            "END MODULE M\n"
        )
        mod = crackfortran.crackfortran([str(fpath)])
        var = mod[0]["body"][0]["vars"]["X"]
        assert var["kindselector"] == {"kind": "real64"}
        assert capi_maps.getctype(var) == "complex_double"
class TestUseAssociation:
    # gh-3562: module data accessed by USE belongs to the using module too
    def test_use_associated_vars(self, tmp_path, monkeypatch):
        fpath = tmp_path / "gh3562.f90"
        fpath.write_text(textwrap.dedent("""\
            module chained
                use test2
            end module chained

            module test
                implicit none
                integer, parameter :: a = 1
                integer, private :: hidden = 7
                type :: point
                    integer :: x
                end type point
                type(point) :: origin
            contains
                function twice(x)
                    integer :: x, twice
                    twice = 2*x
                end function twice
            end module test

            module other
                integer, parameter :: a = 2
            end module other

            module test2
                use test
                integer, parameter :: b = 2
            end module test2

            module clash
                use test
                use other
            end module clash

            module quiet
                use test
                use other, only: c => a
                private :: test
            end module quiet
            """))
        monkeypatch.setattr(crackfortran, "f77modulename", "gh3562")
        mod = crackfortran.crackfortran([str(fpath)])
        modvars = {m["name"]: m["vars"] for m in mod[0]["body"][0]["body"]}
        assert set(modvars["test2"]) == {"a", "b"}
        assert modvars["test2"]["a"]["="] == "1"
        assert set(modvars["chained"]) == {"a", "b"}
        # different entities named a: ambiguous, so not exposed
        assert "a" not in modvars["clash"]
        # F2018 8.6.1: a module name in an access statement sets the default
        assert "a" not in modvars["quiet"]
        assert modvars["quiet"]["c"]["="] == "2"


class TestStringLiteralDepend:
    # gh-28700: identifiers inside a character parameter's literal value
    # must not be harvested as dependencies (a self-dependency crashes the
    # dependency sort during a -c build).
    def test_mask_string_literals(self):
        mask = crackfortran.mask_string_literals
        assert mask("'mkdir '") == "''"
        assert mask('"badvar2"') == "''"
        assert mask("'mkdir '//badvar2") == "''//badvar2"
        assert mask("a + b") == "a + b"

    def test_no_self_depend_from_literal(self):
        fpath = util.getpath("tests", "src", "crackfortran", "gh28700.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        m1 = next(b for b in mod if b.get("name") == "mod1")
        vs = m1["vars"]
        # string-only values gain no dependency at all
        assert "depend" not in vs["mkdir"]
        assert "depend" not in vs["badvar2"]
        # a real identifier outside the quotes is still a dependency,
        # while the one inside the literal is dropped
        assert vs["realdep"]["depend"] == ["badvar2"]

    def test_trcon_literal_masking_and_sort_order(self):
        # SciPy flapack ?trcon regression: masking ``diag = 'N'`` must not
        # harvest ``n`` from the literal. Cycle fallback must order
        # ``a`` then ``n = shape(a, 1)`` then ``work`` so getarrdims keeps
        # ``work_Dims[0]=3*n`` (not blanked to assumed-shape).
        fpath = util.getpath("tests", "src", "crackfortran", "gh28700_trcon.pyf")
        mod = crackfortran.crackfortran([str(fpath)])
        iface = next(b for b in mod if b.get("name") == "gh28700_trcon")["body"][0]
        trcon = next(b for b in iface["body"] if b.get("name") == "trcon")
        vs = trcon["vars"]
        assert "depend" not in vs["diag"]
        assert vs["work"]["dimension"] == ["3 * n"]
        assert vs["work"]["depend"] == ["n"]
        order = trcon["sortvars"]
        assert order.index("a") < order.index("n")
        assert order.index("n") < order.index("work")
        assert order.index("n") < order.index("lda")

        lwork_case = next(b for b in iface["body"] if b.get("name") == "trcon_lwork")
        lvs = lwork_case["vars"]
        # depend harvest order follows vars dict iteration; compare as a set
        assert set(lvs["lwork"]["depend"]) == {"norm", "n"}
        assert lvs["lwork"]["="] == "(*norm=='i'?3*n:n)"
        assert lvs["work"]["dimension"] == ["lwork"]

    def test_lda_declared_before_n(self):
        # Declaration order must not put ``lda = MAX(1, n)`` ahead of
        # ``n = shape(a, 1)``. getarrdims blanks a dimension that names a
        # later depargs entry, so ``work`` must follow ``n`` as well.
        fpath = util.getpath("tests", "src", "crackfortran",
                             "gh28700_lda_before_n.pyf")
        mod = crackfortran.crackfortran([str(fpath)])
        iface = next(b for b in mod
                     if b.get("name") == "gh28700_lda_before_n")["body"][0]
        probe = next(b for b in iface["body"] if b.get("name") == "probe")
        order = probe["sortvars"]
        assert order.index("a") < order.index("n")
        assert order.index("n") < order.index("lda")
        assert order.index("n") < order.index("work")
        _, depargs = getargs2(probe)
        capi_maps.depargs = depargs
        work_dims = capi_maps.getarrdims("work", probe["vars"]["work"])
        assert work_dims["setdims"] == "#varname#_Dims[0]=3 * n"
        assert capi_maps.getarrdims("a", probe["vars"]["a"])["setdims"] == ""

    def test_true_cycle_does_not_recurse(self, tmp_path, capsys):
        # Names inside character literals, and a genuine parameter cycle,
        # must be reported instead of blowing the stack in _get_depend_dict.
        src = tmp_path / "cycle.f90"
        src.write_text(
            "module nonchar_literal_cycle\n"
            "  integer, parameter :: n = len('m')\n"
            "  integer, parameter :: m = len('n')\n"
            "end module nonchar_literal_cycle\n"
            "module char_real_cycle\n"
            "  character(len=8), parameter :: aaa = bbb\n"
            "  character(len=8), parameter :: bbb = aaa\n"
            "end module char_real_cycle\n"
            "subroutine uses_cycle\n"
            "  use char_real_cycle\n"
            "end subroutine uses_cycle\n"
        )
        mod = crackfortran.crackfortran([str(src)])
        assert any(b.get("name") == "nonchar_literal_cycle" for b in mod)
        err = capsys.readouterr().err
        assert "cyclic dependencies" in err

    def test_backslash_is_not_an_escape(self):
        # gh-33220: a backslash does not escape the closing quote, so
        # the name after the literal stays visible and the line parses.
        mask = crackfortran.mask_string_literals
        expr = "len('c:\\'//n)"
        assert mask(expr) == "len(''//n)"
        assert crackfortran.markoutercomma(expr) == expr
        fpath = util.getpath("tests", "src", "crackfortran", "gh33220.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        probe = next(b for b in mod if b.get("name") == "probe")
        assert set(probe["vars"]) == {"n", "got"}


@pytest.mark.slow
class TestBackslashCharLiteral(util.F2PyTest):
    # gh-33220. len('c:\') is 3; len(n) follows the Python string.
    sources = [util.getpath("tests", "src", "crackfortran", "gh33220.f90")]

    def test_length_keeps_name_after_backslash(self):
        assert self.module.probe("ABCD") == 7
        assert self.module.probe("Z") == 4


@pytest.mark.slow
class TestLdaBeforeNRuntime(util.F2PyTest):
    # Fortran-order array shape (5, 4): shape(a, 1) is a_Dims[1] == 4,
    # and lda = MAX(1, n) must be computed from that n, not from 0.
    # The .pyf names the extension; f2py ignores -m in that case.
    module_name = "gh28700_lda_before_n"
    sources = [
        util.getpath("tests", "src", "crackfortran", "gh28700_lda_before_n.pyf"),
        util.getpath("tests", "src", "crackfortran", "gh28700_lda_before_n.f90"),
    ]

    def test_shape_then_lda(self):
        a = np.ones((5, 4), dtype=np.float32, order="F")
        gotn, gotlda = self.module.probe(a)
        assert gotn == 4
        assert gotlda == 4


class TestUseRenamedKinds:
    # gh-30352: a kind renamed on use, such as ``dp => real64``, resolves to
    # the intrinsic module constant that capi_maps maps to a C type.
    def test_intrinsic_kind_renames(self):
        fpath = util.getpath("tests", "src", "crackfortran", "gh30352.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        body = next(b for b in mod if b["name"] == "gh30352")["body"]
        kinds = {
            (r["name"], a): (r["vars"][a]["kindselector"]["kind"],
                             capi_maps.getctype(r["vars"][a]))
            for r in body for a in r["args"] if "kindselector" in r["vars"][a]
        }
        assert kinds[("test_function", "arr")] == ("real64", "complex_double")
        assert kinds[("set_cdp", "arr")] == ("c_double", "complex_double")
        assert kinds[("add_dp", "a")] == ("real64", "double")
        assert kinds[("add_ik", "a")] == ("int64", "long_long")

    def test_rename_without_lowercasing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crackfortran, "dolowercase", 0)
        fpath = tmp_path / "upper.f90"
        fpath.write_text(
            "MODULE M\n"
            "  USE ISO_FORTRAN_ENV, ONLY: DP => REAL64\n"
            "CONTAINS\n"
            "  SUBROUTINE S(X)\n"
            "    COMPLEX(KIND=DP), INTENT(INOUT) :: X\n"
            "  END SUBROUTINE S\n"
            "END MODULE M\n"
        )
        mod = crackfortran.crackfortran([str(fpath)])
        x = mod[0]["body"][0]["vars"]["X"]
        assert x["kindselector"] == {"kind": "real64"}
        assert capi_maps.getctype(x) == "complex_double"


@pytest.mark.slow
class TestUseRenamedKindsRuntime(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "gh30352.f90")]

    @pytest.mark.parametrize("name", ["test_function", "set_cdp", "set_wp"])
    def test_complex_inout(self, name):
        arr = np.zeros(10, dtype=np.complex128)
        getattr(self.module.gh30352, name)(arr)
        np.testing.assert_array_equal(arr, np.full(10, 1 - 1j))

    def test_real_function(self):
        assert self.module.gh30352.add_dp(1.0, 2.0**-40) == 1 + 2.0**-40

    def test_integer_function(self):
        assert self.module.gh30352.add_ik(2**32, 2**32) == 2**33


class TestUseModuleNature:
    # A use statement may name the module nature (Fortran 2018, R1409).
    @pytest.mark.parametrize("stmt", [
        "use, intrinsic :: iso_fortran_env, only: dp => real64",
        "use,intrinsic::iso_fortran_env, dp => real64",
        "use :: iso_fortran_env, only: dp => real64",
    ])
    def test_intrinsic(self, tmp_path, capsys, stmt):
        fpath = tmp_path / "nature.f90"
        fpath.write_text(
            "module m\n"
            f"  {stmt}\n"
            "contains\n"
            "  subroutine s(x)\n"
            "    complex(kind=dp), intent(inout) :: x\n"
            "  end subroutine s\n"
            "end module m\n"
        )
        mod = crackfortran.crackfortran([str(fpath)])
        assert "Could not crack the use statement" not in capsys.readouterr().out
        assert mod[0]["use"]["iso_fortran_env"]["map"] == {"dp": "real64"}
        assert mod[0]["body"][0]["vars"]["x"]["kindselector"] == {"kind": "real64"}

    def test_non_intrinsic(self, tmp_path):
        fpath = tmp_path / "nature.f90"
        fpath.write_text(
            "module kinds\n"
            "  integer, parameter :: dp = 8\n"
            "end module kinds\n"
            "module m\n"
            "  use, non_intrinsic :: kinds, only: wp => dp\n"
            "end module m\n"
        )
        mod = crackfortran.crackfortran([str(fpath)])
        assert mod[1]["use"] == {"kinds": {"only": 1, "map": {"wp": "dp"}}}


@pytest.mark.slow
class TestEval(util.F2PyTest):
    def test_eval_scalar(self):
        eval_scalar = crackfortran._eval_scalar

        assert eval_scalar('123', {}) == '123'
        assert eval_scalar('12 + 3', {}) == '15'
        assert eval_scalar('a + b', {"a": 1, "b": 2}) == '3'
        assert eval_scalar('"123"', {}) == "'123'"


@pytest.mark.slow
class TestFortranReader(util.F2PyTest):
    @pytest.mark.parametrize("encoding",
                             ['ascii', 'utf-8', 'utf-16', 'utf-32'])
    def test_input_encoding(self, tmp_path, encoding):
        # gh-635
        f_path = tmp_path / f"input_with_{encoding}_encoding.f90"
        with f_path.open('w', encoding=encoding) as ff:
            ff.write("""
                     subroutine foo()
                     end subroutine foo
                     """)
        mod = crackfortran.crackfortran([str(f_path)])
        assert mod[0]['name'] == 'foo'


@pytest.mark.slow
class TestUnicodeComment(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "unicode_comment.f90")]

    @pytest.mark.skipif(
        (importlib.util.find_spec("charset_normalizer") is None),
        reason="test requires charset_normalizer which is not installed",
    )
    def test_encoding_comment(self):
        self.module.foo(3)


class TestNameArgsPatternBacktracking:
    @pytest.mark.parametrize(
        ['adversary'],
        [
            ('@)@bind@(@',),
            ('@)@bind                         @(@',),
            ('@)@bind foo bar baz@(@',)
        ]
    )
    def test_nameargspattern_backtracking(self, adversary):
        '''address ReDOS vulnerability:
        https://github.com/numpy/numpy/issues/23338'''
        trials_per_batch = 12
        batches_per_regex = 4
        start_reps, end_reps = 15, 25
        for ii in range(start_reps, end_reps):
            repeated_adversary = adversary * ii
            # test times in small batches.
            # this gives us more chances to catch a bad regex
            # while still catching it before too long if it is bad
            for _ in range(batches_per_regex):
                times = []
                for _ in range(trials_per_batch):
                    t0 = time.perf_counter()
                    mtch = nameargspattern.search(repeated_adversary)
                    times.append(time.perf_counter() - t0)
                # our pattern should be much faster than 0.2s per search
                # it's unlikely that a bad regex will pass even on fast CPUs
                assert np.median(times) < 0.2
            assert not mtch
            # if the adversary is capped with @)@, it becomes acceptable
            # according to the old version of the regex.
            # that should still be true.
            good_version_of_adversary = repeated_adversary + '@)@'
            assert nameargspattern.search(good_version_of_adversary)

@pytest.mark.slow
class TestFunctionReturn(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "gh23598.f90")]

    def test_function_rettype(self):
        # gh-23598
        assert self.module.intproduct(3, 4) == 12


@pytest.mark.slow
class TestFortranGroupCounters(util.F2PyTest):
    def test_end_if_comment(self):
        # gh-23533
        fpath = util.getpath("tests", "src", "crackfortran", "gh23533.f")
        try:
            crackfortran.crackfortran([str(fpath)])
        except Exception as exc:
            assert False, f"'crackfortran.crackfortran' raised an exception {exc}"


class TestF77CommonBlockReader:
    def test_gh22648(self, tmp_path):
        fpath = util.getpath("tests", "src", "crackfortran", "gh22648.pyf")
        with contextlib.redirect_stdout(io.StringIO()) as stdout_f2py:
            mod = crackfortran.crackfortran([str(fpath)])
        assert "Mismatch" not in stdout_f2py.getvalue()

    @pytest.mark.parametrize(
        "end", ["end block data", "end blockdata", "endblockdata", "end"])
    def test_end_block_data(self, tmp_path, end):
        fpath = tmp_path / "blkdata.f90"
        fpath.write_text(textwrap.dedent(f"""\
            block data init
              integer n
              common /cblk/ n
              data n /3/
            {end}
            subroutine after(k)
              integer k
            end subroutine after
            """))
        with contextlib.redirect_stdout(io.StringIO()) as stdout_f2py:
            mod = crackfortran.crackfortran([str(fpath)])
        assert "Mismatch" not in stdout_f2py.getvalue()
        assert [(b["block"], b["name"]) for b in mod] == [
            ("block data", "init"), ("subroutine", "after")]


@pytest.mark.slow
class TestBlockConstruct(util.F2PyTest):
    # END BLOCK of a Fortran 2008 BLOCK construct must not end the
    # enclosing procedure
    sources = [util.getpath("tests", "src", "crackfortran", "block_construct.f90")]

    def test_crackfortran(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout_f2py:
            mod = crackfortran.crackfortran([str(self.sources[0])])
        assert "Mismatch" not in stdout_f2py.getvalue()
        assert [b["name"] for b in mod] == [
            "block_construct", "external_with_block", "external_after_block"]
        assert [b["name"] for b in mod[0]["body"]] == [
            "with_block", "with_named_block", "after_blocks"]

    def test_wrappers(self):
        mod = self.module.block_construct
        assert mod.with_block(1.0) == 3.0
        assert mod.with_named_block(1.0) == 3.0
        assert mod.after_blocks(1.0) == 0.0
        assert self.module.external_with_block(1.0) == 11.0
        assert self.module.external_after_block(1.0) == 21.0

class TestParamEval:
    # issue gh-11612, array parameter parsing
    def test_param_eval_nested(self):
        v = '(/3.14, 4./)'
        g_params = {"kind": crackfortran._kind_func,
                "selected_int_kind": crackfortran._selected_int_kind_func,
                "selected_real_kind": crackfortran._selected_real_kind_func}
        params = {'dp': 8, 'intparamarray': {1: 3, 2: 5},
                  'nested': {1: 1, 2: 2, 3: 3}}
        dimspec = '(2)'
        ret = crackfortran.param_eval(v, g_params, params, dimspec=dimspec)
        assert ret == {1: 3.14, 2: 4.0}

    def test_param_eval_nonstandard_range(self):
        v = '(/ 6, 3, 1 /)'
        g_params = {"kind": crackfortran._kind_func,
                "selected_int_kind": crackfortran._selected_int_kind_func,
                "selected_real_kind": crackfortran._selected_real_kind_func}
        params = {}
        dimspec = '(-1:1)'
        ret = crackfortran.param_eval(v, g_params, params, dimspec=dimspec)
        assert ret == {-1: 6, 0: 3, 1: 1}

    def test_param_eval_empty_range(self):
        v = '6'
        g_params = {"kind": crackfortran._kind_func,
                "selected_int_kind": crackfortran._selected_int_kind_func,
                "selected_real_kind": crackfortran._selected_real_kind_func}
        params = {}
        dimspec = ''
        pytest.raises(ValueError, crackfortran.param_eval, v, g_params, params,
                      dimspec=dimspec)

    def test_param_eval_non_array_param(self):
        v = '3.14_dp'
        g_params = {"kind": crackfortran._kind_func,
                "selected_int_kind": crackfortran._selected_int_kind_func,
                "selected_real_kind": crackfortran._selected_real_kind_func}
        params = {}
        ret = crackfortran.param_eval(v, g_params, params, dimspec=None)
        assert ret == '3.14_dp'

    def test_param_eval_too_many_dims(self):
        v = 'reshape((/ (i, i=1, 250) /), (/5, 10, 5/))'
        g_params = {"kind": crackfortran._kind_func,
                "selected_int_kind": crackfortran._selected_int_kind_func,
                "selected_real_kind": crackfortran._selected_real_kind_func}
        params = {}
        dimspec = '(0:4, 3:12, 5)'
        pytest.raises(ValueError, crackfortran.param_eval, v, g_params, params,
                      dimspec=dimspec)


class TestParamParseNestedParens:
    # issue gh-28095: grouping parens in a dimension expression such as
    # (mx_supply_curves + mx_intl_curves) must fold to a concrete size,
    # not be mistaken for pa(index) array-parameter indexing.
    def test_grouping_parens_fold(self):
        params = {"mx_supply_curves": 14, "mx_intl_curves": 12}
        out = crackfortran.param_parse(
            "(mx_supply_curves + mx_intl_curves)", params)
        assert out.replace(" ", "") == "(14+12)"

    def test_grouping_parens_with_trailing_factor(self):
        # A grouping paren need not span the whole token; the factor
        # outside the parentheses must survive substitution.
        params = {"n": 3, "m": 1800}
        out = crackfortran.param_parse("(n + 1)*m", params)
        assert out.replace(" ", "") == "(3+1)*1800"

    def test_array_index_still_works(self):
        params = {"pa": {1: 3, 2: 5}}
        assert crackfortran.param_parse("pa(1)", params) == "3"
        assert crackfortran.param_parse("pa(2)", params) == "5"

    def test_nested_array_index(self):
        params = {"dim": 2, "nested": {1: 1, 2: 2, 3: 3},
                  "myparamarray": {1: 10, 2: 20, 3: 30}}
        assert crackfortran.param_parse(
            "myparamarray(nested(dim))", params) == "20"

    def test_dimension_substituted_in_module(self):
        fpath = util.getpath("tests", "src", "crackfortran", "gh28095.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        vs = mod[0]["vars"]["cmm_cl_btus"]
        assert vs["dimension"] == ["26", "1800"]


@pytest.mark.slow
class TestLowerF2PYDirective(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "gh27697.f90")]
    options = ['--lower']

    def test_no_lower_fail(self):
        with pytest.raises(ValueError, match='aborting directly') as exc:
            self.module.utils.my_abort('aborting directly')


class TestImplicitRules:
    def test_rules_stay_in_their_unit(self):
        fpath = util.getpath("tests", "src", "crackfortran", "implicit_units.f")
        first, second = crackfortran.crackfortran([str(fpath)])
        assert first["vars"]["x"]["typespec"] == "double precision"
        assert second["vars"]["x"]["typespec"] == "real"
        assert crackfortran.defaultimplicitrules["x"] == {"typespec": "real"}

    def test_module_procedure_uses_module_rules(self):
        fpath = util.getpath("tests", "src", "crackfortran", "implicit_module.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        sub = mod[0]["body"][0]
        assert sub["vars"]["x"]["typespec"] == "double precision"

    def test_uppercase_letters(self, monkeypatch):
        # gh-13356: without -h, ``f2py -m`` keeps the case of the source
        monkeypatch.setattr(crackfortran, "dolowercase", 0)
        fpath = util.getpath("tests", "src", "crackfortran", "implicit_upper.f")
        fvars = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        for name in ("TK", "X", "ETA1"):
            assert fvars[name]["typespec"] == "double precision"
        assert fvars["KK"]["typespec"] == "integer"
