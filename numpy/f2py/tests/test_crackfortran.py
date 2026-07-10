import contextlib
import importlib
import io
import textwrap
import time

import pytest

import numpy as np
from numpy.f2py import crackfortran
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
        a = 0
        length = size(a)
      end function

      subroutine get_inv_arr_size_{count}(a, n)
        integer :: n
        ! the value of n is computed in f2py wrapper
        !f2py intent(out) n
        integer, dimension({dimspec}), intent(in) :: a
        if (a({first}).gt.0) then
          ! print*, "a=", a
        endif
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
        lst = [(d.split(":")[0] if ":" in d else "1") for d in dimspec.split(',')]
        code += code_template.format(
            count=count,
            dimspec=dimspec,
            first=", ".join(lst),
        )

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


class TestModuleDeclaration:
    def test_dependencies(self, tmp_path):
        fpath = util.getpath("tests", "src", "crackfortran", "foo_deps.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        assert mod[0]["vars"]["abar"]["="] == "bar('abar')"


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
        # harvest ``n`` from the literal, yet ``a`` must still precede
        # ``n = shape(a, 1)`` in sortvars so ``work`` gets ``3*n``.
        fpath = util.getpath("tests", "src", "crackfortran", "gh28700_trcon.pyf")
        mod = crackfortran.crackfortran([str(fpath)])
        iface = next(b for b in mod if b.get("name") == "gh28700_trcon")["body"][0]
        trcon = next(b for b in iface["body"] if b.get("name") == "trcon")
        vs = trcon["vars"]
        assert "depend" not in vs["diag"]
        assert vs["work"]["dimension"] == ["3 * n"]
        assert vs["work"]["depend"] == ["n"]
        assert trcon["sortvars"].index("a") < trcon["sortvars"].index("n")

        lwork_case = next(b for b in iface["body"] if b.get("name") == "trcon_lwork")
        lvs = lwork_case["vars"]
        assert lvs["lwork"]["depend"] == ["norm", "n"]
        assert lvs["lwork"]["="] == "(*norm=='i'?3*n:n)"
        assert lvs["work"]["dimension"] == ["lwork"]


class TestISOFortranEnvKinds:
    # gh-30352: resolve iso_fortran_env named kind parameters (real64, int32,
    # ...) and their use-renames so kindselectors map to numeric kinds.
    @staticmethod
    def _kindselector(mods, modname, subname, varname):
        for block in mods:
            if block["name"] != modname:
                continue
            for sub in block["body"]:
                if sub["name"] == subname:
                    return sub["vars"][varname].get("kindselector")
        raise AssertionError(f"{modname}:{subname}:{varname} not found")

    def test_iso_fortran_env_kinds(self):
        fpath = util.getpath("tests", "src", "crackfortran", "gh30352.f90")
        mods = crackfortran.crackfortran([str(fpath)])
        ks = self._kindselector
        # only-list rename: complex(kind=dp), dp => real64  ->  kind 8
        assert ks(mods, "test_only_rename", "s_complex", "arr") == {"kind": "8"}
        # bare use imports every kind constant
        assert ks(mods, "test_bare_use", "s_real", "x") == {"kind": "8"}
        assert ks(mods, "test_bare_use", "s_int", "y") == {"kind": "4"}
        # a kind absent from the only-list stays unresolved
        assert ks(mods, "test_only_missing", "s_unresolved", "x") == {
            "kind": "real64"
        }

    def test_resolve_intrinsic_use_scoping(self):
        # bare use: all constants visible
        assert crackfortran._resolve_intrinsic_use(
            "iso_fortran_env", {}, {}
        ) == {
            "int8": "1", "int16": "2", "int32": "4", "int64": "8",
            "real32": "4", "real64": "8", "real128": "16",
        }
        # only-list rename maps the local name to the numeric kind
        assert crackfortran._resolve_intrinsic_use(
            "iso_fortran_env", {"only": 1, "map": {"dp": "real64"}}, {}
        ) == {"dp": "8"}
        # only-list without the requested kind resolves nothing
        assert crackfortran._resolve_intrinsic_use(
            "iso_fortran_env", {"only": 1, "map": {"int32": "int32"}}, {}
        ) == {"int32": "4"}

class TestNonLinearDimSpec:
    # gh-5506: a dimension bound that f2py's C-flavoured expression parser
    # cannot put over a common denominator (here `2**n`, parsed as the
    # pointer product `2 * *n`) must not abort signature generation. The
    # bound is carried through as an opaque expression instead.
    def test_pointer_deref_bound_does_not_crash(self, tmp_path):
        fpath = tmp_path / "gh5506.f90"
        fpath.write_text(textwrap.dedent("""\
            subroutine blah(list, n)
                integer, intent(in) :: n
                real, intent(out) :: list(2*n,0:2**n)
                list = 1.0
            end subroutine
            """))
        mod = crackfortran.crackfortran([str(fpath)])
        (var,) = (mod[0]["vars"][k] for k in ["list"])
        # linear first extent resolves, non-linear second extent is opaque
        assert var["dimension"] == ["2 * n", "1 + 2 * *n"]
        assert var["depend"] == ["n"]


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


class TestPyfInterfaceAttrs:
    """gh-13553: f2py-only attrs must not leak into Fortran wrapper interfaces.

    When a .pyf already carries ``depend``/``check``/``required`` (as written by
    ``f2py -h``), re-cracking it and emitting ``saved_interface`` used to copy
    those attributes into ``*-f2pywrappers*.f90``, which gfortran rejects.
    ``as_interface=True`` must strip them while leaving them in .pyf output.

    Note: ``saved_interface`` is snapshotted *before* ``analyzevars`` peels
    ``check``/``depend`` out of ``attrspec``, so the strip must also filter
    those strings while they still live in ``attrspec``.
    """

    def test_as_interface_strips_f2py_only_attrs(self):
        fpath = util.getpath("tests", "src", "crackfortran", "gh13553.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        # Free-function source cracks to a top-level function block.
        rout = mod[0]
        assert rout["block"] == "function"
        assert rout["name"] == "trapz"

        # .pyf emission keeps f2py-only metadata (as_interface=False).
        pyf = crackfortran.crack2fortrangen(rout, as_interface=False)
        assert "depend(" in pyf
        assert "check(" in pyf
        assert "required" in pyf

        # Fortran interface embedded in wrappers must be clean.
        iface = crackfortran.crack2fortrangen(rout, as_interface=True)
        assert "depend(" not in iface
        assert "check(" not in iface
        assert "required" not in iface
        assert "intent(in)" in iface
        assert "dimension(size(x))" in iface

    def test_as_interface_strips_attrspec_f2py_strings(self):
        # Pre-analyzevars shape of saved_interface input: check/depend still
        # live in attrspec as literal strings (not separate keys yet).
        block = {
            "block": "function",
            "name": "trapz",
            "args": ["x", "y"],
            "vars": {},
            "body": [],
        }
        vars_ = {
            "x": {
                "typespec": "real",
                "kindselector": {"*": "8"},
                "attrspec": ["required", "dimension(:)", "intent(in)"],
            },
            "y": {
                "typespec": "real",
                "kindselector": {"*": "8"},
                "attrspec": [
                    "dimension(size(x))",
                    "intent(in)",
                    "check(shape(y, 0) == size(x))",
                    "depend(x)",
                ],
            },
        }
        block["vars"] = vars_
        out = crackfortran.vars2fortran(
            block, vars_, ["x", "y"], tab="\n    ", as_interface=True
        )
        assert "required" not in out
        assert "check(" not in out
        assert "depend(" not in out
        assert "dimension(size(x))" in out
        assert "intent(in)" in out

    def test_pyf_roundtrip_wrapper_interface_clean(self, tmp_path):
        # Full codegen path: .f90 -> .pyf -> wrappers; inspect wrapper text.
        from numpy.f2py.f2py2e import run_main

        src = util.getpath("tests", "src", "crackfortran", "gh13553.f90")
        work = tmp_path / "gh13553"
        work.mkdir()
        f90 = work / "sub.f90"
        f90.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        pyf = work / "sub.pyf"

        with util.switchdir(work):
            run_main([str(f90.name), "-m", "sub", "-h", str(pyf.name),
                      "--overwrite-signature"])
            pyf_text = pyf.read_text(encoding="utf-8")
            assert "depend(" in pyf_text
            assert "check(" in pyf_text
            assert "required" in pyf_text

            run_main([str(pyf.name)])
            wrap = (work / "sub-f2pywrappers2.f90").read_text(encoding="utf-8")

        assert "depend(" not in wrap
        assert "check(" not in wrap
        assert "required" not in wrap
        # Declared dummies still carry valid Fortran shape/intent.
        assert "dimension(size(x))" in wrap
        assert "intent(in)" in wrap
