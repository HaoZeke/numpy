import contextlib
import importlib
import io
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


class TestImplicitRules:
    @pytest.fixture(autouse=True)
    def _saved_defaults(self):
        saved = {k: dict(v) for k, v in crackfortran.defaultimplicitrules.items()}
        try:
            yield
        finally:
            crackfortran.defaultimplicitrules.clear()
            crackfortran.defaultimplicitrules.update(saved)

    def test_upper_letters_and_no_leak(self, monkeypatch, tmp_path):
        # gh-13356: keep the source case, and do not rewrite the next routine.
        monkeypatch.setattr(crackfortran, "dolowercase", 0)
        by = self._crack(tmp_path, "implicit.f90", """\
            subroutine FIRST(X)
              implicit double precision (A-H,O-Z)
              X = 1
            end

            subroutine SECOND(X)
              X = 1
            end
            """)
        assert by["FIRST"]["vars"]["X"]["typespec"] == "double precision"
        assert by["SECOND"]["vars"]["X"]["typespec"] == "real"
        assert crackfortran.defaultimplicitrules["a"]["typespec"] == "real"
        assert "A" not in crackfortran.defaultimplicitrules

    def test_implicit_none_from_host_stays_none(self):
        parent = {"name": "host", "implicit": None}
        child = {"name": "inner", "parent_block": parent}
        rules, attrs = crackfortran.buildimplicitrules(child)
        assert rules is None
        assert attrs == {}

    def _find(self, block, name):
        if isinstance(block, dict):
            if block.get("name") == name:
                return block
            for child in block.get("body") or []:
                found = self._find(child, name)
                if found:
                    return found
        return None

    def _crack(self, tmp_path, name, source):
        path = tmp_path / name
        path.write_text(textwrap.dedent(source))
        blocks = crackfortran.crackfortran([str(path)])
        return {b["name"]: b for b in blocks if isinstance(b, dict)}

    def test_interface_body_keeps_default_rules(self, tmp_path):
        # A Fortran interface body keeps the default map. A contained
        # routine and a call wrapper still see the host map.
        by = self._crack(tmp_path, "iface.f90", """\
            module none
              implicit none
            contains
              subroutine caller(cb)
                interface
                  subroutine cb(x, i)
                  end subroutine
                end interface
              end subroutine
            end module
            module host
              implicit double precision (a-h,o-z)
              interface
                subroutine ext(a)
                end subroutine
              end interface
            contains
              subroutine inner(a)
              end subroutine
            end module
            subroutine wrap(cb, ext, x)
              implicit double precision (a-h,o-z)
              external cb
              call cb(x)
              interface
                subroutine ext(a)
                end subroutine
              end interface
            end subroutine
            """)
        cb = self._find(by["none"], "cb")
        assert cb["vars"]["x"]["typespec"] == "real"
        assert cb["vars"]["i"]["typespec"] == "integer"
        assert self._find(by["host"], "ext")["vars"]["a"]["typespec"] == "real"
        assert self._find(by["host"], "inner")["vars"]["a"]["typespec"] == \
            "double precision"
        assert self._find(by["wrap__user__routines"], "cb")["vars"]["x"][
            "typespec"] == "double precision"
        assert self._find(by["wrap"], "ext")["vars"]["a"]["typespec"] == "real"

    def test_use_associated_implicit_is_folded_on_host(self, tmp_path):
        # Kind and length come from the module that defined them.
        by = self._crack(tmp_path, "kind.f90", """\
            module kinds
              integer, parameter :: wp = 4
              integer, parameter :: n = 4
            end module
            module host
              use kinds, only: wp, n
              implicit real(kind=wp) (a-b,d-h,o-z)
              implicit character(len=n) (c)
            contains
              subroutine inner(a, c)
                integer, parameter :: wp = 8
                integer, parameter :: n = 8
              end subroutine
              subroutine later(b, c)
              end subroutine
            end module
            """)
        inner = self._find(by["host"], "inner")
        later = self._find(by["host"], "later")
        assert inner["vars"]["a"]["kindselector"]["kind"] == "4"
        assert later["vars"]["b"]["kindselector"]["kind"] == "4"
        assert inner["vars"]["c"]["charselector"]["*"] == "4"
        assert later["vars"]["c"]["charselector"]["*"] == "4"
        assert "kindselector" not in crackfortran.defaultimplicitrules["a"]

    def test_use_visibility_and_scope(self, tmp_path):
        by = self._crack(tmp_path, "visibility.f90", """\
            module kinds
              integer, parameter :: wp = 4
              integer, parameter :: n = 99
              integer, parameter :: k = 8
              integer, parameter :: dp = 8
            end module
            module host
              use kinds, only: wp
              integer, parameter :: k = 4
              integer, parameter :: n = 2
              interface
                subroutine ext(a, n)
                  real :: a(n)
                end subroutine ext
                subroutine extwp(x)
                  real(kind=wp) :: x
                end subroutine extwp
              end interface
            contains
              subroutine inner(c, n)
                integer :: n
                character(len=n) :: c
              end subroutine inner
              subroutine folded(c)
                character(len=n) :: c
              end subroutine folded
              subroutine withk(a)
                real(kind=k) :: a
              end subroutine withk
              subroutine fromother(a)
                use other, only: wp
                real(kind=wp) :: a
              end subroutine fromother
            end module
            module other
              integer, parameter :: wp = 8
              integer, parameter :: dp = 4
            end module
            module secret
              integer, parameter, private :: n = 7
              integer, parameter :: wp = 4
            end module
            module foldmod
              use kinds
            contains
              subroutine uses_n(c)
                character(len=n) :: c
              end subroutine uses_n
            end module
            subroutine onlyfree(c)
              use kinds, only: wp
              character(len=n) :: c
            end subroutine onlyfree
            subroutine renamekind(a, b)
              use kinds, only: rk => wp
              real(kind=rk) :: a
              real(kind=wp) :: b
            end subroutine renamekind
            subroutine twouse(x)
              use other, only: dp
              use kinds, only: wp
              real(kind=dp) :: x
            end subroutine twouse
            subroutine private_extent(a)
              use secret
              real :: a(n)
            end subroutine private_extent
            subroutine private_dummy(a, n)
              use secret
              integer :: n
              real :: a(n)
            end subroutine private_dummy
            subroutine only_extent(a, n)
              use kinds, only: wp
              integer :: n
              real(kind=wp) :: a(n)
            end subroutine only_extent
            module wide
              integer, parameter :: wp = 8
            end module
            module hostwide
              use wide
            contains
              subroutine child(a, b)
                use wide, only: rk => wp
                real(kind=wp) :: a
                real(kind=rk) :: b
              end subroutine child
            end module
            module bothuse
            contains
              subroutine shaped(a)
                use kinds
                use kinds, only: wp
                real(kind=wp) :: a(:)
              end subroutine shaped
            end module
            module bothnames
            contains
              subroutine renameonly(a, b)
                use kinds, only: rk => wp
                use kinds, only: wp
                real(kind=rk) :: a(:)
                real(kind=wp) :: b(:)
              end subroutine renameonly
              subroutine renamebare(a, b)
                use kinds, rk => wp
                use kinds, only: wp
                real(kind=rk) :: a(:)
                real(kind=wp) :: b(:)
              end subroutine renamebare
            end module
            """)
        host = by["host"]
        inner = self._find(host, "inner")
        folded = self._find(host, "folded")
        withk = self._find(host, "withk")
        ext = self._find(host, "ext")
        extwp = self._find(host, "extwp")
        fromother = self._find(host, "fromother")
        uses_n = self._find(by["foldmod"], "uses_n")

        def kind(block, name):
            return block["vars"][name]["kindselector"]["kind"]

        def length(block, name):
            return block["vars"][name]["charselector"]["*"]

        def dim(block, name):
            return block["vars"][name]["dimension"]

        assert length(inner, "c") == "n"
        assert length(folded, "c") == "2"
        assert kind(withk, "a") == "4"
        assert dim(ext, "a") == ["n"]
        assert kind(extwp, "x") == "wp"
        assert kind(fromother, "a") == "8"
        assert length(uses_n, "c") == "99"
        assert length(by["onlyfree"], "c") == "n"
        assert kind(by["renamekind"], "a") == "4"
        assert kind(by["renamekind"], "b") == "wp"
        assert kind(by["twouse"], "x") == "4"
        assert dim(by["private_extent"], "a") == ["n"]
        assert dim(by["private_dummy"], "a") == ["n"]
        assert kind(by["only_extent"], "a") == "4"
        assert dim(by["only_extent"], "a") == ["n"]
        child = self._find(by["hostwide"], "child")
        assert (kind(child, "a"), kind(child, "b")) == ("8", "8")
        shaped = self._find(by["bothuse"], "shaped")
        assert "use kinds\n" in shaped["saved_interface"]

        def use_lines(block):
            return [
                ln.strip()
                for ln in block["saved_interface"].splitlines()
                if ln.strip().startswith("use ")
            ]

        renameonly = self._find(by["bothnames"], "renameonly")
        renamebare = self._find(by["bothnames"], "renamebare")
        assert (
            use_lines(renameonly) == [
                "use kinds, only: rk=>wp",
                "use kinds, only: wp",
            ]
            and use_lines(renamebare) == [
                "use kinds, rk=>wp",
                "use kinds, only: wp",
            ]
            and (kind(renameonly, "a"), kind(renameonly, "b")) == ("4", "4")
            and (kind(renamebare, "a"), kind(renamebare, "b")) == ("4", "4")
        )

    def test_entry_name_gets_its_own_type(self, tmp_path):
        by = self._crack(tmp_path, "entry.f90", """\
            double precision function eta0(tk)
              implicit double precision (a-h,o-z)
              eta0 = tk
              return
            entry eta1(tk, kk)
              eta1 = tk * kk
              return
            end
            """)
        eta = by["eta0"]["vars"]
        assert eta["eta0"]["typespec"] == "double precision"
        assert eta["eta1"]["typespec"] == "double precision"
        assert eta["tk"]["typespec"] == "double precision"
        assert eta["kk"]["typespec"] == "integer"
