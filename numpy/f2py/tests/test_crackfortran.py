import contextlib
import importlib
import io
import os
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

def _flat_blocks(nodes, acc=None):
    if acc is None:
        acc = []
    if isinstance(nodes, dict):
        nodes = [nodes]
    for block in nodes or []:
        if not isinstance(block, dict):
            continue
        acc.append(block)
        _flat_blocks(block.get("body"), acc)
    return acc
def _callback_parts(blocks, module_name, callback_name):
    module = next(block for block in blocks if block.get("name") == module_name)
    for routine in _flat_blocks(module):
        if (routine.get("name") == callback_name
                and routine.get("block") in ("function", "subroutine")):
            arg = routine["args"][0]
            result = routine["vars"].get(routine.get("result") or routine["name"], {})
            return (routine["block"], routine["vars"][arg]["typespec"],
                    result.get("typespec"), routine.get("from"))
    raise AssertionError(module_name)
def _crack_sources(tmp_path, files, module=""):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, body in files:
        text = textwrap.dedent(body).lstrip("\n")
        if not text.endswith("\n"):
            text += "\n"
        (tmp_path / name).write_text(text)
    cwd = os.getcwd()
    saved = crackfortran.f77modulename
    os.chdir(tmp_path)
    crackfortran.f77modulename = module
    try:
        return crackfortran.crackfortran([name for name, _body in files])
    finally:
        crackfortran.f77modulename = saved
        os.chdir(cwd)
def _real(name):
    return (f"function {name}(cb) result(y)\nreal y, x\nexternal cb\n"
            "y = cb(x)\nend\n")
def _sub(name):
    return (f"subroutine {name}(cb)\ninteger n\nexternal cb\n"
            "call cb(n)\nend\n")
def _mod(name, body, use=""):
    return f"module {name}\n{use}contains\n{body}end\n"

class TestPassedExternal:
    def test_missing_interface_is_left_unset(self, tmp_path):
        src = (
            "function plain(a) result(r); implicit none; real(8) :: a, r; "
            "r = a; end\nfunction forward_missing(cb) result(r); "
            "implicit none; real(8), external :: cb, plain; real(8) :: r; "
            "character(len=20) :: message; message = 'plain(cb)'; "
            "r = plain(1.0d0); if (message == 'plain(cb)') r = r; end\n")
        blocks = _crack_sources(tmp_path, [("a.f90", src)])
        names = [block.get("name") for block in blocks]
        assert "forward_missing__user__routines" not in names
        missing = next(block for block in _flat_blocks(blocks)
                       if block.get("name") == "forward_missing")
        assert "cb" in missing["externals"]

    def test_comment_does_not_copy_callback(self, tmp_path):
        src = (
            "function evil(cb) result(r)\nimplicit none\n"
            "integer, external :: cb\ninteger :: r\nr = cb(3)\nend\n"
            "function victim(cb) result(r)\nimplicit none\n"
            "real(8), external :: cb\nreal(8) :: r\n"
            "r = len('c:\\') ! call evil(cb)\nr = 1.0d0\nend\n"
            "function keeps_own(cb) result(r)\nimplicit none\n"
            "real(8), external :: cb\nreal(8) :: r, x\n"
            "x = 1.0d0\nr = cb(x) ! call evil(cb)\nend\n")
        blocks = _crack_sources(tmp_path, [("a.f90", src)])
        names = [block.get("name") for block in blocks]
        assert "victim__user__routines" not in names
        evil = _callback_parts(blocks, "evil__user__routines", "cb")[1]
        own = _callback_parts(blocks, "keeps_own__user__routines", "cb")[1]
        assert (evil, own) == ("integer", "real")

    def test_scope_and_order(self, tmp_path):
        run = "subroutine run(fun)\nexternal fun\ncall apply(fun)\nend\n"
        real_apply = "subroutine apply(cb)\nreal x\nexternal cb\nx = cb(x)\nend\n"
        contained = _sub("apply") + _mod(
            "m", "subroutine outer(fun)\nexternal fun\ncall apply(fun)\n"
            "contains\n" + _real("apply") + "end\n")
        decoy = _mod("decoy", _real("apply")) + (
            "subroutine parent(fun)\nexternal fun\ncall apply(fun)\nend\n")
        mods = _mod("m2", real_apply) + _mod("m1", run + _sub("apply"))
        iface = (
            "subroutine run(fun)\nexternal fun\ninterface\n"
            "subroutine fun(n)\ninteger n\nend\nend interface\nend\n")
        si, fr = ("subroutine", "integer"), ("function", "real")
        jobs = (
            ("contained", [("a.f90", contained)],
             (("outer__user__routines", fr),)),
            ("decoy", [("caller.f90", decoy), ("impl.f90", _sub("apply"))],
             (("parent__user__routines", si),)),
            ("mods", [("mods.f90", mods)], (("run__user__routines", si),)),
            ("fwd", [("a.f90", run + _sub("apply")), ("b.f90", run + real_apply)],
             (("run__user__routines", si), ("run__user__routines_1", fr))),
            ("ord", [("a.f90", run + real_apply), ("b.f90", iface)],
             (("run__user__routines", si), ("run__user__routines_1", fr))),
        )
        for tag, files, expects in jobs:
            blocks = _crack_sources(tmp_path / tag, files)
            for mod, sig in expects:
                got = _callback_parts(blocks, mod, "fun")
                assert got[:2] == sig, (tag, got)

    def test_keyword_uses_interface(self, tmp_path):
        src = (
            "subroutine apply(cb1, cb2); integer n; real x; external cb1, cb2; "
            "call cb1(n); x = cb2(x); end\n"
            "subroutine parent(fun, ident); external fun, ident; interface\n"
            "subroutine apply(cb2, cb1); external cb2, cb1; end; end interface\n"
            "call apply(cb1=fun, cb2=ident); end\n")
        blocks = _crack_sources(tmp_path, [("kw.f90", src)])
        fun = _callback_parts(blocks, "parent__user__routines", "fun")
        ident = _callback_parts(blocks, "parent__user__routines", "ident")
        assert fun[:2] == ("function", "real")
        assert ident[:2] == ("subroutine", "integer")

    def test_rename_keeps_dummy_type(self, tmp_path):
        src = (
            "function apply(cb, fun); double precision apply; integer fun; "
            "external cb; apply = cb(fun); end\n"
            "function parent(fun); double precision parent; external fun; "
            "parent = apply(fun); end\n")
        blocks = _crack_sources(tmp_path, [("a.f90", src)])
        parent = _callback_parts(blocks, "parent__user__routines", "fun")
        apply = _callback_parts(blocks, "apply__user__routines", "cb")
        assert parent[:3] == ("function", "integer", "double precision")
        assert apply[1] == "integer"

    def test_renamed_dummy_rewrites_dimension(self, tmp_path):
        src = (
            "subroutine apply(cb, n); integer n; real p(n); character(len=n) :: s; "
            "character*(n) u(2); character*8 :: t = 'n'; real x(a % n); real z(a% n); "
            "real w(a %n); real c(a%n); real y(n$); external cb\n!f2py check(m % n) p\n"
            "!f2py check(s=='n') s\ncall cb(n, p, s, u, t, x, z, w, c, y); end\n"
            "subroutine parent(n); external n; call apply(n, m); end\n"
            "subroutine applyi(cb, in); integer, intent(in) :: in, m; external cb; "
            "call cb(in, m); end\nsubroutine parenti(in); external in; "
            "call applyi(in, m); end\nsubroutine bapply(cb, size); integer size\n"
            "real p(size); real q(2 * size); real x(a % size); external cb\n"
            "!f2py check(size>0) size\ncall cb(size, p, q, x); end\n"
            "subroutine bparent(size); external size; call bapply(size, m); end\n")
        flat = _flat_blocks(_crack_sources(tmp_path, [("a.f90", src)]))
        v = next(b["vars"] for b in flat if b.get("name") == "n")
        sv = next(b["vars"] for b in flat if b.get("name") == "size_bn")
        mvar = next(b["vars"]["m"] for b in flat if b.get("name") == "in")
        assert (v["p"]["dimension"], v["x"]["dimension"], v["z"]["dimension"],
                v["w"]["dimension"], v["c"]["dimension"], v["y"]["dimension"],
                v["s"]["charselector"].get("len"), v["u"]["charselector"].get("*"),
                v["t"].get("="), v["s"].get("check"), v["p"].get("check"),
                mvar["intent"], sv["p"]["dimension"], sv["q"]["dimension"],
                sv["x"]["dimension"], sv["size_bn_1"].get("check")) == (
            ["n_1"], ["a % n"], ["a% n"], ["a %n"], ["a%n"], ["n$"],
            "n_1", "(n_1)", "'n'", ["s=='n'"], ["m % n"], ["in"],
            ["size_bn_1"], ["2 * size_bn_1"], ["a % size"], ["size_bn_1>0"])

    def test_association_order(self, tmp_path):
        A, O, S, B = (_sub(n) for n in ("apply", "other", "size", "bee"))
        a, o, x, c = (_real(n) for n in ("apply", "other", "x", "cee"))
        M = _mod
        P = lambda use="", call="apply", ext="external fun\n": (
            f"subroutine parent(fun)\n{use}{ext}call {call}(fun)\nend\n")
        IF = ("interface\nfunction apply(cb) result(y)\ninterface\n"
              "function cb(x) result(z)\nreal z, x\nend\nend interface\n"
              "real y\nend\nend interface\n")
        LS = ("interface\nsubroutine apply(cb)\ninterface\n"
              "subroutine cb(n)\ninteger n\nend\nend interface\n"
              "end\nend interface\n")
        LP = "private\n" + IF + "public :: apply\n"
        LSp = LS + "private :: apply\n"
        IFACE = ("subroutine parent(fun)\nexternal fun\ninterface\n"
                 "function apply(cb) result(y)\nreal y\nexternal cb\nend\n"
                 "end interface\ncall apply(fun)\nend\n")
        DIR = (
            "subroutine parent(fun)\nuse m\nexternal fun\ninteger n\n"
            "call fun(n)\ncall apply(fun)\nend\n")
        si, fr = ("subroutine", "integer"), ("function", "real")

        def check(tag, source, sig, origin="", lower=1):
            saved = crackfortran.dolowercase
            crackfortran.dolowercase = lower
            try:
                blocks = _crack_sources(tmp_path / tag, [("a.f90", source)])
            finally:
                crackfortran.dolowercase = saved
            if sig is None:
                names = [b.get("name") or "" for b in _flat_blocks(blocks)]
                ended = "parent__user__routines"
                assert not any(n.endswith(ended) for n in names), tag
                return
            got = _callback_parts(blocks, "parent__user__routines", "fun")
            assert got[:2] == sig, (tag, got)
            if origin:
                assert origin in (got[3] or ""), (tag, got)

        check("ext", A + M("m", a + P(ext="external fun, apply\n")), si)
        check("iface", a + M("m", A + IFACE), fr)
        check("none", M("m", A + P(ext="external fun, apply\n")), None)
        check("host", M("other", a) + M("m", A + P("use other\n")), fr)
        check("kept", M("src", a + O) + M("host", P("use src, only: other\n"),
            "use src\n"), fr)
        check("ren", M("m", A + o) + P("use m, only: apply => other\n"), fr)
        check("bare", M("m", o) + P("use m, apply => other\n"), fr)
        check("loc", M("m", a) + P("use m, only: local => apply\n", "local"), fr)
        check("amb", M("m", A + o) + P("use m, apply => other\n"), None)
        check("omod", M("src", A) + M("othermod", a) + M("host",
            P("use othermod, only: apply\n"), "use src\n"), fr)
        check("oren", M("src", A + o) + M("host", P("use src, only: apply => other\n"),
            "use src\n"), fr)
        check("work", M("other", A) + M("worker", a) + M("host", P("use worker\n"),
            "use other\n"), fr)
        check("hloc", M("m", a) + M("host", P("use m\n", "local"),
            "use m, only: local => apply\n"), fr)
        check("hbare", A + M("m", o) + M("host", P("use m\n"),
            "use m, only: apply => other\n"), fr)
        check("beta", M("alpha", A) + M("beta", o) + M("host",
            P("use beta, only: apply => other\n"), "use alpha\n"), fr)
        check("hidef", O + M("m", o) + P("use m, apply => other\n", "other"), si)
        check("hideh", M("m", o) + M("host", O + P("use m, apply => other\n", "other")),
            si)
        check("bn", M("m", S) + P("use m, only: size\n", "size"), si)
        check("msize", A + M("size", a) + P("use size\n"), fr)
        check("pub", A + M("m", a, "private\npublic :: apply\n") + P("use m\n"), fr)
        check("reonly", A + M("m",
            a + O) + P("use m, only: apply\nuse m, only: other\n"), fr)
        check("rebare", O + M("m", o + A) + P("use m\nuse m, only: apply\n", "other"),
            fr)
        check("onlyo", A + M("m", a + O) + P("use m, only: other\n"), si)
        check("miss", "module m\ninteger :: k\nend\n" + A + P("use m, only: apply\n"),
            None)
        check("unseen", M("host", A + P("use externalmod, only: apply\n")), None)
        check("ifaceu", M("m", "\n", IF) + M("host", A + P("use m, only: apply\n")), fr)
        check("swap", M("m", A + o) + P("use m, apply => other, other => apply\n"), fr)
        check("direct", M("m", a) + A + DIR, si, "parent")
        check("sib", M("m", A + P(), "private :: apply\n"), si)
        check("latepub", A + M("m", "\n", LP) + P("use m\n"), fr)
        check("latepriv", a + M("m", "\n", LSp) + P("use m\n"), fr)
        check("hostif", M("m", P(), LS), si)
        check("caseuse", A + M("size", a) + P("use SIZE\n"), fr, lower=0)
        check("caseonly", A + M("m", a,
            "private\npublic :: APPLY\n") + P("use m, only: APPLY\n"), fr, lower=0)
        both = "use m, only: a => x, b => x\n"
        for call in ("a", "b"):
            check(f"both{call}", M("m", x) + P(both, call), fr)
        check("reamb", M("m",
            B + c) + P("use m, only: apply => bee\nuse m, only: apply => cee\n"), None)
        check("renonly", M("m",
            A + o) + P("use m, apply => other\nuse m, only: apply\n"), None)
        check("reid", M("m",
            o) + P("use m, only: apply => other\nuse m, only: apply => other\n"), fr)
        check("mixuse", A + M("m", a) + P("use m, only: local => apply\nuse M\n"), si,
            "a.f90:apply:unknown_interface", 0)
        check("locif", (
            "subroutine parent(fun)\nexternal fun\ninterface\n"
            "subroutine apply(cb)\ninteger n\nexternal cb\ncall cb(n)\nend\n"
            "end interface\ncall apply(fun)\nend\n"), si)

    @pytest.mark.parametrize("body,expect", [
        ("use kinds, only: wp => dp\nuse kinds\nreal(wp) :: x(:)\n",
         ["use kinds, wp=>dp"]),
        ("use kinds, wp => dp\nuse kinds, only: dp\nreal(dp) :: x(:)\n",
         ["use kinds, wp=>dp,dp=>dp"]),
    ], ids=["rename", "self"])
    def test_bare_use_merges_with_only(self, tmp_path, body, expect):
        src = ("module kinds\ninteger, parameter :: dp = kind(1.0d0)\nend\n"
               + _mod("wrap", f"subroutine parent(x)\n{body}end\n"))
        blocks = _crack_sources(tmp_path, [("a.f90", src)])
        parent = next(b for b in _flat_blocks(blocks) if b.get("name") == "parent")
        text = parent.get("saved_interface") or ""
        uses = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("use ")]
        assert uses == expect
