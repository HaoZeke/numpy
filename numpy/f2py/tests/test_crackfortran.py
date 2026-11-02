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


class TestMarkoutercomma:
    def test_brackets(self):
        # an array constructor in brackets is a single item
        assert (crackfortran.markoutercomma("a = [1, 2], b(2)")
                == "a = [1, 2]@,@ b(2)")


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

    def test_param_eval_reshape(self):
        v = 'reshape((/ (i, i=1, 250) /), (/5, 10, 5/))'
        g_params = {"kind": crackfortran._kind_func,
                "selected_int_kind": crackfortran._selected_int_kind_func,
                "selected_real_kind": crackfortran._selected_real_kind_func}
        params = {}
        dimspec = '(0:4, 3:12, 5)'
        ret = crackfortran.param_eval(v, g_params, params, dimspec=dimspec)
        # the first index varies fastest
        assert len(ret) == 250
        assert ret[0, 3, 1] == 1
        assert ret[1, 3, 1] == 2
        assert ret[0, 4, 1] == 6
        assert ret[4, 12, 5] == 250


class TestUseParameters:
    # parameter arrays keep their bounds in the modules that use them
    def test_array_bounds(self, tmp_path):
        fpath = tmp_path / "uses.pyf"
        fpath.write_text(textwrap.dedent("""\
            python module uses
              module base
                integer, parameter, dimension(2) :: fixed = (/5, 6/)
                integer, parameter, dimension(-1:1) :: offset = (/7, 8, 9/)
                integer, parameter, dimension(*) :: sizes = (/2, 3/)
              end module base
              module user
                use base
                real, dimension(fixed(1)) :: buf
                real, dimension(offset(1)) :: low
                integer, parameter, dimension(*) :: both = [sizes, fixed]
              end module user
            end python module uses
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["body"][1]["vars"]
        assert vs["buf"]["dimension"] == ["5"]
        assert vs["low"]["dimension"] == ["9"]
        assert vs["both"]["dimension"] == ["4"]


class TestParameterSizes:
    # implied shape and assumed length parameters take their size from
    # their value
    def test_sizes(self, tmp_path):
        fpath = tmp_path / "sizes.f90"
        fpath.write_text(textwrap.dedent("""\
            module sizes
              integer :: i
              integer, parameter :: n_max = 2
              integer, parameter :: ints(*) = [n_max, 2 * n_max, 8]
              real(8), parameter :: reals(0:*) = (/1.5d0, 2.5d0/)
              integer, parameter :: unknown(*) = (/huge(i)/)
              character(len=*), parameter :: phrase = 'don''t' // '!'
              character(len=*), parameter :: names(*) = ['ab', 'cd']
            end module sizes
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["ints"]["dimension"] == ["3"]
        assert vs["reals"]["dimension"] == ["2"]
        assert vs["unknown"]["dimension"] == ["*"]
        assert vs["phrase"]["charselector"]["len"] == "6"
        assert vs["names"]["dimension"] == ["2"]
        assert vs["names"]["charselector"]["len"] == "2"


class TestRealParameters:
    # a real literal makes a value a kind only if it cannot be evaluated
    def test_extents(self, tmp_path):
        fpath = tmp_path / "reals.f90"
        fpath.write_text(textwrap.dedent("""\
            module reals
              integer, parameter :: dp = kind(1.0d0)
              integer, parameter :: wp = selected_real_kind(precision(1.0d0))
              real, parameter :: ratio = 1.5
              real(dp), parameter :: half = 0.5_dp
              integer, parameter :: n = int(ratio * 2), m = int(4 * half)
              integer, parameter :: k = int(2.5)
              real :: kinds(dp, wp), buf(n), low(m), two(k)
            end module reals
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["kinds"]["dimension"] == ["8", "8"]
        assert vs["buf"]["dimension"] == ["3"]
        assert vs["low"]["dimension"] == ["2"]
        assert vs["two"]["dimension"] == ["2"]

    def test_not_kinds(self):
        # and only if the value is a kind
        par = {'attrspec': ['parameter']}
        vars = {'big': {**par, 'typespec': 'real', '=': 'huge(1.0)'},
                'z': {**par, 'typespec': 'complex', '=': 'cmplx(1.0, 2.0)'},
                'n': {**par, 'typespec': 'integer', '=': 'int(big / 1e38)'}}
        assert crackfortran.get_parameters(vars) == {
            'big': 'huge(1.0)', 'z': 'cmplx(1.0, 2.0)', 'n': 'int(big / 1e38)'}


class TestLiteralKinds:
    # kinds are dropped from literals, not from names
    def test_extents(self, tmp_path):
        fpath = tmp_path / "kinds.f90"
        fpath.write_text(textwrap.dedent("""\
            module kinds
              integer, parameter :: i8 = selected_int_kind(18)
              integer, parameter :: dp = kind(1.0d0)
              integer, parameter :: n_max = 2, n = n_max + 1
              integer(i8), parameter :: k = 2_i8 + 3_8
              real(dp), parameter :: x = 1.5_dp
              integer, parameter :: m = int(x * 2.0_dp)
              real :: buf(n), big(k), three(m)
            end module kinds
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["buf"]["dimension"] == ["3"]
        assert vs["big"]["dimension"] == ["5"]
        assert vs["three"]["dimension"] == ["3"]

    def test_logical(self):
        var = {'typespec': 'logical', 'kindselector': {'kind': '1'},
               'attrspec': ['parameter'], '=': '.true._1'}
        assert crackfortran.get_parameters({'yes': var}) == {'yes': True}


class TestIntegerDivision:
    # the quotient of two integers is truncated, as in Fortran
    def test_extents(self, tmp_path):
        fpath = tmp_path / "division.f90"
        fpath.write_text(textwrap.dedent("""\
            module division
              integer, parameter :: n = 7, half = n / 2, neg = -n / 2
              real, parameter :: ratio = n / 2.0
              integer, parameter :: twice = int(ratio * 2)
              real :: buf(half), low(neg + 5), every(twice)
              character(len=n / 2) :: word
            end module division
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["buf"]["dimension"] == ["3"]
        assert vs["low"]["dimension"] == ["2"]
        assert vs["every"]["dimension"] == ["7"]
        assert vs["word"]["charselector"]["len"] == "3"

    def test_negative_powers(self, tmp_path):
        # 2**(-2) is 1 / 4
        fpath = tmp_path / "powers.f90"
        fpath.write_text(textwrap.dedent("""\
            module powers
              integer, parameter :: zero = 2**(-2), minus = (-1)**(-3)
              real, parameter :: quarter = 2.0**(-2)
              real :: buf(zero + 1), odd(minus + 2), four(int(1 / quarter))
            end module powers
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["buf"]["dimension"] == ["1"]
        assert vs["odd"]["dimension"] == ["1"]
        assert vs["four"]["dimension"] == ["4"]


class TestArrayElements:
    # an element of a parameter array can define another parameter
    def test_extents(self, tmp_path):
        fpath = tmp_path / "elements.f90"
        fpath.write_text(textwrap.dedent("""\
            module elements
              integer, parameter :: dims(2) = [4, 5]
              integer, parameter :: nx = dims(2), total = dims(1) * dims(2)
              real :: buf(nx), grid(total)
            end module elements
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["buf"]["dimension"] == ["5"]
        assert vs["grid"]["dimension"] == ["20"]


class TestIntrinsics:
    # intrinsics follow Fortran, so nint rounds halves away from zero
    @pytest.mark.parametrize("value, expected", [
        ("nint(2.5)", 3), ("nint(-2.5)", -3),
        ("nint(0.49999999999999994)", 0), ("nint(a=7.5, kind=8)", 8),
        ("int(-2.7)", -2), ("floor(-1.5)", -2), ("ceiling(1.2)", 2),
        ("mod(-7, 3)", -1), ("modulo(-7, 3)", 2), ("sign(4, -1)", -4),
        ("merge(2, 9, 1 > 0)", 2), ("iachar('B')", 66),
    ])
    def test_integer(self, value, expected):
        var = {'typespec': 'integer', 'attrspec': ['parameter'], '=': value}
        assert crackfortran.get_parameters({'n': var}) == {'n': expected}

    @pytest.mark.parametrize("value, expected", [
        ("achar(65)", "A"), ("char(66) // 'c'", "Bc"),
        ("repeat('-', 3)", "---"), ("trim('ab  ')", "ab"),
    ])
    def test_character(self, value, expected):
        var = {'typespec': 'character', 'charselector': {'len': '*'},
               'attrspec': ['parameter'], '=': value}
        assert crackfortran.get_parameters({'s': var}) == {'s': expected}

    def test_extents(self, tmp_path):
        fpath = tmp_path / "intrinsics.f90"
        fpath.write_text(textwrap.dedent("""\
            module intrinsics
              character(len=*), parameter :: names(*) = ['ab', 'cd', 'ef']
              character(len=*), parameter :: letter = achar(65)
              integer, parameter :: k = nint(2.5), n = size(names)
              real :: buf(k), each(n)
            end module intrinsics
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["letter"]["charselector"]["len"] == "1"
        assert vs["buf"]["dimension"] == ["3"]
        assert vs["each"]["dimension"] == ["3"]

    def test_array_arguments(self, tmp_path):
        # these see the elements of an array, not their indices
        fpath = tmp_path / "reductions.f90"
        fpath.write_text(textwrap.dedent("""\
            module reductions
              integer, parameter :: dims(2) = [4, 5]
              character(len=3), parameter :: names(2) = ['ab', 'cd']
              logical, parameter :: flags(2) = [.false., .false.]
              integer, parameter :: total = sum(dims), width = len(names)
              integer, parameter :: area = product(dims)
              integer, parameter :: hi = maxval(dims), lo = minval(dims)
              real :: buf(total), each(width)
              real :: one(merge(1, 2, any(flags))), two(merge(1, 2, all(flags)))
              character(len=len(names)) :: word
            end module reductions
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["total"]["="] == "9"
        assert [vs[n]["="] for n in ("area", "hi", "lo")] == ["20", "5", "4"]
        assert vs["buf"]["dimension"] == ["9"]
        assert vs["each"]["dimension"] == ["3"]
        assert vs["one"]["dimension"] == ["2"]
        assert vs["two"]["dimension"] == ["2"]
        assert vs["word"]["charselector"]["len"] == "3"


class TestConstantDimensions:
    # a dimension that is a constant expression is evaluated as in Fortran
    def test_extents(self, tmp_path):
        fpath = tmp_path / "dimensions.f90"
        fpath.write_text(textwrap.dedent("""\
            module dimensions
              integer, parameter :: n = 7, neg = -3
              integer, parameter :: dims(2) = [4, 5]
              real, parameter :: scale = 2.5
              integer, parameter :: odd(n / 2) = [1, 3, 5]
              real :: even(2 * (n / 2)), pos(-neg), more(dims(1) + 1)
              real :: buf(nint(scale)), third(odd(3))
            end module dimensions
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["even"]["dimension"] == ["6"]
        assert vs["pos"]["dimension"] == ["3"]
        assert vs["more"]["dimension"] == ["5"]
        assert vs["buf"]["dimension"] == ["3"]
        assert vs["odd"]["dimension"] == ["3"]
        assert vs["third"]["dimension"] == ["5"]


class TestImpliedDo:
    @pytest.mark.parametrize("v, expected", [
        ("(/(i, i = 1, n)/)", [1, 2, 3]),
        ("[(2 * i, i = 5, 1, -2)]", [10, 6, 2]),
        ("[(i, -i, i = 1, 2)]", [1, -1, 2, -2]),
        ("[((i * j, i = 1, 2), j = 2, 3)]", [2, 4, 3, 6]),
        ("[0, ([i, 10 * i], i = 1, 2), 9]", [0, 1, 10, 2, 20, 9]),
        ("[(i, i = 1, 0)]", []),
    ])
    def test_values(self, v, expected):
        p = crackfortran.param_eval(v, {}, {'n': 3}, dimspec='(*)')
        assert p == dict(enumerate(expected, start=1))

    def test_extents(self, tmp_path):
        fpath = tmp_path / "implied_do.f90"
        fpath.write_text(textwrap.dedent("""\
            module implied_do
              integer :: i
              integer, parameter :: seq(*) = (/(i, i = 1, 4)/)
              real :: buf(seq(4) - 1)
            end module implied_do
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["seq"]["dimension"] == ["4"]
        assert vs["buf"]["dimension"] == ["3"]


class TestReshape:
    # elements are in array element order, the first index varying fastest
    @pytest.mark.parametrize("v, expected", [
        ("reshape([1, 2, 3, 4], [2, 2])", [1, 2, 3, 4]),
        ("reshape([1, 2, 3, 4], [2, 2], order=[2, 1])", [1, 3, 2, 4]),
        ("reshape(shape=[2, 2], source=[1, 2, 3], pad=[0])", [1, 2, 3, 0]),
        ("0", [0, 0, 0, 0]),
    ])
    def test_values(self, v, expected):
        p = crackfortran.param_eval(v, {}, {}, dimspec='(2, 2)')
        assert list(p) == [(1, 1), (2, 1), (1, 2), (2, 2)]
        assert list(p.values()) == expected

    def test_extents(self, tmp_path):
        fpath = tmp_path / "rank2.f90"
        fpath.write_text(textwrap.dedent("""\
            module rank2
              integer, parameter :: m(*, *) = reshape([1, 2, 3, 4, 5, 6], [2, 3])
              character(len=*), parameter :: s(*, *) = reshape(['ab', 'cd'], [1, 2])
              integer, parameter :: rows = size(m, 1)
              real :: second(m(2, 1)), grid(rows, size(m, 2))
            end module rank2
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["m"]["dimension"] == ["2", "3"]
        assert vs["s"]["dimension"] == ["1", "2"]
        assert vs["s"]["charselector"]["len"] == "2"
        assert vs["second"]["dimension"] == ["2"]
        assert vs["grid"]["dimension"] == ["2", "3"]


class TestArrayArithmetic:
    # Python would repeat the list, where Fortran doubles each element
    def test_unevaluated(self, tmp_path):
        fpath = tmp_path / "arithmetic.f90"
        fpath.write_text(textwrap.dedent("""\
            module arithmetic
              integer, parameter :: twice(*) = 2 * [1, 2, 3]
              complex, parameter :: z = (1.0, 2.0) * 2
              real :: buf(size(2 * [1, 2, 3]))
            end module arithmetic
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["twice"]["dimension"] == ["*"]
        assert vs["z"]["="] == "(1.0, 2.0)*2"
        assert vs["buf"]["dimension"] == ["size(2 * [1, 2, 3])"]


class TestCharacterLengths:
    # a character parameter is blank padded or cut to its length
    def test_lengths(self, tmp_path):
        fpath = tmp_path / "lengths.f90"
        fpath.write_text(textwrap.dedent("""\
            module lengths
              character(len=5), parameter :: padded = 'ab'
              character(len=2), parameter :: cut = 'abcd'
              character(len=*), parameter :: both = padded // cut // '!'
              integer, parameter :: n = len(padded)
              real :: buf(n)
            end module lengths
            """))
        vs = crackfortran.crackfortran([str(fpath)])[0]["vars"]
        assert vs["both"]["charselector"]["len"] == "8"
        assert vs["buf"]["dimension"] == ["5"]


class TestGetStrLength:
    def test_stringarray(self, capsys):
        # get_elsize asks for the length of string arrays as well
        var = {'typespec': 'character', 'charselector': {'len': '2'},
               'dimension': ['3']}
        assert capi_maps.getstrlength(var) == '2'
        assert capsys.readouterr().err == ''


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
