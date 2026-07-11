import os
import platform

import pytest

import numpy as np
import numpy.testing as npt

from . import util


@pytest.mark.slow
class TestIntentInOut(util.F2PyTest):
    # Check that intent(in out) translates as intent(inout)
    sources = [util.getpath("tests", "src", "regression", "inout.f90")]

    def test_inout(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float32)[::2]
        pytest.raises(ValueError, self.module.foo, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float32)
        self.module.foo(x)
        assert np.allclose(x, [3, 1, 2])


@pytest.mark.slow
class TestDataOnlyMultiModule(util.F2PyTest):
    # Check that modules without subroutines work
    sources = [util.getpath("tests", "src", "regression", "datonly.f90")]

    def test_mdat(self):
        assert self.module.datonly.max_value == 100
        assert self.module.dat.max_ == 1009
        int_in = 5
        assert self.module.simple_subroutine(5) == 1014


@pytest.mark.slow
class TestModuleWithDerivedType(util.F2PyTest):
    # Check that modules with derived types work
    sources = [util.getpath("tests", "src", "regression", "mod_derived_types.f90")]

    def test_mtypes(self):
        assert self.module.no_type_subroutine(10) == 110
        assert self.module.type_subroutine(10) == 210


@pytest.mark.slow
class TestNegativeBounds(util.F2PyTest):
    # Check that negative bounds work correctly
    sources = [util.getpath("tests", "src", "negative_bounds", "issue_20853.f90")]

    def test_negbound(self):
        xvec = np.arange(12)
        xlow = -6
        xhigh = 4

        # Calculate the upper bound,
        # Keeping the 1 index in mind

        def ubound(xl, xh):
            return xh - xl + 1
        rval = self.module.foo(is_=xlow, ie_=xhigh,
                        arr=xvec[:ubound(xlow, xhigh)])
        expval = np.arange(11, dtype=np.float32)
        assert np.allclose(rval, expval)


@pytest.mark.slow
class TestNumpyVersionAttribute(util.F2PyTest):
    # Check that th attribute __f2py_numpy_version__ is present
    # in the compiled module and that has the value np.__version__.
    sources = [util.getpath("tests", "src", "regression", "inout.f90")]

    def test_numpy_version_attribute(self):

        # Check that self.module has an attribute named "__f2py_numpy_version__"
        assert hasattr(self.module, "__f2py_numpy_version__")

        # Check that the attribute __f2py_numpy_version__ is a string
        assert isinstance(self.module.__f2py_numpy_version__, str)

        # Check that __f2py_numpy_version__ has the value numpy.__version__
        assert np.__version__ == self.module.__f2py_numpy_version__


def test_include_path():
    incdir = np.f2py.get_include()
    fnames_in_dir = os.listdir(incdir)
    for fname in ("fortranobject.c", "fortranobject.h"):
        assert fname in fnames_in_dir


@pytest.mark.slow
class TestIncludeFiles(util.F2PyTest):
    sources = [util.getpath("tests", "src", "regression", "incfile.f90")]
    options = [f"-I{util.getpath('tests', 'src', 'regression')}",
               f"--include-paths {util.getpath('tests', 'src', 'regression')}"]

    def test_gh25344(self):
        exp = 7.0
        res = self.module.add(3.0, 4.0)
        assert exp == res

@pytest.mark.slow
class TestF77Comments(util.F2PyTest):
    # Check that comments are stripped from F77 continuation lines
    sources = [util.getpath("tests", "src", "regression", "f77comments.f")]

    def test_gh26148(self):
        x1 = np.array(3, dtype=np.int32)
        x2 = np.array(5, dtype=np.int32)
        res = self.module.testsub(x1, x2)
        assert res[0] == 8
        assert res[1] == 15

    def test_gh26466(self):
        # Check that comments after PARAMETER directions are stripped
        expected = np.arange(1, 11, dtype=np.float32) * 2
        res = self.module.testsub2()
        npt.assert_allclose(expected, res)

@pytest.mark.slow
class TestF90Continuation(util.F2PyTest):
    # Check that comments are stripped from F90 continuation lines
    sources = [util.getpath("tests", "src", "regression", "f90continuation.f90")]

    def test_gh26148b(self):
        x1 = np.array(3, dtype=np.int32)
        x2 = np.array(5, dtype=np.int32)
        res = self.module.testsub(x1, x2)
        assert res[0] == 8
        assert res[1] == 15

@pytest.mark.slow
class TestLowerF2PYDirectives(util.F2PyTest):
    # Check variables are cased correctly
    sources = [util.getpath("tests", "src", "regression", "lower_f2py_fortran.f90")]

    def test_gh28014(self):
        self.module.inquire_next(3)
        assert True

@pytest.mark.slow
def test_gh26623():
    # Including libraries with . should not generate an incorrect meson.build
    try:
        aa = util.build_module(
            [util.getpath("tests", "src", "regression", "f90continuation.f90")],
            ["-lfoo.bar"],
            module_name="Blah",
        )
    except RuntimeError as rerr:
        assert "lparen got assign" not in str(rerr)


@pytest.mark.slow
@pytest.mark.skipif(platform.system() == "Windows", reason='Unsupported on this platform for now')
def test_gh25784():
    # Compile dubious file using passed flags
    try:
        aa = util.build_module(
            [util.getpath("tests", "src", "regression", "f77fixedform.f95")],
            options=[
                # Meson will collect and dedup these to pass to fortran_args:
                "--f77flags='-ffixed-form -O2'",
                "--f90flags=\"-ffixed-form -g\"",
            ],
            module_name="Blah",
        )
    except ImportError as rerr:
        assert "unknown_subroutine_" in str(rerr)


@pytest.mark.slow
class TestComplexStructCompat(util.F2PyTest):
    # Check that .r/.i field access works on complex_double pointers in
    # callstatements (scipy compatibility, gh-30966 follow-up)
    sources = [
        util.getpath("tests", "src", "regression", "complex_struct_compat.pyf"),
        util.getpath("tests", "src", "regression", "complex_struct_compat.f90"),
    ]
    module_name = "_complex_struct_compat_test"

    def test_complex_struct_field_access(self):
        c = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        self.module.zero_imag(c)
        npt.assert_array_equal(c.imag, [0.0, 0.0, 0.0])
        npt.assert_array_equal(c.real, [1.0, 3.0, 5.0])


@pytest.mark.slow
class TestAssignmentOnlyModules(util.F2PyTest):
    # Ensure that variables are exposed without functions or subroutines in a module
    sources = [util.getpath("tests", "src", "regression", "assignOnlyModule.f90")]

    def test_gh27167(self):
        assert (self.module.f_globals.n_max == 16)
        assert (self.module.f_globals.i_max == 18)
        assert (self.module.f_globals.j_max == 72)


def test_gh22511_parameter_attr_parse():
    # Pure-Python: crackfortran must mark module PARAMETER attrs (gh-22511).
    from numpy.f2py import crackfortran
    from numpy.f2py.auxfuncs import isparameter

    fpath = util.getpath("tests", "src", "regression", "gh22511.f90")
    mod = crackfortran.crackfortran([str(fpath)])
    assert len(mod) == 1
    vars_ = mod[0]["vars"]
    assert isparameter(vars_["my_const"])
    assert isparameter(vars_["my_real"])
    assert not isparameter(vars_["mutable_var"])


def test_gh22511_parameter_copy_codegen(tmp_path):
    # Codegen inspection only (no compile): PARAMETER values must be copied
    # into static module-owned storage, not aliased by address (gh-22511).
    import shutil

    from numpy.f2py.f2py2e import run_main

    src = util.getpath("tests", "src", "regression", "gh22511.f90")
    work = tmp_path / "gh22511"
    work.mkdir()
    f90 = work / "gh22511.f90"
    shutil.copy(src, f90)

    with util.switchdir(work):
        run_main(["-m", "gh22511_test", str(f90.name)])

    c_path = work / "gh22511_testmodule.c"
    assert c_path.is_file()
    text = c_path.read_text(encoding="utf-8")

    # PARAMETER scalars: static buffer + memcpy, then .data points at the copy.
    assert "static char f2py_gh22511_mod_my_const_data[sizeof(int)];" in text
    assert "memcpy(f2py_gh22511_mod_my_const_data, my_const, sizeof(int));" in text
    assert (
        "f2py_gh22511_mod_def[i_f2py++].data = f2py_gh22511_mod_my_const_data;"
        in text
    )
    assert "static char f2py_gh22511_mod_my_real_data[sizeof(float)];" in text
    assert "memcpy(f2py_gh22511_mod_my_real_data, my_real, sizeof(float));" in text

    # Must not alias the Fortran PARAMETER address directly.
    assert "f2py_gh22511_mod_def[i_f2py++].data = my_const;" not in text
    assert "f2py_gh22511_mod_def[i_f2py++].data = my_real;" not in text

    # Mutable module variables keep direct pointer wrapping.
    assert "f2py_gh22511_mod_def[i_f2py++].data = mutable_var;" in text


@pytest.mark.slow
class TestParameterConstants(util.F2PyTest):
    # gh-22511: PARAMETER constants must survive init (value copy, not alias).
    sources = [util.getpath("tests", "src", "regression", "gh22511.f90")]

    def test_parameter_constant_values(self):
        mod = self.module.gh22511_mod
        assert int(mod.my_const) == 1234
        assert abs(float(mod.my_real) - 3.14) < 0.01
        assert int(mod.mutable_var) == 42


@pytest.mark.slow
class TestBoolLogicalNoCopy(util.F2PyTest):
    # gh-10117: bool arrays pass to logical(kind=1) dummies without a copy
    sources = [util.getpath("tests", "src", "regression", "gh10117.f90")]

    def test_bool_inout_no_copy(self):
        # mutation through intent(inout) is only visible without a copy
        m = np.array([True, False, True])
        self.module.bool_flip_first(m, m.size)
        npt.assert_array_equal(m, [False, False, True])

    def test_bool_intent_in(self):
        m = np.array([True, False, True, True])
        assert self.module.bool_count_true(m, m.size) == 3

    def test_int8_still_works(self):
        b = np.array([1, 0, 1], dtype=np.int8)
        self.module.bool_flip_first(b, b.size)
        npt.assert_array_equal(b, [0, 0, 1])


@pytest.mark.slow
class TestPickleFortranFunction(util.F2PyTest):
    # gh-21767: function wrappers pickle by reference; data-carrying
    # fortran objects refuse with a clear message
    sources = [util.getpath("tests", "src", "regression", "gh21767.f90")]

    def test_pickle_roundtrip(self):
        import pickle

        fn = self.module.double_it
        restored = pickle.loads(pickle.dumps(fn))
        assert restored is fn
        assert restored(21.0) == 42.0

    def test_deepcopy(self):
        import copy

        fn = self.module.double_it
        assert copy.deepcopy(fn) is fn


@pytest.mark.slow
class TestOptionalPresent(util.F2PyTest):
    # gh-4013: omitted source-optional args forward as absent, so
    # present() branches correctly instead of reading zeros
    sources = [util.getpath("tests", "src", "regression", "gh4013.f90")]

    def test_present_branch(self):
        assert self.module.gh4013.foo(20.0) == 20.0

    def test_absent_branch(self):
        assert self.module.gh4013.foo() == 1.0
