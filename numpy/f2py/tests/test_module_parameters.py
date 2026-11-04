import copy
import re

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from numpy.f2py import crackfortran, f90mod_rules
from numpy.f2py.f2py2e import run_main

from . import util


ORIGINAL = util.getpath("tests", "src", "modules", "gh22511_original.f90")
STORAGE = util.getpath("tests", "src", "modules", "gh22511_storage.f90")


def test_original_parameter_attribute():
    module, = crackfortran.crackfortran([str(ORIGINAL)])
    assert "parameter" in module["vars"]["my_const"]["attrspec"]
    assert module["vars"]["my_const"]["="] == "1234"


def test_parameter_storage_codegen(tmp_path):
    with util.switchdir(tmp_path):
        run_main(["-m", "parameter_storage", str(STORAGE)])
    generated = (tmp_path / "parameter_storagemodule.c").read_text()
    setup = generated.index("static void f2py_setup_constants(")
    storage = re.search(r"static complex_double f2py_constants_def_\d+_data_?\[", generated)
    assert storage is not None and storage.start() < setup
    body = generated[setup:generated.index("extern void", setup)]
    assert "PyMem_Malloc" not in body
    assert "memcpy(" not in body
    assert "((unsigned char *)" in body
    assert ".data = scalar;" not in body
    assert ".data = mutable;" in body
    assert "NPY_ARRAY_WRITEABLE" in generated
    # A public name matching generated storage cannot hide the static array.
    assert "static int f2py_constants_def_0_data_[" in generated
    assert "unsigned long long f2py_copy_index_ = 0" in body


@pytest.mark.parametrize("field", ["dimension", "character"])
def test_unresolved_parameter_storage_rejected(field):
    module, = crackfortran.crackfortran([str(STORAGE)])
    module = copy.deepcopy(module)
    if field == "dimension":
        module["vars"]["vector"]["dimension"] = ["missing_extent"]
    else:
        module["vars"]["word"]["charselector"] = dict(len="missing_length")
    with pytest.raises(ValueError, match="Cannot determine module PARAMETER storage extent"):
        f90mod_rules.buildhooks(module)


@pytest.mark.slow
class TestOriginalModuleParameter(util.F2PyTest):
    sources = [ORIGINAL]

    def test_original_reproducer(self):
        module = self.module.f2py_test_module
        assert int(module.my_const) == 1234
        assert_array_equal(module.cross([1, 0, 0], [0, 1, 0]), [0, 0, 1])
        assert int(module.my_const) == 1234

    def test_original_constant_is_readonly(self):
        module = self.module.f2py_test_module
        assert not module.my_const.flags.writeable
        with pytest.raises(AttributeError, match="named constant my_const"):
            module.my_const = 0
        assert int(module.my_const) == 1234


@pytest.mark.slow
class TestModuleParameterStorage(util.F2PyTest):
    sources = [STORAGE]

    def test_parameter_values_and_dtypes(self):
        module = self.module.constants
        assert int(module.scalar) == 1234
        assert float(module.wide) == 2.5
        assert complex(module.z) == 1.5 - 2j
        assert bool(module.truth)
        assert int(module.large) == 1234567890123
        assert module.small.dtype == np.dtype("int8")
        assert module.large.dtype == np.dtype("int64")
        assert module.wide.dtype == np.dtype("float64")
        assert module.z.dtype == np.dtype("complex128")
        assert_array_equal(module.small, [1, -2, 3])
        assert_array_equal(module.vector, [4, 6, 8])
        assert_array_equal(module.matrix, [[1, 3, 5], [2, 4, 6]])
        assert module.empty.shape == (0,)
        assert module.empty_negative.shape == (0,)
        assert module.zero_word == b""
        assert module.word == b"test"
        assert_array_equal(module.words, [b"abcd", b"efgh"])
        assert int(module.memcpy) == 19
        assert int(module.size_t) == 23
        assert int(module.complex_double) == 37
        assert int(module.f2py_copy_index) == 41
        assert int(module.f2py_constants_def_0_data) == 43

    def test_values_survive_stack_reuse(self):
        module = self.module.constants
        for seed in range(32):
            assert module.clobber(seed) == 2 * seed + 8193
            assert int(module.scalar) == 1234
            assert float(module.wide) == 2.5
            assert complex(module.z) == 1.5 - 2j
            assert_array_equal(module.vector, [4, 6, 8])
            assert_array_equal(module.matrix, [[1, 3, 5], [2, 4, 6]])
            assert module.word == b"test"

    @pytest.mark.parametrize("name", ["scalar", "wide", "z", "truth", "small", "large",
                                     "vector", "matrix", "empty", "empty_negative", "zero_word", "word", "words"])
    def test_parameter_storage_is_aligned_and_readonly(self, name):
        value = getattr(self.module.constants, name)
        assert value.flags.aligned
        assert not value.flags.writeable
        assert not value.flags.owndata
        with pytest.raises(ValueError, match="read-only"):
            value[...] = 0

    def test_parameter_attribute_assignment_is_rejected(self):
        module = self.module.constants
        with pytest.raises(AttributeError, match="named constant vector"):
            module.vector = [0, 0, 0]
        assert_array_equal(module.vector, [4, 6, 8])

    def test_mutable_storage_remains_linked(self):
        module = self.module.constants
        original = int(module.mutable)
        try:
            assert module.mutable.flags.writeable
            module.mutable = 91
            assert module.read_mutable() == 91
            module.set_mutable(17)
            assert int(module.mutable) == 17
        finally:
            module.set_mutable(original)
