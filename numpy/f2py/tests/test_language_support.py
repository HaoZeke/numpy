import pytest

import numpy as np
from numpy.testing import assert_array_equal

from . import util


def _source(name):
    return util.getpath("tests", "src", "language_support", name)


@pytest.mark.slow
class TestF77Features(util.F2PyTest):
    sources = [_source("f77_features.f")]

    def test_entry(self):
        assert self.module.incr(1.0) == 2.0
        assert self.module.incr2(1.0) == 3.0

    def test_statement_function(self):
        assert self.module.sqp1(2.0) == 5.0

    def test_implicit_typing(self):
        assert self.module.impl(1.5, 2) == 3.0
        assert "n : input int" in self.module.impl.__doc__
        assert abs(self.module.impldp(1.0) - 1.0 / 3.0) < 1e-15

    def test_equivalence(self):
        assert self.module.eqv(4.0) == 4.0

    def test_save(self):
        first = self.module.countr()
        assert self.module.countr() == first + 1


@pytest.mark.slow
class TestF90Features(util.F2PyTest):
    sources = [_source("f90_features.f90")]

    def test_generic_interface_specifics(self):
        mod = self.module.f90_features
        assert mod.add_int(1, 2) == 3
        assert mod.add_real(1.5, 2.0) == 3.5
        assert not hasattr(mod, "add")

    def test_recursive(self):
        assert self.module.f90_features.factorial(5) == 120

    def test_elemental_scalar_call(self):
        assert self.module.f90_features.square(3.0) == 9.0

    def test_pure(self):
        assert self.module.f90_features.shift(1.0) == 2.0

    def test_internal_procedure(self):
        mod = self.module.f90_features
        assert mod.host(2.0) == 6.0
        assert not hasattr(mod, "triple")

    def test_array_valued_function(self):
        assert_array_equal(self.module.f90_features.twice([1.0, 2.0, 3.0]),
                           [2.0, 4.0, 6.0])

    def test_module_variables(self):
        mod = self.module.f90_features
        mod.counter = 5
        assert mod.counter == 5
        mod.table = np.arange(6.0).reshape(2, 3)
        assert_array_equal(mod.table, np.arange(6.0).reshape(2, 3))
        assert mod.table.flags.f_contiguous


@pytest.mark.slow
class TestF90Allocatable(util.F2PyTest):
    sources = [_source("f90_allocatable.f90")]

    def test_allocate_from_python(self):
        mod = self.module.f90_allocatable
        mod.buffer = [1.0, 2.0]
        assert mod.buffer_size() == 2
        assert_array_equal(mod.buffer, [1.0, 2.0])
        mod.buffer = [1.0, 2.0, 3.0]
        assert mod.buffer_size() == 3
        mod.buffer = None
        assert mod.buffer_size() == -1


@pytest.mark.slow
class TestF2003Features(util.F2PyTest):
    sources = [_source("f2003_features.f90")]

    def test_enumerator_in_fortran(self):
        assert self.module.f2003_features.last_colour() == 3

    def test_protected_variable(self):
        mod = self.module.f2003_features
        before = int(mod.counter)
        mod.bump()
        assert mod.counter == before + 1

    def test_bind_c_variable(self):
        assert self.module.f2003_features.shared_count == 5

    def test_volatile_asynchronous(self):
        assert self.module.f2003_features.copy_value(2.5) == 2.5


@pytest.mark.slow
class TestF2008Features(util.F2PyTest):
    sources = [_source("f2008_features.f90")]

    def test_contiguous(self):
        assert self.module.f2008_features.total([1.0, 2.0, 3.0]) == 6.0

    def test_do_concurrent(self):
        assert_array_equal(self.module.f2008_features.fill(3), [1.0, 2.0, 3.0])


@pytest.mark.slow
class TestF2018Features(util.F2PyTest):
    sources = [_source("f2018_features.f90")]

    def test_implicit_none_spec(self):
        assert self.module.no_implicit(1.5) == 3.0
