import pytest

import numpy as np
from numpy.f2py.crackfortran import crackfortran

from . import util


class TestModuleIndexedData:
    """Element-indexed DATA belongs to the declared array."""

    @pytest.mark.parametrize("implicit", ["", "implicit none"])
    @pytest.mark.parametrize("name, mapped_name", [("d", "d"), ("size", "size_bn")])
    @pytest.mark.parametrize("statements", [
        "data d(1) / 1 /\n  data d(2) / 2 /",
        "data d(1), d(2) / 1, 2 /",
    ])
    def test_element_indexed_data(self, tmp_path, implicit, statements, name, mapped_name):
        path = tmp_path / "indexed_data.f90"
        path.write_text(
            "module minimal\n" + implicit + "\n"
            f"integer, dimension(2) :: {name}\n"
            + statements.replace("d(", f"{name}(")
            + "\nend module minimal\n")
        blocks = crackfortran([str(path)])
        assert len(blocks) == 1
        variables = blocks[0]["vars"]
        assert set(variables) == {mapped_name}
        assert variables[mapped_name]["typespec"] == "integer"
        assert variables[mapped_name]["dimension"] == ["2"]
        assert variables[mapped_name]["="] == "(/1, 2/)"

    @pytest.mark.parametrize("implicit", ["", "implicit none"])
    @pytest.mark.parametrize("statements", [
        "data size(1) / 1 /\n  data size(2) / 2 /",
        "data size(1), size(2) / 1, 2 /",
    ])
    def test_reserved_name_data(self, tmp_path, implicit, statements):
        path = tmp_path / "reserved_name_data.f90"
        path.write_text(
            "module minimal\n" + implicit + "\n"
            "integer, dimension(2) :: size\n" + statements
            + "\nend module minimal\n")
        variables = crackfortran([str(path)])[0]["vars"]
        assert set(variables) == {"size_bn"}
        assert variables["size_bn"]["typespec"] == "integer"
        assert variables["size_bn"]["dimension"] == ["2"]
        assert variables["size_bn"]["="] == "(/1, 2/)"

    def test_mixed_whole_array_data(self, tmp_path):
        path = tmp_path / "mixed_data.f90"
        path.write_text(
            "module minimal\nimplicit none\ninteger :: a(2), d(1)\n"
            "data a, d(1) / 1, 2, 3 /\nend module minimal\n")
        variables = crackfortran([str(path)])[0]["vars"]
        assert set(variables) == {"a", "d"}
        assert variables["a"]["="] == "(/1, 2/)"
        assert variables["d"]["="] == "(/3/)"

    def test_unknown_array_extent_data(self, tmp_path):
        path = tmp_path / "unknown_extent_data.f90"
        path.write_text(
            "module minimal\nimplicit none\ninteger, parameter :: n = 2\n"
            "integer :: a(n), d(1)\n"
            "data a, d(1) / 1, 2, 3 /\nend module minimal\n")
        variables = crackfortran([str(path)])[0]["vars"]
        assert set(variables) == {"n", "a", "d"}
        assert "=" not in variables["d"]

    def test_repeated_element_value(self, tmp_path):
        path = tmp_path / "repeated_data.f90"
        path.write_text(
            "module minimal\ninteger, dimension(2) :: d\n"
            "data d(1), d(2) / 2*3 /\nend module minimal\n")
        variables = crackfortran([str(path)])[0]["vars"]
        assert set(variables) == {"d"}
        assert variables["d"]["="] == "(/3, 3/)"

    def test_whole_array_data(self, tmp_path):
        path = tmp_path / "whole_array_data.f90"
        path.write_text(
            "module minimal\ninteger, dimension(2) :: d\n"
            "data d / 1, 2 /\nend module minimal\n")
        variables = crackfortran([str(path)])[0]["vars"]
        assert set(variables) == {"d"}
        assert variables["d"]["="] == "(/1,  2/)"


@pytest.mark.slow
class TestModuleIndexedDataRuntime(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "gh7189_indexed.f90")]

    def test_module_values(self):
        np.testing.assert_array_equal(self.module.minimal.d, [1, 2])
        np.testing.assert_array_equal(self.module.minimal_none.d, [1, 2])
        np.testing.assert_array_equal(self.module.repeated.d, [3, 3])
        np.testing.assert_array_equal(self.module.reserved_name.size, [1, 2])
        np.testing.assert_array_equal(self.module.mixed.a, [1, 2])
        np.testing.assert_array_equal(self.module.mixed.d, [3])
        np.testing.assert_array_equal(self.module.unknown_extent.a, [1, 2])
        np.testing.assert_array_equal(self.module.unknown_extent.d, [3])


@pytest.mark.slow
class TestData(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_stmts.f90")]

    # For gh-23276
    def test_data_stmts(self):
        assert self.module.cmplxdat.i == 2
        assert self.module.cmplxdat.j == 3
        assert self.module.cmplxdat.x == 1.5
        assert self.module.cmplxdat.y == 2.0
        assert self.module.cmplxdat.pi == 3.1415926535897932384626433832795028841971693993751058209749445923078164062
        assert self.module.cmplxdat.medium_ref_index == np.array(1. + 0.j)
        assert np.all(self.module.cmplxdat.z == np.array([3.5, 7.0]))
        assert np.all(self.module.cmplxdat.my_array == np.array([ 1. + 2.j, -3. + 4.j]))
        assert np.all(self.module.cmplxdat.my_real_array == np.array([ 1., 2., 3.]))
        assert np.all(self.module.cmplxdat.ref_index_one == np.array([13.0 + 21.0j]))
        assert np.all(self.module.cmplxdat.ref_index_two == np.array([-30.0 + 43.0j]))

    def test_crackedlines(self):
        mod = crackfortran(self.sources)
        assert mod[0]['vars']['x']['='] == '1.5'
        assert mod[0]['vars']['y']['='] == '2.0'
        assert mod[0]['vars']['pi']['='] == '3.1415926535897932384626433832795028841971693993751058209749445923078164062d0'
        assert mod[0]['vars']['my_real_array']['='] == '(/1.0d0, 2.0d0, 3.0d0/)'
        assert mod[0]['vars']['ref_index_one']['='] == '(13.0d0, 21.0d0)'
        assert mod[0]['vars']['ref_index_two']['='] == '(-30.0d0, 43.0d0)'
        assert mod[0]['vars']['my_array']['='] == '(/(1.0d0, 2.0d0), (-3.0d0, 4.0d0)/)'
        assert mod[0]['vars']['z']['='] == '(/3.5,  7.0/)'

@pytest.mark.slow
class TestDataF77(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_common.f")]

    # For gh-23276
    def test_data_stmts(self):
        assert self.module.mycom.mydata == 0

    def test_crackedlines(self):
        mod = crackfortran(str(self.sources[0]))
        print(mod[0]['vars'])
        assert mod[0]['vars']['mydata']['='] == '0'


@pytest.mark.slow
class TestDataMultiplierF77(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_multiplier.f")]

    # For gh-23276
    def test_data_stmts(self):
        assert self.module.mycom.ivar1 == 3
        assert self.module.mycom.ivar2 == 3
        assert self.module.mycom.ivar3 == 2
        assert self.module.mycom.ivar4 == 2
        assert self.module.mycom.evar5 == 0


@pytest.mark.slow
class TestDataWithCommentsF77(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_with_comments.f")]

    # For gh-23276
    def test_data_stmts(self):
        assert len(self.module.mycom.mytab) == 3
        assert self.module.mycom.mytab[0] == 0
        assert self.module.mycom.mytab[1] == 4
        assert self.module.mycom.mytab[2] == 0
