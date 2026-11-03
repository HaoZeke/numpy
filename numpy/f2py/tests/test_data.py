import textwrap

import pytest

import numpy as np
from numpy.f2py.crackfortran import crackfortran

from . import util


def _crack_source(tmp_path, filename, source):
    path = tmp_path / filename
    path.write_text(textwrap.dedent(source))
    return crackfortran([str(path)])


class TestElementData:
    """gh-7189: element DATA names the array, not a variable such as ``d(1)``."""

    def test_issue_module(self, tmp_path):
        mod = _crack_source(tmp_path, 'minimal.f90', """\
            module minimal
              integer, dimension(2) :: d
              data d(1) / 1 /
              data d(2) / 2 /
            end module
            """)
        vars_ = mod[0]['vars']
        assert set(vars_) == {'d'}
        assert vars_['d']['typespec'] == 'integer'
        assert vars_['d']['dimension'] == ['2']
        assert vars_['d']['='] == '(/1, 2/)'

    def test_implicit_none(self, tmp_path):
        mod = _crack_source(tmp_path, 'minimal_none.f90', """\
            module minimal
              implicit none
              integer, dimension(2) :: d
              data d(1) / 1 /
              data d(2) / 2 /
            end module
            """)
        vars_ = mod[0]['vars']
        assert set(vars_) == {'d'}
        assert vars_['d']['typespec'] == 'integer'
        assert vars_['d']['='] == '(/1, 2/)'

    def test_element_list(self, tmp_path):
        mod = _crack_source(tmp_path, 'element_list.f90', """\
            module minimal
              integer, dimension(2) :: d
              data d(1), d(2) / 1, 2 /
            end module
            """)
        vars_ = mod[0]['vars']
        assert set(vars_) == {'d'}
        assert vars_['d']['='] == '(/1, 2/)'

    def test_repeat_factor(self, tmp_path):
        mod = _crack_source(tmp_path, 'repeat.f90', """\
            module minimal
              integer, dimension(2) :: q
              data q(1), q(2) / 2*4 /
            end module
            """)
        vars_ = mod[0]['vars']
        assert set(vars_) == {'q'}
        assert vars_['q']['='] == '(/4, 4/)'

    def test_partial_fill(self, tmp_path, capsys):
        mod = _crack_source(tmp_path, 'partial.f90', """\
            module minimal
              integer, dimension(3) :: p
              data p(1) / 1 /
            end module
            """)
        vars_ = mod[0]['vars']
        assert set(vars_) == {'p'}
        assert '=' not in vars_['p']
        captured = capsys.readouterr().out
        assert (
            'analyzeline: element DATA "p(1)" is not folded into an initialiser. '
            'Skipping.'
        ) in captured

    def test_twod_element(self, tmp_path, capsys):
        mod = _crack_source(tmp_path, 'twod.f90', """\
            module minimal
              integer, dimension(2, 2) :: e
              data e(1,1) / 5 /
            end module
            """)
        vars_ = mod[0]['vars']
        assert set(vars_) == {'e'}
        assert '=' not in vars_['e']
        captured = capsys.readouterr().out
        assert (
            'analyzeline: element DATA "e(1,1)" is not folded into an initialiser. '
            'Skipping.'
        ) in captured

    def test_whole_array(self, tmp_path):
        mod = _crack_source(tmp_path, 'whole.f90', """\
            module minimal
              integer, dimension(2) :: d
              data d / 1, 2 /
            end module
            """)
        vars_ = mod[0]['vars']
        assert set(vars_) == {'d'}
        assert vars_['d']['='] == '(/1,  2/)'

    def test_dimension_lower_bound(self, tmp_path):
        mod = _crack_source(tmp_path, 'lower.f90', """\
            module minimal
              integer, dimension(1:2) :: d
              data d(1) / 8 /
              data d(2) / 9 /
            end module
            """)
        vars_ = mod[0]['vars']
        assert set(vars_) == {'d'}
        assert vars_['d']['dimension'] == ['2']
        assert vars_['d']['='] == '(/8, 9/)'


@pytest.mark.slow
class TestGh7189(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "gh7189.f90")]

    def test_module_data(self):
        assert np.all(self.module.minimal.d == np.array([1, 2]))

    def test_subroutine_data(self):
        got = self.module.elemdata()
        assert np.all(got == np.array([3, 4]))


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
