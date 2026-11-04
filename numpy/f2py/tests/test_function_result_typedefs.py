"""Scalar function result types retain their C typedef dependencies."""

import subprocess
import sys

import pytest

from . import util


@pytest.mark.parametrize('declaration,definition', [
    ('logical(kind=c_bool)', 'typedef unsigned char unsigned_char;'),
    ('integer(kind=-1)', 'typedef unsigned char unsigned_char;'),
    ('integer(kind=-2)', 'typedef unsigned short unsigned_short;'),
    ('integer(kind=-8)', 'typedef unsigned long long unsigned_long_long;'),
])
def test_scalar_result_typedef_is_emitted(tmp_path, declaration, definition):
    signature = tmp_path / 'result_types.pyf'
    signature.write_text(
        'python module result_types\n'
        '  interface\n'
        '    function result_value()\n'
        '      use iso_c_binding, only: c_bool\n'
        f'      {declaration} :: result_value\n'
        '    end function result_value\n'
        '  end interface\n'
        'end python module result_types\n', encoding='ascii')
    result = subprocess.run(
        [sys.executable, '-m', 'numpy.f2py', str(signature),
         '--no-wrap-functions'],
        cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    source = (tmp_path / 'result_typesmodule.c').read_text(encoding='utf-8')
    assert source.count(definition) == 1


@pytest.mark.slow
class TestByteLogicalResults(util.F2PyTest):
    options = ['--no-wrap-functions']
    sources = [util.getpath('tests', 'src', 'return_types',
                            'scalar_result_aliases.f90')]

    def test_true_result(self):
        assert self.module.scalar_result_aliases.true_result() == 1

    def test_false_result(self):
        assert self.module.scalar_result_aliases.false_result() == 0
