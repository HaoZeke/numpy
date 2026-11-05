"""Scalar C aliases have declarations, symbols, and integer conversions."""

import subprocess
import sys

import pytest

from . import util


def _generate(tmp_path, signature, module_name):
    path = tmp_path / f'{module_name}.pyf'
    path.write_text(signature, encoding='ascii')
    result = subprocess.run(
        [sys.executable, '-m', 'numpy.f2py', str(path),
         '--no-wrap-functions'],
        cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return (tmp_path / f'{module_name}module.c').read_text(encoding='utf-8')


def test_logical_argument_typedef_precedes_prototype(tmp_path):
    source = _generate(tmp_path, """
python module logical_input
  interface
    subroutine check_flag(flag, result)
      use iso_c_binding, only: c_bool
      logical(kind=c_bool), intent(in) :: flag
      integer, intent(out) :: result
    end subroutine check_flag
  end interface
end python module logical_input
""", 'logical_input')
    definition = 'typedef unsigned char unsigned_char;'
    assert source.count(definition) == 1
    assert source.index(definition) < source.index('unsigned_char*')


def test_unwrapped_module_functions_use_native_symbols(tmp_path):
    path = util.getpath('tests', 'src', 'return_types',
                        'scalar_result_aliases.f90')
    result = subprocess.run(
        [sys.executable, '-m', 'numpy.f2py', str(path),
         '-m', 'native_aliases', '--no-wrap-functions'],
        cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    source = (tmp_path / 'native_aliases-f2pywrappers2.f90').read_text(
        encoding='utf-8').lower()
    assert 'use scalar_result_aliases, only : true_result' in source
    assert 'use scalar_result_aliases, only : false_result' in source
    assert 'f2pywrap_scalar_result_aliases_true_result' not in source
    assert 'f2pywrap_scalar_result_aliases_false_result' not in source


def test_unsigned_result_uses_integer_buildvalue(tmp_path):
    source = _generate(tmp_path, """
python module unsigned_result
  interface
    function result_value()
      integer(kind=-8), intent(c) :: result_value
    end function result_value
  end interface
end python module unsigned_result
""", 'unsigned_result')
    assert 'Py_BuildValue("K",' in source
    assert 'typedef unsigned __int64 unsigned_long_long;' in source
    assert 'typedef unsigned long long unsigned_long_long;' in source


@pytest.mark.slow
class TestScalarAliasReturns(util.F2PyTest):
    suffix = '.pyf'
    module_name = 'c_scalar_alias_results'
    options = ['--no-wrap-functions']
    code = """
python module c_scalar_alias_results
usercode '''
#include <limits.h>
unsigned char max_u8(void) { return UCHAR_MAX; }
unsigned short max_u16(void) { return USHRT_MAX; }
unsigned int max_u32(void) { return UINT_MAX; }
unsigned long long max_u64(void) { return ULLONG_MAX; }
unsigned char byte_true(void) { return 1; }
'''
interface
  function max_u8()
    integer(kind=-1), intent(c) :: max_u8
  end function max_u8
  function max_u16()
    integer(kind=-2), intent(c) :: max_u16
  end function max_u16
  function max_u32()
    integer(kind=-4), intent(c) :: max_u32
  end function max_u32
  function max_u64()
    integer(kind=-8), intent(c) :: max_u64
  end function max_u64
  function byte_true()
    use iso_c_binding, only: c_bool
    logical(kind=c_bool), intent(c) :: byte_true
  end function byte_true
end interface
end python module c_scalar_alias_results
"""

    @pytest.mark.parametrize('name,expected', [
        ('max_u8', 2**8 - 1),
        ('max_u16', 2**16 - 1),
        ('max_u32', 2**32 - 1),
        ('max_u64', 2**64 - 1),
        ('byte_true', 1),
    ])
    def test_integer_result(self, name, expected):
        result = getattr(self.module, name)()
        assert type(result) is int
        assert result == expected
