"""Include directives preserve their argv path payloads."""

import os

import pytest

from numpy.f2py.f2py2e import get_newer_options

from . import util

_INCLUDED_VALUE = 42


@pytest.mark.parametrize('option', ['--include-paths', '--include_paths'])
@pytest.mark.parametrize('form', ['combined', 'separate', 'equals'])
def test_include_directive_argument_forms_preserve_spaces(tmp_path, option, form):
    first = str(tmp_path / 'first include directory')
    second = str(tmp_path / 'second include directory')
    value = os.pathsep.join([first, second])
    if form == 'combined':
        directive = [f'{option} {value}']
    elif form == 'separate':
        directive = [option, value]
    else:
        directive = [f'{option}={value}']
    remaining = ['source with spaces.f90', '-m', 'include_probe']
    original = directive + remaining
    paths, free_threaded, rest = get_newer_options(original)
    assert paths == [first, second]
    assert free_threaded is None
    assert rest == remaining
    assert original == directive + remaining


def test_similar_unknown_option_remains_unknown():
    argument = '--include-paths-extra directory with spaces'
    paths, free_threaded, rest = get_newer_options([argument])
    assert paths == []
    assert free_threaded is None
    assert rest == [argument]


@pytest.mark.slow
@pytest.mark.parametrize('option', ['--include-paths', '--include_paths'])
def test_native_combined_include_directive_with_spaces(tmp_path, option):
    include_dir = tmp_path / 'native include directory'
    include_dir.mkdir()
    (include_dir / 'value.inc').write_text(
        f'integer, parameter :: include_value = {_INCLUDED_VALUE}\n')
    source = '''subroutine read_included(answer)
  integer, intent(out) :: answer
  include 'value.inc'
  answer = include_value
end subroutine read_included
'''
    module = util.build_code(source, suffix='.f90',
                             options=[f'-I{include_dir}', f'{option} {include_dir}'])
    assert module.read_included() == _INCLUDED_VALUE


@pytest.mark.parametrize('option', [
    '--include-paths', '--include_paths', '-I', '--include-', '--include_p',
])
@pytest.mark.parametrize('value', [
    '--include-paths literal directory', '--include_paths literal directory',
])
def test_include_option_value_preserves_literal_directive(option, value):
    arguments = [option, value]
    paths, free_threaded, rest = get_newer_options(arguments)
    assert paths == [value]
    assert free_threaded is None
    assert rest == []
    assert arguments == [option, value]


@pytest.mark.parametrize('arguments', [
    ['--', '--include-paths literal directory'],
    ['--', '-I', '--include_paths literal directory'],
])
def test_end_of_options_preserves_literal_tokens(arguments):
    original = list(arguments)
    paths, free_threaded, rest = get_newer_options(arguments)
    assert paths == []
    assert free_threaded is None
    assert rest == original
    assert arguments == original


@pytest.mark.parametrize('option', ['--incl', '--include'])
def test_ambiguous_include_abbreviation_retains_usage_error(option):
    with pytest.raises(SystemExit) as error:
        get_newer_options([option, 'literal directory'])
    assert error.value.code == 2


def test_standalone_directive_after_a_value_is_recognized():
    paths, free_threaded, rest = get_newer_options([
        '-I', 'first directory', '--include-paths second directory',
    ])
    assert paths == ['first directory', 'second directory']
    assert free_threaded is None
    assert rest == []


def test_identical_text_can_be_a_value_and_a_standalone_directive():
    value = '--include-paths literal directory'
    paths, free_threaded, rest = get_newer_options(['--include-paths', value, value])
    assert paths == [value, 'literal directory']
    assert free_threaded is None
    assert rest == []
