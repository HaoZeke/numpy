from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from numpy.f2py import _compiler_context as context, crackfortran

from . import util


def _context(**kwargs):
    return context.CompilerContext(command=('target-fc',), selected=True,
                                   **kwargs)


def test_environment_runner_keeps_arguments(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('FC', 'target-fc --target=aarch64-linux-gnu')
    monkeypatch.setenv('FFLAGS', '-O2 -Dlabel="two words"')
    monkeypatch.setenv('LDFLAGS', '-L/opt/sysroot/lib')
    monkeypatch.setenv('MESON_EXE_WRAPPER', 'qemu-aarch64 -L "/opt/sysroot a"')
    configured = context.from_environment()
    assert configured.command == ('target-fc', '--target=aarch64-linux-gnu')
    assert configured.compile_args == ('-O2', '-Dlabel=two words')
    assert configured.link_args == ('-O2', '-Dlabel=two words', '-L/opt/sysroot/lib')
    assert configured.runner == ('qemu-aarch64', '-L', '/opt/sysroot a')


def test_meson_machine_files_preserve_layered_expressions(monkeypatch, tmp_path):
    first = tmp_path / 'target.ini'
    second = tmp_path / 'override.ini'
    first.write_text("""[constants]
root = '/opt/target tree'
wrapper = ['qemu-aarch64', '-L', root]
[binaries]
fortran = ['target-fc', '--sysroot=' + root]
exe_wrapper = wrapper
[properties]
needs_exe_wrapper = true
[built-in options]
fortran_args = ['-O2']
fortran_link_args = ['-L' + root / 'lib']
""")
    second.write_text("""[built-in options]
fortran_args = ['-O3', '-Dname=two words']
""")
    monkeypatch.setenv('FC', 'host-fc')
    configured = context.from_machine_files([first, second], source_dir=tmp_path)
    assert configured.command == ('target-fc', '--sysroot=/opt/target tree')
    assert configured.runner == ('qemu-aarch64', '-L', '/opt/target tree')
    assert configured.compile_args == ('-O3', '-Dname=two words')
    assert configured.link_args == ('-L/opt/target tree/lib',)
    assert configured.needs_runner is True


def test_generation_context_restores_on_error(monkeypatch):
    monkeypatch.setenv('FC', 'host-fc')
    monkeypatch.delenv('MESON_EXE_WRAPPER', raising=False)
    original = context.current()
    with pytest.raises(RuntimeError, match='stop generation'):
        with context.generation_context(runner=['qemu', '-L', '/target'],
                                        fortran_args=['-O3']) as selected:
            assert context.current() is selected
            assert selected.runner == ('qemu', '-L', '/target')
            assert selected.compile_args[-1] == '-O3'
            raise RuntimeError('stop generation')
    assert context.current() == original


def test_compile_generation_uses_cross_context_before_backend(monkeypatch, tmp_path):
    import sys

    from numpy.f2py import f2py2e

    machine_file = tmp_path / 'cross.ini'
    machine_file.write_text("""[binaries]
fortran = ['target-fc']
exe_wrapper = ['qemu-aarch64', '-L', '/target tree']
[properties]
needs_exe_wrapper = true
""")
    monkeypatch.setenv('FC', 'host-fc')
    original = context.current()
    observations = []

    def generate(arguments):
        observations.append(('generate', context.current(), arguments))

    def backend(*args):
        observations.append(('backend', context.current(), args[-1]))
        return SimpleNamespace(compile=lambda: None)

    monkeypatch.setattr(f2py2e, 'run_main', generate)
    monkeypatch.setattr(f2py2e, 'f2py_build_generator', lambda name: backend)
    monkeypatch.setattr(sys, 'argv', [
        'f2py', '-c', '-m', 'cross_probe', '--cross-file', str(machine_file),
        '--build-dir', str(tmp_path), '--f90flags=-O3', 'source.f90',
    ])
    f2py2e.run_compile()
    assert [value[0] for value in observations] == ['generate', 'backend']
    selected = observations[0][1]
    assert selected.command == ('target-fc',)
    assert selected.runner == ('qemu-aarch64', '-L', '/target tree')
    assert selected.compile_args[-1] == '-O3'
    assert '--lower' in observations[0][2]
    assert observations[1][1] == original
    assert observations[1][2]['cross_files'] == [str(machine_file)]
    assert context.current() == original


def test_query_compiles_links_then_runs_only_executable_through_wrapper(
    monkeypatch, tmp_path,
):
    invocations = []

    def run(argv, **kwargs):
        invocations.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=b'-1\n', stderr=b'')

    configured = _context(compile_args=('-O3',), link_args=('-lm',),
                          runner=('qemu-aarch64', '-L', '/target tree'))
    monkeypatch.setattr(crackfortran.subprocess, 'run', run)
    crackfortran._selected_real_kind_program.cache_clear()
    try:
        value = crackfortran._execute_selected_real_kind(
            configured, crackfortran._Integer(2**40, 8), None,
            crackfortran._Integer(2**40, 8))
        assert value == -1
        assert invocations[0][0][:3] == ['target-fc', '-O3', '-c']
        object_file = invocations[0][0][-1]
        executable = invocations[1][0][-1]
        assert invocations[1][0] == [
            'target-fc', object_file, '-lm', '-o', executable,
        ]
        assert invocations[2][0][:3] == ['qemu-aarch64', '-L', '/target tree']
        assert invocations[2][1]['input'] == b'5 1099511627776 0 1099511627776\n'
        program = Path(invocations[0][0][3]).read_text()
        assert 'integer(kind=8) :: p' in program
        assert 'integer(kind=8) :: radix' in program
        assert 'integer :: r' in program
    finally:
        crackfortran._selected_real_kind_program.cache_clear()


def test_native_query_has_no_wrapper(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(crackfortran, '_selected_real_kind_program',
                        lambda *args: (SimpleNamespace(name=str(tmp_path)),
                                       tmp_path / 'probe'))

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=b'4\n', stderr=b'')

    monkeypatch.setattr(crackfortran.subprocess, 'run', run)
    assert crackfortran._execute_selected_real_kind(_context(), 0, None, None) == 4
    assert calls == [[str(tmp_path / 'probe')]]


def test_missing_cross_runner_refuses_execution(monkeypatch):
    calls = []
    monkeypatch.setattr(crackfortran.subprocess, 'run',
                        lambda *args, **kwargs: calls.append(args))
    with pytest.raises(context.CompilerExecutionError, match='target runner'):
        crackfortran._execute_selected_real_kind(
            _context(needs_runner=True, cross=True), 0, None, None)
    assert calls == []


def test_context_and_argument_kinds_fence_probe_cache(monkeypatch):
    keys = []

    def program(configured, kinds):
        keys.append((configured, kinds))
        return SimpleNamespace(name='/tmp'), '/tmp/probe'

    monkeypatch.setattr(crackfortran, '_selected_real_kind_program', program)
    monkeypatch.setattr(crackfortran.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(
                            returncode=0, stdout=b'-1\n', stderr=b''))
    original = _context(runner=('qemu', '-L', '/one'))
    configurations = [original,
                      replace(original, runner=('qemu', '-L', '/two')),
                      replace(original, compile_args=('-O2',)),
                      replace(original, link_args=('-lm',)),
                      replace(original, working_directory='/another'),
                      replace(original, search_path='/another/bin')]
    for configured in configurations:
        crackfortran._execute_selected_real_kind(configured, 0, None, None)
    crackfortran._execute_selected_real_kind(
        original, crackfortran._Integer(0, 8), None, None)
    assert len(set(keys)) == len(configurations) + 1
    assert keys[-1][1] == (8, None, None)


def test_binary128_query_uses_execution_context(monkeypatch):
    calls = []
    configured = _context(runner=('qemu', '-L', '/target'))

    def query(selected, p, r, radix):
        calls.append((selected, p, r, radix))
        return 16

    monkeypatch.setattr(crackfortran, '_execute_selected_real_kind', query)
    assert crackfortran._run_selected_real_kind(33, configured) == 16
    assert calls == [(configured, 33, 0, None)]


@pytest.mark.parametrize('entry', [
    lambda: crackfortran._fit_length('a', {'charselector': {'len': 'n'}}, {}, {}),
    lambda: crackfortran._declared_type(
        'a', {'typespec': 'real', 'kindselector': {'kind': 'n'}}, {}),
    lambda: crackfortran._get_typed_parameters(
        {'a': {'typespec': 'integer', 'attrspec': ['parameter'], '=': 'n'}}),
    lambda: crackfortran._eval_dimension('n', {}),
    lambda: crackfortran._eval_scalar('n', {}),
    lambda: crackfortran._extend_parameter_items([], 'n', {}, {}, False),
    lambda: crackfortran.param_eval('n', {}, {}),
])
def test_execution_diagnostics_escape_unresolved_expression_fallbacks(
    monkeypatch, entry,
):
    def fail(*args, **kwargs):
        raise context.CompilerExecutionError('target runner unavailable')

    monkeypatch.setattr(crackfortran, '_fortran_eval', fail)
    monkeypatch.setattr(crackfortran, '_eval_items', fail)
    with pytest.raises(context.CompilerExecutionError, match='target runner'):
        entry()


def test_unknown_source_expression_keeps_ordinary_fallback():
    assert crackfortran._eval_scalar('unresolved_name', {}) == 'unresolved_name'
    assert crackfortran._eval_dimension('unresolved_name', {}) == 'unresolved_name'
    assert crackfortran.param_eval('unresolved_name', {}, {}) == 'unresolved_name'


@pytest.mark.parametrize('failure', ['missing', 'timeout', 'nonzero'])
def test_configured_compiler_failures_have_execution_diagnostics(monkeypatch, failure):
    def run(*args, **kwargs):
        if failure == 'missing':
            raise FileNotFoundError('configured compiler')
        if failure == 'timeout':
            raise crackfortran.subprocess.TimeoutExpired('configured compiler', 1)
        return SimpleNamespace(returncode=2, stdout=b'', stderr=b'compile failed')

    monkeypatch.setattr(crackfortran.subprocess, 'run', run)
    crackfortran._selected_real_kind_program.cache_clear()
    try:
        with pytest.raises(context.CompilerExecutionError, match='probe'):
            crackfortran._execute_selected_real_kind(_context(), 0, None, None)
    finally:
        crackfortran._selected_real_kind_program.cache_clear()


@pytest.mark.parametrize('stdout,returncode', [(b'not an integer', 0),
                                              (b'\xff', 0), (b'4', 2)])
def test_invalid_probe_output_is_not_a_guessed_kind(
    monkeypatch, tmp_path, stdout, returncode,
):
    monkeypatch.setattr(crackfortran, '_selected_real_kind_program',
                        lambda *args: (SimpleNamespace(name=str(tmp_path)),
                                       tmp_path / 'probe'))
    monkeypatch.setattr(crackfortran.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(
                            returncode=returncode, stdout=stdout, stderr=b'bad result'))
    with pytest.raises(context.CompilerExecutionError, match='probe'):
        crackfortran._execute_selected_real_kind(_context(), 0, None, None)


def test_cross_file_can_use_exported_meson_wrapper(monkeypatch, tmp_path):
    machine_file = tmp_path / 'cross.ini'
    machine_file.write_text("""[binaries]
fortran = ['target-fc']
[properties]
needs_exe_wrapper = true
""")
    monkeypatch.setenv('MESON_EXE_WRAPPER', 'qemu-aarch64 -L "/target tree"')
    configured = context.from_machine_files([machine_file], source_dir=tmp_path)
    assert configured.runner == ('qemu-aarch64', '-L', '/target tree')
    assert configured.needs_runner is True


def test_unspecified_machine_without_wrapper_keeps_native_execution(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv('MESON_EXE_WRAPPER', raising=False)
    machine_file = tmp_path / 'native.ini'
    machine_file.write_text("""[binaries]
fortran = ['native-fc']
""")
    configured = context.from_machine_files(native_files=[machine_file],
                                             source_dir=tmp_path)
    assert configured.command == ('native-fc',)
    assert configured.needs_runner is False
    assert configured.cross is False
    assert configured.runner == ()


def test_named_cross_file_uses_meson_data_directory(monkeypatch, tmp_path):
    directory = tmp_path / 'meson' / 'cross'
    directory.mkdir(parents=True)
    filename = directory / 'aarch64'
    filename.write_text("""[binaries]
fortran = ['target-fc']
exe_wrapper = ['qemu-aarch64']
[properties]
needs_exe_wrapper = true
""")
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path))
    monkeypatch.setenv('XDG_DATA_DIRS', str(tmp_path / 'absent'))
    files = ['aarch64']
    with context.generation_context(cross_files=files,
                                    source_dir=tmp_path) as selected:
        assert selected.command == ('target-fc',)
        assert selected.runner == ('qemu-aarch64',)
        assert files == [str(filename)]


def test_fast_integer_probe_uses_same_cross_compiler_context(monkeypatch):
    from numpy.f2py import capi_maps

    received = []
    monkeypatch.setenv('FC', 'target-fc')
    monkeypatch.setattr(capi_maps, '_fast_user_override', set())

    def probe(name, selected):
        received.append((name, selected))
        return None

    monkeypatch.setattr(capi_maps, '_fast_integer_ctype', probe)
    with context.generation_context(runner=['qemu-aarch64', '-L', '/target'],
                                    fortran_args=['-O3']) as selected:
        assert capi_maps._fast_c_type('c_int_fast16_t') is None
        assert received == [('c_int_fast16_t', selected)]
        assert selected.command == ('target-fc',)
        assert selected.compile_args[-1] == '-O3'


@pytest.mark.slow
class TestWideSelectedRealKind(util.F2PyTest):
    suffix = '.f90'
    code = """
      subroutine wide_kind_query(p, r, radix, result)
      implicit none
      integer(kind=8), intent(in) :: p, r, radix
      integer, intent(out) :: result
      result = selected_real_kind(p, r, radix)
      end subroutine wide_kind_query
    """

    @pytest.mark.parametrize('p,r,radix', [
        (2**40, 0, 2),
        (0, 2**40, 2),
        (0, 0, 2**40),
    ])
    def test_wide_integer_intrinsic_arguments(self, p, r, radix):
        expected = self.module.wide_kind_query(p, r, radix)
        actual = crackfortran._selected_real_kind_func(
            *(crackfortran._Integer(value, 8) for value in (p, r, radix)))
        assert actual == expected
