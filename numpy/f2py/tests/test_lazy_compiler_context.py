import builtins
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from numpy.f2py import _compiler_context as context, crackfortran, f2py2e


def test_no_query_forwards_machine_file_names_without_reading(monkeypatch):
    files = ['missing-cross.ini']
    native = ['missing-native.ini']

    def forbidden(*args):
        raise AssertionError('No compiler query requests machine-file contents')

    monkeypatch.setattr(context, '_materialize_machine_files', forbidden)
    original = context.current()
    with context.generation_context(cross_files=files,
                                    native_files=native) as selected:
        assert context.current() is selected
        assert files == ['missing-cross.ini']
        assert native == ['missing-native.ini']
    assert context.current() == original


def test_compile_forwards_missing_files_when_generation_has_no_query(
    monkeypatch, tmp_path,
):
    import sys

    received = []

    def forbidden(*args):
        raise AssertionError('Compiler context is not requested')

    def backend(*args):
        received.append(args[-1])
        return SimpleNamespace(compile=lambda: None)

    monkeypatch.setattr(context, '_materialize_machine_files', forbidden)
    monkeypatch.setattr(f2py2e, 'run_main', lambda args: None)
    monkeypatch.setattr(f2py2e, 'f2py_build_generator', lambda name: backend)
    monkeypatch.setattr(sys, 'argv', [
        'f2py', '-c', '-m', 'forward_probe', '--build-dir', str(tmp_path),
        '--cross-file', 'missing-cross.ini',
        '--native-file', 'missing-native.ini', 'source.f90',
    ])
    f2py2e.run_compile()
    assert received == [{
        'dependencies': [], 'cross_files': ['missing-cross.ini'],
        'native_files': ['missing-native.ini'],
    }]


def test_real_query_diagnoses_missing_cross_file_without_native_fallback(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv('FC', 'native-fc')
    calls = []
    monkeypatch.setattr(crackfortran.subprocess, 'run',
                        lambda *args, **kwargs: calls.append(args))
    original = context.current()
    with pytest.raises(context.CompilerExecutionError, match='missing-cross.ini'):
        with context.generation_context(
            cross_files=[str(tmp_path / 'missing-cross.ini')],
        ):
            crackfortran._selected_real_kind_func(0)
    assert calls == []
    assert context.current() == original


def test_resolved_context_keeps_entry_environment_and_identity(monkeypatch, tmp_path):
    filename = tmp_path / 'cross.ini'
    filename.write_text('[properties]\n')
    monkeypatch.setenv('FC', 'target-fc')
    monkeypatch.setenv('FFLAGS', '-O2')
    monkeypatch.setenv('MESON_EXE_WRAPPER', 'qemu -L /target')
    with context.generation_context(cross_files=[str(filename)],
                                    fortran_args=['-g']) as selected:
        monkeypatch.setenv('FC', 'different-fc')
        monkeypatch.setenv('FFLAGS', '-O0')
        monkeypatch.setenv('MESON_EXE_WRAPPER', 'different-runner')
        assert selected.command == ('target-fc',)
        assert selected.compile_args == ('-O2', '-g')
        assert selected.runner == ('qemu', '-L', '/target')
        resolved = context.coerce(selected)
        assert context.current() is selected
        assert context.coerce(selected) is resolved
        assert selected == resolved
        assert resolved == selected
        assert hash(selected) == hash(resolved)


def test_stream_machine_file_is_copied_once_and_forwarded(monkeypatch, tmp_path):
    source = b"[binaries]\nfortran = ['target-fc']\n"
    stream_name = str(tmp_path / 'stream-input')
    stream_reads = []
    original_open = builtins.open

    def open_file(path, *args, **kwargs):
        if str(path) == stream_name:
            stream_reads.append(str(path))
            return io.BytesIO(source)
        return original_open(path, *args, **kwargs)

    def paths(filenames, kind, working_directory=None):
        if list(filenames) == ['stream-input']:
            return [stream_name]
        return [str(Path(name).absolute()) for name in filenames]

    monkeypatch.setattr(context, '_machine_paths', paths)
    monkeypatch.setattr(builtins, 'open', open_file)
    files = ['stream-input']
    with context.generation_context(cross_files=files,
                                    source_dir=tmp_path) as selected:
        assert selected.command == ('target-fc',)
        resolved = context.coerce(selected)
        assert context.coerce(selected) is resolved
        assert hash(selected) == hash(resolved)
        assert files != ['stream-input']
        assert Path(files[0]).read_bytes() == source
    assert stream_reads == [stream_name]


def test_context_cache_key_still_distinguishes_runner_arguments(tmp_path):
    filename = tmp_path / 'cross.ini'
    filename.write_text("[binaries]\nfortran = ['target-fc']\n")
    contexts = []
    for root in ['/target-one', '/target-two', '/target-one']:
        with context.generation_context(
            cross_files=[str(filename)], runner=['qemu', '-L', root],
        ) as selected:
            contexts.append(context.coerce(selected))
    assert contexts[0] == contexts[2]
    assert contexts[0] != contexts[1]
    assert len(set(contexts)) == 2


def test_resolution_failure_is_cached_and_scope_restored(monkeypatch, tmp_path):
    filename = tmp_path / 'cross.ini'
    filename.write_text('[properties]\n')
    calls = []

    def broken(*args, **kwargs):
        calls.append(args)
        raise ValueError('broken target configuration')

    monkeypatch.setattr(context, 'from_machine_files', broken)
    original = context.current()
    with context.generation_context(cross_files=[str(filename)]) as selected:
        for attempt in range(2):
            with pytest.raises(context.CompilerExecutionError,
                               match='broken target configuration'):
                context.coerce(selected)
    assert len(calls) == 1
    assert context.current() == original


def test_inner_unqueried_context_restores_outer_identity(tmp_path):
    filename = tmp_path / 'outer.ini'
    filename.write_text("[binaries]\nfortran = ['target-fc']\n")
    with context.generation_context(cross_files=[str(filename)]) as outer:
        assert outer.command == ('target-fc',)
        with context.generation_context(cross_files=['missing-inner.ini']) as inner:
            assert context.current() is inner
        assert context.current() is outer
        assert outer.command == ('target-fc',)


@pytest.mark.parametrize('file_option', ['cross_files', 'native_files'])
def test_deferred_paths_use_entry_directory_after_chdir(
    monkeypatch, tmp_path, file_option,
):
    entry = tmp_path / 'entry'
    other = tmp_path / 'other'
    entry.mkdir()
    other.mkdir()
    (entry / 'same.ini').write_text(
        "[binaries]\nfortran = ['entry-fc']\n"
        "exe_wrapper = ['@GLOBAL_SOURCE_ROOT@/runner']\n"
        "[properties]\nneeds_exe_wrapper = true\n")
    (other / 'same.ini').write_text(
        "[binaries]\nfortran = ['different-fc']\n")
    monkeypatch.chdir(entry)
    files = ['same.ini']
    with context.generation_context(**{file_option: files},
                                    source_dir='build') as selected:
        monkeypatch.chdir(other)
        assert selected.command == ('entry-fc',)
        assert selected.runner == (str(entry / 'build/runner'),)
        assert selected.working_directory == str(entry)
        assert files == [str(entry / 'same.ini')]
