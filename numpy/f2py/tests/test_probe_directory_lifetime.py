import gc
import warnings
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from numpy.f2py import _compiler_context, crackfortran


def successful_command(*args, **kwargs):
    return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')


def test_clearing_probe_cache_cleans_without_implicit_tempdir_warning(monkeypatch):
    monkeypatch.setattr(crackfortran.subprocess, 'run', successful_command)
    context = _compiler_context.CompilerContext(command=('test-fc',), selected=True)
    crackfortran._selected_real_kind_program.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ResourceWarning)
        owner, executable = crackfortran._selected_real_kind_program(
            context, (None, None, None))
        directory = Path(owner.name)
        reference = weakref.ref(owner)
        assert executable.parent == directory
        assert directory.is_dir()
        del owner
        crackfortran._selected_real_kind_program.cache_clear()
        gc.collect()
        assert reference() is None
        assert not directory.exists()
    assert not [item for item in caught if issubclass(item.category, ResourceWarning)]


def test_probe_cache_eviction_releases_directory_without_warning(monkeypatch):
    monkeypatch.setattr(crackfortran.subprocess, 'run', successful_command)
    crackfortran._selected_real_kind_program.cache_clear()
    directories = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ResourceWarning)
        for number in range(9):
            context = _compiler_context.CompilerContext(
                command=('test-fc', str(number)), selected=True)
            owner, _ = crackfortran._selected_real_kind_program(
                context, (None, None, None))
            directories.append(Path(owner.name))
            del owner
        assert not directories[0].exists()
        crackfortran._selected_real_kind_program.cache_clear()
        gc.collect()
        assert all(not directory.exists() for directory in directories)
    assert not [item for item in caught if issubclass(item.category, ResourceWarning)]


def test_compiler_failure_still_cleans_probe_directory(monkeypatch):
    directories = []
    original = crackfortran._ProbeDirectory

    def allocate():
        owner = original()
        directories.append(Path(owner.name))
        return owner

    monkeypatch.setattr(crackfortran, '_ProbeDirectory', allocate)
    monkeypatch.setattr(crackfortran.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(
                            returncode=1, stdout=b'', stderr=b'compile failed'))
    context = _compiler_context.CompilerContext(command=('test-fc',), selected=True)
    crackfortran._selected_real_kind_program.cache_clear()
    with pytest.raises(_compiler_context.CompilerExecutionError, match='compile failed'):
        crackfortran._selected_real_kind_program(context, (None, None, None))
    assert directories
    assert all(not directory.exists() for directory in directories)
