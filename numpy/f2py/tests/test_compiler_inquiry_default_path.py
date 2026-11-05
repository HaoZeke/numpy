"""Absent and empty PATH retain different executable-search semantics."""

import os
from types import SimpleNamespace

from numpy.f2py import _compiler_context, _compiler_inquiry, crackfortran


def test_probe_absent_and_empty_path_keep_distinct_lookup(monkeypatch, tmp_path):
    monkeypatch.setenv('FC', 'f2py-context-fc')
    monkeypatch.setenv('FFLAGS', '')
    monkeypatch.setenv('LDFLAGS', '')
    monkeypatch.setenv('MESON_EXE_WRAPPER', 'f2py-context-runner')
    monkeypatch.setenv('F2PY_CONTEXT_SENTINEL', 'preserved')
    commands = []
    owners = []
    contexts = []
    original_program = _compiler_inquiry._selected_real_kind_program.__wrapped__

    def build(context, kinds):
        owner, executable = original_program(context, kinds)
        owners.append(owner)
        return owner, executable

    def run(command, **kwargs):
        assert kwargs['env']['PATH'] == expected_path
        assert kwargs['env']['F2PY_CONTEXT_SENTINEL'] == 'preserved'
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=b'2\n', stderr=b'')

    monkeypatch.setattr(_compiler_inquiry, '_selected_real_kind_program', build)
    monkeypatch.setattr(_compiler_inquiry.subprocess, 'run', run)
    try:
        for absent, expected_path in [(True, os.defpath), (False, '')]:
            if absent:
                monkeypatch.delenv('PATH', raising=False)
            else:
                monkeypatch.setenv('PATH', '')
            with _compiler_context.generation_context() as context:
                assert context.selected
                assert context.search_path == expected_path
                contexts.append(context)
                monkeypatch.setenv('PATH', str(tmp_path / 'different-bin'))
                result = crackfortran._selected_real_kind_func(0)
                assert type(result) is int
                assert result == 2
                assert context.search_path == expected_path
        assert contexts[0] != contexts[1]
        assert len(commands) == 6
        for start in [0, 3]:
            assert commands[start][0] == 'f2py-context-fc'
            assert '-c' in commands[start]
            assert commands[start + 1][0] == 'f2py-context-fc'
            assert commands[start + 2][0] == 'f2py-context-runner'
    finally:
        for owner in owners:
            owner.cleanup()
