"""Automatic compiler selection retains Meson and captured execution settings."""

import json
import os
import subprocess

import pytest

from numpy.f2py import _compiler_context, _specification_compiler


def test_meson_compiler_detection_is_lazy_and_cached(monkeypatch):
    monkeypatch.delenv("FC", raising=False)
    monkeypatch.setenv("PATH", "/captured/compiler/bin")
    monkeypatch.setenv("FFLAGS", "-fcheck=bounds")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "setup":
            return subprocess.CompletedProcess(command, 0, b"setup", b"")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {"host": {"fortran": {"exelist": ["target-fc", "--target=chosen"]}}}
            ).encode(),
            b"",
        )

    monkeypatch.setattr(_specification_compiler.subprocess, "run", run)
    with _compiler_context.generation_context(
        fortran_args=("-fdefault-integer-8",),
        discover=True,
    ) as context:
        assert calls == []
        monkeypatch.setenv("PATH", "/different/compiler/bin")
        assert bool(context)
        assert context.command == ("target-fc", "--target=chosen")
        assert context.compile_args == ("-fcheck=bounds", "-fdefault-integer-8")
        assert context.search_path == "/captured/compiler/bin"
        assert _compiler_context.current() is context
        assert context.resolve() is context.resolve()
    assert len(calls) == 2
    assert calls[0][0][:2] == ["meson", "setup"]
    assert calls[1][0][:3] == ["meson", "introspect", "--compilers"]
    assert all(kwargs["timeout"] == 120 for _, kwargs in calls)
    assert all(kwargs["env"]["PATH"] == "/captured/compiler/bin" for _, kwargs in calls)
    assert all("FC" not in kwargs["env"] for _, kwargs in calls)


@pytest.mark.parametrize(
    "output",
    [
        b"not JSON",
        b"{}",
        b'{"host":{"fortran":{"exelist":[]}}}',
        b'{"host":{"fortran":{"exelist":"target-fc"}}}',
        b'{"host":{"fortran":{"exelist":[null]}}}',
    ],
)
def test_malformed_meson_compiler_result_is_an_error(monkeypatch, output):
    monkeypatch.delenv("FC", raising=False)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, output, b"")

    monkeypatch.setattr(_specification_compiler.subprocess, "run", run)
    with _compiler_context.generation_context(discover=True) as context:
        with pytest.raises(_compiler_context.CompilerExecutionError):
            context.resolve()
        with pytest.raises(_compiler_context.CompilerExecutionError):
            context.resolve()
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["setup", "introspect"])
def test_meson_detection_failure_is_not_a_compiler_fallback(monkeypatch, failure):
    monkeypatch.delenv("FC", raising=False)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, int(command[1] == failure), b"", b"rejected"
        )

    monkeypatch.setattr(_specification_compiler.subprocess, "run", run)
    with _compiler_context.generation_context(discover=True) as context:
        with pytest.raises(_compiler_context.CompilerExecutionError, match="rejected"):
            bool(context)
    assert len(calls) == (1 if failure == "setup" else 2)


def test_explicit_fc_does_not_run_meson_detection(monkeypatch):
    monkeypatch.setenv("FC", "chosen-fc --selected")

    def run(*args, **kwargs):
        raise AssertionError("explicit FC reached compiler discovery")

    monkeypatch.setattr(_specification_compiler.subprocess, "run", run)
    with _compiler_context.generation_context(discover=True) as context:
        assert context.command == ("chosen-fc", "--selected")
        assert bool(context)


@pytest.mark.parametrize("search_path", [None, ""])
def test_auto_selection_distinguishes_absent_and_empty_path(monkeypatch, search_path):
    monkeypatch.delenv("FC", raising=False)
    if search_path is None:
        monkeypatch.delenv("PATH", raising=False)
    else:
        monkeypatch.setenv("PATH", search_path)
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs["env"]["PATH"])
        return subprocess.CompletedProcess(
            command, 0, b'{"host":{"fortran":{"exelist":["chosen-fc"]}}}', b""
        )

    monkeypatch.setattr(_specification_compiler.subprocess, "run", run)
    with _compiler_context.generation_context(discover=True) as context:
        assert bool(context)
        assert context.search_path == (os.defpath if search_path is None else "")
    assert calls == [os.defpath if search_path is None else ""] * 2
