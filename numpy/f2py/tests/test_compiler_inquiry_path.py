"""Probe executable lookup follows the captured compiler context."""

from types import SimpleNamespace

from numpy.f2py import _compiler_context, _compiler_inquiry, crackfortran


def test_probe_compile_link_and_runner_use_captured_path(monkeypatch, tmp_path):
    captured = str(tmp_path / "captured-bin")
    changed = str(tmp_path / "different-bin")
    monkeypatch.setenv("FC", "f2py-context-fc")
    monkeypatch.setenv("FFLAGS", "-O2")
    monkeypatch.setenv("LDFLAGS", "-lcompanion")
    monkeypatch.setenv("MESON_EXE_WRAPPER", "f2py-context-runner --target")
    monkeypatch.setenv("PATH", captured)
    monkeypatch.setenv("F2PY_CONTEXT_SENTINEL", "preserved")
    commands = []
    owners = []
    original_program = _compiler_inquiry._selected_real_kind_program.__wrapped__

    def build(context, kinds):
        owner, executable = original_program(context, kinds)
        owners.append(owner)
        return owner, executable

    def run(command, **kwargs):
        assert kwargs["env"]["PATH"] == captured
        assert kwargs["env"]["F2PY_CONTEXT_SENTINEL"] == "preserved"
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=b"2\n", stderr=b"")

    monkeypatch.setattr(_compiler_inquiry, "_selected_real_kind_program", build)
    monkeypatch.setattr(_compiler_inquiry.subprocess, "run", run)
    try:
        with _compiler_context.generation_context() as context:
            assert context.search_path == captured
            monkeypatch.setenv("PATH", changed)
            result = crackfortran._selected_real_kind_func(0)
            assert type(result) is int
            assert result == 2
            assert context.search_path == captured
        assert len(commands) == 3
        assert commands[0][0] == "f2py-context-fc"
        assert "-c" in commands[0]
        assert "-O2" in commands[0]
        assert commands[1][0] == "f2py-context-fc"
        assert "-lcompanion" in commands[1]
        assert commands[2][:2] == ["f2py-context-runner", "--target"]
    finally:
        for owner in owners:
            owner.cleanup()
