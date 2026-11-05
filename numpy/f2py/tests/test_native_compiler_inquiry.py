"""Native parser inquiries follow the compiler and keep ordinary integers."""

from types import SimpleNamespace

import pytest

from numpy.f2py import _compiler_context, _compiler_inquiry, crackfortran, f2py2e

from . import util


def test_native_inquiry_keeps_plain_int_and_optional_presence(monkeypatch):
    calls = []

    def query(compiler, flags, p, r, radix):
        calls.append((compiler, flags, p, r, radix))
        return 2

    monkeypatch.setenv("FC", "flang-23")
    monkeypatch.setenv("FFLAGS", "-O2 -Dname='two words'")
    monkeypatch.setattr(_compiler_inquiry, "_selected_real_kind_from_compiler", query)
    result = crackfortran._selected_real_kind_func(0)
    assert type(result) is int
    assert result == 2
    assert crackfortran._selected_real_kind_func(r=400) == 2
    assert crackfortran._selected_real_kind_func(radix=10) == 2
    assert calls == [
        (("flang-23",), ("-O2", "-Dname=two words"), 0, None, None),
        (("flang-23",), ("-O2", "-Dname=two words"), None, 400, None),
        (("flang-23",), ("-O2", "-Dname=two words"), None, None, 10),
    ]


def test_configured_probe_errors_survive_parameter_evaluation(monkeypatch):
    def failed(*args):
        raise _compiler_context.CompilerExecutionError("selected compiler failed")

    monkeypatch.setattr(_compiler_inquiry, "selected_real_kind", failed)
    variables = {"rk": {"typespec": "integer", "attrspec": ["parameter"],
                        "=": "selected_real_kind(0)"}}
    with pytest.raises(_compiler_context.CompilerExecutionError,
                       match="selected compiler failed"):
        crackfortran.get_parameters(variables)


def test_configured_probe_errors_survive_array_parameter_evaluation(monkeypatch):
    def failed(*args):
        raise _compiler_context.CompilerExecutionError("selected compiler failed")

    monkeypatch.setattr(_compiler_inquiry, "selected_real_kind", failed)
    names = {"selected_real_kind": crackfortran._selected_real_kind_func}
    with pytest.raises(_compiler_context.CompilerExecutionError,
                       match="selected compiler failed"):
        crackfortran.param_eval(
            "(/ selected_real_kind(0), selected_real_kind(6) /)",
            names, {}, dimspec="(2)")


def test_unconfigured_inquiry_keeps_legacy_plain_values(monkeypatch):
    monkeypatch.delenv("FC", raising=False)
    for precision, expected in [(0, 4), (6, 4), (15, 8)]:
        result = crackfortran._selected_real_kind_func(precision)
        assert type(result) is int
        assert result == expected


def test_cross_inquiry_requires_its_target_runner(monkeypatch):
    context = _compiler_context.CompilerContext(
        command=("target-fc",), selected=True, needs_runner=True)
    monkeypatch.setattr(_compiler_context, "current", lambda *args: context)
    with pytest.raises(_compiler_context.CompilerExecutionError,
                       match="target runner"):
        crackfortran._selected_real_kind_func(0)


def test_generation_inquiry_receives_cli_fortran_flags(monkeypatch, tmp_path):
    import sys

    received = []

    def query(*args):
        received.append(_compiler_context.current().compile_args)
        return 4

    monkeypatch.setenv("FC", "configured-fc")
    monkeypatch.setenv("FFLAGS", "-O1")
    monkeypatch.setattr(_compiler_inquiry, "_selected_real_kind_from_compiler", query)
    monkeypatch.setattr(f2py2e, "run_main",
                        lambda args: crackfortran._selected_real_kind_func(0))
    monkeypatch.setattr(f2py2e, "f2py_build_generator",
                        lambda name: lambda *args: SimpleNamespace(compile=lambda: None))
    monkeypatch.setattr(sys, "argv", [
        "f2py", "-c", "-m", "inquiry_flags", "--backend", "meson",
        "--build-dir", str(tmp_path), "--f90flags=-fdefault-real-8", "source.f90",
    ])
    f2py2e.run_compile()
    assert received == [("-O1", "-fdefault-real-8")]


@pytest.mark.slow
class TestNativeCompilerInquiry(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "native_compiler_inquiry.f90")]

    @pytest.mark.parametrize("precision, exponent_range, radix", [
        (0, 0, 2), (6, 400, 2), (15, 400, 2),
        (33, 4931, 2), (40, 5000, 2), (0, 0, 10),
    ])
    def test_runtime_intrinsic_matches_native_parser(
        self, precision, exponent_range, radix,
    ):
        expected = self.module.native_compiler_inquiry.query(
            7, precision, exponent_range, radix)
        result = crackfortran._selected_real_kind_func(
            precision, exponent_range, radix)
        assert type(result) is int
        assert result == expected

    @pytest.mark.parametrize("mask, p, r, radix", [
        (1, 0, 0, 0), (2, 0, 400, 0), (3, 6, 400, 0),
        (4, 0, 0, 10), (5, 0, 0, 10), (6, 0, 400, 2), (7, 15, 400, 2),
    ])
    def test_native_optional_argument_presence(self, mask, p, r, radix):
        arguments = {name: value for index, (name, value) in enumerate(
            [("p", p), ("r", r), ("radix", radix)]) if mask & (1 << index)}
        expected = self.module.native_compiler_inquiry.query(mask, p, r, radix)
        result = crackfortran._selected_real_kind_func(**arguments)
        assert type(result) is int
        assert result == expected
