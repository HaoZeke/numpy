"""The public generation and compilation entrypoints share wrapper generation."""

import copy
import sys
import textwrap

import pytest

from numpy.f2py import f2py2e


def _write_source(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="ascii")
    return str(path)


def _generated_files(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir()
            if path.is_file()}


def _generate_pair(tmp_path, monkeypatch, capfd, sources, main_flags=(),
                   compile_flags=None, module="shared", backend="meson"):
    calls = []
    keys = []

    class RecordingBackend:
        def __init__(self, *args):
            self.args = args

        def compile(self):
            calls.append(self.args)

    def factory(key):
        keys.append(key)
        return RecordingBackend

    monkeypatch.setattr(f2py2e, "f2py_build_generator", factory)
    main_dir = tmp_path / "generate"
    compile_dir = tmp_path / "compile"
    main_dir.mkdir()
    compile_dir.mkdir()
    monkeypatch.chdir(main_dir)
    result = f2py2e.run_main([*main_flags, "-m", module, *sources])
    main_output = capfd.readouterr().out
    generated = _generated_files(main_dir)
    assert calls == []
    monkeypatch.chdir(compile_dir)
    flags = main_flags if compile_flags is None else compile_flags
    build_dir = str(tmp_path / "backend-build")
    monkeypatch.setattr(sys, "argv", ["f2py", "-c", "--backend", backend,
                                    "--build-dir", build_dir, "-m", module,
                                    *flags, *sources])
    assert f2py2e.run_compile() is None
    compile_output = capfd.readouterr().out
    assert keys == ["meson"]
    assert len(calls) == 1
    assert calls[0][3] == build_dir
    assert _generated_files(compile_dir) == generated
    return result, generated, calls[0], main_output, compile_output


@pytest.mark.parametrize("backend", ["meson", "distutils"])
def test_pyf_is_authoritative_for_both_entrypoints(
        tmp_path, monkeypatch, capfd, backend):
    source = _write_source(tmp_path, "implementation.f90", """
        subroutine exposed(value)
          integer, intent(out) :: value
          value = 1
        end subroutine exposed
        subroutine hidden(value)
          integer, intent(out) :: value
          value = 2
        end subroutine hidden
    """)
    signature = _write_source(tmp_path, "interface.pyf", """
        python module declared
          interface
            subroutine exposed(value)
              integer intent(out) :: value
            end subroutine exposed
          end interface
        end python module declared
    """)
    result, generated, build, main_output, compile_output = _generate_pair(
        tmp_path, monkeypatch, capfd, [source, signature],
        module="ignored", backend=backend)
    assert set(result) == {"declared"}
    assert build[0] == "declared"
    assert build[1] == [signature, source]
    assert "declaredmodule.c" in generated
    assert not any("ignored" in name or "untitled" in name for name in generated)
    for output in (main_output, compile_output):
        assert 'Constructing wrapper function "exposed"' in output
        assert 'Constructing wrapper function "hidden"' not in output


def test_pyf_callback_contract_is_shared(tmp_path, monkeypatch, capfd):
    source = _write_source(tmp_path, "callback.f90", """
              subroutine foo(f,r)
              external f
              real*8 f,r
              r = f(1)
              end
    """)
    signature = _write_source(tmp_path, "callback.pyf", """
        python module __user__routines
          interface
            function fun(i) result(r)
              integer :: i
              real*8 :: r
            end function fun
          end interface
        end python module __user__routines
        python module callback2
          interface
            subroutine foo(f,r)
              use __user__routines, f=>fun
              external f
              real*8 intent(out) :: r
            end subroutine foo
          end interface
        end python module callback2
    """)
    result, generated, build, _, _ = _generate_pair(
        tmp_path, monkeypatch, capfd, [signature, source], module="ignored")
    assert set(result) == {"callback2"}
    assert build[0] == "callback2"
    assert build[1] == [signature, source]
    c_source = generated["callback2module.c"].decode()
    assert "cb_fun_in___user__routines_typedef" in c_source
    assert "cb_fun_in_foo__user__routines_typedef" not in c_source


@pytest.mark.parametrize("main_flags,compile_flags,routine", [
    ([], ["--no-lower"], "MIXED"),
    (["--lower"], [], "mixed"),
    (["--lower"], ["--lower"], "mixed"),
    (["--no-lower"], ["--no-lower"], "MIXED"),
    (["--lower", "--no-lower"], ["--lower", "--no-lower"], "MIXED"),
    (["--no-lower", "--lower"], ["--no-lower", "--lower"], "mixed"),
])
def test_generation_case_policy(tmp_path, monkeypatch, capfd, main_flags,
                                compile_flags, routine):
    source = _write_source(tmp_path, "case.f90", """
        subroutine MIXED(value)
          integer, intent(out) :: value
          value = 1
        end subroutine MIXED
    """)
    _, _, build, main_output, compile_output = _generate_pair(
        tmp_path, monkeypatch, capfd, [source], main_flags, compile_flags)
    assert build[1] == [source]
    for output in (main_output, compile_output):
        assert f'Constructing wrapper function "{routine}"' in output


def test_generation_flags_are_not_backend_source_files(
        tmp_path, monkeypatch, capfd):
    include_dir = tmp_path / "includes"
    include_dir.mkdir()
    (include_dir / "value.inc").write_text(
        "integer, intent(out) :: value\n", encoding="ascii")
    source = _write_source(tmp_path, "include.f90", """
        subroutine provided(value)
          include 'value.inc'
          value = 1
        end subroutine provided
    """)
    mapping = _write_source(tmp_path, "kinds.map", "{}\n")
    flags = ["--lower", "--debug-capi", "--skip-empty-wrappers",
             "--no-wrap-functions", "--no-freethreading-compatible",
             "--rest-doc", "--latex-doc", "--short-latex",
             "-include<stdbool.h>", "--f2cmap", mapping,
             "--include-paths", str(include_dir)]
    _, generated, build, main_output, compile_output = _generate_pair(
        tmp_path, monkeypatch, capfd, [source], flags)
    assert build[1] == [source]
    assert build[4] == [str(include_dir)]
    assert build[9][build[9].index("--f2cmap") + 1] == mapping
    assert "#include <stdbool.h>" in generated["sharedmodule.c"].decode()
    assert "sharedmodule.rst" in generated
    assert "sharedmodule.tex" in generated
    assert "\\documentclass" not in generated["sharedmodule.tex"].decode()
    for output in (main_output, compile_output):
        assert f"Reading f2cmap from '{mapping}'" in output


@pytest.mark.parametrize("flags", [
    ["--quiet", "--verbose"], ["--verbose", "--quiet"],
])
def test_generation_verbosity_order_is_shared(
        tmp_path, monkeypatch, capfd, flags):
    source = _write_source(tmp_path, "verbosity.f90", """
        subroutine verbosity()
        end subroutine verbosity
    """)
    _, _, _, main_output, compile_output = _generate_pair(
        tmp_path, monkeypatch, capfd, [source], ["--lower", *flags])
    expected = flags[-1] == "--verbose"
    assert ('Constructing wrapper function "verbosity"' in main_output) == expected
    assert ('Constructing wrapper function "verbosity"' in compile_output) == expected


@pytest.mark.parametrize("flags,routine", [([], "mixed"), (["--no-lower"], "MIXED")])
def test_signature_generation_keeps_its_case_policy(
        tmp_path, monkeypatch, flags, routine):
    source = _write_source(tmp_path, "signature.f90", """
        subroutine MIXED(value)
          integer, intent(out) :: value
          value = 1
        end subroutine MIXED
    """)
    signature = tmp_path / "generated.pyf"
    monkeypatch.chdir(tmp_path)
    result = f2py2e.run_main(["-h", str(signature), "-m", "declared", *flags, source])
    assert result is None
    generated = signature.read_text()
    assert "python module declared" in generated
    assert f"subroutine {routine}(" in generated
    assert not list(tmp_path.glob("*module.c"))


@pytest.mark.parametrize("compile_mode", [False, True])
def test_multiple_signatures_fail_before_backend_and_restore_settings(
        tmp_path, monkeypatch, compile_mode):
    signatures = [_write_source(tmp_path, name + ".pyf", f"""
        python module {name}
          interface
          end interface
        end python module {name}
    """) for name in ("first", "second")]

    class UnexpectedBackend:
        def __init__(self, *args):
            raise AssertionError("Generation must reject multiple signatures")

    monkeypatch.setattr(f2py2e, "f2py_build_generator",
                        lambda key: UnexpectedBackend)
    monkeypatch.chdir(tmp_path)
    before = [(module, name, copy.deepcopy(getattr(module, name)))
              for module, names in f2py2e._RUN_SETTINGS for name in names]
    with pytest.raises(ValueError, match="Only one .pyf file per call"):
        if compile_mode:
            monkeypatch.setattr(sys, "argv", ["f2py", "-c", "--backend", "meson",
                                            "--build-dir", str(tmp_path / "build"),
                                            "-m", "ignored", *signatures])
            f2py2e.run_compile()
        else:
            f2py2e.run_main(["-m", "ignored", *signatures])
    assert not list(tmp_path.glob("*module.c"))
    for module, name, value in before:
        assert getattr(module, name) == value
