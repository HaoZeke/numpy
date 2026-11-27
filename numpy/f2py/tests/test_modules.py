import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from numpy.f2py import crackfortran, f90mod_rules

from . import util


def _gh32148_source():
    return Path(__file__).resolve().parent / "src" / "modules" / "gh32148.f90"


def _continue_free_form(text):
    # Match the line breaks f2py writes into the f90 wrapper.
    out = []
    for line in text.split("\n"):
        if len(line) > 72 and line.startswith(" "):
            out.append(line[:72] + "&")
            rest = line[72:]
            while len(rest) > 66:
                out.append("     &" + rest[:66] + "&")
                rest = rest[66:]
            out.append("     &" + rest)
        else:
            out.append(line)
    return "\n".join(out)


@pytest.mark.slow
class TestModuleFilterPublicEntities(util.F2PyTest):
    sources = [
        util.getpath(
            "tests", "src", "modules", "gh26920",
            "two_mods_with_one_public_routine.f90"
        )
    ]
    # we filter the only public function mod2
    only = ["mod1_func1", ]

    def test_gh26920(self):
        # if it compiles and can be loaded, things are fine
        pass


@pytest.mark.slow
class TestModuleWithoutPublicEntities(util.F2PyTest):
    sources = [
        util.getpath(
            "tests", "src", "modules", "gh26920",
            "two_mods_with_no_public_entities.f90"
        )
    ]
    only = ["mod1_func1", ]

    def test_gh26920(self):
        # if it compiles and can be loaded, things are fine
        pass


@pytest.mark.slow
class TestModuleDocString(util.F2PyTest):
    sources = [util.getpath("tests", "src", "modules", "module_data_docstring.f90")]

    def test_module_docstring(self):
        assert self.module.mod.__doc__ == textwrap.dedent(
            """\
                     i : 'i'-scalar
                     x : 'i'-array(4)
                     a : 'f'-array(2,3)
                     b : 'f'-array(-1,-1), not allocated
                     foo()\n
                     Wrapper for ``foo``.\n\n"""
        )


@pytest.mark.slow
class TestModuleAndSubroutine(util.F2PyTest):
    module_name = "example"
    sources = [
        util.getpath("tests", "src", "modules", "gh25337", "data.f90"),
        util.getpath("tests", "src", "modules", "gh25337", "use_data.f90"),
        util.getpath("tests", "src", "regression", "datonly.f90"),
    ]

    def test_gh25337(self):
        self.module.data.set_shift(3)
        assert "data" in dir(self.module)

    def test_allocatable_in_dir(self):
        # gh-27696: allocatable arrays should appear in dir()
        names = dir(self.module.datonly)
        assert "data_array" in names
        assert "max_value" in names


@pytest.mark.slow
class TestAllocatableCharacterArray(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "modules", "gh21674_alloc_char.f90")
    ]

    def test_allocatable_char_roundtrip(self):
        # gh-21674: allocatable character arrays should work after allocation
        mod = self.module.alloc_char_mod
        mod.setup_data(3)
        assert mod.charge.shape == (3, 3)
        names = mod.names
        assert names.shape == (3,)
        assert names.dtype.kind == 'U' or names.dtype.kind == 'S'


def test_gh32148_parameter_saved_local(tmp_path):
    # gh-32148: pass a saved local for each module parameter
    source = _gh32148_source()
    blocks = crackfortran.crackfortran([str(source)])
    mod = next(b for b in blocks if b.get("name") == "constants")
    _ret, hook = f90mod_rules.buildhooks(mod)
    call = re.search(r"call f2pysetupfunc\((.*)\)", hook)
    assert call is not None
    args = [item.strip() for item in call.group(1).split(",") if item.strip()]
    assert "constant" not in args
    assert "non_parameter" in args
    assert re.search(r"f2py_par_\d+ = constant\b", hook)
    assert "save ::" in hook
    wrapper = tmp_path / "wrap.f90"
    wrapper.write_text(_continue_free_form(hook))
    if shutil.which("gfortran"):
        syntax = subprocess.run(
            ["gfortran", "-fsyntax-only", str(source), str(wrapper)],
            capture_output=True, text=True,
        )
        assert syntax.returncode == 0, syntax.stderr


@pytest.mark.slow
class TestGh32148Parameters(util.F2PyTest):
    sources = [
        str(Path(__file__).resolve().parent / "src" / "modules" / "gh32148.f90")
    ]

    def test_parameter_values(self):
        mod = self.module.constants
        assert mod.constant == 4
        assert mod.get_constant() == 4
        assert mod.non_parameter == 5
        assert mod.vector.tolist() == [1, 2, 3]
        assert mod.ratio == 1.5
        assert mod.word == b"abc"
        assert mod.names.tolist() == [b"ab", b"cd"]
        assert mod.dp == 8
        assert mod.wide == 2.5
        assert mod.counted == 6
        assert mod.sized == 11
        assert mod.from_c == 7
        assert mod.flag == 1 or mod.flag == True  # noqa: E712
        assert complex(mod.zed) == 1 + 2j


@pytest.mark.slow
class TestUsedModule(util.F2PyTest):
    module_name = "fmath"
    sources = [
        util.getpath("tests", "src", "modules", "use_modules.f90"),
    ]

    def test_gh25867(self):
        compiled_mods = [x for x in dir(self.module) if "__" not in x]
        assert "useops" in compiled_mods
        assert self.module.useops.sum_and_double(3, 7) == 20
        assert "mathops" in compiled_mods
        assert self.module.mathops.add(3, 7) == 10
