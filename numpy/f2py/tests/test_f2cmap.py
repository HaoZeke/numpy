import sys

import pytest

import numpy as np
from numpy.f2py.auxfuncs import process_f2cmap_dict
from numpy.f2py.capi_maps import c2py_map

from . import util


def test_c_type_spelling_aliases(capfd):
    # gh-25229: C spellings such as "short int" name types f2py
    # already converts. They used to be ignored.
    new = {
        "integer": {
            "c_short": "short int",
            "c_long": "long int",
            "c_long_long": "long long int",
            "c_schar": "signed char",
            "c_i8": "int8_t",
            "c_i64": "int64_t",
        },
        "real": {"c_ld": "long double"},
    }
    out, mapped = process_f2cmap_dict({}, new, c2py_map, verbose=True)
    assert out["integer"] == {
        "c_short": "short",
        "c_long": "long",
        "c_long_long": "long_long",
        "c_schar": "signed_char",
        "c_i8": "signed_char",
        "c_i64": "long_long",
    }
    assert out["real"]["c_ld"] == "long_double"
    assert "short" in mapped
    err = capfd.readouterr().err
    assert "Ignoring map" not in err

    process_f2cmap_dict({}, {"integer": {"x": "not_a_type"}}, c2py_map, True)
    assert "Ignoring map" in capfd.readouterr().err


def test_short_int_alias_reaches_the_wrapper(tmp_path):
    src = tmp_path / "k.f90"
    src.write_text(
        "subroutine kindof(n)\n"
        "  integer(c_short), intent(out) :: n\n"
        "  n = 1\n"
        "end subroutine\n"
    )
    cmap = tmp_path / "map"
    cmap.write_text("{'integer': {'c_short': 'short int'}}\n")
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "numpy.f2py", str(src), "-m", "kshort",
         "--f2cmap", str(cmap), "--build-dir", str(tmp_path)],
        check=True, cwd=tmp_path, capture_output=True,
    )
    text = (tmp_path / "kshortmodule.c").read_text()
    assert "short *n" in text or "short n" in text


@pytest.mark.slow
class TestF2Cmap(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90"),
        util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap")
    ]

    # gh-15095
    def test_gh15095(self):
        inp = np.ones(3)
        out = self.module.func1(inp)
        exp_out = 3
        assert out == exp_out
