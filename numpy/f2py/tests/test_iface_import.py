import pytest

import numpy as np
from numpy.f2py import crackfortran

from . import util


@pytest.mark.slow
class TestIfaceImport(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "iface_import", "apply.f90"),
        util.getpath("tests", "src", "iface_import", "csum.c"),
    ]

    @pytest.mark.parametrize("name", ["apply_renamed", "apply_named"])
    def test_c_sum(self, name):
        x = np.array([1, 2, 3, 4], dtype=np.int32)
        assert getattr(self.module, name)(x) == 10


def test_import_only_clause(tmp_path):
    # gfortran 13 parses "import, only:" as a name list, so this checks
    # the regenerated text rather than compiling the clause.
    src = tmp_path / "import_only.f90"
    src.write_text(
        "subroutine s\n"
        "  interface\n"
        "    subroutine g\n"
        "      import, only: a\n"
        "    end subroutine g\n"
        "  end interface\n"
        "end subroutine s\n"
    )
    text = crackfortran.crack2fortran(crackfortran.crackfortran([str(src)]))
    assert "import, only: a" in text
    assert "import ," not in text
