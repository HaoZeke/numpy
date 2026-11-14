import pytest

from numpy.f2py import crackfortran

from . import util


# Six leading spaces: fixed form. Dedent would put MODULE in column 1.
FIXED_MODULE = """\
      MODULE M
      CONTAINS
      SUBROUTINE ADD(A, B, C)
      INTEGER, INTENT(IN) :: A, B
      INTEGER, INTENT(OUT) :: C
      C = A + B
      END
      END MODULE
"""


def _blocks(post):
    found = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        found.append((node.get('block'), node.get('name')))
        walk(node.get('body') or [])

    walk(post)
    return found


class TestFixedFormModuleCrack:
    def test_endmodule_closes_the_module_it_opened(self, tmp_path):
        path = tmp_path / "m.f"
        path.write_text(FIXED_MODULE)
        blocks = _blocks(crackfortran.crackfortran([str(path)]))
        assert ('module', 'm') in blocks
        assert ('subroutine', 'add') in blocks
        # The subroutine belongs to the module, so the wrapper calls
        # the module procedure rather than an external ``add_``.
        modules = [b for b in blocks if b[0] == 'module']
        assert modules == [('module', 'm')]


@pytest.mark.slow
class TestFixedFormModule(util.F2PyTest):
    suffix = ".f"
    code = FIXED_MODULE

    def test_module_procedure_is_callable(self):
        assert self.module.m.add(1, 2) == 3
