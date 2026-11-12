import pytest

from numpy.f2py import crackfortran

from . import util


def _blocks(path):
    return crackfortran.crackfortran([str(path)])


def _walk(blocks):
    found = []

    def visit(block):
        if isinstance(block, dict) and "block" in block:
            found.append((block.get("block"), block.get("name")))
            for child in block.get("body") or []:
                visit(child)
        elif isinstance(block, list):
            for child in block:
                visit(child)

    visit(blocks)
    return found


def test_fixed_form_module_contains_its_function():
    # gh-32360: MODULE in a .f file was not a group opener, but
    # ENDMODULE closed whatever group was open.
    path = util.getpath("tests", "src", "crackfortran", "gh32360_mod.f")
    found = _walk(_blocks(path))
    assert ("module", "side_a_mod") in found
    assert ("function", "addone") in found
    assert ("program", "gh32360_mod") not in found


def test_fixed_form_interface_stays_inside_the_subroutine():
    path = util.getpath("tests", "src", "crackfortran", "gh32360_iface.f")
    found = _walk(_blocks(path))
    assert found[0] == ("subroutine", "run")
    assert ("interface", "unknown_interface") in found
    assert ("subroutine", "f") in found


def test_fixed_form_type_star_is_not_a_derived_type():
    path = util.getpath("tests", "src", "crackfortran", "gh32360_typeio.f")
    found = _walk(_blocks(path))
    assert found == [("subroutine", "show")]


@pytest.mark.slow
class TestFixedFormModule(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "gh32360_mod.f")]

    def test_addone(self):
        assert self.module.side_a_mod.addone(1.0) == 2.0
