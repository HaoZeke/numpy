import pytest

from . import util


@pytest.mark.slow
@pytest.mark.parametrize(
    "parent_name, parent, include_name, include",
    [
        (
            "add.f90",
            "function add(a, b)\n"
            "  include 'kinds.f'\n"
            "  add = a + b\n"
            "end function\n",
            "kinds.f",
            "  integer :: a, b, add\n",
        ),
        (
            "add.f",
            "      function add(a, b)\n"
            "      include 'kinds.f90'\n"
            "      add = a + b\n"
            "      end\n",
            "kinds.f90",
            "      integer a\n"
            "     &,b\n"
            "     &,add\n",
        ),
    ],
)
def test_include_uses_parent_source_form(tmp_path, parent_name, parent,
                                         include_name, include):
    src = tmp_path / parent_name
    (tmp_path / include_name).write_text(include)
    src.write_text(parent)
    mod = util.build_module(
        [str(src)],
        options=[f"-I{tmp_path}", f"--include-paths {tmp_path}"],
    )
    assert mod.add(2, 3) == 5
