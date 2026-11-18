import textwrap

import pytest

from numpy.f2py import crackfortran


@pytest.mark.parametrize("prefix", ["non_recursive", "pure non_recursive"])
def test_non_recursive_subroutine(tmp_path, monkeypatch, prefix):
    fpath = tmp_path / "nonrec.f90"
    fpath.write_text(textwrap.dedent(f"""
        {prefix} subroutine foo(n)
          integer, intent(inout) :: n
        end subroutine foo
    """))
    monkeypatch.setattr(crackfortran, "verbose", 0)
    post = crackfortran.crackfortran([str(fpath)])
    assert [block["name"] for block in post] == ["foo"]
    foo = post[0]
    assert foo["block"] == "subroutine"
    assert foo["vars"]["n"]["intent"] == ["inout"]
