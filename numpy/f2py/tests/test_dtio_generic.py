import pytest

from numpy.f2py import crackfortran


@pytest.mark.parametrize("source_spec,spec", [
    ("read(formatted)", "read(formatted)"),
    ("write ( unformatted )", "write(unformatted)"),
])
def test_defined_io_generic_spec(tmp_path, source_spec, spec):
    src = tmp_path / "dtio.f90"
    src.write_text(
        "module m\n"
        f"  interface {source_spec}\n"
        "    module procedure p\n"
        "  end interface\n"
        "end module m\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    iface = blocks[0]["body"][0]
    assert iface["name"] == spec
    assert iface.get("implementedby") == ["p"]
    assert not iface.get("args")
    assert not iface.get("vars")

    signature = tmp_path / "dtio.pyf"
    rendered = crackfortran.crack2fortran(blocks)
    signature.write_text(rendered)
    assert f"interface {spec}" in rendered
    assert "real :: formatted" not in rendered
    assert "real :: unformatted" not in rendered
    crackfortran.crackfortran([str(signature)])
