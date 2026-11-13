from numpy.f2py import crackfortran


def test_kinded_upper_bound(tmp_path):
    # 2_int_hash is the integer 2 when int_hash is not a number yet.
    src = tmp_path / "sizes.f90"
    src.write_text(
        "module m\n"
        "  integer, parameter :: int_hash = int32\n"
        "  integer :: a(2_int_hash)\n"
        "end module m\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    found = []

    def walk(block):
        if isinstance(block, list):
            for item in block:
                walk(item)
            return
        if not isinstance(block, dict):
            return
        if block.get("name") == "m" and block.get("block") == "module":
            found.append(block)
        for child in block.get("body") or []:
            walk(child)

    walk(blocks)
    assert found
    assert found[0]["vars"]["a"]["dimension"] == ["2"]
