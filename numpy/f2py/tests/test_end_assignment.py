from numpy.f2py import crackfortran


def test_assignment_to_end(tmp_path):
    # A variable named end must not close the subroutine.
    src = tmp_path / "bounds.f90"
    src.write_text(
        "subroutine bounds(n, end)\n"
        "  integer, intent(in) :: n\n"
        "  integer, intent(out) :: end\n"
        "  end = n\n"
        "end subroutine bounds\n"
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
        if block.get("name") == "bounds" and block.get("block") == "subroutine":
            found.append(block)
        for child in block.get("body") or []:
            walk(child)

    walk(blocks)
    assert found
    assert found[0]["args"] == ["n", "end"]
