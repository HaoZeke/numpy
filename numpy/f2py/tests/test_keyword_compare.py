from numpy.f2py import crackfortran
from numpy.f2py.symbolic import Language, fromstring


def test_keyword_value_may_be_a_comparison():
    # mask=1<dim is the keyword mask, and the value is 1<dim.
    expr = fromstring("merge(size(x, 1), size(x, 2), mask=1<dim)")
    text = expr.tostring(language=Language.C).replace(" ", "")
    assert "mask=1<dim" in text


def test_dimension_keyword_comparison(tmp_path):
    src = tmp_path / "mean.f90"
    src.write_text(
        "subroutine mean(x, dim, res)\n"
        "  integer, intent(in) :: dim\n"
        "  real, intent(in) :: x(:,:)\n"
        "  real :: res(merge(size(x, 1), size(x, 2), mask=1<dim))\n"
        "end subroutine mean\n"
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
        if block.get("name") == "mean" and block.get("block") == "subroutine":
            found.append(block)
        for child in block.get("body") or []:
            walk(child)

    walk(blocks)
    assert found
    dim = found[0]["vars"]["res"]["dimension"]
    assert len(dim) == 1
    assert "merge" in dim[0]
    assert "1 < dim" in dim[0] or "1<dim" in dim[0].replace(" ", "")
