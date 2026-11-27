import pytest

from . import util


@pytest.mark.slow
class TestCallbackArrayReturn(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "callback", "arrreturn.f90"),
        util.getpath("tests", "src", "callback", "arrreturn.pyf"),
    ]
    module_name = "cbarr"

    def test_fixed_shape(self):
        seen = []

        def cb(n):
            seen.append(n)
            return [n + 1, n + 2]

        out = self.module.run(cb, 7)
        assert seen == [7]
        assert out.tolist() == [8, 9]

    def test_extent_from_argument(self):
        def cbn(n):
            return [n + i for i in range(n)]

        out = self.module.runn(cbn, 4)
        assert out.tolist() == [4, 5, 6, 7]

    def test_rank2(self):
        def cb2(n):
            return [[n, n + 1, n + 2], [n + 3, n + 4, n + 5]]

        out = self.module.fill(cb2, 10)
        assert out.tolist() == [[10, 11, 12], [13, 14, 15]]
