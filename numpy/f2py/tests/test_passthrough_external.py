import pytest

from . import util


@pytest.mark.slow
class TestPassthroughExternal(util.F2PyTest):
    # gh-24284: an external that is only forwarded still needs the
    # callback typedef of the routine that calls it.
    suffix = ".f"
    code = """\
      subroutine diva(n, divaf)
      integer n
      external divaf
      call divaa(n, divaf)
      end
      subroutine divaa(n, divaf)
      integer n
      external divaf
      call divaf(n)
      end
"""

    def test_forwarded_callback_is_invoked(self):
        seen = []

        def divaf(n):
            seen.append(int(n))

        self.module.diva(3, divaf)
        assert seen == [3]
