import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestAssumedCharArrayCallback(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine feed(cb, fixed, s, n)
  implicit none
  integer, intent(in) :: n
  character(len=3), dimension(n), intent(in) :: fixed
  character(len=*), dimension(n), intent(in) :: s
  external cb
  call cb(fixed, s, n)
end subroutine feed
"""

    @pytest.mark.parametrize("words", [
        (b"ab", b"cd"),
        (b"abcd", b"efgh", b"ijkl"),
    ])
    def test_assumed_length_character_array(self, words):
        seen = []

        def cb(fixed, s, _n):
            seen.append((fixed.copy(), s.copy()))

        fixed = np.full(len(words), b"abc", dtype="S3")
        arr = np.array(words, dtype=f"S{len(words[0])}")
        self.module.feed(cb, fixed, arr, len(words))
        assert seen[0][0].tolist() == fixed.tolist()
        assert seen[0][1].tolist() == arr.tolist()
