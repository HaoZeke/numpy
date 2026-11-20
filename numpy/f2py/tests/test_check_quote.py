import pytest

from numpy.f2py.tests import util

SOURCE = """\
subroutine quoted(s)
  character(len=*) :: s
!f2py intent(in) s
!f2py check(s[0] == "a"[0]) s
end subroutine quoted

subroutine backslash(s)
  character(len=*) :: s
!f2py intent(in) s
!f2py check(s[0] == "\\\\"[0]) s
end subroutine backslash

subroutine percent_str(s)
  character(len=*) :: s
!f2py intent(in) s
!f2py check(s[0] == "%s"[0]) s
end subroutine percent_str

subroutine percent_scalar(n)
  integer, intent(in) :: n
!f2py check(n % 2 == 0) n
end subroutine percent_scalar

subroutine comment_close(s)
  character(len=*) :: s
!f2py intent(in) s
!f2py check(s[0] == "*/"[0]) s
end subroutine comment_close
"""


@pytest.mark.slow
def test_check_with_quote():
    mod = util.build_code(SOURCE, suffix=".f90", module_name="check_quote")
    err = mod._check_quote_error
    mod.quoted("ab")
    with pytest.raises(err, match=r'\(s\[0\] == "a"\[0\]\) failed for'):
        mod.quoted("ba")
    mod.backslash("\\x")
    with pytest.raises(err, match=r'\(s\[0\] == "\\\\"\[0\]\) failed for'):
        mod.backslash("ab")
    mod.percent_str("%x")
    with pytest.raises(err, match=r'\(s\[0\] == "%s"\[0\]\) failed for'):
        mod.percent_str("ab")
    mod.percent_scalar(4)
    with pytest.raises(err, match=r"\(n % 2 == 0\) failed for"):
        mod.percent_scalar(3)
    mod.comment_close("*x")
    with pytest.raises(err, match=r'\(s\[0\] == "\*/"\[0\]\) failed for'):
        mod.comment_close("ab")
