import pytest

from . import util


@pytest.mark.slow
class TestRoutineUsercode(util.F2PyTest):
    suffix = ".pyf"
    module_name = "routine_usercode_mod"
    code = """
python module routine_usercode_mod
interface
  subroutine addone(n)
    fortranname
    integer intent(out) :: n
    usercode '''
    n = 1;
    '''
    usercode '''
    n = 9;
    '''
    callstatement /* leave n */
    callprotoargument void
  end subroutine addone
end interface
end python module routine_usercode_mod
"""

    def test_second_block(self):
        assert self.module.addone() == 9
