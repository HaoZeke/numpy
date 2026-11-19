import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestCInOut(util.F2PyTest):
    suffix = ".pyf"
    module_name = "c_inout_ext"
    code = """
python module c_inout_ext
    usercode '''
void c_inc(int *n) { *n = *n + 1; }
void c_set(int *n) { *n = 1; }
'''
    interface
        subroutine c_inc(n)
            intent(c) c_inc
            integer, intent(c, inout) :: n
        end subroutine c_inc
        subroutine c_set(n)
            intent(c) c_set
            integer, intent(c, out) :: n
        end subroutine c_set
    end interface
end python module c_inout_ext
"""

    def test_inout_pointer_writeback(self):
        x = np.array(3, dtype=np.intc)
        self.module.c_inc(x)
        assert int(x) == 4

    def test_out_pointer_writeback(self):
        assert self.module.c_set() == 1
