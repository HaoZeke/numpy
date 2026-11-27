import pytest

from . import util


@pytest.mark.slow
class TestBindResult(util.F2PyTest):
    suffix = ".f90"
    code = """
module bindres
  implicit none
contains
  function addbad(x) bind(c) result(r)
    integer, intent(in) :: x
    integer :: r
    r = x + 2
  end function addbad

  function addok(x) result(r) bind(c)
    integer, intent(in) :: x
    integer :: r
    r = x + 1
  end function addok

  function addsp(x) bind (c) result(r)
    integer, intent(in) :: x
    integer :: r
    r = x + 3
  end function addsp

  function addnm(x) bind(c, name="addnm") result(r)
    integer, intent(in) :: x
    integer :: r
    r = x + 4
  end function addnm
end module bindres
"""

    def test_bind_before_result(self):
        fn = self.module.bindres.addbad
        got = fn(5)
        assert got == 7
        assert type(got) is int
        assert "float" not in fn.__doc__.split("Returns", 1)[1]

    def test_result_before_bind(self):
        fn = self.module.bindres.addok
        got = fn(5)
        assert got == 6
        assert type(got) is int

    def test_spaced_bind_before_result(self):
        fn = self.module.bindres.addsp
        got = fn(5)
        assert got == 8
        assert type(got) is int

    def test_named_bind_before_result(self):
        fn = self.module.bindres.addnm
        got = fn(5)
        assert got == 9
        assert type(got) is int
