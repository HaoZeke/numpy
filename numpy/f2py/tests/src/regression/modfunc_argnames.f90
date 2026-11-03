module argnames
  implicit none
contains
  function addf(xaddf, n) result(r)
    integer, intent(in) :: xaddf, n
    integer :: r
    r = xaddf + n
  end function addf

  function three() result(r)
    integer :: r
    r = 3
  end function three
end module argnames
