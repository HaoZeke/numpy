subroutine run(cb, n, out)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: out(2)
  interface
    function cb(n) result(r)
      integer, intent(in) :: n
      integer :: r(2)
    end function cb
  end interface
  out = cb(n)
end subroutine run

subroutine runn(cbn, n, out)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: out(n)
  interface
    function cbn(n) result(r)
      integer, intent(in) :: n
      integer :: r(n)
    end function cbn
  end interface
  out = cbn(n)
end subroutine runn

subroutine fill(cb2, n, out)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: out(2, 3)
  interface
    function cb2(n) result(r)
      integer, intent(in) :: n
      integer :: r(2, 3)
    end function cb2
  end interface
  out = cb2(n)
end subroutine fill
