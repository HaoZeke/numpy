module unwrappable
  implicit none
  real, target :: storage(3)
  real, pointer :: view(:)
  integer, pointer :: single
  ! f2py cannot evaluate log_gamma, but the compiler can
  real :: sized(nint(log_gamma(5.0)))
  integer :: plain = 7
contains
  subroutine attach()
    storage(1) = 1.0
    storage(2) = 2.0
    storage(3) = 3.0
    view => storage
  end subroutine attach
end module unwrappable
