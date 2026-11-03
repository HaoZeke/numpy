! Assumed-shape arguments make f2py write F90 wrappers with an explicit
! interface; CONTAINS procedures must not appear in it.
module gh20103_consts
  implicit none
  real(8) :: a = 2.0d0
end module gh20103_consts

subroutine example(array)
  implicit none
  real(8), intent(inout) :: array(:, :)
  integer :: i, j
  do j = 1, size(array, 2)
    do i = 1, size(array, 1)
      array(i, j) = 10 * i + j
    end do
  end do
  call tentimes(array)
contains
  subroutine tentimes(array)
    real(8), intent(inout) :: array(:, :)
    array = array * 10
  end subroutine tentimes
end subroutine example

function total(x) result(s)
  implicit none
  real(8), intent(in) :: x(:)
  real(8) :: s
  s = twice(sum(x))
contains
  function twice(v) result(w)
    real(8), intent(in) :: v
    real(8) :: w
    w = 2 * v
  end function twice
end function total

module gh20103_mod
  implicit none
contains
  subroutine scale(a)
    real(8), intent(inout) :: a(:)
    call inner(a)
  contains
    ! The USE makes gh20103_consts' `a` visible here, which would clash
    ! with the dummy `a` if it were copied into the wrapper.
    subroutine inner(b)
      use gh20103_consts
      real(8), intent(inout) :: b(:)
      b = b * a
    end subroutine inner
  end subroutine scale
end module gh20103_mod
