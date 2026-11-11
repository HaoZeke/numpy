! gh-21672: character gamma(10)*8 is 10 elements of length 8 (free form)
module gh21672_free
  implicit none
  character gamma(10)*8
  character*8 delta(10)
  character(len=8) :: epsilon(10)
contains
  subroutine store_gamma(inp)
    character inp(10)*8
    intent(in) :: inp
    integer i
    do i = 1, 10
      gamma(i) = inp(i)
    end do
  end subroutine store_gamma

  subroutine store_delta(inp)
    character*8 inp(10)
    intent(in) :: inp
    integer i
    do i = 1, 10
      delta(i) = inp(i)
    end do
  end subroutine store_delta

  subroutine store_epsilon(inp)
    character(len=8) :: inp(10)
    intent(in) :: inp
    integer i
    do i = 1, 10
      epsilon(i) = inp(i)
    end do
  end subroutine store_epsilon

  subroutine echo_entity(gamma_in, echoed)
    character gamma_in(10)*8
    character echoed(10)*8
    intent(in) :: gamma_in
    intent(out) :: echoed
    integer i
    do i = 1, 10
      echoed(i) = gamma_in(i)
    end do
  end subroutine echo_entity
end module gh21672_free
