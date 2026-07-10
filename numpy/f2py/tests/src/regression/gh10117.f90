! gh-10117: numpy bool arrays must pass to logical dummies without a copy
subroutine bool_flip_first(mask, n)
  integer, intent(in) :: n
  logical(kind=1), dimension(n), intent(inout) :: mask
  mask(1) = .not. mask(1)
end subroutine bool_flip_first

function bool_count_true(mask, n) result(c)
  integer, intent(in) :: n
  logical(kind=1), dimension(n), intent(in) :: mask
  integer :: c, i
  c = 0
  do i = 1, n
     if (mask(i)) c = c + 1
  end do
end function bool_count_true
