module block_construct
  implicit none
contains
  subroutine with_block(x, y)
    real(8), intent(in) :: x
    real(8), intent(out) :: y
    block
      real(8) :: t
      t = 2 * x
      y = t + 1
    end block
  end subroutine with_block

  subroutine with_named_block(x, y)
    real(8), intent(in) :: x
    real(8), intent(out) :: y
    outer: block
      real(8) :: t
      t = 3 * x
      y = t
    endblock outer
  end subroutine with_named_block

  subroutine after_blocks(x, y)
    real(8), intent(in) :: x
    real(8), intent(out) :: y
    y = x - 1
  end subroutine after_blocks
end module block_construct

subroutine external_with_block(x, y)
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  block
    y = x + 10
  end block
end subroutine external_with_block

subroutine external_after_block(x, y)
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  y = x + 20
end subroutine external_after_block
