      subroutine echo(x, y)
        real(kind=10), intent(in) :: x
        real(kind=8), intent(out) :: y
        y = x + 1
      end
