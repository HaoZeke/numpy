subroutine inc_d(x)
  real(8), intent(inout) :: x
  x = x + 1
end subroutine inc_d

subroutine inc_i8(n)
  integer(8), intent(inout) :: n
  n = n + 1_8
end subroutine inc_i8

subroutine scale_z(z)
  complex(8), intent(inout) :: z
  z = 2 * z
end subroutine scale_z
