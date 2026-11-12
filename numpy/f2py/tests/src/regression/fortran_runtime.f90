subroutine format_int(n, s)
   ! The internal write calls into the Fortran runtime library
   integer, intent(in) :: n
   character(len=12), intent(out) :: s
   write (s, '(i0)') n
end subroutine format_int
