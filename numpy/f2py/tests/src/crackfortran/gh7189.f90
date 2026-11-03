module minimal
  integer, dimension(2) :: d
  data d(1) / 1 /
  data d(2) / 2 /
end module minimal

subroutine elemdata(out)
  integer, intent(out) :: out(2)
  integer, dimension(2) :: a
  data a(1) / 3 /
  data a(2) / 4 /
  out = a
end subroutine elemdata
