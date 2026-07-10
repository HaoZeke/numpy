subroutine one(r)
  integer :: r
  external cb
  integer :: cb
  r = cb(1)
end subroutine
subroutine two(r)
  integer :: r
  external cb
  integer :: cb
  r = cb(2) * 10
end subroutine
