module useonly
  use srcmod, only: n
  implicit none
  integer :: local(n)
contains
  subroutine fill()
    local = n
  end subroutine fill
end module useonly

module userename
  use srcmod, only: m => n
  implicit none
  integer, parameter :: n = 2
  integer :: renamed(m)
  integer :: shadowed(n)
contains
  subroutine fill()
    renamed = m
    shadowed = n
  end subroutine fill
end module userename
