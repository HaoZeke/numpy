module test2
  use test
  integer, parameter :: b = 2
end module test2

module renamed
  use test, only: alias => a
  implicit none
end module renamed

module restricted
  use test
  implicit none
  private
  public :: counter
end module restricted
