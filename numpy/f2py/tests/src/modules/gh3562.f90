! gh-3562: parameters from a USEd module should be visible.
module test
  implicit none
  integer, parameter :: a = 1
  integer :: not_param = 5
end module test

module test2
  use test, only: c => a
  implicit none
  integer, parameter :: b = 2
end module test2
