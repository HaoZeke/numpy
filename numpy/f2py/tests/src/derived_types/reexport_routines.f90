module reexport_owner
  use iso_c_binding
  implicit none
  type, bind(c) :: item
    integer(c_int) :: value
  end type
end module

module reexport_other_owner
  use iso_c_binding
  implicit none
  type :: item
    real(c_double) :: value
  end type
end module

module reexport_facade
  use reexport_owner, only: renamed => item
  implicit none
end module

module reexport_second_facade
  use reexport_owner, only: renamed => item
  implicit none
end module

module reexport_chain
  use reexport_facade, only: chain_name => renamed
  implicit none
end module

module reexport_private_facade
  use reexport_owner, only: hidden => item
  implicit none
  private :: hidden
end module

module reexport_direct_consumer
  use reexport_owner, only: direct => item
  implicit none
contains
  function sum_direct(values) result(total)
    type(direct), intent(in) :: values(:)
    integer :: total
    total = sum(values%value)
  end function
end module

module reexport_consumer
  use reexport_facade, only: local => renamed
  implicit none
contains
  function sum_transitive(values) result(total)
    type(local), intent(in) :: values(:)
    integer :: total
    total = sum(values%value)
  end function

  subroutine bump_transitive(values, increment)
    type(local), intent(inout) :: values(:)
    integer, intent(in) :: increment
    values%value = values%value + increment
  end subroutine
end module

module reexport_chain_consumer
  use reexport_chain, only: element => chain_name
  implicit none
contains
  function sum_chain(values) result(total)
    type(element), intent(in) :: values(:, :)
    integer :: total
    total = sum(values%value)
  end function
end module

module reexport_routine_local
  implicit none
contains
  function sum_local(values) result(total)
    use reexport_facade, only: here => renamed
    type(here), intent(in) :: values(:)
    integer :: total
    total = sum(values%value)
  end function
end module

module reexport_two_routes
  use reexport_facade, only: first => renamed
  use reexport_second_facade, only: second => renamed
  implicit none
contains
  function sum_two_routes(a, b) result(total)
    type(first), intent(in) :: a(:)
    type(second), intent(in) :: b(:)
    integer :: total
    total = sum(a%value) + sum(b%value)
  end function
end module
