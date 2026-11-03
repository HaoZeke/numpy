module storage_args
  use iso_c_binding, only: c_int
  implicit none

  type :: item
    integer :: n = 0
  end type item

  type, bind(c) :: pair
    integer(c_int) :: a = 0, b = 0
  end type pair

  type(item), target, save :: pool(3)
  type(pair), target, save :: pairs(2)

contains

  subroutine make(x, n)
    type(item), allocatable, intent(out) :: x
    integer, intent(in) :: n
    if (n > 0) then
      allocate(x)
      x%n = n
    end if
  end subroutine make

  subroutine bump(x)
    type(item), allocatable, intent(inout) :: x
    if (allocated(x)) x%n = x%n + 1
  end subroutine bump

  subroutine pick(p, i)
    type(item), pointer, intent(out) :: p
    integer, intent(in) :: i
    nullify(p)
    if (i > 0) then
      pool(i)%n = 10 * i
      p => pool(i)
    end if
  end subroutine pick

  function look(p) result(n)
    type(item), pointer, intent(in) :: p
    integer :: n
    n = -1
    if (associated(p)) n = p%n
  end function look

  function at(i) result(p)
    integer, intent(in) :: i
    type(item), pointer :: p
    nullify(p)
    if (i > 0) p => pool(i)
  end function at

  subroutine pick_pair(p, i)
    type(pair), pointer, intent(out) :: p
    integer, intent(in) :: i
    nullify(p)
    if (i > 0) then
      pairs(i) = pair(i, -i)
      p => pairs(i)
    end if
  end subroutine pick_pair

  subroutine grow(v, m)
    type(item), allocatable, intent(inout) :: v(:)
    integer, intent(in) :: m
    v%n = v%n + m
  end subroutine grow

  subroutine regrow(v)
    type(item), allocatable, intent(inout) :: v(:)
    deallocate(v)
    allocate(v(1))
    v%n = 99
  end subroutine regrow

  function total(v) result(s)
    type(item), pointer, intent(in) :: v(:)
    integer :: s
    s = sum(v%n)
  end function total

end module storage_args
