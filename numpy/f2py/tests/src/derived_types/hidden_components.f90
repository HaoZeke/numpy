module hidden_components
  implicit none

  type :: node
    integer :: value
    type(node), pointer :: next
  end type node

  type :: holder
    logical :: flag
    logical :: flags(3)
    integer(2) :: small
    complex*16 :: z
    integer(1) :: tiny
    character(len=2) :: codes(2)
    type(node), allocatable :: nodes(:)
    type(node) :: grid(2, 2)
  end type holder

contains

  subroutine push(head, v)
    type(node), intent(inout) :: head
    integer, intent(in) :: v
    type(node), pointer :: p
    allocate(p)
    p%value = v
    p%next => head%next
    head%next => p
  end subroutine push

  function length(head) result(n)
    type(node), intent(in) :: head
    integer :: n
    type(node), pointer :: p
    n = 0
    p => head%next
    do while (associated(p))
      n = n + 1
      p => p%next
    end do
  end function length

  subroutine fill(h, n)
    type(holder), intent(inout) :: h
    integer, intent(in) :: n
    allocate(h%nodes(n))
    h%nodes%value = 1
    h%tiny = int(n, kind(h%tiny))
    h%codes = ['ab', 'cd']
    h%grid%value = n
  end subroutine fill

  function census(h) result(total)
    type(holder), intent(in) :: h
    integer :: total
    total = sum(h%nodes%value) + h%tiny + sum(h%grid%value)
    if (h%flag) total = total + 1000
    if (all(h%codes == ['ab', 'cd'])) total = total + 100
  end function census

end module hidden_components
