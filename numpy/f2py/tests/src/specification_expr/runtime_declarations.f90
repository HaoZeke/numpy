module declaration_runtime
  implicit none
  integer, parameter :: kind(2) = (/7,9/)
  integer, parameter :: chosen = 1
  real, parameter, private :: threshold = tiny(1.0)
contains
  subroutine shadowed_inquiry(values)
    integer, intent(out) :: values(kind(chosen))
    values = 67
  end subroutine shadowed_inquiry

  function divided_values(n) result(values)
    integer, intent(in) :: n
    integer :: values(0:n/2*2-1)
    values = 61
  end function divided_values

  function character_result(n) result(text)
    integer, intent(in) :: n
    character(len=n/2*2) :: text
    text = repeat('x',n/2*2)
  end function character_result

  subroutine indexed_shape(widths,values)
    integer, intent(in) :: widths(-2:-1)
    integer, intent(out) :: values(widths(-2)/2*2)
    values = 71
  end subroutine indexed_shape

  subroutine selected_axis(array,axis,values)
    integer, intent(in) :: array(-2:,5:), axis
    integer, intent(out) :: values(size(array,dim=axis))
    values = 73
  end subroutine selected_axis

  function bound_result(array,axis) result(values)
    integer, intent(in) :: array(-2:,5:), axis
    integer :: values(max(0,lbound(array,dim=axis)))
    values = 79
  end function bound_result

  subroutine blank_shape(text,values)
    character(len=*), intent(in) :: text
    integer, intent(out) :: values(len_trim(text))
    values = 83
  end subroutine blank_shape

  subroutine signed_shape(n,values)
    integer, intent(in) :: n
    integer, intent(out) :: values(modulo(n,3)+abs(mod(n,3)))
    values = 89
  end subroutine signed_shape

  subroutine conditional_shape(n,values)
    integer, intent(in) :: n
    integer, intent(out) :: values(merge(n,2,n>0))
    values = 97
  end subroutine conditional_shape

  subroutine division_shape(n,divisor,values)
    integer, intent(in) :: n, divisor
    integer, intent(out) :: values(n/divisor)
    values = 101
  end subroutine division_shape

  function private_constant_result(n) result(values)
    integer, intent(in) :: n
    integer :: values(n+merge(1,0,threshold>0.0))
    values = 103
  end function private_constant_result

  subroutine allocate_rank_three(n,total)
    integer, intent(in) :: n
    integer, intent(out) :: total
    integer, allocatable :: work(:,:,:)
    allocate(work(-2:n-3,0:1,-1:1))
    work = 3
    total = sum(work)+size(work)
    deallocate(work)
  end subroutine allocate_rank_three
end module declaration_runtime
