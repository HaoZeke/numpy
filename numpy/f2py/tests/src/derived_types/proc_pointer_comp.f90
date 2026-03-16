! Test procedure pointer components (F2018 7.5.4.4, R741)
module proc_pointer_mod
  implicit none

  abstract interface
    function real_unary_func(x) result(y)
      real, intent(in) :: x
      real :: y
    end function real_unary_func
  end interface

  type :: with_proc_ptr
    integer :: id = 0
    procedure(real_unary_func), pointer, nopass :: func => null()
  end type with_proc_ptr

contains

  function square_func(x) result(y)
    real, intent(in) :: x
    real :: y
    y = x * x
  end function square_func

  function double_func(x) result(y)
    real, intent(in) :: x
    real :: y
    y = x * 2.0
  end function double_func

  subroutine set_func_square(obj)
    type(with_proc_ptr), intent(inout) :: obj
    obj%func => square_func
  end subroutine set_func_square

  subroutine set_func_double(obj)
    type(with_proc_ptr), intent(inout) :: obj
    obj%func => double_func
  end subroutine set_func_double

  function apply_func(obj, x) result(y)
    type(with_proc_ptr), intent(in) :: obj
    real, intent(in) :: x
    real :: y
    if (associated(obj%func)) then
      y = obj%func(x)
    else
      y = 0.0
    end if
  end function apply_func

end module proc_pointer_mod
