! gh-7577: a dummy procedure with an explicit interface must be a callback,
! and gfortran must return COMPLEX by value.
module gh7577
  implicit none
contains
  function foo(y, func_complex) result(n)
    complex, intent(in) :: y
    complex :: n
    interface
       function func_complex(z) result(f)
         complex :: f
         complex, intent(in) :: z
       end function func_complex
    end interface
    n = func_complex(y)
  end function foo
end module gh7577
