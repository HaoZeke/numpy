function apply_renamed(x) result(y)
  use iso_c_binding, only: ci => c_int
  integer(ci), intent(in) :: x(:)
  integer(ci) :: y
  interface
    integer(ci) function c_sum(n, x) bind(c, name="c_sum")
      import :: ci
      integer(ci), value :: n
      integer(ci), intent(in) :: x(*)
    end function c_sum
  end interface
  y = c_sum(size(x), x)
end function apply_renamed

function apply_named(x) result(y)
  use iso_c_binding, only: c_int
  integer(c_int), intent(in) :: x(:)
  integer(c_int) :: y
  interface
    integer(c_int) function c_sum(n, x) bind(c, name="c_sum")
      import :: c_int
      integer(c_int), value :: n
      integer(c_int), intent(in) :: x(*)
    end function c_sum
  end interface
  y = c_sum(size(x), x)
end function apply_named
