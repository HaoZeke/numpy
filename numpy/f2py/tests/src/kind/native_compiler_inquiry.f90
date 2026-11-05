module native_compiler_inquiry
  implicit none
contains
  subroutine query(mask, p, r, radix, result)
    integer, intent(in) :: mask, p, r, radix
    integer, intent(out) :: result
    select case (mask)
    case (1)
      result = selected_real_kind(p=p)
    case (2)
      result = selected_real_kind(r=r)
    case (3)
      result = selected_real_kind(p=p, r=r)
    case (4)
      result = selected_real_kind(radix=radix)
    case (5)
      result = selected_real_kind(p=p, radix=radix)
    case (6)
      result = selected_real_kind(r=r, radix=radix)
    case (7)
      result = selected_real_kind(p=p, r=r, radix=radix)
    end select
  end subroutine query
end module native_compiler_inquiry
