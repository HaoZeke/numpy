module scalar_result_aliases
  use iso_c_binding, only: c_bool
contains
  logical(kind=c_bool) function true_result()
    true_result = .true.
  end function true_result

  logical(kind=c_bool) function false_result()
    false_result = .false.
  end function false_result
end module scalar_result_aliases
