module p
  integer, parameter :: ik = selected_int_kind(8)
  integer(kind=ik), parameter :: n = 5_ik
  integer, parameter :: m = 2
end module

module p_alt
  integer, parameter :: n = 3
end module
