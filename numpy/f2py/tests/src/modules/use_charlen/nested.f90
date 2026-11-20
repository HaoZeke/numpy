module p_mid
  use p
  integer, parameter :: twice = 2 * n
end module

module u_nested
  use p_mid
  character(len=n) :: reexported = "hello"
  character(len=twice) :: derived = "0123456789"
end module
