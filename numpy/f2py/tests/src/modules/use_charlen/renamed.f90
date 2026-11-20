module ur
  use p, only: width => n
  character(len=width) :: name = "hello"
end module

module ur_swap
  use p, m => n, n => m
  character(len=m) :: long_s = "hello"
  character(len=n) :: short_s = "ab"
end module

module ur_twice
  use p, only: n
  use p, only: m
  character(len=n) :: s = "hello"
  character(len=m) :: t = "ab"
end module

module ur_collide
  use p_alt
  use p, only: m
  character(len=n) :: first = "abc"
end module

module ur_collide_rev
  use p, only: m
  use p_alt
  character(len=n) :: last = "abc"
end module
