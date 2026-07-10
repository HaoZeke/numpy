! gh-28605: typed EXTERNAL dummy without a call site
Real*8 function tcb(p)
  external p
  real*8 p,a(6),c(6)
  integer i
  a = [(i, i = 1, 6)]
  c = [(i, i = 1, 6)]
  print *, a
  print *, c
  tcb = 1.0
  return
end function tcb
