! tcb, scb and forward never call their EXTERNAL dummies
real*8 function tcb(p)
  external p
  real*8 p
  tcb = 1.0
end function tcb

subroutine scb(s, n)
  external s
  integer, intent(out) :: n
  n = 2
end subroutine scb

real*8 function callee(f)
  real*8, external :: f
  callee = f()
end function callee

real*8 function forward(g)
  real*8, external :: g, callee
  forward = callee(g)
end function forward
