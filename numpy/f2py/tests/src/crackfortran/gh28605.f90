! gh-28605: typed external with no call and no interface.
real*8 function tcb(p)
  external p
  real*8 p
  tcb = 1.0d0
end function tcb
