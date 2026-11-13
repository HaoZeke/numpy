      subroutine foo(fun, r)
      external fun
      integer i
      real*8 r, fun
Cf2py intent(out) r
      r = 0d0
      do i=-5,5
         r = r + fun(i)
      enddo
      end
