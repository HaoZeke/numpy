      subroutine compute(fun, n, x, y)
      external fun
      integer n
      double precision x(n), y(n)
cf2py intent(out) y
      call fun(n, x, y)
      end
