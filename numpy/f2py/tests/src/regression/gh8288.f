      subroutine first(pyfunc, x, r)
      external pyfunc
      double precision x, r, pyfunc
      r = pyfunc(x)
      end

      subroutine second(pyfunc, x, r)
      external pyfunc
      double precision x, r, pyfunc
      r = pyfunc(x) * 2.0d0
      end
