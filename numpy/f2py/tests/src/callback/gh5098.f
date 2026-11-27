      subroutine fixed2d(a)
      double precision a(3, 4)
      a(1, 1) = a(1, 1)
      end

      subroutine array_result(array_cb, n, x)
      external array_cb
      integer n
      double precision x(n)
      call array_cb(n, x)
      end
