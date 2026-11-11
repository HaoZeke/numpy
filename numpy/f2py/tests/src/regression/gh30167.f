      subroutine scale(n, x, factor)
      integer n, i
      double precision x(n), factor
      do i = 1, n
         x(i) = factor*x(i)
      end do
      end

      subroutine apply(fun, n, x)
      external fun
      integer n, i
      double precision x(n), fun
      do i = 1, n
         x(i) = fun(x(i))
      end do
      end
