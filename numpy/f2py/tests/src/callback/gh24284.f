      subroutine parent(n, fun)
      integer n
      external fun
      call apply(fun, n)
      end

      subroutine apply(cb, n)
      integer n
      external cb
      call cb(n)
      end

      function finner(cb)
      double precision finner, cb, x
      external cb
      x = 1.0d0
      finner = cb(x)
      end

      function fouter(fun)
      double precision fouter, fun, finner
      external fun
      fouter = finner(fun)
      end
