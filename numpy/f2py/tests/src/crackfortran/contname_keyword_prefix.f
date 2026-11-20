      integer function real
     &x(a)
      integer a
      realx = a + 1
      end

      integer function integer
     &val(a)
      integer a
      integerval = a + 2
      end

      integer function double
     &x(a)
      integer a
      doublex = a + 3
      end

      integer function end
     &val(a)
      integer a
      endval = a + 4
      end

      integer function to
     &do(a)
      integer a
      todo = a + 5
      end

      integer function vec3
     &d(a)
      integer a
      vec3d = a + 6
      end

      integer function farg(real
     &val, do
     &it)
      integer :: real
     &val
      integer a, do
     &it
      a = 0
      farg = realval + doit + a
      end

      function fres(a) result(real
     &res)
      integer a, realres
      realres = a + 7
      end

      subroutine entrbase(a, b)
cf2py intent(out) b
      integer a, b
      b = a + 1
      return
      entry real
C comment between continuations
     &ent(a, b)
      b = a + 8
      return
      end
