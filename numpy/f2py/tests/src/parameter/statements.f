C     Sizes from parameter statements, which f2py evaluates as Fortran
      subroutine fill(a, b, c)
      parameter (n = 7 / 2, x = 1, m = int(x / 2 * 10))
      parameter (big = huge(1.0), k = kind(big))
      real a(n), b(m), c(k)
Cf2py intent(out) a, b, c
      a = n
      b = m
      c = k
      end
