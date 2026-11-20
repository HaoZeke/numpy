      subroutine dps(a)
cf2py intent(out) a
      doubleprecision
     &a
      a = 5.5d0
      end

      subroutine dcs(z)
cf2py intent(out) z
      doublecomplex
     &z
      z = (1d0, 2d0)
      end

      pure doubleprecision
     &function dpf(x)
      doubleprecision, intent(in) :: x
      dpf = 2d0 * x
      end
