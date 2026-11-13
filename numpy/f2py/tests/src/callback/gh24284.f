      subroutine parent(x, fun)
      double precision x
      external fun
cf2py intent(out) x
      call child(x, fun)
      end

      subroutine child(x, fun)
      double precision x
      external fun
cf2py intent(out) x
      call fun(x)
      end
