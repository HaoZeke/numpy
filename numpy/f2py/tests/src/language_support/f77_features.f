      SUBROUTINE INCR(X, Y)
      DOUBLE PRECISION X, Y
Cf2py intent(out) y
      Y = X + 1D0
      RETURN
      ENTRY INCR2(X, Y)
      Y = X + 2D0
      RETURN
      END

      SUBROUTINE SQP1(X, Y)
      DOUBLE PRECISION X, Y, F, T
Cf2py intent(out) y
      F(T) = T*T + 1D0
      Y = F(X)
      END

      SUBROUTINE IMPL(X, N, Y)
Cf2py intent(out) y
      Y = X * N
      END

      SUBROUTINE IMPLDP(X, Y)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
Cf2py intent(out) y
      Y = X / 3D0
      END

      SUBROUTINE EQV(X, Y)
      DOUBLE PRECISION X, Y, A(2), B
      EQUIVALENCE (A(1), B)
Cf2py intent(out) y
      A(1) = X
      Y = B
      END

      SUBROUTINE COUNTR(K)
      INTEGER K, N
      SAVE N
      DATA N /0/
Cf2py intent(out) k
      N = N + 1
      K = N
      END
