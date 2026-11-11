C gh-13356: ENTRY must be wrapped, not crash.
      DOUBLE PRECISION FUNCTION ETA0(TK)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      ETA0 = TK
      RETURN
      ENTRY ETA1(TK, KK)
      ETA1 = TK + KK
      RETURN
      END

      REAL FUNCTION HOST(X)
      REAL X
      HOST = X
      RETURN
      ENTRY IADD(X)
      IADD = 3
      RETURN
      END
