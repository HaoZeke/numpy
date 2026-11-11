      double precision function eta0(tk, x)
      implicit double precision (a-h,o-z)
      dimension x(*)
      eta0 = tk + x(1)
      return
      entry eta1(tk, kk)
      eta1 = tk * kk
      return
      end
