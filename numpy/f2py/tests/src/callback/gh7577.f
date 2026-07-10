      subroutine complex_cb_test(callback, z, r)
      external callback
      complex z, r, callback
cf2py  intent(callback) callback
cf2py  intent(in) z
cf2py  intent(out) r
      r = callback(z)
      end