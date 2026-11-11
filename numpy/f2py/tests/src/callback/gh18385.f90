subroutine primary()
  external callback
  !f2py intent(callback, hide) :: callback
  call callback(1, 2)
end subroutine primary

subroutine secondary()
  external callback
  call callback(3, 4)
end subroutine secondary

subroutine tertiary()
  external callback
  !f2py intent(callback, hide) :: callback
  call callback(5, 6)
end subroutine tertiary
