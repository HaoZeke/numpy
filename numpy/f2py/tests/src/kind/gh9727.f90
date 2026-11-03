subroutine f_get_pi(pi)
    real(KIND=16), intent(out) :: pi
    pi = 4._16*ATAN(1._16)
end subroutine

subroutine f_deg2rad(deg, rad)
    real*16, intent(in) :: deg
    real*16, intent(out) :: rad
    !
    real*16 :: pi
    !
    CALL f_get_pi(pi)
    rad = pi/180._16 * deg
end subroutine
