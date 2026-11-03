! Host-side code, built into a library before the f2py module (gh-31249).
! gh31249_cb is the intent(callback) wrapper defined by the f2py module.
subroutine gh31249_apply(n)
    integer, intent(in) :: n
    external gh31249_cb
    call gh31249_cb(n)
end subroutine gh31249_apply
