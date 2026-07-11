! gh-4013: present() must see the truth for omitted optional args
module gh4013
contains
  real function foo(x)
    real, intent(in), optional :: x
    if (present(x)) then
      foo = x
    else
      foo = 1
    end if
  end function foo
end module gh4013
