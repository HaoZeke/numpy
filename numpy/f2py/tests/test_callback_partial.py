import functools

import pytest

from . import util


def nested_partial(func, /, *args, **keywords):
    # A __dict__ stops the constructor from merging the inner partial.
    inner = functools.partial(func, *args, **keywords)
    inner._keep = 1
    return functools.partial(inner)


class Acc:
    def m(self, a, b):
        return a + b

    def n(self, a, b=0):
        return a + b


@pytest.mark.slow
class TestCallbackPartial(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine call_two(fun, n)
  implicit none
  external fun
  integer :: fun
  integer, intent(out) :: n
  n = fun(1, 2)
end subroutine call_two

subroutine call_none(fun, n)
  implicit none
  external fun
  integer :: fun
  integer, intent(out) :: n
  n = fun()
end subroutine call_none
"""

    @pytest.mark.parametrize(
        "cb,expected",
        [
            (functools.partial(lambda a: 10 + a), 11),
            (functools.partial(lambda k, a: k + a, 10), 11),
            (nested_partial(lambda a, b: a + b, 10), 11),
            (functools.partial(lambda *a: int(sum(a))), 3),
            (functools.partial(lambda a, *rest: a + sum(rest), 10), 13),
            (
                functools.partial(
                    lambda a, b=0, *rest: a + 10 * b + sum(rest), b=5
                ),
                51,
            ),
            (functools.partial(lambda a, b, c=0: a + b + c, b=5), 6),
            (
                functools.partial(
                    lambda a, b, c=0: a + 10 * b + 100 * c, b=7
                ),
                71,
            ),
            (
                functools.partial(
                    lambda a, b=0, c=0: a + 10 * b + 100 * c, b=7
                ),
                71,
            ),
            (functools.partial(lambda a, b=0: a + b, a=3), 3),
            (functools.partial(Acc().m, 10), 11),
            (functools.partial(Acc().n, 4), 5),
        ],
    )
    def test_partial_arity(self, cb, expected):
        assert self.module.call_two(cb) == expected

    def test_varargs_fun_extra_args(self):
        star = functools.partial(lambda *a: int(sum(a)))
        assert self.module.call_two(star, fun_extra_args=(9,)) == 12
        assert self.module.call_two(star, fun_extra_args=(10,)) == 13
        assert self.module.call_none(star, fun_extra_args=(6,)) == 6
        arest = functools.partial(lambda a, *rest: a + sum(rest), 10)
        assert self.module.call_two(arest, fun_extra_args=(9,)) == 22
        assert self.module.call_two(arest, fun_extra_args=(7,)) == 20

    def test_keyword_on_default_is_required(self):
        plain = lambda a, b=0: a + b
        with pytest.raises(Exception) as plain_err:
            self.module.call_none(plain)
        with pytest.raises(type(plain_err.value)) as bound_err:
            self.module.call_none(functools.partial(plain, b=5))
        assert "argument list" in str(plain_err.value)
        assert str(bound_err.value) == str(plain_err.value)
