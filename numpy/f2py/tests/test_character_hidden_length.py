import textwrap

from numpy.f2py.auxfuncs import getcallprotoargument

from . import util


def test_mixed_call_prototype():
    rout = {
        "args": ["bare", "fixed", "bare_array", "strings",
                 "c_bare", "value_bare", "out_bare"],
        "vars": {
            "bare": {"typespec": "character"},
            "fixed": {
                "typespec": "character",
                "charselector": {"len": "3"},
            },
            "bare_array": {
                "typespec": "character",
                "dimension": ["2"],
            },
            "strings": {
                "typespec": "character",
                "charselector": {"len": "4"},
                "dimension": ["2"],
            },
            "c_bare": {
                "typespec": "character",
                "intent": ["c"],
            },
            "value_bare": {
                "typespec": "character",
                "attrspec": ["value"],
            },
            "out_bare": {
                "typespec": "character",
                "intent": ["out"],
            },
        },
    }
    expected = (
        "character*,string,character*,string*,character,character,character*,"
        "size_t,size_t,size_t,size_t,size_t,size_t"
    )
    assert getcallprotoargument(rout) == expected


class TestBareCharacterHiddenLength(util.F2PyTest):
    suffix = ".f90"
    code = textwrap.dedent("""
        subroutine assumed_length_after_bare(bare, assumed, observed)
          implicit none
          character, intent(in) :: bare
          character(*), intent(in) :: assumed
          integer, intent(out) :: observed
          observed = len(assumed)
        end subroutine assumed_length_after_bare
        """)

    def test_assumed_length_order(self):
        assert self.module.assumed_length_after_bare(b"B", b"XYZ") == 3


class TestBareCharacterCallback(util.F2PyTest):
    suffix = ".f90"
    code = textwrap.dedent("""
        subroutine callback_mix(callback, status)
          implicit none
          integer, external :: callback
          integer, intent(out) :: status
          character :: bare
          character(3) :: fixed
          bare = 'B'
          fixed = 'XYZ'
          status = callback(bare, fixed)
        end subroutine callback_mix
        """)

    def test_callback(self):
        def callback(bare, fixed):
            return int(bare != "B" or fixed != "XYZ")

        assert self.module.callback_mix(callback) == 0
