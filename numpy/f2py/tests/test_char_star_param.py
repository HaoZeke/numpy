import textwrap

import pytest

import numpy as np
from numpy.f2py import crackfortran
from numpy.f2py.capi_maps import getstrlength
from numpy.f2py.func2subr import var2fixfortran

from . import util


@pytest.mark.slow
class TestCharacterStarParameter(util.F2PyTest):
    suffix = ".f90"
    code = """
        module m
          character(len=*), parameter :: hello = 'hello'
          character*(*), parameter :: star = 'xyz'
          character(len=*), parameter :: names(2) = &
            [character(len=4) :: 'ab', 'cdef']
          character(len=*), parameter :: pair(2) = (/'mn', 'op'/)
          character(len=*), parameter :: cat = 'foo' // 'bar'
          character(len=*), parameter :: hi = "hi" // "!"
          character(len=*), parameter :: many = 'ab' // 'cd' // 'ef'
          character(len=*), parameter :: abde = 'ab' // 'cde'
          character(len=*), parameter :: empty = ''
          character(len=*), parameter :: blank = ""
          character(len=*), parameter :: apos = 'it''s'
          character(len=*), parameter :: boxa(1) = (/ apos /)
          character(len=*), parameter :: dq = "it""s"
          character(len=2), parameter :: some_long_name = 'z'
          character(len=*), parameter :: early = some_long_name
          character(len=*), parameter :: box(1) = (/ early /)
          character(len=*), parameter :: ver = '1.2.3'
          character(len=*), parameter :: vp(2) = (/ ver, 'ab' /)
          character(len=2), parameter :: clipped = 'abcd'
          character(len=*), parameter :: mix(2) = (/ clipped, 'xyz' /)
          integer, parameter :: nk = 4_4*2
          character(len=*), parameter :: kn(1) = [character(len=nk) :: 'abcdefgh']
          character(len=*), parameter :: slash = 'a//b'
          character(len=*), parameter :: joined(2) = (/'ab' // 'c', 'd' // 'ef'/)
          character(len=*), parameter :: zeros(2) = (/ '', '' /)
          integer, parameter :: n = 8
          character(len=*), parameter :: wide(2) = [character(len=n) :: 'ab', 'cd']
          character(len=*), parameter :: prod(1) = [character(len=2*3) :: 'ab']
          character(len=*), parameter :: pads(2) = [character(len=4) :: 'ab', 'cd']
          character(len=*), parameter :: eq(2) = ['ab', 'cd']
          character(len=*), parameter :: name = 'abc'
          character(len=*), parameter :: named(2) = (/'ab' // 'c', name/)
          character(len=*), parameter :: divlen(1) = [character(len=8/2) :: 'ab']
          character(len=*), parameter :: divprod(1) = [character(len=5/2*2) :: 'ab']
          integer :: after = 7
        end module m
    """

    def test_read(self):
        m = self.module.m
        assert m.hello.dtype == np.dtype("S5")
        assert m.hello.tobytes() == b"hello"
        assert m.star.dtype == np.dtype("S3")
        assert m.star.tobytes() == b"xyz"
        assert m.names.dtype == np.dtype("S4")
        assert m.names.tobytes() == b"ab  cdef"
        assert m.pair.dtype == np.dtype("S2")
        assert m.pair.tobytes() == b"mnop"
        assert m.cat.dtype == np.dtype("S6")
        assert m.cat.tobytes() == b"foobar"
        assert m.hi.dtype == np.dtype("S3")
        assert m.hi.tobytes() == b"hi!"
        assert m.many.dtype == np.dtype("S6")
        assert m.many.tobytes() == b"abcdef"
        assert m.abde.dtype == np.dtype("S5")
        assert m.abde.tobytes() == b"abcde"
        assert m.empty.dtype == np.dtype("|S0")
        assert m.empty.tobytes() == b""
        assert m.blank.dtype == np.dtype("|S0")
        assert m.blank.tobytes() == b""
        assert m.apos.dtype == np.dtype("S4")
        assert m.apos.tobytes() == b"it's"
        assert m.boxa.dtype == np.dtype("S4")
        assert m.boxa.tobytes() == b"it's"
        assert m.dq.dtype == np.dtype("S4")
        assert m.dq.tobytes() == b'it"s'
        assert m.early.dtype == np.dtype("S2")
        assert m.early.tobytes() == b"z "
        assert m.box.dtype == np.dtype("S2")
        assert m.box.tobytes() == b"z "
        assert m.ver.dtype == np.dtype("S5")
        assert m.ver.tobytes() == b"1.2.3"
        assert m.vp.dtype == np.dtype("S5")
        assert m.vp.tobytes() == b"1.2.3ab   "
        assert m.mix.dtype == np.dtype("S2")
        assert m.mix.tobytes() == b"abxy"
        assert m.kn.dtype == np.dtype("S8")
        assert m.kn.tobytes() == b"abcdefgh"
        assert m.slash.dtype == np.dtype("S4")
        assert m.slash.tobytes() == b"a//b"
        assert m.joined.dtype == np.dtype("S3")
        assert m.joined.tobytes() == b"abcdef"
        assert m.zeros.dtype == np.dtype("|S0")
        assert m.zeros.tobytes() == b""
        assert m.wide.dtype == np.dtype("S8")
        assert m.wide.tobytes() == b"ab      cd      "
        assert m.prod.dtype == np.dtype("S6")
        assert m.prod.tobytes() == b"ab    "
        assert m.pads.dtype == np.dtype("S4")
        assert m.pads.tobytes() == b"ab  cd  "
        assert m.eq.dtype == np.dtype("S2")
        assert m.eq.tobytes() == b"abcd"
        assert m.name.dtype == np.dtype("S3")
        assert m.name.tobytes() == b"abc"
        assert m.named.dtype == np.dtype("S3")
        assert m.named.tobytes() == b"abcabc"
        assert m.divlen.dtype == np.dtype("S4")
        assert m.divlen.tobytes() == b"ab  "
        assert m.divprod.dtype == np.dtype("S4")
        assert m.divprod.tobytes() == b"ab  "
        assert int(m.after) == 7


class TestCharacterStarUnknown:
    def test_unknown_length(self, tmp_path):
        src = textwrap.dedent(
            """
            module m
              character(len=*), parameter :: mixed(2) = (/'ab' // 'c', trim('xy')/)
              character(len=*), parameter :: implied(3) = (/ ('ab' // 'xy', i = 1, 3) /)
              character(len=*), parameter :: cat = 'ab' // 'c'
              character(len=*), parameter :: via(2) = (/ cat, 'z' /)
              character(len=*), parameter :: bare(2) = ['ab', 'cdef']
              character(len=*), parameter :: modern(2) = ['hello', 'x']
              character(len=*), parameter :: one(1) = [character(len=*) :: 'hello']
              character(len=*), parameter :: stars(2) = &
                [character(len=*) :: 'ab', 'cdef']
              character(len=*), parameter :: divlen(1) = [character(len=8/2) :: 'ab']
              character(len=*), parameter :: divprod(1) = [character(len=5/2*2) :: 'ab']
              character(len=*), parameter :: name = 'z'
              character(len=*), parameter :: named(2) = (/'ab' // 'c', name/)
              integer :: a, b
              character(len=5), parameter :: fixed = 'hello'
              character(len=*) :: dummy
            end module m
            """
        )
        path = tmp_path / "unknown.f90"
        path.write_text(src)
        mod = crackfortran.crackfortran([str(path)])
        vars = mod[0]["vars"]

        def length(name):
            sel = vars[name]["charselector"]
            return sel.get("len", sel.get("*"))

        assert length("mixed") == "*"
        assert length("implied") == "*"
        assert length("via") == "3"
        assert length("cat") == "3"
        assert length("bare") == "4"
        assert vars["bare"]["="] == "['ab', 'cdef']"
        assert length("modern") == "5"
        assert length("one") == "5"
        assert vars["one"]["="] == "[character(len=*):: 'hello']"
        assert length("stars") == "4"
        assert vars["stars"]["="] == "[character(len=*):: 'ab', 'cdef']"
        assert length("divlen") == "4"
        assert length("divprod") == "4"
        assert length("name") == "1"
        assert length("named") == "3"
        assert length("fixed") == "5"
        assert length("dummy") == "*"
        assert "a" in vars and "b" in vars

    def test_bracket_comma(self):
        marked = crackfortran.markoutercomma("['ab', 'cdef']")
        assert "@,@" not in marked
        parts = [
            p.strip()
            for p in crackfortran.markoutercomma("a, b").split("@,@")
        ]
        assert parts == ["a", "b"]

    def test_length_division(self):
        fold = crackfortran._fold_char_len
        bad = crackfortran._CHAR_LEN_BAD
        got = {"8/2": "4", "5/2*2": "4", "2*3": "6", "4_4": "4",
               "8_4/2": "4", "2*4_4": "8", "2_4*3": "6", "4_4*2": "8",
               "2_4*2": "4", "2*2_4": "4", "8/2_4": "4",
               "5/(-2)*(-2)": "4", "-(5/(-2))": "2",
               "(-5)/2*(-2)": "4", "(-7)/2+4": "1",
               "2**3": "8", "2_4**3": "8", "4_4**2": "16", "2**4_4": "16",
               "2*4_4**2": "32", "2**3**2": "512", "(2_4)**3": "8",
               "2**(-1)": "0", "2_4**(-1)": "0", "4_4**(-1)": "0",
               "4_4**(-2)": "0", "2**(-4_4)": "0", "1_4**(-3)": "1",
               "-5": "0", "1-4": "0"}
        assert {expr: fold(expr, {}) for expr in got} == got
        assert fold("n/2", {"n": 8}) == "4"
        assert fold("n**2", {"n": 4}) == "16"
        assert fold("n**(-1)", {"n": 4}) == "0"
        assert all(fold(e, {}) is bad for e in
                   ("n/2", "8/2.0", "8.0/2", "m/2", "len('a/b')",
                    "0**(-1)", "0_4**(-1)"))
        assert fold("n/2.0", {"n": 8}) is bad

    def test_folded_lengths(self, tmp_path):
        src = textwrap.dedent(
            """
            module m
              integer, parameter :: n = 8
              real, parameter :: r = 8.0
              character(len=*), parameter :: qn(1) = [character(len=n/2) :: 'ab']
              character(len=*), parameter :: qr(1) = [character(len=r/2) :: 'abcd']
              character(len=*), parameter :: qdot(1) = [character(len=8/2.0) :: 'ab']
              character(len=*), parameter :: qm(1) = [character(len=m/2) :: 'ab']
              character(len=*), parameter :: qcall(1) = [character(len=len('a/b')) :: 'abc']
              character(len=*), parameter :: qkind(1) = [character(len=8_4/2) :: 'ab']
              character(len=*), parameter :: qneg(1) = [character(len=5/(-2)*(-2)) :: 'ab']
              character(len=*), parameter :: qto(1) = [character(len=(-7)/2+4) :: 'a']
              character(len=5), parameter :: wide_name = 'z'
              character(len=*), parameter :: pair(2) = (/ wide_name, 'ab' /)
              character(len=2), parameter :: clipped = 'abcd'
              character(len=*), parameter :: only(1) = (/ clipped /)
              character(len=*), parameter :: name = 'abc'
              character(len=*), parameter :: named(2) = (/'ab' // 'c', name/)
              character(len=*), parameter :: glyph = achar(65)
              character(len=*), parameter :: ac(2) = (/ glyph, 'z' /)
            end module m
            """
        )
        path = tmp_path / "folded.f90"
        path.write_text(src)
        mod = crackfortran.crackfortran([str(path)])
        vars = mod[0]["vars"]

        def length(name):
            sel = vars[name]["charselector"]
            return sel.get("len", sel.get("*"))

        expect = {
            "qn": "4", "qr": "*", "qdot": "*", "qm": "*", "qcall": "*",
            "qkind": "4", "qneg": "4", "qto": "1", "wide_name": "5",
            "pair": "5", "clipped": "2", "only": "2", "name": "3",
            "named": "3", "glyph": "*", "ac": "*",
        }
        assert {name: length(name) for name in expect} == expect

    def test_fortran_char_len(self, tmp_path):
        src = textwrap.dedent(
            """
            module m
              integer, parameter :: nk = 4_4*2
              integer, parameter :: nd = 8_4/2
              integer, parameter :: nm = 2_4*3
              integer, parameter :: half = nk/2
              integer, parameter :: n = 8
              integer, parameter :: bad = 4_4**2
              character(len=*), parameter :: kn(1) = [character(len=nk) :: 'abcdefgh']
              character(len=*), parameter :: dn(1) = [character(len=nd) :: 'abcd']
              character(len=*), parameter :: mn(1) = [character(len=nm) :: 'abcdef']
              character(len=*), parameter :: hn(1) = [character(len=half) :: 'abcd']
              character(len=*), parameter :: nn(1) = [character(len=n/2) :: 'abcd']
              character(len=*), parameter :: bn(1) = [character(len=bad) :: 'ab']
              character(len=nk), parameter :: declared = 'abcdefgh'
              character(len=*), parameter :: boxd(1) = (/ declared /)
              character(len=*), parameter :: inlinek(1) = [character(len=8_4/2) :: 'ab']
              character(len=*), parameter :: before(1) = (/ shortn /)
              character(len=*), parameter :: shortn = lat
              character(len=2), parameter :: lat = 'abcd'
              character(len=2), parameter :: some_long_name = 'z'
              character(len=*), parameter :: early = some_long_name
              character(len=*), parameter :: box(1) = (/ early /)
              character(len=*), parameter :: apos = 'it''s'
              character(len=*), parameter :: boxa(1) = (/ apos /)
              character(len=*), parameter :: dq = "it""s"
              character(len=*), parameter :: boxq(1) = (/ dq /)
              character(len=*), parameter :: bs = 'a\\nb'
              character(len=*), parameter :: bsv(1) = (/ bs /)
              character(len=*), parameter :: slashb = 'a\\b'
              character(len=*), parameter :: ver = '1.2.3'
              character(len=*), parameter :: vp(2) = (/ ver, 'ab' /)
              character(len=*), parameter :: dot = '2.0'
              character(len=*), parameter :: dp(1) = (/ dot /)
              character(len=*), parameter :: phase = 'phase2e'
              character(len=*), parameter :: pp(1) = (/ phase /)
              character(len=*), parameter :: wq = "1.2.3"
              character(len=*), parameter :: wp(2) = (/ wq, 'ab' /)
              character(len=*), parameter :: direct(2) = (/ '1.2.3', 'ab' /)
              character(len=2), parameter :: clipped = 'abcd'
              character(len=*), parameter :: alias = clipped
              character(len=*), parameter :: via_alias(1) = (/ alias /)
              character(len=*), parameter :: mix(2) = (/ clipped, 'xyz' /)
              character(len=*), parameter :: mixb(2) = [clipped, 'xyz']
              character(len=2), parameter :: ay = 'xy'
              character(len=5), parameter :: bee = 'abcde'
              character(len=*), parameter :: q(2) = (/ ay, bee /)
              character(len=*), parameter :: longer(2) = (/ bee, ay /)
              character(len=5), parameter :: wide_name = 'z'
              character(len=*), parameter :: pair(2) = (/ wide_name, 'ab' /)
              character(len=*), parameter :: alias5 = wide_name
              character(len=*), parameter :: pair5(2) = (/ alias5, 'ab' /)
              character(len=*), parameter :: plain = 'abcd'
              character(len=*), parameter :: plaina(1) = (/ plain /)
              character(len=*), parameter :: cat = 'ab' // 'c'
              character(len=*), parameter :: via(2) = (/ cat, 'z' /)
              character(len=*), parameter :: glyph = achar(65)
              character(len=*), parameter :: ac(2) = (/ glyph, 'z' /)
            end module m
            """
        )
        path = tmp_path / "charlen.f90"
        path.write_text(src)
        mod = crackfortran.crackfortran([str(path)])
        vars = mod[0]["vars"]

        def length(name):
            sel = vars[name]["charselector"]
            return sel.get("len", sel.get("*"))

        expect = {
            "kn": "8", "dn": "4", "mn": "6", "hn": "4", "nn": "4", "bn": "16",
            "declared": "8", "boxd": "8", "inlinek": "4",
            "before": "2", "shortn": "2", "lat": "2",
            "some_long_name": "2", "early": "2", "box": "2",
            "apos": "4", "boxa": "4", "dq": "4", "boxq": "4",
            "bs": "4", "bsv": "4", "slashb": "3",
            "ver": "5", "vp": "5", "dot": "3", "dp": "3",
            "phase": "7", "pp": "7", "wq": "5", "wp": "5", "direct": "5",
            "clipped": "2", "alias": "2", "via_alias": "2",
            "mix": "2", "mixb": "2", "q": "2", "longer": "5",
            "wide_name": "5", "pair": "5", "alias5": "5", "pair5": "5",
            "plain": "4", "plaina": "4", "cat": "3", "via": "3",
            "glyph": "*", "ac": "*",
        }
        assert {name: length(name) for name in expect} == expect

    def test_kind_power_length(self, tmp_path):
        src = textwrap.dedent(
            """
            module m
              integer, parameter :: bad = 4_4**2
              integer, parameter :: negbad = 4_4**(-1)
              character(len=2_4**3), parameter :: midk = '01234567'
              character(len=4_4**2), parameter :: p44 = 'x'
              character(len=2**4_4), parameter :: p24 = 'x'
              character(len=bad), parameter :: frombad = 'x'
              character(len=*), parameter :: bn(1) = [character(len=bad) :: 'ab']
              character(len=2**3), parameter :: plain = 'xxxxxxxx'
              character(len=2_4**(-1)), parameter :: negk = 'x'
              character(len=2**(-1)), parameter :: negp = 'x'
              character(len=negbad), parameter :: fromneg = 'x'
              character(len=1_4**(-3)), parameter :: one = 'x'
              character(len=-3), parameter :: neglen = 'x'
            end module m
            """
        )
        path = tmp_path / "kindpow.f90"
        path.write_text(src)
        mod = crackfortran.crackfortran([str(path)])
        vars = mod[0]["vars"]

        def length(name):
            sel = vars[name]["charselector"]
            return sel.get("len", sel.get("*"))

        expect = {
            "midk": "8", "p44": "16", "p24": "16", "frombad": "16",
            "bn": "16", "plain": "8", "negk": "0", "negp": "0",
            "fromneg": "0", "one": "1", "neglen": "0",
        }
        assert {name: length(name) for name in expect} == expect

    def test_variable_length_expr(self, tmp_path):
        src = textwrap.dedent(
            """
            subroutine s(n, n1_2, r8_kind, buf, pown, mulname, idlen, kindlen, kindn, prod)
              integer :: n, n1_2, r8_kind
              character(len=n**2), intent(out) :: buf
              character(len=2**n), intent(out) :: pown
              character(len=2*n1_2), intent(out) :: mulname
              character(len=n1_2), intent(out) :: idlen
              character(len=r8_kind), intent(out) :: kindlen
              character(len=4_4*n), intent(out) :: kindn
              character(len=2*n), intent(out) :: prod
            end subroutine s
            function f(n)
              integer :: n
              character(len=4_4*n) :: f
            end function f
            function g(n)
              integer :: n
              character(len=n**2) :: g
            end function g
            """
        )
        path = tmp_path / "openlen.f90"
        path.write_text(src)
        blocks = {b["name"]: b for b in crackfortran.crackfortran([str(path)])}
        vars = blocks["s"]["vars"]

        def star(name):
            return vars[name]["charselector"]["*"]

        expect = {
            "buf": "n**2", "pown": "2**n", "mulname": "2*n1_2",
            "idlen": "n1_2", "kindlen": "r8_kind", "kindn": "4*n",
            "prod": "2*n",
        }
        assert {name: star(name) for name in expect} == expect
        assert {name: getstrlength(vars[name]) for name in expect} == expect
        assert blocks["f"]["vars"]["f"]["charselector"]["len"] == "4*n"
        assert "character(len=4*n)" in var2fixfortran(
            blocks["f"]["vars"], "f", f90mode=True)
        assert blocks["g"]["vars"]["g"]["charselector"]["len"] == "n**2"
        assert getstrlength(blocks["g"]["vars"]["g"]) == "n**2"

    def test_star_selector_power(self, tmp_path):
        src = textwrap.dedent(
            """
            subroutine star(n, buf, buf2, buf3)
              integer :: n
              character*(2_4**3), intent(out) :: buf
              character*(2**4_4), intent(out) :: buf2
              character*(4_4*n), intent(out) :: buf3
            end subroutine star
            module pm
              character*(2_4**3), parameter :: starred = '01234567'
              character*(2**4_4), parameter :: p24 = 'xxxxxxxxxxxxxxxx'
            end module pm
            """
        )
        path = tmp_path / "starpow.f90"
        path.write_text(src)
        blocks = {b["name"]: b for b in crackfortran.crackfortran([str(path)])}
        got = blocks["star"]["vars"]
        assert got["buf"]["charselector"]["*"] == "8"
        assert got["buf2"]["charselector"]["*"] == "16"
        assert got["buf3"]["charselector"]["*"] == "(4*n)"
        assert getstrlength(got["buf"]) == "8"
        assert getstrlength(got["buf3"]) == "(4*n)"
        mod = blocks["pm"]["vars"]
        assert mod["starred"]["charselector"]["*"] == "8"
        assert mod["p24"]["charselector"]["*"] == "16"

    def test_parameter_power_dimension(self, tmp_path):
        src = textwrap.dedent(
            """
            subroutine d(arr)
              integer, parameter :: bad = 4_4**2
              integer :: arr(bad)
            end subroutine d
            """
        )
        path = tmp_path / "dimpow.f90"
        path.write_text(src)
        block = crackfortran.crackfortran([str(path)])[0]
        assert block["vars"]["arr"]["dimension"] == ["16"]
