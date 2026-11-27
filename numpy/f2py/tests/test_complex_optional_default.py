import textwrap

import pytest

import numpy as np
from numpy.f2py import capi_maps, crackfortran

from . import util


@pytest.mark.slow
class TestComplexOptionalDefault(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine pick(a, b, out)
  complex(kind=4), intent(in) :: a
  complex(kind=4), intent(in) :: b
  complex(kind=4), intent(out) :: out
  !f2py complex(kind=4), optional, intent(in), depend(a), check(b.r>0) :: b = a
  out = b
end subroutine pick
"""

    @pytest.mark.parametrize("override", [None, 3 + 4j])
    def test_default_names_other_argument(self, override):
        a = 1 + 2j
        if override is None:
            out = self.module.pick(np.complex64(a))
            expect = a
        else:
            out = self.module.pick(np.complex64(a), np.complex64(override))
            expect = override
        assert complex(out) == complex(expect)


@pytest.mark.slow
class TestComplexOptionalDefaultPyf(util.F2PyTest):
    suffix = ".pyf"
    module_name = "complex_optional_pyf"
    code = """
python module complex_optional_pyf
    interface
        subroutine pick(a, b, out)
            fortranname
            callstatement out=b
            callprotoargument void
            complex(kind=4), intent(in) :: a
            complex(kind=4), optional, intent(in), depend(a), check(b.r>0) :: b = a
            complex(kind=4), intent(out) :: out
        end subroutine pick
    end interface
end python module complex_optional_pyf
"""

    @pytest.mark.parametrize("override", [None, 3 + 4j])
    def test_pyf_default_names_other_argument(self, override):
        a = 1 + 2j
        if override is None:
            out = self.module.pick(np.complex64(a))
            expect = a
        else:
            out = self.module.pick(np.complex64(a), np.complex64(override))
            expect = override
        assert complex(out) == complex(expect)


def test_parenthesized_complex_name_default(tmp_path):
    pyf = tmp_path / "complex_optional_paren.pyf"
    pyf.write_text(textwrap.dedent("""\
    python module complex_optional_paren
        interface
            subroutine pick(a, b, out)
                complex(kind=4), intent(in) :: a
                complex(kind=4), optional, intent(in), depend(a) :: b = (a)
                complex(kind=4), intent(out) :: out
            end subroutine pick
        end interface
    end python module complex_optional_paren
    """))
    mod = crackfortran.crackfortran([str(pyf)])
    routine = mod[0]["body"][0]["body"][0]
    assert routine["name"] == "pick"
    mapped = capi_maps.sign2map("b", routine["vars"]["b"])
    assert mapped["init.r"] == "(a).r"
    assert mapped["init.i"] == "(a).i"
    for expr in ("(a)", "( a )", " (a) "):
        mapped = capi_maps.sign2map("b", {"typespec": "complex", "=": expr})
        assert mapped["init.r"] == "(a).r"
        assert mapped["init.i"] == "(a).i"


def test_complex_default_rejects_non_identifier():
    for expr in ("a+1", "(a+1)", "((a))"):
        with pytest.raises(ValueError):
            capi_maps.getinit("b", {"typespec": "complex", "=": expr})


def test_complex_literal_default_real_imag():
    init, _show = capi_maps.getinit(
        "b", {"typespec": "complex", "=": "1+2j", "dimension": ["n"]}
    )
    assert init == "(capi_c.r=1.0,capi_c.i=2.0,capi_c)"


def test_complex_default_call_is_not_run(tmp_path):
    sentinel = tmp_path / "ran"
    expr = (
        "(__import__('pathlib').Path("
        + repr(str(sentinel))
        + ").write_text('x'))"
    )
    pyf = tmp_path / "complex_call.pyf"
    pyf.write_text(textwrap.dedent(f"""\
    python module complex_call
        interface
            subroutine pick(a, b)
                complex, intent(in) :: a
                complex, optional :: b = {expr}
            end subroutine pick
        end interface
    end python module complex_call
    """))
    # f2py leaves case unchanged unless a signature file is requested.
    saved = crackfortran.dolowercase
    crackfortran.dolowercase = 0
    try:
        mod = crackfortran.crackfortran([str(pyf)])
    finally:
        crackfortran.dolowercase = saved
    routine = mod[0]["body"][0]["body"][0]
    assert not sentinel.exists()
    with pytest.raises(ValueError):
        capi_maps.getinit("b", routine["vars"]["b"])
    block = {"name": "pick", "vars": routine["vars"]}
    crackfortran.vars2fortran(block, routine["vars"], ["b"])
    assert not sentinel.exists()
