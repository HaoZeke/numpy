import pytest

from . import util


def _call_code(modname):
    return f"""
python module {modname}
    interface
        subroutine setv(x)
            intent(c) setv
            fortranname
            integer intent(out) :: x
            callstatement
            '''
            x = 42;
            '''
        end subroutine setv
    end interface
end python module {modname}
"""


def _method_code(modname):
    return f"""
python module {modname}
    usercode '''
    static PyObject* {modname}_ping(PyObject *self, PyObject *args) {{
        return PyLong_FromLong(7);
    }}
    '''
    pymethoddef
    '''
    {{"ping", {modname}_ping, METH_NOARGS, "ping"}},
    '''
    interface
    end interface
end python module {modname}
"""


@pytest.mark.parametrize("kind", ["callstatement", "pymethoddef"])
def test_quote_body_on_next_line(kind):
    modname = "q_" + kind
    if kind == "callstatement":
        mod = util.build_code(
            _call_code(modname), suffix=".pyf", module_name=modname)
        assert mod.setv() == 42
    else:
        mod = util.build_code(
            _method_code(modname), suffix=".pyf", module_name=modname)
        assert mod.ping() == 7
