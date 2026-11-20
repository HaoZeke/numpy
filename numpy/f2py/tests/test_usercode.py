import subprocess
import sys

import pytest

from . import util


class TestUserCode(util.F2PyTest):
    suffix = ".pyf"
    module_name = "user_code_success"

    # The failure path in this code isn't expected to happen (and would be
    # untestable if it did, because it would happen in the test setup).
    # However, it does need to compile correctly.
    code = f"""
python module {module_name}
    interface
        usercode '''
        {{
            PyObject *value = PyUnicode_FromString("Hello from the user code");
            if (!value) return NULL;
            if (PyModule_AddObjectRef(m, "foo", value) < 0) return NULL;
        }}
        '''
    end interface
end python module {module_name}
    """

    def test_user_code_success(self):
        assert self.module.foo == "Hello from the user code"


@pytest.mark.slow
@pytest.mark.parametrize(("suffix", "source", "call"), [
    (
        ".pyf",
        """\
python module two_uc_pyf
    usercode '''
    static int first_val = 10;
    '''
    usercode '''
    static int second_val(void) { return first_val + 5; }
    '''
    interface
        subroutine get_val(x)
            fortranname
            integer, intent(out) :: x
            callstatement x = second_val();
        end subroutine get_val
    end interface
end python module two_uc_pyf
""",
        ("get_val", (), 15),
    ),
    (
        ".f90",
        """\
subroutine echo(x, y)
  !f2py usercode int a = 1;
  !f2py usercode int b = 2;
  integer, intent(in) :: x
  integer, intent(out) :: y
  y = x
end subroutine echo
""",
        ("echo", (7,), 7),
    ),
])
def test_multiple_usercode_signature_roundtrip(tmp_path, suffix, source, call):
    src = tmp_path / f"src{suffix}"
    src.write_text(source)
    pyf = tmp_path / "out.pyf"
    cmd = [sys.executable, "-m", "numpy.f2py", "-h", str(pyf)]
    if suffix == ".f90":
        cmd.extend(["-m", "two_uc_f90"])
    cmd.append(str(src))
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "usercode [" not in pyf.read_text()
    build_sources = [pyf] if suffix == ".pyf" else [pyf, src]
    mod_name = "two_uc_pyf" if suffix == ".pyf" else "two_uc_f90"
    mod = util.build_module(build_sources, module_name=mod_name)
    fn_name, args, expected = call
    assert getattr(mod, fn_name)(*args) == expected
