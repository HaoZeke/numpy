import pytest

from . import util


def test_dict_setitem_keeps_error(capfd):
    # gh-20053: F2PyDict_SetItemString printed and cleared the error raised
    # while creating the value, so the import failed with a SystemError
    code = """
python module gh20053_usercode
    interface
        usercode '''
            PyErr_SetString(PyExc_RuntimeError, "creating the value failed");
            if (F2PyDict_SetItemString(d, "value", NULL) < 0) return NULL;
        '''
    end interface
end python module gh20053_usercode
"""
    with pytest.raises(RuntimeError, match="creating the value failed"):
        util.build_code(code, suffix=".pyf", module_name="gh20053_usercode")
    assert "Error loading" not in capfd.readouterr().err


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
            int status = PyModule_AddObjectRef(m, "foo", value);
            Py_DECREF(value);
            if (status < 0) return NULL;
        }}
        '''
    end interface
end python module {module_name}
    """

    def test_user_code_success(self):
        assert self.module.foo == "Hello from the user code"
