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


class TestUserCodeInterfaces(util.F2PyTest):
    suffix = ".pyf"
    module_name = "user_code_interfaces"

    code = f"""
python module {module_name}
    interface
        subroutine interface_0()
            fortranname
        end subroutine interface_0
        usercode '''
        {{
            PyObject *value = PyLong_FromLong(1);
            if (!value) return NULL;
            if (PyModule_AddObjectRef(m, "answer0", value) < 0) return NULL;
        }}
        '''
    end interface
    interface
        subroutine interface_1()
            fortranname
        end subroutine interface_1
        usercode '''
        {{
            PyObject *value = PyLong_FromLong(2);
            if (!value) return NULL;
            if (PyModule_AddObjectRef(m, "answer1", value) < 0) return NULL;
        }}
        '''
    end interface
end python module {module_name}
"""

    @pytest.mark.parametrize("name, value", [("answer0", 1), ("answer1", 2)])
    def test_usercode_in_each_interface(self, name, value):
        assert getattr(self.module, name) == value
