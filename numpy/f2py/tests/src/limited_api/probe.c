#include "fortranobject.h"

#ifdef Py_TARGET_ABI3T
#if Py_LIMITED_API != 0x030F0000
#error "The free-threaded probe must use the Python 3.15 Limited API"
#endif
#else
#if Py_LIMITED_API != 0x030C0000
#error "The probe must use NumPy's Python 3.12 API floor"
#endif
#endif

#ifdef F2PY_TEST_PRIVATE_API
#define F2PY_TEST_NAME "test_f2py_limited_api_reject"
static const char *
private_type_layout(PyObject *obj)
{
    return Py_TYPE(obj)->tp_name;
}
#else
#define F2PY_TEST_NAME "test_f2py_limited_api"
#endif

static PyObject *
identity(PyObject *self, PyObject *args, PyObject *kw, void *data)
{
    Py_RETURN_NONE;
}

static FortranDataDef routine_def = {
    .name = "identity",
    .rank = -1,
    .func = (f2py_init_func)identity,
};

static double array_data = 0;
static FortranDataDef invalid_defs[] = {
    {
        .name = "invalid",
        .rank = 1,
        .dims = {{-1}},
        .type = NPY_DOUBLE,
        .data = (char *)&array_data,
    },
    {NULL}
};

static PyObject *
new_fortran(PyObject *module, PyObject *Py_UNUSED(args))
{
    return PyFortranObject_NewAsAttr(module, &routine_def);
}

static PyObject *
new_invalid_fortran(PyObject *module, PyObject *Py_UNUSED(args))
{
    return PyFortranObject_New(module, invalid_defs, NULL);
}

static PyObject *
describe(PyObject *module, PyObject *obj)
{
    return f2py_describe_obj(obj);
}

static PyMethodDef probe_methods[] = {
    {"new_fortran", new_fortran, METH_NOARGS, NULL},
    {"new_invalid_fortran", new_invalid_fortran, METH_NOARGS, NULL},
    {"describe", describe, METH_O, NULL},
    {NULL, NULL, 0, NULL}
};

static int
probe_exec(PyObject *module)
{
    import_array2("cannot initialize the Limited API probe", -1);
    return F2PyRuntime_Init(module);
}

#ifndef Py_TARGET_ABI3T
static PyModuleDef_Slot probe_slots[] = {
    {Py_mod_exec, (void *)probe_exec},
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
    {0, NULL}
};

static struct PyModuleDef probe_def = {
    PyModuleDef_HEAD_INIT,
    F2PY_TEST_NAME,
    NULL,
    sizeof(F2PyRuntimeState),
    probe_methods,
    probe_slots,
    F2PyRuntime_Traverse,
    F2PyRuntime_Clear,
    F2PyRuntime_Free
};

#ifdef F2PY_TEST_PRIVATE_API
PyMODINIT_FUNC PyInit_test_f2py_limited_api_reject(void)
#else
PyMODINIT_FUNC PyInit_test_f2py_limited_api(void)
#endif
{
    return PyModuleDef_Init(&probe_def);
}
#else
PyABIInfo_VAR(probe_abi_info);
static PySlot probe_pyslots[] = {
    PySlot_STATIC_DATA(Py_mod_abi, &probe_abi_info),
    PySlot_STATIC_DATA(Py_mod_name, F2PY_TEST_NAME),
    PySlot_STATIC_DATA(Py_mod_methods, probe_methods),
    PySlot_SIZE(Py_mod_state_size, sizeof(F2PyRuntimeState)),
    PySlot_FUNC(Py_mod_state_traverse, F2PyRuntime_Traverse),
    PySlot_FUNC(Py_mod_state_clear, F2PyRuntime_Clear),
    PySlot_FUNC(Py_mod_state_free, F2PyRuntime_Free),
    PySlot_DATA(Py_mod_gil, Py_MOD_GIL_NOT_USED),
    PySlot_DATA(Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED),
    PySlot_FUNC(Py_mod_exec, probe_exec),
    PySlot_END
};

#ifdef F2PY_TEST_PRIVATE_API
PyMODEXPORT_FUNC PyModExport_test_f2py_limited_api_reject(void)
#else
PyMODEXPORT_FUNC PyModExport_test_f2py_limited_api(void)
#endif
{
    return probe_pyslots;
}
#endif
