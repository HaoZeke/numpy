/*
 * cmocka unit tests of the f2py runtime, numpy/f2py/src/fortranobject.c.
 *
 * test_fortranobject.py builds this file with the f2py meson backend, which
 * compiles fortranobject.c exactly as for an f2py generated module, and runs
 * the cmocka group in a subprocess.  Every test listed in run() is reported
 * as a separate pytest test.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "fortranobject.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cmocka.h>

#define assert_no_pyerr()                                   \
    do {                                                    \
        if (PyErr_Occurred()) {                             \
            PyErr_Print();                                  \
            fail_msg("unexpected Python exception");        \
        }                                                   \
    } while (0)

/*
 * Check that an exception of type exc is pending, that its message contains
 * message (if not NULL), and clear it.
 */
static void
check_pyerr(PyObject *exc, const char *message)
{
    PyObject *raised = PyErr_GetRaisedException();
    if (raised == NULL) {
        fail_msg("expected a Python exception");
    }
    if (!PyErr_GivenExceptionMatches(raised, exc)) {
        PyErr_SetRaisedException(raised);
        PyErr_Print();
        fail_msg("unexpected Python exception type");
    }
    if (message != NULL) {
        PyObject *str = PyObject_Str(raised);
        assert_non_null(str);
        const char *text = PyUnicode_AsUTF8(str);
        assert_non_null(text);
        if (strstr(text, message) == NULL) {
            fail_msg("expected '%s' in the exception message '%s'",
                     message, text);
        }
        Py_DECREF(str);
    }
    Py_DECREF(raised);
}

static void
check_str(PyObject *obj, const char *expected)
{
    assert_non_null(obj);
    assert_true(PyUnicode_Check(obj));
    assert_string_equal(PyUnicode_AsUTF8(obj), expected);
}

/* A new array of the given shape and type holding 0, 1, 2, ... */
static PyArrayObject *
arange(int nd, npy_intp *dims, int type_num)
{
    PyArrayObject *arr = (PyArrayObject *)PyArray_SimpleNew(nd, dims,
                                                            type_num);
    assert_non_null(arr);
    for (npy_intp i = 0; i < PyArray_SIZE(arr); i++) {
        PyObject *item = PyLong_FromSsize_t(i);
        assert_non_null(item);
        assert_int_equal(PyArray_SETITEM(
                arr, (char *)PyArray_DATA(arr) + i * PyArray_ITEMSIZE(arr),
                item), 0);
        Py_DECREF(item);
    }
    return arr;
}

/* Stand-in for a Fortran routine and for its f2py generated C/API wrapper. */

static int
twice(int x)
{
    return 2 * x;
}

static PyObject *
twice_wrapper(PyObject *self, PyObject *args, PyObject *kwds, void *f2py_func)
{
    int x;
    (void)self;
    (void)kwds;
    if (!PyArg_ParseTuple(args, "i", &x)) {
        return NULL;
    }
    return PyLong_FromLong(((int (*)(int))f2py_func)(x));
}

/* Wrapper of a dummy routine: one without a Fortran routine to call. */
static PyObject *
dummy_wrapper(PyObject *self, PyObject *args, PyObject *kwds, void *f2py_func)
{
    (void)self;
    (void)args;
    (void)kwds;
    return PyBool_FromLong(f2py_func == NULL);
}

/*
 * Stand-in for the getdims routine that f2py generates for a rank-1 module
 * allocatable array (see fgetdims1 and fgetdims2 in f90mod_rules.py).
 */
static double *alloc_data = NULL;
static npy_intp alloc_size = 0;

static void
alloc_getdims(int *rank, npy_intp *dims, f2py_set_data_func set_data,
              int *flag)
{
    npy_intp allocated;
    assert_int_equal(*rank, 1);
    if (alloc_data != NULL && dims[0] >= 0 && dims[0] != alloc_size) {
        free(alloc_data);
        alloc_data = NULL;
    }
    if (alloc_data == NULL && dims[0] >= 1) {
        alloc_data = calloc(dims[0], sizeof(double));
        assert_non_null(alloc_data);
        alloc_size = dims[0];
    }
    if (alloc_data != NULL) {
        dims[0] = alloc_size;
    }
    *flag = 1;
    allocated = (alloc_data != NULL);
    set_data((char *)alloc_data, &allocated);
}

static void
free_alloc_data(void)
{
    free(alloc_data);
    alloc_data = NULL;
    alloc_size = 0;
}

/* Module data of a module object, see module_defs(). */
static int module_int;
static double module_array[3];
static int module_init_calls;

static void
module_init(void)
{
    module_init_calls++;
}

/* The FortranDataDef table f2py generates for a module (f90mod_rules.py). */
static FortranDataDef *
module_defs(void)
{
    static const FortranDataDef init[] = {
        {"ival", 0, {{-1}}, NPY_INT, sizeof(int), (char *)&module_int},
        {"darr", 1, {{3}}, NPY_DOUBLE, sizeof(double), (char *)module_array},
        {"alloc", 1, {{-1}}, NPY_DOUBLE, sizeof(double), NULL, alloc_getdims},
        {"twice", -1, {{-1}}, 0, 0, (char *)twice,
         (f2py_init_func)twice_wrapper, "twice(x)"},
        {NULL},
    };
    static FortranDataDef defs[sizeof(init) / sizeof(init[0])];
    memcpy(defs, init, sizeof(init));
    free_alloc_data();
    module_int = 7;
    for (int i = 0; i < 3; i++) {
        module_array[i] = i + 1;
    }
    module_init_calls = 0;
    return defs;
}

static void
test_dict_setitemstring(void **state)
{
    (void)state;
    PyObject *dict = PyDict_New();
    PyObject *value = PyLong_FromLong(42);
    assert_non_null(dict);
    assert_non_null(value);

    assert_int_equal(F2PyDict_SetItemString(dict, "value", value), 0);
    PyObject *stored = PyMapping_GetItemString(dict, "value");
    assert_ptr_equal(stored, value);
    Py_DECREF(stored);

    /* A NULL object is reported and not stored. */
    assert_int_equal(F2PyDict_SetItemString(dict, "missing", NULL), -1);
    assert_no_pyerr();
    PyErr_SetString(PyExc_RuntimeError, "failed to create the object");
    assert_int_equal(F2PyDict_SetItemString(dict, "missing", NULL), -1);
    assert_no_pyerr();
    assert_int_equal(PyDict_Size(dict), 1);

    Py_DECREF(value);
    Py_DECREF(dict);
}

static void
test_capsule(void **state)
{
    (void)state;
    int target = 0;
    PyObject *capsule = F2PyCapsule_FromVoidPtr(&target, NULL);
    assert_non_null(capsule);
    assert_true(F2PyCapsule_Check(capsule));
    assert_ptr_equal(F2PyCapsule_AsVoidPtr(capsule), &target);
    Py_DECREF(capsule);

    /* Failures return NULL without leaving an exception set. */
    assert_false(F2PyCapsule_Check(Py_None));
    assert_null(F2PyCapsule_AsVoidPtr(Py_None));
    assert_no_pyerr();
    assert_null(F2PyCapsule_FromVoidPtr(NULL, NULL));
    assert_no_pyerr();
}

static int destructor_calls;

static void
count_destructor(PyObject *capsule)
{
    (void)capsule;
    destructor_calls++;
}

static void
test_capsule_destructor(void **state)
{
    (void)state;
    int target = 0;
    destructor_calls = 0;
    PyObject *capsule = F2PyCapsule_FromVoidPtr(&target, count_destructor);
    assert_non_null(capsule);
    Py_DECREF(capsule);
    assert_int_equal(destructor_calls, 1);
}

static void
test_thread_local_callback_ptr(void **state)
{
    (void)state;
    char key[] = "test_fortranobject_callback";
    int first, second;

    assert_null(F2PyGetThreadLocalCallbackPtr(key));
    assert_null(F2PySwapThreadLocalCallbackPtr(key, &first));
    assert_ptr_equal(F2PyGetThreadLocalCallbackPtr(key), &first);
    assert_ptr_equal(F2PySwapThreadLocalCallbackPtr(key, &second), &first);
    assert_ptr_equal(F2PyGetThreadLocalCallbackPtr(key), &second);
    assert_ptr_equal(F2PySwapThreadLocalCallbackPtr(key, NULL), &second);
    assert_null(F2PyGetThreadLocalCallbackPtr(key));
    assert_no_pyerr();

    assert_int_equal(PyDict_DelItemString(PyThreadState_GetDict(), key), 0);
}

static void
test_routine_object(void **state)
{
    (void)state;
    static FortranDataDef def = {
            "twice", -1, {{-1}}, 0, 0, (char *)twice,
            (f2py_init_func)twice_wrapper, "twice(x)"};
    PyObject *routine = PyFortranObject_NewAsAttr(&def);
    assert_non_null(routine);
    assert_true(PyFortran_Check(routine));

    PyObject *name = PyObject_GetAttrString(routine, "__name__");
    check_str(name, "function twice");
    Py_DECREF(name);
    PyObject *repr = PyObject_Repr(routine);
    check_str(repr, "<fortran function twice>");
    Py_DECREF(repr);
    PyObject *doc = PyObject_GetAttrString(routine, "__doc__");
    check_str(doc, "twice(x)\n");
    Py_DECREF(doc);

    /* Calls go to the wrapper, which gets the routine as last argument. */
    PyObject *result = PyObject_CallFunction(routine, "i", 21);
    assert_non_null(result);
    assert_int_equal(PyLong_AsLong(result), 42);
    Py_DECREF(result);

    PyObject *cpointer = PyObject_GetAttrString(routine, "_cpointer");
    assert_non_null(cpointer);
    assert_true(F2PyCapsule_Check(cpointer));
    assert_ptr_equal(F2PyCapsule_AsVoidPtr(cpointer), (void *)twice);
    Py_DECREF(cpointer);

    Py_DECREF(routine);
}

static void
test_routine_without_function(void **state)
{
    (void)state;
    static FortranDataDef nowrapper = {"nowrapper", -1, {{-1}}, 0, 0, NULL,
                                       NULL, NULL};
    static FortranDataDef dummy = {"dummy", -1, {{-1}}, 0, 0, NULL,
                                   (f2py_init_func)dummy_wrapper, NULL};

    PyObject *routine = PyFortranObject_NewAsAttr(&nowrapper);
    assert_non_null(routine);
    PyObject *doc = PyObject_GetAttrString(routine, "__doc__");
    check_str(doc, "nowrapper - no docs available\n");
    Py_DECREF(doc);
    assert_null(PyObject_CallNoArgs(routine));
    check_pyerr(PyExc_RuntimeError, "no function to call");
    Py_DECREF(routine);

    /* Dummy routines are called with a NULL Fortran routine. */
    routine = PyFortranObject_NewAsAttr(&dummy);
    assert_non_null(routine);
    PyObject *result = PyObject_CallNoArgs(routine);
    assert_ptr_equal(result, Py_True);
    Py_DECREF(result);
    Py_DECREF(routine);
}

static void
test_data_object_not_callable(void **state)
{
    (void)state;
    static int scalar;
    static double array[3];
    static FortranDataDef scalar_def = {"scalar", 0, {{-1}}, NPY_INT,
                                        sizeof(int), (char *)&scalar};
    static FortranDataDef array_def = {"array", 1, {{3}}, NPY_DOUBLE,
                                       sizeof(double), (char *)array};

    PyObject *obj = PyFortranObject_NewAsAttr(&scalar_def);
    assert_non_null(obj);
    PyObject *name = PyObject_GetAttrString(obj, "__name__");
    check_str(name, "scalar scalar");
    Py_DECREF(name);
    assert_null(PyObject_CallNoArgs(obj));
    check_pyerr(PyExc_TypeError, "this fortran object is not callable");
    Py_DECREF(obj);

    obj = PyFortranObject_NewAsAttr(&array_def);
    assert_non_null(obj);
    name = PyObject_GetAttrString(obj, "__name__");
    check_str(name, "array array");
    Py_DECREF(name);
    Py_DECREF(obj);
}

static void
test_module_object(void **state)
{
    (void)state;
    FortranDataDef *defs = module_defs();
    PyObject *module = PyFortranObject_New(defs, module_init);
    assert_non_null(module);
    assert_int_equal(module_init_calls, 1);

    /* Module data is exposed as arrays that share the Fortran storage. */
    PyObject *ival = PyObject_GetAttrString(module, "ival");
    assert_non_null(ival);
    assert_true(PyArray_Check(ival));
    assert_int_equal(PyArray_NDIM((PyArrayObject *)ival), 0);
    assert_int_equal(PyArray_TYPE((PyArrayObject *)ival), NPY_INT);
    assert_ptr_equal(PyArray_DATA((PyArrayObject *)ival), &module_int);
    Py_DECREF(ival);

    PyArrayObject *darr = (PyArrayObject *)PyObject_GetAttrString(module,
                                                                  "darr");
    assert_non_null(darr);
    assert_true(PyArray_Check(darr));
    assert_int_equal(PyArray_NDIM(darr), 1);
    assert_int_equal(PyArray_DIM(darr, 0), 3);
    assert_true(PyArray_ISFARRAY(darr));
    assert_ptr_equal(PyArray_DATA(darr), module_array);
    Py_DECREF(darr);

    PyObject *routine = PyObject_GetAttrString(module, "twice");
    assert_non_null(routine);
    PyObject *result = PyObject_CallFunction(routine, "i", 4);
    assert_non_null(result);
    assert_int_equal(PyLong_AsLong(result), 8);
    Py_DECREF(result);
    Py_DECREF(routine);

    PyObject *doc = PyObject_GetAttrString(module, "__doc__");
    check_str(doc,
              "ival : 'i'-scalar\n"
              "darr : 'd'-array(3)\n"
              "alloc : 'd'-array(-1), not allocated\n"
              "twice(x)\n");
    Py_DECREF(doc);

    PyObject *repr = PyObject_Repr(module);
    check_str(repr, "<fortran object>");
    Py_DECREF(repr);

    /* dir() lists every Fortran object, including unallocated arrays. */
    PyObject *dir = PyObject_Dir(module);
    assert_non_null(dir);
    const char *names[] = {"alloc", "darr", "ival", "twice"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        PyObject *item = PyUnicode_FromString(names[i]);
        assert_int_equal(PySequence_Contains(dir, item), 1);
        Py_DECREF(item);
    }
    Py_DECREF(dir);

    Py_DECREF(module);
}

static void
test_module_setattr(void **state)
{
    (void)state;
    PyObject *module = PyFortranObject_New(module_defs(), NULL);
    assert_non_null(module);

    /* Assignments copy into the Fortran storage. */
    PyObject *value = Py_BuildValue("[ddd]", 4.0, 5.0, 6.0);
    assert_int_equal(PyObject_SetAttrString(module, "darr", value), 0);
    Py_DECREF(value);
    assert_true(module_array[0] == 4.0 && module_array[1] == 5.0
                && module_array[2] == 6.0);
    value = PyLong_FromLong(9);
    assert_int_equal(PyObject_SetAttrString(module, "ival", value), 0);
    Py_DECREF(value);
    assert_int_equal(module_int, 9);

    /* The shape of non-allocatable arrays is fixed. */
    value = Py_BuildValue("[dd]", 1.0, 2.0);
    assert_int_equal(PyObject_SetAttrString(module, "darr", value), -1);
    check_pyerr(PyExc_ValueError,
                "0-th dimension must be fixed to 3 but got 2");
    Py_DECREF(value);
    assert_true(module_array[0] == 4.0);

    assert_int_equal(PyObject_SetAttrString(module, "twice", Py_None), -1);
    check_pyerr(PyExc_AttributeError, "over-writing fortran routine");

    /* Other attributes live in the object dictionary. */
    value = PyLong_FromLong(1);
    assert_int_equal(PyObject_SetAttrString(module, "extra", value), 0);
    PyObject *extra = PyObject_GetAttrString(module, "extra");
    assert_ptr_equal(extra, value);
    Py_DECREF(extra);
    Py_DECREF(value);
    assert_int_equal(PyObject_DelAttrString(module, "extra"), 0);
    assert_int_equal(PyObject_DelAttrString(module, "extra"), -1);
    check_pyerr(PyExc_AttributeError, "delete non-existing fortran attribute");

    Py_DECREF(module);
}

static void
test_module_allocatable(void **state)
{
    (void)state;
    PyObject *module = PyFortranObject_New(module_defs(), NULL);
    assert_non_null(module);

    PyObject *alloc = PyObject_GetAttrString(module, "alloc");
    assert_ptr_equal(alloc, Py_None);
    Py_DECREF(alloc);

    /* Assigning allocates (or reallocates) to the shape of the value. */
    PyObject *value = Py_BuildValue("[ddd]", 1.0, 2.0, 3.0);
    assert_int_equal(PyObject_SetAttrString(module, "alloc", value), 0);
    Py_DECREF(value);
    assert_non_null(alloc_data);
    assert_int_equal(alloc_size, 3);
    assert_true(alloc_data[0] == 1.0 && alloc_data[2] == 3.0);

    PyArrayObject *arr = (PyArrayObject *)PyObject_GetAttrString(module,
                                                                 "alloc");
    assert_non_null(arr);
    assert_true(PyArray_Check(arr));
    assert_int_equal(PyArray_DIM(arr, 0), 3);
    assert_ptr_equal(PyArray_DATA(arr), alloc_data);
    Py_DECREF(arr);

    value = Py_BuildValue("[dd]", 7.0, 8.0);
    assert_int_equal(PyObject_SetAttrString(module, "alloc", value), 0);
    Py_DECREF(value);
    assert_int_equal(alloc_size, 2);
    assert_true(alloc_data[0] == 7.0 && alloc_data[1] == 8.0);

    /* Assigning None deallocates. */
    assert_int_equal(PyObject_SetAttrString(module, "alloc", Py_None), 0);
    assert_null(alloc_data);
    alloc = PyObject_GetAttrString(module, "alloc");
    assert_ptr_equal(alloc, Py_None);
    Py_DECREF(alloc);

    Py_DECREF(module);
    free_alloc_data();
}

static void
test_ndarray_intent_in_reuses_array(void **state)
{
    (void)state;
    npy_intp shape[] = {3};
    PyArrayObject *arr = arange(1, shape, NPY_DOUBLE);
    Py_ssize_t refcnt = Py_REFCNT(arr);

    npy_intp dims[] = {-1};
    PyArrayObject *res = ndarray_from_pyobj(
            NPY_DOUBLE, sizeof(double), dims, 1, F2PY_INTENT_IN,
            (PyObject *)arr, NULL);
    assert_ptr_equal(res, arr);
    assert_int_equal(Py_REFCNT(arr), refcnt);
    assert_int_equal(dims[0], 3);

    /* intent(out) returns a new reference. */
    dims[0] = -1;
    res = ndarray_from_pyobj(NPY_DOUBLE, sizeof(double), dims, 1,
                             F2PY_INTENT_IN | F2PY_INTENT_OUT,
                             (PyObject *)arr, NULL);
    assert_ptr_equal(res, arr);
    assert_int_equal(Py_REFCNT(arr), refcnt + 1);
    Py_DECREF(res);

    Py_DECREF(arr);
}

static void
test_ndarray_intent_in_copies(void **state)
{
    (void)state;
    npy_intp shape[] = {2, 3};
    PyArrayObject *arr = arange(2, shape, NPY_DOUBLE);

    /* A C-ordered array is copied into Fortran order... */
    npy_intp dims[] = {-1, -1};
    PyArrayObject *res = ndarray_from_pyobj(
            NPY_DOUBLE, sizeof(double), dims, 2, F2PY_INTENT_IN,
            (PyObject *)arr, NULL);
    assert_non_null(res);
    assert_ptr_not_equal(res, arr);
    assert_true(PyArray_ISFARRAY(res));
    assert_int_equal(dims[0], 2);
    assert_int_equal(dims[1], 3);
    for (npy_intp i = 0; i < 2; i++) {
        for (npy_intp j = 0; j < 3; j++) {
            assert_true(*(double *)PyArray_GETPTR2(res, i, j)
                        == *(double *)PyArray_GETPTR2(arr, i, j));
        }
    }
    Py_DECREF(res);

    /* ...unless intent(c) asks for C order. */
    dims[0] = dims[1] = -1;
    res = ndarray_from_pyobj(NPY_DOUBLE, sizeof(double), dims, 2,
                             F2PY_INTENT_IN | F2PY_INTENT_C, (PyObject *)arr,
                             NULL);
    assert_ptr_equal(res, arr);

    /* Other types are cast. */
    dims[0] = dims[1] = -1;
    res = ndarray_from_pyobj(NPY_INT, sizeof(int), dims, 2, F2PY_INTENT_IN,
                             (PyObject *)arr, NULL);
    assert_non_null(res);
    assert_int_equal(PyArray_TYPE(res), NPY_INT);
    assert_int_equal(*(int *)PyArray_GETPTR2(res, 1, 2), 5);
    Py_DECREF(res);

    Py_DECREF(arr);
}

static void
test_ndarray_intent_in_sequence(void **state)
{
    (void)state;
    PyObject *list = Py_BuildValue("[iii]", 1, 2, 3);
    npy_intp dims[] = {-1};
    PyArrayObject *res = ndarray_from_pyobj(
            NPY_DOUBLE, sizeof(double), dims, 1, F2PY_INTENT_IN, list, NULL);
    assert_non_null(res);
    assert_int_equal(PyArray_TYPE(res), NPY_DOUBLE);
    assert_int_equal(dims[0], 3);
    assert_true(*(double *)PyArray_GETPTR1(res, 2) == 3.0);
    Py_DECREF(res);

    /* array_from_pyobj takes the element size from the type. */
    dims[0] = -1;
    res = array_from_pyobj(NPY_INT, dims, 1, F2PY_INTENT_IN, list);
    assert_non_null(res);
    assert_int_equal(PyArray_TYPE(res), NPY_INT);
    assert_int_equal(PyArray_ITEMSIZE(res), sizeof(int));
    Py_DECREF(res);

    Py_DECREF(list);
}

static void
test_ndarray_intent_inout(void **state)
{
    (void)state;
    npy_intp shape[] = {2, 3};
    PyArrayObject *arr = arange(2, shape, NPY_DOUBLE);
    npy_intp dims[] = {-1, -1};

    /* intent(inout) works in place and cannot copy. */
    assert_null(ndarray_from_pyobj(NPY_DOUBLE, sizeof(double), dims, 2,
                                   F2PY_INTENT_INOUT, (PyObject *)arr, NULL));
    check_pyerr(PyExc_ValueError,
                "failed to initialize intent(inout) array"
                " -- input not fortran contiguous");

    dims[0] = dims[1] = -1;
    PyArrayObject *res = ndarray_from_pyobj(
            NPY_DOUBLE, sizeof(double), dims, 2,
            F2PY_INTENT_INOUT | F2PY_INTENT_C, (PyObject *)arr, NULL);
    assert_ptr_equal(res, arr);

    PyObject *list = Py_BuildValue("[ddd]", 1.0, 2.0, 3.0);
    dims[0] = -1;
    assert_null(ndarray_from_pyobj(NPY_DOUBLE, sizeof(double), dims, 1,
                                   F2PY_INTENT_INOUT, list, NULL));
    check_pyerr(PyExc_TypeError, "input 'list' object is not an array");
    Py_DECREF(list);

    Py_DECREF(arr);
}

static void
test_ndarray_intent_inplace(void **state)
{
    (void)state;
    npy_intp shape[] = {6};
    PyArrayObject *base = arange(1, shape, NPY_DOUBLE);
    PyObject *two = PyLong_FromLong(2);
    PyObject *every_other = PySlice_New(NULL, NULL, two);
    Py_DECREF(two);
    PyArrayObject *view = (PyArrayObject *)PyObject_GetItem((PyObject *)base,
                                                            every_other);
    Py_DECREF(every_other);
    assert_non_null(view);
    assert_false(PyArray_ISONESEGMENT(view));

    /* A non-contiguous input is copied and written back on resolve. */
    npy_intp dims[] = {-1};
    PyArrayObject *res = ndarray_from_pyobj(
            NPY_DOUBLE, sizeof(double), dims, 1, F2PY_INTENT_INPLACE,
            (PyObject *)view, NULL);
    assert_non_null(res);
    assert_ptr_not_equal(res, view);
    assert_true(PyArray_FLAGS(res) & NPY_ARRAY_WRITEBACKIFCOPY);
    for (npy_intp i = 0; i < 3; i++) {
        *(double *)PyArray_GETPTR1(res, i) = -1.0;
    }
    assert_int_equal(PyArray_ResolveWritebackIfCopy(res), 1);
    Py_DECREF(res);
    for (npy_intp i = 0; i < 6; i++) {
        assert_true(*(double *)PyArray_GETPTR1(base, i)
                    == (i % 2 ? (double)i : -1.0));
    }

    /* The type must already match: no casting in place. */
    PyArrayObject *ints = arange(1, shape, NPY_INT);
    dims[0] = -1;
    assert_null(ndarray_from_pyobj(NPY_DOUBLE, sizeof(double), dims, 1,
                                   F2PY_INTENT_INPLACE, (PyObject *)ints,
                                   NULL));
    check_pyerr(PyExc_ValueError,
                "failed to initialize intent(inplace) array -- input");
    Py_DECREF(ints);

    Py_DECREF(view);
    Py_DECREF(base);
}

static void
test_ndarray_intent_hide(void **state)
{
    (void)state;
    npy_intp dims[] = {2, 3};
    PyArrayObject *res = ndarray_from_pyobj(
            NPY_DOUBLE, sizeof(double), dims, 2,
            F2PY_INTENT_HIDE | F2PY_INTENT_OUT, Py_None, NULL);
    assert_non_null(res);
    assert_int_equal(PyArray_DIM(res, 0), 2);
    assert_int_equal(PyArray_DIM(res, 1), 3);
    assert_true(PyArray_ISFARRAY(res));
    for (npy_intp i = 0; i < 2; i++) {
        for (npy_intp j = 0; j < 3; j++) {
            assert_true(*(double *)PyArray_GETPTR2(res, i, j) == 0.0);
        }
    }
    Py_DECREF(res);

    /* So do optional arguments that are not given. */
    npy_intp opt_dims[] = {4};
    res = ndarray_from_pyobj(NPY_INT, sizeof(int), opt_dims, 1, F2PY_OPTIONAL,
                             Py_None, NULL);
    assert_non_null(res);
    assert_int_equal(PyArray_DIM(res, 0), 4);
    Py_DECREF(res);

    /* The shape must be known. */
    npy_intp undefined[] = {2, -1};
    assert_null(ndarray_from_pyobj(NPY_DOUBLE, sizeof(double), undefined, 2,
                                   F2PY_INTENT_HIDE, Py_None, NULL));
    check_pyerr(PyExc_ValueError, "must have defined dimensions but got (2,-1,)");
}

static void
test_ndarray_intent_cache(void **state)
{
    (void)state;
    npy_intp shape[] = {3};
    PyArrayObject *arr = arange(1, shape, NPY_DOUBLE);
    Py_ssize_t refcnt = Py_REFCNT(arr);

    npy_intp dims[] = {-1};
    PyArrayObject *res = ndarray_from_pyobj(
            NPY_DOUBLE, sizeof(double), dims, 1, F2PY_INTENT_CACHE,
            (PyObject *)arr, NULL);
    assert_ptr_equal(res, arr);
    assert_int_equal(Py_REFCNT(arr), refcnt);

    PyObject *list = Py_BuildValue("[ddd]", 1.0, 2.0, 3.0);
    dims[0] = -1;
    assert_null(ndarray_from_pyobj(NPY_DOUBLE, sizeof(double), dims, 1,
                                   F2PY_INTENT_CACHE, list, NULL));
    check_pyerr(PyExc_TypeError, "input 'list' object is not an array");
    Py_DECREF(list);

    Py_DECREF(arr);
}

static void
test_ndarray_dimensions(void **state)
{
    (void)state;
    npy_intp shape[] = {3};
    PyArrayObject *arr = arange(1, shape, NPY_DOUBLE);

    npy_intp fixed[] = {4};
    assert_null(ndarray_from_pyobj(NPY_DOUBLE, sizeof(double), fixed, 1,
                                   F2PY_INTENT_IN, (PyObject *)arr,
                                   "wrapper.call"));
    check_pyerr(PyExc_ValueError,
                "wrapper.call -- 0-th dimension must be fixed to 4 but got 3");

    /* Missing trailing dimensions are 1... */
    npy_intp extended[] = {-1, -1};
    PyArrayObject *res = ndarray_from_pyobj(
            NPY_DOUBLE, sizeof(double), extended, 2, F2PY_INTENT_IN,
            (PyObject *)arr, NULL);
    assert_ptr_equal(res, arr);
    assert_int_equal(extended[0], 3);
    assert_int_equal(extended[1], 1);

    /* ...and dimensions of length 1 can be dropped. */
    npy_intp row_shape[] = {1, 3};
    PyArrayObject *row = arange(2, row_shape, NPY_DOUBLE);
    npy_intp reduced[] = {-1};
    res = ndarray_from_pyobj(NPY_DOUBLE, sizeof(double), reduced, 1,
                             F2PY_INTENT_IN, (PyObject *)row, NULL);
    assert_ptr_equal(res, row);
    assert_int_equal(reduced[0], 3);
    Py_DECREF(row);

    Py_DECREF(arr);
}

static void
test_ndarray_string_elsize(void **state)
{
    (void)state;
    PyObject *bytes = PyBytes_FromString("hello");
    npy_intp dims[] = {-1};

    /* A negative elsize takes the element size from the object. */
    PyArrayObject *res = ndarray_from_pyobj(NPY_STRING, -1, dims, 0,
                                            F2PY_INTENT_IN, bytes, NULL);
    assert_non_null(res);
    assert_int_equal(PyArray_TYPE(res), NPY_STRING);
    assert_int_equal(PyArray_ITEMSIZE(res), 5);
    assert_memory_equal(PyArray_DATA(res), "hello", 5);
    Py_DECREF(res);

    res = ndarray_from_pyobj(NPY_STRING, 8, dims, 0, F2PY_INTENT_IN, bytes,
                             NULL);
    assert_non_null(res);
    assert_int_equal(PyArray_ITEMSIZE(res), 8);
    Py_DECREF(res);

    Py_DECREF(bytes);
}

static void
test_failed_array_creation_keeps_dtype(void **state)
{
    (void)state;
    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_DOUBLE);
    Py_ssize_t refcnt = Py_REFCNT(dtype);

    /* PyArray_FromAny fails on an input of inhomogeneous shape. */
    PyObject *ragged = Py_BuildValue("[[d][dd]]", 1.0, 1.0, 2.0);
    npy_intp dims[] = {-1};
    assert_null(ndarray_from_pyobj(NPY_DOUBLE, sizeof(double), dims, 1,
                                   F2PY_INTENT_IN, ragged, NULL));
    check_pyerr(PyExc_ValueError, NULL);
    Py_DECREF(ragged);
    assert_int_equal(Py_REFCNT(dtype), refcnt);

    /* PyArray_NewFromDescr fails on an array that is too big. */
    npy_intp huge[] = {NPY_MAX_INTP};
    assert_null(ndarray_from_pyobj(NPY_DOUBLE, sizeof(double), huge, 1,
                                   F2PY_INTENT_HIDE, Py_None, NULL));
    check_pyerr(PyExc_ValueError, NULL);
    assert_int_equal(Py_REFCNT(dtype), refcnt);

    /* PyFortranObject_New fails on module data of undefined shape. */
    static double data[1];
    static FortranDataDef defs[] = {
        {"undefined", 1, {{-1}}, NPY_DOUBLE, sizeof(double), (char *)data},
        {NULL},
    };
    assert_null(PyFortranObject_New(defs, NULL));
    check_pyerr(PyExc_ValueError, NULL);
    assert_int_equal(Py_REFCNT(dtype), refcnt);

    Py_DECREF(dtype);
}

static void
test_copy_ND_array(void **state)
{
    (void)state;
    npy_intp shape[] = {3};
    PyArrayObject *in = arange(1, shape, NPY_INT);
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(1, shape, NPY_DOUBLE,
                                                        1);
    assert_non_null(out);
    assert_int_equal(copy_ND_array(in, out), 0);
    for (npy_intp i = 0; i < 3; i++) {
        assert_true(*(double *)PyArray_GETPTR1(out, i) == (double)i);
    }
    Py_DECREF(out);
    Py_DECREF(in);
}

static void
test_f2py_size(void **state)
{
    (void)state;
    npy_intp shape[] = {2, 3};
    PyArrayObject *arr = arange(2, shape, NPY_DOUBLE);
    assert_int_equal(f2py_size_impl(arr, -1), 6);
    assert_int_equal(f2py_size_impl(arr, 1, -1), 2);
    assert_int_equal(f2py_size_impl(arr, 2, -1), 3);
    /* Out of range dimensions are reported and give 0. */
    assert_int_equal(f2py_size_impl(arr, 3, -1), 0);
    Py_DECREF(arr);
}

static void
check_describe(PyObject *obj, const char *expected)
{
    assert_non_null(obj);
    PyObject *description = f2py_describe_obj(obj);
    assert_no_pyerr();
    check_str(description, expected);
    Py_DECREF(description);
}

static void
test_describe_obj(void **state)
{
    (void)state;
    npy_intp shape[] = {2, 3};
    PyArrayObject *arr = arange(2, shape, NPY_DOUBLE);
    check_describe((PyObject *)arr, "(2, 3,)-f8-numpy.ndarray");
    Py_DECREF(arr);

    PyObject *zero_dim = PyArray_ZEROS(0, NULL, NPY_DOUBLE, 0);
    check_describe(zero_dim, "f8-numpy.ndarray-scalar");
    Py_DECREF(zero_dim);

    PyObject *list = Py_BuildValue("[ii]", 1, 2);
    check_describe(list, "2-list");
    Py_DECREF(list);

    check_describe(Py_None, "NoneType instance");
}

static void
test_header_macros(void **state)
{
    (void)state;
    assert_int_equal(F2PY_GET_ALIGNMENT(F2PY_INTENT_IN), 1);
    assert_int_equal(F2PY_GET_ALIGNMENT(F2PY_INTENT_ALIGNED4), 4);
    assert_int_equal(F2PY_GET_ALIGNMENT(F2PY_INTENT_ALIGNED8), 8);
    assert_int_equal(F2PY_GET_ALIGNMENT(F2PY_INTENT_ALIGNED16), 16);

    npy_intp shape[] = {3};
    PyArrayObject *arr = arange(1, shape, NPY_DOUBLE);
    assert_true(F2PY_CHECK_ALIGNMENT(arr, F2PY_INTENT_ALIGNED8));
    Py_DECREF(arr);

    PyArray_Descr *s1 = PyArray_DescrNewFromType(NPY_STRING);
    assert_non_null(s1);
    PyDataType_SET_ELSIZE(s1, 1);
    PyArrayObject *chars = (PyArrayObject *)PyArray_Zeros(1, shape, s1, 1);
    assert_non_null(chars);
    assert_true(F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(chars));
    assert_false(F2PY_IS_UNICODE_ARRAY(chars));
    Py_DECREF(chars);

    PyArrayObject *bytes = arange(1, shape, NPY_UINT8);
    assert_true(F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(bytes));
    Py_DECREF(bytes);

    PyArrayObject *ints = arange(1, shape, NPY_INT32);
    assert_false(F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(ints));
    Py_DECREF(ints);

    PyArray_Descr *u1 = PyArray_DescrNewFromType(NPY_UNICODE);
    assert_non_null(u1);
    PyDataType_SET_ELSIZE(u1, sizeof(npy_ucs4));
    PyArrayObject *unicode = (PyArrayObject *)PyArray_Zeros(1, shape, u1, 1);
    assert_non_null(unicode);
    assert_true(F2PY_IS_UNICODE_ARRAY(unicode));
    Py_DECREF(unicode);
}

static PyObject *
run(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_dict_setitemstring),
        cmocka_unit_test(test_capsule),
        cmocka_unit_test(test_capsule_destructor),
        cmocka_unit_test(test_thread_local_callback_ptr),
        cmocka_unit_test(test_routine_object),
        cmocka_unit_test(test_routine_without_function),
        cmocka_unit_test(test_data_object_not_callable),
        cmocka_unit_test(test_module_object),
        cmocka_unit_test(test_module_setattr),
        cmocka_unit_test(test_module_allocatable),
        cmocka_unit_test(test_ndarray_intent_in_reuses_array),
        cmocka_unit_test(test_ndarray_intent_in_copies),
        cmocka_unit_test(test_ndarray_intent_in_sequence),
        cmocka_unit_test(test_ndarray_intent_inout),
        cmocka_unit_test(test_ndarray_intent_inplace),
        cmocka_unit_test(test_ndarray_intent_hide),
        cmocka_unit_test(test_ndarray_intent_cache),
        cmocka_unit_test(test_ndarray_dimensions),
        cmocka_unit_test(test_ndarray_string_elsize),
        cmocka_unit_test(test_failed_array_creation_keeps_dtype),
        cmocka_unit_test(test_copy_ND_array),
        cmocka_unit_test(test_f2py_size),
        cmocka_unit_test(test_describe_obj),
        cmocka_unit_test(test_header_macros),
    };
    return PyLong_FromLong(
            cmocka_run_group_tests_name("fortranobject", tests, NULL, NULL));
}

static PyMethodDef methods[] = {
    {"run", run, METH_NOARGS,
     "Run the cmocka tests, return the number of failed tests."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_test_fortranobject_ext",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC
PyInit__test_fortranobject_ext(void)
{
    /* The same set up as in the module init of f2py generated modules. */
    Py_SET_TYPE((PyObject *)&PyFortran_Type, &PyType_Type);
    import_array();
    if (PyType_Ready(&PyFortran_Type) < 0) {
        return NULL;
    }
    PyObject *m = PyModule_Create(&moduledef);
#ifdef Py_GIL_DISABLED
    if (m != NULL) {
        PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
    }
#endif
    return m;
}
