#ifndef Py_FORTRANOBJECT_H
#define Py_FORTRANOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <stdarg.h>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#endif
#ifdef FORTRANOBJECT_C
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL _npy_f2py_ARRAY_API
#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"

#ifdef F2PY_REPORT_ATEXIT
#include <sys/timeb.h>
// clang-format off
extern void f2py_start_clock(void);
extern void f2py_stop_clock(void);
extern void f2py_start_call_clock(void);
extern void f2py_stop_call_clock(void);
extern void f2py_cb_start_clock(void);
extern void f2py_cb_stop_clock(void);
extern void f2py_cb_start_call_clock(void);
extern void f2py_cb_stop_call_clock(void);
extern void f2py_report_on_exit(int, void *);
// clang-format on
#endif

#ifdef DMALLOC
#include "dmalloc.h"
#endif

/* Fortran object interface */

/*
123456789-123456789-123456789-123456789-123456789-123456789-123456789-12

PyFortranObject represents various Fortran objects:
Fortran (module) routines, COMMON blocks, module data.

Author: Pearu Peterson <pearu@cens.ioc.ee>
*/

/* Declaration arithmetic preserves Fortran grouping at pointer-sized width. */
static NPY_INLINE npy_intp
f2py_spec_overflow(void)
{
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_OverflowError, "Fortran declaration integer exceeds npy_intp");
    }
    return 0;
}

static NPY_INLINE npy_intp
f2py_spec_add(npy_intp a, npy_intp b)
{
    if ((b > 0 && a > NPY_MAX_INTP - b) || (b < 0 && a < NPY_MIN_INTP - b)) {
        return f2py_spec_overflow();
    }
    return a + b;
}

static NPY_INLINE npy_intp
f2py_spec_sub(npy_intp a, npy_intp b)
{
    if ((b > 0 && a < NPY_MIN_INTP + b) || (b < 0 && a > NPY_MAX_INTP + b)) {
        return f2py_spec_overflow();
    }
    return a - b;
}

static NPY_INLINE npy_intp
f2py_spec_mul(npy_intp a, npy_intp b)
{
    if ((a > 0 && ((b > 0 && a > NPY_MAX_INTP / b) ||
                   (b < 0 && b < NPY_MIN_INTP / a))) ||
        (a < 0 && ((b > 0 && a < NPY_MIN_INTP / b) ||
                   (b < 0 && a < NPY_MAX_INTP / b)))) {
        return f2py_spec_overflow();
    }
    return a * b;
}

static NPY_INLINE npy_intp
f2py_spec_div(npy_intp a, npy_intp b)
{
    if (b == 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ZeroDivisionError, "zero divisor in Fortran declaration");
        }
        return 0;
    }
    if (a == NPY_MIN_INTP && b == -1) {
        return f2py_spec_overflow();
    }
    return a / b;
}

static NPY_INLINE npy_intp
f2py_spec_mod(npy_intp a, npy_intp b)
{
    if (b == 0) {
        (void)f2py_spec_div(a, b);
        return 0;
    }
    return a == NPY_MIN_INTP && b == -1 ? 0 : a % b;
}

static NPY_INLINE npy_intp
f2py_spec_modulo(npy_intp a, npy_intp b)
{
    npy_intp remainder = f2py_spec_mod(a, b);
    return remainder && ((remainder < 0) != (b < 0)) ? remainder + b : remainder;
}

static NPY_INLINE npy_intp
f2py_spec_pow(npy_intp base, npy_intp exponent)
{
    npy_intp result = 1;
    if (exponent < 0) {
        return base == 1 ? 1 : base == -1 ? (exponent % 2 ? -1 : 1) : 0;
    }
    while (exponent && !PyErr_Occurred()) {
        if (exponent & 1) {
            result = f2py_spec_mul(result, base);
        }
        exponent /= 2;
        if (exponent) {
            base = f2py_spec_mul(base, base);
        }
    }
    return result;
}

static NPY_INLINE npy_intp f2py_spec_cast(npy_intp a) { return a; }
static NPY_INLINE npy_intp f2py_spec_min(npy_intp a, npy_intp b) { return a < b ? a : b; }
static NPY_INLINE npy_intp f2py_spec_max(npy_intp a, npy_intp b) { return a > b ? a : b; }
static NPY_INLINE npy_intp f2py_spec_abs(npy_intp a) { return a < 0 ? f2py_spec_sub(0, a) : a; }
static NPY_INLINE npy_intp f2py_spec_sign(npy_intp a, npy_intp b) { return b < 0 ? (a > 0 ? -a : a) : f2py_spec_abs(a); }
static NPY_INLINE npy_intp f2py_spec_dim(npy_intp a, npy_intp b) { return a > b ? f2py_spec_sub(a, b) : 0; }
static NPY_INLINE npy_intp f2py_spec_merge(npy_intp a, npy_intp b, npy_intp mask) { return mask ? a : b; }
static NPY_INLINE npy_intp f2py_spec_not(npy_intp a) { return !a; }
static NPY_INLINE npy_intp f2py_spec_and(npy_intp a, npy_intp b) { return a && b; }
static NPY_INLINE npy_intp f2py_spec_or(npy_intp a, npy_intp b) { return a || b; }
static NPY_INLINE npy_intp f2py_spec_eq(npy_intp a, npy_intp b) { return a == b; }
static NPY_INLINE npy_intp f2py_spec_ne(npy_intp a, npy_intp b) { return a != b; }
static NPY_INLINE npy_intp f2py_spec_lt(npy_intp a, npy_intp b) { return a < b; }
static NPY_INLINE npy_intp f2py_spec_le(npy_intp a, npy_intp b) { return a <= b; }
static NPY_INLINE npy_intp f2py_spec_gt(npy_intp a, npy_intp b) { return a > b; }
static NPY_INLINE npy_intp f2py_spec_ge(npy_intp a, npy_intp b) { return a >= b; }
static NPY_INLINE npy_intp f2py_spec_lbound(npy_intp lower, npy_intp extent) { return extent == 0 ? 1 : lower; }
static NPY_INLINE npy_intp f2py_spec_ubound(npy_intp lower, npy_intp extent) { return extent == 0 ? 0 : f2py_spec_add(lower, extent - 1); }

static NPY_INLINE npy_intp
f2py_spec_size_impl(PyArrayObject *array, npy_intp dim)
{
    if (array == NULL || dim < 0 || dim > PyArray_NDIM(array)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "invalid DIM in Fortran declaration inquiry");
        }
        return 0;
    }
    return dim == 0 ? PyArray_SIZE(array) : PyArray_DIM(array, dim - 1);
}

static NPY_INLINE npy_intp
f2py_spec_select(npy_intp dim, int rank, ...)
{
    va_list arguments;
    npy_intp result = 0;
    int axis;
    if (dim < 1 || dim > rank) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "invalid DIM in Fortran bound inquiry");
        }
        return 0;
    }
    va_start(arguments, rank);
    for (axis = 1; axis <= rank; axis++) {
        npy_intp value = va_arg(arguments, npy_intp);
        if (axis == dim) {
            result = value;
        }
    }
    va_end(arguments);
    return result;
}

static NPY_INLINE npy_intp
f2py_spec_item_impl(PyArrayObject *array, ...)
{
    npy_intp indexes[NPY_MAXDIMS], result;
    PyObject *item, *integer;
    va_list arguments;
    int axis;
    if (PyErr_Occurred()) {
        return 0;
    }
    if (array == NULL || !PyArray_ISINTEGER(array)) {
        PyErr_SetString(PyExc_TypeError, "Fortran declaration needs an INTEGER array");
        return 0;
    }
    va_start(arguments, array);
    for (axis = 0; axis < PyArray_NDIM(array); axis++) {
        indexes[axis] = va_arg(arguments, npy_intp);
        if (indexes[axis] < 0 || indexes[axis] >= PyArray_DIM(array, axis)) {
            va_end(arguments);
            PyErr_SetString(PyExc_IndexError, "Fortran declaration subscript is out of bounds");
            return 0;
        }
    }
    va_end(arguments);
    item = PyArray_GETITEM(array, PyArray_GetPtr(array, indexes));
    if (item == NULL) {
        return 0;
    }
    integer = PyNumber_Index(item);
    Py_DECREF(item);
    if (integer == NULL) {
        return 0;
    }
    result = PyLong_AsSsize_t(integer);
    Py_DECREF(integer);
    return result;
}

static NPY_INLINE npy_intp
f2py_spec_len_trim(const char *string, npy_intp length)
{
    while (length > 0 && string[length - 1] == ' ') {
        length--;
    }
    return length;
}

#define f2py_spec_size(var, dim) f2py_spec_size_impl((PyArrayObject *)capi_ ## var ## _as_array, dim)
#define f2py_spec_item(var, ...) f2py_spec_item_impl((PyArrayObject *)capi_ ## var ## _as_array, __VA_ARGS__)

#define F2PY_MAX_DIMS 40

typedef void (*f2py_set_data_func)(char *, npy_intp *);
typedef void (*f2py_void_func)(void);
typedef void (*f2py_init_func)(int *, npy_intp *, f2py_set_data_func, int *);

/*typedef void* (*f2py_c_func)(void*,...);*/

typedef void *(*f2pycfunc)(void);

typedef struct {
    char *name; /* attribute (array||routine) name */
    int rank;   /* array rank, 0 for scalar, max is F2PY_MAX_DIMS,
                   || rank=-1 for Fortran routine */
    struct {
        npy_intp d[F2PY_MAX_DIMS];
    } dims;              /* dimensions of the array, || not used */
    int type;            /* PyArray_<type> || not used */
    int elsize;                /* Element size || not used */
    char *data;          /* pointer to array || Fortran routine */
    f2py_init_func func; /* initialization function for
                            allocatable arrays:
                            func(&rank,dims,set_ptr_func,name,len(name))
                            || C/API wrapper for Fortran routine */
    char *doc;           /* documentation string; only recommended
                            for routines. */
} FortranDataDef;

typedef struct {
    PyObject_HEAD
    int len;              /* Number of attributes */
    FortranDataDef *defs; /* An array of FortranDataDef's */
    PyObject *dict;       /* Fortran object attribute dictionary */
} PyFortranObject;

#define PyFortran_Check(op) (Py_TYPE(op) == &PyFortran_Type)
#define PyFortran_Check1(op) (0 == strcmp(Py_TYPE(op)->tp_name, "fortran"))

extern PyTypeObject PyFortran_Type;
extern int
F2PyDict_SetItemString(PyObject *dict, char *name, PyObject *obj);
extern PyObject *
PyFortranObject_New(FortranDataDef *defs, f2py_void_func init);
extern PyObject *
PyFortranObject_NewAsAttr(FortranDataDef *defs);

PyObject *
F2PyCapsule_FromVoidPtr(void *ptr, void (*dtor)(PyObject *));
void *
F2PyCapsule_AsVoidPtr(PyObject *obj);
int
F2PyCapsule_Check(PyObject *ptr);

extern void *
F2PySwapThreadLocalCallbackPtr(char *key, void *ptr);
extern void *
F2PyGetThreadLocalCallbackPtr(char *key);

#define ISCONTIGUOUS(m) (PyArray_FLAGS(m) & NPY_ARRAY_C_CONTIGUOUS)
#define F2PY_INTENT_IN 1
#define F2PY_INTENT_INOUT 2
#define F2PY_INTENT_OUT 4
#define F2PY_INTENT_HIDE 8
#define F2PY_INTENT_CACHE 16
#define F2PY_INTENT_COPY 32
#define F2PY_INTENT_C 64
#define F2PY_OPTIONAL 128
#define F2PY_INTENT_INPLACE 256
#define F2PY_INTENT_ALIGNED4 512
#define F2PY_INTENT_ALIGNED8 1024
#define F2PY_INTENT_ALIGNED16 2048

#define ARRAY_ISALIGNED(ARR, SIZE) ((size_t)(PyArray_DATA(ARR)) % (SIZE) == 0)
#define F2PY_ALIGN4(intent) (intent & F2PY_INTENT_ALIGNED4)
#define F2PY_ALIGN8(intent) (intent & F2PY_INTENT_ALIGNED8)
#define F2PY_ALIGN16(intent) (intent & F2PY_INTENT_ALIGNED16)

#define F2PY_GET_ALIGNMENT(intent) \
    (F2PY_ALIGN4(intent)           \
             ? 4                   \
             : (F2PY_ALIGN8(intent) ? 8 : (F2PY_ALIGN16(intent) ? 16 : 1)))
#define F2PY_CHECK_ALIGNMENT(arr, intent) \
    ARRAY_ISALIGNED(arr, F2PY_GET_ALIGNMENT(intent))
#define F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(arr) ((PyDataType_TYPENUM(PyArray_DESCR(arr)) == NPY_STRING && PyArray_ITEMSIZE(arr) >= 1) \
                                                 || PyDataType_TYPENUM(PyArray_DESCR(arr)) == NPY_UINT8)
#define F2PY_IS_UNICODE_ARRAY(arr) (PyDataType_TYPENUM(PyArray_DESCR(arr)) == NPY_UNICODE)

extern PyArrayObject *
ndarray_from_pyobj(const int type_num, const int elsize_, npy_intp *dims,
                   const int rank, const int intent, PyObject *obj,
                   const char *errmess);

extern PyArrayObject *
array_from_pyobj(const int type_num, npy_intp *dims, const int rank,
                 const int intent, PyObject *obj);
extern int
copy_ND_array(const PyArrayObject *in, PyArrayObject *out);

#ifdef DEBUG_COPY_ND_ARRAY
extern void
dump_attrs(const PyArrayObject *arr);
#endif

extern PyObject *f2py_describe_obj(PyObject *obj);

  /* Utility CPP macros and functions that can be used in signature file
     expressions. See signature-file.rst for documentation.
  */

#define f2py_itemsize(var) (PyArray_ITEMSIZE(capi_ ## var ## _as_array))
#define f2py_size(var, ...) f2py_size_impl((PyArrayObject *)(capi_ ## var ## _as_array), ## __VA_ARGS__, -1)
#define f2py_rank(var) var ## _Rank
#define f2py_shape(var,dim) var ## _Dims[dim]
#define f2py_len(var) f2py_shape(var,0)
#define f2py_fshape(var,dim) f2py_shape(var,rank(var)-dim-1)
#define f2py_flen(var) f2py_fshape(var,0)
#define f2py_slen(var) capi_ ## var ## _len

  extern npy_intp f2py_size_impl(PyArrayObject* var, ...);

#ifdef __cplusplus
}
#endif
#endif /* !Py_FORTRANOBJECT_H */
