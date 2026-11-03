"""
C declarations, CPP macros, and C functions for f2py2e.
Only required declarations/macros/functions will be used.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import copy
import sys

from . import __version__

f2py_version = __version__.version


def errmess(s: str) -> None:
    """
    Write an error message to stderr.

    This indirection is needed because sys.stderr might not always be available (see #26862).
    """
    if sys.stderr is not None:
        sys.stderr.write(s)

##################### Definitions ##################


outneeds = {'includes0': [], 'includes': [], 'typedefs': [], 'typedefs_generated': [],
            'userincludes': [],
            'cppmacros': [], 'cfuncs': [], 'callbacks': [], 'f90modhooks': [],
            'commonhooks': []}
needs = {}
includes0 = {'includes0': '/*need_includes0*/'}
includes = {'includes': '/*need_includes*/'}
userincludes = {'userincludes': '/*need_userincludes*/'}
typedefs = {'typedefs': '/*need_typedefs*/'}
typedefs_generated = {'typedefs_generated': '/*need_typedefs_generated*/'}
cppmacros = {'cppmacros': '/*need_cppmacros*/'}
cfuncs = {'cfuncs': '/*need_cfuncs*/'}
callbacks = {'callbacks': '/*need_callbacks*/'}
f90modhooks = {'f90modhooks': '/*need_f90modhooks*/',
               'initf90modhooksstatic': '/*initf90modhooksstatic*/',
               'initf90modhooksdynamic': '/*initf90modhooksdynamic*/',
               }
commonhooks = {'commonhooks': '/*need_commonhooks*/',
               'initcommonhooks': '/*need_initcommonhooks*/',
               }

############ Includes ###################

includes0['math.h'] = '#include <math.h>'
includes0['string.h'] = '#include <string.h>'
includes0['setjmp.h'] = '#include <setjmp.h>'
includes0['limits.h'] = '#include <limits.h>'

includes['arrayobject.h'] = '''#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "arrayobject.h"'''
includes['npy_math.h'] = '#include "numpy/npy_math.h"'

includes['arrayobject.h'] = '#include "fortranobject.h"'
includes['stdarg.h'] = '#include <stdarg.h>'
includes['complex.h'] = '#include <complex.h>'

############# Type definitions ###############

typedefs['unsigned_char'] = 'typedef unsigned char unsigned_char;'
typedefs['unsigned_short'] = 'typedef unsigned short unsigned_short;'
typedefs['unsigned_long'] = 'typedef unsigned long unsigned_long;'
typedefs['signed_char'] = 'typedef signed char signed_char;'
typedefs['long_long'] = """
#if defined(NPY_OS_WIN32)
typedef __int64 long_long;
#else
typedef long long long_long;
typedef unsigned long long unsigned_long_long;
#endif
"""
typedefs['unsigned_long_long'] = """
#if defined(NPY_OS_WIN32)
typedef __uint64 long_long;
#else
typedef unsigned long long unsigned_long_long;
#endif
"""
typedefs['long_double'] = """
#ifndef _LONG_DOUBLE
typedef long double long_double;
#endif
"""
typedefs['float16'] = """
#ifndef __FLT16_MANT_DIG__
#error real(kind=2) needs _Float16, which this C compiler does not provide
#endif
typedef _Float16 float16;
"""
typedefs['complex_long_double'] = 'typedef union { struct {long double r,i;}; npy_clongdouble _npy; } complex_long_double;'
typedefs['complex_float'] = 'typedef union { struct {float r,i;}; npy_cfloat _npy; } complex_float;'
typedefs['complex_double'] = 'typedef union { struct {double r,i;}; npy_cdouble _npy; } complex_double;'
typedefs['string'] = """typedef char * string;"""
typedefs['character'] = """typedef char character;"""


############### CPP macros ####################
cppmacros['CFUNCSMESS'] = """
#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,\"debug-capi:\"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \\
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\
    fprintf(stderr,\"\\n\");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif
"""
cppmacros['F_FUNC'] = """
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif
"""
cppmacros['F_WRAPPEDFUNC'] = """
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F
#else
#define F_WRAPPEDFUNC(f,F) _f2pywrap##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F##_
#else
#define F_WRAPPEDFUNC(f,F) _f2pywrap##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F
#else
#define F_WRAPPEDFUNC(f,F) f2pywrap##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F##_
#else
#define F_WRAPPEDFUNC(f,F) f2pywrap##f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f##_,F##_)
#else
#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f,F)
#endif
"""
cppmacros['F_MODFUNC'] = """
#if defined(F90MOD2CCONV1) /*E.g. Compaq Fortran */
#if defined(NO_APPEND_FORTRAN)
#define F_MODFUNCNAME(m,f) $ ## m ## $ ## f
#else
#define F_MODFUNCNAME(m,f) $ ## m ## $ ## f ## _
#endif
#endif

#if defined(F90MOD2CCONV2) /*E.g. IBM XL Fortran, not tested though */
#if defined(NO_APPEND_FORTRAN)
#define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f
#else
#define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f ## _
#endif
#endif

#if defined(F90MOD2CCONV3) /*E.g. MIPSPro Compilers */
#if defined(NO_APPEND_FORTRAN)
#define F_MODFUNCNAME(m,f)  f ## .in. ## m
#else
#define F_MODFUNCNAME(m,f)  f ## .in. ## m ## _
#endif
#endif
/*
#if defined(UPPERCASE_FORTRAN)
#define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(M,F)
#else
#define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(m,f)
#endif
*/

#define F_MODFUNC(m,f) (*(f2pymodstruct##m##.##f))
"""
cppmacros['SWAPUNSAFE'] = """
#define SWAP(a,b) (size_t)(a) = ((size_t)(a) ^ (size_t)(b));\\
 (size_t)(b) = ((size_t)(a) ^ (size_t)(b));\\
 (size_t)(a) = ((size_t)(a) ^ (size_t)(b))
"""
cppmacros['SWAP'] = """
#define SWAP(a,b,t) {\\
    t *c;\\
    c = a;\\
    a = b;\\
    b = c;}
"""
# cppmacros['ISCONTIGUOUS']='#define ISCONTIGUOUS(m) (PyArray_FLAGS(m) &
# NPY_ARRAY_C_CONTIGUOUS)'
cppmacros['PRINTPYOBJERR'] = """
#define PRINTPYOBJERR(obj)\\
    fprintf(stderr,\"#modulename#.error is related to \");\\
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\
    fprintf(stderr,\"\\n\");
"""
cppmacros['MINMAX'] = """
#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif
"""
cppmacros['len..'] = """
/* See fortranobject.h for definitions. The macros here are provided for BC. */
#define rank f2py_rank
#define shape f2py_shape
#define fshape f2py_shape
#define len f2py_len
#define flen f2py_flen
#define slen f2py_slen
#define size f2py_size
"""
cppmacros['pyobj_from_char1'] = r"""
#define pyobj_from_char1(v) (PyLong_FromLong(v))
"""
cppmacros['pyobj_from_short1'] = r"""
#define pyobj_from_short1(v) (PyLong_FromLong(v))
"""
needs['pyobj_from_int1'] = ['signed_char']
cppmacros['pyobj_from_int1'] = r"""
#define pyobj_from_int1(v) (PyLong_FromLong(v))
"""
cppmacros['pyobj_from_long1'] = r"""
#define pyobj_from_long1(v) (PyLong_FromLong(v))
"""
needs['pyobj_from_long_long1'] = ['long_long']
cppmacros['pyobj_from_long_long1'] = """
#ifdef HAVE_LONG_LONG
#define pyobj_from_long_long1(v) (PyLong_FromLongLong(v))
#else
#warning HAVE_LONG_LONG is not available. Redefining pyobj_from_long_long.
#define pyobj_from_long_long1(v) (PyLong_FromLong(v))
#endif
"""
# A Python float or complex would round a long double to a double
needs['pyobj_from_float161'] = ['float16', 'string.h']
cfuncs['pyobj_from_float161'] = """
static PyObject *
pyobj_from_float161(float16 v)
{
    npy_half h;
    memcpy(&h, &v, sizeof(h));
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_HALF);
    PyObject *obj = PyArray_Scalar(&h, descr, NULL);
    Py_DECREF(descr);
    return obj;
}
"""
cppmacros['pyobj_from_unsigned_char1'] = """
#define pyobj_from_unsigned_char1(v) (PyLong_FromUnsignedLong((unsigned long)(v)))
"""
cppmacros['pyobj_from_unsigned_short1'] = """
#define pyobj_from_unsigned_short1(v) (PyLong_FromUnsignedLong((unsigned long)(v)))
"""
cppmacros['pyobj_from_unsigned1'] = """
#define pyobj_from_unsigned1(v) (PyLong_FromUnsignedLong((unsigned long)(v)))
"""
needs['pyobj_from_unsigned_long_long1'] = ['unsigned_long_long']
cppmacros['pyobj_from_unsigned_long_long1'] = """
#define pyobj_from_unsigned_long_long1(v) (PyLong_FromUnsignedLongLong(v))
"""
needs['pyobj_from_long_double1'] = ['long_double']
cfuncs['pyobj_from_long_double1'] = """
static PyObject *
pyobj_from_long_double1(long_double v)
{
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_LONGDOUBLE);
    PyObject *obj = PyArray_Scalar(&v, descr, NULL);
    Py_DECREF(descr);
    return obj;
}
"""
cppmacros['pyobj_from_double1'] = """
#define pyobj_from_double1(v) (PyFloat_FromDouble(v))"""
cppmacros['pyobj_from_float1'] = """
#define pyobj_from_float1(v) (PyFloat_FromDouble(v))"""
needs['pyobj_from_complex_long_double1'] = ['complex_long_double']
cfuncs['pyobj_from_complex_long_double1'] = """
static PyObject *
pyobj_from_complex_long_double1(complex_long_double v)
{
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_CLONGDOUBLE);
    PyObject *obj = PyArray_Scalar(&v._npy, descr, NULL);
    Py_DECREF(descr);
    return obj;
}
"""
needs['pyobj_from_complex_double1'] = ['complex_double', 'npy_math.h']
cppmacros['pyobj_from_complex_double1'] = """
#define pyobj_from_complex_double1(v) (PyComplex_FromDoubles(npy_creal(v._npy),npy_cimag(v._npy)))"""
needs['pyobj_from_complex_float1'] = ['complex_float', 'npy_math.h']
cppmacros['pyobj_from_complex_float1'] = """
#define pyobj_from_complex_float1(v) (PyComplex_FromDoubles((double)npy_crealf(v._npy),(double)npy_cimagf(v._npy)))"""
needs['pyobj_from_string1'] = ['string']
cppmacros['pyobj_from_string1'] = """
#define pyobj_from_string1(v) (PyUnicode_FromString((char *)v))"""
needs['pyobj_from_string1size'] = ['string']
cppmacros['pyobj_from_string1size'] = """
#define pyobj_from_string1size(v,len) (PyUnicode_FromStringAndSize((char *)v, len))"""
needs['TRYPYARRAYTEMPLATE'] = ['PRINTPYOBJERR']
cppmacros['TRYPYARRAYTEMPLATE'] = """
/* New SciPy */
#define TRYPYARRAYTEMPLATECHAR case NPY_STRING: *(char *)(PyArray_DATA(arr))=*v; break;
#define TRYPYARRAYTEMPLATELONG case NPY_LONG: *(long *)(PyArray_DATA(arr))=*v; break;
#define TRYPYARRAYTEMPLATEOBJECT case NPY_OBJECT: PyArray_SETITEM(arr,PyArray_DATA(arr),pyobj_from_ ## ctype ## 1(*v)); break;

#define TRYPYARRAYTEMPLATE(ctype,typecode) \\
        PyArrayObject *arr = NULL;\\
        if (!obj) return -2;\\
        if (!PyArray_Check(obj)) return -1;\\
        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,\"TRYPYARRAYTEMPLATE:\");PRINTPYOBJERR(obj);return 0;}\\
        if (PyArray_DESCR(arr)->type==typecode)  {*(ctype *)(PyArray_DATA(arr))=*v; return 1;}\\
        switch (PyArray_TYPE(arr)) {\\
                case NPY_DOUBLE: *(npy_double *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_INT: *(npy_int *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_LONG: *(npy_long *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_FLOAT: *(npy_float *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_CDOUBLE: *(npy_double *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_CFLOAT: *(npy_float *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=(*v!=0); break;\\
                case NPY_UBYTE: *(npy_ubyte *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_BYTE: *(npy_byte *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_SHORT: *(npy_short *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_OBJECT: PyArray_SETITEM(arr, PyArray_DATA(arr), pyobj_from_ ## ctype ## 1(*v)); break;\\
        default: return -2;\\
        };\\
        return 1
"""

needs['TRYCOMPLEXPYARRAYTEMPLATE'] = ['PRINTPYOBJERR', 'npy_math.h']
cppmacros['TRYCOMPLEXPYARRAYTEMPLATE'] = """
#define TRYCOMPLEXPYARRAYTEMPLATEOBJECT case NPY_OBJECT: PyArray_SETITEM(arr, PyArray_DATA(arr), pyobj_from_complex_ ## ctype ## 1((*v))); break;
/* Helper to extract real/imag from complex value via cast to npy_cdouble */
static inline double _f2py_creal_as_double(const void *p, int typenum) {
    if (typenum == NPY_CFLOAT) return (double)npy_crealf(*(const npy_cfloat *)p);
    if (typenum == NPY_CLONGDOUBLE) return (double)npy_creall(*(const npy_clongdouble *)p);
    return npy_creal(*(const npy_cdouble *)p);
}
static inline double _f2py_cimag_as_double(const void *p, int typenum) {
    if (typenum == NPY_CFLOAT) return (double)npy_cimagf(*(const npy_cfloat *)p);
    if (typenum == NPY_CLONGDOUBLE) return (double)npy_cimagl(*(const npy_clongdouble *)p);
    return npy_cimag(*(const npy_cdouble *)p);
}
#define TRYCOMPLEXPYARRAYTEMPLATE(ctype,typenum)\\
        PyArrayObject *arr = NULL;\\
        double _re, _im;\\
        if (!obj) return -2;\\
        if (!PyArray_Check(obj)) return -1;\\
        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,\"TRYCOMPLEXPYARRAYTEMPLATE:\");PRINTPYOBJERR(obj);return 0;}\\
        _re = _f2py_creal_as_double(v, typenum);\\
        _im = _f2py_cimag_as_double(v, typenum);\\
        if (PyArray_TYPE(arr)==typenum) {\\
            memcpy(PyArray_DATA(arr), v, sizeof(*v));\\
            return 1;\\
        }\\
        switch (PyArray_TYPE(arr)) {\\
                case NPY_CDOUBLE: npy_csetreal((npy_cdouble *)PyArray_DATA(arr), _re);\\
                                  npy_csetimag((npy_cdouble *)PyArray_DATA(arr), _im);\\
                                  break;\\
                case NPY_CFLOAT: npy_csetrealf((npy_cfloat *)PyArray_DATA(arr), (float)_re);\\
                                 npy_csetimagf((npy_cfloat *)PyArray_DATA(arr), (float)_im);\\
                                 break;\\
                case NPY_DOUBLE: *(npy_double *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_LONG: *(npy_long *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_FLOAT: *(npy_float *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_INT: *(npy_int *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_SHORT: *(npy_short *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_UBYTE: *(npy_ubyte *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_BYTE: *(npy_byte *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=(_re!=0 && _im!=0); break;\\
                case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=_re; break;\\
                case NPY_CLONGDOUBLE: npy_csetreall((npy_clongdouble *)PyArray_DATA(arr), (npy_longdouble)_re);\\
                                      npy_csetimagl((npy_clongdouble *)PyArray_DATA(arr), (npy_longdouble)_im);\\
                                      break;\\
                case NPY_OBJECT: PyArray_SETITEM(arr, PyArray_DATA(arr), pyobj_from_complex_ ## ctype ## 1((*v))); break;\\
                default: return -2;\\
        };\\
        return -1;
"""
# cppmacros['NUMFROMARROBJ']="""
# define NUMFROMARROBJ(typenum,ctype) \\
#     if (PyArray_Check(obj)) arr = (PyArrayObject *)obj;\\
#     else arr = (PyArrayObject *)PyArray_ContiguousFromObject(obj,typenum,0,0);\\
#     if (arr) {\\
#         if (PyArray_TYPE(arr)==NPY_OBJECT) {\\
#             if (!ctype ## _from_pyobj(v,(PyArray_DESCR(arr)->getitem)(PyArray_DATA(arr)),\"\"))\\
#             goto capi_fail;\\
#         } else {\\
#             (PyArray_DESCR(arr)->cast[typenum])(PyArray_DATA(arr),1,(char*)v,1,1);\\
#         }\\
#         if ((PyObject *)arr != obj) { Py_DECREF(arr); }\\
#         return 1;\\
#     }
# """
# XXX: Note that CNUMFROMARROBJ is identical with NUMFROMARROBJ
# cppmacros['CNUMFROMARROBJ']="""
# define CNUMFROMARROBJ(typenum,ctype) \\
#     if (PyArray_Check(obj)) arr = (PyArrayObject *)obj;\\
#     else arr = (PyArrayObject *)PyArray_ContiguousFromObject(obj,typenum,0,0);\\
#     if (arr) {\\
#         if (PyArray_TYPE(arr)==NPY_OBJECT) {\\
#             if (!ctype ## _from_pyobj(v,(PyArray_DESCR(arr)->getitem)(PyArray_DATA(arr)),\"\"))\\
#             goto capi_fail;\\
#         } else {\\
#             (PyArray_DESCR(arr)->cast[typenum])((void *)(PyArray_DATA(arr)),1,(void *)(v),1,1);\\
#         }\\
#         if ((PyObject *)arr != obj) { Py_DECREF(arr); }\\
#         return 1;\\
#     }
# """


needs['GETSTRFROMPYTUPLE'] = ['STRINGCOPYN', 'PRINTPYOBJERR']
cppmacros['GETSTRFROMPYTUPLE'] = """
#define GETSTRFROMPYTUPLE(tuple,index,str,len) {\\
        PyObject *rv_cb_str = PyTuple_GetItem((tuple),(index));\\
        if (rv_cb_str == NULL)\\
            goto capi_fail;\\
        if (PyBytes_Check(rv_cb_str)) {\\
            str[len-1]='\\0';\\
            STRINGCOPYN((str),PyBytes_AS_STRING((PyBytesObject*)rv_cb_str),(len));\\
        } else {\\
            PRINTPYOBJERR(rv_cb_str);\\
            PyErr_SetString(#modulename#_error,\"string object expected\");\\
            goto capi_fail;\\
        }\\
    }
"""
cppmacros['GETSCALARFROMPYTUPLE'] = """
#define GETSCALARFROMPYTUPLE(tuple,index,var,ctype,mess) {\\
        if ((capi_tmp = PyTuple_GetItem((tuple),(index)))==NULL) goto capi_fail;\\
        if (!(ctype ## _from_pyobj((var),capi_tmp,mess)))\\
            goto capi_fail;\\
    }
"""

cppmacros['FAILNULL'] = """\
#define FAILNULL(p) do {                                            \\
    if ((p) == NULL) {                                              \\
        PyErr_SetString(PyExc_MemoryError, "NULL pointer found");   \\
        goto capi_fail;                                             \\
    }                                                               \\
} while (0)
"""
needs['MEMCOPY'] = ['string.h', 'FAILNULL']
cppmacros['MEMCOPY'] = """
#define MEMCOPY(to,from,n)\\
    do { FAILNULL(to); FAILNULL(from); (void)memcpy(to,from,n); } while (0)
"""
cppmacros['STRINGMALLOC'] = """
#define STRINGMALLOC(str,len)\\
    if ((str = (string)malloc(len+1)) == NULL) {\\
        PyErr_SetString(PyExc_MemoryError, \"out of memory\");\\
        goto capi_fail;\\
    } else {\\
        (str)[len] = '\\0';\\
    }
"""
cppmacros['STRINGFREE'] = """
#define STRINGFREE(str) do {if (!(str == NULL)) free(str);} while (0)
"""
needs['STRINGPADN'] = ['string.h']
cppmacros['STRINGPADN'] = """
/*
STRINGPADN replaces null values with padding values from the right.

`to` must have size of at least N bytes.

If the `to[N-1]` has null value, then replace it and all the
preceding, nulls with the given padding.

STRINGPADN(to, N, PADDING, NULLVALUE) is an inverse operation.
*/
#define STRINGPADN(to, N, NULLVALUE, PADDING)                   \\
    do {                                                        \\
        int _m = (N);                                           \\
        char *_to = (to);                                       \\
        for (_m -= 1; _m >= 0 && _to[_m] == NULLVALUE; _m--) {  \\
             _to[_m] = PADDING;                                 \\
        }                                                       \\
    } while (0)
"""
needs['STRINGCOPYN'] = ['string.h', 'FAILNULL']
cppmacros['STRINGCOPYN'] = """
/*
STRINGCOPYN copies N bytes.

`to` and `from` buffers must have sizes of at least N bytes.
*/
#define STRINGCOPYN(to,from,N)                                  \\
    do {                                                        \\
        int _m = (N);                                           \\
        char *_to = (to);                                       \\
        char *_from = (from);                                   \\
        FAILNULL(_to); FAILNULL(_from);                         \\
        (void)strncpy(_to, _from, _m);             \\
    } while (0)
"""
needs['STRINGCOPY'] = ['string.h', 'FAILNULL']
cppmacros['STRINGCOPY'] = """
#define STRINGCOPY(to,from)\\
    do { FAILNULL(to); FAILNULL(from); (void)strcpy(to,from); } while (0)
"""
cppmacros['CHECKGENERIC'] = """
#define CHECKGENERIC(check,tcheck,name) \\
    if (!(check)) {\\
        PyErr_SetString(#modulename#_error,\"(\"tcheck\") failed for \"name);\\
        /*goto capi_fail;*/\\
    } else """
cppmacros['CHECKARRAY'] = """
#define CHECKARRAY(check,tcheck,name) \\
    if (!(check)) {\\
        PyErr_SetString(#modulename#_error,\"(\"tcheck\") failed for \"name);\\
        /*goto capi_fail;*/\\
    } else """
cppmacros['CHECKSTRING'] = """
#define CHECKSTRING(check,tcheck,name,show,var)\\
    if (!(check)) {\\
        PyErr_Format(#modulename#_error, \"(\"tcheck\") failed for \"name\": \"show, slen(var), var);\\
        /*goto capi_fail;*/\\
    } else """
cppmacros['CHECKSCALAR'] = """
#define CHECKSCALAR(check,tcheck,name,show,var)\\
    if (!(check)) {\\
        PyErr_Format(#modulename#_error, \"(\"tcheck\") failed for \"name\": \"show, var);\\
        /*goto capi_fail;*/\\
    } else """
# cppmacros['CHECKDIMS']="""
# define CHECKDIMS(dims,rank) \\
#     for (int i=0;i<(rank);i++)\\
#         if (dims[i]<0) {\\
#             fprintf(stderr,\"Unspecified array argument requires a complete dimension specification.\\n\");\\
#             goto capi_fail;\\
#         }
# """
cppmacros[
    'ARRSIZE'] = '#define ARRSIZE(dims,rank) (_PyArray_multiply_list(dims,rank))'
cppmacros['OLDPYNUM'] = """
#ifdef OLDPYNUM
#error You need to install NumPy version 0.13 or higher. See https://scipy.org/install.html
#endif
"""

# Defining the correct value to indicate thread-local storage in C without
# running a compile-time check (which we have no control over in generated
# code used outside of NumPy) is hard. Therefore we support overriding this
# via an external define - the f2py-using package can then use the same
# compile-time checks as we use for `NPY_TLS` when building NumPy (see
# scipy#21860 for an example of that).
#
# __STDC_NO_THREADS__ should not be coupled to the availability of _Thread_local.
# In case we get a bug report, guard it with __STDC_NO_THREADS__ after all.
#
# `thread_local` has become a keyword in C23, but don't try to use that yet
# (too new, doing so while C23 support is preliminary will likely cause more
#  problems than it solves).
#
# Note: do not try to use `threads.h`, its availability is very low
# *and* threads.h isn't actually used where `F2PY_THREAD_LOCAL_DECL` is
# in the generated code. See gh-27718 for more details.
cppmacros["F2PY_THREAD_LOCAL_DECL"] = """
#ifndef F2PY_THREAD_LOCAL_DECL
#if defined(_MSC_VER)
#define F2PY_THREAD_LOCAL_DECL __declspec(thread)
#elif defined(NPY_OS_MINGW)
#define F2PY_THREAD_LOCAL_DECL __thread
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define F2PY_THREAD_LOCAL_DECL _Thread_local
#elif defined(__GNUC__) \\
      && (__GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 4)))
#define F2PY_THREAD_LOCAL_DECL __thread
#endif
#endif
"""
################# C functions ###############

cfuncs['calcarrindex'] = """
static int calcarrindex(int *i,PyArrayObject *arr) {
    int k,ii = i[0];
    for (k=1; k < PyArray_NDIM(arr); k++)
        ii += (ii*(PyArray_DIM(arr,k) - 1)+i[k]); /* assuming contiguous arr */
    return ii;
}"""
cfuncs['calcarrindextr'] = """
static int calcarrindextr(int *i,PyArrayObject *arr) {
    int k,ii = i[PyArray_NDIM(arr)-1];
    for (k=1; k < PyArray_NDIM(arr); k++)
        ii += (ii*(PyArray_DIM(arr,PyArray_NDIM(arr)-k-1) - 1)+i[PyArray_NDIM(arr)-k-1]); /* assuming contiguous arr */
    return ii;
}"""
cfuncs['forcomb'] = """
struct ForcombCache { int nd;npy_intp *d;int *i,*i_tr,tr; };
static int initforcomb(struct ForcombCache *cache, npy_intp *dims,int nd,int tr) {
  int k;
  if (dims==NULL) return 0;
  if (nd<0) return 0;
  cache->nd = nd;
  cache->d = dims;
  cache->tr = tr;

  cache->i = (int *)malloc(sizeof(int)*nd);
  if (cache->i==NULL) return 0;
  cache->i_tr = (int *)malloc(sizeof(int)*nd);
  if (cache->i_tr==NULL) {free(cache->i); return 0;};

  for (k=1;k<nd;k++) {
    cache->i[k] = cache->i_tr[nd-k-1] = 0;
  }
  cache->i[0] = cache->i_tr[nd-1] = -1;
  return 1;
}
static int *nextforcomb(struct ForcombCache *cache) {
  if (cache==NULL) return NULL;
  int j,*i,*i_tr,k;
  int nd=cache->nd;
  if ((i=cache->i) == NULL) return NULL;
  if ((i_tr=cache->i_tr) == NULL) return NULL;
  if (cache->d == NULL) return NULL;
  i[0]++;
  if (i[0]==cache->d[0]) {
    j=1;
    while ((j<nd) && (i[j]==cache->d[j]-1)) j++;
    if (j==nd) {
      free(i);
      free(i_tr);
      return NULL;
    }
    for (k=0;k<j;k++) i[k] = i_tr[nd-k-1] = 0;
    i[j]++;
    i_tr[nd-j-1]++;
  } else
    i_tr[nd-1]++;
  if (cache->tr) return i_tr;
  return i;
}"""
needs['try_pyarr_from_string'] = ['STRINGCOPYN', 'PRINTPYOBJERR', 'string']
cfuncs['try_pyarr_from_string'] = """
/*
  try_pyarr_from_string copies str[:len(obj)] to the data of an `ndarray`.

  If obj is an `ndarray`, it is assumed to be contiguous.

  If the specified len==-1, str must be null-terminated.
*/
static int try_pyarr_from_string(PyObject *obj,
                                 const string str, const int len) {
#ifdef DEBUGCFUNCS
fprintf(stderr, "try_pyarr_from_string(str='%s', len=%d, obj=%p)\\n",
        (char*)str,len, obj);
#endif
    if (!obj) return -2; /* Object missing */
    if (obj == Py_None) return -1; /* None */
    if (!PyArray_Check(obj)) goto capi_fail; /* not an ndarray */
    if (PyArray_Check(obj)) {
        PyArrayObject *arr = (PyArrayObject *)obj;
        assert(ISCONTIGUOUS(arr));
        string buf = PyArray_DATA(arr);
        npy_intp n = len;
        if (n == -1) {
            /* Assuming null-terminated str. */
            n = strlen(str);
        }
        if (n > PyArray_NBYTES(arr)) {
            n = PyArray_NBYTES(arr);
        }
        STRINGCOPYN(buf, str, n);
        return 1;
    }
capi_fail:
    PRINTPYOBJERR(obj);
    PyErr_SetString(#modulename#_error, \"try_pyarr_from_string failed\");
    return 0;
}
"""
needs['string_from_pyobj'] = ['string', 'STRINGMALLOC', 'STRINGCOPYN']
cfuncs['string_from_pyobj'] = """
/*
  Create a new string buffer `str` of at most length `len` from a
  Python string-like object `obj`.

  The string buffer has given size (len) or the size of inistr when len==-1.

  The string buffer is padded with blanks: in Fortran, trailing blanks
  are insignificant contrary to C nulls.
 */
static int
string_from_pyobj(string *str, int *len, const string inistr, PyObject *obj,
                  const char *errmess)
{
    PyObject *tmp = NULL;
    string buf = NULL;
    npy_intp n = -1;
#ifdef DEBUGCFUNCS
fprintf(stderr,\"string_from_pyobj(str='%s',len=%d,inistr='%s',obj=%p)\\n\",
               (char*)str, *len, (char *)inistr, obj);
#endif
    if (obj == Py_None) {
        n = strlen(inistr);
        buf = inistr;
    }
    else if (PyArray_Check(obj)) {
        PyArrayObject *arr = (PyArrayObject *)obj;
        if (!ISCONTIGUOUS(arr)) {
            PyErr_SetString(PyExc_ValueError,
                            \"array object is non-contiguous.\");
            goto capi_fail;
        }
        n = PyArray_NBYTES(arr);
        buf = PyArray_DATA(arr);
        n = strnlen(buf, n);
    }
    else {
        if (PyBytes_Check(obj)) {
            tmp = obj;
            Py_INCREF(tmp);
        }
        else if (PyUnicode_Check(obj)) {
            tmp = PyUnicode_AsASCIIString(obj);
        }
        else {
            PyObject *tmp2;
            tmp2 = PyObject_Str(obj);
            if (tmp2) {
                tmp = PyUnicode_AsASCIIString(tmp2);
                Py_DECREF(tmp2);
            }
            else {
                tmp = NULL;
            }
        }
        if (tmp == NULL) goto capi_fail;
        n = PyBytes_GET_SIZE(tmp);
        buf = PyBytes_AS_STRING(tmp);
    }
    if (*len == -1) {
        /* TODO: change the type of `len` so that we can remove this */
        if (n > NPY_MAX_INT) {
            PyErr_SetString(PyExc_OverflowError,
                            "object too large for a 32-bit int");
            goto capi_fail;
        }
        *len = n;
    }
    else if (*len < n) {
        /* discard the last (len-n) bytes of input buf */
        n = *len;
    }
    if (n < 0 || *len < 0 || buf == NULL) {
        goto capi_fail;
    }
    STRINGMALLOC(*str, *len);  // *str is allocated with size (*len + 1)
    if (n < *len) {
        /*
          Pad fixed-width string with nulls. The caller will replace
          nulls with blanks when the corresponding argument is not
          intent(c).
        */
        memset(*str + n, '\\0', *len - n);
    }
    STRINGCOPYN(*str, buf, n);
    Py_XDECREF(tmp);
    return 1;
capi_fail:
    Py_XDECREF(tmp);
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = #modulename#_error;
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}
"""

cfuncs['character_from_pyobj'] = """
static int
character_from_pyobj(character* v, PyObject *obj, const char *errmess) {
    if (PyBytes_Check(obj)) {
        /* empty bytes has trailing null, so dereferencing is always safe */
        *v = PyBytes_AS_STRING(obj)[0];
        return 1;
    } else if (PyUnicode_Check(obj)) {
        PyObject* tmp = PyUnicode_AsASCIIString(obj);
        if (tmp != NULL) {
            *v = PyBytes_AS_STRING(tmp)[0];
            Py_DECREF(tmp);
            return 1;
        }
    } else if (PyArray_Check(obj)) {
        PyArrayObject* arr = (PyArrayObject*)obj;
        if (F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(arr)) {
            *v = PyArray_BYTES(arr)[0];
            return 1;
        } else if (F2PY_IS_UNICODE_ARRAY(arr)) {
            // TODO: update when numpy will support 1-byte and
            // 2-byte unicode dtypes
            PyObject* tmp = PyUnicode_FromKindAndData(
                              PyUnicode_4BYTE_KIND,
                              PyArray_BYTES(arr),
                              (PyArray_NBYTES(arr)>0?1:0));
            if (tmp != NULL) {
                if (character_from_pyobj(v, tmp, errmess)) {
                    Py_DECREF(tmp);
                    return 1;
                }
                Py_DECREF(tmp);
            }
        }
    } else if (PySequence_Check(obj)) {
        PyObject* tmp = PySequence_GetItem(obj,0);
        if (tmp != NULL) {
            if (character_from_pyobj(v, tmp, errmess)) {
                Py_DECREF(tmp);
                return 1;
            }
            Py_DECREF(tmp);
        }
    }
    {
        PyObject *describe_obj;
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = PyExc_TypeError;
            Py_INCREF(err);
        }
        else {
            Py_INCREF(err);
            PyErr_Clear();
        }
        describe_obj = f2py_describe_obj(obj);
        if (describe_obj) {
            PyErr_Format(err, "%s -- expected str|bytes|sequence-of-str-or-bytes, got %U", errmess, describe_obj);
            Py_DECREF(describe_obj);
        }
        Py_DECREF(err);
    }
    return 0;
}
"""

# TODO: These should be dynamically generated, too many mapped to int things,
# see note in _isocbind.py
needs['char_from_pyobj'] = ['int_from_pyobj']
cfuncs['char_from_pyobj'] = """
static int
char_from_pyobj(char* v, PyObject *obj, const char *errmess) {
    int i = 0;
    if (int_from_pyobj(&i, obj, errmess)) {
        *v = (char)i;
        return 1;
    }
    return 0;
}
"""


needs['signed_char_from_pyobj'] = ['int_from_pyobj', 'signed_char']
cfuncs['signed_char_from_pyobj'] = """
static int
signed_char_from_pyobj(signed_char* v, PyObject *obj, const char *errmess) {
    int i = 0;
    if (int_from_pyobj(&i, obj, errmess)) {
        *v = (signed_char)i;
        return 1;
    }
    return 0;
}
"""


needs['short_from_pyobj'] = ['int_from_pyobj']
cfuncs['short_from_pyobj'] = """
static int
short_from_pyobj(short* v, PyObject *obj, const char *errmess) {
    int i = 0;
    if (int_from_pyobj(&i, obj, errmess)) {
        *v = (short)i;
        return 1;
    }
    return 0;
}
"""


cfuncs['int_from_pyobj'] = """
static int
int_from_pyobj(int* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;

    if (PyLong_Check(obj)) {
        *v = Npy__PyLong_AsInt(obj);
        return !(*v == -1 && PyErr_Occurred());
    }

    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = Npy__PyLong_AsInt(tmp);
        Py_DECREF(tmp);
        return !(*v == -1 && PyErr_Occurred());
    }

    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj,\"real\");
    }
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    if (tmp) {
        if (int_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }

    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = #modulename#_error;
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}
"""


cfuncs['long_from_pyobj'] = """
static int
long_from_pyobj(long* v, PyObject *obj, const char *errmess) {
    PyObject* tmp = NULL;

    if (PyLong_Check(obj)) {
        *v = PyLong_AsLong(obj);
        return !(*v == -1 && PyErr_Occurred());
    }

    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = PyLong_AsLong(tmp);
        Py_DECREF(tmp);
        return !(*v == -1 && PyErr_Occurred());
    }

    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj,\"real\");
    }
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    if (tmp) {
        if (long_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = #modulename#_error;
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}
"""


needs['long_long_from_pyobj'] = ['long_long']
cfuncs['long_long_from_pyobj'] = """
static int
long_long_from_pyobj(long_long* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;

    if (PyLong_Check(obj)) {
        *v = PyLong_AsLongLong(obj);
        return !(*v == -1 && PyErr_Occurred());
    }

    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = PyLong_AsLongLong(tmp);
        Py_DECREF(tmp);
        return !(*v == -1 && PyErr_Occurred());
    }

    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj,\"real\");
    }
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    if (tmp) {
        if (long_long_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = #modulename#_error;
        }
        PyErr_SetString(err,errmess);
    }
    return 0;
}
"""


needs['long_double_from_pyobj'] = ['double_from_pyobj', 'long_double']
cfuncs['long_double_from_pyobj'] = """
static int
long_double_from_pyobj(long_double* v, PyObject *obj, const char *errmess)
{
    double d=0;
    if (PyArray_CheckScalar(obj)){
        if PyArray_IsScalar(obj, LongDouble) {
            PyArray_ScalarAsCtype(obj, v);
            return 1;
        }
        else if (PyArray_Check(obj)) {
            PyArrayObject *arr = (PyArrayObject *)obj;
            if (PyArray_TYPE(arr) == NPY_LONGDOUBLE) {
                (*v) = *((npy_longdouble *)PyArray_DATA(arr));
                return 1;
            }
        }
    }
    if (double_from_pyobj(&d, obj, errmess)) {
        *v = (long_double)d;
        return 1;
    }
    return 0;
}
"""


cfuncs['double_from_pyobj'] = """
static int
double_from_pyobj(double* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;
    if (PyFloat_Check(obj)) {
        *v = PyFloat_AsDouble(obj);
        return !(*v == -1.0 && PyErr_Occurred());
    }

    tmp = PyNumber_Float(obj);
    if (tmp) {
        *v = PyFloat_AsDouble(tmp);
        Py_DECREF(tmp);
        return !(*v == -1.0 && PyErr_Occurred());
    }

    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj,\"real\");
    }
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    if (tmp) {
        if (double_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL) err = #modulename#_error;
        PyErr_SetString(err,errmess);
    }
    return 0;
}
"""


needs['float_from_pyobj'] = ['double_from_pyobj']
cfuncs['float_from_pyobj'] = """
static int
float_from_pyobj(float* v, PyObject *obj, const char *errmess)
{
    double d=0.0;
    if (double_from_pyobj(&d,obj,errmess)) {
        *v = (float)d;
        return 1;
    }
    return 0;
}
"""


needs['float16_from_pyobj'] = ['double_from_pyobj', 'float16']
cfuncs['float16_from_pyobj'] = """
static int
float16_from_pyobj(float16* v, PyObject *obj, const char *errmess)
{
    double d = 0.0;
    /* every float16 value is a double, so the conversion is exact */
    if (double_from_pyobj(&d, obj, errmess)) {
        *v = (float16)d;
        return 1;
    }
    return 0;
}
"""


needs['unsigned_long_long_from_pyobj'] = ['unsigned_long_long']
cfuncs['unsigned_long_long_from_pyobj'] = """
static int
_f2py_ulonglong_from_pyobj(unsigned long long *v, PyObject *obj, const char *errmess)
{
    PyObject *tmp = NULL;
    if (PyLong_Check(obj)) {
        *v = PyLong_AsUnsignedLongLong(obj);
        return !(*v == (unsigned long long)-1 && PyErr_Occurred());
    }
    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = PyLong_AsUnsignedLongLong(tmp);
        Py_DECREF(tmp);
        return !(*v == (unsigned long long)-1 && PyErr_Occurred());
    }
    if (PySequence_Check(obj) && !PyBytes_Check(obj) && !PyUnicode_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
        if (tmp) {
            int ok = _f2py_ulonglong_from_pyobj(v, tmp, errmess);
            Py_DECREF(tmp);
            return ok;
        }
    }
    {
        PyObject *err = PyErr_Occurred();
        if (err == NULL) {
            err = #modulename#_error;
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}
static int
unsigned_long_long_from_pyobj(unsigned_long_long *v, PyObject *obj, const char *errmess)
{
    unsigned long long x = 0;
    if (!_f2py_ulonglong_from_pyobj(&x, obj, errmess)) {
        return 0;
    }
    *v = x;
    return 1;
}
"""


def _unsigned_from_pyobj(ctype, limit):
    return f"""
static int
{ctype}_from_pyobj({ctype} *v, PyObject *obj, const char *errmess)
{{
    unsigned long long x = 0;
    if (!_f2py_ulonglong_from_pyobj(&x, obj, errmess)) {{
        return 0;
    }}
    if (x > (unsigned long long)({limit})) {{
        PyErr_SetString(PyExc_OverflowError, errmess);
        return 0;
    }}
    *v = ({ctype})x;
    return 1;
}}
"""


needs['unsigned_char_from_pyobj'] = [
    'unsigned_char', 'unsigned_long_long_from_pyobj', 'limits.h']
cfuncs['unsigned_char_from_pyobj'] = _unsigned_from_pyobj(
    'unsigned_char', 'UCHAR_MAX')
needs['unsigned_short_from_pyobj'] = [
    'unsigned_short', 'unsigned_long_long_from_pyobj', 'limits.h']
cfuncs['unsigned_short_from_pyobj'] = _unsigned_from_pyobj(
    'unsigned_short', 'USHRT_MAX')
needs['unsigned_from_pyobj'] = ['unsigned_long_long_from_pyobj', 'limits.h']
cfuncs['unsigned_from_pyobj'] = _unsigned_from_pyobj('unsigned', 'UINT_MAX')


needs['complex_long_double_from_pyobj'] = ['complex_long_double', 'long_double',
                                           'complex_double_from_pyobj', 'npy_math.h']
cfuncs['complex_long_double_from_pyobj'] = """
static int
complex_long_double_from_pyobj(complex_long_double* v, PyObject *obj, const char *errmess)
{
    complex_double cd = {.r=0, .i=0};
    if (PyArray_CheckScalar(obj)){
        if PyArray_IsScalar(obj, CLongDouble) {
            PyArray_ScalarAsCtype(obj, v);
            return 1;
        }
        else if (PyArray_Check(obj)) {
            PyArrayObject *arr = (PyArrayObject *)obj;
            if (PyArray_TYPE(arr)==NPY_CLONGDOUBLE) {
                npy_clongdouble tmp = *(npy_clongdouble *)PyArray_DATA(arr);
                npy_csetreall(&v->_npy, npy_creall(tmp));
                npy_csetimagl(&v->_npy, npy_cimagl(tmp));
                return 1;
            }
        }
    }
    if (complex_double_from_pyobj(&cd,obj,errmess)) {
        npy_csetreall(&v->_npy, (long_double)npy_creal(cd._npy));
        npy_csetimagl(&v->_npy, (long_double)npy_cimag(cd._npy));
        return 1;
    }
    return 0;
}
"""


needs['complex_double_from_pyobj'] = ['complex_double', 'npy_math.h']
cfuncs['complex_double_from_pyobj'] = """
static int
complex_double_from_pyobj(complex_double* v, PyObject *obj, const char *errmess) {
    Py_complex c;
    if (PyComplex_Check(obj)) {
        c = PyComplex_AsCComplex(obj);
        npy_csetreal(&v->_npy, c.real);
        npy_csetimag(&v->_npy, c.imag);
        return 1;
    }
    if (PyArray_IsScalar(obj, ComplexFloating)) {
        if (PyArray_IsScalar(obj, CFloat)) {
            npy_cfloat tmp;
            PyArray_ScalarAsCtype(obj, &tmp);
            npy_csetreal(&v->_npy, (double)npy_crealf(tmp));
            npy_csetimag(&v->_npy, (double)npy_cimagf(tmp));
        }
        else if (PyArray_IsScalar(obj, CLongDouble)) {
            npy_clongdouble tmp;
            PyArray_ScalarAsCtype(obj, &tmp);
            npy_csetreal(&v->_npy, (double)npy_creall(tmp));
            npy_csetimag(&v->_npy, (double)npy_cimagl(tmp));
        }
        else { /* if (PyArray_IsScalar(obj, CDouble)) */
            PyArray_ScalarAsCtype(obj, v);
        }
        return 1;
    }
    if (PyArray_CheckScalar(obj)) { /* 0-dim array or still array scalar */
        PyArrayObject *arr;
        if (PyArray_Check(obj)) {
            arr = (PyArrayObject *)PyArray_Cast((PyArrayObject *)obj, NPY_CDOUBLE);
        }
        else {
            arr = (PyArrayObject *)PyArray_FromScalar(obj, PyArray_DescrFromType(NPY_CDOUBLE));
        }
        if (arr == NULL) {
            return 0;
        }
        npy_cdouble tmp = *(npy_cdouble *)PyArray_DATA(arr);
        npy_csetreal(&v->_npy, npy_creal(tmp));
        npy_csetimag(&v->_npy, npy_cimag(tmp));
        Py_DECREF(arr);
        return 1;
    }
    /* Python does not provide PyNumber_Complex function :-( */
    npy_csetimag(&v->_npy, 0.0);
    if (PyFloat_Check(obj)) {
        npy_csetreal(&v->_npy, PyFloat_AsDouble(obj));
        return !(npy_creal(v->_npy) == -1.0 && PyErr_Occurred());
    }
    if (PyLong_Check(obj)) {
        npy_csetreal(&v->_npy, PyLong_AsDouble(obj));
        return !(npy_creal(v->_npy) == -1.0 && PyErr_Occurred());
    }
    if (PySequence_Check(obj) && !(PyBytes_Check(obj) || PyUnicode_Check(obj))) {
        PyObject *tmp = PySequence_GetItem(obj,0);
        if (tmp) {
            if (complex_double_from_pyobj(v,tmp,errmess)) {
                Py_DECREF(tmp);
                return 1;
            }
            Py_DECREF(tmp);
        }
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL)
            err = PyExc_TypeError;
        PyErr_SetString(err,errmess);
    }
    return 0;
}
"""


needs['complex_float_from_pyobj'] = [
    'complex_float', 'complex_double_from_pyobj']
cfuncs['complex_float_from_pyobj'] = """
static int
complex_float_from_pyobj(complex_float* v,PyObject *obj,const char *errmess)
{
    complex_double cd = {.r=0, .i=0};
    if (complex_double_from_pyobj(&cd,obj,errmess)) {
        npy_csetrealf(&v->_npy, (float)npy_creal(cd._npy));
        npy_csetimagf(&v->_npy, (float)npy_cimag(cd._npy));
        return 1;
    }
    return 0;
}
"""


cfuncs['try_pyarr_from_character'] = """
static int try_pyarr_from_character(PyObject* obj, character* v) {
    PyArrayObject *arr = (PyArrayObject*)obj;
    if (!obj) return -2;
    if (PyArray_Check(obj)) {
        if (F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(arr))  {
            *(character *)(PyArray_DATA(arr)) = *v;
            return 1;
        }
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            PyObject *describe_obj;
            err = PyExc_ValueError;
            describe_obj = f2py_describe_obj(obj);
            if (describe_obj) {
                PyErr_Format(
                    err,
                    "try_pyarr_from_character failed  -- expected bytes array-scalar|array, got %U",
                    describe_obj
                );
                Py_DECREF(describe_obj);
            }
        }
    }
    return 0;
}
"""

needs['try_pyarr_from_char'] = ['pyobj_from_char1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_char'] = 'static int try_pyarr_from_char(PyObject* obj,char* v) {\n    TRYPYARRAYTEMPLATE(char,\'c\');\n}\n'
needs['try_pyarr_from_unsigned_char'] = [
    'pyobj_from_unsigned_char1', 'TRYPYARRAYTEMPLATE', 'unsigned_char']
cfuncs[
    'try_pyarr_from_unsigned_char'] = 'static int try_pyarr_from_unsigned_char(PyObject* obj,unsigned_char* v) {\n    TRYPYARRAYTEMPLATE(unsigned_char,\'b\');\n}\n'
needs['try_pyarr_from_unsigned_short'] = [
    'pyobj_from_unsigned_short1', 'TRYPYARRAYTEMPLATE', 'unsigned_short']
cfuncs[
    'try_pyarr_from_unsigned_short'] = 'static int try_pyarr_from_unsigned_short(PyObject* obj,unsigned_short* v) {\n    TRYPYARRAYTEMPLATE(unsigned_short,\'H\');\n}\n'
needs['try_pyarr_from_unsigned'] = ['pyobj_from_unsigned1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_unsigned'] = 'static int try_pyarr_from_unsigned(PyObject* obj,unsigned* v) {\n    TRYPYARRAYTEMPLATE(unsigned,\'I\');\n}\n'
needs['try_pyarr_from_unsigned_long_long'] = [
    'pyobj_from_unsigned_long_long1', 'TRYPYARRAYTEMPLATE', 'unsigned_long_long']
cfuncs[
    'try_pyarr_from_unsigned_long_long'] = 'static int try_pyarr_from_unsigned_long_long(PyObject* obj,unsigned_long_long* v) {\n    TRYPYARRAYTEMPLATE(unsigned_long_long,\'Q\');\n}\n'
needs['try_pyarr_from_float16'] = [
    'pyobj_from_float161', 'TRYPYARRAYTEMPLATE', 'float16']
cfuncs[
    'try_pyarr_from_float16'] = 'static int try_pyarr_from_float16(PyObject* obj,float16* v) {\n    TRYPYARRAYTEMPLATE(float16,\'e\');\n}\n'
needs['try_pyarr_from_signed_char'] = ['TRYPYARRAYTEMPLATE', 'signed_char']
cfuncs[
    'try_pyarr_from_signed_char'] = 'static int try_pyarr_from_signed_char(PyObject* obj,signed_char* v) {\n    TRYPYARRAYTEMPLATE(signed_char,\'1\');\n}\n'
needs['try_pyarr_from_short'] = ['pyobj_from_short1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_short'] = 'static int try_pyarr_from_short(PyObject* obj,short* v) {\n    TRYPYARRAYTEMPLATE(short,\'s\');\n}\n'
needs['try_pyarr_from_int'] = ['pyobj_from_int1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_int'] = 'static int try_pyarr_from_int(PyObject* obj,int* v) {\n    TRYPYARRAYTEMPLATE(int,\'i\');\n}\n'
needs['try_pyarr_from_long'] = ['pyobj_from_long1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_long'] = 'static int try_pyarr_from_long(PyObject* obj,long* v) {\n    TRYPYARRAYTEMPLATE(long,\'l\');\n}\n'
needs['try_pyarr_from_long_long'] = [
    'pyobj_from_long_long1', 'TRYPYARRAYTEMPLATE', 'long_long']
cfuncs[
    'try_pyarr_from_long_long'] = 'static int try_pyarr_from_long_long(PyObject* obj,long_long* v) {\n    TRYPYARRAYTEMPLATE(long_long,\'L\');\n}\n'
needs['try_pyarr_from_float'] = ['pyobj_from_float1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_float'] = 'static int try_pyarr_from_float(PyObject* obj,float* v) {\n    TRYPYARRAYTEMPLATE(float,\'f\');\n}\n'
needs['try_pyarr_from_double'] = ['pyobj_from_double1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_double'] = 'static int try_pyarr_from_double(PyObject* obj,double* v) {\n    TRYPYARRAYTEMPLATE(double,\'d\');\n}\n'
needs['try_pyarr_from_long_double'] = [
    'pyobj_from_long_double1', 'TRYPYARRAYTEMPLATE', 'long_double']
cfuncs[
    'try_pyarr_from_long_double'] = 'static int try_pyarr_from_long_double(PyObject* obj,long_double* v) {\n    TRYPYARRAYTEMPLATE(long_double,\'g\');\n}\n'
needs['try_pyarr_from_complex_float'] = [
    'pyobj_from_complex_float1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_float']
cfuncs[
    'try_pyarr_from_complex_float'] = 'static int try_pyarr_from_complex_float(PyObject* obj,complex_float* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(float,NPY_CFLOAT);\n}\n'
needs['try_pyarr_from_complex_double'] = [
    'pyobj_from_complex_double1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_double']
cfuncs[
    'try_pyarr_from_complex_double'] = 'static int try_pyarr_from_complex_double(PyObject* obj,complex_double* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(double,NPY_CDOUBLE);\n}\n'
needs['try_pyarr_from_complex_long_double'] = [
    'pyobj_from_complex_long_double1', 'TRYCOMPLEXPYARRAYTEMPLATE',
    'complex_long_double']
cfuncs[
    'try_pyarr_from_complex_long_double'] = 'static int try_pyarr_from_complex_long_double(PyObject* obj,complex_long_double* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(long_double,NPY_CLONGDOUBLE);\n}\n'


needs['create_cb_arglist'] = ['CFUNCSMESS', 'PRINTPYOBJERR', 'MINMAX']
# create the list of arguments to be used when calling back to python
cfuncs['create_cb_arglist'] = """
static int
create_cb_arglist(PyObject* fun, PyTupleObject* xa , const int maxnofargs,
                  const int nofoptargs, int *nofargs, PyTupleObject **args,
                  const char *errmess)
{
    PyObject *tmp = NULL;
    PyObject *tmp_fun = NULL;
    Py_ssize_t tot, opt, ext, siz, i, di = 0;
    CFUNCSMESS(\"create_cb_arglist\\n\");
    tot=opt=ext=siz=0;
    /* Get the total number of arguments */
    if (PyFunction_Check(fun)) {
        tmp_fun = fun;
        Py_INCREF(tmp_fun);
    }
    else {
        di = 1;
        if (PyObject_HasAttrString(fun,\"im_func\")) {
            tmp_fun = PyObject_GetAttrString(fun,\"im_func\");
        }
        else if (PyObject_HasAttrString(fun,\"__call__\")) {
            tmp = PyObject_GetAttrString(fun,\"__call__\");
            if (PyObject_HasAttrString(tmp,\"im_func\"))
                tmp_fun = PyObject_GetAttrString(tmp,\"im_func\");
            else {
                tmp_fun = fun; /* built-in function */
                Py_INCREF(tmp_fun);
                tot = maxnofargs;
                if (PyCFunction_Check(fun)) {
                    /* In case the function has a co_argcount */
                    di = 0;
                }
                if (xa != NULL)
                    tot += PyTuple_Size((PyObject *)xa);
            }
            Py_XDECREF(tmp);
        }
        else if (PyFortran_Check(fun) || PyFortran_Check1(fun)) {
            tot = maxnofargs;
            if (xa != NULL)
                tot += PyTuple_Size((PyObject *)xa);
            tmp_fun = fun;
            Py_INCREF(tmp_fun);
        }
        else if (F2PyCapsule_Check(fun)) {
            tot = maxnofargs;
            if (xa != NULL)
                ext = PyTuple_Size((PyObject *)xa);
            if(ext>0) {
                fprintf(stderr,\"extra arguments tuple cannot be used with PyCapsule call-back\\n\");
                goto capi_fail;
            }
            tmp_fun = fun;
            Py_INCREF(tmp_fun);
        }
    }

    if (tmp_fun == NULL) {
        const char *tp_name = \"NULL\";
#if defined(Py_LIMITED_API)
        PyObject *tp_name_obj=NULL;
#endif
        if (fun != NULL) {
#if defined(Py_LIMITED_API)
            tp_name_obj = PyType_GetFullyQualifiedName(Py_TYPE(fun));
            if (!tp_name_obj) goto capi_fail;
            tp_name = PyUnicode_AsUTF8AndSize(tp_name_obj, NULL);
            if (tp_name == NULL) {
                Py_DECREF(tp_name_obj); goto capi_fail;
            }
#else
            tp_name = Py_TYPE(fun)->tp_name;
#endif
        }
        fprintf(stderr,
                \"Call-back argument must be function|instance|instance.__call__|f2py-function \"
                \"but got %s.\\n\",
                tp_name);
#if defined(Py_LIMITED_API)
        Py_XDECREF(tp_name_obj);
#endif
        goto capi_fail;
    }

    if (PyObject_HasAttrString(tmp_fun,\"__code__\")) {
        if (PyObject_HasAttrString(tmp = PyObject_GetAttrString(tmp_fun,\"__code__\"),\"co_argcount\")) {
            PyObject *tmp_argcount = PyObject_GetAttrString(tmp,\"co_argcount\");
            Py_DECREF(tmp);
            if (tmp_argcount == NULL) {
                goto capi_fail;
            }
            tot = PyLong_AsSsize_t(tmp_argcount) - di;
            Py_DECREF(tmp_argcount);
        }
    }
    /* Get the number of optional arguments */
    if (PyObject_HasAttrString(tmp_fun,\"__defaults__\")) {
        if (PyTuple_Check(tmp = PyObject_GetAttrString(tmp_fun,\"__defaults__\")))
            opt = PyTuple_Size(tmp);
        Py_XDECREF(tmp);
    }
    /* Get the number of extra arguments */
    if (xa != NULL)
        ext = PyTuple_Size((PyObject *)xa);
    /* Calculate the size of call-backs argument list */
    siz = MIN(maxnofargs+ext,tot);
    *nofargs = MAX(0,siz-ext);

#ifdef DEBUGCFUNCS
    fprintf(stderr,
            \"debug-capi:create_cb_arglist:maxnofargs(-nofoptargs),\"
            \"tot,opt,ext,siz,nofargs = %d(-%d), %zd, %zd, %zd, %zd, %d\\n\",
            maxnofargs, nofoptargs, tot, opt, ext, siz, *nofargs);
#endif

    if (siz < tot-opt) {
        fprintf(stderr,
                \"create_cb_arglist: Failed to build argument list \"
                \"(siz) with enough arguments (tot-opt) required by \"
                \"user-supplied function (siz,tot,opt=%zd, %zd, %zd).\\n\",
                siz, tot, opt);
        goto capi_fail;
    }

    /* Initialize argument list */
    *args = (PyTupleObject *)PyTuple_New(siz);
    for (i=0;i<*nofargs;i++) {
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM((PyObject *)(*args),i,Py_None);
    }
    if (xa != NULL)
        for (i=(*nofargs);i<siz;i++) {
            tmp = PyTuple_GetItem((PyObject *)xa,i-(*nofargs));
            Py_INCREF(tmp);
            PyTuple_SET_ITEM(*args,i,tmp);
        }
    CFUNCSMESS(\"create_cb_arglist-end\\n\");
    Py_DECREF(tmp_fun);
    return 1;

capi_fail:
    if (PyErr_Occurred() == NULL)
        PyErr_SetString(#modulename#_error, errmess);
    Py_XDECREF(tmp_fun);
    return 0;
}
"""

# Instances of wrapped derived types share this layout, so that routine
# arguments can be converted without knowing the type.
typedefs['f2py_dt_object'] = """\
typedef struct {
    PyObject_HEAD
    PyObject *capsule;  /* holds the Fortran object */
    int type_tag;       /* concrete type, for class(T) dispatch */
    int kind_value;     /* KIND parameter of a parameterized type */
} f2py_dt_object;"""
needs['f2py_dt_from_pyobj'] = ['f2py_dt_object']
cfuncs['f2py_dt_from_pyobj'] = """
/* Set *ptr to the Fortran object of an instance of type.  A negative
   want_tag accepts extensions of type, as class(T) does; otherwise the
   instance has to be of type itself.  capname, unless NULL, is the
   specialization of a parameterized type that the instance has to hold. */
static int
f2py_dt_from_pyobj(void **ptr, int *tag, PyObject *obj, PyTypeObject *type,
                   int want_tag, const char *capname, const char *errmess)
{
    f2py_dt_object *dt = (f2py_dt_object *)obj;
    if (!PyObject_TypeCheck(obj, type)
            || (want_tag >= 0 && dt->type_tag != want_tag)) {
        PyErr_Format(PyExc_TypeError, "%s: expected %s, not %s",
                     errmess, type->tp_name, Py_TYPE(obj)->tp_name);
        return 0;
    }
    if (dt->capsule == NULL) {
        PyErr_Format(PyExc_ValueError, "%s: %s instance is not initialized",
                     errmess, type->tp_name);
        return 0;
    }
    if (capname != NULL && !PyCapsule_IsValid(dt->capsule, capname)) {
        PyErr_Format(PyExc_TypeError, "%s: expected %s, not %s",
                     errmess, capname, PyCapsule_GetName(dt->capsule));
        return 0;
    }
    *ptr = PyCapsule_GetPointer(dt->capsule, PyCapsule_GetName(dt->capsule));
    if (*ptr == NULL) {
        return 0;
    }
    *tag = dt->type_tag;
    return 1;
}
"""
needs['f2py_dt_to_pyobj'] = ['f2py_dt_object']
cfuncs['f2py_dt_to_pyobj'] = """
/* Return a new instance of type that owns the Fortran object at ptr, or
   None for no object, which is an unallocated or disassociated dummy or
   result.  On failure the object is destroyed, unless not even its capsule
   is made. */
static PyObject *
f2py_dt_to_pyobj(void *ptr, PyTypeObject *type, int type_tag, int kind_value,
                 const char *capname, PyCapsule_Destructor destructor)
{
    f2py_dt_object *dt;
    PyObject *capsule;
    if (ptr == NULL) {
        Py_RETURN_NONE;
    }
    capsule = PyCapsule_New(ptr, capname, destructor);
    if (capsule == NULL) {
        return NULL;
    }
    dt = (f2py_dt_object *)type->tp_alloc(type, 0);
    if (dt == NULL) {
        Py_DECREF(capsule);
        return NULL;
    }
    dt->capsule = capsule;
    dt->type_tag = type_tag;
    dt->kind_value = kind_value;
    return (PyObject *)dt;
}
"""
needs['f2py_dt_handles_from_pyobj'] = ['f2py_dt_from_pyobj']
cfuncs['f2py_dt_handles_from_pyobj'] = """
/* Return an array of the Fortran objects of a sequence of instances of
   type, in Fortran order.  dims are the extents the argument has to have,
   where -1 takes the extent of the sequence. */
static PyArrayObject *
f2py_dt_handles_from_pyobj(PyObject *obj, PyTypeObject *type, int want_tag,
                           const char *capname, int rank, npy_intp *dims,
                           const char *errmess)
{
    PyArrayObject *items, *handles;
    PyObject **item;
    void **ptr;
    npy_intp i, n;
    int tag, d;
    items = (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_OBJECT,
                                              NPY_ARRAY_F_CONTIGUOUS);
    if (items == NULL) {
        return NULL;
    }
    if (PyArray_NDIM(items) != rank) {
        PyErr_Format(PyExc_TypeError,
                     "%s: expected a rank-%d sequence of %s, got rank %d",
                     errmess, rank, type->tp_name, PyArray_NDIM(items));
        Py_DECREF(items);
        return NULL;
    }
    for (d = 0; d < rank; d++) {
        if (dims[d] >= 0 && dims[d] != PyArray_DIM(items, d)) {
            PyErr_Format(PyExc_ValueError,
                         "%s: expected %" NPY_INTP_FMT " items in dimension %d,"
                         " got %" NPY_INTP_FMT,
                         errmess, dims[d], d, PyArray_DIM(items, d));
            Py_DECREF(items);
            return NULL;
        }
        dims[d] = PyArray_DIM(items, d);
    }
    handles = (PyArrayObject *)PyArray_EMPTY(rank, PyArray_DIMS(items),
                                             NPY_UINTP, 1);
    if (handles == NULL) {
        Py_DECREF(items);
        return NULL;
    }
    n = PyArray_SIZE(items);
    item = (PyObject **)PyArray_DATA(items);
    ptr = (void **)PyArray_DATA(handles);
    for (i = 0; i < n; i++) {
        if (!f2py_dt_from_pyobj(ptr + i, &tag, item[i], type, want_tag,
                                capname, errmess)) {
            Py_DECREF(items);
            Py_DECREF(handles);
            return NULL;
        }
    }
    Py_DECREF(items);
    return handles;
}
"""
needs['f2py_dt_from_handles'] = ['f2py_dt_to_pyobj']
cfuncs['f2py_dt_from_handles'] = """
/* Return new instances of type for an array of Fortran objects, as a list
   for rank 1 and as an object array in Fortran order otherwise.  Objects
   that cannot be returned are destroyed. */
static PyObject *
f2py_dt_from_handles(void **ptr, int rank, npy_intp *dims, PyTypeObject *type,
                     int type_tag, int kind_value, const char *capname,
                     PyCapsule_Destructor destructor)
{
    PyObject *ret, **items;
    npy_intp i, n = 1;
    int d;
    for (d = 0; d < rank; d++) {
        n *= dims[d];
    }
    if (rank == 1) {
        ret = PyList_New(n);
    }
    else {
        ret = PyArray_EMPTY(rank, dims, NPY_OBJECT, 1);
    }
    for (i = 0; ret != NULL && i < n; i++) {
        PyObject *item = f2py_dt_to_pyobj(ptr[i], type, type_tag, kind_value,
                                          capname, destructor);
        if (item == NULL) {
            Py_CLEAR(ret);
            i++;
            break;
        }
        if (rank == 1) {
            PyList_SET_ITEM(ret, i, item);
        }
        else {
            items = (PyObject **)PyArray_DATA((PyArrayObject *)ret);
            Py_XSETREF(items[i], item);
        }
    }
    if (ret == NULL) {
        PyObject *exc = PyErr_GetRaisedException();
        for (; i < n; i++) {
            if (ptr[i] != NULL) {
                PyObject *rest = PyCapsule_New(ptr[i], capname, destructor);
                Py_XDECREF(rest);
            }
        }
        PyErr_SetRaisedException(exc);
    }
    return ret;
}
"""

needs['f2py_dt_from_vector'] = ['f2py_dt_from_handles', 'string.h']
cfuncs['f2py_dt_from_vector'] = """
/* Return new instances of type for an allocatable or pointer array of
   derived type that Fortran made, as f2py_dt_from_handles does, or None
   when it is not allocated, which a negative first extent tells.  buffer
   holds the handles of copies of its elements and is released with
   free_handles.  A bind(c) type has its elements copied to memory that
   its instances own, copy_size bytes each. */
static PyObject *
f2py_dt_from_vector(void *buffer, int rank, const npy_intp *extents,
                    PyTypeObject *type, int type_tag, int kind_value,
                    const char *capname, PyCapsule_Destructor destructor,
                    size_t copy_size, void (*free_handles)(void *, int))
{
    npy_intp dims[F2PY_MAX_DIMS], i, n = 1;
    void **ptr = (void **)buffer, **copies = NULL;
    PyObject *ret = NULL;
    int d;
    if (extents[0] < 0) {
        Py_RETURN_NONE;
    }
    for (d = 0; d < rank; d++) {
        dims[d] = extents[d];
        n *= dims[d];
    }
    if (copy_size > 0 && n > 0) {
        copies = PyMem_Calloc(n, sizeof(void *));
        for (i = 0; copies != NULL && i < n; i++) {
            copies[i] = PyMem_Malloc(copy_size);
            if (copies[i] == NULL) {
                for (; i > 0; i--) {
                    PyMem_Free(copies[i - 1]);
                }
                PyMem_Free(copies);
                copies = NULL;
                break;
            }
            memcpy(copies[i], ptr[i], copy_size);
        }
        if (copies == NULL) {
            PyErr_NoMemory();
        }
        ptr = copies;
    }
    if (copy_size == 0 || n == 0 || copies != NULL) {
        ret = f2py_dt_from_handles(ptr, rank, dims, type, type_tag,
                                   kind_value, capname, destructor);
    }
    PyMem_Free(copies);
    if (buffer != NULL) {
        free_handles(buffer, (int)n);
    }
    return ret;
}
"""
cfuncs['f2py_dt_call_bound'] = """
/* Call routine name of Fortran module modname of module, the binding of a
   type-bound procedure, with self as its passed-object argument.  That is
   positional argument pos of the routine, or keyword passname when fewer
   positional arguments are given, and none when pos is negative. */
static PyObject *
f2py_dt_call_bound(PyObject *module, const char *modname, const char *name,
                   PyObject *self, Py_ssize_t pos, const char *passname,
                   PyObject *args, PyObject *kwds)
{
    PyObject *fmod, *routine, *ret = NULL, *newargs = NULL, *newkwds = NULL;
    Py_ssize_t i, n = PyTuple_GET_SIZE(args);
    fmod = PyObject_GetAttrString(module, modname);
    if (fmod == NULL) {
        return NULL;
    }
    routine = PyObject_GetAttrString(fmod, name);
    Py_DECREF(fmod);
    if (routine == NULL) {
        return NULL;
    }
    if (pos < 0) {
        ret = PyObject_Call(routine, args, kwds);
    }
    else if (n >= pos) {
        newargs = PyTuple_New(n + 1);
        if (newargs != NULL) {
            for (i = 0; i <= n; i++) {
                PyObject *item = (i < pos ? PyTuple_GET_ITEM(args, i)
                                  : i == pos ? self
                                  : PyTuple_GET_ITEM(args, i - 1));
                Py_INCREF(item);
                PyTuple_SET_ITEM(newargs, i, item);
            }
            ret = PyObject_Call(routine, newargs, kwds);
        }
    }
    else {
        newkwds = (kwds == NULL ? PyDict_New() : PyDict_Copy(kwds));
        if (newkwds != NULL
                && PyDict_SetItemString(newkwds, passname, self) == 0) {
            ret = PyObject_Call(routine, args, newkwds);
        }
    }
    Py_XDECREF(newargs);
    Py_XDECREF(newkwds);
    Py_DECREF(routine);
    return ret;
}
"""
typedefs['f2py_dt_dummy'] = """\
typedef struct {
    const char *name;
    char kind;           /* i, f, c, b, s, t for a derived type, o for class(*) */
    int ndim;            /* not rank, which the module defines as a macro */
    PyTypeObject *type;  /* of a derived type */
    int required;
} f2py_dt_dummy;"""
needs['f2py_dt_args_match'] = ['f2py_dt_dummy']
cfuncs['f2py_dt_args_match'] = """
/* Tell whether args and kwds fit the dummy arguments of a specific
   procedure, ending with an entry without name, by their types and
   ranks, so that a generic type-bound procedure can pick its specific
   procedure (F2018 15.4.3.4.5). */
static int
f2py_dt_args_match(PyObject *args, PyObject *kwds, const f2py_dt_dummy *dummy)
{
    Py_ssize_t i, n = PyTuple_GET_SIZE(args), keywords = 0;
    for (i = 0; dummy->name != NULL; i++, dummy++) {
        PyObject *v = NULL;
        int ok;
        if (i < n) {
            v = PyTuple_GET_ITEM(args, i);
        }
        else if (kwds != NULL) {
            v = PyDict_GetItemString(kwds, dummy->name);
            keywords += (v != NULL);
        }
        if (v == NULL) {
            if (dummy->required) {
                return 0;
            }
            continue;
        }
        if (dummy->ndim > 0 || PyArray_Check(v)) {
            if (!PyArray_Check(v)) {
                ok = (dummy->ndim > 0 && PySequence_Check(v)
                      && !PyUnicode_Check(v) && !PyBytes_Check(v));
            }
            else {
                char kind = PyArray_DESCR((PyArrayObject *)v)->kind;
                ok = (PyArray_NDIM((PyArrayObject *)v) == dummy->ndim
                      && (kind == dummy->kind
                          || (kind == 'u' && dummy->kind == 'i')
                          || (kind == 'S' && dummy->kind == 's')
                          || (kind == 'U' && dummy->kind == 's')
                          || (kind == 'O' && dummy->kind == 't')));
            }
        }
        else if (dummy->kind == 'o') {
            ok = 1;
        }
        else if (dummy->kind == 't') {
            ok = PyObject_TypeCheck(v, dummy->type);
        }
        else if (dummy->kind == 'b') {
            ok = PyBool_Check(v) || PyArray_IsScalar(v, Bool);
        }
        else if (dummy->kind == 'i') {
            ok = ((PyLong_Check(v) && !PyBool_Check(v))
                  || PyArray_IsScalar(v, Integer));
        }
        else if (dummy->kind == 'f') {
            ok = PyFloat_Check(v) || PyArray_IsScalar(v, Floating);
        }
        else if (dummy->kind == 'c') {
            ok = PyComplex_Check(v) || PyArray_IsScalar(v, ComplexFloating);
        }
        else {
            ok = PyUnicode_Check(v) || PyBytes_Check(v);
        }
        if (!ok) {
            return 0;
        }
    }
    return n <= i && (kwds == NULL || PyDict_Size(kwds) == keywords);
}
"""

# CLASS(*) arguments take Python and NumPy scalars as these Fortran types,
# and instances of the derived types of their module with their type tags.
# Each entry is (tag, Fortran type, member of f2py_dt_any).
dt_any_intrinsics = [
    (-1, 'logical', 'l'),
    (-2, 'integer', 'i'),
    (-3, 'integer(c_int8_t)', 'i8'),
    (-4, 'integer(c_int16_t)', 'i16'),
    (-5, 'integer(c_int32_t)', 'i32'),
    (-6, 'integer(c_int64_t)', 'i64'),
    (-7, 'real(c_float)', 'f'),
    (-8, 'real(c_double)', 'd'),
    (-9, 'complex(c_float_complex)', 'cf'),
    (-10, 'complex(c_double_complex)', 'cd'),
]
_dt_any_tags = {member: tag for tag, _, member in dt_any_intrinsics}
typedefs['f2py_dt_any'] = """\
typedef struct {
    PyTypeObject *type;
    int type_tag;
} f2py_dt_any_type;
typedef union {
    int l;
    int i;
    npy_int8 i8;
    npy_int16 i16;
    npy_int32 i32;
    npy_int64 i64;
    float f;
    double d;
    npy_cfloat cf;
    npy_cdouble cd;
} f2py_dt_any;"""
needs['f2py_dt_any_from_pyobj'] = ['f2py_dt_any', 'f2py_dt_from_pyobj',
                                   'npy_math.h']
cfuncs['f2py_dt_any_from_pyobj'] = """
/* Set *ptr to the value of a CLASS(*) argument and *tag to its type.
   Scalars are stored in *value; types lists the derived types the
   argument may also have, up to an entry with type NULL. */
static int
f2py_dt_any_from_pyobj(void **ptr, int *tag, f2py_dt_any *value,
                       PyObject *obj, const f2py_dt_any_type *types,
                       const char *errmess)
{
    *ptr = value;
    if (PyBool_Check(obj)) {
        value->l = (obj == Py_True);
        *tag = @l@;
        return 1;
    }
    if (PyArray_IsScalar(obj, Generic)) {
        PyArray_Descr *descr = PyArray_DescrFromScalar(obj);
        int elsize, kind;
        if (descr == NULL) {
            return 0;
        }
        elsize = (int)PyDataType_ELSIZE(descr);
        kind = descr->kind;
        Py_DECREF(descr);
        *tag = 0;
        if (kind == 'b') {
            value->l = PyObject_IsTrue(obj);
            *tag = @l@;
        }
        else if (kind == 'i' && elsize == sizeof(npy_int8)) {
            *tag = @i8@;
        }
        else if (kind == 'i' && elsize == sizeof(npy_int16)) {
            *tag = @i16@;
        }
        else if (kind == 'i' && elsize == sizeof(npy_int32)) {
            *tag = @i32@;
        }
        else if (kind == 'i' && elsize == sizeof(npy_int64)) {
            *tag = @i64@;
        }
        else if (kind == 'f' && elsize == sizeof(float)) {
            *tag = @f@;
        }
        else if (kind == 'f' && elsize == sizeof(double)) {
            *tag = @d@;
        }
        else if (kind == 'c' && elsize == sizeof(npy_cfloat)) {
            *tag = @cf@;
        }
        else if (kind == 'c' && elsize == sizeof(npy_cdouble)) {
            *tag = @cd@;
        }
        if (*tag == 0) {
            PyErr_Format(PyExc_TypeError, "%s: no Fortran type for %s",
                         errmess, Py_TYPE(obj)->tp_name);
            return 0;
        }
        if (kind != 'b') {
            PyArray_ScalarAsCtype(obj, value);
        }
        return 1;
    }
    if (PyLong_Check(obj)) {
        long v = PyLong_AsLong(obj);
        if (v == -1 && PyErr_Occurred()) {
            return 0;
        }
        if (v < INT_MIN || v > INT_MAX) {
            PyErr_Format(PyExc_OverflowError,
                         "%s: %ld does not fit a default integer",
                         errmess, v);
            return 0;
        }
        value->i = (int)v;
        *tag = @i@;
        return 1;
    }
    if (PyFloat_Check(obj)) {
        value->d = PyFloat_AsDouble(obj);
        *tag = @d@;
        return 1;
    }
    if (PyComplex_Check(obj)) {
        value->cd = npy_cpack(PyComplex_RealAsDouble(obj),
                              PyComplex_ImagAsDouble(obj));
        *tag = @cd@;
        return 1;
    }
    for (; types->type != NULL; types++) {
        if (PyObject_TypeCheck(obj, types->type)
                && ((f2py_dt_object *)obj)->type_tag == types->type_tag) {
            return f2py_dt_from_pyobj(ptr, tag, obj, types->type,
                                      types->type_tag, NULL, errmess);
        }
    }
    PyErr_Format(PyExc_TypeError, "%s: no Fortran type for %s",
                 errmess, Py_TYPE(obj)->tp_name);
    return 0;
}
"""
for _member, _tag in _dt_any_tags.items():
    cfuncs['f2py_dt_any_from_pyobj'] = (
        cfuncs['f2py_dt_any_from_pyobj'].replace(f'@{_member}@', str(_tag)))


def buildcfuncs():
    from .capi_maps import c2capi_map
    for k in c2capi_map.keys():
        m = f'pyarr_from_p_{k}1'
        cppmacros[
            m] = f'#define {m}(v) (PyArray_SimpleNewFromData(0,NULL,{c2capi_map[k]},(char *)v))'
    k = 'string'
    m = f'pyarr_from_p_{k}1'
    # NPY_CHAR compatibility, NPY_STRING with itemsize 1
    cppmacros[
        m] = f'#define {m}(v,dims) (PyArray_New(&PyArray_Type, 1, dims, NPY_STRING, NULL, v, 1, NPY_ARRAY_CARRAY, NULL))'


############ Auxiliary functions for sorting needs ###################

def append_needs(need, flag=1):
    # This function modifies the contents of the global `outneeds` dict.
    if isinstance(need, list):
        for n in need:
            append_needs(n, flag)
    elif isinstance(need, str):
        if not need:
            return
        if need in includes0:
            n = 'includes0'
        elif need in includes:
            n = 'includes'
        elif need in typedefs:
            n = 'typedefs'
        elif need in typedefs_generated:
            n = 'typedefs_generated'
        elif need in cppmacros:
            n = 'cppmacros'
        elif need in cfuncs:
            n = 'cfuncs'
        elif need in callbacks:
            n = 'callbacks'
        elif need in f90modhooks:
            n = 'f90modhooks'
        elif need in commonhooks:
            n = 'commonhooks'
        else:
            errmess(f'append_needs: unknown need {repr(need)}\n')
            return
        if need in outneeds[n]:
            return
        if flag:
            tmp = {}
            if need in needs:
                for nn in needs[need]:
                    t = append_needs(nn, 0)
                    if isinstance(t, dict):
                        for nnn in t.keys():
                            if nnn in tmp:
                                tmp[nnn] = tmp[nnn] + t[nnn]
                            else:
                                tmp[nnn] = t[nnn]
            for nn in tmp.keys():
                for nnn in tmp[nn]:
                    if nnn not in outneeds[nn]:
                        outneeds[nn] = [nnn] + outneeds[nn]
            outneeds[n].append(need)
        else:
            tmp = {}
            if need in needs:
                for nn in needs[need]:
                    t = append_needs(nn, flag)
                    if isinstance(t, dict):
                        for nnn in t.keys():
                            if nnn in tmp:
                                tmp[nnn] = t[nnn] + tmp[nnn]
                            else:
                                tmp[nnn] = t[nnn]
            if n not in tmp:
                tmp[n] = []
            tmp[n].append(need)
            return tmp
    else:
        errmess(f'append_needs: expected list or string but got :{repr(need)}\n')


def get_needs():
    # This function modifies the contents of the global `outneeds` dict.
    res = {}
    for n in outneeds.keys():
        out = []
        saveout = copy.copy(outneeds[n])
        while len(outneeds[n]) > 0:
            if outneeds[n][0] not in needs:
                out.append(outneeds[n][0])
                del outneeds[n][0]
            else:
                flag = 0
                for k in outneeds[n][1:]:
                    if k in needs[outneeds[n][0]]:
                        flag = 1
                        break
                if flag:
                    outneeds[n] = outneeds[n][1:] + [outneeds[n][0]]
                else:
                    out.append(outneeds[n][0])
                    del outneeds[n][0]
            if saveout and (0 not in map(lambda x, y: x == y, saveout, outneeds[n])) \
                    and outneeds[n] != []:
                print(n, saveout)
                errmess(
                    'get_needs: no progress in sorting needs, probably circular dependence, skipping.\n')
                out = out + saveout
                break
            saveout = copy.copy(outneeds[n])
        if out == []:
            out = [n]
        res[n] = out
    return res
