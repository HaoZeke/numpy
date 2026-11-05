"""
C code generation for bind(c) and opaque derived type wrappers.

Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

from ._dt_array_components import record_array_c, record_array_declarations
from ._dt_helpers import (
    _C_TO_NPY_ENUM,
    _C_TO_PYFORMAT,
    _C_TO_PYOBJ,
    _PYOBJ_TO_C,
    _coarray_as_local,
    _fortran_sym,
    _get_alloc_ndim,
    _get_array_dims,
    _get_c_return_type,
    _get_char_len,
    _get_member_ctype,
    _get_pointer_ndim,
    _is_allocatable_member,
    _is_array_member,
    _is_char_member,
    _is_coarray_member,
    _is_complex_member,
    _is_deferred_char_member,
    _is_len_sized_array,
    _is_pointer_member,
    _is_private_member,
    _is_type_array_member,
    _is_type_member,
)
from ._dt_operators import (
    _gen_operator_c_code as _gen_operator_c_code,
    _gen_operator_fortran_wrappers as _gen_operator_fortran_wrappers,
    _scan_operator_interfaces as _scan_operator_interfaces,
)


def _gen_array_from_any_helper():
    """Generate a C helper that accepts Array API / DLPack objects.

    Returns a string containing a static C function
    ``f2py_dt_array_from_any`` that tries:
    1. Fast-path: input is already a numpy array
    2. DLPack: input has ``__dlpack__``, convert via numpy.from_dlpack
    3. Fallback: PyArray_FROM_OTF (handles lists, __array__, etc.)
    """
    return """\
/* Array API / DLPack helper for derived type array setters.
 * Accepts numpy arrays, DLPack objects, and generic array-likes. */
static PyObject *
f2py_dt_array_from_any(PyObject *value, int typenum, int requirements)
{
    /* Fast path: already a numpy array */
    if (PyArray_Check(value)) {
        return PyArray_FROM_OTF(value, typenum, requirements);
    }
    /* DLPack path: check for __dlpack__ attribute */
    if (PyObject_HasAttrString(value, "__dlpack__")) {
        PyObject *np_mod = PyImport_ImportModule("numpy");
        if (np_mod != NULL) {
            PyObject *arr = PyObject_CallMethod(np_mod, "from_dlpack", "O",
                                                value);
            Py_DECREF(np_mod);
            if (arr != NULL) {
                /* Ensure correct dtype and contiguity */
                PyObject *result = PyArray_FROM_OTF(arr, typenum,
                                                    requirements);
                Py_DECREF(arr);
                return result;
            }
            /* from_dlpack failed; clear error and fall through */
            PyErr_Clear();
        } else {
            PyErr_Clear();
        }
    }
    /* Fallback: lists, __array__ protocol, etc. */
    return PyArray_FROM_OTF(value, typenum, requirements);
}

static int
f2py_dt_shape_product(int rank, const npy_intp *dims, npy_intp *total)
{
    *total = 1;
    for (int d = 0; d < rank; d++) {
        npy_intp extent = dims[d];
        if (extent < 0 || (extent != 0 && *total > NPY_MAX_INTP / extent)) {
            PyErr_SetString(PyExc_OverflowError, "derived array shape is out of range");
            return 0;
        }
        *total *= extent;
    }
    return 1;
}

static int
f2py_dt_checked_shape(PyArrayObject *arr, int rank)
{
    npy_intp total;
    if (PyArray_NDIM(arr) != rank) {
        PyErr_Format(PyExc_ValueError,
                     "derived array requires a %dD array, got %dD",
                     rank, PyArray_NDIM(arr));
        return 0;
    }
    if (!f2py_dt_shape_product(rank, PyArray_DIMS(arr), &total)) return 0;
    for (int d = 0; d < rank; d++) {
        npy_intp extent = PyArray_DIM(arr, d);
        if ((npy_intp)(int)extent != extent) {
            PyErr_SetString(PyExc_OverflowError, "derived array extent exceeds native index range");
            return 0;
        }
    }
    return 1;
}

/* The capsule pins Fortran storage while result allocation can collect. */
static PyObject *
f2py_dt_copy_array(PyObject *owner, int rank, npy_intp *dims, int typenum,
                    void *data, size_t bytes)
{
    Py_INCREF(owner);
    PyObject *arr = PyArray_EMPTY(rank, dims, typenum, 1);
    if (arr != NULL && bytes > 0) {
        memcpy(PyArray_DATA((PyArrayObject *)arr), data, bytes);
    }
    Py_DECREF(owner);
    return arr;
}
"""


def _gen_bindc_struct(typename, members):
    """Generate C typedef struct for a bind(c) type."""
    lines = ['typedef struct {']
    for mname, mvar in members.items():
        if _is_type_array_member(mvar):
            inner = mvar.get('typename', '').lower()
            dims = _get_array_dims(mvar)
            dim_str = ''.join(f'[{d}]' for d in reversed(dims))
            lines.append(f'    f2py_{inner}_t {mname}{dim_str};')
        elif _is_type_member(mvar):
            inner = mvar.get('typename', '').lower()
            lines.append(f'    f2py_{inner}_t {mname};')
        else:
            ctype = _get_member_ctype(mvar)
            dims = _get_array_dims(mvar)
            if dims:
                dim_str = ''.join(f'[{d}]' for d in reversed(dims))
                lines.append(f'    {ctype} {mname}{dim_str};')
            else:
                lines.append(f'    {ctype} {mname};')
    lines.append(f'}} f2py_{typename}_t;\n')
    return '\n'.join(lines)


def _gen_pytype_struct(typename, has_kind=False, len_info=None):
    """Generate the Python type object struct.

    It starts like f2py_dt_object, which routine arguments are read as.
    """
    len_fields = ''
    if len_info:
        for li in len_info:
            lname = li['name']
            len_fields += (
                f'\n    int len_{lname};'
                f'  /* LEN parameter {lname} */')
    return f"""\
typedef struct {{
    PyObject_HEAD
    PyObject *capsule;  /* PyCapsule wrapping Fortran {typename} data */
    int type_tag;  /* F2018 7.3.2.3: type discriminator for polymorphic dispatch */
    int kind_value;  /* KIND parameter value for dispatch */{len_fields}
}} Py{typename}Object;
"""


def _gen_capsule_destructor(typename):
    """Generate PyCapsule destructor that frees the C struct."""
    capsule_name = f'f2py.{typename}'
    return f"""\
static void
f2py_{typename}_capsule_destructor(PyObject *capsule)
{{
    void *ptr = PyCapsule_GetPointer(capsule, "{capsule_name}");
    if (ptr != NULL) {{
        PyMem_Free(ptr);
    }}
}}
"""


def _gen_tp_new(typename, type_tag=0, *, pdt=False):
    """Generate tp_new for the Python type."""
    code = f"""\
static PyObject *
Py{typename}_tp_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{{
    Py{typename}Object *self;
    self = (Py{typename}Object *)type->tp_alloc(type, 0);
    if (self != NULL) {{
        self->capsule = NULL;
        self->type_tag = {type_tag};  /* F2018 7.3.2.3: polymorphic dispatch tag */
    }}
    return (PyObject *)self;
}}

static PyTypeObject Py{typename}_Type;

static PyObject *
f2py_op_{typename}_owner(PyObject *operand)
{{
    Py{typename}Object *value = (Py{typename}Object *)operand;
    if (value->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError, "operand not initialized");
        return NULL;
    }}
    return Py_NewRef(value->capsule);
}}

static int
f2py_op_{typename}_writable(PyObject *operand)
{{
    return 1;
}}

static PyObject *
f2py_op_{typename}_adopt(void *pointer, PyCapsule_Destructor destroy)
{{
    PyObject *result = Py{typename}_tp_new(&Py{typename}_Type, NULL, NULL);
    if (result == NULL) return NULL;
    ((Py{typename}Object *)result)->capsule = PyCapsule_New(
        pointer, "f2py.{typename}", destroy);
    if (((Py{typename}Object *)result)->capsule == NULL) {{
        Py_DECREF(result);
        return NULL;
    }}
    return result;
}}
"""

    if pdt:
        from ._dt_c_handles import _handle_operator_adopt
        marker = f'static PyObject *\nf2py_op_{typename}_adopt('
        assert code.count(marker) == 1
        code = code[:code.index(marker)]
        code = code.replace('    return Py_NewRef(value->capsule);',
                            '    const char *name = PyCapsule_GetName(value->capsule);\n'
                            f'    if (name == NULL || f2py_{typename}_checked(operand, name, 0) == NULL) return NULL;\n'
                            '    return Py_NewRef(value->capsule);')
        code = code.replace('    return 1;',
                            f'    Py{typename}Object *value = (Py{typename}Object *)operand;\n'
                            '    if (value->capsule == NULL) {\n'
                            '        PyErr_SetString(PyExc_RuntimeError, "operand not initialized");\n'
                            '        return 0;\n    }\n'
                            '    const char *name = PyCapsule_GetName(value->capsule);\n'
                            f'    return name != NULL && f2py_{typename}_checked(operand, name, 1) != NULL;')
        code += _handle_operator_adopt(typename)
        code = (f'static const char *f2py_{typename}_capsule_name(void *);\n'
                f'static int f2py_{typename}_hydrate(PyObject *, void *);\n'
                f'static void *f2py_{typename}_checked(PyObject *, const char *, int);\n'
                + code)
    return code


def _gen_tp_init(typename, members):
    """Generate tp_init that allocates the C struct and populates from args."""
    capsule_name = f'f2py.{typename}'
    # Build format string and argument extraction (scalar members only;
    # array members are zero-initialized and set via properties)
    kwlist = []
    fmt_parts = []
    extract_lines = []
    for mname, mvar in members.items():
        if _is_array_member(mvar) or _is_type_member(mvar) or _is_type_array_member(mvar):
            continue  # arrays and nested types set via properties
        ctype = _get_member_ctype(mvar)
        fmt = _C_TO_PYFORMAT.get(ctype)
        if fmt is None:
            continue
        kwlist.append(mname)
        fmt_parts.append(fmt)
        if ctype in ('npy_cfloat', 'npy_cdouble'):
            if ctype == 'npy_cfloat':
                extract_lines.append(
                    f'    data->{mname} = npy_cpackf('
                    f'(float){mname}.real, (float){mname}.imag);')
            else:
                extract_lines.append(
                    f'    data->{mname} = npy_cpack('
                    f'{mname}.real, {mname}.imag);')
        else:
            extract_lines.append(f'    data->{mname} = {mname};')

    kwlist_str = ', '.join(f'"{k}"' for k in kwlist)
    fmt_str = ''.join(fmt_parts)

    # Declare temp variables for parsed args
    decl_lines = []
    parse_args = []
    for mname, mvar in members.items():
        if _is_array_member(mvar) or _is_type_member(mvar) or _is_type_array_member(mvar):
            continue
        ctype = _get_member_ctype(mvar)
        if _C_TO_PYFORMAT.get(ctype) is None:
            continue
        if ctype in ('npy_cfloat', 'npy_cdouble'):
            decl_lines.append(f'    Py_complex {mname} = {{0, 0}};')
        else:
            decl_lines.append(f'    {ctype} {mname} = 0;')
        parse_args.append(f'&{mname}')

    decl_str = '\n'.join(decl_lines)
    parse_args_str = ', '.join(parse_args)
    extract_str = '\n'.join(extract_lines)

    # When no scalar members exist (all arrays), skip arg parsing
    if kwlist:
        parse_block = f"""\
    static char *kwlist[] = {{{kwlist_str}, NULL}};
{decl_str}

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|{fmt_str}", kwlist,
                                     {parse_args_str}))
        return -1;"""
    else:
        parse_block = """\
    /* No scalar members; set via properties */"""

    return f"""\
static int
Py{typename}_tp_init(PyObject *selfobj, PyObject *args, PyObject *kwds)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
{parse_block}

    f2py_{typename}_t *data = (f2py_{typename}_t *)PyMem_Malloc(
        sizeof(f2py_{typename}_t));
    if (data == NULL) {{
        PyErr_NoMemory();
        return -1;
    }}
    memset(data, 0, sizeof(f2py_{typename}_t));

{extract_str}

    /* Clean up old capsule if re-initializing */
    Py_XDECREF(self->capsule);
    self->capsule = PyCapsule_New(
        (void *)data, "{capsule_name}",
        f2py_{typename}_capsule_destructor);
    if (self->capsule == NULL) {{
        PyMem_Free(data);
        return -1;
    }}

    return 0;
}}
"""


def _gen_tp_dealloc(typename):
    """Generate tp_dealloc."""
    return f"""\
static void
Py{typename}_tp_dealloc(PyObject *selfobj)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    Py_XDECREF(self->capsule);
    Py_TYPE(self)->tp_free((PyObject *)self);
}}
"""


def _gen_getset(typename, members, type_map=None):
    """Generate tp_getset array with property getters/setters."""
    capsule_name = f'f2py.{typename}'
    funcs = []
    getset_entries = []

    for mname, mvar in members.items():
        if _is_coarray_member(mvar):
            mvar = _coarray_as_local(mvar)  # F2018 7.5.4.3: local data
        if _is_private_member(mvar):
            continue  # F2018 7.5.4.8: private components not exposed
        if _is_type_array_member(mvar):
            code, entry = record_array_c(
                typename, mname, mvar, type_map, bindc=True)
            funcs.append(code)
            getset_entries.append(entry)
            continue

        if _is_type_member(mvar):
            # Nested derived type member
            inner = mvar.get('typename', '').lower()
            inner_capsule = f'f2py.{inner}'

            getter_name = f'Py{typename}_get_{mname}'
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    f2py_{typename}_t *data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "{capsule_name}");
    if (data == NULL) return NULL;
    /* Create a new inner type object with a copy of the nested data */
    PyObject *obj = Py{inner}_tp_new(&Py{inner}_Type, NULL, NULL);
    if (obj == NULL) return NULL;
    f2py_{inner}_t *copy = (f2py_{inner}_t *)PyMem_Malloc(
        sizeof(f2py_{inner}_t));
    if (copy == NULL) {{
        Py_DECREF(obj);
        return PyErr_NoMemory();
    }}
    memcpy(copy, &data->{mname}, sizeof(f2py_{inner}_t));
    ((Py{inner}Object *)obj)->capsule = PyCapsule_New(
        copy, "{inner_capsule}", f2py_{inner}_capsule_destructor);
    if (((Py{inner}Object *)obj)->capsule == NULL) {{
        PyMem_Free(copy);
        Py_DECREF(obj);
        return NULL;
    }}
    return obj;
}}
""")

            setter_name = f'Py{typename}_set_{mname}'
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    if (!Py_IS_TYPE(value, &Py{inner}_Type)) {{
        PyErr_SetString(PyExc_TypeError,
                        "{mname} must be a {inner} instance");
        return -1;
    }}
    f2py_{typename}_t *data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "{capsule_name}");
    if (data == NULL) return -1;
    Py{inner}Object *inner_obj = (Py{inner}Object *)value;
    if (inner_obj->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{inner} object not initialized");
        return -1;
    }}
    f2py_{inner}_t *inner_data = (f2py_{inner}_t *)PyCapsule_GetPointer(
        inner_obj->capsule, "{inner_capsule}");
    if (inner_data == NULL) return -1;
    memcpy(&data->{mname}, inner_data, sizeof(f2py_{inner}_t));
    return 0;
}}
""")

            getset_entries.append(
                f'    {{"{mname}", {getter_name}, {setter_name}, '
                f'"{mname} member", NULL}},'
            )
            continue

        if _is_complex_member(mvar):
            ctype = _get_member_ctype(mvar)
            is_single = (ctype == 'npy_cfloat')
            creal_fn = 'npy_crealf' if is_single else 'npy_creal'
            cimag_fn = 'npy_cimagf' if is_single else 'npy_cimag'
            cpack_fn = 'npy_cpackf' if is_single else 'npy_cpack'
            cast = '(float)' if is_single else ''

            getter_name = f'Py{typename}_get_{mname}'
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    f2py_{typename}_t *data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "{capsule_name}");
    if (data == NULL) return NULL;
    return PyComplex_FromDoubles(
        (double){creal_fn}(data->{mname}),
        (double){cimag_fn}(data->{mname}));
}}
""")

            setter_name = f'Py{typename}_set_{mname}'
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    f2py_{typename}_t *data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "{capsule_name}");
    if (data == NULL) return -1;
    Py_complex c = PyComplex_AsCComplex(value);
    if (PyErr_Occurred()) return -1;
    data->{mname} = {cpack_fn}({cast}c.real, {cast}c.imag);
    return 0;
}}
""")

            getset_entries.append(
                f'    {{"{mname}", {getter_name}, {setter_name}, '
                f'"{mname} member", NULL}},'
            )
            continue

        ctype = _get_member_ctype(mvar)
        dims = _get_array_dims(mvar)

        if dims:
            # Array member -- getter returns NumPy array view,
            # setter copies from input array
            npy_enum = _C_TO_NPY_ENUM.get(ctype)
            if npy_enum is None:
                continue
            ndim = len(dims)
            total = 1
            for d in dims:
                total *= d
            dims_init = ', '.join(str(d) for d in dims)
            if ndim == 1:
                create_array = (
                    f'PyArray_SimpleNewFromData({ndim}, dims, {npy_enum}, '
                    f'(void *)data->{mname})')
            else:
                create_array = (
                    f'PyArray_New(&PyArray_Type, {ndim}, dims, {npy_enum}, '
                    f'NULL, (void *)data->{mname}, 0, NPY_ARRAY_FARRAY, NULL)')

            getter_name = f'Py{typename}_get_{mname}'
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    f2py_{typename}_t *data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "{capsule_name}");
    if (data == NULL) return NULL;
    npy_intp dims[{ndim}] = {{{dims_init}}};
    /* The capsule owns this allocation even if self is reinitialized. */
    PyObject *owner = Py_NewRef(self->capsule);
    PyObject *arr = {create_array};
    if (arr == NULL) {{
        Py_DECREF(owner);
        return NULL;
    }}
    if (PyArray_SetBaseObject((PyArrayObject *)arr, owner) < 0) {{
        Py_DECREF(arr);
        return NULL;
    }}
    return arr;
}}
""")

            setter_name = f'Py{typename}_set_{mname}'
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    f2py_{typename}_t *data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "{capsule_name}");
    if (data == NULL) return -1;
    PyObject *arr = f2py_dt_array_from_any(value, {npy_enum},
                                            NPY_ARRAY_FARRAY_RO);
    if (arr == NULL) return -1;
    if (!f2py_dt_checked_shape((PyArrayObject *)arr, {ndim})) {{
        Py_DECREF(arr);
        return -1;
    }}
    data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "{capsule_name}");
    if (data == NULL) {{
        Py_DECREF(arr);
        return -1;
    }}
    if (PyArray_NDIM((PyArrayObject *)arr) != {ndim}) {{
        PyErr_SetString(PyExc_ValueError, "{mname} has an incorrect rank");
        Py_DECREF(arr);
        return -1;
    }}
    const npy_intp expected_dims[{ndim}] = {{{dims_init}}};
    for (int d = 0; d < {ndim}; d++) {{
        if (PyArray_DIM((PyArrayObject *)arr, d) != expected_dims[d]) {{
            PyErr_SetString(PyExc_ValueError,
                            "{mname} has an incorrect shape");
            Py_DECREF(arr);
            return -1;
        }}
    }}
    if ((void *)data->{mname} != PyArray_DATA((PyArrayObject *)arr)) {{
        memcpy(data->{mname}, PyArray_DATA((PyArrayObject *)arr),
               {total} * sizeof({ctype}));
    }}
    Py_DECREF(arr);
    return 0;
}}
""")
        else:
            # Scalar member
            pyobj_expr = _C_TO_PYOBJ.get(ctype)
            frompy_expr = _PYOBJ_TO_C.get(ctype)
            if pyobj_expr is None or frompy_expr is None:
                continue

            getter_name = f'Py{typename}_get_{mname}'
            val_expr = pyobj_expr.format(val=f'data->{mname}')
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    f2py_{typename}_t *data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "{capsule_name}");
    if (data == NULL) return NULL;
    return {val_expr};
}}
""")

            setter_name = f'Py{typename}_set_{mname}'
            conv_expr = frompy_expr.format(obj='value')
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    f2py_{typename}_t *data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "{capsule_name}");
    if (data == NULL) return -1;
    data->{mname} = {conv_expr};
    if (PyErr_Occurred()) return -1;
    return 0;
}}
""")

        getset_entries.append(
            f'    {{"{mname}", {getter_name}, {setter_name}, '
            f'"{mname} member", NULL}},'
        )

    getset_array = (
        f'static PyGetSetDef Py{typename}_getset[] = {{\n'
        + '\n'.join(getset_entries) + '\n'
        + '    {NULL}  /* sentinel */\n'
        + '};\n'
    )

    return '\n'.join(funcs) + '\n' + getset_array


# One extra byte so a Fortran character value can be stored as a C string.
# C99 7.1.1 requires the terminating NUL, which the Fortran length does not count.
_C_STRING_NUL = 1


def _gen_repr_return(typename, fmt_str, args):
    """Return the C statements that return the repr of a type instance.

    The text is formatted twice, first to measure it (C99 7.21.6.5), so
    that no repr is cut off.  args are the arguments of the format, each
    after a comma.
    """
    fmt = f'"{typename}({fmt_str})"'
    return f"""\
    int f2py_len = snprintf(NULL, 0, {fmt}{args});
    if (f2py_len < 0) {{
        PyErr_SetString(PyExc_ValueError, "cannot format {typename}");
        return NULL;
    }}
    char *f2py_buf = PyMem_Malloc((size_t)f2py_len + {_C_STRING_NUL});
    if (f2py_buf == NULL) {{
        return PyErr_NoMemory();
    }}
    snprintf(f2py_buf, (size_t)f2py_len + {_C_STRING_NUL}, {fmt}{args});
    PyObject *f2py_repr = PyUnicode_FromStringAndSize(f2py_buf, f2py_len);
    PyMem_Free(f2py_buf);
    return f2py_repr;"""


def _gen_tp_repr(typename, members):
    """Generate tp_repr for nice string representation.

    Uses snprintf to format values since PyUnicode_FromFormat does not
    support floating-point format specifiers.
    """
    capsule_name = f'f2py.{typename}'
    fmt_parts = []
    val_args = []
    for mname, mvar in members.items():
        if _is_type_array_member(mvar):
            inner = mvar.get('typename', '').lower()
            dims = _get_array_dims(mvar)
            total = 1
            for d in dims:
                total *= d
            fmt_parts.append(f'{mname}=[{total}x{inner}]')
            continue
        if _is_type_member(mvar):
            inner = mvar.get('typename', '').lower()
            fmt_parts.append(f'{mname}=<{inner}>')
            continue
        dims = _get_array_dims(mvar)
        if dims:
            total = 1
            for d in dims:
                total *= d
            dim_str = 'x'.join(str(d) for d in dims)
            fmt_parts.append(f'{mname}=<array({dim_str})>')
            continue
        ctype = _get_member_ctype(mvar)
        if ctype in ('npy_cfloat', 'npy_cdouble'):
            creal_fn = 'npy_crealf' if ctype == 'npy_cfloat' else 'npy_creal'
            cimag_fn = 'npy_cimagf' if ctype == 'npy_cfloat' else 'npy_cimag'
            fmt_parts.append(f'{mname}=(%g+%gj)')
            val_args.append(f'(double){creal_fn}(data->{mname})')
            val_args.append(f'(double){cimag_fn}(data->{mname})')
        elif ctype in ('float', 'double'):
            fmt_parts.append(f'{mname}=%g')
            if ctype == 'float':
                val_args.append(f'(double)data->{mname}')
            else:
                val_args.append(f'data->{mname}')
        elif ctype in ('int',):
            fmt_parts.append(f'{mname}=%d')
            val_args.append(f'data->{mname}')
        elif ctype in ('long',):
            fmt_parts.append(f'{mname}=%ld')
            val_args.append(f'data->{mname}')
        elif ctype in ('long long',):
            fmt_parts.append(f'{mname}=%lld')
            val_args.append(f'data->{mname}')
        else:
            continue

    fmt_str = ', '.join(fmt_parts)
    val_str = ', '.join(val_args)
    if val_str:
        val_str = ',\n        ' + val_str

    return f"""\
static PyObject *
Py{typename}_tp_repr(PyObject *selfobj)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        return PyUnicode_FromString("{typename}(<uninitialized>)");
    }}
    f2py_{typename}_t *data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "{capsule_name}");
    if (data == NULL) {{
        PyErr_Clear();
        return PyUnicode_FromString("{typename}(<invalid>)");
    }}
{_gen_repr_return(typename, fmt_str, val_str)}
}}
"""


def _gen_typeobject(typename, has_methods=False, parent_typename=None,
                    has_number=False, has_richcompare=False,
                    has_tp_str=False):
    """Generate PyTypeObject definition."""
    methods_line = ''
    if has_methods:
        methods_line = f'\n    .tp_methods = Py{typename}_methods,'
    base_line = ''
    if parent_typename:
        base_line = f'\n    .tp_base = &Py{parent_typename}_Type,'
    number_line = ''
    if has_number:
        number_line = f'\n    .tp_as_number = &Py{typename}_as_number,'
    richcmp_line = ''
    if has_richcompare:
        richcmp_line = (f'\n    .tp_richcompare = '
                        f'Py{typename}_tp_richcompare,')
    str_line = ''
    if has_tp_str:
        str_line = f'\n    .tp_str = Py{typename}_tp_str,'
    return f"""\
static PyTypeObject Py{typename}_Type = {{
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "f2py.{typename}",
    .tp_doc = "Wrapper for Fortran derived type {typename}",
    .tp_basicsize = sizeof(Py{typename}Object),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Py{typename}_tp_new,
    .tp_init = Py{typename}_tp_init,
    .tp_dealloc = Py{typename}_tp_dealloc,
    .tp_repr = Py{typename}_tp_repr,{str_line}
    .tp_getset = Py{typename}_getset,{methods_line}{base_line}{number_line}{richcmp_line}
}};
"""


def _gen_init_code(typename, modulename):
    """Generate code for module init to register the type.

    The type is an attribute of the extension module and of the object
    of its Fortran module.
    """
    # Use lowercase for the Python-accessible attribute name to match
    # Fortran's case-insensitive convention
    pyname = typename.lower()
    return [
        '\t{',
        f'\t\tPyObject *fmod = PyDict_GetItemString(d, "{modulename}");',
        f'\t\tif (PyType_Ready(&Py{typename}_Type) < 0) return -1;',
        (f'\t\tif (PyModule_AddObjectRef(m, "{pyname}", '
         f'(PyObject *)&Py{typename}_Type) < 0) return -1;'),
        (f'\t\tif (fmod != NULL && PyObject_SetAttrString(fmod, "{pyname}", '
         f'(PyObject *)&Py{typename}_Type) < 0) return -1;'),
        '\t}',
    ]


def _gen_opaque_extern_decls(typename, members, specializations=None,
                             len_info=None):
    """Generate extern declarations for the Fortran bind(c) wrappers.

    If specializations is provided, generates separate declarations for
    each KIND variant with suffixed function names and resolved types.
    If len_info is provided, LEN params are added as extra int arguments
    to all extern declarations (destructor, getters, setters).
    """
    if specializations:
        all_lines = ['/* Extern declarations for Fortran wrappers */']
        for suffix, kind_dict, resolved_members in specializations:
            sym = _fortran_sym(typename + suffix)
            all_lines.append(
                f'/* KIND specialization {suffix} */')
            all_lines.append(
                _gen_opaque_extern_decls_one(sym, resolved_members, len_info))
        return '\n'.join(all_lines)

    sym = _fortran_sym(typename)
    lines = []
    lines.append('/* Extern declarations for Fortran wrappers */')

    # LEN parameter extra args for extern declarations.
    # For LEN types, every function after the constructor needs
    # these extra int args so Fortran can reconstruct the type.
    len_param_names = []
    if len_info:
        len_param_names = [li['name'] for li in len_info]
    len_extra = ', '.join(['int'] * len(len_param_names))
    # e.g. ', int' or ', int, int' or ''
    len_extra_prefix = (', ' + len_extra) if len_extra else ''
    len_param_name_set = set(len_param_names)

    # Constructor: returns void* (c_ptr) -- LEN params + scalar args
    args = []
    for li in (len_info or []):
        args.append(f'int {li["name"]}')
    for mname, mvar in members.items():
        if (_is_array_member(mvar) or _is_type_member(mvar)
                or _is_type_array_member(mvar) or _is_char_member(mvar)
                or _is_allocatable_member(mvar)
                or _is_pointer_member(mvar)
                or _is_deferred_char_member(mvar)):
            continue
        if (len_param_name_set
                and _is_len_sized_array(mvar, len_param_name_set)):
            continue
        ctype = _get_member_ctype(mvar)
        if ctype and _C_TO_PYFORMAT.get(ctype):
            args.append(f'{ctype} {mname}')
    args_str = ', '.join(args) if args else 'void'
    lines.append(f'extern void *f2py_create_{sym}({args_str});')

    # Destructor: takes void* + LEN params
    destr_args = f'void *{len_extra_prefix}'
    lines.append(f'extern void f2py_destroy_{sym}({destr_args});')

    # Getters and setters -- all take LEN params after void*
    for mname, mvar in members.items():
        if _is_type_array_member(mvar):
            lines.extend(record_array_declarations(
                sym, mname, mvar, len_extra_prefix))
            continue

        if _is_type_member(mvar):
            lines.append(
                f'extern void *f2py_get_{sym}_{mname}'
                f'(void *{len_extra_prefix});')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *{len_extra_prefix}, void *);')
            continue

        if _is_char_member(mvar):
            lines.append(
                f'extern void f2py_get_{sym}_{mname}'
                f'(void *{len_extra_prefix}, char *, int);')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *{len_extra_prefix}, const char *, int);')
            continue

        if _is_allocatable_member(mvar):
            ndim = _get_alloc_ndim(mvar)
            lines.append(
                f'extern unsigned char f2py_get_{sym}_{mname}'
                f'_allocated(void *{len_extra_prefix});')
            lines.append(
                f'extern int f2py_get_{sym}_{mname}'
                f'_ndim(void *{len_extra_prefix});')
            lines.append(
                f'extern void f2py_get_{sym}_{mname}'
                f'_shape(void *{len_extra_prefix}, int *);')
            lines.append(
                f'extern void *f2py_get_{sym}_{mname}'
                f'_data(void *{len_extra_prefix});')
            set_args = ', '.join(
                ['void *'] + ['int'] * len(len_param_names)
                + ['int'] * ndim + ['void *'])
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'({set_args});')
            continue

        if _is_pointer_member(mvar):
            ndim = _get_pointer_ndim(mvar)
            lines.append(
                f'extern unsigned char f2py_get_{sym}_{mname}'
                f'_associated(void *{len_extra_prefix});')
            lines.append(
                f'extern int f2py_get_{sym}_{mname}'
                f'_ndim(void *{len_extra_prefix});')
            lines.append(
                f'extern void f2py_get_{sym}_{mname}'
                f'_shape(void *{len_extra_prefix}, int *);')
            lines.append(
                f'extern void *f2py_get_{sym}_{mname}'
                f'_data(void *{len_extra_prefix});')
            lines.append(
                f'extern void f2py_get_{sym}_{mname}'
                f'_copy(void *{len_extra_prefix}, void *);')
            continue

        if _is_deferred_char_member(mvar):
            lines.append(
                f'extern unsigned char f2py_get_{sym}_{mname}'
                f'_allocated(void *{len_extra_prefix});')
            lines.append(
                f'extern int f2py_get_{sym}_{mname}'
                f'_len(void *{len_extra_prefix});')
            lines.append(
                f'extern void f2py_get_{sym}_{mname}'
                f'(void *{len_extra_prefix}, char *, int);')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *{len_extra_prefix}, const char *, int);')
            continue

        # LEN-sized array members
        if (len_param_name_set
                and _is_len_sized_array(mvar, len_param_name_set)):
            ctype = _get_member_ctype(mvar)
            if ctype is None or _C_TO_NPY_ENUM.get(ctype) is None:
                continue
            ndim = len(mvar.get('dimension', []))
            lines.append(
                f'extern void *f2py_get_{sym}_{mname}'
                f'_data(void *{len_extra_prefix});')
            lines.append(
                f'extern void f2py_get_{sym}_{mname}'
                f'_shape(void *{len_extra_prefix}, int *);')
            set_args = ', '.join(
                ['void *'] + ['int'] * len(len_param_names)
                + ['int'] * ndim + ['void *'])
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'({set_args});')
            continue

        ctype = _get_member_ctype(mvar)
        dims = _get_array_dims(mvar)

        if dims:
            if _C_TO_NPY_ENUM.get(ctype) is None:
                continue
            lines.append(
                f'extern void *f2py_get_{sym}_{mname}'
                f'(void *{len_extra_prefix});')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *{len_extra_prefix}, {ctype} *);')
        else:
            if ctype is None or _C_TO_PYFORMAT.get(ctype) is None:
                continue
            rtype = _get_c_return_type(ctype)
            lines.append(
                f'extern {rtype} f2py_get_{sym}_{mname}'
                f'(void *{len_extra_prefix});')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *{len_extra_prefix}, {ctype});')

    lines.append('')
    return '\n'.join(lines)


def _gen_opaque_extern_decls_one(sym, members, len_info=None):
    """Generate the complete member interface for one specialization."""
    return _gen_opaque_extern_decls(sym, members, len_info=len_info)


def _gen_opaque_capsule_destructor(typename, specializations=None,
                                    len_info=None):
    """Generate PyCapsule destructor that calls Fortran deallocator."""
    if specializations or len_info:
        from ._dt_c_handles import _handle_capsule_destructor
        return _handle_capsule_destructor(typename)
    if specializations:
        # Multi-kind: dispatch based on capsule name
        sym = _fortran_sym(typename)
        branches = []
        for suffix, kind_dict, _ in specializations:
            cap_name = f'f2py.{typename}{suffix}'
            dsym = _fortran_sym(typename + suffix)
            cond = 'if' if not branches else 'else if'
            branches.append(
                f'    {cond} (strcmp(name, "{cap_name}") == 0) {{\n'
                f'        f2py_destroy_{dsym}(ptr);\n'
                f'    }}')
        dispatch = '\n'.join(branches)
        return f"""\
static void
f2py_{typename}_capsule_destructor(PyObject *capsule)
{{
    const char *name = PyCapsule_GetName(capsule);
    void *ptr = PyCapsule_GetPointer(capsule, name);
    if (ptr == NULL) return;
{dispatch}
}}
"""
    sym = _fortran_sym(typename)
    capsule_name = f'f2py.{typename}'
    if len_info:
        # LEN types: retrieve LEN values from capsule context
        # (stored as a malloc'd int array by tp_init)
        nlen = len(len_info)
        len_args = ', '.join(
            f'len_ctx[{i}]' for i in range(nlen))
        return f"""\
static void
f2py_{typename}_capsule_destructor(PyObject *capsule)
{{
    void *ptr = PyCapsule_GetPointer(capsule, "{capsule_name}");
    if (ptr == NULL) return;
    int *len_ctx = (int *)PyCapsule_GetContext(capsule);
    if (len_ctx != NULL) {{
        f2py_destroy_{sym}(ptr, {len_args});
        free(len_ctx);
    }}
}}
"""
    return f"""\
static void
f2py_{typename}_capsule_destructor(PyObject *capsule)
{{
    PyObject *error_type, *error_value, *error_traceback;
    PyErr_Fetch(&error_type, &error_value, &error_traceback);
    void *ptr = PyCapsule_GetPointer(capsule, "{capsule_name}");
    if (ptr != NULL) {{
        f2py_destroy_{sym}(ptr);
    }}
    if (PyErr_Occurred()) PyErr_WriteUnraisable(capsule);
    PyErr_Restore(error_type, error_value, error_traceback);
}}
"""


def _gen_opaque_tp_init(typename, members, specializations=None,
                        kind_info=None, len_info=None, *,
                        symbol_override=None, capsule_override=None,
                        boxed=False):
    """Generate tp_init that calls the Fortran constructor."""
    if specializations and kind_info:
        from ._dt_c_handles import _handle_init_multi
        return _handle_init_multi(
            typename, specializations, kind_info, len_info,
            _gen_opaque_tp_init)
    boxed = boxed or bool(len_info)
    sym = _fortran_sym(symbol_override or typename)
    capsule_name = capsule_override or f'f2py.{typename}'

    # Collect LEN param names for skipping LEN-sized arrays
    len_param_names = set()
    if len_info:
        len_param_names = {li['name'] for li in len_info}

    # Build kwlist: LEN params first (required), then scalar members
    kwlist = []
    fmt_required = []  # required args (LEN params)
    fmt_optional = []  # optional args (scalar members)
    if len_info:
        for li in len_info:
            kwlist.append(li['name'])
            fmt_required.append('i')

    for mname, mvar in members.items():
        if (_is_array_member(mvar) or _is_type_member(mvar)
                or _is_type_array_member(mvar) or _is_char_member(mvar)
                or _is_allocatable_member(mvar)
                or _is_pointer_member(mvar)
                or _is_deferred_char_member(mvar)):
            continue  # arrays, nested types, chars, allocs set via props
        if (len_param_names
                and _is_len_sized_array(mvar, len_param_names)):
            continue
        ctype = _get_member_ctype(mvar)
        fmt = _C_TO_PYFORMAT.get(ctype)
        if fmt is None:
            continue
        kwlist.append(mname)
        fmt_optional.append(fmt)

    kwlist_str = ', '.join(f'"{k}"' for k in kwlist)
    # Format: required args, then '|', then optional args
    if fmt_required:
        fmt_str = ''.join(fmt_required) + '|' + ''.join(fmt_optional)
    else:
        fmt_str = '|' + ''.join(fmt_optional)

    decl_lines = []
    parse_args = []
    call_args = []

    # LEN param declarations and parse args
    if len_info:
        for li in len_info:
            lname = li['name']
            decl_lines.append(f'    int {lname} = 0;')
            parse_args.append(f'&{lname}')
            call_args.append(lname)

    for mname, mvar in members.items():
        if (_is_array_member(mvar) or _is_type_member(mvar)
                or _is_type_array_member(mvar) or _is_char_member(mvar)
                or _is_allocatable_member(mvar)
                or _is_pointer_member(mvar)
                or _is_deferred_char_member(mvar)):
            continue
        if (len_param_names
                and _is_len_sized_array(mvar, len_param_names)):
            continue
        ctype = _get_member_ctype(mvar)
        if _C_TO_PYFORMAT.get(ctype) is None:
            continue
        fortran_default = mvar.get('=')
        if ctype in ('npy_cfloat', 'npy_cdouble'):
            decl_lines.append(f'    Py_complex {mname} = {{0, 0}};')
            parse_args.append(f'&{mname}')
            if ctype == 'npy_cfloat':
                call_args.append(
                    f'npy_cpackf((float){mname}.real, (float){mname}.imag)')
            else:
                call_args.append(
                    f'npy_cpack({mname}.real, {mname}.imag)')
        else:
            if fortran_default is not None:
                c_default = str(fortran_default).strip()
                c_default = c_default.replace('d', 'e').replace('D', 'E')
                if c_default.lower() == '.true.':
                    c_default = '1'
                elif c_default.lower() == '.false.':
                    c_default = '0'
                decl_lines.append(f'    {ctype} {mname} = {c_default};')
            else:
                decl_lines.append(f'    {ctype} {mname} = 0;')
            parse_args.append(f'&{mname}')
            call_args.append(mname)

    decl_str = '\n'.join(decl_lines)
    parse_args_str = ', '.join(parse_args)
    call_args_str = ', '.join(call_args)

    if kwlist:
        parse_block = f"""\
    static char *kwlist[] = {{{kwlist_str}, NULL}};
{decl_str}

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "{fmt_str}", kwlist,
                                     {parse_args_str}))
        return -1;"""
    else:
        parse_block = """\
    /* No scalar members; all members are arrays set via properties */"""

    # Ordinary handles carry their destructor arguments in context.
    if len_info and not boxed:
        nlen = len(len_info)
        store_len = '\n'.join(
            f'    self->len_{li["name"]} = {li["name"]};'
            for li in len_info)
        len_destr_args = ', '.join(
            li['name'] for li in len_info)
        ctx_lines = f"""\

{store_len}

    /* Store LEN params in capsule context for destructor */
    int *len_ctx = (int *)malloc({nlen} * sizeof(int));
    if (len_ctx == NULL) {{
        f2py_destroy_{sym}(ptr, {len_destr_args});
        PyErr_NoMemory();
        return -1;
    }}"""
        for i, li in enumerate(len_info):
            ctx_lines += f'\n    len_ctx[{i}] = {li["name"]};'
        ctx_lines += """
    PyCapsule_SetContext(self->capsule, len_ctx);"""
        post_capsule = ctx_lines
    else:
        post_capsule = ''

    release = (f'f2py_{typename}_release_preserving_error(ptr);' if boxed else
               f'f2py_destroy_{sym}(ptr' +
               (', ' + ', '.join(li['name'] for li in len_info)
                if len_info else '') + ');')
    hydrate = (f'    if (f2py_{typename}_hydrate(selfobj, ptr) < 0) return -1;\n'
               if boxed else '')

    return f"""\
static int
Py{typename}_tp_init(PyObject *selfobj, PyObject *args, PyObject *kwds)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
{parse_block}

    /* Call Fortran constructor (allocates the derived type) */
    void *ptr = f2py_create_{sym}({call_args_str});
    if (ptr == NULL) {{
        PyErr_SetString(PyExc_MemoryError,
                        "Fortran allocate failed for {typename}");
        return -1;
    }}

    /* Clean up old capsule if re-initializing */
    Py_XDECREF(self->capsule);
    self->capsule = PyCapsule_New(
        ptr, "{capsule_name}",
        f2py_{typename}_capsule_destructor);
    if (self->capsule == NULL) {{
        {release}
        return -1;
    }}{post_capsule}

    {hydrate}return 0;
}}
"""


def _gen_opaque_tp_init_multi(typename, members, specializations, kind_info):
    """Generate multi-kind tp_init with dispatch."""
    # Use the first specialization's members for the kwlist
    # (all specializations have the same member names)
    _, _, first_members = specializations[0]
    ki = kind_info[0]  # First (typically only) KIND param
    kname = ki['name']
    has_default = ki['has_default']
    default_val = ki['default']

    # Build kwlist: kind param first, then data members
    kwlist = [kname]
    fmt_parts = ['i']  # kind is always int
    for mname, mvar in first_members.items():
        if (_is_array_member(mvar) or _is_type_member(mvar)
                or _is_type_array_member(mvar) or _is_char_member(mvar)
                or _is_allocatable_member(mvar)
                or _is_pointer_member(mvar)
                or _is_deferred_char_member(mvar)):
            continue
        ctype = _get_member_ctype(mvar)
        fmt = _C_TO_PYFORMAT.get(ctype)
        if fmt is None:
            continue
        kwlist.append(mname)
        # Parse all numeric args as double (widest), cast on call
        fmt_parts.append('d')

    kwlist_str = ', '.join(f'"{k}"' for k in kwlist)
    fmt_str = ''.join(fmt_parts)

    # Declarations: kind + all members as double
    decl_lines = []
    if has_default:
        decl_lines.append(f'    int {kname} = {default_val};')
    else:
        decl_lines.append(f'    int {kname} = -1;')
    parse_args = [f'&{kname}']

    member_names = []
    for mname, mvar in first_members.items():
        if (_is_array_member(mvar) or _is_type_member(mvar)
                or _is_type_array_member(mvar) or _is_char_member(mvar)
                or _is_allocatable_member(mvar)
                or _is_pointer_member(mvar)
                or _is_deferred_char_member(mvar)):
            continue
        ctype = _get_member_ctype(mvar)
        if _C_TO_PYFORMAT.get(ctype) is None:
            continue
        decl_lines.append(f'    double {mname} = 0;')
        parse_args.append(f'&{mname}')
        member_names.append(mname)

    decl_str = '\n'.join(decl_lines)
    parse_args_str = ', '.join(parse_args)

    # Build dispatch branches
    branches = []
    for suffix, kind_dict, resolved_members in specializations:
        kv = kind_dict[kname]
        sym = _fortran_sym(typename + suffix)
        cap_name = f'f2py.{typename}{suffix}'

        # Build call args with casts
        call_parts = []
        for mname in member_names:
            mvar = resolved_members.get(mname, {})
            ctype = _get_member_ctype(mvar) or 'double'
            if ctype == 'float':
                call_parts.append(f'(float){mname}')
            elif ctype == 'int':
                call_parts.append(f'(int){mname}')
            elif ctype == 'long long':
                call_parts.append(f'(long long){mname}')
            else:
                call_parts.append(mname)
        call_args_str = ', '.join(call_parts)

        cond = 'if' if not branches else 'else if'
        branches.append(f"""\
    {cond} ({kname} == {kv}) {{
        ptr = f2py_create_{sym}({call_args_str});
        capsule_name = "{cap_name}";
    }}""")

    dispatch = '\n'.join(branches)

    # Required kind check for types without default
    if not has_default:
        required_check = f"""\
    if ({kname} == -1) {{
        PyErr_SetString(PyExc_TypeError,
                        "{typename}() requires '{kname}=' keyword argument");
        return -1;
    }}"""
    else:
        required_check = ''

    # Valid kind values for error message
    valid_kinds = [str(kind_dict[kname])
                   for _, kind_dict, _ in specializations]
    valid_str = ', '.join(valid_kinds)

    return f"""\
static int
Py{typename}_tp_init(PyObject *selfobj, PyObject *args, PyObject *kwds)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    static char *kwlist[] = {{{kwlist_str}, NULL}};
{decl_str}

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|{fmt_str}", kwlist,
                                     {parse_args_str}))
        return -1;
{required_check}

    void *ptr = NULL;
    const char *capsule_name = NULL;
{dispatch}
    else {{
        PyErr_Format(PyExc_ValueError,
                     "unsupported {kname}=%d for {typename} "
                     "(valid: {valid_str})", {kname});
        return -1;
    }}

    if (ptr == NULL) {{
        PyErr_SetString(PyExc_MemoryError,
                        "Fortran allocate failed for {typename}");
        return -1;
    }}

    self->kind_value = {kname};

    /* Clean up old capsule if re-initializing */
    Py_XDECREF(self->capsule);
    self->capsule = PyCapsule_New(
        ptr, capsule_name,
        f2py_{typename}_capsule_destructor);
    if (self->capsule == NULL) {{
        /* capsule creation failed -- leak is acceptable (extremely rare) */
        return -1;
    }}

    return 0;
}}
"""


def _gen_opaque_getset(typename, members, specializations=None,
                       kind_info=None, len_info=None, type_map=None, *,
                       symbol_override=None, capsule_override=None,
                       return_parts=False, boxed=False):
    """Generate getters/setters that call Fortran accessor wrappers."""
    if specializations and kind_info:
        from ._dt_c_handles import _handle_getset_multi
        return _handle_getset_multi(
            typename, specializations, kind_info, len_info,
            type_map, _gen_opaque_getset)
    boxed = boxed or bool(len_info)
    sym = _fortran_sym(symbol_override or typename)
    capsule_name = capsule_override or f'f2py.{typename}'
    funcs = []
    getset_entries = []

    # For LEN types, build the extra args string for Fortran calls.
    # e.g. ', self->len_n' or ', self->len_m, self->len_n' or ''.
    len_param_names = set()
    len_call_extra = ''
    if len_info:
        len_param_names = {li['name'] for li in len_info}
        len_call_extra = ', '.join(
            f'self->len_{li["name"]}' for li in len_info)
        len_call_extra = ', ' + len_call_extra

    # Add read-only LEN properties
    if len_info:
        for li in len_info:
            lname = li['name']
            getter_name = f'Py{typename}_get_len_{lname}'
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    return PyLong_FromLong((long)self->len_{lname});
}}
""")
            getset_entries.append(
                f'    {{"{lname}", {getter_name}, NULL, '
                f'"{lname} (LEN parameter, read-only)", NULL}},')

    for mname, mvar in members.items():
        if _is_coarray_member(mvar):
            mvar = _coarray_as_local(mvar)  # F2018 7.5.4.3: local data
        if _is_private_member(mvar):
            continue  # F2018 7.5.4.8: private components not exposed
        if _is_type_array_member(mvar):
            code, entry = record_array_c(
                typename, mname, mvar, type_map, symbol=sym,
                len_call=len_call_extra)
            funcs.append(code)
            getset_entries.append(entry)
            continue

        if _is_type_member(mvar):
            # Nested derived type member via opaque path
            inner = mvar.get('typename', '').lower()
            inner_sym = _fortran_sym(inner)
            inner_capsule = f'f2py.{inner}'
            from ._dt_handles import _handle_description
            inner_spec = inner + ('(' + mvar['typeparams'] + ')'
                                  if mvar.get('typeparams') else '')
            inner_description = _handle_description(type_map[inner], inner_spec)
            from ._dt_helpers import _can_wrap_bindc
            if _can_wrap_bindc(type_map[inner], type_map):
                make_owned = f"""\
    void *owned = PyMem_Malloc(sizeof(f2py_{inner}_t));
    if (owned == NULL) return PyErr_NoMemory();
    memcpy(owned, inner_ptr, sizeof(f2py_{inner}_t));
    inner_ptr = owned;"""
                release = 'PyMem_Free(inner_ptr);'
                if boxed:
                    free = f'f2py_freecopy_{sym}_{mname}'
                    make_owned = f"""\
    extern void {free}(void *);
    void *owned = PyMem_Malloc(sizeof(f2py_{inner}_t));
    if (owned == NULL) {{
        {free}(inner_ptr);
        return PyErr_NoMemory();
    }}
    memcpy(owned, inner_ptr, sizeof(f2py_{inner}_t));
    {free}(inner_ptr);
    inner_ptr = owned;"""
            else:
                make_owned = ''
                release = ('f2py_pdt_destroy(inner_ptr);' if inner_description
                           else f'f2py_destroy_{inner_sym}(inner_ptr);')

            if inner_description is not None:
                make_object = (f'    return f2py_{inner}_from_handle(\n'
                               f'        inner_ptr, NULL, f2py_{inner}_capsule_destructor, NULL);')
            else:
                make_object = f"""\
    PyObject *obj = Py{inner}_tp_new(&Py{inner}_Type, NULL, NULL);
    if (obj == NULL) {{
        {release}
        return NULL;
    }}
    ((Py{inner}Object *)obj)->capsule = PyCapsule_New(
        inner_ptr, "{inner_capsule}",
        f2py_{inner}_capsule_destructor);
    if (((Py{inner}Object *)obj)->capsule == NULL) {{
        {release}
        Py_DECREF(obj);
        return NULL;
    }}
    return obj;
"""

            getter_name = f'Py{typename}_get_{mname}'
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    /* Get an opaque copy of the nested type from Fortran */
    void *inner_ptr = f2py_get_{sym}_{mname}(ptr{len_call_extra});
    if (inner_ptr == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "Fortran returned NULL for {mname}");
        return NULL;
    }}
{make_owned}
{make_object}
}}
""")

            if inner_description is not None:
                source_pointer = (
                    f'f2py_{inner}_checked(value, PyCapsule_GetName(inner_obj->capsule), 0)')
                source_identity_check = f'''    if (f2py_pdt_identity(inner_ptr) != {inner_description['identity_number']}LL) {{
        PyErr_SetString(PyExc_TypeError, "member needs the declared KIND specialization");
        return -1;
    }}'''
            else:
                source_pointer = f'PyCapsule_GetPointer(inner_obj->capsule, \"{inner_capsule}\")'
                source_identity_check = ''

            setter_name = f'Py{typename}_set_{mname}'
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    if (!Py_IS_TYPE(value, &Py{inner}_Type)) {{
        PyErr_SetString(PyExc_TypeError,
                        "{mname} must be a {inner} instance");
        return -1;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return -1;
    Py{inner}Object *inner_obj = (Py{inner}Object *)value;
    if (inner_obj->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{inner} object not initialized");
        return -1;
    }}
    void *inner_ptr = {source_pointer};
    if (inner_ptr == NULL) return -1;
{source_identity_check}
    PyObject *parent_owner = Py_NewRef(self->capsule);
    PyObject *inner_owner = Py_NewRef(inner_obj->capsule);
    f2py_set_{sym}_{mname}(ptr{len_call_extra}, inner_ptr);
    Py_DECREF(inner_owner);
    Py_DECREF(parent_owner);
    return PyErr_Occurred() ? -1 : 0;
}}
""")

            getset_entries.append(
                f'    {{"{mname}", {getter_name}, {setter_name}, '
                f'"{mname} member", NULL}},'
            )
            continue

        if _is_char_member(mvar):
            char_len = _get_char_len(mvar)
            getter_name = f'Py{typename}_get_{mname}'
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    char buf[{char_len + _C_STRING_NUL}];
    memset(buf, 0, {char_len + _C_STRING_NUL});
    f2py_get_{sym}_{mname}(ptr{len_call_extra}, buf, {char_len});
    /* Trim trailing spaces */
    int len = {char_len};
    while (len > 0 && buf[len - 1] == ' ') len--;
    return PyUnicode_FromStringAndSize(buf, len);
}}
""")

            setter_name = f'Py{typename}_set_{mname}'
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    if (!PyUnicode_Check(value)) {{
        PyErr_SetString(PyExc_TypeError,
                        "{mname} must be a string");
        return -1;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return -1;
    Py_ssize_t slen;
    const char *str = PyUnicode_AsUTF8AndSize(value, &slen);
    if (str == NULL) return -1;
    f2py_set_{sym}_{mname}(ptr{len_call_extra}, str, (int)slen);
    return 0;
}}
""")

            getset_entries.append(
                f'    {{"{mname}", {getter_name}, {setter_name}, '
                f'"{mname} member", NULL}},'
            )
            continue

        if _is_complex_member(mvar):
            ctype = _get_member_ctype(mvar)
            is_single = (ctype == 'npy_cfloat')
            creal_fn = 'npy_crealf' if is_single else 'npy_creal'
            cimag_fn = 'npy_cimagf' if is_single else 'npy_cimag'
            cpack_fn = 'npy_cpackf' if is_single else 'npy_cpack'
            cast = '(float)' if is_single else ''

            getter_name = f'Py{typename}_get_{mname}'
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    {ctype} val = f2py_get_{sym}_{mname}(ptr{len_call_extra});
    return PyComplex_FromDoubles(
        (double){creal_fn}(val),
        (double){cimag_fn}(val));
}}
""")

            setter_name = f'Py{typename}_set_{mname}'
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return -1;
    Py_complex c = PyComplex_AsCComplex(value);
    if (PyErr_Occurred()) return -1;
    {ctype} cval = {cpack_fn}({cast}c.real, {cast}c.imag);
    f2py_set_{sym}_{mname}(ptr{len_call_extra}, cval);
    return 0;
}}
""")

            getset_entries.append(
                f'    {{"{mname}", {getter_name}, {setter_name}, '
                f'"{mname} member", NULL}},'
            )
            continue

        if _is_allocatable_member(mvar):
            alloc_ctype = _get_member_ctype(mvar)
            npy_enum = _C_TO_NPY_ENUM.get(alloc_ctype)
            if npy_enum is None:
                continue
            ndim = _get_alloc_ndim(mvar)

            getter_name = f'Py{typename}_get_{mname}'
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    unsigned char is_alloc = f2py_get_{sym}_{mname}_allocated(ptr{len_call_extra});
    if (!is_alloc) Py_RETURN_NONE;
    /* Query rank and shape from Fortran (F2018 16.9.182 size intrinsic) */
    int rank = f2py_get_{sym}_{mname}_ndim(ptr{len_call_extra});
    int fortran_shape[{ndim}];
    f2py_get_{sym}_{mname}_shape(ptr{len_call_extra}, fortran_shape);
    npy_intp numpy_dims[{ndim}];
    npy_intp total_elements = 1;
    for (int dim_idx = 0; dim_idx < rank; dim_idx++) {{
        numpy_dims[dim_idx] = (npy_intp)fortran_shape[dim_idx];
    }}
    if (!f2py_dt_shape_product(rank, numpy_dims, &total_elements)) return NULL;
    void *data = f2py_get_{sym}_{mname}_data(ptr{len_call_extra});
    return f2py_dt_copy_array(self->capsule, rank, numpy_dims,
        {npy_enum}, data, total_elements * sizeof({alloc_ctype}));
}}
""")

            # Build setter args for the Fortran set call
            # f2py_set_TYPE_MEMBER(ptr, n1, n2, ..., src)
            set_dim_args = ', '.join(
                f'(int)PyArray_DIM((PyArrayObject *)arr, {i})'
                for i in range(ndim))
            dealloc_args = ', '.join(['0'] * ndim + ['NULL'])

            setter_name = f'Py{typename}_set_{mname}'
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return -1;
    if (value == Py_None) {{
        f2py_set_{sym}_{mname}(ptr{len_call_extra}, {dealloc_args});
        return 0;
    }}
    PyObject *arr = f2py_dt_array_from_any(value, {npy_enum},
                                            NPY_ARRAY_FARRAY_RO);
    if (arr == NULL) return -1;
    if (!f2py_dt_checked_shape((PyArrayObject *)arr, {ndim})) {{
        Py_DECREF(arr);
        return -1;
    }}
    ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) {{
        Py_DECREF(arr);
        return -1;
    }}
    if (PyArray_NDIM((PyArrayObject *)arr) != {ndim}) {{
        PyErr_Format(PyExc_ValueError,
                     "{mname} requires a {ndim}D array, got %dD",
                     PyArray_NDIM((PyArrayObject *)arr));
        Py_DECREF(arr);
        return -1;
    }}
    f2py_set_{sym}_{mname}(ptr{len_call_extra}, {set_dim_args},
                            PyArray_DATA((PyArrayObject *)arr));
    Py_DECREF(arr);
    return 0;
}}
""")

            getset_entries.append(
                f'    {{"{mname}", {getter_name}, {setter_name}, '
                f'"{mname} member", NULL}},'
            )
            continue

        if _is_pointer_member(mvar):
            # Pointer member: read-only getter, no setter
            # F2018 7.5.4.6 pointer components, 16.9.16 associated()
            ptr_ctype = _get_member_ctype(mvar)
            npy_enum = _C_TO_NPY_ENUM.get(ptr_ctype)
            if npy_enum is None:
                continue
            ndim = _get_pointer_ndim(mvar)

            getter_name = f'Py{typename}_get_{mname}'
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    /* Check pointer association (F2018 16.9.16 associated()) */
    unsigned char is_assoc = f2py_get_{sym}_{mname}_associated(ptr{len_call_extra});
    if (!is_assoc) Py_RETURN_NONE;
    /* Query shape from Fortran (F2018 16.9.182 size()) */
    int rank = f2py_get_{sym}_{mname}_ndim(ptr{len_call_extra});
    int fortran_shape[{ndim}];
    f2py_get_{sym}_{mname}_shape(ptr{len_call_extra}, fortran_shape);
    npy_intp numpy_dims[{ndim}];
    npy_intp total_elements = 1;
    for (int dim_idx = 0; dim_idx < rank; dim_idx++) {{
        numpy_dims[dim_idx] = (npy_intp)fortran_shape[dim_idx];
    }}
    if (!f2py_dt_shape_product(rank, numpy_dims, &total_elements)) return NULL;
    PyObject *owner = Py_NewRef(self->capsule);
    PyObject *arr = PyArray_EMPTY(rank, numpy_dims, {npy_enum}, 1);
    if (arr != NULL && total_elements > 0) {{
        f2py_get_{sym}_{mname}_copy(ptr{len_call_extra},
            PyArray_DATA((PyArrayObject *)arr));
    }}
    Py_DECREF(owner);
    return arr;
}}
""")

            # Read-only: setter raises AttributeError
            setter_name = f'Py{typename}_set_{mname}'
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    PyErr_SetString(PyExc_AttributeError,
                    "{mname} is a Fortran pointer member (read-only)");
    return -1;
}}
""")

            getset_entries.append(
                f'    {{"{mname}", {getter_name}, {setter_name}, '
                f'"{mname} pointer member (read-only)", NULL}},'
            )
            continue

        if _is_deferred_char_member(mvar):
            # Deferred-length allocatable character (F2018 7.4.4.2)
            # Getter: query length via len() (F2018 16.9.109),
            # copy to buffer, return as Python string
            getter_name = f'Py{typename}_get_{mname}'
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    /* Check allocation status (F2018 16.9.3 allocated()) */
    unsigned char is_alloc = f2py_get_{sym}_{mname}_allocated(ptr{len_call_extra});
    if (!is_alloc) Py_RETURN_NONE;
    /* Query runtime length (F2018 16.9.109 len()) */
    int str_length = f2py_get_{sym}_{mname}_len(ptr{len_call_extra});
    if (str_length <= 0) return PyUnicode_FromString("");
    char *buffer = (char *)malloc(str_length + {_C_STRING_NUL});
    if (buffer == NULL) return PyErr_NoMemory();
    f2py_get_{sym}_{mname}(ptr{len_call_extra}, buffer, str_length);
    buffer[str_length] = '\\0';
    /* Trim trailing spaces (Fortran pads with spaces) */
    int trimmed_length = str_length;
    while (trimmed_length > 0 && buffer[trimmed_length - 1] == ' ')
        trimmed_length--;
    PyObject *result = PyUnicode_FromStringAndSize(buffer, trimmed_length);
    free(buffer);
    return result;
}}
""")

            # Setter: allocate with new length, copy from Python string
            setter_name = f'Py{typename}_set_{mname}'
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return -1;
    if (value == Py_None) {{
        /* Deallocate: pass empty string with length 0 */
        f2py_set_{sym}_{mname}(ptr{len_call_extra}, "", 0);
        return 0;
    }}
    if (!PyUnicode_Check(value)) {{
        PyErr_SetString(PyExc_TypeError,
                        "{mname} must be a string or None");
        return -1;
    }}
    Py_ssize_t py_str_length;
    const char *str_data = PyUnicode_AsUTF8AndSize(value, &py_str_length);
    if (str_data == NULL) return -1;
    f2py_set_{sym}_{mname}(ptr{len_call_extra}, str_data, (int)py_str_length);
    return 0;
}}
""")

            getset_entries.append(
                f'    {{"{mname}", {getter_name}, {setter_name}, '
                f'"{mname} member", NULL}},'
            )
            continue

        # LEN-sized array members: dynamic size from LEN param
        if (len_param_names
                and _is_len_sized_array(mvar, len_param_names)):
            ctype = _get_member_ctype(mvar)
            npy_enum = _C_TO_NPY_ENUM.get(ctype)
            if npy_enum is None:
                continue
            ndim = len(mvar.get('dimension', []))
            # Determine shape from LEN params stored in self
            # For 1D arrays sized by single LEN param, shape = (len_n,)
            dim_exprs = []
            for d in mvar.get('dimension', []):
                ds = str(d).strip().lower()
                if ds in len_param_names:
                    dim_exprs.append(f'self->len_{ds}')
                else:
                    dim_exprs.append(ds)

            getter_name = f'Py{typename}_get_{mname}'
            dims_assign = '\n'.join(
                f'    dims[{i}] = (npy_intp){dim_exprs[i]};'
                for i in range(ndim))
            total_expr = ' * '.join(dim_exprs)
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    int native_dims[{ndim}];
    f2py_get_{sym}_{mname}_shape(ptr{len_call_extra}, native_dims);
    npy_intp dims[{ndim}];
    for (int d = 0; d < {ndim}; d++) dims[d] = native_dims[d];
    npy_intp total;
    if (!f2py_dt_shape_product({ndim}, dims, &total)) return NULL;
    if (total <= 0) {{
        return PyArray_EMPTY({ndim}, dims, {npy_enum}, 1);
    }}
    void *data = f2py_get_{sym}_{mname}_data(ptr{len_call_extra});
    if (data == NULL) {{
        return PyArray_EMPTY({ndim}, dims, {npy_enum}, 1);
    }}
    return f2py_dt_copy_array(self->capsule, {ndim}, dims,
        {npy_enum}, data, total * sizeof({ctype}));
}}
""")

            # Build setter args for Fortran
            set_dim_args = ', '.join(
                f'(int)PyArray_DIM((PyArrayObject *)arr, {i})'
                for i in range(ndim))

            setter_name = f'Py{typename}_set_{mname}'
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return -1;
    PyObject *arr = f2py_dt_array_from_any(value, {npy_enum},
                                            NPY_ARRAY_FARRAY_RO);
    if (arr == NULL) return -1;
    if (!f2py_dt_checked_shape((PyArrayObject *)arr, {ndim})) {{
        Py_DECREF(arr);
        return -1;
    }}
    ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) {{
        Py_DECREF(arr);
        return -1;
    }}
    if (PyArray_NDIM((PyArrayObject *)arr) != {ndim}) {{
        PyErr_SetString(PyExc_ValueError, "{mname} has an incorrect rank");
        Py_DECREF(arr);
        return -1;
    }}
    int expected_dims[{ndim}];
    f2py_get_{sym}_{mname}_shape(ptr{len_call_extra}, expected_dims);
    for (int d = 0; d < {ndim}; d++) {{
        if (PyArray_DIM((PyArrayObject *)arr, d) != expected_dims[d]) {{
            PyErr_SetString(PyExc_ValueError,
                            "{mname} has an incorrect shape");
            Py_DECREF(arr);
            return -1;
        }}
    }}
    f2py_set_{sym}_{mname}(ptr{len_call_extra}, {set_dim_args},
                            PyArray_DATA((PyArrayObject *)arr));
    Py_DECREF(arr);
    return 0;
}}
""")
            getset_entries.append(
                f'    {{"{mname}", {getter_name}, {setter_name}, '
                f'"{mname} member", NULL}},')
            continue

        ctype = _get_member_ctype(mvar)
        dims = _get_array_dims(mvar)

        if dims:
            # Array member -- getter gets c_ptr from Fortran, wraps
            # as NumPy array (copy, since Fortran manages the memory)
            npy_enum = _C_TO_NPY_ENUM.get(ctype)
            if npy_enum is None:
                continue
            ndim = len(dims)
            total = 1
            for d in dims:
                total *= d
            dims_init = ', '.join(str(d) for d in dims)

            getter_name = f'Py{typename}_get_{mname}'
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    if ({total} == 0) {{
        npy_intp dims[{ndim}] = {{{dims_init}}};
        return PyArray_EMPTY({ndim}, dims, {npy_enum}, 1);
    }}
    void *arrptr = f2py_get_{sym}_{mname}(ptr{len_call_extra});
    if (arrptr == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "Fortran returned NULL for {mname}");
        return NULL;
    }}
    npy_intp dims[{ndim}] = {{{dims_init}}};
    /* Copy data out of Fortran memory */
    return f2py_dt_copy_array(self->capsule, {ndim}, dims,
        {npy_enum}, arrptr, {total} * sizeof({ctype}));
}}
""")

            setter_name = f'Py{typename}_set_{mname}'
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return -1;
    PyObject *arr = f2py_dt_array_from_any(value, {npy_enum},
                                            NPY_ARRAY_FARRAY_RO);
    if (arr == NULL) return -1;
    if (!f2py_dt_checked_shape((PyArrayObject *)arr, {ndim})) {{
        Py_DECREF(arr);
        return -1;
    }}
    ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) {{
        Py_DECREF(arr);
        return -1;
    }}
    if (PyArray_NDIM((PyArrayObject *)arr) != {ndim}) {{
        PyErr_SetString(PyExc_ValueError, "{mname} has an incorrect rank");
        Py_DECREF(arr);
        return -1;
    }}
    const npy_intp expected_dims[{ndim}] = {{{dims_init}}};
    for (int d = 0; d < {ndim}; d++) {{
        if (PyArray_DIM((PyArrayObject *)arr, d) != expected_dims[d]) {{
            PyErr_SetString(PyExc_ValueError,
                            "{mname} has an incorrect shape");
            Py_DECREF(arr);
            return -1;
        }}
    }}
    f2py_set_{sym}_{mname}(ptr{len_call_extra},
        ({ctype} *)PyArray_DATA((PyArrayObject *)arr));
    Py_DECREF(arr);
    return 0;
}}
""")
        else:
            # Scalar member
            pyobj_expr = _C_TO_PYOBJ.get(ctype)
            frompy_expr = _PYOBJ_TO_C.get(ctype)
            if pyobj_expr is None or frompy_expr is None:
                continue

            getter_name = f'Py{typename}_get_{mname}'
            rtype = _get_c_return_type(ctype)
            val_expr = pyobj_expr.format(val='val')
            funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    {rtype} val = f2py_get_{sym}_{mname}(ptr{len_call_extra});
    return {val_expr};
}}
""")

            setter_name = f'Py{typename}_set_{mname}'
            conv_expr = frompy_expr.format(obj='value')
            funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return -1;
    {ctype} cval = {conv_expr};
    if (PyErr_Occurred()) return -1;
    f2py_set_{sym}_{mname}(ptr{len_call_extra}, cval);
    return 0;
}}
""")

        getset_entries.append(
            f'    {{"{mname}", {getter_name}, {setter_name}, '
            f'"{mname} member", NULL}},'
        )

    getset_array = (
        f'static PyGetSetDef Py{typename}_getset[] = {{\n'
        + '\n'.join(getset_entries) + '\n'
        + '    {NULL}  /* sentinel */\n'
        + '};\n'
    )

    if boxed:
        from ._dt_c_handles import _guard_boxed_accessors
        from ._dt_numeric_handles import _numeric_getters
        funcs = _numeric_getters(
            funcs, typename, sym, members, capsule_name, type_map)
        funcs = _guard_boxed_accessors(funcs, typename)
    if return_parts:
        return funcs, getset_entries
    return '\n'.join(funcs) + '\n' + getset_array


def _gen_kind_array_getset(typename, members, specializations, kind_info,
                           type_map):
    """Dispatch numeric array access to the selected KIND representation."""
    arrays = {name: var for name, var in members.items()
              if var.get('typespec') != 'type'
              and (_is_array_member(var) or _is_allocatable_member(var)
                   or _is_pointer_member(var))
              and not _is_private_member(var)}
    if not arrays:
        return [], []
    functions, entries = [], []
    kind_name = kind_info[0]['name']
    for suffix, _, resolved in specializations:
        selected = {name: resolved[name] for name in arrays}
        body = _gen_opaque_getset(typename, selected, type_map=type_map)
        body = body.rsplit('static PyGetSetDef ', 1)[0]
        body = body.replace(f'Py{typename}_get_', f'Py{typename}{suffix}_get_')
        body = body.replace(f'Py{typename}_set_', f'Py{typename}{suffix}_set_')
        symbol = _fortran_sym(typename)
        specialized = _fortran_sym(typename + suffix)
        body = body.replace(f'f2py_get_{symbol}_', f'f2py_get_{specialized}_')
        body = body.replace(f'f2py_set_{symbol}_', f'f2py_set_{specialized}_')
        body = body.replace(f'"f2py.{typename}"', f'"f2py.{typename}{suffix}"')
        functions.append(body)
    for name in arrays:
        for operation, returns, arguments, call in (
                ('get', 'PyObject *', '', 'selfobj, closure'),
                ('set', 'int', ', PyObject *value', 'selfobj, value, closure')):
            branches = []
            for suffix, kinds, _ in specializations:
                branches.append(
                    f'    if (self->kind_value == {kinds[kind_name]})\n'
                    f'        return Py{typename}{suffix}_{operation}_{name}({call});')
            error = 'NULL' if operation == 'get' else '-1'
            if operation == 'get':
                signature = 'PyObject *selfobj, void *closure'
            else:
                signature = 'PyObject *selfobj, PyObject *value, void *closure'
            functions.append(
                f'static {returns}\nPy{typename}_{operation}_{name}({signature})\n'
                '{\n'
                f'    Py{typename}Object *self = (Py{typename}Object *)selfobj;\n'
                + '\n'.join(branches) + '\n'
                '    PyErr_SetString(PyExc_ValueError, "Unsupported KIND specialization");\n'
                f'    return {error};\n' + '}\n')
        entries.append(
            f'    {{"{name}", Py{typename}_get_{name}, '
            f'Py{typename}_set_{name}, "{name} member", NULL}},')
    return functions, entries


def _gen_opaque_getset_multi(typename, members, specializations, kind_info,
                            type_map=None):
    """Generate multi-kind dispatch getters/setters for PDTs."""
    ki = kind_info[0]
    kname = ki['name']
    funcs, getset_entries = _gen_kind_array_getset(
        typename, members, specializations, kind_info, type_map)

    # Add read-only 'k' property for the kind value
    getter_name = f'Py{typename}_get_{kname}'
    funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    return PyLong_FromLong((long)self->kind_value);
}}
""")
    getset_entries.append(
        f'    {{"{kname}", {getter_name}, NULL, '
        f'"{kname} (KIND parameter, read-only)", NULL}},')

    for mname, mvar in members.items():
        if _is_coarray_member(mvar):
            mvar = _coarray_as_local(mvar)  # F2018 7.5.4.3: local data
        if _is_private_member(mvar):
            continue  # F2018 7.5.4.8
        # Skip complex member types for now (array-of-types, nested, etc.)
        if (_is_type_array_member(mvar) or _is_type_member(mvar)
                or _is_char_member(mvar) or _is_allocatable_member(mvar)
                or _is_pointer_member(mvar)
                or _is_deferred_char_member(mvar)
                or _is_array_member(mvar)):
            continue
        if _is_complex_member(mvar):
            continue  # Complex dispatch not yet implemented

        ctype = _get_member_ctype(mvar)
        if ctype is None or _C_TO_PYFORMAT.get(ctype) is None:
            continue

        # Build dispatch getter
        getter_name = f'Py{typename}_get_{mname}'
        getter_branches = []
        for suffix, kind_dict, resolved_members in specializations:
            kv = kind_dict[kname]
            dsym = _fortran_sym(typename + suffix)
            cap_name = f'f2py.{typename}{suffix}'
            rmvar = resolved_members.get(mname, mvar)
            rctype = _get_member_ctype(rmvar) or ctype
            if rctype in ('float', 'double'):
                val_expr = (f'(double)f2py_get_{dsym}_{mname}(ptr)')
                ret_expr = f'PyFloat_FromDouble({val_expr})'
            elif rctype in ('int',):
                ret_expr = (
                    f'PyLong_FromLong((long)'
                    f'f2py_get_{dsym}_{mname}(ptr))')
            elif rctype in ('long long',):
                ret_expr = (
                    f'PyLong_FromLongLong('
                    f'f2py_get_{dsym}_{mname}(ptr))')
            else:
                ret_expr = (
                    f'PyFloat_FromDouble((double)'
                    f'f2py_get_{dsym}_{mname}(ptr))')

            cond = 'if' if not getter_branches else 'else if'
            getter_branches.append(
                f'    {cond} (self->kind_value == {kv}) {{\n'
                f'        return {ret_expr};\n'
                f'    }}')

        getter_dispatch = '\n'.join(getter_branches)
        funcs.append(f"""\
static PyObject *
{getter_name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    const char *cap = PyCapsule_GetName(self->capsule);
    void *ptr = PyCapsule_GetPointer(self->capsule, cap);
    if (ptr == NULL) return NULL;
{getter_dispatch}
    PyErr_SetString(PyExc_RuntimeError, "invalid kind_value");
    return NULL;
}}
""")

        # Build dispatch setter
        setter_name = f'Py{typename}_set_{mname}'
        setter_branches = []
        for suffix, kind_dict, resolved_members in specializations:
            kv = kind_dict[kname]
            dsym = _fortran_sym(typename + suffix)
            rmvar = resolved_members.get(mname, mvar)
            rctype = _get_member_ctype(rmvar) or ctype
            if rctype == 'float':
                parse = '(float)PyFloat_AsDouble(value)'
            elif rctype == 'double':
                parse = 'PyFloat_AsDouble(value)'
            elif rctype == 'int':
                parse = '(int)PyLong_AsLong(value)'
            elif rctype == 'long long':
                parse = 'PyLong_AsLongLong(value)'
            else:
                parse = f'({rctype})PyFloat_AsDouble(value)'

            cond = 'if' if not setter_branches else 'else if'
            setter_branches.append(
                f'    {cond} (self->kind_value == {kv}) {{\n'
                f'        f2py_set_{dsym}_{mname}(ptr, {parse});\n'
                f'    }}')

        setter_dispatch = '\n'.join(setter_branches)
        funcs.append(f"""\
static int
{setter_name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError,
                        "Cannot delete {mname} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return -1;
    }}
    const char *cap = PyCapsule_GetName(self->capsule);
    void *ptr = PyCapsule_GetPointer(self->capsule, cap);
    if (ptr == NULL) return -1;
{setter_dispatch}
    if (PyErr_Occurred()) return -1;
    return 0;
}}
""")
        getset_entries.append(
            f'    {{"{mname}", {getter_name}, {setter_name}, '
            f'"{mname} member", NULL}},')

    getset_array = (
        f'static PyGetSetDef Py{typename}_getset[] = {{\n'
        + '\n'.join(getset_entries) + '\n'
        + '    {NULL}  /* sentinel */\n'
        + '};\n'
    )
    return '\n'.join(funcs) + '\n' + getset_array


def _gen_opaque_tp_repr(typename, members, specializations=None,
                        kind_info=None, len_info=None):
    """Generate tp_repr that calls Fortran getters for display."""
    if specializations and kind_info:
        return _gen_opaque_tp_repr_multi(
            typename, members, specializations, kind_info)
    sym = _fortran_sym(typename)
    capsule_name = f'f2py.{typename}'
    # Build LEN call suffix for Fortran getter calls
    len_call_extra = ''
    if len_info:
        len_param_names = {li['name'] for li in len_info}
        len_call_extra = ', ' + ', '.join(
            f'self->len_{li["name"]}' for li in len_info)
    else:
        len_param_names = set()
    fmt_parts = []
    val_args = []
    for mname, mvar in members.items():
        if _is_type_array_member(mvar):
            inner = mvar.get('typename', '').lower()
            dims = _get_array_dims(mvar)
            if dims is None:
                fmt_parts.append(f'{mname}=<array>')
                continue
            total = 1
            for d in dims:
                total *= d
            fmt_parts.append(f'{mname}=[{total}x{inner}]')
            continue
        if _is_type_member(mvar):
            inner = mvar.get('typename', '').lower()
            fmt_parts.append(f'{mname}=<{inner}>')
            continue
        if _is_char_member(mvar):
            char_len = _get_char_len(mvar)
            fmt_parts.append(f'{mname}=<char({char_len})>')
            continue
        if _is_allocatable_member(mvar):
            fmt_parts.append(f'{mname}=<allocatable>')
            continue
        if _is_pointer_member(mvar):
            fmt_parts.append(f'{mname}=<pointer>')
            continue
        if _is_deferred_char_member(mvar):
            fmt_parts.append(f'{mname}=<deferred-char>')
            continue
        # LEN-sized arrays show as placeholder
        if (len_param_names
                and _is_len_sized_array(mvar, len_param_names)):
            fmt_parts.append(f'{mname}=<array>')
            continue
        ctype = _get_member_ctype(mvar)
        dims = _get_array_dims(mvar)
        if dims:
            dim_str = 'x'.join(str(d) for d in dims)
            fmt_parts.append(f'{mname}=<array({dim_str})>')
            continue
        if ctype in ('npy_cfloat', 'npy_cdouble'):
            creal_fn = 'npy_crealf' if ctype == 'npy_cfloat' else 'npy_creal'
            cimag_fn = 'npy_cimagf' if ctype == 'npy_cfloat' else 'npy_cimag'
            fmt_parts.append(f'{mname}=(%g+%gj)')
            val_args.append(
                f'(double){creal_fn}(f2py_get_{sym}_{mname}'
                f'(ptr{len_call_extra}))')
            val_args.append(
                f'(double){cimag_fn}(f2py_get_{sym}_{mname}'
                f'(ptr{len_call_extra}))')
        elif ctype in ('float', 'double'):
            fmt_parts.append(f'{mname}=%g')
            val_args.append(
                f'(double)f2py_get_{sym}_{mname}(ptr{len_call_extra})')
        elif ctype in ('int',):
            fmt_parts.append(f'{mname}=%d')
            val_args.append(
                f'f2py_get_{sym}_{mname}(ptr{len_call_extra})')
        elif ctype in ('long',):
            fmt_parts.append(f'{mname}=%ld')
            val_args.append(
                f'f2py_get_{sym}_{mname}(ptr{len_call_extra})')
        elif ctype in ('long long',):
            fmt_parts.append(f'{mname}=%lld')
            val_args.append(
                f'f2py_get_{sym}_{mname}(ptr{len_call_extra})')
        else:
            continue

    fmt_str = ', '.join(fmt_parts)
    val_str = ', '.join(val_args)
    if val_str:
        val_str = ',\n        ' + val_str

    return f"""\
static PyObject *
Py{typename}_tp_repr(PyObject *selfobj)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        return PyUnicode_FromString("{typename}(<uninitialized>)");
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) {{
        PyErr_Clear();
        return PyUnicode_FromString("{typename}(<invalid>)");
    }}
{_gen_repr_return(typename, fmt_str, val_str)}
}}
"""


def _gen_opaque_tp_repr_multi(typename, members, specializations, kind_info):
    """Generate multi-kind dispatch repr for PDTs.

    Uses Python-level attribute access via getset (which already
    dispatches per kind) to avoid duplicating dispatch logic.
    """
    ki = kind_info[0]
    kname = ki['name']

    member_names = []
    for mname, mvar in members.items():
        if (_is_type_array_member(mvar) or _is_type_member(mvar)
                or _is_char_member(mvar) or _is_allocatable_member(mvar)
                or _is_pointer_member(mvar)
                or _is_deferred_char_member(mvar)
                or _is_array_member(mvar) or _is_complex_member(mvar)):
            continue
        ctype = _get_member_ctype(mvar)
        if ctype is None or _C_TO_PYFORMAT.get(ctype) is None:
            continue
        member_names.append(mname)

    get_lines = []
    fmt_parts = [f'{kname}=%d']
    val_parts = ['self->kind_value']
    for mname in member_names:
        fmt_parts.append(f'{mname}=%g')
        get_lines.append(
            f'    PyObject *py_{mname} = '
            f'PyObject_GetAttrString(selfobj, "{mname}");')
        get_lines.append(
            f'    double v_{mname} = py_{mname} ? '
            f'PyFloat_AsDouble(py_{mname}) : 0.0;')
        get_lines.append(f'    Py_XDECREF(py_{mname});')
        val_parts.append(f'v_{mname}')

    get_str = '\n'.join(get_lines)
    fmt_str = ', '.join(fmt_parts)
    val_str = ', '.join(val_parts)

    return f"""\
static PyObject *
Py{typename}_tp_repr(PyObject *selfobj)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        return PyUnicode_FromString("{typename}(<uninitialized>)");
    }}
{get_str}
    PyErr_Clear();
{_gen_repr_return(typename, fmt_str, ', ' + val_str)}
}}
"""
