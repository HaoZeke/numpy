"""
C code generation for bind(c) and opaque derived type wrappers.

Copyright 2024 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

from .auxfuncs import (
    hasbody,
    isfunction,
    isroutine,
)

from ._dt_helpers import (
    _C_TO_NPY_ENUM,
    _C_TO_PYFORMAT,
    _C_TO_PYOBJ,
    _FORTRAN_ARITH_OPS,
    _FORTRAN_CMP_OPS,
    _PYOBJ_TO_C,
    _fortran_sym,
    _get_array_dims,
    _get_c_return_type,
    _get_char_len,
    _get_member_ctype,
    _get_member_isoc_type,
    _get_alloc_ndim,
    _is_allocatable_member,
    _is_array_member,
    _is_char_member,
    _is_complex_member,
    _is_type_array_member,
    _is_type_member,
)


def _gen_bindc_struct(typename, members):
    """Generate C typedef struct for a bind(c) type."""
    lines = [f'typedef struct {{']
    for mname, mvar in members.items():
        if _is_type_array_member(mvar):
            inner = mvar.get('typename', '').lower()
            dims = _get_array_dims(mvar)
            dim_str = ''.join(f'[{d}]' for d in dims)
            lines.append(f'    f2py_{inner}_t {mname}{dim_str};')
        elif _is_type_member(mvar):
            inner = mvar.get('typename', '').lower()
            lines.append(f'    f2py_{inner}_t {mname};')
        else:
            ctype = _get_member_ctype(mvar)
            dims = _get_array_dims(mvar)
            if dims:
                dim_str = ''.join(f'[{d}]' for d in dims)
                lines.append(f'    {ctype} {mname}{dim_str};')
            else:
                lines.append(f'    {ctype} {mname};')
    lines.append(f'}} f2py_{typename}_t;\n')
    return '\n'.join(lines)


def _gen_pytype_struct(typename):
    """Generate the Python type object struct."""
    return f"""\
typedef struct {{
    PyObject_HEAD
    PyObject *capsule;  /* PyCapsule wrapping Fortran {typename} data */
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


def _gen_tp_new(typename):
    """Generate tp_new for the Python type."""
    return f"""\
static PyObject *
Py{typename}_tp_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{{
    Py{typename}Object *self;
    self = (Py{typename}Object *)type->tp_alloc(type, 0);
    if (self != NULL) {{
        self->capsule = NULL;
    }}
    return (PyObject *)self;
}}
"""


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
        if ctype in ('float _Complex', 'double _Complex'):
            cast = '(float)' if ctype == 'float _Complex' else ''
            extract_lines.append(
                f'    data->{mname} = {cast}{mname}.real'
                f' + {cast}{mname}.imag * _Complex_I;')
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
        if ctype in ('float _Complex', 'double _Complex'):
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


def _gen_getset(typename, members):
    """Generate tp_getset array with property getters/setters."""
    capsule_name = f'f2py.{typename}'
    funcs = []
    getset_entries = []

    for mname, mvar in members.items():
        if _is_type_array_member(mvar):
            # Array of derived types -- getter returns list, setter
            # accepts list
            inner = mvar.get('typename', '').lower()
            inner_capsule = f'f2py.{inner}'
            dims = _get_array_dims(mvar)
            total = 1
            for d in dims:
                total *= d

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
    PyObject *list = PyList_New({total});
    if (list == NULL) return NULL;
    for (Py_ssize_t i = 0; i < {total}; i++) {{
        PyObject *obj = Py{inner}_tp_new(&Py{inner}_Type, NULL, NULL);
        if (obj == NULL) {{
            Py_DECREF(list);
            return NULL;
        }}
        f2py_{inner}_t *copy = (f2py_{inner}_t *)PyMem_Malloc(
            sizeof(f2py_{inner}_t));
        if (copy == NULL) {{
            Py_DECREF(obj);
            Py_DECREF(list);
            return PyErr_NoMemory();
        }}
        memcpy(copy, &data->{mname}[i], sizeof(f2py_{inner}_t));
        ((Py{inner}Object *)obj)->capsule = PyCapsule_New(
            copy, "{inner_capsule}", f2py_{inner}_capsule_destructor);
        if (((Py{inner}Object *)obj)->capsule == NULL) {{
            PyMem_Free(copy);
            Py_DECREF(obj);
            Py_DECREF(list);
            return NULL;
        }}
        PyList_SET_ITEM(list, i, obj);
    }}
    return list;
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
    if (!PyList_Check(value) || PyList_Size(value) != {total}) {{
        PyErr_SetString(PyExc_ValueError,
                        "{mname} must be a list of {total} {inner} objects");
        return -1;
    }}
    f2py_{typename}_t *data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "{capsule_name}");
    if (data == NULL) return -1;
    for (Py_ssize_t i = 0; i < {total}; i++) {{
        PyObject *item = PyList_GET_ITEM(value, i);
        if (!Py_IS_TYPE(item, &Py{inner}_Type)) {{
            PyErr_Format(PyExc_TypeError,
                         "{mname}[%zd] must be a {inner} instance", i);
            return -1;
        }}
        Py{inner}Object *inner_obj = (Py{inner}Object *)item;
        if (inner_obj->capsule == NULL) {{
            PyErr_Format(PyExc_RuntimeError,
                         "{mname}[%zd] not initialized", i);
            return -1;
        }}
        f2py_{inner}_t *inner_data = (f2py_{inner}_t *)
            PyCapsule_GetPointer(inner_obj->capsule, "{inner_capsule}");
        if (inner_data == NULL) return -1;
        memcpy(&data->{mname}[i], inner_data, sizeof(f2py_{inner}_t));
    }}
    return 0;
}}
""")

            getset_entries.append(
                f'    {{"{mname}", {getter_name}, {setter_name}, '
                f'"{mname} member", NULL}},'
            )
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
    char buf[{char_len + 1}];
    memset(buf, 0, {char_len + 1});
    f2py_get_{sym}_{mname}(ptr, buf, {char_len});
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
    f2py_set_{sym}_{mname}(ptr, str, (int)slen);
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
            is_single = (ctype == 'float _Complex')
            creal_fn = 'crealf' if is_single else 'creal'
            cimag_fn = 'cimagf' if is_single else 'cimag'
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
    data->{mname} = {cast}c.real + {cast}c.imag * _Complex_I;
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
    PyObject *arr = PyArray_SimpleNewFromData(
        {ndim}, dims, {npy_enum}, (void *)data->{mname});
    if (arr == NULL) return NULL;
    /* Set self as base so struct stays alive while array is in use */
    if (PyArray_SetBaseObject((PyArrayObject *)arr,
                              (PyObject *)self) < 0) {{
        Py_DECREF(arr);
        return NULL;
    }}
    Py_INCREF(self);
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
    PyObject *arr = PyArray_FROM_OTF(value, {npy_enum},
                                      NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) return -1;
    if (PyArray_SIZE((PyArrayObject *)arr) != {total}) {{
        PyErr_SetString(PyExc_ValueError,
                        "{mname} must have {total} elements");
        Py_DECREF(arr);
        return -1;
    }}
    memcpy(data->{mname}, PyArray_DATA((PyArrayObject *)arr),
           {total} * sizeof({ctype}));
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
        if ctype in ('float _Complex', 'double _Complex'):
            creal_fn = 'crealf' if ctype == 'float _Complex' else 'creal'
            cimag_fn = 'cimagf' if ctype == 'float _Complex' else 'cimag'
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
    char buf[256];
    snprintf(buf, sizeof(buf), "{typename}({fmt_str})"{val_str});
    return PyUnicode_FromString(buf);
}}
"""


def _scan_operator_interfaces(module_block, typename, type_map):
    """Find operator interfaces that involve a given derived type.

    Scans module body for interface blocks with names like
    'operator(+)' or 'operator(==)', and finds implementing procedures
    whose arguments involve the specified typename.

    Returns dict mapping operator symbol to list of
    (proc_name, proc_block, result_kind) tuples, where result_kind
    is 'type' (returns derived type) or 'logical' (returns logical).
    """
    if not hasbody(module_block):
        return {}

    # Collect all routines in module for lookup
    routines = {}
    for b in module_block['body']:
        if isroutine(b):
            routines[b['name'].lower()] = b

    ops = {}
    for b in module_block['body']:
        if b.get('block') != 'interface':
            continue
        iname = b.get('name', '')
        # Match 'operator(+)' or 'operator(==)' etc.
        if not iname.startswith(('operator(', 'assignment(')):
            continue
        op_sym = iname[iname.index('(') + 1:iname.rindex(')')]

        # Check if this is an arithmetic or comparison op we support
        if (op_sym not in _FORTRAN_ARITH_OPS
                and op_sym not in _FORTRAN_CMP_OPS):
            continue

        procs = b.get('implementedby', [])
        for pname in procs:
            pname_lower = pname.lower()
            if pname_lower not in routines:
                continue
            proc = routines[pname_lower]

            # Check if this procedure involves our type
            args = proc.get('args', [])
            involves_type = False
            for argname in args:
                var = proc['vars'].get(argname, {})
                if (var.get('typespec') == 'type'
                        and var.get('typename', '').lower()
                        == typename.lower()):
                    involves_type = True
                    break
            if not involves_type:
                continue

            # Determine result kind
            if isfunction(proc):
                result_var = proc['vars'].get(
                    proc.get('result', proc['name']), {})
                if result_var.get('typespec') == 'type':
                    result_kind = 'type'
                elif result_var.get('typespec') == 'logical':
                    result_kind = 'logical'
                else:
                    continue
            else:
                continue  # subroutines can't be operators

            if op_sym not in ops:
                ops[op_sym] = []
            ops[op_sym].append((pname_lower, proc, result_kind))

    return ops


def _gen_operator_fortran_wrappers(typename, ops, type_map):
    """Generate Fortran bind(c) wrapper functions for operator impls.

    Returns list of Fortran source lines.
    """
    lines = []
    for op_sym, proc_list in ops.items():
        for pname, proc, result_kind in proc_list:
            args = proc.get('args', [])
            wrapper_name = f'f2py_op_{typename}_{pname}'

            if result_kind == 'type':
                # function(a, b) result(c) where c is derived type
                result_var = proc['vars'].get(
                    proc.get('result', proc['name']), {})
                result_typename = result_var.get(
                    'typename', '').lower()

                lines.append(
                    f'  function {wrapper_name}('
                    + ', '.join(f'{a}_ptr' for a in args)
                    + ') result(cptr) bind(c)')
                lines.append(f'    type(c_ptr) :: cptr')
                for argname in args:
                    var = proc['vars'].get(argname, {})
                    if var.get('typespec') == 'type':
                        lines.append(
                            f'    type(c_ptr), value :: {argname}_ptr')
                    else:
                        isoc = _get_member_isoc_type(var)
                        if isoc:
                            lines.append(
                                f'    {isoc}, value :: {argname}_ptr')
                # Local variables
                for argname in args:
                    var = proc['vars'].get(argname, {})
                    if var.get('typespec') == 'type':
                        tname = var.get('typename', '').lower()
                        lines.append(
                            f'    type({tname}), pointer :: {argname}')
                lines.append(
                    f'    type({result_typename}), pointer :: res')
                # c_f_pointer for type args
                for argname in args:
                    var = proc['vars'].get(argname, {})
                    if var.get('typespec') == 'type':
                        lines.append(
                            f'    call c_f_pointer({argname}_ptr, '
                            f'{argname})')
                # Call original function
                call_args = ', '.join(args)
                lines.append(f'    allocate(res)')
                lines.append(f'    res = {pname}({call_args})')
                lines.append(f'    cptr = c_loc(res)')
                lines.append(f'  end function {wrapper_name}')
                lines.append('')

            elif result_kind == 'logical':
                # function(a, b) result(eq) where eq is logical
                lines.append(
                    f'  function {wrapper_name}('
                    + ', '.join(f'{a}_ptr' for a in args)
                    + ') result(res) bind(c)')
                lines.append(f'    integer(c_int) :: res')
                for argname in args:
                    var = proc['vars'].get(argname, {})
                    if var.get('typespec') == 'type':
                        lines.append(
                            f'    type(c_ptr), value :: {argname}_ptr')
                    else:
                        isoc = _get_member_isoc_type(var)
                        if isoc:
                            lines.append(
                                f'    {isoc}, value :: {argname}_ptr')
                for argname in args:
                    var = proc['vars'].get(argname, {})
                    if var.get('typespec') == 'type':
                        tname = var.get('typename', '').lower()
                        lines.append(
                            f'    type({tname}), pointer :: {argname}')
                for argname in args:
                    var = proc['vars'].get(argname, {})
                    if var.get('typespec') == 'type':
                        lines.append(
                            f'    call c_f_pointer({argname}_ptr, '
                            f'{argname})')
                call_args = ', '.join(args)
                lines.append(
                    f'    if ({pname}({call_args})) then')
                lines.append(f'      res = 1')
                lines.append(f'    else')
                lines.append(f'      res = 0')
                lines.append(f'    end if')
                lines.append(f'  end function {wrapper_name}')
                lines.append('')

    return lines


def _gen_operator_c_code(typename, ops, type_map):
    """Generate C code for Python operator slots.

    Returns (code_parts, has_number, has_richcompare) where code_parts
    is a list of C code strings, has_number indicates PyNumberMethods
    were generated, has_richcompare indicates tp_richcompare was generated.
    """
    sym = _fortran_sym(typename)
    capsule_name = f'f2py.{typename}'
    # Forward declare the PyTypeObject so operator functions can
    # reference it (the actual definition comes later)
    code_parts = [
        f'static PyTypeObject Py{typename}_Type;  '
        f'/* forward decl for operators */'
    ]
    nb_slots = {}
    cmp_ops = {}

    for op_sym, proc_list in ops.items():
        for pname, proc, result_kind in proc_list:
            wrapper_sym = f'f2py_op_{sym}_{pname}'
            args = proc.get('args', [])

            if result_kind == 'type' and op_sym in _FORTRAN_ARITH_OPS:
                result_var = proc['vars'].get(
                    proc.get('result', proc['name']), {})
                result_typename = result_var.get(
                    'typename', '').lower()
                result_capsule = f'f2py.{result_typename}'

                # Extern declaration
                extern_args = []
                for argname in args:
                    var = proc['vars'].get(argname, {})
                    if var.get('typespec') == 'type':
                        extern_args.append('void *')
                    else:
                        ctype = _get_member_ctype(var)
                        if ctype:
                            extern_args.append(ctype)
                code_parts.append(
                    f'extern void *{wrapper_sym}'
                    f'({", ".join(extern_args)});')

                # nb_* slot function
                slot_name = _FORTRAN_ARITH_OPS[op_sym]
                func_name = f'Py{typename}_{slot_name}'

                code_parts.append(f"""\
static PyObject *
{func_name}(PyObject *left, PyObject *right)
{{
    if (!Py_IS_TYPE(left, &Py{typename}_Type) ||
        !Py_IS_TYPE(right, &Py{typename}_Type)) {{
        Py_RETURN_NOTIMPLEMENTED;
    }}
    Py{typename}Object *a = (Py{typename}Object *)left;
    Py{typename}Object *b = (Py{typename}Object *)right;
    if (a->capsule == NULL || b->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError, "operand not initialized");
        return NULL;
    }}
    void *a_ptr = PyCapsule_GetPointer(a->capsule, "{capsule_name}");
    void *b_ptr = PyCapsule_GetPointer(b->capsule, "{capsule_name}");
    if (a_ptr == NULL || b_ptr == NULL) return NULL;
    void *res_ptr = {wrapper_sym}(a_ptr, b_ptr);
    if (res_ptr == NULL) {{
        PyErr_SetString(PyExc_RuntimeError, "operator returned NULL");
        return NULL;
    }}
    PyObject *res = Py{result_typename}_tp_new(
        &Py{result_typename}_Type, NULL, NULL);
    if (res == NULL) return NULL;
    ((Py{result_typename}Object *)res)->capsule = PyCapsule_New(
        res_ptr, "{result_capsule}",
        f2py_{result_typename}_capsule_destructor);
    if (((Py{result_typename}Object *)res)->capsule == NULL) {{
        f2py_destroy_{_fortran_sym(result_typename)}(res_ptr);
        Py_DECREF(res);
        return NULL;
    }}
    return res;
}}
""")
                nb_slots[slot_name] = func_name

            elif result_kind == 'logical' and op_sym in _FORTRAN_CMP_OPS:
                # Extern declaration
                extern_args = []
                for argname in args:
                    var = proc['vars'].get(argname, {})
                    if var.get('typespec') == 'type':
                        extern_args.append('void *')
                    else:
                        ctype = _get_member_ctype(var)
                        if ctype:
                            extern_args.append(ctype)
                code_parts.append(
                    f'extern int {wrapper_sym}'
                    f'({", ".join(extern_args)});')

                py_cmp = _FORTRAN_CMP_OPS[op_sym]
                cmp_ops[py_cmp] = (wrapper_sym, pname)

    # Generate PyNumberMethods if any arithmetic ops
    has_number = bool(nb_slots)
    if has_number:
        lines = [f'static PyNumberMethods Py{typename}_as_number = {{']
        for slot, func in nb_slots.items():
            lines.append(f'    .{slot} = {func},')
        lines.append('};')
        code_parts.append('\n'.join(lines))

    # Generate tp_richcompare if any comparison ops
    has_richcompare = bool(cmp_ops)
    if has_richcompare:
        cases = []
        for py_cmp, (wrapper, pname) in cmp_ops.items():
            cases.append(f"""\
    case {py_cmp}:
        result = {wrapper}(a_ptr, b_ptr);
        if (result) Py_RETURN_TRUE;
        Py_RETURN_FALSE;""")
        cases_str = '\n'.join(cases)

        code_parts.append(f"""\
static PyObject *
Py{typename}_tp_richcompare(PyObject *left, PyObject *right, int op)
{{
    if (!Py_IS_TYPE(left, &Py{typename}_Type) ||
        !Py_IS_TYPE(right, &Py{typename}_Type)) {{
        Py_RETURN_NOTIMPLEMENTED;
    }}
    Py{typename}Object *a = (Py{typename}Object *)left;
    Py{typename}Object *b = (Py{typename}Object *)right;
    if (a->capsule == NULL || b->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError, "operand not initialized");
        return NULL;
    }}
    void *a_ptr = PyCapsule_GetPointer(a->capsule, "{capsule_name}");
    void *b_ptr = PyCapsule_GetPointer(b->capsule, "{capsule_name}");
    if (a_ptr == NULL || b_ptr == NULL) return NULL;
    int result;
    switch (op) {{
{cases_str}
    default:
        Py_RETURN_NOTIMPLEMENTED;
    }}
}}
""")

    return code_parts, has_number, has_richcompare


def _gen_typeobject(typename, has_methods=False, parent_typename=None,
                    has_number=False, has_richcompare=False):
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
    .tp_repr = Py{typename}_tp_repr,
    .tp_getset = Py{typename}_getset,{methods_line}{base_line}{number_line}{richcmp_line}
}};
"""


def _gen_init_code(typename, modulename):
    """Generate code for module init to register the type."""
    # Use lowercase for the Python-accessible attribute name to match
    # Fortran's case-insensitive convention
    pyname = typename.lower()
    return [
        '\t{',
        f'\t\tif (PyType_Ready(&Py{typename}_Type) < 0) return NULL;',
        f'\t\tPy_INCREF(&Py{typename}_Type);',
        f'\t\tPyModule_AddObject(m, "{pyname}", '
        f'(PyObject *)&Py{typename}_Type);',
        '\t}',
    ]


def _gen_opaque_extern_decls(typename, members):
    """Generate extern declarations for the Fortran bind(c) wrappers."""
    sym = _fortran_sym(typename)
    lines = []
    lines.append(f'/* Extern declarations for Fortran wrappers */')

    # Constructor: returns void* (c_ptr) -- scalar args only
    args = []
    for mname, mvar in members.items():
        if (_is_array_member(mvar) or _is_type_member(mvar)
                or _is_type_array_member(mvar) or _is_char_member(mvar)
                or _is_allocatable_member(mvar)):
            continue
        ctype = _get_member_ctype(mvar)
        if ctype and _C_TO_PYFORMAT.get(ctype):
            args.append(f'{ctype} {mname}')
    args_str = ', '.join(args) if args else 'void'
    lines.append(f'extern void *f2py_create_{sym}({args_str});')

    # Destructor: takes void*
    lines.append(f'extern void f2py_destroy_{sym}(void *);')

    # Getters and setters
    for mname, mvar in members.items():
        if _is_type_array_member(mvar):
            # Array of derived types: indexed get/set
            lines.append(
                f'extern void *f2py_get_{sym}_{mname}'
                f'(void *, int);')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *, int, void *);')
            continue

        if _is_type_member(mvar):
            # Nested type: getter returns void* (new allocation),
            # setter takes void* (copies data)
            lines.append(
                f'extern void *f2py_get_{sym}_{mname}(void *);')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *, void *);')
            continue

        if _is_char_member(mvar):
            char_len = _get_char_len(mvar)
            lines.append(
                f'extern void f2py_get_{sym}_{mname}'
                f'(void *, char *, int);')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *, const char *, int);')
            continue

        if _is_allocatable_member(mvar):
            ndim = _get_alloc_ndim(mvar)
            # Allocatable array: allocated, ndim, shape, data, set
            lines.append(
                f'extern unsigned char f2py_get_{sym}_{mname}'
                f'_allocated(void *);')
            lines.append(
                f'extern int f2py_get_{sym}_{mname}'
                f'_ndim(void *);')
            lines.append(
                f'extern void f2py_get_{sym}_{mname}'
                f'_shape(void *, int *);')
            lines.append(
                f'extern void *f2py_get_{sym}_{mname}'
                f'_data(void *);')
            # setter takes ndim dimension args then data pointer
            set_args = ', '.join(['void *'] + ['int'] * ndim + ['void *'])
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'({set_args});')
            continue

        ctype = _get_member_ctype(mvar)
        dims = _get_array_dims(mvar)

        if dims:
            # Array member: getter returns c_ptr to array data,
            # setter accepts pointer + copies
            if _C_TO_NPY_ENUM.get(ctype) is None:
                continue
            lines.append(
                f'extern void *f2py_get_{sym}_{mname}(void *);')
            total = 1
            for d in dims:
                total *= d
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *, {ctype} *);')
        else:
            if ctype is None or _C_TO_PYFORMAT.get(ctype) is None:
                continue
            rtype = _get_c_return_type(ctype)
            lines.append(
                f'extern {rtype} f2py_get_{sym}_{mname}(void *);')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *, {ctype});')

    lines.append('')
    return '\n'.join(lines)


def _gen_opaque_capsule_destructor(typename):
    """Generate PyCapsule destructor that calls Fortran deallocator."""
    sym = _fortran_sym(typename)
    capsule_name = f'f2py.{typename}'
    return f"""\
static void
f2py_{typename}_capsule_destructor(PyObject *capsule)
{{
    void *ptr = PyCapsule_GetPointer(capsule, "{capsule_name}");
    if (ptr != NULL) {{
        f2py_destroy_{sym}(ptr);
    }}
}}
"""


def _gen_opaque_tp_init(typename, members):
    """Generate tp_init that calls the Fortran constructor."""
    sym = _fortran_sym(typename)
    capsule_name = f'f2py.{typename}'
    kwlist = []
    fmt_parts = []
    for mname, mvar in members.items():
        if (_is_array_member(mvar) or _is_type_member(mvar)
                or _is_type_array_member(mvar) or _is_char_member(mvar)
                or _is_allocatable_member(mvar)):
            continue  # arrays, nested types, chars, allocs set via props
        ctype = _get_member_ctype(mvar)
        fmt = _C_TO_PYFORMAT.get(ctype)
        if fmt is None:
            continue
        kwlist.append(mname)
        fmt_parts.append(fmt)

    kwlist_str = ', '.join(f'"{k}"' for k in kwlist)
    fmt_str = ''.join(fmt_parts)

    decl_lines = []
    parse_args = []
    call_args = []
    for mname, mvar in members.items():
        if (_is_array_member(mvar) or _is_type_member(mvar)
                or _is_type_array_member(mvar) or _is_char_member(mvar)
                or _is_allocatable_member(mvar)):
            continue
        ctype = _get_member_ctype(mvar)
        if _C_TO_PYFORMAT.get(ctype) is None:
            continue
        if ctype in ('float _Complex', 'double _Complex'):
            decl_lines.append(f'    Py_complex {mname} = {{0, 0}};')
            parse_args.append(f'&{mname}')
            cast = '(float)' if ctype == 'float _Complex' else ''
            call_args.append(
                f'{cast}{mname}.real + {cast}{mname}.imag * _Complex_I')
        else:
            decl_lines.append(f'    {ctype} {mname} = 0;')
            parse_args.append(f'&{mname}')
            call_args.append(mname)

    decl_str = '\n'.join(decl_lines)
    parse_args_str = ', '.join(parse_args)
    call_args_str = ', '.join(call_args)

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
    /* No scalar members; all members are arrays set via properties */"""

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
        f2py_destroy_{sym}(ptr);
        return -1;
    }}

    return 0;
}}
"""


def _gen_opaque_getset(typename, members):
    """Generate getters/setters that call Fortran accessor wrappers."""
    sym = _fortran_sym(typename)
    capsule_name = f'f2py.{typename}'
    funcs = []
    getset_entries = []

    for mname, mvar in members.items():
        if _is_type_array_member(mvar):
            # Array of derived types via opaque path
            inner = mvar.get('typename', '').lower()
            inner_sym = _fortran_sym(inner)
            inner_capsule = f'f2py.{inner}'
            dims = _get_array_dims(mvar)
            total = 1
            for d in dims:
                total *= d

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
    PyObject *list = PyList_New({total});
    if (list == NULL) return NULL;
    for (Py_ssize_t i = 0; i < {total}; i++) {{
        void *inner_ptr = f2py_get_{sym}_{mname}(ptr, (int)(i + 1));
        if (inner_ptr == NULL) {{
            Py_DECREF(list);
            PyErr_SetString(PyExc_RuntimeError,
                            "Fortran returned NULL for {mname} element");
            return NULL;
        }}
        PyObject *obj = Py{inner}_tp_new(&Py{inner}_Type, NULL, NULL);
        if (obj == NULL) {{
            f2py_destroy_{inner_sym}(inner_ptr);
            Py_DECREF(list);
            return NULL;
        }}
        ((Py{inner}Object *)obj)->capsule = PyCapsule_New(
            inner_ptr, "{inner_capsule}",
            f2py_{inner}_capsule_destructor);
        if (((Py{inner}Object *)obj)->capsule == NULL) {{
            f2py_destroy_{inner_sym}(inner_ptr);
            Py_DECREF(obj);
            Py_DECREF(list);
            return NULL;
        }}
        PyList_SET_ITEM(list, i, obj);
    }}
    return list;
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
    if (!PyList_Check(value) || PyList_Size(value) != {total}) {{
        PyErr_SetString(PyExc_ValueError,
                        "{mname} must be a list of {total} {inner} objects");
        return -1;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return -1;
    for (Py_ssize_t i = 0; i < {total}; i++) {{
        PyObject *item = PyList_GET_ITEM(value, i);
        if (!Py_IS_TYPE(item, &Py{inner}_Type)) {{
            PyErr_Format(PyExc_TypeError,
                         "{mname}[%zd] must be a {inner} instance", i);
            return -1;
        }}
        Py{inner}Object *inner_obj = (Py{inner}Object *)item;
        if (inner_obj->capsule == NULL) {{
            PyErr_Format(PyExc_RuntimeError,
                         "{mname}[%zd] not initialized", i);
            return -1;
        }}
        void *inner_ptr = PyCapsule_GetPointer(
            inner_obj->capsule, "{inner_capsule}");
        if (inner_ptr == NULL) return -1;
        f2py_set_{sym}_{mname}(ptr, (int)(i + 1), inner_ptr);
    }}
    return 0;
}}
""")

            getset_entries.append(
                f'    {{"{mname}", {getter_name}, {setter_name}, '
                f'"{mname} member", NULL}},'
            )
            continue

        if _is_type_member(mvar):
            # Nested derived type member via opaque path
            inner = mvar.get('typename', '').lower()
            inner_sym = _fortran_sym(inner)
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
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    /* Get an opaque copy of the nested type from Fortran */
    void *inner_ptr = f2py_get_{sym}_{mname}(ptr);
    if (inner_ptr == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "Fortran returned NULL for {mname}");
        return NULL;
    }}
    PyObject *obj = Py{inner}_tp_new(&Py{inner}_Type, NULL, NULL);
    if (obj == NULL) return NULL;
    ((Py{inner}Object *)obj)->capsule = PyCapsule_New(
        inner_ptr, "{inner_capsule}",
        f2py_{inner}_capsule_destructor);
    if (((Py{inner}Object *)obj)->capsule == NULL) {{
        f2py_destroy_{inner_sym}(inner_ptr);
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
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return -1;
    Py{inner}Object *inner_obj = (Py{inner}Object *)value;
    if (inner_obj->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{inner} object not initialized");
        return -1;
    }}
    void *inner_ptr = PyCapsule_GetPointer(
        inner_obj->capsule, "{inner_capsule}");
    if (inner_ptr == NULL) return -1;
    f2py_set_{sym}_{mname}(ptr, inner_ptr);
    return 0;
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
    char buf[{char_len + 1}];
    memset(buf, 0, {char_len + 1});
    f2py_get_{sym}_{mname}(ptr, buf, {char_len});
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
    f2py_set_{sym}_{mname}(ptr, str, (int)slen);
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
            is_single = (ctype == 'float _Complex')
            creal_fn = 'crealf' if is_single else 'creal'
            cimag_fn = 'cimagf' if is_single else 'cimag'
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
    {ctype} val = f2py_get_{sym}_{mname}(ptr);
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
    {ctype} cval = {cast}c.real + {cast}c.imag * _Complex_I;
    f2py_set_{sym}_{mname}(ptr, cval);
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
    unsigned char is_alloc = f2py_get_{sym}_{mname}_allocated(ptr);
    if (!is_alloc) Py_RETURN_NONE;
    /* Query rank and shape from Fortran (F2018 16.9.182 size intrinsic) */
    int rank = f2py_get_{sym}_{mname}_ndim(ptr);
    int fortran_shape[{ndim}];
    f2py_get_{sym}_{mname}_shape(ptr, fortran_shape);
    npy_intp numpy_dims[{ndim}];
    npy_intp total_elements = 1;
    for (int dim_idx = 0; dim_idx < rank; dim_idx++) {{
        numpy_dims[dim_idx] = (npy_intp)fortran_shape[dim_idx];
        total_elements *= numpy_dims[dim_idx];
    }}
    void *data = f2py_get_{sym}_{mname}_data(ptr);
    /* Allocate NumPy array matching Fortran memory layout.
       Fortran allocatables are column-major (F2018 8.5.8.1),
       so multi-dimensional arrays use Fortran order. 1D arrays
       have identical layout in both C and Fortran order. */
    PyObject *arr;
    if (rank > 1) {{
        arr = PyArray_EMPTY(rank, numpy_dims, {npy_enum},
                            1 /* fortran_order */);
    }} else {{
        arr = PyArray_SimpleNew(rank, numpy_dims, {npy_enum});
    }}
    if (arr == NULL) return NULL;
    memcpy(PyArray_DATA((PyArrayObject *)arr), data,
           total_elements * sizeof({alloc_ctype}));
    return arr;
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
        f2py_set_{sym}_{mname}(ptr, {dealloc_args});
        return 0;
    }}
    PyObject *arr = PyArray_FROM_OTF(value, {npy_enum},
                                      NPY_ARRAY_F_CONTIGUOUS);
    if (arr == NULL) return -1;
    if (PyArray_NDIM((PyArrayObject *)arr) != {ndim}) {{
        PyErr_Format(PyExc_ValueError,
                     "{mname} requires a {ndim}D array, got %dD",
                     PyArray_NDIM((PyArrayObject *)arr));
        Py_DECREF(arr);
        return -1;
    }}
    f2py_set_{sym}_{mname}(ptr, {set_dim_args},
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
    void *arrptr = f2py_get_{sym}_{mname}(ptr);
    if (arrptr == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "Fortran returned NULL for {mname}");
        return NULL;
    }}
    npy_intp dims[{ndim}] = {{{dims_init}}};
    /* Copy data out of Fortran memory */
    PyObject *arr = PyArray_SimpleNew({ndim}, dims, {npy_enum});
    if (arr == NULL) return NULL;
    memcpy(PyArray_DATA((PyArrayObject *)arr), arrptr,
           {total} * sizeof({ctype}));
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
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return -1;
    PyObject *arr = PyArray_FROM_OTF(value, {npy_enum},
                                      NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) return -1;
    if (PyArray_SIZE((PyArrayObject *)arr) != {total}) {{
        PyErr_SetString(PyExc_ValueError,
                        "{mname} must have {total} elements");
        Py_DECREF(arr);
        return -1;
    }}
    f2py_set_{sym}_{mname}(ptr,
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
    {rtype} val = f2py_get_{sym}_{mname}(ptr);
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
    f2py_set_{sym}_{mname}(ptr, cval);
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


def _gen_opaque_tp_repr(typename, members):
    """Generate tp_repr that calls Fortran getters for display."""
    sym = _fortran_sym(typename)
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
        if _is_char_member(mvar):
            char_len = _get_char_len(mvar)
            fmt_parts.append(f'{mname}=<char({char_len})>')
            continue
        if _is_allocatable_member(mvar):
            fmt_parts.append(f'{mname}=<allocatable>')
            continue
        ctype = _get_member_ctype(mvar)
        dims = _get_array_dims(mvar)
        if dims:
            dim_str = 'x'.join(str(d) for d in dims)
            fmt_parts.append(f'{mname}=<array({dim_str})>')
            continue
        if ctype in ('float _Complex', 'double _Complex'):
            creal_fn = 'crealf' if ctype == 'float _Complex' else 'creal'
            cimag_fn = 'cimagf' if ctype == 'float _Complex' else 'cimag'
            fmt_parts.append(f'{mname}=(%g+%gj)')
            val_args.append(
                f'(double){creal_fn}(f2py_get_{sym}_{mname}(ptr))')
            val_args.append(
                f'(double){cimag_fn}(f2py_get_{sym}_{mname}(ptr))')
        elif ctype in ('float', 'double'):
            fmt_parts.append(f'{mname}=%g')
            val_args.append(
                f'(double)f2py_get_{sym}_{mname}(ptr)')
        elif ctype in ('int',):
            fmt_parts.append(f'{mname}=%d')
            val_args.append(f'f2py_get_{sym}_{mname}(ptr)')
        elif ctype in ('long',):
            fmt_parts.append(f'{mname}=%ld')
            val_args.append(f'f2py_get_{sym}_{mname}(ptr)')
        elif ctype in ('long long',):
            fmt_parts.append(f'{mname}=%lld')
            val_args.append(f'f2py_get_{sym}_{mname}(ptr)')
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
    char buf[256];
    snprintf(buf, sizeof(buf), "{typename}({fmt_str})"{val_str});
    return PyUnicode_FromString(buf);
}}
"""
