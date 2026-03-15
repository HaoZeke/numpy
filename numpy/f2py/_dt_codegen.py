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
    _get_pointer_ndim,
    _is_allocatable_member,
    _is_array_member,
    _is_deferred_char_member,
    _is_len_sized_array,
    _is_pointer_member,
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


def _gen_pytype_struct(typename, has_kind=False, len_info=None):
    """Generate the Python type object struct."""
    kind_field = ''
    if has_kind:
        kind_field = (
            f'\n    int kind_value;'
            f'  /* KIND parameter value for dispatch */')
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
    PyObject *capsule;  /* PyCapsule wrapping Fortran {typename} data */{kind_field}{len_fields}
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
                _gen_opaque_extern_decls_one(sym, resolved_members))
        return '\n'.join(all_lines)

    sym = _fortran_sym(typename)
    lines = []
    lines.append(f'/* Extern declarations for Fortran wrappers */')

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
            lines.append(
                f'extern void *f2py_get_{sym}_{mname}'
                f'(void *{len_extra_prefix}, int);')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *{len_extra_prefix}, int, void *);')
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


def _gen_opaque_extern_decls_one(sym, members):
    """Generate extern declarations for one specialization."""
    lines = []
    args = []
    for mname, mvar in members.items():
        if (_is_array_member(mvar) or _is_type_member(mvar)
                or _is_type_array_member(mvar) or _is_char_member(mvar)
                or _is_allocatable_member(mvar)
                or _is_pointer_member(mvar)
                or _is_deferred_char_member(mvar)):
            continue
        ctype = _get_member_ctype(mvar)
        if ctype and _C_TO_PYFORMAT.get(ctype):
            args.append(f'{ctype} {mname}')
    args_str = ', '.join(args) if args else 'void'
    lines.append(f'extern void *f2py_create_{sym}({args_str});')
    lines.append(f'extern void f2py_destroy_{sym}(void *);')
    for mname, mvar in members.items():
        ctype = _get_member_ctype(mvar)
        if _is_type_array_member(mvar) or _is_type_member(mvar):
            continue
        if _is_char_member(mvar) or _is_allocatable_member(mvar):
            continue
        if _is_pointer_member(mvar) or _is_deferred_char_member(mvar):
            continue
        dims = _get_array_dims(mvar)
        if dims:
            continue
        if ctype is None or _C_TO_PYFORMAT.get(ctype) is None:
            continue
        rtype = _get_c_return_type(ctype)
        lines.append(f'extern {rtype} f2py_get_{sym}_{mname}(void *);')
        lines.append(
            f'extern void f2py_set_{sym}_{mname}(void *, {ctype});')
    return '\n'.join(lines)


def _gen_opaque_capsule_destructor(typename, specializations=None,
                                    len_info=None):
    """Generate PyCapsule destructor that calls Fortran deallocator."""
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
    void *ptr = PyCapsule_GetPointer(capsule, "{capsule_name}");
    if (ptr != NULL) {{
        f2py_destroy_{sym}(ptr);
    }}
}}
"""


def _gen_opaque_tp_init(typename, members, specializations=None,
                        kind_info=None, len_info=None):
    """Generate tp_init that calls the Fortran constructor."""
    if specializations and kind_info:
        return _gen_opaque_tp_init_multi(
            typename, members, specializations, kind_info)
    sym = _fortran_sym(typename)
    capsule_name = f'f2py.{typename}'

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

    # Store LEN values and set capsule context
    if len_info:
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
        ctx_lines += f"""
    PyCapsule_SetContext(self->capsule, len_ctx);"""
        post_capsule = ctx_lines
    else:
        post_capsule = ''

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
    }}{post_capsule}

    return 0;
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
                       kind_info=None, len_info=None):
    """Generate getters/setters that call Fortran accessor wrappers."""
    if specializations and kind_info:
        return _gen_opaque_getset_multi(
            typename, members, specializations, kind_info)
    sym = _fortran_sym(typename)
    capsule_name = f'f2py.{typename}'
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
    void *inner_ptr = f2py_get_{sym}_{mname}(ptr{len_call_extra});
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
    unsigned char is_assoc = f2py_get_{sym}_{mname}_associated(ptr);
    if (!is_assoc) Py_RETURN_NONE;
    /* Query shape from Fortran (F2018 16.9.182 size()) */
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
    /* Copy data from Fortran pointer target.
       Column-major layout for rank > 1 (F2018 8.5.8.1). */
    PyObject *arr;
    if (rank > 1) {{
        arr = PyArray_EMPTY(rank, numpy_dims, {npy_enum},
                            1 /* fortran_order */);
    }} else {{
        arr = PyArray_SimpleNew(rank, numpy_dims, {npy_enum});
    }}
    if (arr == NULL) return NULL;
    memcpy(PyArray_DATA((PyArrayObject *)arr), data,
           total_elements * sizeof({ptr_ctype}));
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
    unsigned char is_alloc = f2py_get_{sym}_{mname}_allocated(ptr);
    if (!is_alloc) Py_RETURN_NONE;
    /* Query runtime length (F2018 16.9.109 len()) */
    int str_length = f2py_get_{sym}_{mname}_len(ptr);
    if (str_length <= 0) return PyUnicode_FromString("");
    char *buffer = (char *)malloc(str_length + 1);
    if (buffer == NULL) return PyErr_NoMemory();
    f2py_get_{sym}_{mname}(ptr, buffer, str_length);
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
        f2py_set_{sym}_{mname}(ptr, "", 0);
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
    f2py_set_{sym}_{mname}(ptr, str_data, (int)py_str_length);
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
    npy_intp dims[{ndim}];
{dims_assign}
    npy_intp total = {total_expr};
    if (total <= 0) {{
        return PyArray_SimpleNew({ndim}, dims, {npy_enum});
    }}
    void *data = f2py_get_{sym}_{mname}_data(ptr{len_call_extra});
    if (data == NULL) {{
        return PyArray_SimpleNew({ndim}, dims, {npy_enum});
    }}
    PyObject *arr = PyArray_SimpleNew({ndim}, dims, {npy_enum});
    if (arr == NULL) return NULL;
    memcpy(PyArray_DATA((PyArrayObject *)arr), data,
           total * sizeof({ctype}));
    return arr;
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
    PyObject *arr = PyArray_FROM_OTF(value, {npy_enum},
                                      NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) return -1;
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
    void *arrptr = f2py_get_{sym}_{mname}(ptr{len_call_extra});
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

    return '\n'.join(funcs) + '\n' + getset_array


def _gen_opaque_getset_multi(typename, members, specializations, kind_info):
    """Generate multi-kind dispatch getters/setters for PDTs."""
    ki = kind_info[0]
    kname = ki['name']
    funcs = []
    getset_entries = []

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
                parse = f'(float)PyFloat_AsDouble(value)'
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
        if _is_pointer_member(mvar):
            fmt_parts.append(f'{mname}=<pointer>')
            continue
        if _is_deferred_char_member(mvar):
            fmt_parts.append(f'{mname}=<deferred-char>')
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
    char buf[256];
    snprintf(buf, sizeof(buf), "{typename}({fmt_str})",
        {val_str});
    return PyUnicode_FromString(buf);
}}
"""
