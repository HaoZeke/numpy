"""
Derived type support for f2py.

Generates Python class wrappers for Fortran derived types using PyCapsule.

For bind(c) types with scalar members: the C struct layout matches Fortran
directly; Python classes wrap these via PyCapsule pointing to heap-allocated
C structs.

For non-bind(c) types (opaque pointer path): auto-generated Fortran wrappers
with bind(c) provide constructor/destructor/getters/setters operating on
c_ptr. Python classes call these through PyCapsule.

Polymorphic arguments declared as ``class(T)`` are normalized to ``type(T)``
by crackfortran (see crackfortran.py line ~1636). Dynamic dispatch is lost;
the wrapped code treats ``class(T)`` identically to ``type(T)``.

Copyright 2024 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

import os

from .auxfuncs import (
    _SIMPLE_SCALAR_TYPESPECS,
    get_type_members,
    hasbody,
    is_fixed_array,
    is_simple_derived_type,
    isallocatable,
    isarray,
    isbindctype,
    isfunction,
    isroutine,
    issubroutine,
    outmess,
)

# Map Fortran typespec (+ optional kind) to C types for struct members
_FORTRAN_TO_C = {
    ('real', None): 'float',
    ('real', '4'): 'float',
    ('real', '8'): 'double',
    ('real', 'c_float'): 'float',
    ('real', 'c_double'): 'double',
    ('double precision', None): 'double',
    ('integer', None): 'int',
    ('integer', '4'): 'int',
    ('integer', '8'): 'long long',
    ('integer', 'c_int'): 'int',
    ('integer', 'c_long'): 'long',
    ('integer', 'c_long_long'): 'long long',
    ('logical', None): 'int',
    ('logical', 'c_bool'): 'unsigned char',
    ('complex', None): 'float _Complex',
    ('complex', '8'): 'float _Complex',
    ('complex', '16'): 'double _Complex',
    ('complex', 'c_float_complex'): 'float _Complex',
    ('complex', 'c_double_complex'): 'double _Complex',
    ('double complex', None): 'double _Complex',
}

# Map C type to Python format code for Py_BuildValue / PyArg_Parse
_C_TO_PYFORMAT = {
    'float': 'f',
    'double': 'd',
    'int': 'i',
    'long': 'l',
    'long long': 'L',
    'unsigned char': 'b',
    'float _Complex': 'D',
    'double _Complex': 'D',
}

# Map C type to PyObject conversion for getters
_C_TO_PYOBJ = {
    'float': 'PyFloat_FromDouble((double){val})',
    'double': 'PyFloat_FromDouble({val})',
    'int': 'PyLong_FromLong((long){val})',
    'long': 'PyLong_FromLong({val})',
    'long long': 'PyLong_FromLongLong({val})',
    'unsigned char': 'PyBool_FromLong((long){val})',
}

# Map C type to C-from-PyObject conversion for setters
_PYOBJ_TO_C = {
    'float': '(float)PyFloat_AsDouble({obj})',
    'double': 'PyFloat_AsDouble({obj})',
    'int': '(int)PyLong_AsLong({obj})',
    'long': 'PyLong_AsLong({obj})',
    'long long': 'PyLong_AsLongLong({obj})',
    'unsigned char': '(unsigned char)PyLong_AsLong({obj})',
}


# Map C type to NumPy type enum for array wrapping
_C_TO_NPY_ENUM = {
    'float': 'NPY_FLOAT',
    'double': 'NPY_DOUBLE',
    'int': 'NPY_INT',
    'long': 'NPY_LONG',
    'long long': 'NPY_LONGLONG',
    'unsigned char': 'NPY_UBYTE',
    'float _Complex': 'NPY_CFLOAT',
    'double _Complex': 'NPY_CDOUBLE',
}


# Map Fortran operator to Python number protocol slot
_FORTRAN_ARITH_OPS = {
    '+': 'nb_add',
    '-': 'nb_subtract',
    '*': 'nb_multiply',
    '/': 'nb_true_divide',
}

# Map Fortran comparison operator to Python richcompare constant
_FORTRAN_CMP_OPS = {
    '==': 'Py_EQ',
    '.eq.': 'Py_EQ',
    '/=': 'Py_NE',
    '.ne.': 'Py_NE',
    '<': 'Py_LT',
    '.lt.': 'Py_LT',
    '<=': 'Py_LE',
    '.le.': 'Py_LE',
    '>': 'Py_GT',
    '.gt.': 'Py_GT',
    '>=': 'Py_GE',
    '.ge.': 'Py_GE',
}


def _get_member_ctype(var):
    """Get C type string for a type member variable."""
    typespec = var.get('typespec', '').lower()
    kind = None
    if 'kindselector' in var:
        ks = var['kindselector']
        kind = ks.get('kind') or ks.get('*')
        if kind:
            kind = str(kind).lower()
    return _FORTRAN_TO_C.get((typespec, kind),
                             _FORTRAN_TO_C.get((typespec, None)))


def _get_array_dims(var):
    """Get fixed array dimensions as a list of ints, or None if scalar."""
    if not is_fixed_array(var):
        return None
    return [int(d) for d in var['dimension']]


def _is_array_member(var):
    """Check if a member is a fixed-size array."""
    return is_fixed_array(var)


def _get_extends_parent(typeblock):
    """Extract parent type name from extends(parent) attribute, or None.

    crackfortran stores type attributes in the parent module's vars dict,
    not on the type block itself. Check both locations.
    """
    for attr in typeblock.get('attrspec', []):
        if isinstance(attr, str) and attr.startswith('extends(') and attr.endswith(')'):
            return attr[len('extends('):-1]
    # Check parent module's vars dict
    parent = typeblock.get('parent_block')
    if parent and typeblock.get('name'):
        tname = typeblock['name']
        parent_var = parent.get('vars', {}).get(tname, {})
        for attr in parent_var.get('attrspec', []):
            if isinstance(attr, str) and attr.startswith('extends(') and attr.endswith(')'):
                return attr[len('extends('):-1]
    return None


def _get_all_members(typeblock, type_map):
    """Get all members including inherited ones from parent types.

    Returns (parent_members, own_members) where parent_members is an
    OrderedDict of members inherited from the parent chain and
    own_members is the type's own members.
    """
    from collections import OrderedDict
    parent_members = OrderedDict()
    parent_name = _get_extends_parent(typeblock)
    if parent_name and type_map:
        parent_tb = type_map.get(parent_name)
        if parent_tb:
            # Recursively get parent's full member set
            grandparent_members, parent_own = _get_all_members(
                parent_tb, type_map)
            parent_members.update(grandparent_members)
            parent_members.update(parent_own)
    own_members = get_type_members(typeblock)
    return parent_members, own_members


def _find_derived_types(module):
    """Find all type blocks in a module body.

    Sets parent_block on each type block so that functions like
    _get_extends_parent and isbindctype can check the parent module's
    vars dict (where crackfortran stores type attributes).
    """
    if not hasbody(module):
        return []
    types = [b for b in module['body'] if b.get('block') == 'type']
    for tb in types:
        tb['parent_block'] = module
    return types


def _topo_visit(tb, all_blocks, type_map, generated, gen_order):
    """Topological sort visit: ensure dependencies come first."""
    tname = tb['name'].lower()
    if tname in generated or tname not in type_map:
        return
    generated.add(tname)
    # Visit extends parent first (inheritance dependency)
    parent_name = _get_extends_parent(tb)
    if parent_name:
        for dep_tb in all_blocks:
            if dep_tb['name'].lower() == parent_name:
                _topo_visit(dep_tb, all_blocks, type_map,
                            generated, gen_order)
                break
    # Visit nested type member dependencies
    for mname, mvar in get_type_members(tb).items():
        if _is_type_member(mvar):
            inner = mvar.get('typename', '').lower()
            for dep_tb in all_blocks:
                if dep_tb['name'].lower() == inner:
                    _topo_visit(dep_tb, all_blocks, type_map,
                                generated, gen_order)
                    break
    gen_order.append(tb)


def _get_char_len(var):
    """Extract fixed character length as int, or None."""
    cs = var.get('charselector', {})
    char_len = cs.get('len') or cs.get('*')
    if char_len is None:
        return None
    try:
        return int(char_len)
    except (ValueError, TypeError):
        return None


def _is_char_member(var):
    """Check if a member is a fixed-length character."""
    return var.get('typespec', '') == 'character' and _get_char_len(var) is not None


def _is_complex_member(var):
    """Check if a member is a complex scalar (not array)."""
    ctype = _get_member_ctype(var)
    return ctype in ('float _Complex', 'double _Complex') and not _is_array_member(var)


def _is_allocatable_member(var):
    """Check if a member is a 1D allocatable numeric array."""
    if not isallocatable(var) or not isarray(var):
        return False
    typespec = var.get('typespec', '').lower()
    if typespec not in _SIMPLE_SCALAR_TYPESPECS:
        return False
    dims = var.get('dimension', [])
    return len(dims) == 1 and str(dims[0]).strip() == ':'


def _is_type_member(var):
    """Check if a member is a scalar nested derived type."""
    return (var.get('typespec') == 'type'
            and not _is_array_member(var))


def _is_type_array_member(var):
    """Check if a member is a fixed-size array of derived types."""
    return (var.get('typespec') == 'type'
            and is_fixed_array(var))


def _can_wrap_bindc(typeblock, type_map=None):
    """Check if a bind(c) type can be wrapped as a direct C struct.

    type_map is a dict of already-known wrappable types, used to
    validate nested type members.
    """
    if not isbindctype(typeblock):
        return False
    if not is_simple_derived_type(typeblock):
        return False
    for name, var in get_type_members(typeblock).items():
        if _is_type_member(var) or _is_type_array_member(var):
            inner = var.get('typename', '').lower()
            if type_map is None or inner not in type_map:
                return False
        elif _is_char_member(var):
            continue
        elif _is_allocatable_member(var):
            continue
        elif _get_member_ctype(var) is None:
            return False
    return True


def _can_wrap_opaque(typeblock, type_map=None):
    """Check if a non-bind(c) type can be wrapped via opaque pointers.

    type_map is a dict of already-known wrappable types, used to
    validate nested type members and extends(parent) dependencies.
    """
    if isbindctype(typeblock):
        return False
    if not is_simple_derived_type(typeblock):
        return False
    # Check extends parent is already wrappable
    parent_name = _get_extends_parent(typeblock)
    if parent_name:
        if type_map is None or parent_name not in type_map:
            return False
    for name, var in get_type_members(typeblock).items():
        if _is_type_member(var) or _is_type_array_member(var):
            inner = var.get('typename', '').lower()
            if type_map is None or inner not in type_map:
                return False
        elif _is_char_member(var):
            continue
        elif _is_allocatable_member(var):
            continue
        elif _get_member_ctype(var) is None:
            return False
    return True


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


def buildhooks(pymod):
    """Build C code hooks for derived type support in a module.

    Scans the module for wrappable derived types and generates:
    - C struct typedefs
    - Python type objects with getters/setters
    - Module init registration code
    - Wrapper functions for routines with derived type arguments

    Returns a dict with 'f90modhooks' and 'initf90modhooks' keys
    compatible with the existing f90mod_rules pipeline.
    """
    ret = {
        'f90modhooks': [],
        'initf90modhooks': [],
    }

    # Find modules in the pymod
    from .f90mod_rules import findf90modules
    for m in findf90modules(pymod):
        modulename = m['name']
        type_blocks = _find_derived_types(m)

        # Collect module-level routines (for TBP and routine wrapping)
        all_routines = [b for b in m.get('body', [])
                        if isroutine(b)]

        # Get source file for TBP scanning
        # The 'from' field format varies:
        #   direct crackfortran: '/path/to/file.f90'
        #   full f2py pipeline: ':modulename:/path/to/file.f90'
        source_file = m.get('from', '')
        if ':' in source_file:
            parts = source_file.split(':')
            for part in reversed(parts):
                if part and os.path.isfile(part):
                    source_file = part
                    break
            else:
                source_file = ''

        # Multi-pass type resolution: resolve leaf types first, then
        # types with nested type members (dependency ordering)
        type_map = {}
        remaining = list(type_blocks)
        max_passes = len(type_blocks) + 1
        for _ in range(max_passes):
            if not remaining:
                break
            still_remaining = []
            for tb in remaining:
                typename = tb['name']
                if (_can_wrap_bindc(tb, type_map)
                        or _can_wrap_opaque(tb, type_map)):
                    type_map[typename.lower()] = tb
                else:
                    still_remaining.append(tb)
            if len(still_remaining) == len(remaining):
                break  # no progress, stop
            remaining = still_remaining

        # Generate type wrappers in dependency order (leaf types first)
        generated = set()
        gen_order = []
        for tb in type_blocks:
            _topo_visit(tb, type_blocks, type_map,
                        generated, gen_order)

        for tb in gen_order:
            typename = tb['name']

            # Scan for type-bound procedures
            bound_procs = _scan_type_bound_procedures(
                source_file, typename)
            if bound_procs:
                outmess(f'\t\tFound type-bound procedures for '
                        f'"{typename}": '
                        f'{", ".join(bound_procs.keys())}\n')

            if _can_wrap_bindc(tb, type_map):
                outmess(f'\t\tGenerating bind(c) type wrapper '
                        f'for "{typename}"...\n')
                _generate_bindc_hooks(
                    ret, typename, tb, modulename, m,
                    bound_procs=bound_procs,
                    routines=all_routines,
                    type_map=type_map)
            elif _can_wrap_opaque(tb, type_map):
                outmess(f'\t\tGenerating opaque pointer type wrapper '
                        f'for "{typename}"...\n')
                _generate_opaque_hooks(
                    ret, typename, tb, modulename, m,
                    bound_procs=bound_procs,
                    routines=all_routines,
                    type_map=type_map)
            else:
                outmess(f'\t\tSkipping derived type "{typename}" '
                        f'(not wrappable yet)...\n')

        # Process routines with derived type arguments
        if type_map:
            wrappable_routines = _get_wrappable_routines(m, type_map)
            method_entries = []
            for routine in wrappable_routines:
                rname = routine['name']
                outmess(f'\t\tGenerating derived type routine wrapper '
                        f'for "{rname}"...\n')
                c_code, method_def = _gen_routine_c_wrapper(
                    modulename, routine, type_map)
                if c_code is not None:
                    ret['f90modhooks'].append(c_code)
                    method_entries.append(method_def)
            if method_entries:
                ret['f90modhooks'].append(
                    _gen_routine_method_table(modulename, method_entries))
                ret['initf90modhooks'].extend(
                    _gen_routine_init_code(modulename, method_entries))

    # Add <complex.h> to needs if any type has complex members
    if ret['f90modhooks']:
        for m in findf90modules(pymod):
            for tb in _find_derived_types(m):
                for mvar in get_type_members(tb).values():
                    ctype = _get_member_ctype(mvar)
                    if ctype in ('float _Complex', 'double _Complex'):
                        ret.setdefault('need', []).append('complex.h')
                        break
                if 'need' in ret:
                    break
            if 'need' in ret:
                break

    return ret


def _generate_bindc_hooks(ret, typename, typeblock, modulename,
                          module_block, bound_procs=None, routines=None,
                          type_map=None):
    """Generate hooks for a bind(c) derived type."""
    members = get_type_members(typeblock)

    code_parts = []
    code_parts.append(_gen_bindc_struct(typename, members))
    code_parts.append(_gen_pytype_struct(typename))
    code_parts.append(_gen_capsule_destructor(typename))
    code_parts.append(_gen_tp_new(typename))
    code_parts.append(_gen_tp_init(typename, members))
    code_parts.append(_gen_tp_dealloc(typename))
    code_parts.append(_gen_getset(typename, members))
    code_parts.append(_gen_tp_repr(typename, members))

    # Type-bound procedures
    has_methods = False
    if bound_procs and routines and type_map:
        method_funcs, method_entries = _gen_type_methods(
            typename, bound_procs, routines, type_map)
        if method_funcs:
            code_parts.extend(method_funcs)
            methods_table = (
                f'static PyMethodDef Py{typename}_methods[] = {{\n'
                + '\n'.join(method_entries) + '\n'
                + '    {NULL}  /* sentinel */\n'
                + '};\n')
            code_parts.append(methods_table)
            has_methods = True

    # Operator overloading
    has_number = False
    has_richcompare = False
    ops = _scan_operator_interfaces(module_block, typename, type_map)
    if ops:
        op_code, has_number, has_richcompare = _gen_operator_c_code(
            typename, ops, type_map)
        code_parts.extend(op_code)

    code_parts.append(_gen_typeobject(
        typename, has_methods=has_methods,
        has_number=has_number, has_richcompare=has_richcompare))

    ret['f90modhooks'].append('\n'.join(code_parts))
    ret['initf90modhooks'].extend(_gen_init_code(typename, modulename))


def _generate_opaque_hooks(ret, typename, typeblock, modulename,
                           module_block, bound_procs=None, routines=None,
                           type_map=None):
    """Generate hooks for a non-bind(c) derived type via opaque pointers.

    Uses the 3-layer approach: Python -> C wrapper -> Fortran accessor.
    The C code calls auto-generated bind(c) Fortran functions that use
    c_f_pointer to access the actual Fortran type. No C struct needed;
    the PyCapsule wraps a raw void* (Fortran c_ptr).

    For types with extends(parent), only child-specific members are
    added to getset (parent members inherited via tp_base). The Fortran
    wrappers and tp_init cover all members (parent + child).
    """
    own_members = get_type_members(typeblock)
    parent_name = _get_extends_parent(typeblock)

    # For inheritance: all_members = parent + own for extern/init/repr/getset
    # Each getset entry uses the child's capsule name and child's Fortran
    # accessors, so parent getset cannot be inherited via tp_base.
    if parent_name and type_map:
        parent_members, own = _get_all_members(typeblock, type_map)
        from collections import OrderedDict
        all_members = OrderedDict()
        all_members.update(parent_members)
        all_members.update(own)
    else:
        all_members = own_members
        parent_name = None

    code_parts = []
    code_parts.append(_gen_opaque_extern_decls(typename, all_members))
    code_parts.append(_gen_pytype_struct(typename))
    code_parts.append(_gen_opaque_capsule_destructor(typename))
    code_parts.append(_gen_tp_new(typename))
    code_parts.append(_gen_opaque_tp_init(typename, all_members))
    code_parts.append(_gen_tp_dealloc(typename))
    # All members in getset (capsule names differ per type, so parent
    # getset cannot be reused directly)
    code_parts.append(_gen_opaque_getset(typename, all_members))
    code_parts.append(_gen_opaque_tp_repr(typename, all_members))

    # Type-bound procedures
    has_methods = False
    if bound_procs and routines and type_map:
        method_funcs, method_entries = _gen_type_methods(
            typename, bound_procs, routines, type_map)
        if method_funcs:
            code_parts.extend(method_funcs)
            methods_table = (
                f'static PyMethodDef Py{typename}_methods[] = {{\n'
                + '\n'.join(method_entries) + '\n'
                + '    {NULL}  /* sentinel */\n'
                + '};\n')
            code_parts.append(methods_table)
            has_methods = True

    # Operator overloading
    has_number = False
    has_richcompare = False
    ops = _scan_operator_interfaces(module_block, typename, type_map)
    if ops:
        op_code, has_number, has_richcompare = _gen_operator_c_code(
            typename, ops, type_map)
        code_parts.extend(op_code)

    # Capitalize parent name to match PyTypeObject naming convention
    parent_typename = None
    if parent_name and type_map:
        parent_tb = type_map.get(parent_name)
        if parent_tb:
            parent_typename = parent_tb['name']

    code_parts.append(_gen_typeobject(
        typename, has_methods=has_methods,
        parent_typename=parent_typename,
        has_number=has_number, has_richcompare=has_richcompare))

    ret['f90modhooks'].append('\n'.join(code_parts))
    ret['initf90modhooks'].extend(_gen_init_code(typename, modulename))


def _get_c_return_type(ctype):
    """Map C type to the return type used by Fortran bind(c) wrappers."""
    # Fortran bind(c) functions return C-compatible types
    # logical is returned as int (c_int) from the Fortran side
    return ctype


def _fortran_sym(typename):
    """Return the lowercased symbol name for Fortran bind(c) wrappers.

    Fortran's bind(c) without an explicit name= clause always produces
    a lowercase binding label (F2018 18.10.2). The C extern declarations
    must match this exactly.
    """
    return typename.lower()


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
            # Character member: getter fills char buf, setter copies in
            char_len = _get_char_len(mvar)
            lines.append(
                f'extern void f2py_get_{sym}_{mname}'
                f'(void *, char *, int);')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *, const char *, int);')
            continue

        if _is_allocatable_member(mvar):
            # Allocatable array: 4 externs (allocated, size, data, set)
            lines.append(
                f'extern unsigned char f2py_get_{sym}_{mname}'
                f'_allocated(void *);')
            lines.append(
                f'extern int f2py_get_{sym}_{mname}'
                f'_size(void *);')
            lines.append(
                f'extern void *f2py_get_{sym}_{mname}'
                f'_data(void *);')
            lines.append(
                f'extern void f2py_set_{sym}_{mname}'
                f'(void *, int, void *);')
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
    int n = f2py_get_{sym}_{mname}_size(ptr);
    void *data = f2py_get_{sym}_{mname}_data(ptr);
    npy_intp dims[1] = {{n}};
    PyObject *arr = PyArray_SimpleNew(1, dims, {npy_enum});
    if (arr == NULL) return NULL;
    memcpy(PyArray_DATA((PyArrayObject *)arr), data,
           n * sizeof({alloc_ctype}));
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
    if (value == Py_None) {{
        f2py_set_{sym}_{mname}(ptr, 0, NULL);
        return 0;
    }}
    PyObject *arr = PyArray_FROM_OTF(value, {npy_enum},
                                      NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) return -1;
    int n = (int)PyArray_SIZE((PyArrayObject *)arr);
    f2py_set_{sym}_{mname}(ptr, n, PyArray_DATA((PyArrayObject *)arr));
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


# Fortran typespec to iso_c_binding type map for wrapper generation
_FORTRAN_TO_ISOC = {
    ('real', None): 'real(c_float)',
    ('real', '4'): 'real(c_float)',
    ('real', '8'): 'real(c_double)',
    ('real', 'c_float'): 'real(c_float)',
    ('real', 'c_double'): 'real(c_double)',
    ('double precision', None): 'real(c_double)',
    ('integer', None): 'integer(c_int)',
    ('integer', '4'): 'integer(c_int)',
    ('integer', '8'): 'integer(c_long_long)',
    ('integer', 'c_int'): 'integer(c_int)',
    ('integer', 'c_long_long'): 'integer(c_long_long)',
    ('logical', None): 'integer(c_int)',
    ('complex', None): 'complex(c_float_complex)',
    ('complex', '8'): 'complex(c_float_complex)',
    ('complex', '16'): 'complex(c_double_complex)',
    ('complex', 'c_float_complex'): 'complex(c_float_complex)',
    ('complex', 'c_double_complex'): 'complex(c_double_complex)',
    ('double complex', None): 'complex(c_double_complex)',
}


def _get_member_isoc_type(var):
    """Get iso_c_binding type for a member variable."""
    typespec = var.get('typespec', '').lower()
    kind = None
    if 'kindselector' in var:
        ks = var['kindselector']
        kind = ks.get('kind') or ks.get('*')
        if kind:
            kind = str(kind).lower()
    return _FORTRAN_TO_ISOC.get((typespec, kind),
                                _FORTRAN_TO_ISOC.get((typespec, None)))


def generate_fortran_wrappers(modulename, type_blocks, routines=None,
                              module_block=None):
    """Generate Fortran wrapper source code for opaque pointer types,
    routines with derived type arguments, and operator implementations.

    Returns the complete Fortran source as a string, or None if no
    wrappers are needed.
    """
    # Multi-pass type resolution (same as buildhooks)
    type_map = {}
    remaining = list(type_blocks)
    for _ in range(len(type_blocks) + 1):
        if not remaining:
            break
        still_remaining = []
        for tb in remaining:
            tname = tb['name'].lower()
            if (_can_wrap_bindc(tb, type_map)
                    or _can_wrap_opaque(tb, type_map)):
                type_map[tname] = tb
            else:
                still_remaining.append(tb)
        if len(still_remaining) == len(remaining):
            break
        remaining = still_remaining

    opaque_types = [tb for tb in type_blocks
                    if _can_wrap_opaque(tb, type_map)]

    # Find routines needing wrappers
    routine_wrappers = []
    if routines:
        for routine in routines:
            if _has_derived_type_args(routine):
                wrapper_name, fortran_code = _gen_routine_fortran_wrapper(
                    modulename, routine, type_map)
                if fortran_code is not None:
                    routine_wrappers.append(fortran_code)

    # Find operator wrappers needed
    operator_wrappers = []
    if module_block and type_map:
        for tb in type_blocks:
            typename = tb['name']
            if typename.lower() not in type_map:
                continue
            ops = _scan_operator_interfaces(
                module_block, typename, type_map)
            if ops:
                op_lines = _gen_operator_fortran_wrappers(
                    typename, ops, type_map)
                if op_lines:
                    operator_wrappers.extend(op_lines)

    if not opaque_types and not routine_wrappers and not operator_wrappers:
        return None

    lines = []
    lines.append('! Auto-generated by f2py: derived type opaque pointer wrappers')
    lines.append(f'module f2py_{modulename}_derived_wrappers')
    lines.append(f'  use {modulename}')
    lines.append('  use iso_c_binding')
    lines.append('  implicit none')
    lines.append('')

    lines.append('contains')
    lines.append('')

    for tb in opaque_types:
        typename = tb['name']
        # For types with extends(parent), include all inherited members
        parent_name = _get_extends_parent(tb)
        if parent_name and parent_name in type_map:
            from collections import OrderedDict
            parent_members, own_members = _get_all_members(
                tb, type_map)
            members = OrderedDict()
            members.update(parent_members)
            members.update(own_members)
        else:
            members = get_type_members(tb)

        # Constructor: allocate + populate + return c_ptr
        # (scalar members only; arrays are zero-initialized by allocate)
        args = []
        decls = []
        assigns = []
        for mname, mvar in members.items():
            if (_is_array_member(mvar) or _is_type_member(mvar)
                    or _is_type_array_member(mvar) or _is_char_member(mvar)
                    or _is_allocatable_member(mvar)):
                continue  # arrays, nested types, chars, allocs skip ctor
            isoc_type = _get_member_isoc_type(mvar)
            if isoc_type is None:
                continue
            args.append(mname)
            decls.append(f'    {isoc_type}, intent(in), value :: {mname}')
            assigns.append(f'    obj%{mname} = {mname}')

        args_str = ', '.join(args)
        decls_str = '\n'.join(decls)
        assigns_str = '\n'.join(assigns)

        lines.append(
            f'  function f2py_create_{typename}({args_str}) '
            f'result(cptr) bind(c)')
        lines.append(f'    type(c_ptr) :: cptr')
        lines.append(decls_str)
        lines.append(f'    type({typename}), pointer :: obj')
        lines.append(f'    allocate(obj)')
        # Zero-initialize all members (arrays default to 0 from allocate)
        for mname, mvar in members.items():
            if _is_type_array_member(mvar):
                inner = mvar.get('typename', '').lower()
                inner_tb = type_map.get(inner)
                if inner_tb:
                    dims = _get_array_dims(mvar)
                    total = 1
                    for d in dims:
                        total *= d
                    for idx in range(1, total + 1):
                        for imname, imvar in get_type_members(
                                inner_tb).items():
                            if not _is_array_member(imvar):
                                lines.append(
                                    f'    obj%{mname}({idx})%'
                                    f'{imname} = 0')
            elif _is_type_member(mvar):
                inner = mvar.get('typename', '').lower()
                inner_tb = type_map.get(inner)
                if inner_tb:
                    for imname, imvar in get_type_members(
                            inner_tb).items():
                        if not _is_array_member(imvar):
                            lines.append(
                                f'    obj%{mname}%{imname} = 0')
            elif _is_char_member(mvar):
                lines.append(f'    obj%{mname} = \' \'')
        lines.append(assigns_str)
        lines.append(f'    cptr = c_loc(obj)')
        lines.append(f'  end function f2py_create_{typename}')
        lines.append('')

        # Destructor
        lines.append(
            f'  subroutine f2py_destroy_{typename}(cptr) bind(c)')
        lines.append(f'    type(c_ptr), value :: cptr')
        lines.append(f'    type({typename}), pointer :: obj')
        lines.append(f'    call c_f_pointer(cptr, obj)')
        lines.append(f'    deallocate(obj)')
        lines.append(f'  end subroutine f2py_destroy_{typename}')
        lines.append('')

        # Getters and setters for each member
        for mname, mvar in members.items():
            if _is_type_array_member(mvar):
                inner = mvar.get('typename', '').lower()
                # Array of types: indexed getter (1-based)
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'(cptr, idx) '
                    f'result(inner_cptr) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(
                    f'    integer(c_int), value :: idx')
                lines.append(f'    type(c_ptr) :: inner_cptr')
                lines.append(
                    f'    type({typename}), pointer :: obj')
                lines.append(
                    f'    type({inner}), pointer :: inner_obj')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(f'    allocate(inner_obj)')
                lines.append(
                    f'    inner_obj = obj%{mname}(idx)')
                lines.append(
                    f'    inner_cptr = c_loc(inner_obj)')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}')
                lines.append('')

                # Array of types: indexed setter (1-based)
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr, idx, inner_cptr) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(
                    f'    integer(c_int), value :: idx')
                lines.append(
                    f'    type(c_ptr), value :: inner_cptr')
                lines.append(
                    f'    type({typename}), pointer :: obj')
                lines.append(
                    f'    type({inner}), pointer :: inner_obj')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    call c_f_pointer(inner_cptr, inner_obj)')
                lines.append(
                    f'    obj%{mname}(idx) = inner_obj')
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
                continue

            if _is_type_member(mvar):
                inner = mvar.get('typename', '').lower()
                # Nested type getter: allocate copy, return c_ptr
                lines.append(
                    f'  function f2py_get_{typename}_{mname}(cptr) '
                    f'result(inner_cptr) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(f'    type(c_ptr) :: inner_cptr')
                lines.append(f'    type({typename}), pointer :: obj')
                lines.append(f'    type({inner}), pointer :: inner_obj')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(f'    allocate(inner_obj)')
                lines.append(f'    inner_obj = obj%{mname}')
                lines.append(f'    inner_cptr = c_loc(inner_obj)')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}')
                lines.append('')

                # Nested type setter: copy from inner c_ptr
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr, inner_cptr) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(f'    type(c_ptr), value :: inner_cptr')
                lines.append(f'    type({typename}), pointer :: obj')
                lines.append(f'    type({inner}), pointer :: inner_obj')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    call c_f_pointer(inner_cptr, inner_obj)')
                lines.append(f'    obj%{mname} = inner_obj')
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
                continue

            if _is_char_member(mvar):
                char_len = _get_char_len(mvar)
                # Character getter: copies member into C buffer
                lines.append(
                    f'  subroutine f2py_get_{typename}_{mname}'
                    f'(cptr, buf, buflen) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(
                    f'    integer(c_int), value :: buflen')
                lines.append(
                    f'    character(c_char), intent(out) :: buf(buflen)')
                lines.append(
                    f'    type({typename}), pointer :: obj')
                lines.append(f'    integer :: i')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    do i = 1, min(buflen, '
                    f'len_trim(obj%{mname}))')
                lines.append(
                    f'      buf(i) = obj%{mname}(i:i)')
                lines.append(f'    end do')
                lines.append(
                    f'    do i = len_trim(obj%{mname})+1, buflen')
                lines.append(f'      buf(i) = \' \'')
                lines.append(f'    end do')
                lines.append(
                    f'  end subroutine f2py_get_{typename}_{mname}')
                lines.append('')

                # Character setter: copies C buffer into member
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr, buf, buflen) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(
                    f'    integer(c_int), value :: buflen')
                lines.append(
                    f'    character(c_char), intent(in) :: buf(buflen)')
                lines.append(
                    f'    type({typename}), pointer :: obj')
                lines.append(f'    integer :: i, copy_len')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(f'    obj%{mname} = \' \'')
                lines.append(
                    f'    copy_len = min(buflen, len(obj%{mname}))')
                lines.append(f'    do i = 1, copy_len')
                lines.append(
                    f'      obj%{mname}(i:i) = buf(i)')
                lines.append(f'    end do')
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
                continue

            if _is_allocatable_member(mvar):
                isoc_type = _get_member_isoc_type(mvar)
                if isoc_type is None:
                    continue
                # _allocated: returns logical(c_bool)
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_allocated(cptr) '
                    f'result(is_alloc) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(
                    f'    logical(c_bool) :: is_alloc')
                lines.append(
                    f'    type({typename}), pointer :: obj')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    is_alloc = allocated(obj%{mname})')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_allocated')
                lines.append('')

                # _size: returns integer(c_int)
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_size(cptr) '
                    f'result(n) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(f'    integer(c_int) :: n')
                lines.append(
                    f'    type({typename}), pointer :: obj')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    if (allocated(obj%{mname})) then')
                lines.append(
                    f'      n = size(obj%{mname})')
                lines.append(f'    else')
                lines.append(f'      n = 0')
                lines.append(f'    end if')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_size')
                lines.append('')

                # _data: returns type(c_ptr)
                # F2018 18.2.3.3: c_loc on allocatable requires F2008+
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_data(cptr) '
                    f'result(dptr) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(f'    type(c_ptr) :: dptr')
                lines.append(
                    f'    type({typename}), pointer :: obj')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    if (allocated(obj%{mname})) then')
                lines.append(
                    f'      dptr = c_loc(obj%{mname})')
                lines.append(f'    else')
                lines.append(f'      dptr = c_null_ptr')
                lines.append(f'    end if')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_data')
                lines.append('')

                # setter: deallocate if allocated, allocate(n), copy
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr, n, src) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(f'    integer(c_int), value :: n')
                lines.append(f'    type(c_ptr), value :: src')
                lines.append(
                    f'    type({typename}), pointer :: obj')
                lines.append(
                    f'    {isoc_type}, pointer :: src_arr(:)')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    if (allocated(obj%{mname})) '
                    f'deallocate(obj%{mname})')
                lines.append(f'    if (n > 0) then')
                lines.append(
                    f'      allocate(obj%{mname}(n))')
                lines.append(
                    f'      call c_f_pointer(src, src_arr, [n])')
                lines.append(
                    f'      obj%{mname} = src_arr')
                lines.append(f'    end if')
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
                continue

            isoc_type = _get_member_isoc_type(mvar)
            if isoc_type is None:
                continue
            dims = _get_array_dims(mvar)

            if dims:
                # Array getter: returns c_ptr to array data via c_loc
                total = 1
                for d in dims:
                    total *= d
                dim_str = ', '.join(str(d) for d in dims)

                lines.append(
                    f'  function f2py_get_{typename}_{mname}(cptr) '
                    f'result(arrptr) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(f'    type(c_ptr) :: arrptr')
                lines.append(f'    type({typename}), pointer :: obj')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(f'    arrptr = c_loc(obj%{mname})')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}')
                lines.append('')

                # Array setter: accepts pointer to array data, copies in
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr, arr) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(f'    {isoc_type}, intent(in) :: arr({total})')
                lines.append(f'    type({typename}), pointer :: obj')
                lines.append(f'    integer :: i')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                # Reshape from flat array into member shape
                if len(dims) == 1:
                    lines.append(f'    obj%{mname} = arr')
                else:
                    lines.append(
                        f'    obj%{mname} = reshape(arr, (/{dim_str}/))')
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
            else:
                # Scalar getter
                lines.append(
                    f'  function f2py_get_{typename}_{mname}(cptr) '
                    f'result(val) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(f'    {isoc_type} :: val')
                lines.append(f'    type({typename}), pointer :: obj')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(f'    val = obj%{mname}')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}')
                lines.append('')

                # Scalar setter
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr, val) bind(c)')
                lines.append(f'    type(c_ptr), value :: cptr')
                lines.append(f'    {isoc_type}, value :: val')
                lines.append(f'    type({typename}), pointer :: obj')
                lines.append(f'    call c_f_pointer(cptr, obj)')
                lines.append(f'    obj%{mname} = val')
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')

    # Add routine wrappers
    for wrapper_code in routine_wrappers:
        lines.append(wrapper_code)

    # Add operator wrappers
    lines.extend(operator_wrappers)

    lines.append(f'end module f2py_{modulename}_derived_wrappers')
    lines.append('')

    return '\n'.join(lines)


def write_fortran_wrappers(buildpath, modulename, type_blocks,
                           routines=None, module_block=None):
    """Write Fortran wrapper file if opaque types or wrapped routines exist.

    Returns the filename if written, None otherwise.
    """
    source = generate_fortran_wrappers(modulename, type_blocks,
                                       routines=routines,
                                       module_block=module_block)
    if source is None:
        return None

    fname = f'{modulename}-f2pyderivedwrappers.f90'
    fpath = os.path.join(buildpath, fname)
    with open(fpath, 'w') as f:
        f.write(source)
    outmess(f'\tFortran derived type wrappers saved to "{fpath}"\n')
    return fname


# --- Feature 4: Derived type arguments in routines ---

def _is_array_type_arg(var):
    """Check if a variable is an array of a derived type."""
    return (var.get('typespec') == 'type'
            and 'dimension' in var
            and len(var['dimension']) > 0)


def _has_derived_type_args(routine):
    """Check if a routine has any derived type arguments or returns one."""
    for argname in routine.get('args', []):
        var = routine['vars'].get(argname, {})
        if var.get('typespec') == 'type':
            return True
    if isfunction(routine):
        result_var = routine.get('result', routine['name'])
        rvar = routine['vars'].get(result_var, {})
        if rvar.get('typespec') == 'type':
            return True
    return False


def _get_wrappable_routines(module, type_map):
    """Find routines with derived type args where all types are wrappable.

    type_map: dict mapping typename -> typeblock for wrappable types.
    Returns list of routine blocks.
    """
    if not hasbody(module):
        return []
    result = []
    for b in module['body']:
        if not isroutine(b):
            continue
        if not _has_derived_type_args(b):
            continue
        # Check all derived type args reference wrappable types
        all_ok = True
        for argname in b.get('args', []):
            var = b['vars'].get(argname, {})
            if var.get('typespec') == 'type':
                tname = var.get('typename', '').lower()
                if tname not in type_map:
                    all_ok = False
                    break
        # Check function return type
        if all_ok and isfunction(b):
            result_var = b.get('result', b['name'])
            rvar = b['vars'].get(result_var, {})
            if rvar.get('typespec') == 'type':
                tname = rvar.get('typename', '').lower()
                if tname not in type_map:
                    all_ok = False
        if all_ok:
            result.append(b)
    return result


def _gen_routine_fortran_wrapper(modulename, routine, type_map):
    """Generate Fortran bind(c) wrapper for a routine with derived type args.

    The wrapper accepts void* (c_ptr) for each derived type argument,
    converts via c_f_pointer, and calls the original routine.
    """
    rname = routine['name']
    is_func = isfunction(routine)
    args = routine.get('args', [])

    wrapper_name = f'f2py_wrap_{rname}'
    wrapper_args = []
    decls = []
    local_decls = []
    pre_call = []
    post_call = []
    call_args = []

    result_var = None
    result_type = None
    returns_type = False
    if is_func:
        result_var = routine.get('result', rname)
        rvar = routine['vars'].get(result_var, {})
        if rvar.get('typespec') == 'type':
            returns_type = True
            result_type = 'type'
            tname_ret = rvar.get('typename', '').lower()
            local_decls.append(
                f'    type({tname_ret}), pointer :: f2py_temp_result')
        else:
            result_type = rvar.get('typespec', '')

    for argname in args:
        var = routine['vars'].get(argname, {})
        if var.get('typespec') == 'type' and _is_array_type_arg(var):
            tname = var.get('typename', '').lower()
            dim_expr = var['dimension'][0]
            intent = var.get('intent', [])
            if 'out' in intent and isallocatable(var):
                # Intent(out) allocatable array of derived type
                # Fortran allocates; we extract heap copies post-call
                wrapper_args.append(f'{argname}_ptrs')
                wrapper_args.append(f'{argname}_n')
                decls.append(
                    f'    type(c_ptr) :: {argname}_ptrs(*)')
                decls.append(
                    f'    integer(c_int) :: {argname}_n')
                local_decls.append(
                    f'    type({tname}), allocatable :: {argname}(:)')
                local_decls.append(
                    f'    type({tname}), pointer :: {argname}_heap_tmp')
                local_decls.append(f'    integer :: {argname}_i')
                # Post-call: extract each element as heap copy
                post_call.append(
                    f'    {argname}_n = size({argname})')
                post_call.append(
                    f'    do {argname}_i = 1, {argname}_n')
                post_call.append(
                    f'      allocate({argname}_heap_tmp)')
                post_call.append(
                    f'      {argname}_heap_tmp = '
                    f'{argname}({argname}_i)')
                post_call.append(
                    f'      {argname}_ptrs({argname}_i) = '
                    f'c_loc({argname}_heap_tmp)')
                post_call.append(
                    f'    end do')
                call_args.append(argname)
            else:
                # Input array of derived type argument
                wrapper_args.append(f'{argname}_ptrs')
                wrapper_args.append(f'{argname}_n')
                decls.append(
                    f'    type(c_ptr), intent(in) :: {argname}_ptrs(*)')
                decls.append(
                    f'    integer(c_int), value :: {argname}_n')
                local_decls.append(
                    f'    type({tname}), pointer :: {argname}_tmp')
                local_decls.append(
                    f'    type({tname}), allocatable :: {argname}(:)')
                local_decls.append(f'    integer :: {argname}_i')
                pre_call.append(
                    f'    allocate({argname}({argname}_n))')
                pre_call.append(
                    f'    do {argname}_i = 1, {argname}_n')
                pre_call.append(
                    f'      call c_f_pointer('
                    f'{argname}_ptrs({argname}_i), {argname}_tmp)')
                pre_call.append(
                    f'      {argname}({argname}_i) = {argname}_tmp')
                pre_call.append(
                    f'    end do')
                call_args.append(argname)
            # Skip the corresponding size arg from regular processing
            # by noting the dimension expression
        elif var.get('typespec') == 'type':
            tname = var.get('typename', '').lower()
            wrapper_args.append(f'{argname}_ptr')
            decls.append(f'    type(c_ptr), value :: {argname}_ptr')
            local_decls.append(
                f'    type({tname}), pointer :: {argname}')
            intent = var.get('intent', [])
            if 'out' in intent:
                # For intent(out), allocate a new one, call, return ptr
                pre_call.append(f'    allocate({argname})')
                post_call.append(
                    f'    {argname}_ptr = c_loc({argname})')
                # Rewrite decl: _ptr is inout (returns updated ptr)
                decls[-1] = f'    type(c_ptr) :: {argname}_ptr'
            else:
                pre_call.append(
                    f'    call c_f_pointer({argname}_ptr, {argname})')
            call_args.append(argname)
        else:
            # Scalar arg -- pass through
            isoc_type = _get_member_isoc_type(var)
            if isoc_type is None:
                return None, None
            wrapper_args.append(argname)
            intent = var.get('intent', [])
            if 'out' in intent or 'inout' in intent:
                decls.append(f'    {isoc_type} :: {argname}')
            else:
                decls.append(f'    {isoc_type}, value :: {argname}')
            call_args.append(argname)

    wrapper_args_str = ', '.join(wrapper_args)
    decls_str = '\n'.join(decls)
    local_decls_str = '\n'.join(local_decls)
    pre_call_str = '\n'.join(pre_call)
    post_call_str = '\n'.join(post_call)
    call_args_str = ', '.join(call_args)

    lines = []
    if is_func:
        if returns_type:
            lines.append(
                f'  function {wrapper_name}({wrapper_args_str}) '
                f'result(retval) bind(c)')
            lines.append(f'    type(c_ptr) :: retval')
        else:
            # Function returning scalar
            ret_isoc = _get_member_isoc_type(
                routine['vars'].get(result_var, {}))
            if ret_isoc is None:
                return None, None
            lines.append(
                f'  function {wrapper_name}({wrapper_args_str}) '
                f'result(retval) bind(c)')
            lines.append(f'    {ret_isoc} :: retval')
    else:
        lines.append(
            f'  subroutine {wrapper_name}({wrapper_args_str}) bind(c)')

    lines.append(decls_str)
    if local_decls_str:
        lines.append(local_decls_str)
    if pre_call_str:
        lines.append(pre_call_str)

    if is_func:
        if returns_type:
            lines.append(f'    allocate(f2py_temp_result)')
            lines.append(
                f'    f2py_temp_result = {rname}({call_args_str})')
            lines.append(f'    retval = c_loc(f2py_temp_result)')
        else:
            lines.append(
                f'    retval = {rname}({call_args_str})')
    else:
        lines.append(f'    call {rname}({call_args_str})')

    if post_call_str:
        lines.append(post_call_str)

    if is_func:
        lines.append(f'  end function {wrapper_name}')
    else:
        lines.append(f'  end subroutine {wrapper_name}')
    lines.append('')

    return wrapper_name, '\n'.join(lines)


def _gen_routine_c_wrapper(modulename, routine, type_map):
    """Generate C wrapper function for a routine with derived type args.

    Returns (c_code, method_def_entry) or (None, None) if not wrappable.
    """
    rname = routine['name']
    is_func = isfunction(routine)
    args = routine.get('args', [])
    wrapper_sym = f'f2py_wrap_{rname}'

    # Build extern declaration
    extern_args = []
    # Track which scalar args are consumed as array size parameters
    array_size_args = set()
    for argname in args:
        var = routine['vars'].get(argname, {})
        if _is_array_type_arg(var):
            # The dimension expression names the size arg
            dim_expr = var['dimension'][0]
            # Assumed-shape (:) has no named size variable
            if dim_expr != ':':
                array_size_args.add(dim_expr.lower())

    for argname in args:
        var = routine['vars'].get(argname, {})
        if _is_array_type_arg(var):
            intent = var.get('intent', [])
            extern_args.append('void **')
            if 'out' in intent and isallocatable(var):
                # Intent(out): count is passed by reference (inout)
                extern_args.append('int *')
            else:
                extern_args.append('int')
        elif var.get('typespec') == 'type':
            intent = var.get('intent', [])
            if 'out' in intent:
                extern_args.append('void **')
            else:
                extern_args.append('void *')
        elif argname.lower() in array_size_args:
            # Size arg consumed by array-of-type; still passed to Fortran
            ctype = _get_member_ctype(var)
            if ctype is None:
                return None, None
            extern_args.append(ctype)
        else:
            ctype = _get_member_ctype(var)
            if ctype is None:
                return None, None
            intent = var.get('intent', [])
            if 'out' in intent or 'inout' in intent:
                extern_args.append(f'{ctype} *')
            else:
                extern_args.append(ctype)

    result_var = None
    result_ctype = None
    returns_type_name = None
    if is_func:
        result_var = routine.get('result', rname)
        rvar = routine['vars'].get(result_var, {})
        if rvar.get('typespec') == 'type':
            returns_type_name = rvar.get('typename', '').lower()
            extern_ret = 'void *'
        else:
            result_ctype = _get_member_ctype(rvar)
            if result_ctype is None:
                return None, None
            extern_ret = result_ctype
    else:
        extern_ret = 'void'

    extern_args_str = ', '.join(extern_args) if extern_args else 'void'
    extern_line = f'extern {extern_ret} {wrapper_sym}({extern_args_str});'

    # Build Python wrapper function
    lines = []
    lines.append(extern_line)
    lines.append('')
    lines.append(f'static PyObject *')
    lines.append(
        f'f2py_routine_{rname}(PyObject *self, PyObject *args, '
        f'PyObject *kwds)')
    lines.append('{')

    # Build kwlist and format string
    kwlist = []
    fmt_parts = []
    parse_args_decl = []
    parse_args_ref = []
    call_parts = []

    for argname in args:
        var = routine['vars'].get(argname, {})
        if _is_array_type_arg(var):
            tname = var.get('typename', '').lower()
            intent = var.get('intent', [])
            if 'out' in intent and isallocatable(var):
                # Intent(out) allocatable array of type: output only
                # No Python input needed; handled post-call
                pass
            else:
                # Input array of derived type: accept Python list
                kwlist.append(argname)
                fmt_parts.append('O')
                parse_args_decl.append(
                    f'    PyObject *{argname}_list = NULL;')
                parse_args_ref.append(f'&{argname}_list')
        elif var.get('typespec') == 'type':
            tname = var.get('typename', '').lower()
            intent = var.get('intent', [])
            kwlist.append(argname)
            if 'out' in intent:
                # intent(out) type arg: not passed from Python
                # Actually we still need to pass something -- allocate on
                # Fortran side. Skip from Python args; handled separately.
                kwlist.pop()  # remove from kwlist
                continue
            fmt_parts.append('O')
            parse_args_decl.append(f'    PyObject *{argname}_obj = NULL;')
            parse_args_ref.append(f'&{argname}_obj')
        elif argname.lower() in array_size_args:
            # Size arg consumed by array-of-type; derived from list length
            ctype = _get_member_ctype(var)
            parse_args_decl.append(f'    {ctype} {argname} = 0;')
            # Do NOT add to kwlist/fmt/refs -- computed from list length
        else:
            ctype = _get_member_ctype(var)
            fmt = _C_TO_PYFORMAT.get(ctype)
            if fmt is None:
                return None, None
            intent = var.get('intent', [])
            kwlist.append(argname)
            if 'out' in intent:
                # Skip from Python args for intent(out) scalars
                kwlist.pop()
                fmt_parts.append('')  # placeholder
                parse_args_decl.append(f'    {ctype} {argname} = 0;')
                continue
            fmt_parts.append(fmt)
            parse_args_decl.append(f'    {ctype} {argname} = 0;')
            parse_args_ref.append(f'&{argname}')

    kwlist_str = ', '.join(f'"{k}"' for k in kwlist)
    fmt_str = '|' + ''.join(fmt_parts)
    parse_args_str = ', '.join(parse_args_ref)

    for decl in parse_args_decl:
        lines.append(decl)
    lines.append('')

    if kwlist:
        lines.append(
            f'    static char *kwlist[] = {{{kwlist_str}, NULL}};')
        lines.append(
            f'    if (!PyArg_ParseTupleAndKeywords(args, kwds, '
            f'"{fmt_str}", kwlist,')
        lines.append(f'                                     {parse_args_str}))')
        lines.append('        return NULL;')
    lines.append('')

    # Extract capsule pointers for type args
    # Track array-of-type args that need cleanup
    array_type_cleanup = []
    # Track intent(out) array-of-type args for post-call list building
    out_array_type_args = []
    for argname in args:
        var = routine['vars'].get(argname, {})
        if _is_array_type_arg(var):
            tname = var.get('typename', '').lower()
            intent = var.get('intent', [])
            if 'out' in intent and isallocatable(var):
                # Intent(out) allocatable array of type
                # Allocate buffer for Fortran to fill with pointers
                lines.append(
                    f'    void *{argname}_ptrs[4096];')
                lines.append(
                    f'    int {argname}_n = 0;')
                out_array_type_args.append((argname, tname))
            else:
                capsule_name = f'f2py.{tname}'
                dim_expr = var['dimension'][0].lower()
                lines.append(
                    f'    if ({argname}_list == NULL || '
                    f'!PyList_Check({argname}_list)) {{')
                lines.append(
                    f'        PyErr_SetString(PyExc_TypeError, '
                    f'"{argname} must be a list of {tname} instances");')
                lines.append('        return NULL;')
                lines.append('    }')
                lines.append(
                    f'    Py_ssize_t {argname}_len = '
                    f'PyList_GET_SIZE({argname}_list);')
                # Set the size arg from the list length
                # (skip for assumed-shape ':' -- no named size variable)
                if dim_expr != ':':
                    lines.append(
                        f'    {dim_expr} = (int){argname}_len;')
                lines.append(
                    f'    void **{argname}_ptrs = (void **)malloc('
                    f'sizeof(void *) * {argname}_len);')
                lines.append(
                    f'    if ({argname}_ptrs == NULL) {{')
                lines.append(
                    f'        PyErr_NoMemory();')
                lines.append('        return NULL;')
                lines.append('    }')
                lines.append(
                    f'    for (Py_ssize_t _i = 0; '
                    f'_i < {argname}_len; _i++) {{')
                lines.append(
                    f'        PyObject *_item = '
                    f'PyList_GET_ITEM({argname}_list, _i);')
                lines.append(
                    f'        if (!Py_IS_TYPE(_item, '
                    f'&Py{tname}_Type)) {{')
                lines.append(
                    f'            free({argname}_ptrs);')
                lines.append(
                    f'            PyErr_SetString(PyExc_TypeError, '
                    f'"{argname} items must be {tname} instances");')
                lines.append('            return NULL;')
                lines.append('        }')
                lines.append(
                    f'        Py{tname}Object *_typed = '
                    f'(Py{tname}Object *)_item;')
                lines.append(
                    f'        if (_typed->capsule == NULL) {{')
                lines.append(
                    f'            free({argname}_ptrs);')
                lines.append(
                    f'            PyErr_SetString(PyExc_RuntimeError, '
                    f'"{argname} item not initialized");')
                lines.append('            return NULL;')
                lines.append('        }')
                lines.append(
                    f'        {argname}_ptrs[_i] = PyCapsule_GetPointer('
                    f'_typed->capsule, "{capsule_name}");')
                lines.append(
                    f'        if ({argname}_ptrs[_i] == NULL) {{')
                lines.append(
                    f'            free({argname}_ptrs);')
                lines.append('            return NULL;')
                lines.append('        }')
                lines.append('    }')
                array_type_cleanup.append(f'{argname}_ptrs')
        elif var.get('typespec') == 'type':
            tname = var.get('typename', '').lower()
            intent = var.get('intent', [])
            capsule_name = f'f2py.{tname}'
            if 'out' in intent:
                lines.append(f'    void *{argname}_ptr = NULL;')
            else:
                lines.append(
                    f'    if ({argname}_obj == NULL) {{')
                lines.append(
                    f'        PyErr_SetString(PyExc_TypeError, '
                    f'"{argname} is required");')
                lines.append('        return NULL;')
                lines.append('    }')
                # Check it's the right type
                lines.append(
                    f'    if (!Py_IS_TYPE({argname}_obj, '
                    f'&Py{tname}_Type)) {{')
                lines.append(
                    f'        PyErr_SetString(PyExc_TypeError, '
                    f'"{argname} must be a {tname} instance");')
                lines.append('        return NULL;')
                lines.append('    }')
                lines.append(
                    f'    Py{tname}Object *{argname}_typed = '
                    f'(Py{tname}Object *){argname}_obj;')
                lines.append(
                    f'    if ({argname}_typed->capsule == NULL) {{')
                lines.append(
                    f'        PyErr_SetString(PyExc_RuntimeError, '
                    f'"{argname} not initialized");')
                lines.append('        return NULL;')
                lines.append('    }')
                lines.append(
                    f'    void *{argname}_ptr = PyCapsule_GetPointer('
                    f'{argname}_typed->capsule, "{capsule_name}");')
                lines.append(
                    f'    if ({argname}_ptr == NULL) return NULL;')
    lines.append('')

    # Build call
    call_args = []
    for argname in args:
        var = routine['vars'].get(argname, {})
        if _is_array_type_arg(var):
            intent = var.get('intent', [])
            if 'out' in intent and isallocatable(var):
                # Intent(out): pass buffer + pointer to count
                call_args.append(f'{argname}_ptrs')
                call_args.append(f'&{argname}_n')
            else:
                # Input: pass void** array and int count
                call_args.append(f'{argname}_ptrs')
                call_args.append(f'(int){argname}_len')
        elif var.get('typespec') == 'type':
            intent = var.get('intent', [])
            if 'out' in intent:
                call_args.append(f'&{argname}_ptr')
            else:
                call_args.append(f'{argname}_ptr')
        elif argname.lower() in array_size_args:
            # Size arg -- pass the value derived from list length
            call_args.append(argname)
        else:
            intent = var.get('intent', [])
            if 'out' in intent or 'inout' in intent:
                call_args.append(f'&{argname}')
            else:
                call_args.append(argname)

    call_args_str = ', '.join(call_args)

    if is_func:
        if returns_type_name:
            tname = returns_type_name
            capsule_name = f'f2py.{tname}'
            lines.append(
                f'    void *result = {wrapper_sym}({call_args_str});')
            for cleanup_var in array_type_cleanup:
                lines.append(f'    free({cleanup_var});')
            lines.append(f'    if (result == NULL) {{')
            lines.append(
                f'        PyErr_SetString(PyExc_RuntimeError, '
                f'"Fortran function returned NULL");')
            lines.append('        return NULL;')
            lines.append('    }')
            lines.append(
                f'    Py{tname}Object *out = (Py{tname}Object *)'
                f'Py{tname}_Type.tp_alloc(&Py{tname}_Type, 0);')
            lines.append('    if (out == NULL) return NULL;')
            lines.append(
                f'    out->capsule = PyCapsule_New('
                f'result, "{capsule_name}", '
                f'f2py_{tname}_capsule_destructor);')
            lines.append(
                '    if (out->capsule == NULL) { '
                'Py_DECREF(out); return NULL; }')
            lines.append('    return (PyObject *)out;')
        else:
            lines.append(
                f'    {result_ctype} result = '
                f'{wrapper_sym}({call_args_str});')
            # Free array-of-type temporary arrays
            for cleanup_var in array_type_cleanup:
                lines.append(f'    free({cleanup_var});')
            # Convert result to PyObject
            pyobj_expr = _C_TO_PYOBJ.get(result_ctype)
            if pyobj_expr:
                lines.append(
                    f'    return {pyobj_expr.format(val="result")};')
            else:
                lines.append('    Py_RETURN_NONE;')
    else:
        lines.append(f'    {wrapper_sym}({call_args_str});')
        # Free array-of-type temporary arrays
        for cleanup_var in array_type_cleanup:
            lines.append(f'    free({cleanup_var});')

        # Build return value -- collect intent(out) and intent(inout) args
        out_parts = []
        for argname in args:
            var = routine['vars'].get(argname, {})
            intent = var.get('intent', [])
            if (_is_array_type_arg(var) and 'out' in intent
                    and isallocatable(var)):
                tname = var.get('typename', '').lower()
                out_parts.append((argname, 'type_array', tname))
            elif var.get('typespec') == 'type' and 'out' in intent:
                tname = var.get('typename', '').lower()
                out_parts.append((argname, 'type', tname))
            elif 'out' in intent or 'inout' in intent:
                ctype = _get_member_ctype(var)
                out_parts.append((argname, 'scalar', ctype))

        if not out_parts:
            lines.append('    Py_RETURN_NONE;')
        else:
            # Helper: emit code for a single out_part into `lines`
            def _emit_out_part(aname, kind, info, var_suffix=''):
                """Emit lines to build a PyObject for one output part.

                var_suffix: suffix for local variable names (e.g. '0')
                Returns the variable name holding the PyObject*.
                """
                if kind == 'scalar':
                    pyobj_expr = _C_TO_PYOBJ.get(info)
                    if pyobj_expr:
                        vname = f'out_val{var_suffix}'
                        lines.append(
                            f'    PyObject *{vname} = '
                            f'{pyobj_expr.format(val=aname)};')
                        return vname
                    return None
                elif kind == 'type':
                    tname = info
                    capsule_name = f'f2py.{tname}'
                    vname = f'out{var_suffix}'
                    lines.append(
                        f'    Py{tname}Object *{vname} = '
                        f'(Py{tname}Object *)'
                        f'Py{tname}_Type.tp_alloc('
                        f'&Py{tname}_Type, 0);')
                    lines.append(
                        f'    if ({vname} == NULL) return NULL;')
                    lines.append(
                        f'    {vname}->capsule = PyCapsule_New('
                        f'{aname}_ptr, "{capsule_name}", '
                        f'f2py_{tname}_capsule_destructor);')
                    lines.append(
                        f'    if ({vname}->capsule == NULL) {{ '
                        f'Py_DECREF({vname}); return NULL; }}')
                    return f'(PyObject *){vname}'
                elif kind == 'type_array':
                    tname = info
                    capsule_name = f'f2py.{tname}'
                    vname = f'out_list{var_suffix}'
                    lines.append(
                        f'    PyObject *{vname} = '
                        f'PyList_New({aname}_n);')
                    lines.append(
                        f'    if ({vname} == NULL) return NULL;')
                    lines.append(
                        f'    for (int _j = 0; '
                        f'_j < {aname}_n; _j++) {{')
                    lines.append(
                        f'        Py{tname}Object *_item = '
                        f'(Py{tname}Object *)'
                        f'Py{tname}_Type.tp_alloc('
                        f'&Py{tname}_Type, 0);')
                    lines.append(
                        f'        if (_item == NULL) {{ '
                        f'Py_DECREF({vname}); return NULL; }}')
                    lines.append(
                        f'        _item->capsule = PyCapsule_New('
                        f'{aname}_ptrs[_j], "{capsule_name}", '
                        f'f2py_{tname}_capsule_destructor);')
                    lines.append(
                        f'        if (_item->capsule == NULL) {{ '
                        f'Py_DECREF(_item); Py_DECREF({vname}); '
                        f'return NULL; }}')
                    lines.append(
                        f'        PyList_SET_ITEM({vname}, _j, '
                        f'(PyObject *)_item);')
                    lines.append('    }')
                    return vname
                return None

            if len(out_parts) == 1:
                aname, kind, info = out_parts[0]
                result_var_name = _emit_out_part(aname, kind, info)
                if result_var_name:
                    lines.append(
                        f'    return {result_var_name};')
                else:
                    lines.append('    Py_RETURN_NONE;')
            else:
                # Multiple outputs -- return tuple
                lines.append(
                    f'    PyObject *ret = '
                    f'PyTuple_New({len(out_parts)});')
                lines.append(
                    '    if (ret == NULL) return NULL;')
                for i, (aname, kind, info) in enumerate(out_parts):
                    result_var_name = _emit_out_part(
                        aname, kind, info, var_suffix=str(i))
                    if result_var_name:
                        lines.append(
                            f'    PyTuple_SET_ITEM(ret, {i}, '
                            f'{result_var_name});')
                lines.append('    return ret;')

    lines.append('}')
    lines.append('')

    c_code = '\n'.join(lines)

    # Method definition entry
    method_def = (
        f'    {{"{rname}", (PyCFunction)f2py_routine_{rname}, '
        f'METH_VARARGS | METH_KEYWORDS, '
        f'"Wrapper for Fortran routine {rname}"}},'
    )

    return c_code, method_def


def _scan_type_bound_procedures(source_file, typename):
    """Scan Fortran source for type-bound procedure declarations.

    Returns a dict mapping method_name -> impl_name from lines like:
        procedure :: method => impl
        procedure :: method  (impl == method when no =>)
    within the 'contains' section of the named type block.
    """
    import re
    result = {}
    if not source_file or not os.path.isfile(source_file):
        return result

    in_type = False
    in_contains = False
    type_pat = re.compile(
        r'^\s*type\b(?:\s*,\s*\w+)*\s*::\s*' + re.escape(typename),
        re.I)
    end_type_pat = re.compile(
        r'^\s*end\s+type\b', re.I)
    contains_pat = re.compile(r'^\s*contains\b', re.I)
    proc_pat = re.compile(
        r'^\s*procedure\s*::\s*(\w+)\s*(?:=>\s*(\w+))?\s*$', re.I)

    with open(source_file) as f:
        for line in f:
            stripped = line.split('!')[0].strip()  # remove comments
            if not stripped:
                continue
            if not in_type:
                if type_pat.match(stripped):
                    in_type = True
                continue
            if end_type_pat.match(stripped):
                break
            if contains_pat.match(stripped):
                in_contains = True
                continue
            if in_contains:
                m = proc_pat.match(stripped)
                if m:
                    method_name = m.group(1).lower()
                    impl_name = (m.group(2) or m.group(1)).lower()
                    result[method_name] = impl_name

    return result


def _gen_type_methods(typename, bound_procs, routines, type_map):
    """Generate PyMethodDef entries and C method functions for type-bound
    procedures.

    Each method is a thin wrapper that calls the already-generated
    module-level routine wrapper, automatically passing 'self' as the
    first (derived type) argument.
    """
    funcs = []
    method_entries = []

    # Build lookup from routine name to routine block
    routine_map = {}
    for r in routines:
        routine_map[r['name'].lower()] = r

    for method_name, impl_name in bound_procs.items():
        routine = routine_map.get(impl_name)
        if routine is None:
            continue

        args = routine.get('args', [])
        if not args:
            continue

        # First arg should be the self/this type arg
        self_arg = args[0]
        self_var = routine['vars'].get(self_arg, {})
        if self_var.get('typespec') != 'type':
            continue

        # Remaining args (non-self)
        other_args = args[1:]

        # Generate a C method that takes the type instance as self
        # and remaining args as positional/keyword args
        is_func = isfunction(routine)
        wrapper_sym = f'f2py_wrap_{impl_name}'

        # Generate extern declaration for the Fortran wrapper
        extern_args = []
        for argname in args:
            var = routine['vars'].get(argname, {})
            if var.get('typespec') == 'type':
                intent = var.get('intent', [])
                if 'out' in intent:
                    extern_args.append('void **')
                else:
                    extern_args.append('void *')
            else:
                ctype = _get_member_ctype(var)
                if ctype is None:
                    break
                intent = var.get('intent', [])
                if 'out' in intent or 'inout' in intent:
                    extern_args.append(f'{ctype} *')
                else:
                    extern_args.append(ctype)

        if is_func:
            result_var = routine.get('result', impl_name)
            rvar = routine['vars'].get(result_var, {})
            result_ctype = _get_member_ctype(rvar)
            extern_ret = result_ctype if result_ctype else 'void'
        else:
            extern_ret = 'void'

        extern_args_str = (', '.join(extern_args)
                           if extern_args else 'void')

        # Build the method C function
        mfunc_name = f'Py{typename}_method_{method_name}'
        lines = []
        lines.append(
            f'extern {extern_ret} {wrapper_sym}({extern_args_str});')
        lines.append('')
        lines.append(f'static PyObject *')
        lines.append(
            f'{mfunc_name}(PyObject *selfobj, PyObject *args, '
            f'PyObject *kwds)')
        lines.append('{')
        lines.append(
            f'    Py{typename}Object *self = '
            f'(Py{typename}Object *)selfobj;')

        capsule_name = f'f2py.{typename.lower()}'
        lines.append(f'    if (self->capsule == NULL) {{')
        lines.append(
            f'        PyErr_SetString(PyExc_RuntimeError, '
            f'"{typename} not initialized");')
        lines.append('        return NULL;')
        lines.append('    }')
        lines.append(
            f'    void *self_ptr = PyCapsule_GetPointer('
            f'self->capsule, "{capsule_name}");')
        lines.append('    if (self_ptr == NULL) return NULL;')
        lines.append('')

        # Parse remaining args
        kwlist = []
        fmt_parts = []
        parse_decls = []
        parse_refs = []
        call_args = ['self_ptr']

        for argname in other_args:
            var = routine['vars'].get(argname, {})
            if var.get('typespec') == 'type':
                tname = var.get('typename', '').lower()
                intent = var.get('intent', [])
                if 'out' in intent:
                    call_args.append(f'&{argname}_ptr')
                    lines.append(f'    void *{argname}_ptr = NULL;')
                    continue
                kwlist.append(argname)
                fmt_parts.append('O')
                parse_decls.append(
                    f'    PyObject *{argname}_obj = NULL;')
                parse_refs.append(f'&{argname}_obj')
                # Will extract pointer below
            else:
                ctype = _get_member_ctype(var)
                fmt = _C_TO_PYFORMAT.get(ctype)
                if fmt is None:
                    break  # skip this method
                intent = var.get('intent', [])
                if 'out' in intent:
                    parse_decls.append(f'    {ctype} {argname} = 0;')
                    call_args.append(f'&{argname}')
                    continue
                kwlist.append(argname)
                fmt_parts.append(fmt)
                parse_decls.append(f'    {ctype} {argname} = 0;')
                parse_refs.append(f'&{argname}')
                intent = var.get('intent', [])
                if 'out' in intent or 'inout' in intent:
                    call_args.append(f'&{argname}')
                else:
                    call_args.append(argname)

        for decl in parse_decls:
            lines.append(decl)

        if kwlist:
            kwlist_str = ', '.join(f'"{k}"' for k in kwlist)
            fmt_str = '|' + ''.join(fmt_parts)
            parse_refs_str = ', '.join(parse_refs)
            lines.append(
                f'    static char *kwlist[] = {{{kwlist_str}, NULL}};')
            lines.append(
                f'    if (!PyArg_ParseTupleAndKeywords(args, kwds, '
                f'"{fmt_str}", kwlist,')
            lines.append(
                f'                                     {parse_refs_str}))')
            lines.append('        return NULL;')
        lines.append('')

        # Extract capsule pointers for type args (non-self)
        for argname in other_args:
            var = routine['vars'].get(argname, {})
            if var.get('typespec') == 'type':
                tname = var.get('typename', '').lower()
                intent = var.get('intent', [])
                if 'out' in intent:
                    continue
                cap_name = f'f2py.{tname}'
                lines.append(
                    f'    if ({argname}_obj == NULL || '
                    f'!Py_IS_TYPE({argname}_obj, &Py{tname}_Type)) {{')
                lines.append(
                    f'        PyErr_SetString(PyExc_TypeError, '
                    f'"{argname} must be a {tname} instance");')
                lines.append('        return NULL;')
                lines.append('    }')
                lines.append(
                    f'    void *{argname}_ptr = PyCapsule_GetPointer('
                    f'((Py{tname}Object *){argname}_obj)->capsule, '
                    f'"{cap_name}");')
                lines.append(
                    f'    if ({argname}_ptr == NULL) return NULL;')
                call_args.append(f'{argname}_ptr')

        call_args_str = ', '.join(call_args)

        if is_func:
            result_var = routine.get('result', impl_name)
            rvar = routine['vars'].get(result_var, {})
            result_ctype = _get_member_ctype(rvar)
            if result_ctype:
                lines.append(
                    f'    {result_ctype} result = '
                    f'{wrapper_sym}({call_args_str});')
                pyobj_expr = _C_TO_PYOBJ.get(result_ctype)
                if pyobj_expr:
                    lines.append(
                        f'    return '
                        f'{pyobj_expr.format(val="result")};')
                else:
                    lines.append('    Py_RETURN_NONE;')
            else:
                lines.append(f'    {wrapper_sym}({call_args_str});')
                lines.append('    Py_RETURN_NONE;')
        else:
            lines.append(f'    {wrapper_sym}({call_args_str});')
            lines.append('    Py_RETURN_NONE;')

        lines.append('}')
        lines.append('')

        funcs.append('\n'.join(lines))
        method_entries.append(
            f'    {{"{method_name}", (PyCFunction){mfunc_name}, '
            f'METH_VARARGS | METH_KEYWORDS, '
            f'"Type-bound procedure {method_name}"}},')

    return funcs, method_entries


def _gen_routine_method_table(modulename, method_entries):
    """Generate PyMethodDef table for wrapped routines."""
    lines = []
    lines.append(
        f'static PyMethodDef f2py_{modulename}_derived_methods[] = {{')
    for entry in method_entries:
        lines.append(entry)
    lines.append('    {NULL, NULL, 0, NULL}')
    lines.append('};')
    return '\n'.join(lines)


def _gen_routine_init_code(modulename, method_entries):
    """Generate module init code to register wrapped routine methods."""
    if not method_entries:
        return []
    lines = []
    lines.append('\t{')
    lines.append(f'\t\tPyMethodDef *mdef = f2py_{modulename}_derived_methods;')
    lines.append('\t\twhile (mdef->ml_name != NULL) {')
    lines.append('\t\t\tPyObject *func = PyCFunction_New(mdef, NULL);')
    lines.append('\t\t\tif (func != NULL) {')
    lines.append('\t\t\t\tPyDict_SetItemString(d, mdef->ml_name, func);')
    lines.append('\t\t\t\tPy_DECREF(func);')
    lines.append('\t\t\t}')
    lines.append('\t\t\tmdef++;')
    lines.append('\t\t}')
    lines.append('\t}')
    return lines
