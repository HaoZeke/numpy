"""
Type maps and member inspection helpers for derived type support.

Copyright 2024 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

from .auxfuncs import (
    _SIMPLE_SCALAR_TYPESPECS,
    get_type_members,
    hasbody,
    is_fixed_array,
    is_simple_derived_type,
    isallocatable,
    isarray,
    isbindctype,
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


def _is_deferred_char_member(var):
    """Check if a member is a deferred-length allocatable character.

    Deferred-length character components (F2018 7.4.4.2 paragraph 3)
    use a colon for the length type parameter:
        character(:), allocatable :: name
    The length is determined at runtime when allocated.
    """
    if var.get('typespec', '') != 'character':
        return False
    if not isallocatable(var):
        return False
    char_selector = var.get('charselector', {})
    char_len = char_selector.get('len') or char_selector.get('*')
    return char_len is not None and str(char_len).strip() == ':'


def _is_complex_member(var):
    """Check if a member is a complex scalar (not array)."""
    ctype = _get_member_ctype(var)
    return ctype in ('float _Complex', 'double _Complex') and not _is_array_member(var)


def _is_allocatable_member(var):
    """Check if a member is an allocatable numeric array (any rank)."""
    if not isallocatable(var) or not isarray(var):
        return False
    typespec = var.get('typespec', '').lower()
    if typespec not in _SIMPLE_SCALAR_TYPESPECS:
        return False
    dims = var.get('dimension', [])
    return len(dims) >= 1 and all(str(d).strip() == ':' for d in dims)


def _get_alloc_ndim(var):
    """Return the rank (number of dimensions) of an allocatable member."""
    return len(var.get('dimension', []))


def _is_pointer_member(var):
    """Check if a member is a pointer numeric array (any rank).

    Fortran pointer components (F2018 7.5.4.6 "Pointer components")
    point to existing data via pointer association (F2018 10.2.2).
    Distinct from allocatables (F2018 7.5.4.7): the wrapper does not
    manage the target memory. Exposed as read-only (getter) from Python
    since pointer assignment from C requires careful lifetime management.
    """
    attrspec = var.get('attrspec', [])
    if 'pointer' not in attrspec:
        return False
    if not isarray(var):
        return False
    typespec = var.get('typespec', '').lower()
    if typespec not in _SIMPLE_SCALAR_TYPESPECS:
        return False
    dims = var.get('dimension', [])
    return len(dims) >= 1 and all(str(d).strip() == ':' for d in dims)


def _get_pointer_ndim(var):
    """Return the rank (number of dimensions) of a pointer member."""
    return len(var.get('dimension', []))


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
        elif _is_deferred_char_member(var):
            continue
        elif _is_allocatable_member(var):
            continue
        elif _is_pointer_member(var):
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
        elif _is_deferred_char_member(var):
            continue
        elif _is_allocatable_member(var):
            continue
        elif _is_pointer_member(var):
            continue
        elif _get_member_ctype(var) is None:
            return False
    return True


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
