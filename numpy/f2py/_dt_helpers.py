"""
Type maps and member inspection helpers for derived type support.

Copyright 2011 -- present NumPy Developers.
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
    outmess,
)
from .crackfortran import _ISO_FORTRAN_ENV_VARS, _fortran_eval

# Map Fortran typespec (+ optional kind) to C types for struct members
_FORTRAN_TO_C = {
    ('real', None): 'float',
    ('real', '4'): 'float',
    ('real', '8'): 'double',
    ('real', 'c_float'): 'float',
    ('real', 'c_double'): 'double',
    ('double precision', None): 'double',
    ('integer', None): 'int',
    ('integer', '2'): 'short',
    ('integer', '4'): 'int',
    ('integer', '8'): 'long long',
    ('integer', 'c_short'): 'short',
    ('integer', 'c_int'): 'int',
    ('integer', 'c_long'): 'long',
    ('integer', 'c_long_long'): 'long long',
    ('logical', None): 'int',
    ('logical', '4'): 'int',
    ('logical', 'c_bool'): 'unsigned char',
    ('complex', None): 'npy_cfloat',
    ('complex', '4'): 'npy_cfloat',
    ('complex', '8'): 'npy_cdouble',
    ('complex', 'c_float_complex'): 'npy_cfloat',
    ('complex', 'c_double_complex'): 'npy_cdouble',
    ('double complex', None): 'npy_cdouble',
}

# Map C type to Python format code for Py_BuildValue / PyArg_Parse
_C_TO_PYFORMAT = {
    'float': 'f',
    'double': 'd',
    'short': 'h',
    'int': 'i',
    'long': 'l',
    'long long': 'L',
    'unsigned char': 'b',
    'npy_cfloat': 'D',
    'npy_cdouble': 'D',
}

# Map C type to PyObject conversion for getters
_C_TO_PYOBJ = {
    'float': 'PyFloat_FromDouble((double){val})',
    'double': 'PyFloat_FromDouble({val})',
    'short': 'PyLong_FromLong((long){val})',
    'int': 'PyLong_FromLong((long){val})',
    'long': 'PyLong_FromLong({val})',
    'long long': 'PyLong_FromLongLong({val})',
    'unsigned char': 'PyBool_FromLong((long){val})',
}

# Map C type to C-from-PyObject conversion for setters
_PYOBJ_TO_C = {
    'float': '(float)PyFloat_AsDouble({obj})',
    'double': 'PyFloat_AsDouble({obj})',
    'short': '(short)PyLong_AsLong({obj})',
    'int': '(int)PyLong_AsLong({obj})',
    'long': 'PyLong_AsLong({obj})',
    'long long': 'PyLong_AsLongLong({obj})',
    'unsigned char': '(unsigned char)PyLong_AsLong({obj})',
}


# Map C type to NumPy type enum for array wrapping
_C_TO_NPY_ENUM = {
    'float': 'NPY_FLOAT',
    'double': 'NPY_DOUBLE',
    'short': 'NPY_SHORT',
    'int': 'NPY_INT',
    'long': 'NPY_LONG',
    'long long': 'NPY_LONGLONG',
    'unsigned char': 'NPY_UBYTE',
    'npy_cfloat': 'NPY_CFLOAT',
    'npy_cdouble': 'NPY_CDOUBLE',
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


def _is_type_parameter(var):
    """Check if a member is a type parameter (KIND or LEN).

    Fortran parameterized derived types (F2018 7.5.3) declare type
    parameters as integer components with 'kind' or 'len' in attrspec:
        integer, kind :: k = kind(0.0d0)
        integer, len :: n
    These are not real data members and must be excluded from wrapping.
    """
    attrspec = var.get('attrspec', [])
    return 'kind' in attrspec or 'len' in attrspec


def _eval_kind_expr(expr):
    """Evaluate a Fortran kind() expression to an integer.

    Handles common patterns:
        kind(0.0)    -> 4  (single precision)
        kind(0.0d0)  -> 8  (double precision)
        kind(0)      -> 4  (default integer)
        integer literal -> itself
        real64, real32 etc. from iso_fortran_env -> resolved values
    """
    import re
    expr = str(expr).strip().lower()

    # Already an integer literal
    try:
        return int(expr)
    except (ValueError, TypeError):
        pass

    if expr in _ISO_FORTRAN_ENV_VARS:
        return int(_ISO_FORTRAN_ENV_VARS[expr]['='])

    # iso_c_binding kinds whose C types are IEEE 754 binary32 and binary64
    # or have the same width on every target of gfortran and LLVM flang;
    # others, such as c_long and c_long_double, depend on the target
    _iso_c_kinds = {
        'c_float': 4, 'c_double': 8, 'c_int': 4, 'c_long_long': 8,
        'c_float_complex': 4, 'c_double_complex': 8,
    }
    if expr in _iso_c_kinds:
        return _iso_c_kinds[expr]

    # kind() goes through crackfortran's evaluator, which reads a
    # literal's exponent or its kind suffix. The old kind() took a
    # source string; this one takes a value, so the whole call is
    # evaluated.
    m = re.fullmatch(r'kind\s*\(\s*(.+?)\s*\)', expr)
    if m:
        try:
            value = _fortran_eval(expr, {}, {})
        except (NameError, SyntaxError, TypeError, ValueError, OverflowError):
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return None


def _resolve_kind_params(typeblock):
    """Identify KIND type parameters and resolve their default values.

    Returns a dict mapping parameter name -> resolved integer kind value.
    Only parameters with resolvable defaults are included. Parameters
    without defaults (no '=' field) or with unresolvable expressions
    are omitted.
    """
    params = {}
    for mname, mvar in get_type_members(typeblock).items():
        if not _is_type_parameter(mvar):
            continue
        attrspec = mvar.get('attrspec', [])
        if 'len' in attrspec:
            # LEN parameters are runtime -- skip for now
            continue
        if 'kind' in attrspec:
            default = mvar.get('=')
            if default is not None:
                resolved = _eval_kind_expr(default)
                if resolved is not None:
                    params[mname.lower()] = resolved
    return params


def _has_len_params(typeblock):
    """Check if a type has any LEN type parameters.

    LEN parameters require runtime allocation and cannot be wrapped
    with the current opaque pointer approach.
    """
    for mvar in get_type_members(typeblock).values():
        attrspec = mvar.get('attrspec', [])
        if 'len' in attrspec:
            return True
    return False


def _has_unresolved_kind_params(typeblock):
    """Check if a type has KIND parameters that cannot be enumerated.

    Returns True if there are KIND type parameters whose dependent
    member typespecs have no enumerable kind values in _FORTRAN_TO_C.
    Types with KIND params (even without defaults) are wrappable as
    long as all kind-parameterized members have enumerable typespecs.
    """
    kind_param_names = set()
    for mname, mvar in get_type_members(typeblock).items():
        attrspec = mvar.get('attrspec', [])
        if 'kind' in attrspec:
            kind_param_names.add(mname.lower())

    if not kind_param_names:
        return False

    # Check that all members referencing kind params have enumerable types
    for mname, mvar in get_type_members(typeblock).items():
        if _is_type_parameter(mvar):
            continue
        if 'kindselector' not in mvar:
            continue
        ks = mvar['kindselector']
        kind_val = ks.get('kind') or ks.get('*')
        if kind_val and str(kind_val).strip().lower() in kind_param_names:
            typespec = mvar.get('typespec', '').lower()
            if not _enumerate_kind_values(typespec):
                return True
    return False


def _resolve_parameterized_type(typeblock):
    """Pre-resolve KIND parameters in a parameterized derived type.

    Mutates the typeblock's vars dict in-place:
    1. Resolves kindselector references to KIND parameters (e.g.
       kindselector={'kind': 'k'} becomes kindselector={'kind': '8'}
       when k has default kind(0.0d0))

    Returns the kind_params dict (parameter name -> int value).
    Does nothing if the type has no KIND parameters.
    """
    kind_params = _resolve_kind_params(typeblock)
    if not kind_params:
        return kind_params

    for mname, mvar in get_type_members(typeblock).items():
        if _is_type_parameter(mvar):
            continue
        # Resolve kindselector references
        if 'kindselector' in mvar:
            ks = mvar['kindselector']
            kind_val = ks.get('kind') or ks.get('*')
            if kind_val:
                kind_key = str(kind_val).strip().lower()
                if kind_key in kind_params:
                    resolved = str(kind_params[kind_key])
                    if 'kind' in ks:
                        ks['kind'] = resolved
                    elif '*' in ks:
                        ks['*'] = resolved
    return kind_params


def _enumerate_kind_values(typespec):
    """Return supported kind integers for a Fortran typespec.

    Scans _FORTRAN_TO_C for integer kind keys associated with the
    given typespec. Returns sorted list of ints.
    """
    values = set()
    for (ts, k), _ in _FORTRAN_TO_C.items():
        if ts == typespec and k is not None:
            try:
                values.add(int(k))
            except (ValueError, TypeError):
                pass
    return sorted(values)


def _get_kind_param_info(typeblock):
    """Return info about KIND type parameters.

    Returns list of dicts:
        [{'name': 'k', 'default': 8, 'has_default': True}]
    """
    result = []
    for mname, mvar in get_type_members(typeblock).items():
        attrspec = mvar.get('attrspec', [])
        if 'kind' not in attrspec:
            continue
        default_expr = mvar.get('=')
        default_val = None
        has_default = False
        if default_expr is not None:
            resolved = _eval_kind_expr(default_expr)
            if resolved is not None:
                default_val = resolved
                has_default = True
        result.append({
            'name': mname.lower(),
            'default': default_val,
            'has_default': has_default,
        })
    return result


def _get_len_param_info(typeblock):
    """Return info about LEN type parameters.

    Returns list of dicts:
        [{'name': 'n', 'default': None, 'has_default': False}]
    """
    result = []
    for mname, mvar in get_type_members(typeblock).items():
        attrspec = mvar.get('attrspec', [])
        if 'len' not in attrspec:
            continue
        default_expr = mvar.get('=')
        default_val = None
        has_default = default_expr is not None
        if has_default:
            try:
                default_val = int(default_expr)
            except (ValueError, TypeError):
                default_val = None
                has_default = False
        result.append({
            'name': mname.lower(),
            'default': default_val,
            'has_default': has_default,
        })
    return result


def _enumerate_specializations(typeblock):
    """Return all kind specializations to generate for a PDT.

    Returns list of (kind_dict, suffix) tuples. For RealVec(k) with
    default k=8 and real(k) members: [({'k': 4}, '_k4'), ({'k': 8}, '_k8')].

    Returns empty list for non-parameterized types.
    """
    kind_info = _get_kind_param_info(typeblock)
    if not kind_info:
        return []

    # Find which typespecs are parameterized by each KIND param
    param_typespecs = {}
    for mname, mvar in get_type_members(typeblock).items():
        if _is_type_parameter(mvar):
            continue
        if 'kindselector' not in mvar:
            continue
        ks = mvar['kindselector']
        kind_ref = ks.get('kind') or ks.get('*')
        if not kind_ref:
            continue
        kind_key = str(kind_ref).strip().lower()
        for ki in kind_info:
            if ki['name'] == kind_key:
                typespec = mvar.get('typespec', '').lower()
                if typespec not in param_typespecs.get(kind_key, set()):
                    param_typespecs.setdefault(kind_key, set()).add(typespec)

    # For single KIND param, enumerate kind values from the typespecs
    # For now, support single KIND param only (most common case)
    if len(kind_info) == 1:
        ki = kind_info[0]
        pname = ki['name']
        typespecs = param_typespecs.get(pname, set())
        if not typespecs:
            # KIND param not referenced by any member -- just use default
            if ki['has_default']:
                return [({pname: ki['default']},
                         f'_k{ki["default"]}')]
            return []
        # Intersect supported kind values across all parameterized typespecs
        kind_sets = [set(_enumerate_kind_values(ts)) for ts in typespecs]
        common_kinds = kind_sets[0]
        for ks in kind_sets[1:]:
            common_kinds &= ks
        if not common_kinds:
            return []
        specs = []
        for kv in sorted(common_kinds):
            specs.append(({pname: kv}, f'_{pname}{kv}'))
        return specs

    # Multiple KIND params: generate cross-product
    import itertools
    param_values = []
    param_names = []
    for ki in kind_info:
        pname = ki['name']
        param_names.append(pname)
        typespecs = param_typespecs.get(pname, set())
        if typespecs:
            kind_sets = [set(_enumerate_kind_values(ts))
                         for ts in typespecs]
            common = kind_sets[0]
            for ks in kind_sets[1:]:
                common &= ks
            param_values.append(sorted(common))
        elif ki['has_default']:
            param_values.append([ki['default']])
        else:
            return []  # unreferenced param without default

    specs = []
    for combo in itertools.product(*param_values):
        kind_dict = dict(zip(param_names, combo))
        suffix = '_' + '_'.join(f'{n}{v}' for n, v in
                                zip(param_names, combo))
        specs.append((kind_dict, suffix))
    return specs


def _resolve_type_for_kind(typeblock, kind_dict):
    """Resolve a typeblock copy for specific KIND values.

    Deep-copies the typeblock and resolves kindselector references
    using kind_dict values instead of defaults. Returns the copy.
    """
    import copy
    tb = copy.deepcopy(typeblock)
    for mname, mvar in get_type_members(tb).items():
        if _is_type_parameter(mvar):
            continue
        if 'kindselector' not in mvar:
            continue
        ks = mvar['kindselector']
        kind_val = ks.get('kind') or ks.get('*')
        if kind_val:
            kind_key = str(kind_val).strip().lower()
            if kind_key in kind_dict:
                resolved = str(kind_dict[kind_key])
                if 'kind' in ks:
                    ks['kind'] = resolved
                elif '*' in ks:
                    ks['*'] = resolved
    return tb


def _get_fortran_type_spec(typeblock, kind_override=None):
    """Return the Fortran type specifier string for wrapper generation.

    For non-parameterized types, returns just the type name, e.g.
    'Point'. For parameterized types with resolved KIND parameters,
    returns the name with kind values, e.g. 'RealVec(8)'.

    If kind_override is provided (dict of param_name -> int), uses
    those values instead of defaults.
    """
    typename = typeblock['name']
    if kind_override:
        # Use override values
        param_values = []
        for mname, mvar in get_type_members(typeblock).items():
            if not _is_type_parameter(mvar):
                continue
            attrspec = mvar.get('attrspec', [])
            if 'kind' in attrspec:
                val = kind_override.get(mname.lower())
                if val is not None:
                    param_values.append(str(val))
            elif 'len' in attrspec:
                # LEN params use variable name in Fortran spec
                param_values.append(f'{mname.lower()}={mname.lower()}')
        if param_values:
            return f'{typename}({", ".join(param_values)})'
        return typename

    kind_params = _resolve_kind_params(typeblock)
    if not kind_params:
        # Check for kinds without defaults -- use first specialization
        kind_info = _get_kind_param_info(typeblock)
        if kind_info:
            specs = _enumerate_specializations(typeblock)
            if specs:
                # Use the largest kind (last entry, typically double)
                kind_dict = specs[-1][0]
                return _get_fortran_type_spec(typeblock,
                                              kind_override=kind_dict)
        # Check for LEN params
        len_info = _get_len_param_info(typeblock)
        if len_info:
            param_values = [f'{li["name"]}={li["name"]}'
                            for li in len_info]
            return f'{typename}({", ".join(param_values)})'
        return typename
    # Build parameter list in declaration order
    param_values = []
    for mname, mvar in get_type_members(typeblock).items():
        if not _is_type_parameter(mvar):
            continue
        attrspec = mvar.get('attrspec', [])
        if 'kind' in attrspec:
            val = kind_params.get(mname.lower())
            if val is not None:
                param_values.append(str(val))
    if param_values:
        return f'{typename}({", ".join(param_values)})'
    return typename


def _member_kind(var):
    """Return the kind of a member as a lower case string, or None."""
    ks = var.get('kindselector', {})
    if ks.get('kind'):
        return str(ks['kind']).strip().lower()
    star = str(ks.get('*', '')).strip()
    if not star:
        return None
    # TYPE*N gives the storage size, which for COMPLEX is twice the kind
    # (gfortran and LLVM flang kinds are byte sizes)
    if var.get('typespec', '').lower() == 'complex' and star.isdigit():
        return str(int(star) // 2)
    return star.lower()


def _get_member_ctype(var):
    """Get the C type of a type member variable, or None if it has none.

    A kind that does not map to a C type gives None rather than the type of
    the default kind, which would change the values of the member.
    """
    typespec = var.get('typespec', '').lower()
    kind = _member_kind(var)
    result = _FORTRAN_TO_C.get((typespec, kind))
    if result is not None or kind is None:
        return result
    # Try resolving kind expressions (e.g. kind(0.0d0) -> '8')
    resolved = _eval_kind_expr(kind)
    if resolved is None:
        return None
    return _FORTRAN_TO_C.get((typespec, str(resolved)))


def _get_array_dims(var):
    """Get fixed array dimensions as a list of ints, or None if scalar."""
    if not is_fixed_array(var):
        return None
    return [int(d) for d in var['dimension']]


def _is_array_member(var):
    """Check if a member is a fixed-size array."""
    return is_fixed_array(var)


def _is_abstract_type(typeblock):
    """Check if a type block has the 'abstract' attribute.

    Fortran abstract types (F2003 4.5.7) cannot be instantiated directly.
    crackfortran stores type attributes in attrspec on the type block
    and/or in the parent module's vars dict.
    """
    for attr in typeblock.get('attrspec', []):
        if isinstance(attr, str) and attr.lower() == 'abstract':
            return True
    parent = typeblock.get('parent_block')
    if parent and typeblock.get('name'):
        tname = typeblock['name']
        parent_var = parent.get('vars', {}).get(tname, {})
        for attr in parent_var.get('attrspec', []):
            if isinstance(attr, str) and attr.lower() == 'abstract':
                return True
    return False


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
    own_members is the type's own members. Type parameters (KIND/LEN)
    are excluded from both sets.
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
    own_members = {
        name: var for name, var in get_type_members(typeblock).items()
        if not _is_type_parameter(var)
    }
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


def _routine_args(routine):
    """Return the dummy arguments of a routine as it declares them.

    Wrapping a routine adds arguments to it: the result of a function, and
    hidden extents and type tags.  prepare_routine keeps the declared ones.
    """
    return routine.get('f2py_dt_args', routine.get('args', []))


def _module_types(module):
    """Return the derived types of a module that f2py wraps.

    Returns (type_map, gen_order, type_tags): the wrappable type blocks by
    lower case name, the same blocks with every type after the types it
    depends on, and the tag of each type for class(T) dispatch, which is
    its position in gen_order, so that parents have lower tags than their
    extensions.
    """
    type_blocks = _find_derived_types(module)
    type_map = _wrap_types(type_blocks)
    generated = set()
    gen_order = []
    for tb in type_blocks:
        _topo_visit(tb, type_blocks, type_map, generated, gen_order)
    type_tags = {tb['name'].lower(): i
                 for i, tb in enumerate(gen_order, start=1)}
    return type_map, gen_order, type_tags


def _describe_types(module):
    """Describe how routines pass the wrapped derived types of a module.

    Returns a dict from the lower case type names to what the rules for
    derived type arguments and the routine wrappers need, see
    _dt_routines.prepare_routine.
    """
    type_map, gen_order, type_tags = _module_types(module)
    types = {}
    for tb in gen_order:
        name = tb['name'].lower()
        kind_info = _get_kind_param_info(tb)
        parent = _get_extends_parent(tb)
        types[name] = {
            'name': tb['name'],
            'module': module['name'],
            'block': tb,
            'tag': type_tags[name],
            'spec': _get_fortran_type_spec(tb),
            'bindc': _can_wrap_bindc(tb, type_map),
            'abstract': _is_abstract_type(tb),
            'parent': parent.lower() if parent else None,
            'specs': _enumerate_specializations(tb) if kind_info else [],
            'kind_params': kind_info,
            'len_params': _get_len_param_info(tb),
            'type_param_order': [
                n.lower() for n, v in get_type_members(tb).items()
                if _is_type_parameter(v)],
        }
    return types


def _get_char_len(var):
    """Extract fixed character length as int, or None."""
    cs = var.get('charselector') or {}
    char_len = cs.get('len') or cs.get('*')
    if char_len is None:
        # a length that is not specified is 1 (F2018 7.4.4.2)
        return 1
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
    return ctype in ('npy_cfloat', 'npy_cdouble') and not _is_array_member(var)


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

    The CONTIGUOUS attribute (F2018 8.5.7, R738) on pointer components
    is a compiler optimization hint guaranteeing contiguous storage.
    It does not affect wrapping and is silently ignored.
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


def _get_type_extensions(base_name, type_map):
    """Find all types that extend base_name (directly or transitively).

    Returns a list of (typename, typeblock) pairs for all types in
    type_map that are extensions of base_name, including base_name
    itself.  Ordered with base type first, then children.
    """
    base_name = base_name.lower()
    result = []
    if base_name in type_map:
        result.append((base_name, type_map[base_name]))
    for tname, tb in type_map.items():
        if tname == base_name:
            continue
        # Walk up the parent chain to see if this type extends base_name
        current = tname
        for _ in range(10):  # prevent infinite loops
            parent = _get_extends_parent(type_map.get(current, {}))
            if parent is None:
                break
            if parent.lower() == base_name:
                result.append((tname, tb))
                break
            current = parent.lower()
    return result


# F2018 7.3.2.3: type tags for unlimited polymorphic dispatch.
# Intrinsic types get fixed tags; derived types get 100 + type_tag.
_UNLIMITED_POLY_TAGS = {
    'integer_4': 1, 'integer_8': 2,
    'real_4': 3, 'real_8': 4,
    'complex_4': 5, 'complex_8': 6,
    'logical_4': 7,
    'character': 8,
}


def _is_unlimited_polymorphic_arg(var):
    """Check if a variable was declared with CLASS(*) (unlimited polymorphic).

    Per F2018 7.3.2.3 paragraph 4, CLASS(*) accepts any type including
    intrinsic types. crackfortran adds 'unlimited_polymorphic' to attrspec.
    """
    return 'unlimited_polymorphic' in var.get('attrspec', [])


def _is_polymorphic_arg(var):
    """Check if a variable was declared with CLASS(T) (polymorphic).

    Per F2018 7.3.2.3, CLASS declares polymorphic entities that can
    hold any extension type.  crackfortran normalizes class(T) to
    type(T) but adds 'polymorphic' to attrspec.
    """
    return 'polymorphic' in var.get('attrspec', [])


def _is_private_member(var):
    """Check if a component has the PRIVATE accessibility attribute.

    Per F2018 7.5.4.8, components declared in a type with a
    ``PRIVATE`` statement default to private unless individually
    marked ``PUBLIC``.  Private components are not exposed as
    Python properties.
    """
    return 'private' in var.get('attrspec', [])


def _is_coarray_member(var):
    """Check if a member has the codimension (coarray) attribute.

    Fortran coarray components (F2018 7.5.4.3) use CODIMENSION or
    bracket syntax (e.g. ``integer :: x[*]``).
    """
    attrspec = var.get('attrspec', [])
    for attr in attrspec:
        if isinstance(attr, str) and attr.lower().startswith('codimension'):
            return True
    return False


def _get_coarray_compile_flag():
    """Return the Fortran compiler flag needed for coarray support.

    Uses -fcoarray=single (gfortran) which is a zero-dependency mode
    making all coarray syntax valid.  In single-image mode,
    this_image()=1 and num_images()=1.  For multi-image execution,
    the user provides their own runtime flags.

    Returns the flag string, or empty string if detection fails.
    """
    import subprocess
    import tempfile
    test_src = "program t; integer :: x[*]; x = this_image(); end\n"
    # Try gfortran -fcoarray=single
    try:
        with tempfile.NamedTemporaryFile(suffix='.f90', mode='w',
                                         delete=False) as f:
            f.write(test_src)
            f.flush()
            result = subprocess.run(
                ['gfortran', '-fcoarray=single', '-fsyntax-only', f.name],
                capture_output=True, timeout=10)
            if result.returncode == 0:
                return '-fcoarray=single'
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    # ifort/ifx have built-in coarray support (no flag needed for single)
    for compiler in ['ifort', 'ifx']:
        try:
            with tempfile.NamedTemporaryFile(suffix='.f90', mode='w',
                                             delete=False) as f:
                f.write(test_src)
                f.flush()
                result = subprocess.run(
                    [compiler, '-coarray=single', '-syntax-only', f.name],
                    capture_output=True, timeout=10)
                if result.returncode == 0:
                    return '-coarray=single'
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
    return ''


def _has_coarray_members(members):
    """Check if any member in the dict is a coarray component."""
    return any(_is_coarray_member(v) for v in members.values())


def _coarray_as_local(var):
    """Create a copy of a coarray member with codimension stripped.

    On any single Fortran image, a coarray's local data is an ordinary
    array.  Accessing ``obj%grid`` without cosubscripts ``[...]`` gives
    the local partition.  This function returns a modified variable dict
    that can be processed by the regular allocatable/array code paths.
    """
    import copy
    local = copy.deepcopy(var)
    local['attrspec'] = [
        a for a in local.get('attrspec', [])
        if not (isinstance(a, str) and a.lower().startswith('codimension'))
    ]
    return local


def _is_type_member(var):
    """Check if a member is a scalar nested derived type."""
    return (var.get('typespec') == 'type'
            and not _is_array_member(var))


def _is_type_array_member(var):
    """Check if a member is a fixed-size array of derived types."""
    return (var.get('typespec') == 'type'
            and is_fixed_array(var))


def _member_problem(var, len_param_names=()):
    """Return why f2py cannot expose a component to Python, or None.

    Whether f2py wraps the type of a component of derived type is up to
    the caller.
    """
    if _is_coarray_member(var):
        var = _coarray_as_local(var)  # F2018 7.5.4.3: local data
    typespec = var.get('typespec', '').lower()
    attrs = var.get('attrspec', [])
    dims = var.get('dimension', [])
    if typespec == 'type':
        if 'pointer' in attrs or 'allocatable' in attrs:
            return 'it is a pointer or allocatable of derived type'
        if dims and not is_fixed_array(var):
            return 'it is an array of derived type without a constant shape'
        if len(dims) > 1:
            return f'it is an array of derived type of rank {len(dims)}'
        return None
    if typespec == 'character':
        if _is_deferred_char_member(var) or (_is_char_member(var)
                                             and not dims):
            return None
        return 'it is a character array or has no constant length'
    if typespec not in _SIMPLE_SCALAR_TYPESPECS:
        return f'f2py has no Python type for {typespec} components'
    if _get_member_ctype(var) is None:
        return f'f2py has no C type for {typespec}({_member_kind(var)})'
    if _is_allocatable_member(var) or _is_pointer_member(var):
        return None
    if 'pointer' in attrs or 'allocatable' in attrs:
        return 'it is a pointer or allocatable scalar'
    if dims and not (is_fixed_array(var)
                     or _is_len_sized_array(var, len_param_names)):
        return 'its shape is not constant'
    return None


def _settle_members(typeblock, checked, type_map, settle):
    """Hide the components of ``typeblock`` that f2py cannot expose.

    ``checked`` is the type block with its KIND parameters resolved.  A
    component of a derived type that is not wrapped yet makes this return
    False, unless ``settle`` is true, which hides such components too.
    Hidden components get no accessors and are left out of
    get_type_members; as for PRIVATE components, the type stays wrapped
    and routines still take its objects.
    """
    hidden = typeblock.setdefault('f2py_hidden', {})
    len_param_names = {li['name'] for li in _get_len_param_info(checked)}
    waiting = False
    for name, var in checked.get('vars', {}).items():
        if (name in hidden or _is_type_parameter(var)
                or _is_private_member(var)):
            continue
        problem = _member_problem(var, len_param_names)
        if problem is None and var.get('typespec') == 'type':
            inner = var.get('typename', '').lower()
            if inner not in (type_map or {}):
                if not settle:
                    waiting = True
                    continue
                problem = f'f2py does not wrap its type {inner}'
        if problem:
            hidden[name] = problem
            outmess(f'\t\tf2py does not expose component {name} of type '
                    f'{typeblock["name"]}, as {problem}\n')
    return not waiting


def _has_coarray_ultimate(typeblock, type_map, seen=()):
    """Check if a type has a coarray ultimate component, directly or in a
    nonpointer nonallocatable component of derived type."""
    for var in typeblock.get('vars', {}).values():
        if _is_coarray_member(var):
            return True
        inner = type_map.get(var.get('typename', '').lower())
        attrs = var.get('attrspec', [])
        if (var.get('typespec') == 'type' and inner is not None
                and inner['name'] not in seen and 'pointer' not in attrs
                and 'allocatable' not in attrs
                and _has_coarray_ultimate(inner, type_map,
                                          (*seen, typeblock['name']))):
            return True
    return False


def _unwrapped_reason(typeblock, type_map):
    """Return why _wrap_types leaves a type unwrapped."""
    parent = _get_extends_parent(typeblock)
    if _has_coarray_ultimate(typeblock, type_map):
        return ('an object with a coarray component cannot be a pointer '
                '(F2018 C825)')
    if parent and parent.lower() not in type_map:
        return f'it extends {parent}, which f2py does not wrap'
    return 'f2py cannot enumerate the values of its KIND parameters'


def _wrap_types(type_blocks):
    """Return the type blocks f2py wraps, by lower case name.

    A type waits until the types of its components are wrapped.  When all
    remaining types wait, the first one that can be wrapped with those
    components hidden is wrapped so, and the rest wait again.  As the
    type of a component that is neither a pointer nor allocatable must be
    defined before it (F2018 7.5.4.1), and other components of derived
    type are hidden anyway, this hides only components whose types f2py
    cannot wrap.
    """
    type_map = {}
    remaining = list(type_blocks)
    while remaining:
        for settle in (False, True):
            tb = next((tb for tb in remaining
                       if _can_wrap_bindc(tb, type_map)
                       or _can_wrap_opaque(tb, type_map, settle)
                       or _can_wrap_abstract(tb, type_map, settle)), None)
            if tb is not None:
                break
        else:
            break
        type_map[tb['name'].lower()] = tb
        remaining.remove(tb)
    for tb in remaining:
        if not tb.get('f2py_unwrapped'):
            tb['f2py_unwrapped'] = _unwrapped_reason(tb, type_map)
            outmess(f'\t\tf2py does not wrap derived type {tb["name"]}, '
                    f'as {tb["f2py_unwrapped"]}\n')
    return type_map


def _can_wrap_bindc(typeblock, type_map=None):
    """Check if a bind(c) type can be wrapped as a direct C struct.

    type_map is a dict of already-known wrappable types, used to
    validate nested type members.  A C struct needs every component, so
    a type with a component f2py cannot expose is wrapped as an opaque
    pointer instead.
    """
    if not isbindctype(typeblock) or typeblock.get('f2py_hidden'):
        return False
    if _has_len_params(typeblock):
        return False
    if _has_unresolved_kind_params(typeblock):
        return False
    # Resolve KIND parameters before wrappability check
    _resolve_parameterized_type(typeblock)
    if not is_simple_derived_type(typeblock):
        return False
    for name, var in get_type_members(typeblock).items():
        if _is_type_parameter(var):
            continue
        # the struct has no C type for characters, which opaque types handle
        if _member_problem(var) is not None or _is_char_member(var):
            return False
        if var.get('typespec') == 'type':
            inner = var.get('typename', '').lower()
            if type_map is None or inner not in type_map:
                return False
    return True


def _can_wrap_abstract(typeblock, type_map=None, settle=False):
    """Check if an abstract type can be wrapped as a skeleton base type.

    Abstract types (F2003 4.5.7) cannot be instantiated, but their
    concrete extensions need the abstract parent's PyTypeObject for
    tp_base (isinstance() support).  See _settle_members for ``settle``.
    """
    if not _is_abstract_type(typeblock):
        return False
    if isbindctype(typeblock):
        return False
    if _has_len_params(typeblock):
        return False
    if (_has_unresolved_kind_params(typeblock)
            or _has_coarray_ultimate(typeblock, type_map or {})):
        return False
    _resolve_parameterized_type(typeblock)
    return _settle_members(typeblock, typeblock, type_map, settle)


def _is_len_sized_array(var, len_param_names):
    """Check if a member is an array sized by a LEN type parameter.

    E.g. real :: data(n) where n is a LEN param. These are automatic
    arrays with runtime-determined shape, not allocatable.
    """
    if not isarray(var):
        return False
    if isallocatable(var):
        return False
    dims = var.get('dimension', [])
    return any(str(d).strip().lower() in len_param_names for d in dims)


def _can_wrap_opaque(typeblock, type_map=None, settle=False):
    """Check if a type can be wrapped via opaque pointers.

    type_map is a dict of already-known wrappable types, used to
    validate nested type members and extends(parent) dependencies.
    Abstract types are excluded (handled by _can_wrap_abstract), and so
    are bind(c) types that _can_wrap_bindc wraps as C structs.  See
    _settle_members for ``settle``.

    Parameterized types with KIND parameters are supported (with or
    without defaults). LEN parameters are also supported when all
    LEN-sized members are simple numeric arrays.
    """
    if _is_abstract_type(typeblock):
        return False
    if isbindctype(typeblock) and _can_wrap_bindc(typeblock, type_map):
        return False
    if (_has_unresolved_kind_params(typeblock)
            or _has_coarray_ultimate(typeblock, type_map or {})):
        return False
    # Resolve KIND parameters before wrappability check
    # Use default specialization if available
    kind_info = _get_kind_param_info(typeblock)
    specs = _enumerate_specializations(typeblock)
    checked = typeblock
    if specs:
        # Use first specialization (default or smallest kind) for check
        checked = _resolve_type_for_kind(typeblock, specs[0][0])
    elif kind_info:
        # Has KIND params but no enumerable specializations
        return False
    else:
        _resolve_parameterized_type(typeblock)
    # Check extends parent is already wrappable
    parent_name = _get_extends_parent(typeblock)
    if parent_name:
        if type_map is None or parent_name not in type_map:
            return False
    return _settle_members(typeblock, checked, type_map, settle)


# The iso_c_binding type of the wrapper arguments that carry each C type;
# default logical travels as int, see _member_value and _member_assign
_C_TO_ISOC = {
    'float': 'real(c_float)',
    'double': 'real(c_double)',
    'short': 'integer(c_short)',
    'int': 'integer(c_int)',
    'long': 'integer(c_long)',
    'long long': 'integer(c_long_long)',
    'unsigned char': 'logical(c_bool)',
    'npy_cfloat': 'complex(c_float_complex)',
    'npy_cdouble': 'complex(c_double_complex)',
}


def _get_member_isoc_type(var):
    """Get iso_c_binding type for a member variable."""
    return _C_TO_ISOC.get(_get_member_ctype(var))


def _is_int_logical(var):
    """Check if a logical member reaches C as an int, which needs a
    conversion in both directions."""
    return (var.get('typespec', '').lower() == 'logical'
            and _get_member_ctype(var) == 'int')


def _member_value(var, ref):
    """Return the value of the member ``ref`` in its wrapper argument type."""
    if _is_int_logical(var):
        return f'merge(1_c_int, 0_c_int, {ref})'
    return ref


def _member_actual(var, name):
    """Return the actual argument for a dummy declared like var, from the
    wrapper argument ``name``."""
    if _is_int_logical(var):
        return f'{name} /= 0'
    return name


def _member_assign(var, ref, value):
    """Return the statement that stores a wrapper argument in a member."""
    if _is_int_logical(var):
        return f'{ref} = {value} /= 0'
    return f'{ref} = {value}'


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
