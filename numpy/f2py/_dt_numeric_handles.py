"""Copy intrinsic PDT components without exporting module storage addresses."""

import hashlib

from ._dt_handles import _handle_description, _handle_operation_parts
from ._dt_helpers import (
    _C_TO_NPY_ENUM,
    _coarray_as_local,
    _fortran_sym,
    _get_alloc_ndim,
    _get_member_ctype,
    _get_member_isoc_type,
    _get_pointer_ndim,
    _is_allocatable_member,
    _is_array_member,
    _is_char_member,
    _is_coarray_member,
    _is_deferred_char_member,
    _is_pointer_member,
    _is_private_member,
    _is_type_array_member,
    _is_type_member,
    _member_value,
)


def _numeric_root(owner, symbol, member):
    identity = owner.lower() + ':' + symbol.lower() + ':' + member.lower()
    return 'f2py_pna_' + hashlib.sha256(identity.encode('ascii')).hexdigest()[:24]


def _numeric_array(var):
    """Return normalized metadata for an exposed intrinsic array component."""
    if _is_coarray_member(var):
        var = _coarray_as_local(var)
    if (_is_private_member(var) or _is_type_member(var)
            or _is_type_array_member(var) or _is_char_member(var)
            or _is_deferred_char_member(var)):
        return None
    allocatable = _is_allocatable_member(var)
    pointer = _is_pointer_member(var)
    if not (_is_array_member(var) or allocatable or pointer):
        return None
    rank = len(var.get('dimension', []))
    if not rank:
        rank = (_get_alloc_ndim(var) if allocatable
                else _get_pointer_ndim(var) if pointer else 0)
    ctype = _get_member_ctype(var)
    native_type = _get_member_isoc_type(var)
    if not rank or ctype not in _C_TO_NPY_ENUM or native_type is None:
        return None
    return dict(var=var, rank=rank, allocatable=allocatable, pointer=pointer,
                native_type=native_type, npy_enum=_C_TO_NPY_ENUM[ctype])


def _numeric_fortran(typename, type_spec, block, members):
    """Query and copy arrays through the selected actual Fortran object."""
    description = _handle_description(block, type_spec)
    if description is None:
        return [], set()
    lines = []
    removed = set()
    suffix = typename[len(block['name']):]
    symbol = _fortran_sym(block.get('f2py_c_name', block['name']) + suffix)
    for name, var in members.items():
        info = _numeric_array(var)
        if info is None:
            continue
        rank = info['rank']
        root = _numeric_root(description['owner'], symbol, name)
        removed.update([f'f2py_get_{typename}_{name}', f'f2py_get_{typename}_{name}_data'])
        available = ('allocated' if info['allocatable'] else
                     'associated' if info['pointer'] else None)
        reference = 'object%' + name
        query = ['status = 1', 'dims = 0', 'available = 0']
        if available:
            query += [f'if (.not. {available}({reference})) then',
                      '  status = 0', '  return', 'end if']
        query += [f'dims = shape({reference}, kind=c_size_t)',
                  'available = 1', 'status = 0']
        copy = ['status = 1']
        if available:
            copy += [f'if (.not. {available}({reference})) return']
        copy += [f'if (any(shape({reference}, kind=c_size_t) /= dims)) return',
                 f'if (size({reference}, kind=c_size_t) > 0_c_size_t) then',
                 '  call c_f_pointer(buffer, values, dims)',
                 '  values = ' + _member_value(info['var'], reference),
                 'end if', 'status = 0']
        for operation, body in [('shape', query), ('copy', copy)]:
            uses, declarations, body = _handle_operation_parts(
                [], [f'type({description["pointer_spec"]}), pointer :: object'], body,
                {'object': {'handle': 'handle', 'description': description}},
                failure=('status = 1', 'return'))
            args = 'handle, dims, available, status' if operation == 'shape' else (
                'handle, dims, buffer, status')
            intent = 'out' if operation == 'shape' else 'in'
            header = [f'subroutine {root}_{operation}({args}) bind(c)', *uses,
                      'type(c_ptr), value :: handle',
                      f'integer(c_size_t), intent({intent}) :: dims({rank})',
                      'integer(c_int), intent(out) :: status']
            if operation == 'shape':
                header.append('integer(c_int), intent(out) :: available')
            else:
                header += ['type(c_ptr), value :: buffer',
                           f'{info["native_type"]}, pointer :: values(' +
                           ','.join(':' for _ in range(rank)) + ')']
            lines += [*header, *declarations, *body, f'end subroutine {root}_{operation}', '']
    return lines, removed


def _numeric_getter(typename, root, name, info, capsule):
    rank = info['rank']
    return f'''extern void {root}_shape(void *, size_t *, int *, int *);
extern void {root}_copy(void *, const size_t *, void *, int *);
static PyObject *
Py{typename}_get_{name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    void *pointer = f2py_{typename}_checked(selfobj, "{capsule}", 0);
    if (pointer == NULL) return NULL;
    PyObject *owner = Py_NewRef(self->capsule);
    size_t native_dims[{rank}];
    npy_intp dims[{rank}], total;
    int available = 0, status = 1;
    {root}_shape(pointer, native_dims, &available, &status);
    if (status != 0) {{
        Py_DECREF(owner);
        PyErr_SetString(PyExc_ReferenceError, "native component selection is invalid");
        return NULL;
    }}
    if (!available) {{
        Py_DECREF(owner);
        Py_RETURN_NONE;
    }}
    for (int axis = 0; axis < {rank}; axis++) {{
        if (native_dims[axis] > (size_t)NPY_MAX_INTP) {{
            Py_DECREF(owner);
            PyErr_SetString(PyExc_OverflowError, "native component shape exceeds NumPy index range");
            return NULL;
        }}
        dims[axis] = (npy_intp)native_dims[axis];
    }}
    if (!f2py_dt_shape_product({rank}, dims, &total)) {{
        Py_DECREF(owner);
        return NULL;
    }}
    PyObject *array = PyArray_EMPTY({rank}, dims, {info['npy_enum']}, 1);
    if (array == NULL) {{
        Py_DECREF(owner);
        return NULL;
    }}
    {root}_copy(pointer, native_dims, PyArray_DATA((PyArrayObject *)array), &status);
    Py_DECREF(owner);
    if (status != 0) {{
        Py_DECREF(array);
        PyErr_SetString(PyExc_ReferenceError, "native component changes during array creation");
        return NULL;
    }}
    return array;
}}
'''


def _numeric_getters(functions, typename, symbol, members, capsule, type_map):
    """Replace only intrinsic array getters, retaining all generated setters."""
    replacements = {}
    block = type_map[typename.lower()]
    owner = block.get('f2py_origin_module', block.get('parent_block', {}).get('name'))
    if not owner:
        raise ValueError('a PDT numeric component needs its declaring module')
    for name, var in members.items():
        info = _numeric_array(var)
        if info is not None:
            marker = f'Py{typename}_get_{name}(PyObject *selfobj, void *closure)'
            root = _numeric_root(owner, symbol, name)
            replacements[marker] = _numeric_getter(typename, root, name, info, capsule)
    result = []
    for source in functions:
        selected = [replacement for marker, replacement in replacements.items()
                    if marker in source]
        if len(selected) > 1:
            raise ValueError('numeric getter source contains multiple component entries')
        result.append(selected[0] if selected else source)
    return result


def _drop_numeric_address_getters(source, names):
    """Remove generated address getters replaced by query/copy interfaces."""
    import re

    for name in sorted(names):
        start = re.compile(r'(?im)^[ \t]*function[ \t]+' + re.escape(name) + r'\b')
        end = re.compile(r'(?im)^[ \t]*end[ \t]+function[ \t]+' + re.escape(name)
                         + r'[ \t]*(?:\n|$)')
        starts = list(start.finditer(source))
        if not starts:
            continue
        if len(starts) != 1:
            raise ValueError('duplicate generated component address getter')
        stop = end.search(source, starts[0].end())
        if stop is None:
            raise ValueError('unterminated generated component address getter')
        source = source[:starts[0].start()] + source[stop.end():]
    return source
