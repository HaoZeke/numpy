"""Generate shape-preserving accessors for arrays of derived components."""

from ._dt_helpers import _get_array_dims, _get_fortran_type_spec


def record_array_c(typename, name, var, type_map, *, bindc=False,
                   symbol=None, len_call=''):
    """Return C accessors and their getset entry for a record component."""
    from ._dt_helpers import _can_wrap_bindc

    inner = var['typename'].lower()
    inner_c = _can_wrap_bindc(type_map[inner], type_map)
    rank = len(var['dimension'])
    symbol = symbol or typename.lower()
    attrs = var.get('attrspec', [])
    alloc = 'allocatable' in attrs
    pointer = 'pointer' in attrs
    dims = _get_array_dims(var)
    fixed = dims is not None
    if bindc:
        load_parent = f'''    f2py_{typename}_t *data = (f2py_{typename}_t *)
        PyCapsule_GetPointer(self->capsule, "f2py.{typename}");
    if (data == NULL) return NULL;'''
        shape = ', '.join(str(d) for d in dims)
        get_shape = f'    npy_intp dims[{rank}] = {{{shape}}};'
        get_element = f'&((f2py_{inner}_t *)data->{name})[i]'
    else:
        load_parent = f'''    void *ptr = PyCapsule_GetPointer(self->capsule,
                                           "f2py.{typename}");
    if (ptr == NULL) return NULL;'''
        get_shape = f'''    if (!f2py_get_{symbol}_{name}_available(ptr{len_call}))
    {{
        Py_DECREF(owner);
        Py_RETURN_NONE;
    }}
    int shape[{rank}];
    f2py_get_{symbol}_{name}_shape(ptr{len_call}, shape);
    npy_intp dims[{rank}];
    for (int d = 0; d < {rank}; d++) dims[d] = shape[d];'''
        get_element = f'f2py_get_{symbol}_{name}(ptr{len_call}, (int)(i + 1))'
    if inner_c:
        make_copy = f'''        f2py_{inner}_t *owned = PyMem_Malloc(sizeof(*owned));
        if (owned == NULL) {{
            Py_DECREF(result);
            Py_DECREF(owner);
            return PyErr_NoMemory();
        }}
        memcpy(owned, source, sizeof(*owned));'''
        release = 'PyMem_Free(owned);'
    else:
        make_copy = '        void *owned = source;'
        release = f'f2py_destroy_{inner}(owned);'
    put = ('        PyList_SET_ITEM(result, i, item);' if rank == 1 else
           '''        PyObject **objects = (PyObject **)PyArray_DATA(
            (PyArrayObject *)result);
        Py_XSETREF(objects[i], item);''')
    create = ('PyList_New(total)' if rank == 1 else
              f'PyArray_EMPTY({rank}, dims, NPY_OBJECT, 1)')
    get = f'''
static PyObject *
Py{typename}_get_{name}(PyObject *selfobj, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError, "{typename} object not initialized");
        return NULL;
    }}
    PyObject *owner = Py_NewRef(self->capsule);
{load_parent.replace('self->capsule', 'owner').replace('return NULL;', '{ Py_DECREF(owner); return NULL; }')}

{get_shape}
    npy_intp total;
    if (!f2py_dt_shape_product({rank}, dims, &total)) {{
        Py_DECREF(owner);
        return NULL;
    }}
    if ((npy_intp)(int)total != total) {{
        PyErr_SetString(PyExc_OverflowError, "{name} exceeds the wrapper index range");
        Py_DECREF(owner);
        return NULL;
    }}
    PyObject *result = {create};
    if (result == NULL) {{
        Py_DECREF(owner);
        return NULL;
    }}
    for (npy_intp i = 0; i < total; i++) {{
        void *source = {get_element};
        if (source == NULL) {{
            Py_DECREF(result);
            Py_DECREF(owner);
            PyErr_SetString(PyExc_RuntimeError, "{name} has no element storage");
            return NULL;
        }}
{make_copy}
        PyObject *item = Py{inner}_tp_new(&Py{inner}_Type, NULL, NULL);
        if (item == NULL) {{
            {release}
            Py_DECREF(result);
            Py_DECREF(owner);
            return NULL;
        }}
        ((Py{inner}Object *)item)->capsule = PyCapsule_New(
            owned, "f2py.{inner}", f2py_{inner}_capsule_destructor);
        if (((Py{inner}Object *)item)->capsule == NULL) {{
            {release}
            Py_DECREF(item);
            Py_DECREF(result);
            Py_DECREF(owner);
            return NULL;
        }}
{put}
    }}
    Py_DECREF(owner);
    return result;
}}
'''
    entry = (f'    {{"{name}", Py{typename}_get_{name}, '
             f'Py{typename}_set_{name}, "{name} member", NULL}},')
    if pointer:
        setter = f'''
static int
Py{typename}_set_{name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    PyErr_SetString(PyExc_AttributeError,
                    "{name} is a Fortran pointer member (read-only)");
    return -1;
}}
'''
        return get + setter, entry
    parent_set = load_parent.replace('return NULL;', 'return -1;')
    if bindc:
        refresh_parent = f"""    data = (f2py_{typename}_t *)PyCapsule_GetPointer(
        self->capsule, "f2py.{typename}");
    if (data == NULL) {{
        Py_DECREF(items);
        return -1;
    }}"""
    else:
        refresh_parent = f"""    ptr = PyCapsule_GetPointer(self->capsule,
                                   "f2py.{typename}");
    if (ptr == NULL) {{
        Py_DECREF(items);
        return -1;
    }}"""
    if fixed:
        shape = ', '.join(str(d) for d in dims)
        shape_check = f'''    const npy_intp expected[{rank}] = {{{shape}}};
    for (int d = 0; d < {rank}; d++) {{
        if (PyArray_DIM(items, d) != expected[d]) {{
            Py_DECREF(items);
            PyErr_SetString(PyExc_ValueError, "{name} has an incorrect shape");
            return -1;
        }}
    }}'''
    elif not alloc:
        shape_check = f"""    int expected[{rank}];
    f2py_get_{symbol}_{name}_shape(ptr{len_call}, expected);
    for (int d = 0; d < {rank}; d++) {{
        if (PyArray_DIM(items, d) != expected[d]) {{
            Py_DECREF(items);
            PyErr_SetString(PyExc_ValueError, "{name} has an incorrect shape");
            return -1;
        }}
    }}"""
    else:
        shape_check = ''
    resize = ''
    none = ''
    if alloc:
        none = f'''    if (value == Py_None) {{
        f2py_clear_{symbol}_{name}(ptr{len_call});
        return 0;
    }}'''
        dims_call = ', '.join(f'(int)PyArray_DIM(items, {d})' for d in range(rank))
        resize = f'    f2py_resize_{symbol}_{name}(ptr{len_call}, {dims_call});'
    if bindc:
        set_element = f'''        memcpy(&((f2py_{inner}_t *)data->{name})[i], source,
               sizeof(f2py_{inner}_t));'''
    else:
        set_element = f'        f2py_set_{symbol}_{name}(ptr{len_call}, (int)(i + 1), source);'
    setter = f'''
static int
Py{typename}_set_{name}(PyObject *selfobj, PyObject *value, void *closure)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (value == NULL) {{
        PyErr_SetString(PyExc_TypeError, "Cannot delete {name} attribute");
        return -1;
    }}
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError, "{typename} object not initialized");
        return -1;
    }}
{parent_set}
{none}
    PyArrayObject *items = (PyArrayObject *)PyArray_FROM_OTF(
        value, NPY_OBJECT, NPY_ARRAY_FARRAY_RO);
    if (items == NULL) return -1;
    if (!f2py_dt_checked_shape(items, {rank})) {{
        Py_DECREF(items);
        return -1;
    }}
{refresh_parent}
    if (PyArray_NDIM(items) != {rank}) {{
        Py_DECREF(items);
        PyErr_SetString(PyExc_ValueError, "{name} has an incorrect rank");
        return -1;
    }}
{shape_check}
    npy_intp total = PyArray_SIZE(items);
    if ((npy_intp)(int)total != total) {{
        Py_DECREF(items);
        PyErr_SetString(PyExc_OverflowError, "{name} exceeds the wrapper index range");
        return -1;
    }}
    PyObject **objects = (PyObject **)PyArray_DATA(items);
    for (npy_intp i = 0; i < total; i++) {{
        if (!Py_IS_TYPE(objects[i], &Py{inner}_Type) ||
                !PyCapsule_IsValid(((Py{inner}Object *)objects[i])->capsule,
                                  "f2py.{inner}")) {{
            Py_DECREF(items);
            PyErr_SetString(PyExc_TypeError, "{name} needs initialized {inner} objects");
            return -1;
        }}
    }}
{resize}
    for (npy_intp i = 0; i < total; i++) {{
        void *source = PyCapsule_GetPointer(
            ((Py{inner}Object *)objects[i])->capsule, "f2py.{inner}");
{set_element}
    }}
    Py_DECREF(items);
    return 0;
}}
'''
    return get + setter, entry


def record_array_fortran(typename, type_spec, name, var, type_map,
                         len_prefix='', len_decl=''):
    """Return interoperable shape and element accessors for an opaque type."""
    from ._dt_helpers import _can_wrap_bindc

    inner = var['typename'].lower()
    rank = len(var['dimension'])
    attrs = var.get('attrspec', [])
    alloc = 'allocatable' in attrs
    pointer = 'pointer' in attrs
    made = f'allocated(obj%{name})' if alloc else (
        f'associated(obj%{name})' if pointer else '.true.')
    inner_c = _can_wrap_bindc(type_map[inner], type_map)
    inner_block = type_map[inner]
    inner_spec = _get_fortran_type_spec(inner_block)
    native_name = inner_block['name']
    inner_spec = inner + inner_spec[len(native_name):]
    common = ['    type(c_ptr), value :: cptr']
    if len_decl:
        common.extend(len_decl.splitlines())
    obj_decl = f'    type({type_spec}), pointer :: obj'
    root = f'f2py_get_{typename}_{name}'
    lines = [f'  function {root}_available(cptr{len_prefix}) result(made) bind(c)',
             *common, '    logical(c_bool) :: made', obj_decl,
             '    call c_f_pointer(cptr, obj)', f'    made = {made}',
             f'  end function {root}_available', '']
    lines += [f'  subroutine {root}_shape(cptr{len_prefix}, dims) bind(c)',
              *common, f'    integer(c_int) :: dims({rank})', obj_decl,
              '    call c_f_pointer(cptr, obj)', '    dims = 0',
              f'    if ({made}) then']
    lines += [f'      dims({d}) = size(obj%{name}, {d})' for d in range(1, rank + 1)]
    lines += ['    end if', f'  end subroutine {root}_shape', '']
    indices = []
    for d in range(1, rank + 1):
        stride = ' * '.join(f'max(size(obj%{name}, {k}), 1)' for k in range(1, d)) or '1'
        indices.append(f'lbound(obj%{name}, {d}) + mod((idx - 1) / ({stride}), '
                       f'max(size(obj%{name}, {d}), 1))')
    element = f'obj%{name}(' + ', &\n      & '.join(indices) + ')'
    lines += [f'  function {root}(cptr{len_prefix}, idx) result(inner_cptr) bind(c)',
              *common, '    integer(c_int), value :: idx',
              '    type(c_ptr) :: inner_cptr', obj_decl]
    if not inner_c:
        lines.append(f'    type({inner_spec}), pointer :: inner_obj')
    lines.append('    call c_f_pointer(cptr, obj)')
    if inner_c:
        lines.append(f'    inner_cptr = c_loc({element})')
    else:
        lines += [f'    allocate(inner_obj, source={element})',
                  '    inner_cptr = c_loc(inner_obj)']
    lines += [f'  end function {root}', '']
    if pointer:
        return lines
    set_root = f'f2py_set_{typename}_{name}'
    lines += [f'  subroutine {set_root}(cptr{len_prefix}, idx, inner_cptr) bind(c)',
              *common, '    integer(c_int), value :: idx',
              '    type(c_ptr), value :: inner_cptr', obj_decl,
              f'    type({inner_spec}), pointer :: inner_obj',
              '    call c_f_pointer(cptr, obj)',
              '    call c_f_pointer(inner_cptr, inner_obj)',
              f'    {element} = inner_obj', f'  end subroutine {set_root}', '']
    if alloc:
        clear = f'f2py_clear_{typename}_{name}'
        resize = f'f2py_resize_{typename}_{name}'
        args = ', '.join(f'n{d}' for d in range(1, rank + 1))
        lines += [f'  subroutine {clear}(cptr{len_prefix}) bind(c)', *common,
                  obj_decl, '    call c_f_pointer(cptr, obj)',
                  f'    if (allocated(obj%{name})) deallocate(obj%{name})',
                  f'  end subroutine {clear}', '',
                  f'  subroutine {resize}(cptr{len_prefix}, {args}) bind(c)',
                  *common, f'    integer(c_int), value :: {args}', obj_decl,
                  '    call c_f_pointer(cptr, obj)',
                  f'    if (allocated(obj%{name})) then',
                  f'      if (all(shape(obj%{name}) == [{args}])) return',
                  f'      deallocate(obj%{name})', '    end if',
                  f'    allocate(obj%{name}({args}))',
                  f'  end subroutine {resize}', '']
    return lines


def record_array_declarations(symbol, name, var, len_extra=''):
    """Return C declarations for the opaque component accessors."""
    rank = len(var['dimension'])
    attrs = var.get('attrspec', [])
    decls = [f'extern unsigned char f2py_get_{symbol}_{name}_available(void *{len_extra});',
             f'extern void f2py_get_{symbol}_{name}_shape(void *{len_extra}, int *);',
             f'extern void *f2py_get_{symbol}_{name}(void *{len_extra}, int);']
    if 'pointer' not in attrs:
        decls.append(f'extern void f2py_set_{symbol}_{name}(void *{len_extra}, int, void *);')
    if 'allocatable' in attrs:
        dims = ', int' * rank
        decls += [f'extern void f2py_resize_{symbol}_{name}(void *{len_extra}{dims});',
                  f'extern void f2py_clear_{symbol}_{name}(void *{len_extra});']
    return decls
