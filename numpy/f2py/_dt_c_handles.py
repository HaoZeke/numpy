"""C ownership and native metadata for boxed derived handles."""

import hashlib
import re


def _handle_c_support(typename, len_info, identities):
    """Keep returned and module-bound objects in the native kind/LEN context."""
    lengths = '\n'.join(
        f'''    int length_{index} = f2py_pdt_length(pointer, {index});
    if (length_{index} < 0) {{
        PyErr_SetString(PyExc_RuntimeError, "invalid native LEN metadata");
        return -1;
    }}
    self->len_{item['name']} = length_{index};'''
        for index, item in enumerate(len_info or []))
    identity_cases = '\n'.join(
        f'    case {identity}LL: return "{name}";'
        for identity, name in identities)
    code = f'''extern void f2py_pdt_destroy(void *);
extern int f2py_pdt_validate(void *);
extern int f2py_pdt_storage(void *);
extern int f2py_pdt_kind(void *);
extern int f2py_pdt_length(void *, int);
extern npy_int64 f2py_pdt_identity(void *);
extern int f2py_pdt_writable(void *);

static void
f2py_{typename}_release_preserving_error(void *pointer)
{{
    PyObject *error_type, *error_value, *error_traceback;
    PyErr_Fetch(&error_type, &error_value, &error_traceback);
    f2py_pdt_destroy(pointer);
    if (PyErr_Occurred()) PyErr_WriteUnraisable(Py_None);
    PyErr_Restore(error_type, error_value, error_traceback);
}}

static const char *
f2py_{typename}_capsule_name(void *pointer)
{{
    switch (f2py_pdt_identity(pointer)) {{
{identity_cases}
    default:
        PyErr_SetString(PyExc_TypeError, "native handle has the wrong type identity");
        return NULL;
    }}
}}

static int
f2py_{typename}_hydrate(PyObject *object, void *pointer)
{{
    Py{typename}Object *self = (Py{typename}Object *)object;
    if (f2py_{typename}_capsule_name(pointer) == NULL) return -1;
    int status = f2py_pdt_validate(pointer);
    if (status != 0) {{
        PyErr_Format(PyExc_ReferenceError,
                     "native module-variable selection is invalid (status=%d)",
                     status);
        return -1;
    }}
    self->kind_value = f2py_pdt_kind(pointer);
{lengths}
    return 0;
}}

static void *
f2py_{typename}_checked(PyObject *object, const char *name, int write)
{{
    Py{typename}Object *self = (Py{typename}Object *)object;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError, "object not initialized");
        return NULL;
    }}
    void *pointer = PyCapsule_GetPointer(self->capsule, name);
    if (pointer == NULL) return NULL;
    if (f2py_{typename}_hydrate(object, pointer) < 0) return NULL;
    if (write && !f2py_pdt_writable(pointer)) {{
        PyErr_SetString(PyExc_AttributeError, "Fortran binding is read-only");
        return NULL;
    }}
    return pointer;
}}

static int
f2py_{typename}_require_writable(void *pointer)
{{
    if (f2py_{typename}_capsule_name(pointer) == NULL) return 0;
    int status = f2py_pdt_validate(pointer);
    if (status != 0) {{
        PyErr_SetString(PyExc_ReferenceError, "native module-variable selection is invalid");
        return 0;
    }}
    if (!f2py_pdt_writable(pointer)) {{
        PyErr_SetString(PyExc_AttributeError, "Fortran binding is read-only");
        return 0;
    }}
    return 1;
}}

/* The handle is owned on entry; owner is retained only by its capsule. */
static PyObject *
f2py_{typename}_from_handle_as(void *pointer, const char *name,
                              PyCapsule_Destructor destroy, PyObject *owner,
                              PyTypeObject *type)
{{
    if (pointer == NULL) return PyErr_NoMemory();
    if (!PyType_Check((PyObject *)type) || !PyType_IsSubtype(type, &Py{typename}_Type)) {{
        PyErr_SetString(PyExc_TypeError, "class is not a subtype of the native derived type");
        f2py_{typename}_release_preserving_error(pointer);
        return NULL;
    }}
    const char *actual_name = f2py_{typename}_capsule_name(pointer);
    if (actual_name == NULL || (name != NULL && strcmp(name, actual_name) != 0)) {{
        if (actual_name != NULL)
            PyErr_SetString(PyExc_TypeError, "capsule name disagrees with native type identity");
        f2py_{typename}_release_preserving_error(pointer);
        return NULL;
    }}
    name = actual_name;
    Py{typename}Object *self = (Py{typename}Object *)Py{typename}_tp_new(
        type, NULL, NULL);
    if (self == NULL) {{
        f2py_{typename}_release_preserving_error(pointer);
        return NULL;
    }}
    self->capsule = PyCapsule_New(pointer, name, destroy);
    if (self->capsule == NULL) {{
        f2py_{typename}_release_preserving_error(pointer);
        Py_DECREF(self);
        return NULL;
    }}
    if (owner != NULL) {{
        Py_INCREF(owner);
        if (PyCapsule_SetContext(self->capsule, owner) < 0) {{
            Py_DECREF(owner);
            Py_DECREF(self);
            return NULL;
        }}
    }}
    if (f2py_{typename}_hydrate((PyObject *)self, pointer) < 0) {{
        Py_DECREF(self);
        return NULL;
    }}
    return (PyObject *)self;
}}

static PyObject *
f2py_{typename}_from_handle(void *pointer, const char *name,
                           PyCapsule_Destructor destroy, PyObject *owner)
{{
    return f2py_{typename}_from_handle_as(pointer, name, destroy, owner,
                                        &Py{typename}_Type);
}}
'''
    registration = [
        '\t{',
        f'\t\tPyObject *metadata = PyCapsule_New((void *)f2py_{typename}_hydrate,',
        '\t\t    "f2py.native_metadata", NULL);',
        '\t\tif (metadata == NULL) return -1;',
        f'\t\tint status = PyDict_SetItemString(Py{typename}_Type.tp_dict,',
        '\t\t    "_f2py_native_metadata", metadata);',
        '\t\tPy_DECREF(metadata);',
        '\t\tif (status < 0) return -1;',
        '\t}',
        '\t{',
        f'\t\tPyObject *metadata = PyCapsule_New((void *)f2py_{typename}_require_writable,',
        '\t\t    "f2py.native_writable", NULL);',
        '\t\tif (metadata == NULL) return -1;',
        f'\t\tint status = PyDict_SetItemString(Py{typename}_Type.tp_dict,',
        '\t\t    "_f2py_native_writable", metadata);',
        '\t\tPy_DECREF(metadata);',
        '\t\tif (status < 0) return -1;',
        '\t}',
    ]
    return code, registration


def _handle_operator_adopt(typename):
    """Keep the caller's failure ownership while hydrating an operator result."""
    return f'''static PyObject *
f2py_op_{typename}_adopt(void *pointer, PyCapsule_Destructor destroy)
{{
    const char *name = f2py_{typename}_capsule_name(pointer);
    if (name == NULL) return NULL;
    Py{typename}Object *self = (Py{typename}Object *)Py{typename}_tp_new(
        &Py{typename}_Type, NULL, NULL);
    if (self == NULL) return NULL;
    self->capsule = PyCapsule_New(pointer, name, destroy);
    if (self->capsule == NULL) {{
        Py_DECREF(self);
        return NULL;
    }}
    if (f2py_{typename}_hydrate((PyObject *)self, pointer) < 0) {{
        PyCapsule_SetDestructor(self->capsule, NULL);
        Py_DECREF(self);
        return NULL;
    }}
    return (PyObject *)self;
}}
'''


def _guard_boxed_accessors(functions, typename):
    """Refresh exact descriptor metadata at each generated capsule read."""
    result = []
    pattern = re.compile(r'(?m)^static (?:PyObject\s*\*|int)\n\w+\([^;]*?\)\n\{')
    for text in functions:
        starts = [match.start() for match in pattern.finditer(text)] + [len(text)]
        chunks = [text[:starts[0]]] if len(starts) > 1 else [text]
        for begin, end in zip(starts, starts[1:]):
            chunk = text[begin:end]
            signature = chunk.partition('{')[0]
            write = int('PyObject *value' in signature)
            chunk = re.sub(
                r'PyCapsule_GetPointer\(self->capsule, ("[^"\n]*")\)',
                lambda match: (f'f2py_{typename}_checked(selfobj, '
                               f'{match[1]}, {write})'), chunk)
            chunks.append(chunk)
        result.append(''.join(chunks))
    return result


def _handle_capsule_destructor(typename):
    """Release descriptor boxes without reconstructing LEN from C arguments."""
    return f'''extern void f2py_pdt_destroy(void *);
static void
f2py_{typename}_capsule_destructor(PyObject *capsule)
{{
    PyObject *error_type, *error_value, *error_traceback;
    PyErr_Fetch(&error_type, &error_value, &error_traceback);
    const char *name = PyCapsule_GetName(capsule);
    if (name != NULL) {{
        void *pointer = PyCapsule_GetPointer(capsule, name);
        if (pointer != NULL) {{
            void *lengths = PyCapsule_GetContext(capsule);
            if (!PyErr_Occurred()) {{
                f2py_pdt_destroy(pointer);
                free(lengths);
            }}
        }}
    }}
    if (PyErr_Occurred()) PyErr_WriteUnraisable(capsule);
    PyErr_Restore(error_type, error_value, error_traceback);
}}
'''


_hydrate_c_api = '''
typedef int (*f2py_dt_metadata_func)(PyObject *, void *);
typedef int (*f2py_dt_writable_func)(void *);
static PyObject *
f2py_dt_metadata(PyTypeObject *type, const char *key)
{
    PyObject *mro = type->tp_mro;
    if (mro == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "native object type has no method resolution order");
        return NULL;
    }
    for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(mro); index++) {
        PyTypeObject *base = (PyTypeObject *)PyTuple_GET_ITEM(mro, index);
        if (base->tp_dict == NULL) continue;
        PyObject *capsule = PyDict_GetItemString(base->tp_dict, key);
        if (capsule != NULL || PyErr_Occurred()) return capsule;
    }
    return NULL;
}

static int
f2py_dt_hydrate(PyObject *object, PyTypeObject *type, void *pointer)
{
    (void)type;
    PyObject *capsule = f2py_dt_metadata(Py_TYPE(object), "_f2py_native_metadata");
    if (capsule == NULL) return PyErr_Occurred() == NULL;
    void *raw = PyCapsule_GetPointer(capsule, "f2py.native_metadata");
    if (raw == NULL) return 0;
    f2py_dt_metadata_func hydrate = (f2py_dt_metadata_func)raw;
    return hydrate(object, pointer) == 0;
}

static int
f2py_dt_require_writable(PyTypeObject *type, void *pointer)
{
    PyObject *capsule = f2py_dt_metadata(type, "_f2py_native_writable");
    if (capsule == NULL) return PyErr_Occurred() == NULL;
    void *raw = PyCapsule_GetPointer(capsule, "f2py.native_writable");
    if (raw == NULL) return 0;
    return ((f2py_dt_writable_func)raw)(pointer);
}
'''


def _rename_c_functions(code, names):
    """Rename generated function identifiers, leaving literals untouched."""
    result = []
    position = 0
    while position < len(code):
        if code.startswith('//', position):
            end = code.find('\n', position)
            end = len(code) if end < 0 else end
        elif code.startswith('/*', position):
            end = code.find('*/', position + 2)
            end = len(code) if end < 0 else end + 2
        elif code[position] in "\"'":
            quote = code[position]
            end = position + 1
            while end < len(code):
                if code[end] == '\\':
                    end += 2
                elif code[end] == quote:
                    end += 1
                    break
                else:
                    end += 1
        elif code[position].isalpha() or code[position] == '_':
            end = position + 1
            while end < len(code) and (code[end].isalnum() or code[end] == '_'):
                end += 1
            token = code[position:end]
            result.append(names.get(token, token))
            position = end
            continue
        else:
            end = position + 1
        result.append(code[position:end])
        position = end
    return ''.join(result)


def _variant_functions(code, identity):
    """Give one specialization's C functions private collision-free names."""
    token = hashlib.sha256(str(identity).encode('ascii')).hexdigest()[:24]
    defined = re.findall(r'(?m)^static[^\n;={}]*\n(\w+)\s*\(', code)
    names = {name: 'f2py_variant_' + token + '_' + name for name in defined}
    return _rename_c_functions(code, names), names


def _handle_getset_multi(typename, specializations, kind_info, len_info,
                         type_map, one):
    """Dispatch every generated component through its exact native identity."""
    from ._dt_handles import _handle_description

    block = type_map[typename.lower()]
    variants = []
    code = []
    attributes = {}
    entry_pattern = re.compile(
        r'\{"(?P<name>\w+)",\s*(?P<getter>\w+),\s*(?P<setter>\w+),')
    for suffix, kinds, members in specializations:
        description = _handle_description(block, kind_override=kinds)
        functions, entries = one(
            typename, members, len_info=len_info, type_map=type_map,
            symbol_override=typename + suffix,
            capsule_override='f2py.' + typename + suffix,
            return_parts=True, boxed=True)
        rendered, renamed = _variant_functions('\n'.join(functions), description['identity'])
        code.append(rendered)
        variant = dict(description=description, kinds=kinds, members={})
        for entry in entries:
            match = entry_pattern.search(entry)
            if match is None:
                raise ValueError('unrecognized generated getset entry')
            record = dict(name=match['name'],
                          getter=renamed.get(match['getter'], match['getter']),
                          setter=renamed.get(match['setter'], match['setter']))
            variant['members'][record['name']] = record
            attributes.setdefault(record['name'], []).append(record)
        variants.append(variant)

    def dispatch(name, operation):
        setter = operation == 'setter'
        arguments = ('PyObject *selfobj, PyObject *value, void *closure' if setter
                     else 'PyObject *selfobj, void *closure')
        returned = 'int' if setter else 'PyObject *'
        error = '-1' if setter else 'NULL'
        actual = 'selfobj, value, closure' if setter else 'selfobj, closure'
        function = f'Py{typename}_' + ('set_' if setter else 'get_') + name
        lines = [f'static {returned}', f'{function}({arguments})', '{',
                 f'    Py{typename}Object *self = (Py{typename}Object *)selfobj;',
                 '    if (self->capsule == NULL) {',
                 '        PyErr_SetString(PyExc_RuntimeError, "object not initialized");',
                 f'        return {error};', '    }',
                 '    const char *name = PyCapsule_GetName(self->capsule);',
                 f'    if (name == NULL) return {error};',
                 '    void *pointer = PyCapsule_GetPointer(self->capsule, name);',
                 f'    if (pointer == NULL) return {error};',
                 f'    if (f2py_{typename}_hydrate(selfobj, pointer) < 0) return {error};',
                 '    switch (f2py_pdt_identity(pointer)) {']
        for variant in variants:
            member = variant['members'].get(name)
            if member and member[operation] != 'NULL':
                lines += [f"    case {variant['description']['identity_number']}LL:",
                          f'        return {member[operation]}({actual});']
        lines += ['    default:',
                  '        PyErr_SetString(PyExc_AttributeError,',
                  f'                        "{name} is unavailable for this KIND specialization");',
                  f'        return {error};', '    }', '}']
        return '\n'.join(lines)

    entries = []
    for name, members in attributes.items():
        getter = f'Py{typename}_get_{name}' if any(
            item['getter'] != 'NULL' for item in members) else 'NULL'
        setter = f'Py{typename}_set_{name}' if any(
            item['setter'] != 'NULL' for item in members) else 'NULL'
        if getter != 'NULL':
            code.append(dispatch(name, 'getter'))
        if setter != 'NULL':
            code.append(dispatch(name, 'setter'))
        entries.append(f'    {{"{name}", {getter}, {setter}, "{name}", NULL}},')
    for parameter in kind_info:
        name = parameter['name']
        function = f'Py{typename}_get_kind_{name}'
        lines = ['static PyObject *', f'{function}(PyObject *selfobj, void *closure)', '{',
                 f'    Py{typename}Object *self = (Py{typename}Object *)selfobj;',
                 '    if (self->capsule == NULL) {',
                 '        PyErr_SetString(PyExc_RuntimeError, "object not initialized");',
                 '        return NULL;', '    }',
                 '    const char *name = PyCapsule_GetName(self->capsule);',
                 '    if (name == NULL) return NULL;',
                 '    void *pointer = PyCapsule_GetPointer(self->capsule, name);',
                 '    if (pointer == NULL) return NULL;',
                 '    switch (f2py_pdt_identity(pointer)) {']
        for variant in variants:
            lines += [f"    case {variant['description']['identity_number']}LL:",
                      f"        return PyLong_FromLong({variant['kinds'][name]});"]
        lines += ['    default:',
                  '        PyErr_SetString(PyExc_RuntimeError, "invalid native KIND metadata");',
                  '        return NULL;', '    }', '}']
        code.append('\n'.join(lines))
        entries.append(f'    {{"{name}", {function}, NULL, "KIND parameter", NULL}},')
    code.append(f'static PyGetSetDef Py{typename}_getset[] = {{\n'
                + '\n'.join(entries) + '\n    {NULL}\n};\n')
    return '\n'.join(code)


def _handle_init_multi(typename, specializations, kind_info, len_info, one):
    """Parse KIND choices, then use the matching ordinary constructor."""
    code = []
    choices = []
    for index, parameter in enumerate(kind_info):
        name = parameter['name']
        default = parameter['default']
        if not parameter['has_default']:
            default = -1
        choices.append((name, f'f2py_kind_{index}', default,
                        parameter['has_default']))
    variants = []
    for suffix, kinds, members in specializations:
        variant = one(
            typename, members, len_info=len_info,
            symbol_override=typename + suffix,
            capsule_override='f2py.' + typename + suffix, boxed=True)
        rendered, names = _variant_functions(variant, sorted(kinds.items()))
        code.append(rendered)
        variants.append((kinds, names[f'Py{typename}_tp_init']))
    lines = ['static int',
             f'Py{typename}_tp_init(PyObject *selfobj, PyObject *args, PyObject *kwds)',
             '{', '    Py_ssize_t provided = PyTuple_GET_SIZE(args);',
             f'    Py_ssize_t consumed = provided < {len(choices)} ? provided : {len(choices)};',
             '    PyObject *member_args = PyTuple_GetSlice(args, consumed, provided);',
             '    if (member_args == NULL) return -1;',
             '    PyObject *member_kwds = kwds == NULL ? PyDict_New() : PyDict_Copy(kwds);',
             '    if (member_kwds == NULL) {',
             '        Py_DECREF(member_args);', '        return -1;', '    }',
             '    int status = -1;']
    for index, (name, variable, default, has_default) in enumerate(choices):
        lines += [f'    long {variable} = {default};',
                  f'    PyObject *value_{index} = PyDict_GetItemString(member_kwds, "{name}");',
                  f'    if (provided > {index}) {{',
                  f'        if (value_{index} != NULL) {{',
                  f'            PyErr_SetString(PyExc_TypeError, "multiple values for {name}");',
                  '            goto finish;', '        }',
                  f'        value_{index} = PyTuple_GET_ITEM(args, {index});',
                  '    }',
                  f'    if (value_{index} != NULL) {{',
                  f'        {variable} = PyLong_AsLong(value_{index});',
                  f'        if ({variable} == -1 && PyErr_Occurred()) goto finish;',
                  f'        if ({variable} < INT_MIN || {variable} > INT_MAX) {{',
                  '            PyErr_SetString(PyExc_OverflowError, "KIND does not fit a C int");',
                  '            goto finish;', '        }',
                  f'        if (provided <= {index}',
                  f'                && PyDict_DelItemString(member_kwds, "{name}") < 0) goto finish;',
                  '    }']
        if not has_default:
            lines += [f'    if ({variable} == -1) {{',
                      '        PyErr_SetString(PyExc_TypeError,',
                      f'                        "{typename}() requires \'{name}=\' keyword argument");',
                      '        goto finish;', '    }']
    for kinds, function in variants:
        condition = ' && '.join(variable + ' == ' + str(kinds[name])
                                for name, variable, _, _ in choices)
        lines += [f'    if ({condition}) {{',
                  f'        status = {function}(selfobj, member_args, member_kwds);',
                  '        if (status == 0) {',
                  f'            Py{typename}Object *self = (Py{typename}Object *)selfobj;',
                  '            const char *name = PyCapsule_GetName(self->capsule);',
                  '            void *pointer = name == NULL ? NULL : PyCapsule_GetPointer(self->capsule, name);',
                  '            if (pointer == NULL) status = -1;',
                  f'            else status = f2py_{typename}_hydrate(selfobj, pointer);',
                  '        }', '        goto finish;', '    }']
    if len(choices) == 1:
        name, variable, _, _ = choices[0]
        valid = ', '.join(str(kinds[name]) for _, kinds, _ in specializations)
        lines += ['    PyErr_Format(PyExc_ValueError,',
                  f'                 "unsupported {name}=%ld for {typename} (valid: {valid})", {variable});']
    else:
        lines += [f'    PyErr_SetString(PyExc_ValueError, "unsupported KIND tuple for {typename}");']
    lines += ['finish:', '    Py_DECREF(member_args);', '    Py_DECREF(member_kwds);',
              '    return status;', '}']
    code.append('\n'.join(lines))
    return '\n'.join(code)
