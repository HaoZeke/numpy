"""Match canonical PDT operator identities and propagate native status."""

from ._dt_helpers import _C_TO_PYOBJ, _fortran_sym, _get_member_ctype
from ._dt_operator_handles import _operator_handle_signature


def support(typename):
    return f'''extern npy_int64 f2py_pdt_identity(void *);
static void
f2py_op_{typename}_release_owner(PyObject *owner)
{{
    PyObject *exception = PyErr_GetRaisedException();
    Py_DECREF(owner);
    if (PyErr_Occurred()) PyErr_WriteUnraisable(Py_None);
    PyErr_SetRaisedException(exception);
}}

static int
f2py_op_{typename}_pdt_match(PyObject *object,
                            PyObject *(*owner)(PyObject *), npy_int64 expected)
{{
    PyObject *capsule = owner(object);
    if (capsule == NULL) return -1;
    const char *name = PyCapsule_GetName(capsule);
    void *pointer = name == NULL ? NULL : PyCapsule_GetPointer(capsule, name);
    int matches = pointer != NULL && f2py_pdt_identity(pointer) == expected;
    f2py_op_{typename}_release_owner(capsule);
    return PyErr_Occurred() ? -1 : matches;
}}
'''


def match_expression(typename, entries, proc, type_map, reverse, legacy):
    from ._dt_operators import _operator_kind, _operator_preferred, _routine_args

    signature = _operator_handle_signature(proc, type_map)
    if not signature['active']:
        return legacy(typename, entries, proc, reverse)
    checks = []
    for index, (name, info) in enumerate(zip(_routine_args(proc), signature['args'])):
        var = proc['vars'][name]
        obj = (('right', 'left') if reverse else ('left', 'right'))[index]
        if _operator_kind(var) == 't':
            native = var['typename'].lower()
            check = f'PyObject_TypeCheck({obj}, &Py{native}_Type)'
            if info:
                check += (f' && (f2py_op_{typename}_pdt_match({obj}, '
                          f'f2py_op_{native}_owner, '
                          f'{info["description"]["identity_number"]}LL) > 0)')
            checks.append('(' + check + ')')
        else:
            ctype = _get_member_ctype(var)
            preferred = int(ctype == _operator_preferred(entries, proc, index))
            checks.append(f'(f2py_op_{typename}_scalar_match({obj}, '
                          f"'{_operator_kind(var)}', sizeof({ctype}), {preferred}) > 0)")
    return ' && '.join(checks)


def convert(typename, proc, type_map, reverse, legacy):
    from ._dt_operators import _routine_args

    signature = _operator_handle_signature(proc, type_map)
    lines, arguments, cleanup = legacy(proc, reverse)
    if not signature['active']:
        return lines, arguments, cleanup
    for index, (name, info) in enumerate(zip(_routine_args(proc), signature['args'])):
        if not info:
            continue
        obj = (('right', 'left') if reverse else ('left', 'right'))[index]
        native = proc['vars'][name]['typename'].lower()
        owner = f'f2py_owner{index}'
        marker = f'        void *f2py_arg{index} = PyCapsule_GetPointer('
        start = lines.index(marker)
        assert lines[start + 1] == f'            {owner}, "f2py.{native}");'
        write = int(info['pointer'] or not info['intent_in'])
        lines[start:start + 2] = [f'        void *f2py_arg{index} = f2py_{native}_checked(',
                                  f'            {obj}, "{info["capsule"]}", {write});']
    for source in (lines, cleanup):
        for index, line in enumerate(source):
            if 'Py_DECREF(f2py_owner' in line:
                source[index] = line.replace('Py_DECREF(', f'f2py_op_{typename}_release_owner(')
    return lines, arguments, cleanup


def capsule_destructor(wrapper, capsule):
    return f'''static void
{wrapper}_release_preserving_error(void *pointer)
{{
    PyObject *exception = PyErr_GetRaisedException();
    if (pointer != NULL) {wrapper}_free(pointer);
    if (PyErr_Occurred()) PyErr_WriteUnraisable(Py_None);
    PyErr_SetRaisedException(exception);
}}

static void
{wrapper}_capsule_destructor(PyObject *capsule)
{{
    PyObject *exception = PyErr_GetRaisedException();
    void *pointer = PyCapsule_GetPointer(capsule, "{capsule}");
    if (pointer != NULL) {wrapper}_free(pointer);
    if (PyErr_Occurred()) PyErr_WriteUnraisable(capsule);
    PyErr_SetRaisedException(exception);
}}
'''


def invoke(typename, entry, type_map, reverse, legacy_convert):
    from ._dt_operators import _operator_kind

    pname, proc, result_kind = entry
    signature = _operator_handle_signature(proc, type_map)
    wrapper = f'f2py_op_{_fortran_sym(typename)}_{pname}'
    lines, arguments, cleanup = convert(typename, proc, type_map, reverse, legacy_convert)
    result = proc['vars'].get(proc.get('result', proc['name']), {})
    arguments += ', &f2py_native_status'
    lines.append('        int f2py_native_status = 4;')
    ctype = ('int' if result_kind in ('assignment', 'logical') else
             'void *' if result_kind == 'type' else _get_member_ctype(result))
    lines.append(f'        {ctype} value = {wrapper}({arguments});')
    lines += ['        if (PyErr_Occurred() || f2py_native_status != 0) {']
    if result_kind == 'type':
        lines.append(f'            if (value != NULL) {wrapper}_release_preserving_error(value);')
    lines += ['    ' + line for line in cleanup]
    lines += ['            if (!PyErr_Occurred()) {',
              '                if (f2py_native_status == 1) PyErr_NoMemory();',
              '                else PyErr_SetString(PyExc_ValueError,',
              '                    "invalid derived operator argument or native selection");',
              '            }', '            return NULL;', '        }']
    if result_kind == 'assignment':
        lines += [*cleanup, '        if (PyErr_Occurred()) return NULL;',
                  '        if (value != 0) {',
                  '            PyErr_SetString(PyExc_ValueError, "native assignment failed");',
                  '            return NULL;', '        }', '        return Py_NewRef(left);']
    elif result_kind == 'type':
        info = signature['result']
        lines += ['        if (value == NULL) {',
                  *['    ' + line for line in cleanup],
                  '            return PyErr_NoMemory();', '        }']
        if info:
            lines += [f'        if (f2py_pdt_identity(value) != {info["description"]["identity_number"]}LL) {{',
                      '            PyErr_SetString(PyExc_TypeError, "native operator result has wrong KIND identity");',
                      f'            {wrapper}_release_preserving_error(value);',
                      *['    ' + line for line in cleanup], '            return NULL;', '        }']
        native = result['typename'].lower()
        lines += [f'        PyObject *result = f2py_op_{native}_adopt(',
                  f'            value, {wrapper}_capsule_destructor);',
                  f'        if (result == NULL) {wrapper}_release_preserving_error(value);',
                  *cleanup, '        return result;']
    elif result_kind == 'logical':
        lines += [*cleanup, '        if (PyErr_Occurred()) return NULL;',
                  '        return PyBool_FromLong(value);']
    else:
        if _operator_kind(result) == 'c':
            suffix = 'f' if ctype == 'npy_cfloat' else ''
            returned = f'PyComplex_FromDoubles(npy_creal{suffix}(value), npy_cimag{suffix}(value))'
        else:
            returned = _C_TO_PYOBJ[ctype].format(val='value')
        lines += [*cleanup, '        if (PyErr_Occurred()) return NULL;',
                  f'        return {returned};']
    return lines
