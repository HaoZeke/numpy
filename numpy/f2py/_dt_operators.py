"""Generate derived-type operator dispatch and matching Fortran lifetimes.

Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.
"""

from ._dt_helpers import (
    _C_TO_PYOBJ,
    _FORTRAN_ARITH_OPS,
    _FORTRAN_CMP_OPS,
    _fortran_sym,
    _get_member_ctype,
    _get_member_isoc_type,
    _member_actual,
    _member_value,
    _routine_args,
)
from .auxfuncs import hasbody, isarray, isfunction, isroutine


def _operator_kind(var):
    if var.get('typespec') == 'type':
        return 't'
    if var.get('typespec') == 'logical':
        return 'b'
    ctype = _get_member_ctype(var)
    if ctype in ('short', 'int', 'long', 'long long'):
        return 'i'
    if ctype in ('float', 'double'):
        return 'f'
    if ctype in ('npy_cfloat', 'npy_cdouble'):
        return 'c'
    raise NotImplementedError('unsupported derived-type operator scalar')


def _operator_signature(proc, type_map):
    args = [proc['vars'][a] for a in _routine_args(proc)]
    result = (proc['vars'].get(proc.get('result', proc['name']), {})
              if isfunction(proc) else args[0])
    for var in [*args, result]:
        if _operator_is_array(var) or 'optional' in var.get('attrspec', []):
            raise NotImplementedError(
                'derived-type operators require scalar, required operands')
        kind = _operator_kind(var)
        if kind == 't':
            if var['typename'].lower() not in type_map:
                raise NotImplementedError(
                    'derived-type operator references an unwrapped type')
        elif _get_member_isoc_type(var) is None:
            raise NotImplementedError('unsupported operator scalar ABI')
    return args, result


def _operator_is_array(var):
    return (isarray(var) or 'dimension' in var
            or any(attr.lower().startswith('dimension(')
                   for attr in var.get('attrspec', [])))


def _module_symbol_private(module, name):
    """Resolve module accessibility, including the default PRIVATE statement."""
    variables = {key.lower().replace(' ', ''): value
                 for key, value in module.get('vars', {}).items()}
    attrs = variables.get(name.lower().replace(' ', ''), {}).get('attrspec', [])
    if 'public' in attrs:
        return False
    if 'private' in attrs:
        return True
    return 'private' in variables.get('', {}).get('attrspec', [])


def _operator_actual_call(proc, pname, arguments):
    """Use an accessible generic when its specific implementation is private."""
    op = proc.get('f2py_operator_generic')
    if op == '=':
        return arguments[0] + ' = (' + arguments[1] + ')'
    if op is not None:
        if len(arguments) == 1:
            return '(' + op + ' (' + arguments[0] + '))'
        return '((' + arguments[0] + ') ' + op + ' (' + arguments[1] + '))'
    return pname + '(' + ', '.join(arguments) + ')'


def _operator_assignment_call(proc, pname, arguments):
    call = _operator_actual_call(proc, pname, arguments)
    return call if proc.get('f2py_operator_generic') == '=' else 'call ' + call


def _scan_operator_interfaces(module_block, typename, type_map):
    """Find supported scalar operator signatures involving a wrapped type."""
    if not hasbody(module_block):
        return {}
    routines = {b['name'].lower(): b for b in module_block['body']
                if isroutine(b)}
    native_name = type_map[typename.lower()].get(
        'f2py_native_name', typename).lower()
    ops = {}
    for block in module_block['body']:
        name = block.get('name', '').lower().replace(' ', '')
        assignment = name == 'assignment(=)'
        if (block.get('block') != 'interface'
                or not (name.startswith('operator(') or assignment)):
            continue
        if _module_symbol_private(module_block, name):
            continue
        op = '=' if assignment else name[len('operator('):-1]
        if (not assignment and op not in _FORTRAN_ARITH_OPS
                and op not in _FORTRAN_CMP_OPS):
            continue
        seen = set()
        for pname in block.get('implementedby', []):
            pname = pname.lower()
            proc = routines.get(pname)
            if (proc is None or pname in seen
                    or isfunction(proc) == assignment):
                continue
            args = _routine_args(proc)
            involved = args[:1] if assignment else args
            if not any(proc['vars'].get(a, {}).get('typespec') == 'type'
                       and proc['vars'][a].get('typename', '').lower()
                       in (typename.lower(), native_name) for a in involved):
                continue
            # Array procedures retain their ordinary module-call interface.
            # A record's Python number slots receive scalar operands.
            values = [proc['vars'][a] for a in args]
            if isfunction(proc):
                values.append(proc['vars'].get(proc.get('result', proc['name']), {}))
            if any(_operator_is_array(var) for var in values):
                continue
            if len(args) not in (1, 2) or (len(args) == 1 and op not in ('+', '-')):
                raise ValueError('invalid derived-type operator arity')
            _, result = _operator_signature(proc, type_map)
            result_kind = ('assignment' if assignment
                           else 'type' if result.get('typespec') == 'type'
                           else 'logical' if result.get('typespec') == 'logical'
                           else 'scalar')
            if op in _FORTRAN_CMP_OPS and result_kind != 'logical':
                raise ValueError('comparison operator must return logical')
            if _module_symbol_private(module_block, pname):
                proc = dict(proc, f2py_operator_generic=op)
            ops.setdefault(op, []).append((pname, proc, result_kind))
            seen.add(pname)
    return ops


def _operator_entries(ops):
    seen = set()
    for proc_list in ops.values():
        for entry in proc_list:
            if entry[0] not in seen:
                seen.add(entry[0])
                yield entry


def _gen_operator_fortran_wrappers(typename, ops, type_map):
    """Emit one ABI wrapper and matching deallocator per implementation."""
    lines = []
    for pname, proc, result_kind in _operator_entries(ops):
        args, result = _operator_signature(proc, type_map)
        wrapper = f'f2py_op_{typename}_{pname}'
        reserved = {pname, *(v.get('typename', '').lower()
                             for v in [*args, result])}
        prefix = 'f2py_op'
        suffixes = ['result', 'value', 'status', 'handle', 'rhs',
                    *(f'arg{i}' for i in range(len(args))),
                    *(f'operand{i}' for i in range(len(args)))]
        while any(f'{prefix}_{suffix}' in reserved for suffix in suffixes):
            prefix += 'x'
        dummy = [f'{prefix}_arg{i}' for i in range(len(args))]
        actual_names = [f'{prefix}_operand{i}' if _operator_kind(v) == 't'
                        else dummy[i] for i, v in enumerate(args)]
        value = f'{prefix}_value'
        returned = f'{prefix}_result'
        status = f'{prefix}_status'
        handle = f'{prefix}_handle'
        lines += [(f'  function {wrapper}({", ".join(dummy)}) '
                   f'result({returned}) bind(c)')]
        alias_copy = (result_kind == 'assignment'
                      and _operator_kind(args[1]) == 't'
                      and args[0]['typename'].lower() == args[1]['typename'].lower())
        rhs = f'{prefix}_rhs'
        if result_kind == 'assignment':
            lines += [f'    integer(c_int) :: {returned}']
            if alias_copy:
                lines += [f'    type({args[1]["typename"]}), pointer :: {rhs}']
        elif result_kind == 'type':
            lines += [f'    type(c_ptr) :: {returned}',
                      f'    type({result["typename"]}), pointer :: {value}',
                      f'    integer :: {status}']
        elif result_kind == 'logical':
            lines += [f'    integer(c_int) :: {returned}']
        else:
            lines += [f'    {_get_member_isoc_type(result)} :: {returned}']
        for arg, actual, var in zip(dummy, actual_names, args):
            if _operator_kind(var) == 't':
                lines += [f'    type(c_ptr), value :: {arg}',
                          f'    type({var["typename"]}), pointer :: {actual}']
            else:
                lines += [f'    {_get_member_isoc_type(var)}, value :: {arg}']
        for arg, actual, var in zip(dummy, actual_names, args):
            if _operator_kind(var) == 't':
                lines += [f'    call c_f_pointer({arg}, {actual})']
        arguments = [a if _operator_kind(v) == 't' else _member_actual(v, a)
                     for a, v in zip(actual_names, args)]
        call = _operator_actual_call(proc, pname, arguments)
        if result_kind == 'assignment':
            assignment = _operator_assignment_call(proc, pname, arguments)
            if alias_copy:
                alias_assignment = _operator_assignment_call(
                    proc, pname, [actual_names[0], rhs])
            lines += [f'    {returned} = 0_c_int']
            if alias_copy:
                lines += [f'    if (c_associated({dummy[0]}, {dummy[1]})) then',
                          (f'      allocate({rhs}, source={actual_names[1]}, '
                           f'stat={returned})'),
                          f'      if ({returned} /= 0) return',
                          f'      {alias_assignment}',
                          f'      deallocate({rhs})', '    else',
                          f'      {assignment}', '    end if']
            else:
                lines += [f'    {assignment}']
        elif result_kind == 'type':
            lines += [f'    {returned} = c_null_ptr',
                      f'    allocate({value}, source={call}, stat={status})',
                      f'    if ({status} /= 0) return',
                      f'    {returned} = c_loc({value})']
        elif result_kind == 'logical':
            lines += [f'    {returned} = merge(1_c_int, 0_c_int, {call})']
        else:
            lines += [f'    {returned} = {_member_value(result, call)}']
        lines += [f'  end function {wrapper}', '']
        if result_kind == 'type':
            lines += [f'  subroutine {wrapper}_free({handle}) bind(c)',
                      f'    type(c_ptr), value :: {handle}',
                      f'    type({result["typename"]}), pointer :: {value}',
                      f'    if (.not. c_associated({handle})) return',
                      f'    call c_f_pointer({handle}, {value})',
                      f'    deallocate({value})',
                      f'  end subroutine {wrapper}_free', '']
    return lines


def _operator_group_key(proc):
    return tuple((v.get('typename', '').lower() if _operator_kind(v) == 't'
                  else _operator_kind(v))
                 for a in _routine_args(proc) for v in (proc['vars'][a],))


def _operator_preferred(entries, proc, position):
    """Use default Fortran kinds for Python scalars and exact NumPy dtypes."""
    key = _operator_group_key(proc)
    variants = [other['vars'][_routine_args(other)[position]]
                for _, other, _ in entries if _operator_group_key(other) == key]
    ctypes = {_get_member_ctype(var) for var in variants}
    order = ('int', 'float', 'npy_cfloat', 'long long', 'long', 'short',
             'double', 'npy_cdouble', 'unsigned char')
    return next(ctype for ctype in order if ctype in ctypes)


def _operator_match(typename, entries, proc, reverse=False):
    checks = []
    for i, arg in enumerate(_routine_args(proc)):
        var = proc['vars'][arg]
        obj = (('right', 'left') if reverse else ('left', 'right'))[i]
        if _operator_kind(var) == 't':
            checks.append(f'Py_IS_TYPE({obj}, '
                          f'&Py{var["typename"].lower()}_Type)')
        else:
            ctype = _get_member_ctype(var)
            preferred = int(ctype == _operator_preferred(entries, proc, i))
            checks.append(f'(f2py_op_{typename}_scalar_match({obj}, '
                          f"'{_operator_kind(var)}', sizeof({ctype}), "
                          f'{preferred}) > 0)')
    return ' && '.join(checks)


def _operator_convert(proc, reverse=False):
    lines = []
    names = _routine_args(proc)
    owners = []
    order = sorted(enumerate(names),
                   key=lambda item: _operator_kind(proc['vars'][item[1]]) == 't')
    for i, arg in order:
        var = proc['vars'][arg]
        obj = (('right', 'left') if reverse else ('left', 'right'))[i]
        val = f'f2py_arg{i}'
        kind = _operator_kind(var)
        ctype = _get_member_ctype(var)
        if kind == 't':
            owner = f'f2py_owner{i}'
            typename = var['typename'].lower()
            cleanup = [f'            Py_DECREF({old});' for old in owners]
            lines += [f'        PyObject *{owner} = f2py_op_{typename}_owner({obj});',
                      f'        if ({owner} == NULL) {{',
                      *cleanup, '            return NULL;', '        }']
            owners.append(owner)
            lines += [f'        void *{val} = PyCapsule_GetPointer(',
                      f'            {owner}, "f2py.{typename}");',
                      f'        if ({val} == NULL) {{',
                      *(f'            Py_DECREF({old});' for old in owners),
                      '            return NULL;', '        }']
        elif kind == 'i':
            lines += [f'        PyObject *index{i} = PyNumber_Index({obj});',
                      f'        if (index{i} == NULL) return NULL;',
                      f'        long long integer{i} = PyLong_AsLongLong(index{i});',
                      f'        Py_DECREF(index{i});',
                      '        if (PyErr_Occurred()) return NULL;',
                      f'        {ctype} {val} = ({ctype})integer{i};',
                      f'        if ((long long){val} != integer{i}) {{',
                      '            PyErr_SetString(PyExc_OverflowError,',
                      '                            "operator integer out of range");',
                      '            return NULL;',
                      '        }']
        elif kind == 'b':
            lines += [f'        int truth{i} = PyObject_IsTrue({obj});',
                      f'        if (truth{i} < 0) return NULL;',
                      f'        {ctype} {val} = ({ctype})truth{i};']
        elif kind == 'f':
            lines += [f'        {ctype} {val} = ({ctype})PyFloat_AsDouble({obj});',
                      '        if (PyErr_Occurred()) return NULL;']
        else:
            suffix = 'f' if ctype == 'npy_cfloat' else ''
            lines += [f'        Py_complex complex{i} = PyComplex_AsCComplex({obj});',
                      '        if (PyErr_Occurred()) return NULL;',
                      (f'        {ctype} {val} = npy_cpack{suffix}('
                       f'complex{i}.real, complex{i}.imag);')]
    args = ', '.join(f'f2py_arg{i}' for i in range(len(names)))
    cleanup = [f'        Py_DECREF({owner});' for owner in owners]
    return lines, args, cleanup


def _operator_invoke(typename, entry, reverse=False):
    pname, proc, result_kind = entry
    wrapper = f'f2py_op_{_fortran_sym(typename)}_{pname}'
    lines, args, cleanup = _operator_convert(proc, reverse)
    result = proc['vars'].get(proc.get('result', proc['name']), {})
    if result_kind == 'assignment':
        lines += [f'        int status = {wrapper}({args});', *cleanup,
                  '        if (PyErr_Occurred()) return NULL;',
                  '        if (status != 0) return PyErr_NoMemory();',
                  '        return Py_NewRef(left);']
    elif result_kind == 'type':
        result_name = result['typename'].lower()
        lines += [f'        void *value = {wrapper}({args});',
                  '        if (PyErr_Occurred()) {',
                  f'            if (value != NULL) {wrapper}_free(value);',
                  *('    ' + line for line in cleanup),
                  '            return NULL;', '        }',
                  '        if (value == NULL) {',
                  *('    ' + line for line in cleanup),
                  '            return PyErr_NoMemory();', '        }',
                  f'        PyObject *result = f2py_op_{result_name}_adopt(',
                  f'            value, {wrapper}_capsule_destructor);',
                  f'        if (result == NULL) {wrapper}_free(value);',
                  *cleanup, '        return result;']
    elif result_kind == 'logical':
        lines += [f'        int value = {wrapper}({args});', *cleanup,
                  '        if (PyErr_Occurred()) return NULL;',
                  '        return PyBool_FromLong(value);']
    else:
        ctype = _get_member_ctype(result)
        lines += [f'        {ctype} value = {wrapper}({args});']
        lines += [*cleanup, '        if (PyErr_Occurred()) return NULL;']
        if _operator_kind(result) == 'c':
            suffix = 'f' if ctype == 'npy_cfloat' else ''
            ret = f'PyComplex_FromDoubles(npy_creal{suffix}(value), '
            ret += f'npy_cimag{suffix}(value))'
        else:
            ret = _C_TO_PYOBJ[ctype].format(val='value')
        lines += [f'        return {ret};']
    return lines


def _operator_dispatch(typename, name, entries, reversed_entries=(), assignment=False):
    all_entries = [*entries, *reversed_entries]
    unary = len(_routine_args(all_entries[0][1])) == 1
    decl = ('PyObject *left' if unary else 'PyObject *left, PyObject *right')
    lines = ['static PyObject *', f'{name}({decl})', '{']
    lines += ['    int matches;']
    variants = [(entry, False) for entry in entries]
    variants += [(entry, True) for entry in reversed_entries]
    for entry, reverse in variants:
        match = _operator_match(typename, all_entries, entry[1], reverse)
        lines += [f'    matches = {match};',
                  '    if (PyErr_Occurred()) return NULL;',
                  '    if (matches) {',
                  *_operator_invoke(typename, entry, reverse), '    }']
    if assignment:
        lines += ['    PyErr_SetString(PyExc_TypeError,',
                  '                    "no matching Fortran assignment interface");',
                  '    return NULL;']
    else:
        lines += ['    Py_RETURN_NOTIMPLEMENTED;']
    lines += ['}']
    return '\n'.join(lines)


def _operator_scalar_match(typename):
    return f"""\
static int
f2py_op_{typename}_scalar_match(PyObject *obj, char kind,
                               npy_intp itemsize, int preferred)
{{
    if (PyArray_IsScalar(obj, Generic)) {{
        PyArray_Descr *descr = PyArray_DescrFromScalar(obj);
        int matches;
        if (descr == NULL) return -1;
        matches = ((descr->kind == kind || (kind == 'i' && descr->kind == 'u'))
                   && (kind == 'b' ? preferred
                                   : PyDataType_ELSIZE(descr) == itemsize));
        Py_DECREF(descr);
        return matches;
    }}
    if (!preferred) return 0;
    switch (kind) {{
    case 'i': return PyIndex_Check(obj) && !PyBool_Check(obj);
    case 'f': return PyFloat_Check(obj);
    case 'c': return PyComplex_Check(obj);
    case 'b': return PyBool_Check(obj);
    default: return 0;
    }}
}}
"""


def _gen_assignment_method_entries(typename, method_entries):
    """Expose defined assignment without replacing a native type-bound method."""
    name = f'f2py_op_{typename}_assignment'
    declaration = f'static PyObject *{name}(PyObject *, PyObject *);'
    doc = 'Apply the Fortran assignment interface and return self.'
    names = ['__fortran_assign__']
    if not any('"assign_from"' in entry for entry in method_entries):
        names.append('assign_from')
    entries = [f'    {{"{method}", (PyCFunction){name}, METH_O, "{doc}"}},'
               for method in names]
    return declaration, entries


def _gen_operator_c_code(typename, ops, type_map):
    """Emit one Python slot per arity with all matching native overloads."""
    code = []
    slots = {}
    comparisons = {}
    types = {typename.lower()}
    scalar_args = False
    for pname, proc, result_kind in _operator_entries(ops):
        args, result = _operator_signature(proc, type_map)
        types.update(v['typename'].lower() for v in [*args, result]
                     if _operator_kind(v) == 't')
        scalar_args |= any(_operator_kind(v) != 't' for v in args)
        wrapper = f'f2py_op_{_fortran_sym(typename)}_{pname}'
        decl = ', '.join('void *' if _operator_kind(v) == 't'
                         else _get_member_ctype(v) for v in args)
        ret = ('int' if result_kind == 'assignment'
               else 'void *' if result_kind == 'type' else 'int'
               if result_kind == 'logical' else _get_member_ctype(result))
        code += [f'extern {ret} {wrapper}({decl});']
        if result_kind == 'type':
            code += [f'extern void {wrapper}_free(void *);', f"""\
static void
{wrapper}_capsule_destructor(PyObject *capsule)
{{
    void *value = PyCapsule_GetPointer(capsule, "f2py.{result['typename'].lower()}");
    if (value != NULL) {wrapper}_free(value);
}}
"""]
    forward = []
    for name in sorted(types):
        forward += [f'static PyTypeObject Py{name}_Type;',
                    f'static PyObject *f2py_op_{name}_owner(PyObject *);',
                    (f'static PyObject *f2py_op_{name}_adopt('
                     'void *, PyCapsule_Destructor);')]
    code = [*forward, *code]
    if scalar_args:
        code += [_operator_scalar_match(typename)]
    for op, entries in ops.items():
        if op == '=':
            code += [_operator_dispatch(
                typename, f'f2py_op_{typename}_assignment', entries, assignment=True)]
            continue
        for entry in entries:
            arity = len(_routine_args(entry[1]))
            if op in _FORTRAN_CMP_OPS:
                comparisons.setdefault(_FORTRAN_CMP_OPS[op], []).append(entry)
            else:
                slot = ({'+': 'nb_positive', '-': 'nb_negative'}[op]
                        if arity == 1 else _FORTRAN_ARITH_OPS[op])
                slots.setdefault(slot, []).append(entry)
    for slot, entries in slots.items():
        code += [_operator_dispatch(typename, f'Py{typename}_{slot}', entries)]
    if slots:
        code += ['\n'.join([f'static PyNumberMethods Py{typename}_as_number = {{',
                            *(f'    .{slot} = Py{typename}_{slot},' for slot in slots),
                            '};'])]
    if comparisons:
        inverse = {'Py_EQ': 'Py_EQ', 'Py_NE': 'Py_NE',
                   'Py_LT': 'Py_GT', 'Py_LE': 'Py_GE',
                   'Py_GT': 'Py_LT', 'Py_GE': 'Py_LE'}
        dispatches = dict(comparisons)
        for cmp in comparisons:
            dispatches.setdefault(inverse[cmp], [])
        for cmp, entries in dispatches.items():
            reversed_entries = comparisons.get(inverse[cmp], [])
            code += [_operator_dispatch(typename, f'Py{typename}_{cmp}', entries,
                                        reversed_entries)]
        lines = ['static PyObject *', (f'Py{typename}_tp_richcompare('
                 'PyObject *left, PyObject *right, int op)'), '{',
                 '    switch (op) {']
        lines += [f'    case {cmp}: return Py{typename}_{cmp}(left, right);'
                  for cmp in dispatches]
        lines += ['    default: Py_RETURN_NOTIMPLEMENTED;', '    }', '}']
        code += ['\n'.join(lines)]
    return code, bool(slots), bool(comparisons)
