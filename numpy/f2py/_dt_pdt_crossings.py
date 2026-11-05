"""Keep defined I/O and procedure calls in their parameterized handle context."""

import hashlib

from ._dt_handles import _handle_description, _handle_operation_parts
from ._dt_helpers import (
    _C_TO_PYFORMAT,
    _C_TO_PYOBJ,
    _enumerate_specializations,
    _get_extends_parent,
    _get_kind_param_info,
    _get_len_param_info,
    _get_member_ctype,
    _get_member_isoc_type,
    _member_actual,
    _member_value,
)


def crossing_context(type_block):
    """Describe supported native identities without guessing KIND or LEN."""
    if type_block is None:
        return None
    kind_info = _get_kind_param_info(type_block)
    len_info = _get_len_param_info(type_block)
    if not kind_info and not len_info:
        return None
    variants = []
    specializations = _enumerate_specializations(type_block)
    if kind_info and not specializations:
        raise ValueError('no supported native KIND specialization for defined I/O')
    for kinds, suffix in specializations or [(None, '')]:
        variants.append(dict(
            kinds=kinds or {}, suffix=suffix,
            description=_handle_description(type_block, kind_override=kinds)))
    return dict(variants=variants, kinds=kind_info, lengths=len_info)


def io_bindings(block, type_map, source_file, active=()):
    """Retain inherited defined-I/O bindings in each native wrapper type."""
    from ._dt_routines import _scan_type_bound_procedures, _source_file

    identity = (block.get('f2py_origin_module'), block.get('f2py_origin_name',
                                                        block['name']))
    if identity in active:
        raise ValueError('cyclic defined-I/O type ancestry')
    scope = block.get('f2py_import_context', type_map)
    parent = (_get_extends_parent(block) or '').lower()
    inherited = (io_bindings(scope[parent], scope, source_file, (*active, identity))
                 if parent in scope else {})
    native = block.get('f2py_origin_name', block.get('f2py_native_name', block['name']))
    own = _scan_type_bound_procedures(_source_file(block) or source_file, native)
    return {**inherited, **own}


def _type_use(description):
    return [f'use {description["owner"]}, only: &',
            f'    {description["local_type"]} => {description["native"]}']


def _symbol(description, operation, component=''):
    """Keep generated names within Fortran limits and canonical namespaces."""
    identity = description['identity'] + ':' + operation + ':' + component.lower()
    return 'f2py_pio_' + hashlib.sha256(identity.encode('ascii')).hexdigest()[:24]


def _native_body(description, body, declarations=(), *, write=False,
                 failure=('return',), handle='cptr', local='obj'):
    uses, declarations, body = _handle_operation_parts(
        _type_use(description),
        [f'type({description["pointer_spec"]}), pointer :: {local}', *declarations],
        body, {local: dict(handle=handle, description=description)},
        write_targets=(local,) if write else (), failure=failure)
    return uses, declarations, body


def _routine(name, dummies, uses, declarations, body):
    return '\n'.join([
        f'  subroutine {name}({", ".join(dummies)}) bind(c)',
        '    use iso_c_binding', *['    ' + line for line in uses],
        *['    ' + line for line in declarations],
        *['    ' + line for line in body], f'  end subroutine {name}', ''])


def _procedure_writes(pp, module_body):
    from ._dt_routines import _find_interface_block

    if pp['nopass']:
        return False
    interface = _find_interface_block(module_body, pp['interface'])
    names = interface.get('args', []) if interface else []
    if not names:
        return False
    name = pp['pass'] if pp['pass'] in names else names[0]
    return set(interface['vars'][name].get('intent', [])) != {'in'}


def procedure_fortran(typename, proc_ptrs, module_body, context):
    """Call actual native components in owned or selected module storage."""
    from ._dt_routines import _proc_pointer_call

    result = []
    for variant in context['variants']:
        desc = variant['description']
        for pp in proc_ptrs:
            args, returned, problem = _proc_pointer_call(pp, module_body)
            if problem:
                continue
            reserved = {name.lower() for name, _ in args}
            prefix = 'f2py_component'
            number = 0
            while any(prefix + suffix in reserved for suffix in
                      ('_handle', '_object', '_ok', '_result')):
                number += 1
                prefix = 'f2py_component' + str(number)
            handle, obj, ok, value = [prefix + part for part in
                                      ('_handle', '_object', '_ok', '_result')]
            dummies = [handle, ok, *[name for name, _ in args]]
            declarations = [f'type(c_ptr), value :: {handle}',
                            f'integer(c_int), intent(out) :: {ok}',
                            *[f'{_get_member_isoc_type(var)}, value :: {name}'
                              for name, var in args]]
            actual = ', '.join(_member_actual(var, name) for name, var in args)
            call = f'{obj}%{pp["name"]}({actual})'
            if returned is not None:
                dummies.append(value)
                declarations.append(
                    f'{_get_member_isoc_type(returned)}, intent(out) :: {value}')
                operation = f'{value} = {_member_value(returned, call)}'
            else:
                operation = 'call ' + call
            body = [f'{ok} = 0',
                    f'if (associated({obj}%{pp["name"]})) then',
                    f'  {ok} = 1', f'  {operation}', 'end if']
            uses, declarations, body = _native_body(
                desc, body, declarations, handle=handle, local=obj,
                write=_procedure_writes(pp, module_body),
                failure=(f'{ok} = -1', 'return'))
            name = _symbol(desc, 'procedure', pp['name'])
            result.append(_routine(name, dummies, uses, declarations, body))
    return '\n'.join(result)


def write_fortran(typename, context):
    """Write through a descriptor while exporting only interoperable text."""
    result = []
    for variant in context['variants']:
        desc = variant['description']
        symbol = _symbol(desc, 'write')
        free_symbol = _symbol(desc, 'write_free')
        declarations = [
            'type(c_ptr), value :: cptr', 'type(c_ptr), intent(out) :: text',
            'integer(c_int), intent(out) :: n, iostat',
            'character(kind=c_char), pointer :: record(:)',
            'character(kind=c_char, len=:), allocatable :: buffer',
            'character(kind=c_char) :: c', 'integer :: unit, index, status']
        body = [
            'text = c_null_ptr', 'n = 0', 'iostat = 1',
            "open(newunit=unit, status='scratch', form='formatted', &",
            "     action='readwrite', iostat=iostat)",
            'if (iostat /= 0) return', "write(unit, '(DT)', iostat=iostat) obj",
            'if (iostat == 0) then', '  rewind(unit)', '  do',
            "    read(unit, '(a)', advance='no', iostat=iostat) c",
            '    if (iostat /= 0) exit', '    if (n == huge(n)) then',
            '      iostat = 1', '      exit', '    end if', '    n = n + 1',
            '  end do', '  if (iostat == iostat_eor) then',
            '    allocate(character(kind=c_char, len=max(n,1)) :: buffer, stat=status)',
            '    if (status /= 0) then', '      iostat = status',
            '    else', '      iostat = 0', '      if (n > 0) then',
            '        rewind(unit)', "        read(unit, '(a)', iostat=iostat) buffer",
            '      end if', '      if (iostat == 0) then',
            '        allocate(record(max(n,1)), stat=status)',
            '        if (status /= 0) then', '          iostat = status',
            '        else', '          record = c_null_char',
            '          do index = 1, n',
            '            record(index) = buffer(index:index)',
            '          end do', '          text = c_loc(record(1))',
            '        end if', '      end if', '    end if', '  end if', 'end if',
            'close(unit)']
        uses, declarations, body = _native_body(
            desc, body, declarations,
            failure=('text = c_null_ptr', 'iostat = 1', 'return'))
        uses.append('use iso_fortran_env, only: iostat_eor')
        result.append(_routine(symbol, ['cptr', 'text', 'n', 'iostat'],
                               uses, declarations, body))
        result.append(_routine(free_symbol, ['text', 'n'], [], [
            'type(c_ptr), value :: text', 'integer(c_int), value :: n',
            'character(kind=c_char), pointer :: record(:)'], [
            'if (.not. c_associated(text)) return',
            'call c_f_pointer(text, record, [max(n,1)])', 'deallocate(record)']))
    return '\n'.join(result)


def read_fortran(typename, context, init_lines, loop_depth):
    """Allocate each native result with its selected KIND and explicit LEN."""
    result = []
    length_args = ['f2py_len_' + str(i) for i in range(len(context['lengths']))]
    for variant in context['variants']:
        desc = variant['description']
        args = [name + '=' + str(value) for name, value in desc['kinds'].items()]
        args += [item['name'] + '=' + name for item, name in
                 zip(context['lengths'], length_args)]
        allocated = desc['local_type'] + '(' + ', '.join(args) + ')'
        uses = _type_use(desc) + [f'use {desc["module"]}, only: {desc["prefix"]}_adopt']
        declarations = [
            'character(kind=c_char), intent(in) :: buf(*)',
            'integer(c_int), value :: buflen',
            *[f'integer(c_int), value :: {name}' for name in length_args],
            'type(c_ptr), intent(out) :: cptr',
            'integer(c_int), intent(out) :: iostat',
            f'type({desc["pointer_spec"]}), pointer :: obj',
            'character(len=buflen) :: tmp',
            'integer :: i' + ''.join(f', f2py_i{d}' for d in range(1, loop_depth + 1))]
        body = ['cptr = c_null_ptr', 'iostat = 1',
                'if (buflen < 0) return',
                f'allocate({allocated} :: obj, stat=iostat)',
                'if (iostat /= 0) return', *init_lines,
                'do i = 1, buflen', '  tmp(i:i) = buf(i)', 'end do',
                "read(tmp, '(DT)', iostat=iostat) obj", 'if (iostat == 0) then',
                f'  cptr = {desc["prefix"]}_adopt(obj)',
                '  if (.not. c_associated(cptr)) iostat = 1',
                'else', '  deallocate(obj)', 'end if']
        name = _symbol(desc, 'read')
        result.append(_routine(name, ['buf', 'buflen', *length_args, 'cptr', 'iostat'],
                               uses, declarations, body))
    return '\n'.join(result)


def _pin_and_check(typename, *, write=False, uninitialized_text=None):
    uninitialized = ([f'        return PyUnicode_FromString("{uninitialized_text}");']
                     if uninitialized_text is not None else [
                         '        PyErr_SetString(PyExc_RuntimeError,',
                         '                        "object not initialized");',
                         '        return NULL;'])
    return [
        f'    Py{typename}Object *self = (Py{typename}Object *)selfobj;',
        '    if (self->capsule == NULL) {', *uninitialized, '    }',
        '    PyObject *owner = Py_NewRef(self->capsule);',
        '    const char *name = PyCapsule_GetName(owner);',
        f'    void *ptr = name == NULL ? NULL : f2py_{typename}_checked(',
        f'        selfobj, name, {int(write)});',
        '    if (ptr == NULL) { Py_DECREF(owner); return NULL; }']


def _result_expression(ctype, value):
    return _C_TO_PYOBJ[ctype].format(val=value)


def procedure_c(typename, proc_ptrs, module_body, context):
    """Convert inputs before extracting and retaining the actual native handle."""
    from ._dt_routines import _proc_pointer_call

    functions, entries = [], []
    for pp in proc_ptrs:
        args, returned, problem = _proc_pointer_call(pp, module_body)
        if problem:
            continue
        types = [_get_member_ctype(var) for _, var in args]
        declarations, parsed, converted, actual = [], [], [], []
        for index, ctype in enumerate(types):
            name = f'f2py_arg_{index}'
            if ctype in ('npy_cfloat', 'npy_cdouble'):
                declarations.append(f'    Py_complex {name}_py = {{0, 0}};')
                parsed.append('&' + name + '_py')
                constructor = 'npy_cpackf' if ctype == 'npy_cfloat' else 'npy_cpack'
                converted.append(f'    {ctype} {name} = {constructor}('
                                 f'{name}_py.real, {name}_py.imag);')
            else:
                declarations.append(f'    {ctype} {name} = 0;')
                parsed.append('&' + name)
            actual.append(name)
        extern_types = ['void *', 'int *', *types]
        call_args = ['ptr', '&ok', *actual]
        if returned is not None:
            rtype = _get_member_ctype(returned)
            extern_types.append(rtype + ' *')
            declarations.append(f'    {rtype} f2py_result = 0;')
            call_args.append('&f2py_result')
            expression = _result_expression(rtype, 'f2py_result')
        else:
            expression = 'Py_NewRef(Py_None)'
        externs, cases = [], []
        for variant in context['variants']:
            symbol = _symbol(variant['description'], 'procedure', pp['name'])
            externs.append(f'extern void {symbol}({", ".join(extern_types)});')
            cases += [f'    case {variant["description"]["identity_number"]}LL:',
                      f'        {symbol}({", ".join(call_args)});', '        break;']
        keyword_names = ', '.join('"' + name + '"' for name, _ in args)
        fmt = ''.join(_C_TO_PYFORMAT[ctype] for ctype in types)
        parse_args = ''.join(', ' + value for value in parsed)
        method = f'Py{typename}_method_{pp["name"]}'
        lines = ['#include "numpy/npy_math.h"', *externs, 'static PyObject *',
                 f'{method}(PyObject *selfobj, PyObject *args, PyObject *kwds)', '{',
                 *declarations,
                 '    static char *kwlist[] = {' + keyword_names
                 + (', ' if keyword_names else '') + 'NULL};',
                 f'    if (!PyArg_ParseTupleAndKeywords(args, kwds, "{fmt}",',
                 f'            kwlist{parse_args})) return NULL;',
                 *converted,
                 *_pin_and_check(typename, write=_procedure_writes(pp, module_body)),
                 '    int ok = 0;', '    switch (f2py_pdt_identity(ptr)) {',
                 *cases, '    default:',
                 '        PyErr_SetString(PyExc_TypeError,',
                 '                        "unsupported native KIND identity");',
                 '        Py_DECREF(owner); return NULL;', '    }',
                 '    if (PyErr_Occurred()) { Py_DECREF(owner); return NULL; }',
                 '    if (ok != 1) {',
                 '        PyErr_SetString(PyExc_TypeError,',
                 f'            "procedure pointer component {pp["name"]} "',
                 '            "is not associated");',
                 '        Py_DECREF(owner); return NULL;', '    }',
                 f'    PyObject *answer = {expression};',
                 '    Py_DECREF(owner);', '    return answer;', '}']
        functions.append('\n'.join(lines))
        entries.append(f'    {{"{pp["name"]}", (PyCFunction){method}, '
                       f'METH_VARARGS | METH_KEYWORDS, "Procedure pointer component"}}')
    return functions, entries


def write_c(typename, context):
    """Retain the native owner through callbacks and text conversion."""
    externs, cases = [], []
    for variant in context['variants']:
        symbol = _symbol(variant['description'], 'write')
        free_symbol = _symbol(variant['description'], 'write_free')
        externs += [f'extern void {symbol}(void *, void **, int *, int *);',
                    f'extern void {free_symbol}(void *, int);']
        cases += [f'    case {variant["description"]["identity_number"]}LL:',
                  f'        release_text = {free_symbol};',
                  f'        {symbol}(ptr, &text, &n, &iostat);', '        break;']
    return '\n'.join([
        *externs, 'static PyObject *', f'Py{typename}_tp_str(PyObject *selfobj)', '{',
        *_pin_and_check(typename, uninitialized_text=typename + '(<uninitialized>)'),
        '    void *text = NULL;',
        '    int n = 0, iostat = 0;', '    void (*release_text)(void *, int) = NULL;',
        '    switch (f2py_pdt_identity(ptr)) {', *cases, '    default:',
        '        PyErr_SetString(PyExc_TypeError, "unsupported native KIND identity");',
        '        Py_DECREF(owner); return NULL;', '    }',
        '    if (PyErr_Occurred()) {',
        '        if (text != NULL) release_text(text, n);',
        '        Py_DECREF(owner); return NULL;', '    }',
        '    if (text == NULL || n < 0 || iostat != 0) {',
        '        if (text != NULL) release_text(text, n);',
        '        PyErr_Format(PyExc_ValueError,',
        f'                     "write(formatted) of {typename} "',
        '                     "failed with iostat=%d", iostat);',
        '        Py_DECREF(owner); return NULL;', '    }',
        '    int length = n;',
        "    while (length > 0 && ((const char *)text)[length - 1] == ' ') --length;",
        '    PyObject *answer = PyUnicode_FromStringAndSize(',
        '        (const char *)text, length);',
        '    release_text(text, n);', '    Py_DECREF(owner);',
        '    return answer;', '}'])


def read_c(typename, context):
    """Select explicit native parameters and adopt a fully hydrated result."""
    parameters = [*context['kinds'], *context['lengths']]
    declarations, parsed, checks = [], [], []
    for index, parameter in enumerate(parameters):
        name = parameter['name']
        default = parameter['default'] if parameter['has_default'] else 0
        declarations += [f'    PyObject *f2py_parameter_{index} = NULL;',
                         f'    long f2py_value_{index} = {default};']
        parsed.append(f'&f2py_parameter_{index}')
        checks += [f'    if (f2py_parameter_{index} != NULL) {{',
                   f'        f2py_value_{index} = PyLong_AsLong(',
                   f'            f2py_parameter_{index});',
                   f'        if (f2py_value_{index} == -1 && PyErr_Occurred())',
                   '            return NULL;',
                   '    }']
        if not parameter['has_default']:
            checks += ['    else {',
                       '        PyErr_SetString(PyExc_TypeError,',
                       f'                        "from_string requires {name}");',
                       '        return NULL;', '    }']
        checks += [f'    if (f2py_value_{index} < INT_MIN ||',
                   f'        f2py_value_{index} > INT_MAX) {{',
                   '        PyErr_SetString(PyExc_OverflowError,',
                   '                        "type parameter does not fit a C int");',
                   '        return NULL;', '    }']
    indices = {parameter['name']: index for index, parameter in enumerate(parameters)}
    lengths = [f'(int)f2py_value_{indices[item["name"]]}'
               for item in context['lengths']]
    call_args = ['text', '(int)text_length', *lengths, '&ptr', '&iostat']
    extern_types = ['const char *', 'int', *['int' for _ in lengths],
                    'void **', 'int *']
    externs, cases = [], []
    cname = '"f2py.' + typename
    for position, variant in enumerate(context['variants']):
        symbol = _symbol(variant['description'], 'read')
        externs.append(f'extern void {symbol}({", ".join(extern_types)});')
        condition = ' && '.join(f'f2py_value_{indices[name]} == {value}'
                                for name, value in variant['kinds'].items()) or '1'
        statement = '    if (' if position == 0 else '    else if ('
        cases += [statement + condition + ') {',
                  f'        name = {cname}{variant["suffix"]}";',
                  f'        {symbol}({", ".join(call_args)});', '    }']
    keywords = ', '.join('"' + item['name'] + '"' for item in parameters)
    parsed_suffix = ''.join(', ' + item for item in parsed)
    fmt = 's#' + ('|' + 'O' * len(parameters) if parameters else '')
    return '\n'.join([
        *externs, 'static PyObject *',
        f'Py{typename}_from_string(PyObject *cls, PyObject *args, PyObject *kwds)', '{',
        '    const char *text;', '    Py_ssize_t text_length;', *declarations,
        '    static char *kwlist[] = {"text", ' + keywords + ', NULL};',
        f'    if (!PyArg_ParseTupleAndKeywords(args, kwds, "{fmt}", kwlist,',
        f'            &text, &text_length{parsed_suffix})) return NULL;',
        '    if (text_length > INT_MAX) {',
        '        PyErr_SetString(PyExc_OverflowError, "string too long");',
        '        return NULL;', '    }', *checks,
        '    void *ptr = NULL;', '    int iostat = 0;', '    const char *name = NULL;',
        *cases, '    else {',
        '        PyErr_SetString(PyExc_ValueError,',
        '                        "unsupported native KIND parameters");',
        '        return NULL;', '    }',
        '    if (PyErr_Occurred()) {',
        f'        if (ptr != NULL) f2py_{typename}_release_preserving_error(ptr);',
        '        return NULL;', '    }', '    if (ptr == NULL || iostat != 0) {',
        '        PyErr_Format(PyExc_ValueError,',
        f'                     "read(formatted) of {typename} "',
        '                     "failed with iostat=%d", iostat);',
        f'        if (ptr != NULL) f2py_{typename}_release_preserving_error(ptr);',
        '        return NULL;', '    }',
        f'    return f2py_{typename}_from_handle_as(ptr, name,',
        f'        f2py_{typename}_capsule_destructor, NULL, (PyTypeObject *)cls);',
        '}'])
