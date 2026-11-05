"""Expose parameterized module variables through validated selector handles."""

import hashlib
import json

from ._dt_handles import _handle_description, _handle_operation_parts, _type_arguments
from ._dt_namespace import _find_type
from ._dt_operators import _module_symbol_private
from ._dt_routines import _dt_specialization, _routine_length_parameters


def _plain(value):
    """Copy source metadata without parent and imported-context graphs."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()
                if key not in {'parent_block', 'f2py_import_context',
                               'f2py_c_import_context'}
                and not key.startswith('f2py_')}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _variable_exports(modules, types):
    from ._dt_imports import _import_items, _public_type
    from .auxfuncs import isintent_hide, isprivate

    indexed = {module['name'].lower(): module for module in modules}
    exports, active = {}, set()

    def visit(module):
        key = module['name'].lower()
        if key in exports:
            return exports[key]
        if key in active:
            raise ValueError('cyclic module-variable USE dependency')
        active.add(key)
        entries = {}
        for name, var in module.get('vars', {}).items():
            if var.get('typespec') not in {'type', 'class'}:
                continue
            if (isintent_hide(var) or isprivate(var)
                    or not _public_type(module, name, var)):
                continue
            info = _find_type(types, var.get('typename'), key, module.get('use'))
            if info is not None and (info['kind_params'] or info['len_params']):
                entries[name.lower()] = (module, name, var, info)
        for dependency, mapping in module.get('use', {}).items():
            used = indexed.get(dependency.lower())
            if used is None:
                continue
            for name, record in _import_items(visit(used), mapping):
                if (name not in entries
                        and _public_type(module, name, {}, imported=True)):
                    entries[name] = record
        active.remove(key)
        exports[key] = entries
        return entries

    for module in modules:
        visit(module)
    return exports


def prepare_module_variable_bindings(modules, types):
    """Attach canonical module selections without duplicating type descriptors."""
    from .crackfortran import get_parameters, get_useparameters

    records = []
    owners = {}
    exports = _variable_exports(modules, types)
    indexed = {module['name'].lower(): module for module in modules}
    origins = sorted({(origin[0]['name'].lower(), origin[1].lower())
                      for entries in exports.values() for origin in entries.values()})
    storage_ids = {origin: index for index, origin in enumerate(origins, 1)}
    for info in types.values():
        block = info['block']
        block['f2py_pdt_bindings'] = []
        owners[(info['module'].lower(), info['name'].lower())] = block
    for module in sorted(modules, key=lambda item: item['name'].lower()):
        module['f2py_pdt_bindings'] = []
        for name, origin in sorted(exports[module['name'].lower()].items()):
            declaring_module, declaring_name, var, info = origin
            params = get_parameters(declaring_module.get('vars', {}),
                                    get_useparameters(declaring_module))
            if var['typespec'] == 'class':
                raise ValueError('polymorphic module variables require dynamic '
                                 f'type selection: {module["name"]}.{name}')
            specialization = _dt_specialization(info, var.get('typeparams'), params)
            if specialization is None:
                raise ValueError('module variable has an unwrapped KIND: '
                                 f'{module["name"]}.{name}')
            spec, capsule, _ = specialization
            description = _handle_description(info['block'], spec)
            description.pop('bindings', None)
            parameters = {}
            for position, argument in enumerate(_type_arguments(
                    'value(' + (var.get('typeparams') or '') + ')')):
                if not argument.strip():
                    continue
                key, separator, value = argument.partition('=')
                if not separator:
                    if position >= len(info['type_param_order']):
                        raise ValueError('too many module-variable type parameters')
                    key, value = info['type_param_order'][position], key
                parameters[key.strip().lower()] = value.strip()
            deferred = [item['name'] for item in description['lengths']
                        if parameters.get(item['name']) == ':']
            rank = len(var.get('dimension', []))
            if rank > 15:
                raise ValueError('module variable rank exceeds Fortran rank 15')
            identity = module['name'].lower() + ':' + name.lower()
            token = hashlib.sha256(identity.encode('ascii')).hexdigest()[:24]
            attrs = set(var.get('attrspec', []))
            assignments = []
            for scope_name in {module['name'].lower(),
                               declaring_module['name'].lower(),
                               description['owner']}:
                scope = indexed.get(scope_name, {})
                visible = any(
                    block.get('block') == 'interface'
                    and block.get('name', '').lower().replace(' ', '')
                    == 'assignment(=)'
                    for block in scope.get('body', []))
                if visible and not _module_symbol_private(scope, 'assignment(=)'):
                    assignments.append(scope_name)
            record = dict(id=len(records) + 1,
                          storage_id=storage_ids[(declaring_module['name'].lower(),
                                                  declaring_name.lower())],
                          module=module['name'].lower(), variable=name.lower(),
                          origin_module=declaring_module['name'].lower(),
                          origin_variable=declaring_name.lower(),
                          alias='f2py_v_' + token, prefix='f2py_m_' + token,
                          identity_number=description['identity_number'],
                          description=description, c_name=info['c_name'],
                          type_tag=info['tag'], capsule=capsule, rank=rank,
                          allocatable='allocatable' in attrs,
                          pointer='pointer' in attrs, target='target' in attrs,
                          readonly=bool(attrs & {'parameter', 'protected'}),
                          deferred_lengths=deferred,
                          length_parameters=_routine_length_parameters(
                              info, var.get('typeparams'), params),
                          assignment_modules=sorted(assignments),
                          var=_plain(var))
            records.append(record)
            module['f2py_pdt_bindings'].append(record)
            info['block']['f2py_pdt_bindings'].append(record)
    # Imported contexts are cloned before the canonical binding lists exist.
    for module in modules:
        for block in module.get('f2py_imported_types', {}).values():
            key = (block['f2py_origin_module'].lower(),
                   block['f2py_origin_name'].lower())
            if key in owners:
                block['f2py_pdt_bindings'] = owners[key]['f2py_pdt_bindings']
    return records


def _bindings(modules):
    return [record for module in modules
            for record in module.get('f2py_pdt_bindings', [])]


def _canonical_bindings(records):
    """Project export records into one native dispatch case per storage slot."""
    selected = {}
    for record in sorted(records, key=lambda item: (
            item['module'] != item.get('origin_module', item['module']),
            item['variable'] != item.get('origin_variable', item['variable']),
            item['module'], item['variable'])):
        storage_id = record.get('storage_id', record['id'])
        if storage_id not in selected:
            selected[storage_id] = {**record, 'id': storage_id,
                                    'export_id': record['id']}
    return [selected[key] for key in sorted(selected)]


def _use(module, alias, native):
    return [f'use {module}, only: &', f'  {alias} => &', f'  {native}']


def _selected(record, header='header'):
    alias = record['alias']
    if not record['rank']:
        return alias
    indices = [f'lbound({alias}, {axis}, kind=c_intptr_t) + &\n'
               f'    int({header}%subscripts({axis}), kind=c_intptr_t)'
               for axis in range(1, record['rank'] + 1)]
    return alias + '(' + ', &\n    '.join(indices) + ')'


def _availability(record, failure):
    alias = record['alias']
    if record['allocatable']:
        return [f'if (.not. allocated({alias})) {failure}']
    if record['pointer']:
        return [f'if (.not. associated({alias})) {failure}']
    return []


def _length_checks(record, expression, target, failure):
    lines = []
    for index, item in enumerate(record['description']['lengths'], 1):
        value = expression + '%' + item['name']
        lines += [f'if (int({value}, c_int64_t) < &',
                  f'    -int(huge(0_c_int), c_int64_t)-1_c_int64_t) {failure}',
                  f'if (int({value}, c_int64_t) > &',
                  f'    int(huge(0_c_int), c_int64_t)) {failure}',
                  f'{target}({index}) = int({value}, c_int)']
    return lines


def _validator(record):
    prefix, alias = record['prefix'], record['alias']
    lengths = record['description']['lengths']
    lines = [f'integer(c_int) function {prefix}_valid(handle) result(status) bind(c)',
             'type(c_ptr), value :: handle',
             'type(f2py_pdt_handle), pointer :: header',
             'integer(c_int), pointer :: lengths(:)', 'status = 1',
             'header => f2py_pdt_header(handle)',
             'if (.not. associated(header)) return',
             f'if (header%identity /= {record["identity_number"]}_c_int64_t) return',
             'if (header%storage /= 1) return',
             f'if (header%binding /= {record.get("storage_id", record["id"])}) return',
             f'if (header%rank /= {record["rank"]}) return',
             f'if (header%length_count /= {len(lengths)}) return',
             *_availability(record, 'return')]
    for axis in range(1, record['rank'] + 1):
        lines += [f'if (header%subscripts({axis}) < 0_c_size_t) return',
                  f'if (header%subscripts({axis}) > &',
                  '    int(huge(0_c_intptr_t), c_size_t)) return',
                  f'if (header%subscripts({axis}) >= &',
                  f'    size({alias}, {axis}, kind=c_size_t)) return']
    if lengths:
        lines += ['if (.not. c_associated(header%lengths)) return',
                  'call c_f_pointer(header%lengths, lengths, [header%length_count])']
        lines += _length_checks(record, _selected(record), 'lengths', 'return')
    lines += ['status = 0', f'end function {prefix}_valid']
    return lines


def _producer(record):
    prefix, alias = record['prefix'], record['alias']
    desc = record['description']
    rank, lengths = record['rank'], desc['lengths']
    lines = [f'subroutine {prefix}_shape(dims, status) bind(c)',
             f'integer(c_size_t), intent(out) :: dims({max(rank, 1)})',
             'integer(c_int), intent(out) :: status',
             'status = 1', 'dims = 0', *_availability(record, 'return')]
    lines += [f'dims({axis}) = size({alias}, {axis}, kind=c_size_t)'
              for axis in range(1, rank + 1)]
    lines += ['status = 0', f'end subroutine {prefix}_shape', '',
              f'function {prefix}_select(offsets, status) result(handle) bind(c)',
              f'integer(c_size_t), intent(in) :: offsets({max(rank, 1)})',
              'integer(c_int), intent(out) :: status',
              'type(c_ptr) :: handle',
              'type(f2py_pdt_handle), pointer :: header',
              f'integer(c_int) :: lengths({len(lengths)})',
              'handle = c_null_ptr', 'status = 2', 'lengths = 0',
              (f'handle = {desc["prefix"]}_bind('
               f'{record.get("storage_id", record["id"])}, &'),
              f'    offsets(:{rank}), lengths, c_funloc({prefix}_valid))',
              'if (.not. c_associated(handle)) return',
              'header => f2py_pdt_header(handle)',
              f'header%writable = {int(not record["readonly"])}',
              f'header%pointer_binding = {int(record["pointer"])}',
              f'header%target_binding = {int(record["target"])}',
              f'status = {prefix}_valid(handle)',
              'if (status /= 0) then',
              '  call f2py_pdt_destroy(handle)',
              '  handle = c_null_ptr', 'end if',
              f'end function {prefix}_select']
    return lines


def _temporary_type(desc):
    args = [name + '=' + str(value) for name, value in desc['kinds'].items()]
    args += [item['name'] + f'=lengths({i})'
             for i, item in enumerate(desc['lengths'], 1)]
    return desc['local_type'] + '(' + ', '.join(args) + ')'


def _setter(record):
    if record['readonly']:
        return []
    prefix, alias = record['prefix'], record['alias']
    rank, desc = record['rank'], record['description']
    shape = ','.join(':' for _ in range(rank))
    dims = '(' + shape + ')' if rank else ''
    indices = [f'f2py_i{i}' for i in range(1, rank + 1)]
    indexed = '(' + ','.join(indices) + ')' if rank else ''
    lines = [f'subroutine {prefix}_write(handles, dims, count, clear, status) bind(c)',
             'type(c_ptr), intent(in) :: handles(*)',
             f'integer(c_size_t), intent(in) :: dims({max(rank, 1)})',
             'integer(c_size_t), value :: count',
             'integer(c_int), value :: clear',
             'integer(c_int), intent(out) :: status',
             f'type({desc["pointer_spec"]}), allocatable :: temp{dims}',
             f'type({desc["pointer_spec"]}), pointer :: source',
             'type :: f2py_snapshot',
             f'  type({desc["pointer_spec"]}), allocatable :: value',
             'end type f2py_snapshot',
             'type(f2py_snapshot), allocatable :: copies(:)',
             f'integer(c_int) :: lengths({len(desc["lengths"])}), current_length, error',
             'integer(c_size_t) :: product, position',
             *(['integer(c_intptr_t) :: ' + ','.join(indices)] if rank else [])]
    body = ['status = 1', 'if (clear /= 0) then']
    if record['allocatable']:
        body += [f'  if (allocated({alias})) deallocate({alias}, stat=error)',
                 f'  if (allocated({alias})) return', '  status = 0']
    elif record['pointer']:
        body += [f'  nullify({alias})', '  status = 0']
    body += ['  return', 'end if', 'product = 1_c_size_t']
    for axis in range(1, rank + 1):
        body += [f'if (dims({axis}) < 0_c_size_t) return',
                 f'if (dims({axis}) > int(huge(0_c_intptr_t), c_size_t)) return',
                 f'if (dims({axis}) /= 0_c_size_t) then',
                 f'  if (product > huge(product) / dims({axis})) return',
                 'end if', f'product = product * dims({axis})']
    body += ['if (product /= count) return']
    if record['pointer'] or not record['allocatable']:
        body += _availability(record, 'return')
        body += [f'if (dims({axis}) /= size({alias}, {axis}, kind=c_size_t)) return'
                 for axis in range(1, rank + 1)]
    body += ['if (count > 0_c_size_t) then',
             '  do position = 1_c_size_t, count',
             '    if (f2py_pdt_validate(handles(position)) /= 0) return',
             '    if (f2py_pdt_identity(handles(position)) /= &',
             f'        {record["identity_number"]}_c_int64_t) return']
    for index in range(len(desc['lengths'])):
        body += [f'    if (f2py_pdt_length_value(handles(position), {index}, &',
                 '        current_length) /= 0) return',
                 f'    if (position == 1) lengths({index + 1}) = current_length',
                 f'    if (current_length /= lengths({index + 1})) return']
    body += ['  end do', 'else']
    if desc['lengths']:
        if record['deferred_lengths']:
            body += ['  ' + line for line in _availability(record, 'return')]
        body += ['  ' + line for line in _length_checks(
            record, alias, 'lengths', 'return')]
    body += ['end if']
    for index, item in enumerate(desc['lengths'], 1):
        if not record['allocatable'] or item['name'] not in record['deferred_lengths']:
            body += [f'if (lengths({index}) /= {alias}%{item["name"]}) return']
    alloc_dims = '(' + ', '.join(f'int(dims({i}), c_intptr_t)'
                                for i in range(1, rank + 1)) + ')' if rank else ''
    body += ['allocate(copies(count), stat=error)',
             'if (error /= 0) then', '  status = 2', '  return', 'end if',
             'do position = 1_c_size_t, count']
    operation = ['allocate(copies(position)%value, source=source, stat=error)',
                 'if (error /= 0) then', '  status = 2', '  return', 'end if']
    uses, declarations, operation = _handle_operation_parts(
        [], [], operation,
        {'source': {'handle': 'handles(position)', 'description': desc}},
        failure=('status = 4', 'return'))
    body += operation + ['end do']
    if record['allocatable']:
        body += [f'allocate({_temporary_type(desc)} :: &',
                 f'    temp{alloc_dims}, stat=error)',
                 'if (error /= 0) then', '  status = 2', '  return', 'end if']
        destination = 'temp' + indexed
    else:
        destination = alias
        if rank:
            destination += '(' + ', '.join(
                f'lbound({alias}, {axis}, kind=c_intptr_t) + {index} - 1_c_intptr_t'
                for axis, index in enumerate(indices, 1)) + ')'
    heads = [f'do {index} = 1_c_intptr_t, int(dims({axis}), c_intptr_t)'
             for axis, index in reversed(list(enumerate(indices, 1)))]
    body += ['position = 0_c_size_t', *heads,
             'position = position + 1_c_size_t',
             f'{destination} = copies(position)%value',
             *['end do' for _ in heads]]
    if record['allocatable']:
        body += [f'call move_alloc(temp, {alias})']
    body += ['status = 0']
    return [*lines[:1], *uses, *lines[1:], *declarations, *body,
            f'end subroutine {prefix}_write']


def generate_module_variable_fortran(modules):
    """Emit selectors after the common runtime and concrete handle modules."""
    result = []
    for record in _bindings(modules):
        desc = record['description']
        desc = {**desc, 'bindings': [
            item for item in _canonical_bindings(_bindings(modules))
            if item['identity_number'] == record['identity_number']]}
        record = {**record, 'description': desc}
        uses = _use(record['module'], record['alias'], record['variable'])
        uses += _use(desc['owner'], desc['local_type'], desc['native'])
        uses += [f'use {name}, only: assignment(=)'
                 for name in record['assignment_modules']]
        result += [f'module {record["prefix"]}_module', 'use iso_c_binding',
                   'use f2py_pdt_handle_runtime', f'use {desc["module"]}',
                   *uses, 'implicit none', 'private',
                   f'public :: {record["prefix"]}_shape, &',
                   f'    {record["prefix"]}_select',
                   *([f'public :: {record["prefix"]}_write']
                     if not record['readonly'] else []),
                   'contains', *_validator(record), '', *_producer(record), '',
                   *_setter(record), f'end module {record["prefix"]}_module', '']
    return '\n'.join(_continuations(result))


def _continuations(lines):
    """Keep free-form statements within the standard source line limit."""
    physical = (line.rstrip() for statement in lines
                for line in statement.split('\n'))
    for line in physical:
        while len(line) > 120:
            quote = None
            cuts = []
            index = 0
            while index < min(len(line), 112):
                char = line[index]
                if quote:
                    if char == quote:
                        if index + 1 < len(line) and line[index + 1] == quote:
                            index += 1
                        else:
                            quote = None
                elif char in {'"', "'"}:
                    quote = char
                elif char == ',' or char.isspace():
                    cuts.append(index + 1)
                index += 1
            if not cuts or cuts[-1] <= len(line) - len(line.lstrip()):
                raise ValueError('generated Fortran token exceeds source line limit')
            split = cuts[-1]
            yield line[:split].rstrip() + ' &'
            line = '  & ' + line[split:].lstrip()
        yield line


def _c_string(value):
    return json.dumps(value, ensure_ascii=True)


def _c_getter(record):
    p, cname, rank = record['prefix'], record['c_name'], record['rank']
    capsule = _c_string(record['capsule'])
    make = (f'f2py_{cname}_from_handle(handle, {capsule}, '
            f'{p}_destroy, owner)')
    lines = [f'extern void {p}_shape(size_t *, int *);',
             f'extern void *{p}_select(const size_t *, int *);',
             f'static void {p}_destroy(PyObject *capsule)', '{',
             '    PyObject *error = PyErr_GetRaisedException();',
             f'    void *ptr = PyCapsule_GetPointer(capsule, {capsule});',
             '    PyObject *owner = (PyObject *)PyCapsule_GetContext(capsule);',
             '    if (ptr != NULL) f2py_pdt_destroy(ptr);',
             '    Py_XDECREF(owner);',
             '    if (PyErr_Occurred()) PyErr_WriteUnraisable(capsule);',
             '    PyErr_SetRaisedException(error);', '}',
             f'static PyObject *{p}_get(PyObject *owner, void *closure)', '{',
             (f'    size_t dims[{max(rank, 1)}] = {{0}}, '
              f'offsets[{max(rank, 1)}] = {{0}};'),
             '    int status;', '    (void)closure;',
             f'    {p}_shape(dims, &status);',
             '    if (PyErr_Occurred()) return NULL;',
             '    if (status == 1) Py_RETURN_NONE;',
             '    if (status != 0) {',
             '        PyErr_SetString(PyExc_RuntimeError,',
             '                        "invalid native module shape");',
             '        return NULL;', '    }']
    if not rank:
        return '\n'.join(lines + [f'    void *handle = {p}_select(offsets, &status);',
               '    if (PyErr_Occurred()) {',
               '        if (handle != NULL) f2py_pdt_destroy(handle);',
               '        return NULL;', '    }',
               '    if (status != 0) {',
               '        PyErr_SetString(PyExc_RuntimeError,',
               '                        "native selector creation failed");',
               '        return NULL;', '    }', f'    return {make};', '}'])
    lines += [f'    npy_intp shape[{rank}], count = 1;']
    lines += [f'''    if (dims[{i}] > (size_t)NPY_MAX_INTP) {{
        PyErr_SetString(PyExc_OverflowError, "native module extent is too large");
        return NULL;
    }}
    shape[{i}] = (npy_intp)dims[{i}];
    if (shape[{i}] != 0 && count > NPY_MAX_INTP / shape[{i}]) {{
        PyErr_SetString(PyExc_OverflowError, "native module shape is too large");
        return NULL;
    }}
    count *= shape[{i}];''' for i in range(rank)]
    lines += ['    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_OBJECT);',
              '    if (dtype == NULL) return NULL;',
              '    PyArrayObject *array = (PyArrayObject *)PyArray_Empty(',
              f'        {rank}, shape, dtype, 1);',
              '    if (array == NULL) return NULL;',
              '    for (npy_intp i = 0; i < count; ++i) {',
              '        npy_intp position = i;']
    lines += [f'        offsets[{i}] = (size_t)(position % shape[{i}]);\n'
              f'        position /= shape[{i}];' for i in range(rank)]
    lines += [f'        void *handle = {p}_select(offsets, &status);',
              '        if (PyErr_Occurred()) {',
              '            if (handle != NULL) f2py_pdt_destroy(handle);',
              '            Py_DECREF(array);', '            return NULL;', '        }',
              '        if (status != 0) {',
              '            PyErr_SetString(PyExc_RuntimeError,',
              '                            "native selection changed during read");',
              '            Py_DECREF(array);', '            return NULL;', '        }',
              f'        PyObject *value = {make};',
              '        if (value == NULL) { Py_DECREF(array); return NULL; }',
              '        PyObject **slot = (PyObject **)PyArray_DATA(array) + i;',
              '        Py_XSETREF(*slot, value);', '    }',
              '    return (PyObject *)array;', '}']
    return '\n'.join(lines)


def _c_setter(record):
    if record['readonly']:
        return ''
    p, rank, cname = record['prefix'], record['rank'], record['c_name']
    capsule = _c_string(record['capsule'])
    input_lines = []
    if rank:
        input_lines = [
            'PyArray_Descr *dtype = PyArray_DescrFromType(NPY_OBJECT);',
            'if (dtype == NULL) return -1;',
            'array = (PyArrayObject *)PyArray_FromAny(value, dtype,',
            f'    {rank}, {rank}, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ENSUREARRAY,',
            '    NULL);', 'if (array == NULL) return -1;',
            'count = (size_t)PyArray_SIZE(array);',
            *[f'dims[{i}] = (size_t)PyArray_DIM(array, {i});'
              for i in range(rank)]]
    input_setup = '\n    '.join(input_lines)
    item = '*((PyObject **)PyArray_DATA(array) + i)' if rank else 'value'
    return f'''extern void {p}_write(void **, const size_t *, size_t, int, int *);
static int {p}_set(PyObject *owner, PyObject *value, void *closure)
{{
    (void)owner;
    (void)closure;
    int status = 0, result = -1;
    size_t dims[{max(rank, 1)}] = {{0}}, count = 1;
    PyArrayObject *array = NULL;
    PyObject **pins = NULL;
    void **handles = NULL;
    if (value == NULL) {{
        PyErr_SetString(PyExc_AttributeError,
                        "native module variables cannot be deleted");
        return -1;
    }}
    if (value == Py_None) {{
        void *empty = NULL;
        {p}_write(&empty, dims, 0, 1, &status);
        if (PyErr_Occurred()) return -1;
        if (status == 0) return 0;
        PyErr_SetString(PyExc_ValueError, "native module variable cannot be cleared");
        return -1;
    }}
    {input_setup}
    if (count > SIZE_MAX / sizeof(void *) || count > SIZE_MAX / sizeof(PyObject *)) {{
        PyErr_NoMemory();
        goto finish;
    }}
    handles = PyMem_Calloc(count ? count : 1, sizeof(void *));
    pins = PyMem_Calloc(count ? count : 1, sizeof(PyObject *));
    if (handles == NULL || pins == NULL) {{ PyErr_NoMemory(); goto finish; }}
    for (size_t i = 0; i < count; ++i) {{
        PyObject *item = {item};
        if (!PyObject_TypeCheck(item, &Py{cname}_Type)) {{
            PyErr_SetString(PyExc_TypeError,
                            "expected the declared native derived type");
            goto finish;
        }}
        pins[i] = ((Py{cname}Object *)item)->capsule;
        Py_XINCREF(pins[i]);
        if (pins[i] == NULL) {{
            PyErr_SetString(PyExc_ValueError, "uninitialized native derived value");
            goto finish;
        }}
        handles[i] = PyCapsule_GetPointer(pins[i], {capsule});
        if (handles[i] == NULL) goto finish;
    }}
    {p}_write(handles, dims, count, 0, &status);
    if (PyErr_Occurred()) goto finish;
    if (status == 0) result = 0;
    else if (status == 2) PyErr_NoMemory();
    else PyErr_Format(PyExc_ValueError,
                      "incompatible native module assignment (status=%d)", status);
finish:
    if (pins != NULL) for (size_t i = 0; i < count; ++i) Py_XDECREF(pins[i]);
    PyMem_Free(pins);
    PyMem_Free(handles);
    Py_XDECREF(array);
    return result;
}}'''


def build_module_variable_hooks(modules):
    """Register static uncached descriptors after the Python classes exist."""
    hooks, initialization = [], []
    for module in modules:
        records = module.get('f2py_pdt_bindings', [])
        if not records:
            continue
        for record in records:
            hooks += [_c_getter(record), _c_setter(record)]
        token = hashlib.sha256(module['name'].encode('ascii')).hexdigest()[:24]
        table = 'f2py_mv_' + token
        entries = []
        for record in records:
            p = record['prefix']
            setter = 'NULL' if record['readonly'] else p + '_set'
            entries.append(f'    {{{_c_string(record["variable"])}, {p}_get, '
                           f'{setter}, "Native module variable", NULL}},')
        hooks.append(f'static PyGetSetDef {table}[] = {{\n' +
                     '\n'.join(entries) + '\n};')
        initialization += ['{', (f'    PyObject *module = PyObject_GetAttrString(m, '
                                 f'{_c_string(module["name"])});'),
                           '    if (module == NULL) return -1;',
                           (f'    int status = F2Py_SetModuleGetSet(module, {table}, '
                            f'{len(records)});'), '    Py_DECREF(module);',
                           '    if (status < 0) return -1;', '}']
    return {'f90modhooks': hooks, 'initf90modhooks': initialization}
