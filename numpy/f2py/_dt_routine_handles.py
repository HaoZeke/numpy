"""Pass parameterized routine arguments through validated native operations."""

import re

from ._dt_handles import _handle_operation_parts, _replace_fortran_name, _type_arguments
from .auxfuncs import isarray, isintent_hide, isintent_in, isintent_inout, isintent_out

_PLACEHOLDER = 'f2py_pdt_native_operation'


def _declaration(description, storage, name, shape=''):
    return f'type({description["pointer_spec"]}), {storage} :: {name}{shape}'


def _type_spec(description, lengths):
    arguments = [name + '=' + str(value)
                 for name, value in description['kinds'].items()]
    arguments.extend(item['name'] + '=' + value
                     for item, value in zip(description['lengths'], lengths))
    return description['local_type'] + '(' + ', '.join(arguments) + ')'


def _len_expressions(dt, source=None):
    declared = dt.get('length_parameters', {})
    expressions = []
    for item in dt['pdt_handle']['lengths']:
        value = declared.get(item['name'], item.get('default'))
        value = '*' if value is None else str(value)
        if value in {'*', ':'}:
            value = source + '%' + item['name'] if source else '0'
        expressions.append(value)
    return expressions


def _check_lengths(dt, handle, status, local):
    lines = []
    for index, item in enumerate(dt['pdt_handle']['lengths']):
        expected = dt.get('length_parameters', {}).get(item['name'])
        if expected is None or str(expected) in {'*', ':'}:
            continue
        lines += [f'if (f2py_pdt_length_value({handle}, {index}, {local}) /= 0) then',
                  f'  {status} = 1', '  return', 'end if',
                  f'if (int({local}, c_int64_t) /= int({expected}, c_int64_t)) then',
                  f'  {status} = 1', '  return', 'end if']
    return lines


def _indices(name, rank):
    return [f'f2py_{name}_i{axis}' for axis in range(1, rank + 1)]


def _loops(indices, dimensions, body):
    lines = []
    for index, extent in reversed(list(zip(indices, dimensions))):
        lines.append(f'do {index} = 1, {extent}')
    lines.extend(body)
    lines.extend('end do' for _ in indices)
    return lines


def _expand(uses, declarations, body, pointers, status, *, write=(), pointer=None,
            optional=(), owned=None, allocatable=(), deferred=None, lengths=None):
    return _handle_operation_parts(uses, declarations, body, pointers,
        write_targets=write, pointer_targets=pointer, optional_targets=optional,
        owned_bodies=owned, allocatable_targets=allocatable,
        deferred_targets=deferred, length_targets=lengths,
        failure=(f'{status} = 1', 'return'))


def _output_copy(description, value, pointer, handle, status, cleanup=()):
    return [f'allocate({pointer}, source={value}, stat={status})',
            f'if ({status} /= 0) then', f'  {status} = 2',
            *['  ' + line for line in cleanup], '  return', 'end if',
            f'{handle} = {description["prefix"]}_adopt({pointer})',
            f'if (.not. c_associated({handle})) then',
            f'  {status} = 2', *['  ' + line for line in cleanup], '  return', 'end if']


def _declared_block(description, parameters, local, actual, storage, body, handle):
    values = []
    before, declarations = [], []
    for index, item in enumerate(description['lengths']):
        expression = str(parameters.get(item['name'], '*'))
        if expression == '*':
            buffer = f'{actual}_len{index}'
            declarations.append(f'integer(c_int) :: {buffer}')
            before += [f'{buffer} = 0',
                       f'if (c_associated({handle})) {buffer} = int({local}%{item["name"]}, c_int)']
            expression = buffer
        values.append(expression)
    lines = ['block', *declarations, *before, 'block',
             f'type({_type_spec(description, values)}), {storage} :: {actual}',
             *body, 'end block', 'end block']
    return lines


def _owned_allocation(name, local, snapshot, description, parameters, status, copy_input):
    """Give owned values an allocatable actual without changing selector storage."""
    def transform(statements):
        before = []
        if copy_input:
            before = [f'if (associated({local})) then',
                      f'  allocate({snapshot}, source={local}, stat={status})',
                      f'  if ({status} /= 0) then', f'    {status} = 2',
                      '    return', '  end if', 'end if']
        body = _replace_fortran_name(statements, local, snapshot)
        after = [f'if (c_associated({name})) then',
                 f'  if (allocated({snapshot})) then',
                 f'    f2py_pdt_length_buffer = {description["prefix"]}_replace({name}, {snapshot})',
                 '  else',
                 f'    f2py_pdt_length_buffer = {description["prefix"]}_clear({name})',
                 '  end if',
                 '  if (f2py_pdt_length_buffer /= 0) then',
                 f'    {status} = merge(2, 1, f2py_pdt_length_buffer == 1)',
                 '    return', '  end if', 'end if']
        return _declared_block(description, parameters(), local, snapshot,
                               'allocatable', [*before, *body, *after], name)
    return transform


def _owned_pointer(handle, local, actual, description, parameters, status, association_input):
    def transform(statements):
        if not association_input:
            return [f'{status} = 1', 'return']
        body = [f'{actual} => {local}',
                *_replace_fortran_name(statements, local, actual)]
        return _declared_block(description, parameters(), local, actual, 'pointer', body, handle)
    return transform


def _translated_parameters(dt, names):
    parameters = {}
    for parameter, value in dt.get('length_parameters', {}).items():
        expression = str(value)
        for name, actual in names.items():
            expression = _replace_fortran_name([expression], name, actual)[0]
        parameters[parameter] = expression
    return parameters


def _array_argument(name, var, dt, status, parameters):
    """Build an array copy while dispatching each element's native selection."""
    desc = dt['pdt_handle']
    rank = dt.get('vector', len(var.get('dimension', [])))
    indices = _indices(name, rank)
    index = ', '.join(indices)
    shape = '(' + ', '.join(':' for _ in indices) + ')'
    local, element = 'f2py_' + name, 'f2py_' + name + '_e'
    pure_input = (isintent_in(var) and not isintent_inout(var) and not isintent_out(var)
                  and 'pointer' not in var.get('attrspec', []))
    hidden_output = isintent_out(var) and isintent_hide(var)
    storage = 'pointer' if 'pointer' in var.get('attrspec', []) else 'allocatable'
    uses = ['use ' + desc['module'], 'use f2py_pdt_handle_runtime']
    declarations = [_declaration(desc, 'pointer', element),
                    'integer :: ' + ', '.join(indices),
                    f'integer(c_int) :: f2py_{name}_length({len(desc["lengths"])})',
                    f'integer(c_size_t) :: f2py_{name}_position, f2py_{name}_cleanup',
                    f'type(c_ptr), pointer :: f2py_{name}_handles(:)']
    pre, post, head = [], [], []

    def scope(statements):
        declared = parameters()
        values = [str(declared.get(item['name'], ':'))
                  for item in desc['lengths']]
        values = [f'f2py_{name}_length({index})' if value == '*' else value
                  for index, value in enumerate(values, 1)]
        specification = _type_spec(desc, values)
        definitions = [f'type({specification}), {storage} :: {local}{shape}']
        if storage == 'pointer' and 'vector' not in dt:
            definitions.append(f'type({specification}), allocatable, target :: {local}_owner{shape}')
        return [*head, 'block', *definitions, *statements, 'end block']

    if storage == 'pointer':
        pre.append(f'nullify({local})')
    if 'vector' not in dt:
        dimensions = list(var['dimension'])
        dimensions_text = ', '.join(dimensions)
        first = name + '(' + ', '.join('1' for _ in indices) + ')'
        for axis, expression in enumerate(_len_expressions(dt), 1):
            head.append(f'f2py_{name}_length({axis}) = {expression}')
        if desc['lengths'] and not hidden_output:
            head.append('if (' + ' .and. '.join('(' + size + ') > 0' for size in dimensions) + ') then')
            for axis in range(len(desc['lengths'])):
                head += [f'  if (f2py_pdt_length_value({first}, {axis}, &',
                        f'      f2py_{name}_length({axis + 1})) /= 0) then',
                        f'    {status} = 1', '    return', '  end if']
            head.append('end if')
        length_values = [f'f2py_{name}_length({i})'
                         for i in range(1, len(desc['lengths']) + 1)]
        pre += [f'allocate({_type_spec(desc, length_values)} :: &',
                f'    {local}({dimensions_text}), stat={status})',
                f'if ({status} /= 0) return']
        if not hidden_output:
            source_handle = name + '(' + index + ')'
            body = _check_lengths(dt, source_handle, status, 'f2py_pdt_length_buffer')
            for axis in range(len(desc['lengths'])):
                body += [f'if (f2py_pdt_length_value({source_handle}, {axis}, &',
                         '    f2py_pdt_length_buffer) /= 0) then',
                         f'  {status} = 1', '  return', 'end if',
                         f'if (f2py_pdt_length_buffer /= f2py_{name}_length({axis + 1})) then',
                         f'  {status} = 1', '  return', 'end if']
            body.append(f'{local}({index}) = {element}')
            uses, declarations, body = _expand(uses, declarations, body,
                {element: dict(handle=source_handle, description=desc)}, status)
            pre += _loops(indices, dimensions, body)
        if hidden_output:
            cleanup_indices = [f'f2py_{name}_j{axis}' for axis in range(1, rank + 1)]
            declarations.append('integer :: ' + ', '.join(cleanup_indices))
            cleanup_handle = name + '(' + ', '.join(cleanup_indices) + ')'
            cleanup = _loops(cleanup_indices, dimensions,
                [f'if (c_associated({cleanup_handle})) then',
                 f'  call f2py_pdt_destroy({cleanup_handle})',
                 f'  {cleanup_handle} = c_null_ptr', 'end if'])
            body = _output_copy(desc, f'{local}({index})', element,
                                f'{name}({index})', status, cleanup)
            post += _loops(indices, dimensions, body)
        elif not pure_input:
            actual_indices = ', '.join(f'lbound({local}, {axis}) + {item} - 1'
                                       for axis, item in enumerate(indices, 1))
            body = [f'{element} = {local}({actual_indices})']
            uses, declarations, body = _expand(uses, declarations, body,
                {element: dict(handle=f'{name}({index})', description=desc)},
                status, write=(element,))
            made = 'associated' if storage == 'pointer' else 'allocated'
            post += [f'if ({made}({local})) then',
                     f'if (all(shape({local}) == shape({name}))) then',
                     *_loops(indices, dimensions, body), 'end if', 'end if']
        if storage == 'pointer':
            pre = [line.replace(f'{local}({dimensions_text}), stat=', f'{local}_owner({dimensions_text}), stat=')
                   for line in pre]
            allocation_end = next(i for i, line in enumerate(pre)
                                  if line == f'if ({status} /= 0) return')
            pre.insert(allocation_end + 1, f'{local} => {local}_owner')
        return uses, declarations, pre, post, local, scope
    made = 'associated' if storage == 'pointer' else 'allocated'
    extents = [f'f2py_{name}_n{axis}' for axis in range(1, rank + 1)]
    post += [f'{name} = c_null_ptr', f'{extents[0]} = -1', f'if ({made}({local})) then']
    for axis, extent in enumerate(extents, 1):
        post.append(f'  {extent} = size({local}, {axis})')
    post += [f'  if (size({local}) > 0) then',
             f'    allocate(f2py_{name}_handles(size({local})), stat={status})',
             f'    if ({status} /= 0) return',
             f'    f2py_{name}_handles = c_null_ptr', f'    f2py_{name}_position = 0']
    bounds = [f'size({local}, {axis})' for axis in range(1, rank + 1)]
    element_indices = ', '.join(f'lbound({local}, {axis}) + {item} - 1'
                               for axis, item in enumerate(indices, 1))
    cleanup = [f'do f2py_{name}_cleanup = 1, f2py_{name}_position - 1',
               f'  call f2py_pdt_destroy(f2py_{name}_handles(f2py_{name}_cleanup))',
               'end do', f'deallocate(f2py_{name}_handles)']
    body = [f'f2py_{name}_position = f2py_{name}_position + 1',
            *_output_copy(desc, f'{local}({element_indices})', element,
                          f'f2py_{name}_handles(f2py_{name}_position)', status, cleanup)]
    post.extend('    ' + line for line in _loops(indices, bounds, body))
    post += [f'    {name} = c_loc(f2py_{name}_handles)', '  end if', 'end if']
    return uses, declarations, pre, post, local, scope


def pdt_wrapper_handles(rout, args, result, fallback):
    """Compose native calls with owned and selector-backed PDT actuals."""
    selected = [name for name in args if rout['vars'][name].get('f2py_dt', {}).get('pdt_handle')]
    ordinary = [name for name in args if name not in selected]
    uses, declarations, pre, post, base_body, helpers = fallback(rout, ordinary, result)
    uses, declarations, pre, post = list(uses), list(declarations), list(pre), list(post)
    status = rout['f2py_pdt_status']
    declarations.append('integer(c_int) :: f2py_pdt_length_buffer')
    pre.insert(0, f'{status} = 0')
    names = {name: 'f2py_' + name for name in selected}
    pointers, writable, pointer_targets, optional = {}, [], {}, []
    owned, allocatable, deferred, early_lengths, checks = {}, [], {}, {}, []
    universal = []
    for name in selected:
        var, dt = rout['vars'][name], rout['vars'][name]['f2py_dt']
        dt = {**dt, 'length_parameters': _translated_parameters(dt, names)}
        desc = dt['pdt_handle']
        uses.extend(['use ' + desc['module'], 'use f2py_pdt_handle_runtime'])
        if isarray(var) or 'vector' in dt:
            more_uses, more_declarations, before, after, local, scope = _array_argument(
                name, var, dt, status, lambda dt=dt: _translated_parameters(dt, names))
            uses.extend(more_uses)
            declarations.extend(more_declarations)
            pre.extend(before)
            post.extend(after)
            names[name] = local
            universal.append(scope)
            if 'vector' in dt:
                from ._dt_routines import _dt_vector_free
                helpers.extend(_dt_vector_free(dt))
            continue
        local = 'f2py_' + name
        source = local + '_source'
        storage = ('allocatable' if 'allocatable' in var.get('attrspec', [])
                   else 'pointer' if 'pointer' in var.get('attrspec', []) else None)
        hidden_output = isintent_out(var) and isintent_hide(var)
        pure_input = isintent_in(var) and not isintent_out(var) and not isintent_inout(var)
        names[name] = local
        if hidden_output or name == result:
            actual_storage = storage or 'allocatable'
            declarations.append(_declaration(desc, 'pointer', source))
            if storage is None:
                declarations.append(_declaration(desc, actual_storage, local))
            else:
                def output_scope(statements, dt=dt, desc=desc, local=local,
                                 actual_storage=actual_storage):
                    parameters = _translated_parameters(dt, names)
                    values = [str(parameters.get(item['name'], ':'))
                              for item in desc['lengths']]
                    values = ['0' if value == '*' else value for value in values]
                    specification = _type_spec(desc, values)
                    return ['block',
                            f'type({specification}), {actual_storage} :: {local}',
                            *statements, 'end block']
                universal.append(output_scope)
            if actual_storage == 'pointer':
                pre.append(f'nullify({local})')
            if storage is None and name != result:
                pre += [f'allocate({_type_spec(desc, _len_expressions(dt))} :: &',
                        f'    {local}, stat={status})', f'if ({status} /= 0) return']
            made = 'associated' if actual_storage == 'pointer' else 'allocated'
            post += [f'{name} = c_null_ptr', f'if ({made}({local})) then',
                     *_output_copy(desc, local, source, name, status), 'end if']
            continue
        if storage == 'allocatable' and not pure_input:
            snapshot = local + '_allocation'
            declarations.append(_declaration(desc, 'pointer', local))
            pointers[local] = dict(handle=name, description=desc)
            allocatable.append(local)
            writable.append(local)
            if 'optional' in var.get('attrspec', []):
                optional.append(local)
            owned[local] = _owned_allocation(name, local, snapshot, desc,
                lambda dt=dt: _translated_parameters(dt, names), status,
                copy_input=not isintent_out(var) or isintent_inout(var))
            deferred[local] = {key for key, value in dt.get('length_parameters', {}).items()
                               if str(value) == ':'}
            early_lengths[local] = {key: value for key, value in dt.get('length_parameters', {}).items()
                                   if '%' not in str(value)}
            checks.append((dt, name))
            continue
        declarations.append(_declaration(desc, 'pointer', local))
        target = local
        if storage == 'allocatable':
            snapshot = local + '_allocation'

            def read_snapshot(statements, desc=desc, dt=dt, local=local, snapshot=snapshot, name=name):
                body = [f'if (c_associated({name})) then',
                        f'  allocate({snapshot}, source={local}, stat={status})',
                        f'  if ({status} /= 0) return', 'end if',
                        *_replace_fortran_name(statements, local, snapshot)]
                return _declared_block(desc, _translated_parameters(dt, names),
                                       local, snapshot, 'allocatable', body, name)
            universal.append(read_snapshot)
        pointers[target] = dict(handle=name, description=desc)
        if not pure_input or storage == 'pointer':
            writable.append(target)
        if storage == 'pointer':
            pointer_targets[target] = pure_input
            deferred[target] = {key for key, value in dt.get('length_parameters', {}).items()
                                if str(value) == ':'}
            owned[target] = _owned_pointer(name, local, local + '_actual', desc,
                lambda dt=dt: _translated_parameters(dt, names), status, pure_input)
        if 'optional' in var.get('attrspec', []):
            optional.append(target)
        early_lengths[target] = {key: value for key, value in dt.get('length_parameters', {}).items()
                                if '%' not in str(value)}
        checks.append((dt, name))
    all_names = list(dict.fromkeys([*names, *ordinary]))

    def placeholder(ordinary_names):
        mapping = {**ordinary_names, **names}
        return ['call ' + _PLACEHOLDER + '(' + ', '.join(mapping.get(n, n) for n in all_names) + ')']

    for dt, name in checks:
        current = {**dt, 'length_parameters': _translated_parameters(dt, names)}
        lines = _check_lengths(current, name, status, 'f2py_pdt_length_buffer')
        if lines and 'optional' in rout['vars'][name].get('attrspec', []):
            lines = [f'if (c_associated({name})) then', *lines, 'end if']
        pre += lines
    template = [*pre, *base_body(placeholder), *post]
    for transform in universal:
        template = transform(template)
    uses, declarations, template = _expand(uses, declarations, template, pointers, status,
        write=writable, pointer=pointer_targets, optional=optional,
        owned=owned, allocatable=allocatable, deferred=deferred, lengths=early_lengths)

    def body(call):
        result = []
        pattern = re.compile(r'^(\s*)call ' + _PLACEHOLDER + r'\((.*)\)\s*$', re.I)
        for line in template:
            match = pattern.match(line)
            if match:
                expressions = _type_arguments('arguments(' + match[2] + ')')
                if len(expressions) != len(all_names):
                    raise ValueError('derived native operation argument count differs')
                result.extend(match[1] + text for text in call(dict(zip(all_names, expressions))))
            else:
                result.append(line)
        return result

    return list(dict.fromkeys(uses)), declarations, [], [], body, helpers
