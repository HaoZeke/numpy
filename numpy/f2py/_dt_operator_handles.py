"""Dispatch parameterized operators without borrowing module storage."""

from ._dt_handles import (
    _handle_description,
    _handle_operation_parts,
    _replace_fortran_name,
    _type_arguments,
)
from ._dt_helpers import (
    _enumerate_specializations,
    _get_kind_param_info,
    _get_len_param_info,
    _get_member_isoc_type,
    _member_actual,
    _member_value,
    _routine_args,
)
from .auxfuncs import isfunction
from .crackfortran import _fortran_eval, get_parameters, get_useparameters


def _operator_parameters(proc, type_map):
    """Resolve constants in the procedure's lexical module and local scope."""
    block = proc
    visited = set()
    while block and id(block) not in visited:
        visited.add(id(block))
        if block.get('block') == 'module':
            break
        block = block.get('parent_block')
    if not block or block.get('block') != 'module':
        module_name = proc.get('modulename', '').lower()
        parents = {id(parent): parent for item in type_map.values()
                   if (parent := item.get('parent_block', {})).get('block')
                   == 'module'}
        matching = [parent for parent in parents.values()
                    if parent.get('name', '').lower() == module_name]
        if len(matching) == 1:
            block = matching[0]
        elif len(parents) == 1:
            block = next(iter(parents.values()))
        else:
            raise ValueError('operator procedure has no unique lexical module')
    params = get_parameters(block.get('vars', {}), get_useparameters(block))
    return get_parameters(proc.get('vars', {}), get_useparameters(proc, params))


def _operator_integer(expression, params):
    """Return a resolved integer parameter, retaining unresolved LEN syntax."""
    try:
        value = _fortran_eval(str(expression), params, params)
    except (NameError, SyntaxError, TypeError, ValueError, OverflowError, KeyError):
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def _operator_type_info(var, type_map, params):
    """Describe the canonical owner, specialization and declared LEN limits."""
    if var.get('typespec') != 'type':
        return None
    block = type_map[var['typename'].lower()]
    kinds = _get_kind_param_info(block)
    lengths = _get_len_param_info(block)
    if not kinds and not lengths:
        return None
    order = [name.lower() for name, value in block.get('vars', {}).items()
             if {'kind', 'len'} & set(value.get('attrspec', []))]
    parameters = {}
    raw = var.get('typeparams') or ''
    for position, argument in enumerate(_type_arguments('value(' + raw + ')')):
        if not argument:
            continue
        name, separator, value = argument.partition('=')
        if not separator:
            if position >= len(order):
                raise ValueError('too many operator type parameters')
            name, value = order[position], name
        name = name.strip().lower()
        if name not in order or name in parameters:
            raise ValueError('invalid or repeated operator type parameter')
        parameters[name] = value.strip()
    owner = block.get('parent_block', {})
    owner_params = get_parameters(owner.get('vars', {}), get_useparameters(owner))
    values = {}
    for item in kinds:
        name = item['name']
        expression = parameters.get(name, block['vars'][name].get('='))
        scope = params if name in parameters else {**owner_params, **values}
        value = _operator_integer(expression, scope)
        if value is None:
            raise ValueError('operator requires a concrete KIND: ' + name)
        values[name] = value
    variants = _enumerate_specializations(block) if kinds else [(values, '')]
    suffix = next((suffix for candidate, suffix in variants
                   if candidate == values), None)
    if suffix is None:
        raise ValueError('operator selects an unwrapped KIND specialization')
    description = _handle_description(block, kind_override=values)
    limits = []
    length_parameters = {}
    for item in lengths:
        name = item['name']
        expression = parameters.get(name, block['vars'][name].get('='))
        if expression is None:
            raise ValueError('operator LEN parameter is unspecified: ' + name)
        expression = str(expression).strip()
        length_parameters[name] = expression
        if expression not in ('*', ':'):
            scope = params if name in parameters else {**owner_params, **values}
            limits.append(dict(name=name, expression=expression,
                               constant=_operator_integer(expression, scope)))
    attrs = set(var.get('attrspec', []))
    return dict(description=description, lengths=limits,
                length_parameters=length_parameters,
                deferred_lengths=[name for name, expression in length_parameters.items()
                                  if expression == ':'],
                capsule='f2py.' + block.get('f2py_c_name', block['name']) + suffix,
                pointer='pointer' in attrs, allocatable='allocatable' in attrs,
                intent_in='in' in var.get('intent', []))


def _operator_handle_signature(proc, type_map):
    """Share exact PDT identities and the trailing status ABI with C dispatch."""
    variables = [proc['vars'][name] for name in _routine_args(proc)]
    result = (proc['vars'].get(proc.get('result', proc['name']), {})
              if isfunction(proc) else variables[0])
    parameterized = any(
        var.get('typespec') == 'type' and (
            _get_kind_param_info(type_map[var['typename'].lower()])
            or _get_len_param_info(type_map[var['typename'].lower()]))
        for var in [*variables, result])
    if not parameterized:
        return dict(active=False, args=[None] * len(variables), result=None)
    params = _operator_parameters(proc, type_map)
    return dict(active=True,
                args=[_operator_type_info(var, type_map, params)
                      for var in variables],
                result=_operator_type_info(result, type_map, params))


def _operator_actual_spec(info, local, names, arguments):
    """Declare a block-local actual with the dummy's deferred LEN modes."""
    description = info['description']
    parameters = [name + '=' + str(value)
                  for name, value in description['kinds'].items()]
    limits = {item['name']: item for item in info['lengths']}
    for name, expression in info['length_parameters'].items():
        if expression == '*':
            expression = local + '%' + name
        elif expression != ':':
            limit = limits[name]
            if limit['constant'] is not None:
                expression = str(limit['constant'])
            else:
                for dummy, argument in zip(names, arguments):
                    expression = _replace_fortran_name([expression], dummy, argument)[0]
        parameters.append(name + '=' + expression)
    return description['local_type'] + '(' + ', '.join(parameters) + ')'


def _operator_constraint_parts(statements):
    boundary = statements.index('! Operator operand constraints hold.') + 1
    return statements[:boundary], statements[boundary:]


def _gen_handle_operator(typename, pname, proc, result_kind, type_map, args, result):
    """Generate a status-bearing whole-operation wrapper for PDT operands."""
    from ._dt_operators import (
        _operator_actual_call,
        _operator_assignment_call,
        _operator_kind,
    )

    signature = _operator_handle_signature(proc, type_map)
    if not signature['active']:
        return None
    reserved = {pname, *proc.get('vars', {}), *type_map}
    prefix = 'f2py_op'
    while any(name.startswith(prefix + '_') for name in reserved):
        prefix += 'x'
    dummy = [f'{prefix}_arg{i}' for i in range(len(args))]
    actual = [f'{prefix}_operand{i}' if _operator_kind(var) == 't'
              else dummy[i] for i, var in enumerate(args)]
    returned = prefix + '_result'
    status = prefix + '_status'
    allocated = prefix + '_allocated'
    value = prefix + '_value'
    rhs = prefix + '_rhs'
    uses = ['use iso_c_binding']
    declarations = [f'integer(c_int), intent(out) :: {status}',
                    f'integer :: {allocated}']
    pointers = {}
    pointer_targets = {}
    allocatable_targets = []
    deferred_targets = {}
    length_targets = {}
    owned_bodies = {}
    casts = []
    snapshots = []
    snapshot_declarations = []
    arguments = []
    native_names = _routine_args(proc)
    native_arguments = [local if _operator_kind(var) == 't'
                        else _member_actual(var, local)
                        for local, var in zip(actual, args)]
    for index, (arg, local, var, info) in enumerate(
            zip(dummy, actual, args, signature['args'])):
        if _operator_kind(var) != 't':
            declarations.append(f'{_get_member_isoc_type(var)}, value :: {arg}')
            arguments.append(_member_actual(var, arg))
            continue
        declarations.append(f'type(c_ptr), value :: {arg}')
        spec = info['description']['pointer_spec'] if info else var['typename']
        declarations.append(f'type({spec}), pointer :: {local}')
        if info:
            pointers[local] = dict(handle=arg, description=info['description'])
            if info['pointer']:
                pointer_targets[local] = info['intent_in']
                deferred_targets[local] = set(info['deferred_lengths'])

                def pointer_actual(statements, *, local=local, arg=arg, info=info):
                    constraints, operation = _operator_constraint_parts(statements)
                    if not info['intent_in']:
                        return [*constraints, f'{status} = 4_c_int', 'return']
                    alias = local + '_pointer'
                    spec = _operator_actual_spec(info, local, native_names,
                                                 native_arguments)
                    selected = _replace_fortran_name(operation, local, alias)
                    return [*constraints, 'block',
                            f'  type({spec}), pointer :: {alias}',
                            f'  {alias} => {local}',
                            *['  ' + line for line in selected], 'end block']

                owned_bodies[local] = pointer_actual
        else:
            casts.append(f'call c_f_pointer({arg}, {local})')
        if info and info['allocatable']:
            copy = f'{prefix}_copy{index}'
            allocation_spec = _operator_actual_spec(info, local, native_names,
                                                    native_arguments)
            if result_kind == 'assignment' and index == 0:
                allocatable_targets.append(local)
                deferred_targets[local] = set(info['deferred_lengths'])

                def owned(statements, *, local=local, copy=copy, handle=arg,
                          description=info['description'], spec=allocation_spec):
                    constraints, operation = _operator_constraint_parts(statements)
                    selected = _replace_fortran_name(operation, local, copy)
                    return [
                        *constraints, 'block', f'  type({spec}), allocatable :: {copy}',
                        f'  allocate({copy}, source={local}, stat={allocated})',
                        f'  if ({allocated} /= 0) then', f'    {status} = 1_c_int',
                        '    return', '  end if', *['  ' + line for line in selected],
                        f'  if (allocated({copy})) then',
                        (f'    {status} = {description["prefix"]}_replace('
                         f'{handle}, {copy})'), '  else',
                        f'    {status} = {description["prefix"]}_clear({handle})',
                        '  end if', f'  if ({status} /= 0) return', 'end block']

                owned_bodies[local] = owned
                arguments.append(local)
                continue
            if not info['intent_in']:
                raise ValueError('operator function operands require INTENT(IN)')
            snapshot_declarations.append(
                f'type({allocation_spec}), allocatable :: {copy}')
            snapshots += [f'allocate({copy}, source={local}, stat={allocated})',
                          f'if ({allocated} /= 0) then',
                          f'  {status} = 1_c_int', '  return', 'end if']
            arguments.append(copy)
        else:
            arguments.append(local)
    result_info = signature['result']
    if result_info:
        uses.append('use ' + result_info['description']['module'])
    if result_kind in ('assignment', 'logical'):
        declarations.append(f'integer(c_int) :: {returned}')
        initial = '4_c_int' if result_kind == 'assignment' else '0_c_int'
    elif result_kind == 'type':
        spec = (result_info['description']['pointer_spec'] if result_info
                else result['typename'])
        declarations += [f'type(c_ptr) :: {returned}',
                         f'type({spec}), pointer :: {value}']
        initial = 'c_null_ptr'
    else:
        declarations.append(f'{_get_member_isoc_type(result)} :: {returned}')
        initial = '0'
    body = []
    original = _routine_args(proc)
    for local, info in zip(actual, signature['args']):
        if info is None:
            continue
        for limit in info['lengths']:
            expression = limit['expression']
            if limit['constant'] is not None:
                expression = str(limit['constant'])
            else:
                for name, argument in zip(original, arguments):
                    expression = _replace_fortran_name([expression], name, argument)[0]
            body += [f'if ({local}%{limit["name"]} /= ({expression})) return']
            if limit['constant'] is not None:
                length_targets.setdefault(local, {})[limit['name']] = expression
    constraints = body
    body = []
    call = _operator_actual_call(proc, pname, arguments)
    if result_kind == 'assignment':
        left_info, right_info = signature['args'][:2]
        alias_copy = (_operator_kind(args[1]) == 't' and (
            left_info and right_info
            and left_info['description']['identity']
            == right_info['description']['identity']
            or not left_info and not right_info
            and args[0]['typename'].lower() == args[1]['typename'].lower()))
        assignment = _operator_assignment_call(proc, pname, arguments)
        if alias_copy:
            info = signature['args'][1]
            spec = info['description']['pointer_spec'] if info else args[1]['typename']
            storage = 'allocatable' if info and info['allocatable'] else 'pointer'
            if info and (info['pointer'] or info['allocatable']):
                spec = _operator_actual_spec(info, actual[1], native_names,
                                             native_arguments)
            left_info = signature['args'][0]
            if info and left_info:
                same = (info['description']['prefix'] + '_same('
                        + ', '.join(dummy[:2]) + ')'
                        if info['description']['identity']
                        == left_info['description']['identity'] else '.false.')
            elif not info and not left_info:
                same = 'c_associated(' + ', '.join(dummy[:2]) + ')'
            else:
                same = '.false.'
            copied = _operator_assignment_call(proc, pname, [arguments[0], rhs])
            body += [f'if ({same}) then', '  block',
                     f'    type({spec}), {storage} :: {rhs}',
                     f'    allocate({rhs}, source={actual[1]}, stat={allocated})',
                     f'    if ({allocated} /= 0) then', f'      {status} = 1_c_int',
                     '      return', '    end if', f'    {copied}',
                     f'    deallocate({rhs})', '  end block', 'else',
                     f'  {assignment}', 'end if', f'{returned} = 0_c_int']
        else:
            body += [assignment, f'{returned} = 0_c_int']
    elif result_kind == 'type':
        address = (result_info['description']['prefix'] + f'_adopt({value})'
                   if result_info else f'c_loc({value})')
        body += [f'allocate({value}, source={call}, stat={allocated})',
                 f'if ({allocated} /= 0) then', f'  {status} = 1_c_int',
                 '  return', 'end if', f'{returned} = {address}',
                 f'if (.not. c_associated({returned})) then',
                 f'  {status} = 1_c_int', '  return', 'end if']
    elif result_kind == 'logical':
        body.append(f'{returned} = merge(1_c_int, 0_c_int, {call})')
    else:
        body.append(f'{returned} = {_member_value(result, call)}')
    body.append(f'{status} = 0_c_int')
    if snapshots:
        body = ['block', *['  ' + line for line in snapshot_declarations],
                *['  ' + line for line in snapshots],
                *['  ' + line for line in body], 'end block']
    body = [*constraints, '! Operator operand constraints hold.', *body]
    writes = [local for local, info in zip(actual, signature['args'])
              if info and info['pointer']]
    if result_kind == 'assignment':
        writes.append(actual[0])
    uses, declarations, body = _handle_operation_parts(
        uses, declarations, body, pointers, write_targets=writes,
        pointer_targets=pointer_targets, owned_bodies=owned_bodies,
        allocatable_targets=allocatable_targets, deferred_targets=deferred_targets,
        length_targets=length_targets)
    wrapper = f'f2py_op_{typename}_{pname}'
    lines = [(f'  function {wrapper}({", ".join([*dummy, status])}) '
              f'result({returned}) bind(c)'),
             *['    ' + line for line in uses],
             *['    ' + line for line in declarations],
             f'    {status} = 4_c_int', f'    {returned} = {initial}',
             *['    ' + line for line in casts],
             *['    ' + line for line in body],
             f'  end function {wrapper}', '']
    if result_kind == 'type':
        handle = prefix + '_handle'
        lines += [f'  subroutine {wrapper}_free({handle}) bind(c)',
                  '    use iso_c_binding']
        if result_info:
            description = result_info['description']
            lines += ['    use ' + description['module'],
                      f'    type(c_ptr), value :: {handle}',
                      f'    call {description["prefix"]}_free({handle})']
        else:
            lines += [f'    type(c_ptr), value :: {handle}',
                      f'    type({result["typename"]}), pointer :: {value}',
                      f'    if (.not. c_associated({handle})) return',
                      f'    call c_f_pointer({handle}, {value})',
                      f'    deallocate({value})']
        lines += [f'  end subroutine {wrapper}_free', '']
    return lines
