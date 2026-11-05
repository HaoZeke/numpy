"""Keep parameterized native descriptors inside fixed scalar handle boxes."""

import hashlib
import re

from ._dt_helpers import (
    _can_wrap_opaque,
    _enumerate_specializations,
    _get_kind_param_info,
    _get_len_param_info,
    _module_types,
)


def _type_arguments(spec):
    """Split a generated type specification without interpreting values."""
    tail = spec.partition('(')[2]
    if not tail:
        return []
    depth = 0
    start = 0
    result = []
    for index, char in enumerate(tail[:-1]):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            result.append(tail[start:index].strip())
            start = index + 1
    result.append(tail[start:-1].strip())
    return result


def _handle_description(block, spec=None, kind_override=None):
    """Describe one canonical owner and concrete KIND specialization."""
    kinds = _get_kind_param_info(block)
    lengths = _get_len_param_info(block)
    if not kinds and not lengths:
        return None
    native = block.get('f2py_origin_name',
                       block.get('f2py_native_name', block['name'])).lower()
    owner = block.get('f2py_origin_module')
    if owner is None:
        parent = block.get('parent_block', {})
        owner = parent.get('name')
    if not owner:
        raise ValueError('a parameterized handle requires its declaring module')
    owner = owner.lower()
    values = {item['name']: item['default'] for item in kinds}
    if kind_override is not None:
        values.update(kind_override)
    if spec:
        parameters = [name for name, var in block.get('vars', {}).items()
                      if {'kind', 'len'} & set(var.get('attrspec', []))]
        for position, argument in enumerate(_type_arguments(spec)):
            name, separator, value = argument.partition('=')
            if not separator:
                if position >= len(parameters):
                    raise ValueError('too many generated type parameters')
                name, value = parameters[position], name
            name = name.strip().lower()
            if name in values:
                values[name] = int(value.strip())
    if any(value is None for value in values.values()):
        raise ValueError('a parameterized handle requires concrete KIND values')
    identity = owner + ':' + native + ':' + ','.join(
        name + '=' + str(values[name]) for name in sorted(values))
    token = hashlib.sha256(identity.encode('ascii')).hexdigest()[:24]
    prefix = 'f2py_h_' + token
    local_type = 'f2py_type_' + token
    kind_args = [item['name'] + '=' + str(values[item['name']]) for item in kinds]

    def type_spec(length):
        args = kind_args + [item['name'] + '=' + length for item in lengths]
        return local_type + '(' + ', '.join(args) + ')'

    bindings = {}
    for item in block.get('f2py_pdt_bindings', []):
        if item['identity_number'] != int(token[:15], 16):
            continue
        storage_id = item.get('storage_id', item['id'])
        bindings.setdefault(storage_id, {**item, 'id': storage_id})
    return dict(owner=owner, native=native, kinds=values, lengths=lengths,
                identity=identity, module='f2py_handles_' + token,
                prefix=prefix, box=prefix + '_box',
                local_type=local_type,
                identity_number=int(token[:15], 16),
                kind_value=next(iter(values.values())) if len(values) == 1 else 0,
                bindings=list(bindings.values()),
                pointer_spec=type_spec(':'), dummy_spec=type_spec('*'))


def _handle_runtime():
    """Emit the interoperable envelope shared by every PDT storage mode."""
    return '''module f2py_pdt_handle_runtime
  use iso_c_binding
  implicit none
  type, bind(c) :: f2py_pdt_handle
    integer(c_int64_t) :: identity = 0
    integer(c_int) :: storage = 0
    integer(c_int) :: binding = 0
    integer(c_int) :: kind_value = 0
    integer(c_int) :: writable = 1
    integer(c_int) :: pointer_binding = 0
    integer(c_int) :: target_binding = 0
    integer(c_int) :: length_count = 0
    integer(c_int) :: rank = 0
    integer(c_size_t) :: subscripts(15) = 0
    type(c_ptr) :: lengths = c_null_ptr
    type(c_ptr) :: payload = c_null_ptr
    type(c_funptr) :: release = c_null_funptr
    type(c_funptr) :: validate = c_null_funptr
  end type f2py_pdt_handle
  abstract interface
    subroutine f2py_pdt_releaser(handle) bind(c)
      import c_ptr
      type(c_ptr), value :: handle
    end subroutine f2py_pdt_releaser
    integer(c_int) function f2py_pdt_validator(handle) bind(c)
      import c_ptr, c_int
      type(c_ptr), value :: handle
    end function f2py_pdt_validator
  end interface
contains
  function f2py_pdt_header(handle) result(header)
    type(c_ptr), value :: handle
    type(f2py_pdt_handle), pointer :: header
    nullify(header)
    if (c_associated(handle)) call c_f_pointer(handle, header)
  end function f2py_pdt_header

  function f2py_pdt_enclose(payload, identity, kind_value, lengths, release) result(handle)
    type(c_ptr), value :: payload
    integer(c_int64_t), value :: identity
    integer(c_int), value :: kind_value
    integer(c_int), intent(in) :: lengths(:)
    type(c_funptr), value :: release
    type(c_ptr) :: handle
    type(f2py_pdt_handle), pointer :: header
    integer(c_int), pointer :: values(:)
    integer :: status
    handle = c_null_ptr
    allocate(header, stat=status)
    if (status /= 0) return
    if (size(lengths) > 0) then
      allocate(values(size(lengths)), stat=status)
      if (status /= 0) then
        deallocate(header)
        return
      end if
      values = lengths
      header%lengths = c_loc(values)
    end if
    header%identity = identity
    header%kind_value = kind_value
    header%payload = payload
    header%length_count = size(lengths)
    header%release = release
    handle = c_loc(header)
  end function f2py_pdt_enclose

  subroutine f2py_pdt_release_header(header)
    type(f2py_pdt_handle), pointer :: header
    integer(c_int), pointer :: values(:)
    if (header%length_count > 0 .and. c_associated(header%lengths)) then
      call c_f_pointer(header%lengths, values, [header%length_count])
      deallocate(values)
    end if
    deallocate(header)
  end subroutine f2py_pdt_release_header

  subroutine f2py_pdt_destroy(handle) bind(c)
    type(c_ptr), value :: handle
    type(f2py_pdt_handle), pointer :: header
    procedure(f2py_pdt_releaser), pointer :: release
    header => f2py_pdt_header(handle)
    if (.not. associated(header)) return
    if (.not. c_associated(header%release)) return
    call c_f_procpointer(header%release, release)
    call release(handle)
  end subroutine f2py_pdt_destroy

  integer(c_int) function f2py_pdt_storage(handle) result(storage) bind(c)
    type(c_ptr), value :: handle
    type(f2py_pdt_handle), pointer :: header
    storage = -1
    header => f2py_pdt_header(handle)
    if (associated(header)) storage = header%storage
  end function f2py_pdt_storage

  integer(c_int) function f2py_pdt_writable(handle) result(writable) bind(c)
    type(c_ptr), value :: handle
    type(f2py_pdt_handle), pointer :: header
    writable = 0
    header => f2py_pdt_header(handle)
    if (associated(header)) writable = header%writable
  end function f2py_pdt_writable

  integer(c_int) function f2py_pdt_validate(handle) result(status) bind(c)
    type(c_ptr), value :: handle
    type(f2py_pdt_handle), pointer :: header
    procedure(f2py_pdt_validator), pointer :: validate
    status = 1
    header => f2py_pdt_header(handle)
    if (.not. associated(header)) return
    if (header%storage == 0) then
      if (c_associated(header%payload)) status = 0
    else if (header%storage == 1 .and. c_associated(header%validate)) then
      call c_f_procpointer(header%validate, validate)
      status = validate(handle)
    end if
  end function f2py_pdt_validate

  integer(c_int64_t) function f2py_pdt_identity(handle) result(identity) bind(c)
    type(c_ptr), value :: handle
    type(f2py_pdt_handle), pointer :: header
    identity = 0
    header => f2py_pdt_header(handle)
    if (associated(header)) identity = header%identity
  end function f2py_pdt_identity

  integer(c_int) function f2py_pdt_kind(handle) result(kind_value) bind(c)
    type(c_ptr), value :: handle
    type(f2py_pdt_handle), pointer :: header
    kind_value = 0
    header => f2py_pdt_header(handle)
    if (associated(header)) kind_value = header%kind_value
  end function f2py_pdt_kind

  integer(c_int) function f2py_pdt_length(handle, index) result(length) bind(c)
    type(c_ptr), value :: handle
    integer(c_int), value :: index
    type(f2py_pdt_handle), pointer :: header
    integer(c_int), pointer :: values(:)
    length = -1
    header => f2py_pdt_header(handle)
    if (.not. associated(header)) return
    if (index < 0 .or. index >= header%length_count) return
    call c_f_pointer(header%lengths, values, [header%length_count])
    length = values(index + 1)
  end function f2py_pdt_length
end module f2py_pdt_handle_runtime
'''


def _handle_module(description):
    """Emit one fixed nonparameterized box and its native operations."""
    prefix = description['prefix']
    box = description['box']
    pointer = description['pointer_spec']
    dummy = description['dummy_spec']
    module = description['module']
    lengths = ', '.join('value%' + item['name'] for item in description['lengths'])
    imports = {}
    for binding in description.get('bindings', []):
        if binding['pointer'] or binding['target']:
            imports.setdefault(binding['module'], set()).add(
                binding['alias'] + ' => ' + binding['variable'])
    alias_uses = '\n'.join('  use ' + owner + ', only: ' + ', '.join(sorted(items))
                           for owner, items in sorted(imports.items()))

    def same_pointer(side, handle, header):
        lines = [f'    if ({header}%storage == 0) then',
                 f'      {side} => {prefix}_unwrap({handle})']
        bindings = [record for record in description.get('bindings', [])
                    if record['pointer'] or record['target']]
        if bindings:
            lines += [f'    else if ({header}%storage == 1) then',
                      f'      select case ({header}%binding)']
            for record in bindings:
                expression = record['alias']
                if record['rank']:
                    indices = [
                        f'lbound({expression}, {axis}, kind=c_intptr_t) + '
                        f'int({header}%subscripts({axis}), kind=c_intptr_t)'
                        for axis in range(1, record['rank'] + 1)]
                    expression += '(' + ', '.join(indices) + ')'
                lines += [f'      case ({record["id"]})',
                          f'        {side} => {expression}']
            lines += ['      end select']
        return '\n'.join(lines + ['    end if'])

    same_left = same_pointer('a', 'left', 'ha')
    same_right = same_pointer('b', 'right', 'hb')
    source = f'''module {module}
  use {description['owner']}, only: {description['local_type']} => {description['native']}
{alias_uses}
  use iso_c_binding, only: c_ptr, c_null_ptr, c_f_pointer, c_loc, c_associated, &
    c_int, c_int64_t, c_size_t, c_intptr_t, c_funloc, c_funptr
  use f2py_pdt_handle_runtime, only: f2py_pdt_handle, f2py_pdt_header, &
    f2py_pdt_enclose, f2py_pdt_release_header, f2py_pdt_validate
  implicit none
  private
  public :: {prefix}_unwrap, {prefix}_adopt, {prefix}_copy
  public :: {prefix}_free, {prefix}_same, {prefix}_bind
  public :: {description['local_type']}
  type :: {box}
    type({pointer}), pointer :: value => null()
    logical :: owns_value = .false.
  end type {box}
contains
  function {prefix}_unwrap(handle) result(value)
    type(c_ptr), value :: handle
    type({pointer}), pointer :: value
    type({box}), pointer :: box
    type(f2py_pdt_handle), pointer :: header
    nullify(value)
    header => f2py_pdt_header(handle)
    if (.not. associated(header)) return
    if (header%identity /= {description['identity_number']}_c_int64_t) return
    if (header%storage /= 0) return
    call c_f_pointer(header%payload, box)
    value => box%value
  end function {prefix}_unwrap

  function {prefix}_adopt(value) result(handle)
    type({pointer}), pointer, intent(inout) :: value
    type(c_ptr) :: handle
    type({box}), pointer :: box
    integer :: status
    handle = c_null_ptr
    if (.not. associated(value)) return
    allocate(box, stat=status)
    if (status /= 0) then
      deallocate(value)
      return
    end if
    box%value => value
    box%owns_value = .true.
    handle = f2py_pdt_enclose(c_loc(box), &
      {description['identity_number']}_c_int64_t, {description['kind_value']}, &
      [integer(c_int) :: {lengths}], c_funloc({prefix}_free))
    if (.not. c_associated(handle)) then
      deallocate(box%value)
      deallocate(box)
      return
    end if
  end function {prefix}_adopt

  function {prefix}_copy(value) result(handle)
    type({dummy}), intent(in) :: value
    type({pointer}), pointer :: duplicate
    type(c_ptr) :: handle
    integer :: status
    handle = c_null_ptr
    allocate(duplicate, source=value, stat=status)
    if (status /= 0) return
    handle = {prefix}_adopt(duplicate)
  end function {prefix}_copy

  function {prefix}_bind(binding, offsets, lengths, validate) result(handle)
    integer(c_int), value :: binding
    integer(c_size_t), intent(in) :: offsets(:)
    integer(c_int), intent(in) :: lengths(:)
    type(c_funptr), value :: validate
    type(c_ptr) :: handle
    type(f2py_pdt_handle), pointer :: header
    handle = c_null_ptr
    if (size(offsets) > 15) return
    handle = f2py_pdt_enclose(c_null_ptr, &
      {description['identity_number']}_c_int64_t, {description['kind_value']}, &
      lengths, c_funloc({prefix}_free))
    if (.not. c_associated(handle)) return
    header => f2py_pdt_header(handle)
    header%storage = 1
    header%binding = binding
    header%rank = size(offsets)
    header%subscripts(:size(offsets)) = offsets
    header%validate = validate
  end function {prefix}_bind

  subroutine {prefix}_free(handle) bind(c)
    type(c_ptr), value :: handle
    type({box}), pointer :: box
    type(f2py_pdt_handle), pointer :: header
    header => f2py_pdt_header(handle)
    if (.not. associated(header)) return
    if (header%identity /= {description['identity_number']}_c_int64_t) return
    if (header%storage == 0 .and. c_associated(header%payload)) then
      call c_f_pointer(header%payload, box)
      if (box%owns_value .and. associated(box%value)) deallocate(box%value)
      deallocate(box)
    end if
    call f2py_pdt_release_header(header)
  end subroutine {prefix}_free

  logical function {prefix}_same(left, right) result(same)
    type(c_ptr), value :: left, right
    type({pointer}), pointer :: a, b
    type(f2py_pdt_handle), pointer :: ha, hb
    same = .false.
    ha => f2py_pdt_header(left)
    hb => f2py_pdt_header(right)
    if (.not. associated(ha) .or. .not. associated(hb)) return
    if (ha%identity /= {description['identity_number']}_c_int64_t) return
    if (hb%identity /= ha%identity) return
    if (f2py_pdt_validate(left) /= 0) return
    if (f2py_pdt_validate(right) /= 0) return
    if (ha%storage == 1 .and. hb%storage == 1) then
      if (ha%binding == hb%binding .and. ha%rank == hb%rank) then
        same = all(ha%subscripts == hb%subscripts)
        if (same) return
      end if
    end if
    nullify(a, b)
{same_left}
{same_right}
    if (associated(a) .and. associated(b)) same = associated(a, b)
  end function {prefix}_same
end module {module}
'''
    return _kind_only_module(source, description) if not description['lengths'] else source


def _kind_only_module(source, description):
    """Store LEN-free native targets without a nested PDT pointer component."""
    prefix = description['prefix']
    box = description['box']
    pointer = description['pointer_spec']
    declaration = f'''  type :: {box}
    type({pointer}), pointer :: value => null()
    logical :: owns_value = .false.
  end type {box}
'''
    if source.count(declaration) != 1:
        raise ValueError('fixed holder declaration is not unique')
    source = source.replace(declaration, '')
    start = source.index(f'  function {prefix}_adopt(')
    end = source.index(f'  end function {prefix}_adopt', start)
    end = source.index('\n', end) + 1
    adopt = f'''  function {prefix}_adopt(value) result(handle)
    type({pointer}), pointer, intent(inout) :: value
    type(c_ptr) :: handle
    handle = c_null_ptr
    if (.not. associated(value)) return
    handle = f2py_pdt_enclose(c_loc(value), &
      {description['identity_number']}_c_int64_t, {description['kind_value']}, &
      [integer(c_int) :: ], c_funloc({prefix}_free))
    if (.not. c_associated(handle)) deallocate(value)
  end function {prefix}_adopt
'''
    source = source[:start] + adopt + source[end:]
    start = source.index(f'  subroutine {prefix}_free(')
    end = source.index(f'  end subroutine {prefix}_free', start)
    end = source.index('\n', end) + 1
    free = f'''  subroutine {prefix}_free(handle) bind(c)
    type(c_ptr), value :: handle
    type({pointer}), pointer :: value
    type(f2py_pdt_handle), pointer :: header
    header => f2py_pdt_header(handle)
    if (.not. associated(header)) return
    if (header%identity /= {description['identity_number']}_c_int64_t) return
    if (header%storage == 0 .and. c_associated(header%payload)) then
      call c_f_pointer(header%payload, value)
      if (associated(value)) deallocate(value)
    end if
    call f2py_pdt_release_header(header)
  end subroutine {prefix}_free
'''
    source = source[:start] + free + source[end:]
    source = source.replace(f'    type({box}), pointer :: box\n', '')
    cast = '    call c_f_pointer(header%payload, box)\n    value => box%value\n'
    if source.count(cast) != 1:
        raise ValueError('fixed holder reconstruction is not unique')
    return source.replace(cast, '    call c_f_pointer(header%payload, value)\n')


def _handle_uses(type_map):
    """Import canonical helpers for local and imported parameterized types."""
    modules = set()
    for block in type_map.values():
        for kinds, _ in _enumerate_specializations(block) or [(None, '')]:
            info = _handle_description(block, kind_override=kinds)
            if info is not None:
                modules.add(info['module'])
    return (['use f2py_pdt_handle_runtime'] if modules else []) + [
        'use ' + module for module in sorted(modules)]


def _handle_modules(modules):
    """Emit helper definitions ahead of every generated consumer."""
    definitions = {}
    for module in modules:
        type_map, blocks, _ = _module_types(module)
        for block in blocks:
            if not _can_wrap_opaque(block, type_map):
                continue
            for kinds, _ in _enumerate_specializations(block) or [(None, '')]:
                info = _handle_description(block, kind_override=kinds)
                if info is not None:
                    definitions[info['identity']] = _handle_module(info)
    return _fortran_source(_handle_runtime() + '\n' + '\n'.join(
        definitions[name] for name in sorted(definitions))) if definitions else ''


def _fortran_source(source):
    """Continue generated statements outside literals within the source limit."""
    result = []
    for line in source.splitlines():
        if line.lstrip().startswith('!'):
            while len(line) > 120:
                split = line.rfind(' ', 0, 116)
                if split < 2:
                    break
                result.append(line[:split])
                line = '! ' + line[split:].lstrip()
        while len(line) > 120:
            quote = None
            cuts = []
            index = 0
            while index < min(112, len(line)):
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
            result.append(line[:split].rstrip() + ' &')
            line = '  & ' + line[split:].lstrip()
        result.append(line)
    return '\n'.join(result) + '\n'


def _handle_from_c(handle, target, block, spec=None):
    info = _handle_description(block, spec)
    if info is None:
        return f'call c_f_pointer({handle}, {target})'
    return f"{target} => {info['prefix']}_unwrap({handle})"


def _handle_address(target, block, spec=None):
    info = _handle_description(block, spec)
    if info is None:
        return f'c_loc({target})'
    return f"{info['prefix']}_adopt({target})"


def _handle_free(handle, target, block, spec=None):
    info = _handle_description(block, spec)
    if info is None:
        return f'deallocate({target})'
    return f"call {info['prefix']}_free({handle})"


def _handle_same(left, right, block, spec=None):
    info = _handle_description(block, spec)
    if info is None:
        return f'c_associated({left}, {right})'
    return f"{info['prefix']}_same({left}, {right})"


def _handle_statements(lines, variables):
    """Render generated routine statements with their declared handle kinds.

    The inputs are statements assembled by the routine templates, never
    user Fortran source. Intrinsic buffers and nonparameterized objects
    retain their existing representation.
    """
    rendered = []
    for line in lines:
        cast = re.search(r'\bcall c_f_pointer\((.*)\)\s*$', line)
        if cast:
            args = _type_arguments('arguments(' + cast[1] + ')')
            info = variables.get(args[1].strip()) if len(args) == 2 else None
            if info is not None:
                target = args[1].strip()
                line = (line[:cast.start()] + target + ' => '
                        + info['prefix'] + '_unwrap(' + args[0] + ')')
        for target, info in variables.items():
            if info is not None:
                line = line.replace('c_loc(' + target + ')',
                                    info['prefix'] + '_adopt(' + target + ')')
        rendered.append(line)
    return rendered


def _replace_fortran_name(lines, name, expression):
    """Replace a generated object reference, preserving members and strings."""
    rendered = []
    for line in lines:
        result = []
        position = 0
        quote = None
        while position < len(line):
            char = line[position]
            if quote is not None:
                result.append(char)
                position += 1
                if char == quote:
                    if position < len(line) and line[position] == quote:
                        result.append(line[position])
                        position += 1
                    else:
                        quote = None
                continue
            if char in "\"'":
                quote = char
            if char == '!':
                result.append(line[position:])
                break
            if char.isalpha() or char == '_':
                end = position + 1
                while end < len(line) and (line[end].isalnum() or line[end] == '_'):
                    end += 1
                token = line[position:end]
                previous = line[:position].rstrip()
                if token.lower() == name.lower() and not previous.endswith('%'):
                    result.append(expression)
                else:
                    result.append(token)
                position = end
                continue
            result.append(char)
            position += 1
        rendered.append(''.join(result))
    return rendered


def _handle_operation_parts(uses, declarations, body, pointers, *,
                            write_targets=(), pointer_targets=None,
                            optional_targets=(), failure=('return',)):
    """Dispatch a whole native operation to owned objects or module variables.

    ``pointers`` maps a generated local name to ``handle`` and a canonical
    ``description``. The body contains the operation without C-pointer
    reconstruction. A module branch uses the actual variable expression;
    no pointer association with non-TARGET storage escapes the operation.
    ``pointer_targets`` maps pointer-dummy locals to whether TARGET actuals
    are permitted by their INTENT(IN) interface.
    """
    pointer_targets = pointer_targets or {}
    uses = list(uses)
    declarations = list(declarations)
    body = list(body)
    reserved = ' '.join([*uses, *declarations, *body])
    prefix = 'f2py_storage'
    while prefix in reserved:
        prefix += 'x'
    entries = list(pointers.items())
    headers = {}
    modules = set()
    imports = {}
    for index, (local, pointer) in enumerate(entries):
        info = pointer['description']
        modules.add(info['module'])
        header = f'{prefix}_{index}'
        headers[local] = header
        declarations.append(f'type(f2py_pdt_handle), pointer :: {header}')
        for binding in info.get('bindings', []):
            imports.setdefault(binding['module'], set()).add(
                binding['alias'] + ' => ' + binding['variable'])
        body = [line for line in body
                if line.strip().lower() != f'nullify({local.lower()})']
    if entries:
        uses.append('use f2py_pdt_handle_runtime')
    uses.extend('use ' + module for module in sorted(modules))
    uses.extend('use ' + module + ', only: ' + ', '.join(sorted(items))
                for module, items in sorted(imports.items()))

    def expand(index, statements):
        if index == len(entries):
            return statements
        local, pointer = entries[index]
        handle = pointer['handle']
        info = pointer['description']
        header = headers[local]
        result = [f'{header} => f2py_pdt_header({handle})',
                  f'if (.not. associated({header})) then']
        if local in optional_targets:
            result += ['  ' + line for line in [f'nullify({local})',
                       *expand(index + 1, statements)]]
        else:
            result += ['  ' + line for line in failure]
        result += ['else',
                   f'  if ({header}%identity /= {info["identity_number"]}_c_int64_t) then',
                   *['    ' + line for line in failure], '  end if',
                   f'  if (f2py_pdt_validate({handle}) /= 0) then',
                   *['    ' + line for line in failure], '  end if',
                   f'  select case ({header}%storage)', '  case (0)',
                   f"    {local} => {info['prefix']}_unwrap({handle})"]
        result += ['    ' + line for line in expand(index + 1, statements)]
        bindings = info.get('bindings', [])
        if bindings:
            result += ['  case (1)', f'    select case ({header}%binding)']
            for binding in bindings:
                if local in write_targets and binding['readonly']:
                    continue
                if local in pointer_targets and not (binding['pointer'] or (
                        pointer_targets[local] and binding['target'])):
                    continue
                expression = binding['alias']
                if binding['rank']:
                    indices = [
                        f'lbound({expression}, {axis}, kind=c_intptr_t) + '
                        f'int({header}%subscripts({axis}), kind=c_intptr_t)'
                        for axis in range(1, binding['rank'] + 1)]
                    expression += '(' + ', '.join(indices) + ')'
                selected = _replace_fortran_name(statements, local, expression)
                result += [f"    case ({binding['id']})"]
                result += ['      ' + line for line in expand(index + 1, selected)]
            result += ['    case default', *['      ' + line for line in failure],
                       '    end select']
        result += ['  case default', *['    ' + line for line in failure],
                   '  end select', 'end if']
        return result

    return list(dict.fromkeys(uses)), declarations, expand(0, body)
