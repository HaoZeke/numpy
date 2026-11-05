"""Match declared PDT LEN modes before generating native actual arguments."""

import re


def _constant(value):
    text = str(value).strip()
    return int(text) if re.fullmatch(r'[+-]?\d+', text) else None


def binding_compatible(binding, local, allocatable_targets, deferred_targets,
                       length_targets):
    if local in allocatable_targets and not (binding['allocatable'] and binding['rank'] == 0):
        return False
    if local in deferred_targets and (
            local in allocatable_targets or binding['pointer']):
        if set(binding.get('deferred_lengths', [])) != set(deferred_targets[local]):
            return False
    actual = binding.get('length_parameters', {})
    for name, expected in length_targets.get(local, {}).items():
        left, right = _constant(actual.get(name)), _constant(expected)
        if left is not None and right is not None and left != right:
            return False
    return True


def operation_length_checks(description, handle, expected, buffer, failure):
    lines = []
    for index, item in enumerate(description['lengths']):
        value = expected.get(item['name'])
        if value is None or str(value) in {'*', ':'}:
            continue
        lines += [f'if (f2py_pdt_length_value({handle}, {index}, {buffer}) /= 0) then',
                  *['  ' + line for line in failure], 'end if',
                  f'if (int({buffer}, c_int64_t) /= int({value}, c_int64_t)) then',
                  *['  ' + line for line in failure], 'end if']
    return ([f'integer(c_int) :: {buffer}'] if lines else []), lines
