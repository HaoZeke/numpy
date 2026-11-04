"""Keep native derived type names separate from generated identities."""

import copy
import hashlib
import re


def _prepare_type_namespaces(modules):
    """Qualify generated identities when native names occur in two modules."""
    entries = [(module, block) for module in modules
               for block in module.get('body', []) if block.get('block') == 'type']
    owners = {}
    for module, block in entries:
        owners.setdefault(block['name'].lower(), set()).add(module['name'].lower())
    reserved = {block['name'].lower() for _, block in entries}
    assigned = set(reserved)
    aliases = set(reserved)
    for module in modules:
        module['f2py_dt_namespace'] = {}
    for module, block in entries:
        name = block['name']
        native = name.lower()
        if len(owners[native]) == 1:
            block['f2py_c_name'] = name
            continue
        identity = module['name'].lower() + ':' + native
        digest = hashlib.sha256(identity.encode('ascii')).hexdigest()
        for length in range(24, 57, 4):
            generated = 'f2pyq_' + digest[:length]
            if generated not in assigned:
                break
        else:
            raise ValueError('derived type generated identity collision')
        assigned.add(generated)
        block['f2py_c_name'] = generated
        alias = module['name'].lower() + '_' + native
        if alias in aliases:
            alias = generated
        aliases.add(alias)
        block['f2py_python_name'] = alias
        module['f2py_dt_namespace'][name] = generated


def _type_in_namespace(types, module, name):
    """Resolve an exported name while retaining its declaring descriptor."""
    module, name = module.lower(), name.lower()
    direct = types.get((module, name), types.get(name))
    if direct is not None and direct['module'].lower() == module:
        return direct
    candidates = []
    for info in types.values():
        if any(owner.lower() == module and alias.lower() == name
               for owner, alias in info.get('public_aliases', ())):
            if all(info is not previous for previous in candidates):
                candidates.append(info)
    return candidates[0] if len(candidates) == 1 else None


def _find_type(types, name, module=None, uses=None, local_uses=None):
    """Resolve a native type through its declaring module or explicit USE."""
    if not name:
        return None
    name = name.lower()

    def imported(scope):
        candidates, explicit = [], False
        for used_module, mapping in scope.items():
            renamed = {key.lower(): value.lower()
                       for key, value in mapping.get('map', {}).items()}
            if name in renamed:
                remote = renamed[name]
                explicit = True
            elif mapping.get('only') or name in {
                    value for key, value in renamed.items() if key != value}:
                continue
            else:
                remote = name
            info = _type_in_namespace(types, used_module, remote)
            if info is not None:
                if all(info is not item for item in candidates):
                    candidates.append(info)
        return candidates, explicit

    candidates, explicit = imported(local_uses or {})
    if candidates or explicit:
        return candidates[0] if len(candidates) == 1 else None
    if module:
        module = module.lower()
        local = _type_in_namespace(types, module, name)
        if local is not None:
            return local
    candidates, explicit = imported(uses or {})
    if candidates or explicit or module:
        return candidates[0] if len(candidates) == 1 else None
    candidates = [info for info in types.values() if info['name'].lower() == name]
    return candidates[0] if len(candidates) == 1 else None


def _local_type_spec(info, spec):
    """Use a generated local alias without renaming the Fortran declaration."""
    native = info['name']
    generated = info.get('c_name', native)
    if generated == native:
        return spec
    if spec[:len(native)].lower() != native.lower():
        raise ValueError('derived type specification does not match its declaration')
    return generated + spec[len(native):]


def _type_use_name(spec, native):
    """Return the USE item importing the local type-specification alias."""
    alias = spec.partition('(')[0].strip()
    return native if alias.lower() == native.lower() else f'{alias} => {native}'


def _c_module(module):
    """Give only the C generator an internal copy of qualified declarations."""
    names = {**module.get('f2py_dt_import_names', {}),
             **module.get('f2py_dt_namespace', {})}
    if not names:
        return module
    names = {name.lower(): generated for name, generated in names.items()}
    result = {}
    memo = {id(module): result}
    context_fields = {'f2py_imported_types', 'f2py_c_imported_types'}
    for key, value in module.items():
        if key not in context_fields:
            result[key] = copy.deepcopy(value, memo)

    def attributes(var):
        attrs = []
        for attr in var.get('attrspec', []):
            match = re.fullmatch(r'extends\((\w+)\)', attr, flags=re.I)
            if match and match[1].lower() in names:
                attr = 'extends(' + names[match[1].lower()] + ')'
            attrs.append(attr)
        if 'attrspec' in var:
            var['attrspec'] = attrs

    def declarations(block):
        if block.get('block') == 'type' and block['name'].lower() in names:
            block['f2py_native_name'] = block['name']
            block['name'] = names[block['name'].lower()]
        for var in block.get('vars', {}).values():
            native = var.get('typename', '').lower()
            if native in names:
                var['typename'] = names[native]
            attributes(var)
        attributes(block)
        for child in block.get('body', []):
            declarations(child)

    declarations(result)
    result['f2py_imported_types'] = module.get('f2py_c_imported_types', {})
    result['vars'] = {names.get(name.lower(), name): var
                      for name, var in result.get('vars', {}).items()}
    return result


def _restore_python_exports(source, module):
    """Keep native module attribute names while qualifying colliding roots."""
    for block in module.get('body', []):
        if block.get('block') != 'type' or 'f2py_python_name' not in block:
            continue
        native, generated = block['name'].lower(), block['f2py_c_name'].lower()
        source = source.replace(
            f'PyModule_AddObjectRef(m, "{generated}",',
            f'PyModule_AddObjectRef(m, "{block["f2py_python_name"]}",')
        source = source.replace(
            f'PyObject_SetAttrString(fmod, "{generated}",',
            f'PyObject_SetAttrString(fmod, "{native}",')
    return source


def _qualify_fortran(source, module):
    """Qualify external bindings without changing native Fortran names."""
    names = {name.lower(): generated for name, generated
             in module.get('f2py_dt_namespace', {}).items()}
    if not names:
        return source
    ordered = sorted(names, key=lambda name: (-len(name), name))
    prefixes = ('f2py_create_', 'f2py_destroy_', 'f2py_get_', 'f2py_set_',
                'f2py_resize_', 'f2py_clear_', 'f2py_call_', 'f2py_op_',
                'f2py_str_free_', 'f2py_str_')

    def declaration(match):
        procedure = match['procedure']
        for prefix in prefixes:
            if not procedure.lower().startswith(prefix):
                continue
            rest = procedure[len(prefix):]
            for name in ordered:
                suffix = rest[len(name):]
                if (rest.lower().startswith(name)
                        and (not suffix or suffix.startswith('_'))):
                    binding = prefix + names[name] + suffix.lower()
                    head = match[0][:-len(match['binding'])].rstrip()
                    clause = 'bind(c, name="' + binding + '")'
                    if len(head) + 1 + len(clause) > 132:
                        return head + ' &\n    & ' + clause
                    return head + ' ' + clause
        return match[0]

    # Each generated wrapper declares its BIND(C) procedure on one line.
    pattern = (r'(?im)^[ \t]*(?:function|subroutine)[ \t]+'
               r'(?P<procedure>f2py_[A-Za-z_0-9]+)\b'
               r'[^\n]*?\b(?P<binding>bind[ \t]*\([ \t]*c[ \t]*\))')
    return re.sub(pattern, declaration, source)
