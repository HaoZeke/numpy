"""Carry USE-associated component types with their declaring contexts."""


def _public_type(module, name, block, *, imported=False):
    attrs = set(module.get('vars', {}).get(name, {}).get('attrspec', []))
    if 'public' in attrs:
        return True
    if 'private' in attrs:
        return False
    if not imported:
        attrs = set(block.get('attrspec', []))
        if 'public' in attrs:
            return True
        if 'private' in attrs:
            return False
    default = module.get('vars', {}).get('', {}).get('attrspec', [])
    return 'private' not in default


def _import_items(exports, mapping):
    """Yield each visible local name and its original type declaration."""
    renamed = {local.lower(): remote.lower()
               for local, remote in mapping.get('map', {}).items()}
    if mapping.get('only'):
        for local, remote in renamed.items():
            if remote in exports:
                yield local, exports[remote]
        return
    hidden = {remote for local, remote in renamed.items() if local != remote}
    for remote, record in exports.items():
        if remote not in hidden:
            yield remote, record
    for local, remote in renamed.items():
        if remote in exports:
            yield local, exports[remote]


def _prepare_imported_types(modules):
    """Return owner-first modules and annotate imported component contexts.

    Each imported type keeps its original member-resolution scope. The C
    generator receives its declaring type's generated identity, while the
    Fortran generator receives the source's local USE alias.
    """
    from ._dt_helpers import _can_wrap_bindc, _module_types
    from ._dt_namespace import _c_module

    indexed = {module['name'].lower(): module for module in modules}
    ordered, active, visited = [], set(), set()

    def visit(module):
        key = module['name'].lower()
        if key in visited:
            return
        if key in active:
            raise ValueError('cyclic Fortran module type dependencies')
        active.add(key)
        for dependency in module.get('use', {}):
            if dependency.lower() in indexed:
                visit(indexed[dependency.lower()])
        active.remove(key)
        visited.add(key)
        ordered.append(module)

    for module in modules:
        visit(module)
    exports = {}
    for module in ordered:
        local = {block['name'].lower() for block in module.get('body', [])
                 if block.get('block') == 'type'}
        records, ambiguous = {}, set()
        for dependency, mapping in module.get('use', {}).items():
            for alias, record in _import_items(
                    exports.get(dependency.lower(), {}), mapping):
                if alias in local or alias in ambiguous:
                    continue
                previous = records.get(alias)
                if previous is not None and previous['identity'] != record['identity']:
                    records.pop(alias)
                    ambiguous.add(alias)
                else:
                    records[alias] = record
        native, c_types, c_names = {}, {}, {}
        for alias, record in records.items():
            imported = dict(record['native'])
            imported['f2py_import_bindc'] = record['bindc']
            imported['f2py_import_context'] = record['native_context']
            imported['f2py_origin_module'] = record['module']
            imported['f2py_origin_name'] = record['name']
            native[alias] = imported
            imported_c = dict(record['c_type'])
            imported_c['f2py_import_bindc'] = record['bindc']
            imported_c['f2py_import_context'] = record['c_context']
            c_types[record['c_name']] = imported_c
            c_names[alias] = record['c_name']
        module['f2py_imported_types'] = native
        module['f2py_c_imported_types'] = c_types
        module['f2py_dt_import_names'] = c_names
        for block in module.get('body', []):
            if block.get('block') != 'type':
                continue
            hidden = block.get('f2py_hidden', {})
            for name, var in block.get('vars', {}).items():
                reason = hidden.get(name, '')
                if (var.get('typename', '').lower() in native
                        and reason.startswith('f2py does not wrap its type ')):
                    hidden.pop(name)
        native_context, blocks, _ = _module_types(module)
        c_module = _c_module(module)
        c_context, c_blocks, _ = _module_types(c_module)
        c_blocks = {block['name'].lower(): block for block in c_blocks}
        visible = {alias: record for alias, record in records.items()
                   if _public_type(module, alias, record['native'], imported=True)}
        for block in blocks:
            name = block['name'].lower()
            if not _public_type(module, name, block):
                continue
            c_name = block.get('f2py_c_name', block['name']).lower()
            visible[name] = {'identity': (module['name'].lower(), name),
                             'module': module['name'], 'name': block['name'],
                             'native': block, 'c_type': c_blocks[c_name],
                             'native_context': native_context,
                             'c_context': c_context, 'c_name': c_name,
                             'bindc': _can_wrap_bindc(block, native_context)}
        exports[module['name'].lower()] = visible
    return ordered


def _import_type_uses(module):
    """Import aliases needed by component accessors, including private aliases."""
    lines = []
    for alias, block in module.get('f2py_imported_types', {}).items():
        native = block['f2py_origin_name']
        item = native if alias.lower() == native.lower() else f'{alias} => {native}'
        lines.append(f'  use {block["f2py_origin_module"]}, only: {item}')
    return sorted(set(lines))


def _has_imported_type_exports(module):
    """Return whether a module exposes a USE-associated derived class."""
    return any(_public_type(module, alias, block, imported=True)
               for alias, block in module.get('f2py_imported_types', {}).items())


def _imported_python_exports(module):
    """Expose public USE aliases on their Fortran module object."""
    lines = []
    for alias, block in module.get('f2py_imported_types', {}).items():
        if not _public_type(module, alias, block, imported=True):
            continue
        c_name = module['f2py_dt_import_names'][alias]
        lines.extend([
            '\t{',
            f'\t\tPyObject *fmod = PyDict_GetItemString(d, "{module["name"]}");',
            (f'\t\tif (fmod != NULL && PyObject_SetAttrString(fmod, "{alias}", '
             f'(PyObject *)&Py{c_name}_Type) < 0) return -1;'),
            '\t}',
        ])
    return lines
