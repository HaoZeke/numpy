"""
Derived type support for f2py.

Generates Python class wrappers for Fortran derived types using PyCapsule.

For bind(c) types with scalar members: the C struct layout matches Fortran
directly; Python classes wrap these via PyCapsule pointing to heap-allocated
C structs.

For non-bind(c) types (opaque pointer path): auto-generated Fortran wrappers
with bind(c) provide constructor/destructor/getters/setters operating on
c_ptr. Python classes call these through PyCapsule.

Polymorphic arguments declared as ``class(T)`` are normalized to ``type(T)``
by crackfortran (see crackfortran.py line ~1636). Dynamic dispatch is lost;
the wrapped code treats ``class(T)`` identically to ``type(T)``.

Copyright 2024 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

import os

from .auxfuncs import (
    get_type_members,
    isroutine,
    outmess,
)

# Re-export everything from submodules for backward compatibility
from ._dt_helpers import (  # noqa: F401
    _C_TO_NPY_ENUM,
    _C_TO_PYFORMAT,
    _C_TO_PYOBJ,
    _FORTRAN_ARITH_OPS,
    _FORTRAN_CMP_OPS,
    _FORTRAN_TO_C,
    _FORTRAN_TO_ISOC,
    _PYOBJ_TO_C,
    _can_wrap_bindc,
    _can_wrap_opaque,
    _find_derived_types,
    _fortran_sym,
    _get_all_members,
    _get_array_dims,
    _get_c_return_type,
    _get_char_len,
    _get_extends_parent,
    _get_member_ctype,
    _get_member_isoc_type,
    _get_alloc_ndim,
    _is_allocatable_member,
    _is_array_member,
    _is_char_member,
    _is_complex_member,
    _is_type_array_member,
    _is_type_member,
    _topo_visit,
)

from ._dt_codegen import (  # noqa: F401
    _gen_bindc_struct,
    _gen_capsule_destructor,
    _gen_getset,
    _gen_init_code,
    _gen_opaque_capsule_destructor,
    _gen_opaque_extern_decls,
    _gen_opaque_getset,
    _gen_opaque_tp_init,
    _gen_opaque_tp_repr,
    _gen_operator_c_code,
    _gen_operator_fortran_wrappers,
    _gen_pytype_struct,
    _gen_tp_dealloc,
    _gen_tp_init,
    _gen_tp_new,
    _gen_tp_repr,
    _gen_typeobject,
    _scan_operator_interfaces,
)

from ._dt_routines import (  # noqa: F401
    _gen_routine_c_wrapper,
    _gen_routine_fortran_wrapper,
    _gen_routine_init_code,
    _gen_routine_method_table,
    _gen_type_methods,
    _get_wrappable_routines,
    _has_derived_type_args,
    _is_array_type_arg,
    _scan_type_bound_procedures,
    generate_fortran_wrappers,
    write_fortran_wrappers,
)


def buildhooks(pymod):
    """Build C code hooks for derived type support in a module.

    Scans the module for wrappable derived types and generates:
    - C struct typedefs
    - Python type objects with getters/setters
    - Module init registration code
    - Wrapper functions for routines with derived type arguments

    Returns a dict with 'f90modhooks' and 'initf90modhooks' keys
    compatible with the existing f90mod_rules pipeline.
    """
    ret = {
        'f90modhooks': [],
        'initf90modhooks': [],
    }

    # Find modules in the pymod
    from .f90mod_rules import findf90modules
    for m in findf90modules(pymod):
        modulename = m['name']
        type_blocks = _find_derived_types(m)

        # Collect module-level routines (for TBP and routine wrapping)
        all_routines = [b for b in m.get('body', [])
                        if isroutine(b)]

        # Get source file for TBP scanning
        # The 'from' field format varies:
        #   direct crackfortran: '/path/to/file.f90'
        #   full f2py pipeline: ':modulename:/path/to/file.f90'
        source_file = m.get('from', '')
        if ':' in source_file:
            parts = source_file.split(':')
            for part in reversed(parts):
                if part and os.path.isfile(part):
                    source_file = part
                    break
            else:
                source_file = ''

        # Multi-pass type resolution: resolve leaf types first, then
        # types with nested type members (dependency ordering)
        type_map = {}
        remaining = list(type_blocks)
        max_passes = len(type_blocks) + 1
        for _ in range(max_passes):
            if not remaining:
                break
            still_remaining = []
            for tb in remaining:
                typename = tb['name']
                if (_can_wrap_bindc(tb, type_map)
                        or _can_wrap_opaque(tb, type_map)):
                    type_map[typename.lower()] = tb
                else:
                    still_remaining.append(tb)
            if len(still_remaining) == len(remaining):
                break  # no progress, stop
            remaining = still_remaining

        # Generate type wrappers in dependency order (leaf types first)
        generated = set()
        gen_order = []
        for tb in type_blocks:
            _topo_visit(tb, type_blocks, type_map,
                        generated, gen_order)

        for tb in gen_order:
            typename = tb['name']

            # Scan for type-bound procedures
            bound_procs = _scan_type_bound_procedures(
                source_file, typename)
            if bound_procs:
                outmess(f'\t\tFound type-bound procedures for '
                        f'"{typename}": '
                        f'{", ".join(bound_procs.keys())}\n')

            if _can_wrap_bindc(tb, type_map):
                outmess(f'\t\tGenerating bind(c) type wrapper '
                        f'for "{typename}"...\n')
                _generate_bindc_hooks(
                    ret, typename, tb, modulename, m,
                    bound_procs=bound_procs,
                    routines=all_routines,
                    type_map=type_map)
            elif _can_wrap_opaque(tb, type_map):
                outmess(f'\t\tGenerating opaque pointer type wrapper '
                        f'for "{typename}"...\n')
                _generate_opaque_hooks(
                    ret, typename, tb, modulename, m,
                    bound_procs=bound_procs,
                    routines=all_routines,
                    type_map=type_map)
            else:
                outmess(f'\t\tSkipping derived type "{typename}" '
                        f'(not wrappable yet)...\n')

        # Process routines with derived type arguments
        if type_map:
            wrappable_routines = _get_wrappable_routines(m, type_map)
            method_entries = []
            for routine in wrappable_routines:
                rname = routine['name']
                outmess(f'\t\tGenerating derived type routine wrapper '
                        f'for "{rname}"...\n')
                c_code, method_def = _gen_routine_c_wrapper(
                    modulename, routine, type_map)
                if c_code is not None:
                    ret['f90modhooks'].append(c_code)
                    method_entries.append(method_def)
            if method_entries:
                ret['f90modhooks'].append(
                    _gen_routine_method_table(modulename, method_entries))
                ret['initf90modhooks'].extend(
                    _gen_routine_init_code(modulename, method_entries))

    # Add <complex.h> to needs if any type has complex members
    if ret['f90modhooks']:
        for m in findf90modules(pymod):
            for tb in _find_derived_types(m):
                for mvar in get_type_members(tb).values():
                    ctype = _get_member_ctype(mvar)
                    if ctype in ('float _Complex', 'double _Complex'):
                        ret.setdefault('need', []).append('complex.h')
                        break
                if 'need' in ret:
                    break
            if 'need' in ret:
                break

    return ret


def _generate_bindc_hooks(ret, typename, typeblock, modulename,
                          module_block, bound_procs=None, routines=None,
                          type_map=None):
    """Generate hooks for a bind(c) derived type."""
    members = get_type_members(typeblock)

    code_parts = []
    code_parts.append(_gen_bindc_struct(typename, members))
    code_parts.append(_gen_pytype_struct(typename))
    code_parts.append(_gen_capsule_destructor(typename))
    code_parts.append(_gen_tp_new(typename))
    code_parts.append(_gen_tp_init(typename, members))
    code_parts.append(_gen_tp_dealloc(typename))
    code_parts.append(_gen_getset(typename, members))
    code_parts.append(_gen_tp_repr(typename, members))

    # Type-bound procedures
    has_methods = False
    if bound_procs and routines and type_map:
        method_funcs, method_entries = _gen_type_methods(
            typename, bound_procs, routines, type_map)
        if method_funcs:
            code_parts.extend(method_funcs)
            methods_table = (
                f'static PyMethodDef Py{typename}_methods[] = {{\n'
                + '\n'.join(method_entries) + '\n'
                + '    {NULL}  /* sentinel */\n'
                + '};\n')
            code_parts.append(methods_table)
            has_methods = True

    # Operator overloading
    has_number = False
    has_richcompare = False
    ops = _scan_operator_interfaces(module_block, typename, type_map)
    if ops:
        op_code, has_number, has_richcompare = _gen_operator_c_code(
            typename, ops, type_map)
        code_parts.extend(op_code)

    code_parts.append(_gen_typeobject(
        typename, has_methods=has_methods,
        has_number=has_number, has_richcompare=has_richcompare))

    ret['f90modhooks'].append('\n'.join(code_parts))
    ret['initf90modhooks'].extend(_gen_init_code(typename, modulename))


def _generate_opaque_hooks(ret, typename, typeblock, modulename,
                           module_block, bound_procs=None, routines=None,
                           type_map=None):
    """Generate hooks for a non-bind(c) derived type via opaque pointers.

    Uses the 3-layer approach: Python -> C wrapper -> Fortran accessor.
    The C code calls auto-generated bind(c) Fortran functions that use
    c_f_pointer to access the actual Fortran type. No C struct needed;
    the PyCapsule wraps a raw void* (Fortran c_ptr).

    For types with extends(parent), only child-specific members are
    added to getset (parent members inherited via tp_base). The Fortran
    wrappers and tp_init cover all members (parent + child).
    """
    own_members = get_type_members(typeblock)
    parent_name = _get_extends_parent(typeblock)

    # For inheritance: all_members = parent + own for extern/init/repr/getset
    # Each getset entry uses the child's capsule name and child's Fortran
    # accessors, so parent getset cannot be inherited via tp_base.
    if parent_name and type_map:
        parent_members, own = _get_all_members(typeblock, type_map)
        from collections import OrderedDict
        all_members = OrderedDict()
        all_members.update(parent_members)
        all_members.update(own)
    else:
        all_members = own_members
        parent_name = None

    code_parts = []
    code_parts.append(_gen_opaque_extern_decls(typename, all_members))
    code_parts.append(_gen_pytype_struct(typename))
    code_parts.append(_gen_opaque_capsule_destructor(typename))
    code_parts.append(_gen_tp_new(typename))
    code_parts.append(_gen_opaque_tp_init(typename, all_members))
    code_parts.append(_gen_tp_dealloc(typename))
    # All members in getset (capsule names differ per type, so parent
    # getset cannot be reused directly)
    code_parts.append(_gen_opaque_getset(typename, all_members))
    code_parts.append(_gen_opaque_tp_repr(typename, all_members))

    # Type-bound procedures
    has_methods = False
    if bound_procs and routines and type_map:
        method_funcs, method_entries = _gen_type_methods(
            typename, bound_procs, routines, type_map)
        if method_funcs:
            code_parts.extend(method_funcs)
            methods_table = (
                f'static PyMethodDef Py{typename}_methods[] = {{\n'
                + '\n'.join(method_entries) + '\n'
                + '    {NULL}  /* sentinel */\n'
                + '};\n')
            code_parts.append(methods_table)
            has_methods = True

    # Operator overloading
    has_number = False
    has_richcompare = False
    ops = _scan_operator_interfaces(module_block, typename, type_map)
    if ops:
        op_code, has_number, has_richcompare = _gen_operator_c_code(
            typename, ops, type_map)
        code_parts.extend(op_code)

    # Capitalize parent name to match PyTypeObject naming convention
    parent_typename = None
    if parent_name and type_map:
        parent_tb = type_map.get(parent_name)
        if parent_tb:
            parent_typename = parent_tb['name']

    code_parts.append(_gen_typeobject(
        typename, has_methods=has_methods,
        parent_typename=parent_typename,
        has_number=has_number, has_richcompare=has_richcompare))

    ret['f90modhooks'].append('\n'.join(code_parts))
    ret['initf90modhooks'].extend(_gen_init_code(typename, modulename))
