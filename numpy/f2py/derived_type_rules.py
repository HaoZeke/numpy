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
    _can_wrap_abstract,
    _can_wrap_bindc,
    _can_wrap_opaque,
    _eval_kind_expr,
    _find_derived_types,
    _fortran_sym,
    _get_all_members,
    _get_fortran_type_spec,
    _get_array_dims,
    _get_c_return_type,
    _get_char_len,
    _get_extends_parent,
    _get_member_ctype,
    _get_member_isoc_type,
    _get_alloc_ndim,
    _get_pointer_ndim,
    _has_len_params,
    _has_unresolved_kind_params,
    _is_abstract_type,
    _is_allocatable_member,
    _is_array_member,
    _is_char_member,
    _is_coarray_member,
    _coarray_as_local,
    _is_complex_member,
    _is_deferred_char_member,
    _is_pointer_member,
    _is_type_array_member,
    _is_type_member,
    _is_type_parameter,
    _resolve_kind_params,
    _resolve_parameterized_type,
    _resolve_type_for_kind,
    _enumerate_specializations,
    _get_kind_param_info,
    _get_len_param_info,
    _is_len_sized_array,
    _topo_visit,
)

from ._dt_codegen import (  # noqa: F401
    _gen_array_from_any_helper,
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
    _scan_final_subroutines,
    _scan_proc_pointer_components,
    _gen_proc_pointer_fortran_wrappers,
    _gen_proc_pointer_c_methods,
    _find_interface_block,
    _scan_type_bound_procedures,
    _gen_coarray_remote_c_method,
    _gen_defined_io_fortran_wrapper,
    _gen_defined_io_read_fortran_wrapper,
    _gen_defined_io_tp_str,
    _gen_defined_io_from_string,
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
                        or _can_wrap_opaque(tb, type_map)
                        or _can_wrap_abstract(tb, type_map)):
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

        # Assign type tags for polymorphic dispatch (F2018 7.3.2.3).
        # Each wrappable type gets a unique integer tag. Tags are
        # ordered so parent types have lower tags than children.
        type_tags = {}
        for i, tb in enumerate(gen_order, start=1):
            type_tags[tb['name'].lower()] = i

        # Collect all final subroutine names across types.
        # These must be excluded from routine wrapping since they
        # are called automatically during deallocation (F2018 7.5.6.3)
        # and should not be exposed as Python-callable wrappers.
        final_subroutine_names = set()
        for tb in gen_order:
            finals = _scan_final_subroutines(
                source_file, tb['name'])
            final_subroutine_names.update(finals)

        # Track bound_procs and proc_ptrs per type for inheritance merging
        all_bound_procs = {}
        all_proc_ptrs = {}

        for tb in gen_order:
            typename = tb['name']

            # Scan for type-bound procedures
            bound_procs = _scan_type_bound_procedures(
                source_file, typename)

            # Scan for procedure pointer components (F2018 7.5.4.4)
            proc_ptrs = _scan_proc_pointer_components(
                source_file, typename)

            # Merge inherited TBPs from parent (F2018 7.5.7)
            parent_name = _get_extends_parent(tb)
            if parent_name:
                parent_bp = all_bound_procs.get(parent_name, {})
                merged = dict(parent_bp)
                merged.update(bound_procs)  # child overrides parent
                bound_procs = merged
                # Inherit proc ptrs from parent
                parent_pp = all_proc_ptrs.get(parent_name, [])
                if parent_pp:
                    parent_names = {p['name'] for p in parent_pp}
                    own_names = {p['name'] for p in proc_ptrs}
                    inherited = [p for p in parent_pp
                                 if p['name'] not in own_names]
                    proc_ptrs = inherited + proc_ptrs

            all_bound_procs[typename.lower()] = bound_procs
            all_proc_ptrs[typename.lower()] = proc_ptrs

            if bound_procs:
                outmess(f'\t\tFound type-bound procedures for '
                        f'"{typename}": '
                        f'{", ".join(bound_procs.keys())}\n')

            if proc_ptrs:
                outmess(f'\t\tFound procedure pointer components for '
                        f'"{typename}": '
                        f'{", ".join(p["name"] for p in proc_ptrs)}\n')

            if _can_wrap_abstract(tb, type_map):
                outmess(f'\t\tGenerating abstract type skeleton '
                        f'for "{typename}"...\n')
                _generate_abstract_hooks(
                    ret, typename, tb, modulename)
            elif _can_wrap_bindc(tb, type_map):
                outmess(f'\t\tGenerating bind(c) type wrapper '
                        f'for "{typename}"...\n')
                _generate_bindc_hooks(
                    ret, typename, tb, modulename, m,
                    bound_procs=bound_procs,
                    routines=all_routines,
                    type_map=type_map,
                    type_tag=type_tags.get(typename.lower(), 0))
            elif _can_wrap_opaque(tb, type_map):
                outmess(f'\t\tGenerating opaque pointer type wrapper '
                        f'for "{typename}"...\n')
                _generate_opaque_hooks(
                    ret, typename, tb, modulename, m,
                    bound_procs=bound_procs,
                    routines=all_routines,
                    type_map=type_map,
                    proc_ptrs=proc_ptrs,
                    type_tag=type_tags.get(typename.lower(), 0))
            else:
                outmess(f'\t\tSkipping derived type "{typename}" '
                        f'(not wrappable yet)...\n')

        # Process routines with derived type arguments
        if type_map:
            wrappable_routines = _get_wrappable_routines(m, type_map)
            # Exclude final subroutines (F2018 7.5.6.1) from
            # routine wrapping -- they run via deallocate, not
            # as user-callable Python methods
            wrappable_routines = [
                routine for routine in wrappable_routines
                if routine['name'].lower() not in final_subroutine_names
            ]
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

    # Prepend the Array API / DLPack helper function (used by all
    # array setters in both bind(c) and opaque code paths).
    if ret['f90modhooks']:
        ret['f90modhooks'][0] = (
            _gen_array_from_any_helper() + ret['f90modhooks'][0])

    # Complex members use npy_cfloat/npy_cdouble (MSVC-compatible) with
    # npy_creal/npy_cpack helpers from npy_math.h. The type definitions
    # come from npy_common.h (included via arrayobject.h), but the
    # accessor functions need npy_math.h explicitly.
    if ret['f90modhooks']:
        for m in findf90modules(pymod):
            for tb in _find_derived_types(m):
                for mvar in get_type_members(tb).values():
                    ctype = _get_member_ctype(mvar)
                    if ctype in ('npy_cfloat', 'npy_cdouble'):
                        ret.setdefault('need', []).append('npy_math.h')
                        break
                if 'need' in ret and 'npy_math.h' in ret.get('need', []):
                    break
            if 'need' in ret and 'npy_math.h' in ret.get('need', []):
                break

    return ret


def _generate_bindc_hooks(ret, typename, typeblock, modulename,
                          module_block, bound_procs=None, routines=None,
                          type_map=None, type_tag=0):
    """Generate hooks for a bind(c) derived type."""
    members = {
        name: var for name, var in get_type_members(typeblock).items()
        if not _is_type_parameter(var)
    }

    code_parts = []
    code_parts.append(_gen_bindc_struct(typename, members))
    code_parts.append(_gen_pytype_struct(typename))
    code_parts.append(_gen_capsule_destructor(typename))
    code_parts.append(_gen_tp_new(typename, type_tag=type_tag))
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


def _generate_abstract_hooks(ret, typename, typeblock, modulename):
    """Generate a skeleton PyTypeObject for an abstract Fortran type.

    Abstract types (F2003 4.5.7) cannot be instantiated, but their
    concrete extensions need the abstract parent's PyTypeObject so
    that tp_base gives isinstance() support. The generated type has:
    - A PyObject struct (no capsule, no data)
    - tp_new that works (required by Python type machinery)
    - tp_init that raises TypeError
    - Py_TPFLAGS_BASETYPE so concrete children can inherit
    - No getters/setters (concrete children have their own)
    - No Fortran wrappers (nothing to construct/destroy)
    """
    code_parts = []

    # Minimal PyObject struct (no capsule member)
    code_parts.append(f"""\
/* Abstract type {typename} -- skeleton for isinstance() support */
typedef struct {{
    PyObject_HEAD
}} Py{typename}Object;
""")

    # tp_new -- standard allocator
    code_parts.append(f"""\
static PyObject *
Py{typename}_tp_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{{
    Py{typename}Object *self;
    self = (Py{typename}Object *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}}
""")

    # tp_init -- raises TypeError (abstract types cannot be instantiated)
    code_parts.append(f"""\
static int
Py{typename}_tp_init(PyObject *self, PyObject *args, PyObject *kwds)
{{
    PyErr_SetString(PyExc_TypeError,
        "Cannot instantiate abstract Fortran type '{typename}'");
    return -1;
}}
""")

    # tp_dealloc -- minimal
    code_parts.append(f"""\
static void
Py{typename}_tp_dealloc(PyObject *self)
{{
    Py_TYPE(self)->tp_free(self);
}}
""")

    # tp_repr
    code_parts.append(f"""\
static PyObject *
Py{typename}_tp_repr(PyObject *self)
{{
    return PyUnicode_FromString("<abstract type {typename}>");
}}
""")

    # Empty getset (sentinel only)
    code_parts.append(f"""\
static PyGetSetDef Py{typename}_getset[] = {{
    {{NULL}}  /* sentinel */
}};
""")

    # PyTypeObject
    code_parts.append(f"""\
static PyTypeObject Py{typename}_Type = {{
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "f2py.{typename}",
    .tp_doc = "Abstract Fortran type {typename} (cannot be instantiated)",
    .tp_basicsize = sizeof(Py{typename}Object),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Py{typename}_tp_new,
    .tp_init = Py{typename}_tp_init,
    .tp_dealloc = Py{typename}_tp_dealloc,
    .tp_repr = Py{typename}_tp_repr,
    .tp_getset = Py{typename}_getset,
}};
""")

    ret['f90modhooks'].append('\n'.join(code_parts))
    ret['initf90modhooks'].extend(_gen_init_code(typename, modulename))


def _generate_opaque_hooks(ret, typename, typeblock, modulename,
                           module_block, bound_procs=None, routines=None,
                           type_map=None, proc_ptrs=None, type_tag=0):
    """Generate hooks for a non-bind(c) derived type via opaque pointers.

    Uses the 3-layer approach: Python -> C wrapper -> Fortran accessor.
    The C code calls auto-generated bind(c) Fortran functions that use
    c_f_pointer to access the actual Fortran type. No C struct needed;
    the PyCapsule wraps a raw void* (Fortran c_ptr).

    For types with extends(parent), only child-specific members are
    added to getset (parent members inherited via tp_base). The Fortran
    wrappers and tp_init cover all members (parent + child).

    For parameterized types with KIND parameters, generates multi-kind
    dispatch: separate Fortran functions per kind value, Python type
    dispatches based on kind_value stored in the object.
    """
    own_members = {
        name: var for name, var in get_type_members(typeblock).items()
        if not _is_type_parameter(var)
    }
    parent_name = _get_extends_parent(typeblock)

    # For inheritance: all_members = parent + own for extern/init/repr/getset
    if parent_name and type_map:
        parent_members, own = _get_all_members(typeblock, type_map)
        from collections import OrderedDict
        all_members = OrderedDict()
        all_members.update(parent_members)
        all_members.update(own)
    else:
        all_members = own_members
        parent_name = None

    # Build specialization list for parameterized types
    specs = _enumerate_specializations(typeblock)
    kind_info = _get_kind_param_info(typeblock)
    len_info = _get_len_param_info(typeblock)

    # Build per-specialization resolved member dicts
    specializations = []
    if specs:
        for kind_dict, suffix in specs:
            resolved_tb = _resolve_type_for_kind(typeblock, kind_dict)
            if parent_name and type_map:
                pm, om = _get_all_members(resolved_tb, type_map)
                resolved_members = OrderedDict()
                resolved_members.update(pm)
                resolved_members.update(om)
            else:
                resolved_members = {
                    n: v for n, v
                    in get_type_members(resolved_tb).items()
                    if not _is_type_parameter(v)
                }
            specializations.append((suffix, kind_dict, resolved_members))

    code_parts = []
    if specializations:
        # Multi-kind: generate dispatch-aware code
        code_parts.append(
            _gen_opaque_extern_decls(typename, all_members,
                                     specializations=specializations))
        code_parts.append(_gen_pytype_struct(
            typename, has_kind=True))
        code_parts.append(
            _gen_opaque_capsule_destructor(
                typename, specializations=specializations))
        code_parts.append(_gen_tp_new(typename, type_tag=type_tag))
        code_parts.append(
            _gen_opaque_tp_init(typename, all_members,
                                specializations=specializations,
                                kind_info=kind_info,
                                len_info=len_info))
        code_parts.append(_gen_tp_dealloc(typename))
        code_parts.append(
            _gen_opaque_getset(typename, all_members,
                               specializations=specializations,
                               kind_info=kind_info))
        code_parts.append(
            _gen_opaque_tp_repr(typename, all_members,
                                specializations=specializations,
                                kind_info=kind_info))
    else:
        code_parts.append(_gen_opaque_extern_decls(typename, all_members,
                                                    len_info=len_info))
        code_parts.append(_gen_pytype_struct(typename,
                                              len_info=len_info))
        code_parts.append(_gen_opaque_capsule_destructor(
            typename, len_info=len_info))
        code_parts.append(_gen_tp_new(typename, type_tag=type_tag))
        code_parts.append(_gen_opaque_tp_init(typename, all_members,
                                              len_info=len_info))
        code_parts.append(_gen_tp_dealloc(typename))
        code_parts.append(_gen_opaque_getset(typename, all_members,
                                              len_info=len_info))
        code_parts.append(_gen_opaque_tp_repr(typename, all_members,
                                               len_info=len_info))

    # Type-bound procedures and procedure pointer components
    has_methods = False
    all_method_funcs = []
    all_method_entries = []
    if bound_procs and routines and type_map:
        method_funcs, method_entries = _gen_type_methods(
            typename, bound_procs, routines, type_map)
        all_method_funcs.extend(method_funcs)
        all_method_entries.extend(method_entries)
    # Procedure pointer components (F2018 7.5.4.4)
    if proc_ptrs and module_block:
        pp_funcs, pp_entries = _gen_proc_pointer_c_methods(
            typename, proc_ptrs, module_block.get('body', []))
        all_method_funcs.extend(pp_funcs)
        all_method_entries.extend(pp_entries)
    # Coarray remote access methods (F2018 7.5.4.3)
    for mname, mvar in all_members.items():
        if _is_coarray_member(mvar):
            local_var = _coarray_as_local(mvar)
            result = _gen_coarray_remote_c_method(
                typename, mname, local_var)
            if result:
                c_func, method_entry = result
                all_method_funcs.append(c_func)
                all_method_entries.append(method_entry)
    if all_method_funcs:
        code_parts.extend(all_method_funcs)
        methods_table = (
            f'static PyMethodDef Py{typename}_methods[] = {{\n'
            + '\n'.join(all_method_entries) + '\n'
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

    # Defined I/O: __str__ via write(formatted) (F2018 12.6.4.8)
    has_tp_str = False
    if bound_procs:
        if '__write_formatted__' in bound_procs:
            code_parts.append(_gen_defined_io_tp_str(typename))
            has_tp_str = True
        if '__read_formatted__' in bound_procs:
            code_parts.append(_gen_defined_io_from_string(typename))
            # Add from_string as a class method
            if all_method_entries is not None:
                all_method_entries.append(
                    f'    {{"from_string", (PyCFunction)'
                    f'Py{typename}_from_string, '
                    f'METH_VARARGS | METH_CLASS, '
                    f'"Create {typename} from formatted string"}}')

    code_parts.append(_gen_typeobject(
        typename, has_methods=has_methods,
        parent_typename=parent_typename,
        has_number=has_number, has_richcompare=has_richcompare,
        has_tp_str=has_tp_str))

    ret['f90modhooks'].append('\n'.join(code_parts))
    ret['initf90modhooks'].extend(_gen_init_code(typename, modulename))
