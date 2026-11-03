"""
Routine wrapping and Fortran wrapper generation for derived types.

Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

import os

from ._dt_codegen import _gen_operator_fortran_wrappers, _scan_operator_interfaces
from ._dt_helpers import (
    _C_TO_PYFORMAT,
    _C_TO_PYOBJ,
    _can_wrap_opaque,
    _coarray_as_local,
    _enumerate_specializations,
    _eval_kind_expr,
    _fortran_sym,
    _get_all_members,
    _get_alloc_ndim,
    _get_array_dims,
    _get_char_len,
    _get_extends_parent,
    _get_fortran_type_spec,
    _get_len_param_info,
    _get_member_ctype,
    _get_member_isoc_type,
    _get_pointer_ndim,
    _is_allocatable_member,
    _is_array_member,
    _is_char_member,
    _is_coarray_member,
    _is_deferred_char_member,
    _is_len_sized_array,
    _is_pointer_member,
    _is_polymorphic_arg,
    _is_private_member,
    _is_type_array_member,
    _is_type_member,
    _is_type_parameter,
    _member_actual,
    _member_assign,
    _member_value,
    _resolve_type_for_kind,
    _routine_args,
    _wrap_types,
)
from .auxfuncs import (
    F2PYError,
    errmess,
    get_type_members,
    isallocatable,
    isarray,
    isderivedtype,
    isfunction,
    isintent_hide,
    isintent_in,
    isintent_inout,
    isintent_nothide,
    isintent_out,
    isoptional,
    isrequired,
    isunlimitedpolymorphic,
    outmess,
)
from .cfuncs import dt_any_intrinsics
from .crackfortran import markoutercomma


def _source_file(module):
    """Return the source file of a module, for scanning what crackfortran
    does not record: type-bound procedures, finalizers and procedure
    pointer components.

    The 'from' field is the file for crackfortran, and is prefixed with
    ':modulename:' in the full f2py pipeline.
    """
    source_file = module.get('from', '')
    if ':' in source_file:
        for part in reversed(source_file.split(':')):
            if part and os.path.isfile(part):
                return part
        return ''
    return source_file


def _initial_state(ref, components, type_map, skip=(), depth=0):
    """Return statements that define every component of ``ref`` right
    after ALLOCATE, and the number of DO variables they use.

    ALLOCATE applies default initialization and leaves allocatable
    components unallocated (F2018 7.5.4.6); every other component is
    undefined.  Pointers are nullified, since ASSOCIATED must not be given
    a pointer with undefined association status (F2018 16.9.16); other
    components are zeroed or blanked.  Elements of arrays of derived type
    are visited with DO variables ``f2py_i1``, ``f2py_i2``, ...
    """
    lines = []
    deepest = depth
    for name, var in components.items():
        attrs = var.get('attrspec', [])
        if (name in skip or '=' in var or 'allocatable' in attrs
                or _is_type_parameter(var) or _is_private_member(var)):
            continue
        part = f'{ref}%{name}'
        typespec = var.get('typespec', '')
        if 'pointer' in attrs:
            lines.append(f'nullify({part})')
        elif typespec == 'type':
            inner = type_map.get(var.get('typename', '').lower())
            if inner is None:
                continue
            rank = len(var.get('dimension', []))
            indices = [f'f2py_i{depth + r}' for r in range(1, rank + 1)]
            element = f'{part}({", ".join(indices)})' if rank else part
            body, used = _initial_state(
                element, _all_components(inner, type_map), type_map,
                depth=depth + rank)
            if not body:
                continue
            deepest = max(deepest, used)
            # Column-major order: the first subscript varies fastest
            heads = [f'do {index} = lbound({part}, {r}), ubound({part}, {r})'
                     for r, index in reversed(list(enumerate(indices, 1)))]
            lines.extend('  ' * level + head
                         for level, head in enumerate(heads))
            lines.extend('  ' * rank + line for line in body)
            lines.extend('  ' * level + 'end do'
                         for level in reversed(range(rank)))
        elif typespec == 'character':
            lines.append(f"{part} = ' '")
        elif typespec == 'logical':
            lines.append(f'{part} = .false.')
        elif typespec in _NUMERIC_TYPESPECS:
            lines.append(f'{part} = 0')
    return lines, deepest


_NUMERIC_TYPESPECS = ('integer', 'real', 'double precision', 'complex',
                      'double complex')


def _all_components(tb, type_map):
    """Return all components of a type, inherited ones first, including
    those that f2py does not expose."""
    parent = (_get_extends_parent(tb) or '').lower()
    inherited = (_all_components(type_map[parent], type_map)
                 if parent in type_map else {})
    return {**inherited, **{name: var for name, var in tb['vars'].items()
                            if not _is_type_parameter(var)}}


def generate_fortran_wrappers(modulename, type_blocks, routines=None,
                              module_block=None):
    """Generate Fortran wrapper source code for opaque pointer types,
    routines with derived type arguments, and operator implementations.

    Returns the complete Fortran source as a string, or None if no
    wrappers are needed.
    """
    type_map = _wrap_types(type_blocks)
    opaque_types = [tb for tb in type_blocks
                    if tb['name'].lower() in type_map
                    and _can_wrap_opaque(tb, type_map)]

    # Find operator wrappers needed
    operator_wrappers = []
    if module_block and type_map:
        for tb in type_blocks:
            typename = tb['name']
            if typename.lower() not in type_map:
                continue
            ops = _scan_operator_interfaces(
                module_block, typename, type_map)
            if ops:
                op_lines = _gen_operator_fortran_wrappers(
                    typename, ops, type_map)
                if op_lines:
                    operator_wrappers.extend(op_lines)

    if not opaque_types and not operator_wrappers:
        return None

    lines = []
    lines.append('! Auto-generated by f2py: derived type opaque pointer wrappers')
    lines.append(f'module f2py_{modulename}_derived_wrappers')
    lines.append(f'  use {modulename}')
    lines.append('  use iso_c_binding')
    lines.append('  implicit none')
    lines.append('')

    lines.append('contains')
    lines.append('')
    source_file = _source_file(module_block) if module_block else ''

    # Expand opaque types into specialization list:
    # [(func_name_base, type_spec, members, tb, len_info), ...]
    opaque_specs = []
    for tb in opaque_types:
        base_name = tb['name']
        specs = _enumerate_specializations(tb)
        len_info = _get_len_param_info(tb)
        if specs:
            for kind_dict, suffix in specs:
                resolved_tb = _resolve_type_for_kind(tb, kind_dict)
                ts = _get_fortran_type_spec(
                    resolved_tb, kind_override=kind_dict)
                parent_name = _get_extends_parent(resolved_tb)
                if parent_name and parent_name in type_map:
                    from collections import OrderedDict
                    pm, om = _get_all_members(resolved_tb, type_map)
                    members = OrderedDict()
                    members.update(pm)
                    members.update(om)
                else:
                    members = {
                        n: v for n, v
                        in get_type_members(resolved_tb).items()
                        if not _is_type_parameter(v)
                    }
                opaque_specs.append(
                    (base_name + suffix, ts, members, tb, len_info))
        else:
            # Non-parameterized or LEN-only type
            type_spec = _get_fortran_type_spec(tb)
            parent_name = _get_extends_parent(tb)
            if parent_name and parent_name in type_map:
                from collections import OrderedDict
                pm, om = _get_all_members(tb, type_map)
                members = OrderedDict()
                members.update(pm)
                members.update(om)
            else:
                members = {
                    n: v for n, v
                    in get_type_members(tb).items()
                    if not _is_type_parameter(v)
                }
            opaque_specs.append(
                (base_name, type_spec, members, tb, len_info))

    for typename, type_spec, members, orig_tb, len_info in opaque_specs:
        # Collect LEN param names for array dim checking
        len_param_names = {li['name'] for li in len_info}

        # Constructor: allocate + populate + return c_ptr
        # (scalar members only; arrays are zero-initialized by allocate)
        args = []
        decls = []
        assigns = []
        # LEN params come first as constructor args
        for li in len_info:
            args.append(li['name'])
            decls.append(
                f'    integer(c_int), intent(in), value :: {li["name"]}')
        for mname, mvar in members.items():
            if _is_coarray_member(mvar):
                mvar = _coarray_as_local(mvar)  # F2018 7.5.4.3: local data
            if _is_private_member(mvar):
                continue  # F2018 7.5.4.8: no accessor for private
            if (_is_array_member(mvar) or _is_type_member(mvar)
                    or _is_type_array_member(mvar) or _is_char_member(mvar)
                    or _is_allocatable_member(mvar)
                    or _is_pointer_member(mvar)
                    or _is_deferred_char_member(mvar)):
                continue  # arrays, nested types, chars, allocs, ptrs skip ctor
            if (len_param_names
                    and _is_len_sized_array(mvar, len_param_names)):
                continue  # LEN-sized arrays skip ctor
            isoc_type = _get_member_isoc_type(mvar)
            if isoc_type is None:
                continue
            args.append(mname)
            decls.append(f'    {isoc_type}, intent(in), value :: {mname}')
            assigns.append(
                '    ' + _member_assign(mvar, f'obj%{mname}', mname))

        args_str = ', '.join(args)
        decls_str = '\n'.join(decls)
        assigns_str = '\n'.join(assigns)

        # For LEN types, allocate with LEN params
        if len_info:
            base_tname = type_spec.split('(')[0] if '(' in type_spec else type_spec
            len_args = ', '.join(
                f'{li["name"]}={li["name"]}' for li in len_info)
            alloc_stmt = f'    allocate({base_tname}({len_args}) :: obj)'
        else:
            alloc_stmt = '    allocate(obj)'

        init_lines, loop_depth = _initial_state(
            'obj', _all_components(orig_tb, type_map), type_map,
            skip=set(args))
        # nor do procedure pointer components (F2018 7.5.4.6)
        init_lines += [
            f"nullify(obj%{pp['name']})"
            for pp in _type_proc_pointers(source_file, orig_tb, type_map)
            if not pp['init_null']]
        lines.append(
            f'  function f2py_create_{typename}({args_str}) '
            f'result(cptr) bind(c)')
        lines.append('    type(c_ptr) :: cptr')
        lines.append(decls_str)
        lines.append(f'    type({type_spec}), pointer :: obj')
        if loop_depth:
            lines.append('    integer :: ' + ', '.join(
                f'f2py_i{d}' for d in range(1, loop_depth + 1)))
        lines.append(alloc_stmt)
        lines.extend('    ' + line for line in init_lines)
        # Apply declared default values (F2018 7.5.4.1 R739
        # component-initialization). These run before user-provided
        # arguments so that defaults are used when args are omitted.
        for mname, mvar in members.items():
            default_val = mvar.get('=')
            if default_val is not None:
                # Only apply defaults for scalar members that are
                # also constructor arguments (skip arrays, types, etc.)
                if (_is_array_member(mvar) or _is_type_member(mvar)
                        or _is_type_array_member(mvar)
                        or _is_char_member(mvar)
                        or _is_allocatable_member(mvar)
                        or _is_pointer_member(mvar)
                        or _is_deferred_char_member(mvar)
                        or _is_coarray_member(mvar)
                        or _is_private_member(mvar)):
                    continue  # F2018 7.5.4.8: private ones are not ours
                lines.append(f'    obj%{mname} = {default_val}')
        lines.append(assigns_str)
        lines.append('    cptr = c_loc(obj)')
        lines.append(f'  end function f2py_create_{typename}')
        lines.append('')

        # Build LEN param argument list for wrappers that need to
        # reconstruct the type via c_f_pointer.  For LEN-parameterized
        # types, every Fortran wrapper (destructor, getters, setters)
        # must receive the LEN values so the type descriptor has the
        # correct layout (F2018 7.5.3.2).
        len_arg_names = [li['name'] for li in len_info]
        len_arg_str = ', '.join(len_arg_names)
        len_arg_str_prefix = (', ' + len_arg_str) if len_arg_str else ''
        len_decl_lines = [
            f'    integer(c_int), intent(in), value :: {ln}'
            for ln in len_arg_names
        ]
        len_decl_str = '\n'.join(len_decl_lines)

        # Destructor
        destr_args = f'cptr{len_arg_str_prefix}'
        lines.append(
            f'  subroutine f2py_destroy_{typename}({destr_args}) bind(c)')
        lines.append('    type(c_ptr), value :: cptr')
        if len_decl_str:
            lines.append(len_decl_str)
        lines.append(f'    type({type_spec}), pointer :: obj')
        lines.append('    call c_f_pointer(cptr, obj)')
        lines.append('    deallocate(obj)')
        lines.append(f'  end subroutine f2py_destroy_{typename}')
        lines.append('')

        # Getters and setters for each member.
        # For LEN-parameterized types, every wrapper that does
        # c_f_pointer(cptr, obj) needs the LEN values as extra
        # arguments (F2018 7.5.3.2: LEN params determine layout).
        # `len_arg_str_prefix` is ', n' or '' (set above).
        for mname, mvar in members.items():
            if _is_coarray_member(mvar):
                mvar = _coarray_as_local(mvar)  # F2018 7.5.4.3: local data
            if _is_private_member(mvar):
                continue  # F2018 7.5.4.8: no accessor for private
            if _is_type_array_member(mvar):
                inner = mvar.get('typename', '').lower()
                # Array of types: indexed getter (1-based)
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}, idx) '
                    f'result(inner_cptr) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    integer(c_int), value :: idx')
                lines.append('    type(c_ptr) :: inner_cptr')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append(
                    f'    type({inner}), pointer :: inner_obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append('    allocate(inner_obj)')
                lines.append(
                    f'    inner_obj = obj%{mname}(idx)')
                lines.append(
                    '    inner_cptr = c_loc(inner_obj)')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}')
                lines.append('')

                # Array of types: indexed setter (1-based)
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}, idx, inner_cptr) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    integer(c_int), value :: idx')
                lines.append(
                    '    type(c_ptr), value :: inner_cptr')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append(
                    f'    type({inner}), pointer :: inner_obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    '    call c_f_pointer(inner_cptr, inner_obj)')
                lines.append(
                    f'    obj%{mname}(idx) = inner_obj')
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
                continue

            if _is_type_member(mvar):
                inner = mvar.get('typename', '').lower()
                # Nested type getter: allocate copy, return c_ptr
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}) '
                    f'result(inner_cptr) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append('    type(c_ptr) :: inner_cptr')
                lines.append(f'    type({type_spec}), pointer :: obj')
                lines.append(f'    type({inner}), pointer :: inner_obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append('    allocate(inner_obj)')
                lines.append(f'    inner_obj = obj%{mname}')
                lines.append('    inner_cptr = c_loc(inner_obj)')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}')
                lines.append('')

                # Nested type setter: copy from inner c_ptr
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}, inner_cptr) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append('    type(c_ptr), value :: inner_cptr')
                lines.append(f'    type({type_spec}), pointer :: obj')
                lines.append(f'    type({inner}), pointer :: inner_obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    '    call c_f_pointer(inner_cptr, inner_obj)')
                lines.append(f'    obj%{mname} = inner_obj')
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
                continue

            if _is_char_member(mvar):
                char_len = _get_char_len(mvar)
                # Character getter: copies member into C buffer
                lines.append(
                    f'  subroutine f2py_get_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}, buf, buflen) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    integer(c_int), value :: buflen')
                lines.append(
                    '    character(c_char), intent(out) :: buf(buflen)')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    integer :: i')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    do i = 1, min(buflen, '
                    f'len_trim(obj%{mname}))')
                lines.append(
                    f'      buf(i) = obj%{mname}(i:i)')
                lines.append('    end do')
                lines.append(
                    f'    do i = len_trim(obj%{mname})+1, buflen')
                lines.append('      buf(i) = \' \'')
                lines.append('    end do')
                lines.append(
                    f'  end subroutine f2py_get_{typename}_{mname}')
                lines.append('')

                # Character setter: copies C buffer into member
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}, buf, buflen) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    integer(c_int), value :: buflen')
                lines.append(
                    '    character(c_char), intent(in) :: buf(buflen)')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    integer :: i, copy_len')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(f'    obj%{mname} = \' \'')
                lines.append(
                    f'    copy_len = min(buflen, len(obj%{mname}))')
                lines.append('    do i = 1, copy_len')
                lines.append(
                    f'      obj%{mname}(i:i) = buf(i)')
                lines.append('    end do')
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
                continue

            if _is_deferred_char_member(mvar):
                # Deferred-length allocatable character
                # F2018 7.4.4.2 paragraph 3: length determined at runtime

                # _allocated: returns logical(c_bool)
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_allocated(cptr{len_arg_str_prefix}) '
                    f'result(is_alloc) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    logical(c_bool) :: is_alloc')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    is_alloc = allocated(obj%{mname})')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_allocated')
                lines.append('')

                # _len: returns integer(c_int)
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_len(cptr{len_arg_str_prefix}) '
                    f'result(str_length) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    integer(c_int) :: str_length')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    if (allocated(obj%{mname})) then')
                lines.append(
                    f'      str_length = len(obj%{mname})')
                lines.append('    else')
                lines.append('      str_length = 0')
                lines.append('    end if')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_len')
                lines.append('')

                # getter: copies character into C buffer
                lines.append(
                    f'  subroutine f2py_get_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}, buf, buflen) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    integer(c_int), value :: buflen')
                lines.append(
                    '    character(c_char), intent(out) '
                    ':: buf(buflen)')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append(
                    '    integer :: char_idx')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    do char_idx = 1, '
                    f'min(buflen, len(obj%{mname}))')
                lines.append(
                    f'      buf(char_idx) = '
                    f'obj%{mname}(char_idx:char_idx)')
                lines.append('    end do')
                lines.append(
                    f'  end subroutine f2py_get_{typename}_{mname}')
                lines.append('')

                # setter: deallocate, allocate with new length, copy
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}, buf, buflen) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    integer(c_int), value :: buflen')
                lines.append(
                    '    character(c_char), intent(in) '
                    ':: buf(buflen)')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append(
                    '    integer :: char_idx')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    if (allocated(obj%{mname})) '
                    f'deallocate(obj%{mname})')
                lines.append('    if (buflen > 0) then')
                lines.append(
                    f'      allocate(character(buflen) '
                    f':: obj%{mname})')
                lines.append(
                    '      do char_idx = 1, buflen')
                lines.append(
                    f'        obj%{mname}(char_idx:char_idx) '
                    f'= buf(char_idx)')
                lines.append('      end do')
                lines.append('    end if')
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
                continue

            if _is_allocatable_member(mvar):
                isoc_type = _get_member_isoc_type(mvar)
                if isoc_type is None:
                    continue
                ndim = _get_alloc_ndim(mvar)

                # _allocated
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_allocated(cptr{len_arg_str_prefix}) '
                    f'result(is_alloc) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    logical(c_bool) :: is_alloc')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    is_alloc = allocated(obj%{mname})')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_allocated')
                lines.append('')

                # _ndim
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_ndim(cptr{len_arg_str_prefix}) '
                    f'result(array_rank) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    integer(c_int) :: array_rank')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    array_rank = {ndim}')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_ndim')
                lines.append('')

                # _shape
                lines.append(
                    f'  subroutine f2py_get_{typename}_{mname}'
                    f'_shape(cptr{len_arg_str_prefix}, shape_out) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    f'    integer(c_int) :: shape_out({ndim})')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append(
                    '    integer :: dim_idx')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    if (allocated(obj%{mname})) then')
                for dim_idx in range(1, ndim + 1):
                    lines.append(
                        f'      shape_out({dim_idx}) = '
                        f'size(obj%{mname}, {dim_idx})')
                lines.append('    else')
                for dim_idx in range(1, ndim + 1):
                    lines.append(
                        f'      shape_out({dim_idx}) = 0')
                lines.append('    end if')
                lines.append(
                    f'  end subroutine f2py_get_{typename}_{mname}'
                    f'_shape')
                lines.append('')

                # _data
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_data(cptr{len_arg_str_prefix}) '
                    f'result(data_ptr) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append('    type(c_ptr) :: data_ptr')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    if (allocated(obj%{mname})) then')
                lines.append(
                    f'      data_ptr = c_loc(obj%{mname})')
                lines.append('    else')
                lines.append('      data_ptr = c_null_ptr')
                lines.append('    end if')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_data')
                lines.append('')

                # setter
                dim_args = ', '.join(
                    f'dim_{idx}' for idx in range(1, ndim + 1))
                alloc_shape = ', '.join(
                    f'dim_{idx}' for idx in range(1, ndim + 1))
                colon_shape = ', '.join([':'] * ndim)
                cf_shape = ', '.join(
                    f'dim_{idx}' for idx in range(1, ndim + 1))

                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}, {dim_args}, src) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                for idx in range(1, ndim + 1):
                    lines.append(
                        f'    integer(c_int), value :: dim_{idx}')
                lines.append('    type(c_ptr), value :: src')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append(
                    f'    {isoc_type}, pointer :: '
                    f'src_arr({colon_shape})')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    if (allocated(obj%{mname})) '
                    f'deallocate(obj%{mname})')
                lines.append('    if (dim_1 > 0) then')
                lines.append(
                    f'      allocate(obj%{mname}'
                    f'({alloc_shape}))')
                lines.append(
                    f'      call c_f_pointer(src, src_arr, '
                    f'[{cf_shape}])')
                lines.append(
                    '      ' + _member_assign(mvar, f'obj%{mname}', 'src_arr'))
                lines.append('    end if')
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
                continue

            if _is_pointer_member(mvar):
                isoc_type = _get_member_isoc_type(mvar)
                if isoc_type is None:
                    continue
                ndim = _get_pointer_ndim(mvar)

                # _associated
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_associated(cptr{len_arg_str_prefix}) '
                    f'result(is_assoc) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    logical(c_bool) :: is_assoc')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    is_assoc = associated(obj%{mname})')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_associated')
                lines.append('')

                # _ndim
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_ndim(cptr{len_arg_str_prefix}) '
                    f'result(array_rank) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    '    integer(c_int) :: array_rank')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    array_rank = {ndim}')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_ndim')
                lines.append('')

                # _shape
                lines.append(
                    f'  subroutine f2py_get_{typename}_{mname}'
                    f'_shape(cptr{len_arg_str_prefix}, shape_out) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(
                    f'    integer(c_int) :: shape_out({ndim})')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    if (associated(obj%{mname})) then')
                for dim_idx in range(1, ndim + 1):
                    lines.append(
                        f'      shape_out({dim_idx}) = '
                        f'size(obj%{mname}, {dim_idx})')
                lines.append('    else')
                for dim_idx in range(1, ndim + 1):
                    lines.append(
                        f'      shape_out({dim_idx}) = 0')
                lines.append('    end if')
                lines.append(
                    f'  end subroutine f2py_get_{typename}_{mname}'
                    f'_shape')
                lines.append('')

                # _data
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_data(cptr{len_arg_str_prefix}) '
                    f'result(data_ptr) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append('    type(c_ptr) :: data_ptr')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    if (associated(obj%{mname})) then')
                lines.append(
                    f'      data_ptr = c_loc(obj%{mname})')
                lines.append('    else')
                lines.append('      data_ptr = c_null_ptr')
                lines.append('    end if')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_data')
                lines.append('')
                continue

            # LEN-sized array members: arrays whose dimension is a
            # LEN type parameter (e.g. real :: data(n) where n is LEN).
            # Unlike allocatables, these exist immediately after
            # allocation with the correct size.  No allocated() check.
            if (len_param_names
                    and _is_len_sized_array(mvar, len_param_names)):
                isoc_type = _get_member_isoc_type(mvar)
                if isoc_type is None:
                    continue
                # _data: returns c_ptr to the array data
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'_data(cptr{len_arg_str_prefix}) '
                    f'result(data_ptr) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append('    type(c_ptr) :: data_ptr')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                # Determine which LEN param sizes this dimension
                dim_exprs = [str(d).strip().lower()
                             for d in mvar.get('dimension', [])]
                # Use size() for robustness
                lines.append(
                    f'    if (size(obj%{mname}) > 0) then')
                lines.append(
                    f'      data_ptr = c_loc(obj%{mname})')
                lines.append('    else')
                lines.append('      data_ptr = c_null_ptr')
                lines.append('    end if')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}'
                    f'_data')
                lines.append('')

                # setter: copy from C source into member array
                ndim = len(mvar.get('dimension', []))
                dim_args = ', '.join(
                    f'dim_{idx}' for idx in range(1, ndim + 1))
                colon_shape = ', '.join([':'] * ndim)
                cf_shape = ', '.join(
                    f'dim_{idx}' for idx in range(1, ndim + 1))

                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}, {dim_args}, src) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                for idx in range(1, ndim + 1):
                    lines.append(
                        f'    integer(c_int), value :: dim_{idx}')
                lines.append('    type(c_ptr), value :: src')
                lines.append(
                    f'    type({type_spec}), pointer :: obj')
                lines.append(
                    f'    {isoc_type}, pointer :: '
                    f'src_arr({colon_shape})')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    call c_f_pointer(src, src_arr, '
                    f'[{cf_shape}])')
                lines.append(
                    '    ' + _member_assign(mvar, f'obj%{mname}', 'src_arr'))
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
                continue

            isoc_type = _get_member_isoc_type(mvar)
            if isoc_type is None:
                continue
            dims = _get_array_dims(mvar)

            if dims:
                # Fixed-size array getter
                total = 1
                for d in dims:
                    total *= d
                dim_str = ', '.join(str(d) for d in dims)

                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}) '
                    f'result(arrptr) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append('    type(c_ptr) :: arrptr')
                lines.append(f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(f'    arrptr = c_loc(obj%{mname})')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}')
                lines.append('')

                # Fixed-size array setter
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}, arr) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(f'    {isoc_type}, intent(in) :: arr({total})')
                lines.append(f'    type({type_spec}), pointer :: obj')
                lines.append('    integer :: i')
                lines.append('    call c_f_pointer(cptr, obj)')
                if len(dims) == 1:
                    lines.append(
                        '    ' + _member_assign(mvar, f'obj%{mname}', 'arr'))
                else:
                    lines.append('    ' + _member_assign(
                        mvar, f'obj%{mname}', f'reshape(arr, (/{dim_str}/))'))
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')
            else:
                # Scalar getter
                lines.append(
                    f'  function f2py_get_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}) '
                    f'result(val) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(f'    {isoc_type} :: val')
                lines.append(f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    f'    val = {_member_value(mvar, f"obj%{mname}")}')
                lines.append(
                    f'  end function f2py_get_{typename}_{mname}')
                lines.append('')

                # Scalar setter
                lines.append(
                    f'  subroutine f2py_set_{typename}_{mname}'
                    f'(cptr{len_arg_str_prefix}, val) bind(c)')
                lines.append('    type(c_ptr), value :: cptr')
                if len_decl_str:
                    lines.append(len_decl_str)
                lines.append(f'    {isoc_type}, value :: val')
                lines.append(f'    type({type_spec}), pointer :: obj')
                lines.append('    call c_f_pointer(cptr, obj)')
                lines.append(
                    '    ' + _member_assign(mvar, f'obj%{mname}', 'val'))
                lines.append(
                    f'  end subroutine f2py_set_{typename}_{mname}')
                lines.append('')

    # Add operator wrappers
    lines.extend(operator_wrappers)

    # Add procedure pointer component wrappers (F2018 7.5.4.4)
    if module_block:
        source_file = _source_file(module_block)
        for tb in opaque_types:
            typename = tb['name']
            proc_ptrs = _type_proc_pointers(source_file, tb, type_map)
            if proc_ptrs:
                pp_fortran = _gen_proc_pointer_fortran_wrappers(
                    typename, proc_ptrs, module_block.get('body', []))
                if pp_fortran:
                    lines.append('')
                    lines.append(
                        f'  ! Procedure pointer wrappers for {typename}')
                    lines.append(pp_fortran)

    # Add defined I/O wrappers (F2018 12.6.4.8)
    if module_block:
        source_file = _source_file(module_block)
        for tb in opaque_types:
            typename = tb['name']
            bound_procs = _scan_type_bound_procedures(source_file, typename)
            if '__write_formatted__' in bound_procs:
                lines.append('')
                lines.append(
                    f'  ! Defined I/O write wrapper for {typename}')
                lines.append(
                    _gen_defined_io_fortran_wrapper(typename))
            if '__read_formatted__' in bound_procs:
                lines.append('')
                lines.append(
                    f'  ! Defined I/O read wrapper for {typename}')
                lines.append(_gen_defined_io_read_fortran_wrapper(
                    typename, *_initial_state(
                        'obj', _all_components(tb, type_map), type_map)))

    lines.append(f'end module f2py_{modulename}_derived_wrappers')
    lines.append('')

    return '\n'.join(lines)


def write_fortran_wrappers(buildpath, modulename, type_blocks,
                           routines=None, module_block=None):
    """Write Fortran wrapper file if opaque types or wrapped routines exist.

    Returns the filename if written, None otherwise.
    """
    source = generate_fortran_wrappers(modulename, type_blocks,
                                       routines=routines,
                                       module_block=module_block)
    if source is None:
        return None

    fname = f'{modulename}-f2pyderivedwrappers.f90'
    fpath = os.path.join(buildpath, fname)
    with open(fpath, 'w') as f:
        f.write(source)
    outmess(f'\tFortran derived type wrappers saved to "{fpath}"\n')
    return fname


# --- Derived type arguments of routines ---
#
# A derived type argument reaches Fortran as a handle, the c_ptr of the
# object that its Python instance holds.  The rules for derived types in
# rules.arg_rules turn instances into handles and back, and the wrapper
# that func2subr writes for the routine points a pointer of the right type
# at each handle with c_f_pointer before it calls the routine.


def _has_derived_type_args(routine):
    """Check if a routine has any derived type arguments or returns one."""
    for argname in routine.get('args', []):
        var = routine['vars'].get(argname, {})
        if var.get('typespec') == 'type':
            return True
    if isfunction(routine):
        result_var = routine.get('result', routine['name'])
        rvar = routine['vars'].get(result_var, {})
        if rvar.get('typespec') == 'type':
            return True
    return False


def _type_param_value(value, params):
    """Evaluate a type parameter value, which may name a parameter."""
    value = value.strip().lower()
    seen = set()
    while value in params and value not in seen:
        seen.add(value)
        value = str(params[value]).strip().lower()
    return _eval_kind_expr(value)


def _dt_specialization(info, typeparams, params):
    """Return (spec, capsule, kind_value) for a dummy of a wrapped type.

    typeparams is the parameter list of the ``type(name(...))`` specifier
    of the dummy, giving KIND parameters by position or by name, and
    params maps module parameter names to their values.  Returns None
    when the dummy selects a specialization that is not wrapped.
    """
    if not info['specs']:
        return info['spec'], f"f2py.{info['name']}", 0
    values = {k['name']: k['default'] for k in info['kind_params']}
    if typeparams:
        order = info['type_param_order']
        for i, item in enumerate(markoutercomma(typeparams).split('@,@')):
            name, _, value = item.rpartition('=')
            name = name.strip().lower()
            if not name and i < len(order):
                name = order[i]
            if name in values:
                values[name] = _type_param_value(value, params)
    for kind_dict, suffix in info['specs']:
        if all(values.get(k) == v for k, v in kind_dict.items()):
            spec = _get_fortran_type_spec(info['block'],
                                          kind_override=kind_dict)
            kind_value = (next(iter(kind_dict.values()))
                          if len(kind_dict) == 1 else 0)
            return spec, f"f2py.{info['name']}{suffix}", kind_value
    return None


def _dt_cases(info, types):
    """Return the concrete types a class(T) dummy of a wrapped type takes.

    Each case is (tag, spec, module, name), the declared type first and
    then its extensions in the module of the declared type.
    """
    cases = []
    for t in types.values():
        if t['module'] != info['module'] or t['abstract'] or t['specs']:
            continue
        parent = t
        while parent is not None and parent is not info:
            parent = types.get(parent['parent'])
        if parent is info:
            cases.append((t['tag'], t['spec'], t['module'], t['name']))
    return sorted(cases)


def prepare_routine(rout, types, params=None):
    """Describe the derived type arguments of rout as handles.

    types describes the wrapped derived types by their lower case names,
    see derived_type_rules.describe_types, and params maps the names of
    module parameters to their values.  Each derived type argument and
    derived type function result gets an 'f2py_dt' entry, which
    capi_maps.getdtmap and the wrapper that func2subr writes read.  The
    dynamic type of a class(T) argument with extensions or of a class(*)
    argument is passed in a hidden argument, f2py_<name>_tag.

    Returns what f2py cannot pass, for the message that skips rout.
    """
    params = params or {}
    missing = []
    rout.setdefault('f2py_dt_args', list(rout['args']))
    names = list(rout['args'])
    if isfunction(rout):
        names.append(rout.get('result', rout['name']))
    module = rout.get('modulename')
    for a in names:
        var = rout['vars'].get(a, {})
        if not isderivedtype(var) or 'f2py_dt' in var:
            continue
        out = isintent_out(var) and isintent_hide(var)
        storage = _dt_storage(var)
        if storage and (isunlimitedpolymorphic(var) or (
                _is_polymorphic_arg(var) and (out or a not in rout['args']))):
            # Python would need the dynamic type of what Fortran allocates
            missing.append(f'the polymorphic {storage} {a}')
            continue
        if isunlimitedpolymorphic(var):
            cases = [(tag, decl, None, None)
                     for tag, decl, _ in dt_any_intrinsics]
            cases += sorted((t['tag'], t['spec'], t['module'], t['name'])
                            for t in types.values()
                            if t['module'] == module and not t['abstract']
                            and not t['specs'])
            var['f2py_dt'] = {'any': True, 'poly': True, 'module': module,
                              'bindc': False, 'cases': cases}
        else:
            typename = var.get('typename', '')
            info = types.get(typename.lower())
            spec = (info and _dt_specialization(
                info, var.get('typeparams'), params))
            if spec is None:
                missing.append(f'derived type {typename}, which f2py does '
                               'not wrap')
                continue
            spec, capsule, kind_value = spec
            vector = out and isarray(var) and (
                isallocatable(var) or 'pointer' in var.get('attrspec', []))
            if (_is_polymorphic_arg(var) and not out and not vector
                    and not info['specs']):
                cases = _dt_cases(info, types)
                poly = True
            else:
                cases = [(info['tag'], spec, info['module'], info['name'])]
                poly = False
            if not cases or (out and info['abstract']):
                missing.append(f'{a} of the abstract type {typename}')
                continue
            var['f2py_dt'] = {'any': False, 'poly': poly,
                              'name': info['name'],
                              'module': info['module'], 'spec': spec,
                              'capsule': capsule, 'kind': kind_value,
                              'tag': info['tag'], 'bindc': info['bindc'],
                              'cases': cases}
            if vector:
                # Fortran gives a buffer of handles of copies of the
                # elements, and their extents in hidden arguments
                rank = len(var.pop('dimension'))
                var['f2py_dt']['vector'] = rank
                var['f2py_dt']['vector_free'] = (
                    f"f2py_{module}_{rout['name']}_free_{a}".lower())
                for k in range(1, rank + 1):
                    extent = f'f2py_{a}_n{k}'
                    rout['vars'][extent] = {'typespec': 'integer',
                                            'intent': ['hide'],
                                            'f2py_dt_hidden': a}
                    rout['args'].append(extent)
        if len(var['f2py_dt']['cases']) > 1 and a in rout['args']:
            tag = f'f2py_{a}_tag'
            rout['vars'][tag] = {'typespec': 'integer', 'intent': ['hide'],
                                 '=': f'{a}_tag', 'depend': [a],
                                 'f2py_dt_hidden': a}
            rout['args'].append(tag)
    return missing


def _dt_loops(a, rank, body):
    """Return nested loops over the elements of handle array a."""
    lines = []
    for k in range(rank, 0, -1):
        lines.append('  ' * (rank - k)
                     + f'do f2py_{a}_i{k} = 1, size({a}, {k})')
    lines += ['  ' * rank + line for line in body]
    for k in range(1, rank + 1):
        lines.append('  ' * (rank - k) + 'end do')
    return lines


def _dt_bound_loops(a, rank, body):
    """Return nested loops over the elements of array f2py_<a>."""
    lines = []
    for k in range(rank, 0, -1):
        lines.append('  ' * (rank - k) + f'do f2py_{a}_i{k} = '
                     f'lbound(f2py_{a}, {k}), ubound(f2py_{a}, {k})')
    lines += ['  ' * rank + line for line in body]
    for k in range(1, rank + 1):
        lines.append('  ' * (rank - k) + 'end do')
    return lines


def _dt_vector_free(dt):
    """Return a bind(c) subroutine that releases a buffer of handles."""
    free = dt['vector_free']
    lines = [f'subroutine {free}(h, n) bind(c)', 'use iso_c_binding']
    if dt['bindc']:
        lines.append(f"use {dt['module']}, only : {dt['name']}")
    lines += ['type(c_ptr), value :: h', 'integer(c_int), value :: n',
              'type(c_ptr), pointer :: f2py_h(:)']
    if dt['bindc']:
        # instances own copies of the elements of a bind(c) type
        lines += [f"type({dt['spec']}), pointer :: f2py_e",
                  'integer f2py_i',
                  'call c_f_pointer(h, f2py_h, [n])',
                  'do f2py_i = 1, n',
                  '  call c_f_pointer(f2py_h(f2py_i), f2py_e)',
                  '  deallocate(f2py_e)',
                  'end do']
    else:
        lines.append('call c_f_pointer(h, f2py_h, [n])')
    lines += ['deallocate(f2py_h)', f'end subroutine {free}']
    return lines


def _dt_select(selects, names, call):
    """Return the lines of call inside the selection of dynamic types."""
    if not selects:
        return call(names)
    (a, cases), rest = selects[0], selects[1:]
    lines = [f'select case (f2py_{a}_tag)']
    for tag, local in cases:
        lines.append(f'case ({tag})')
        lines.append(f'  call c_f_pointer({a}, {local})')
        lines += ['  ' + line
                  for line in _dt_select(rest, {**names, a: local}, call)]
    lines.append('end select')
    return lines


_DT_MADE = {'allocatable': 'allocated', 'pointer': 'associated'}


def _dt_storage(var):
    """Return 'allocatable' or 'pointer' for such a dummy, else None."""
    if isallocatable(var):
        return 'allocatable'
    if 'pointer' in var.get('attrspec', []):
        return 'pointer'
    return None


def wrapper_handles(rout, args, result=None):
    """Return the code that the f2py wrapper of rout needs for its handles.

    args are the dummy arguments of the wrapper, including ``result``,
    the one that returns the value of a function.  Returns (uses, decls,
    pre, post, body, helpers): the use statements and declarations of
    local variables the wrapper needs, the statements before and after the
    call, body, and the lines of the procedures it needs besides.  body
    takes a function from {argument: actual argument} to the lines of the
    call, and returns these lines inside the selection of the dynamic
    types of class(T) and class(*) arguments.

    Allocatable and pointer dummies and results that Fortran gives back
    are copied into new instances, or are None if not allocated or
    associated.  An allocatable dummy gets an allocated copy of its
    object, and an allocatable or pointer array copies of its elements,
    which are copied back unless the dummy is intent(in), changed its
    dynamic type or its shape.
    """
    vars = rout['vars']
    uses, decls, pre, post, helpers = {}, [], [], [], []
    names, selects = {}, []
    for a in args:
        dt = vars[a].get('f2py_dt')
        if dt is None:
            continue
        var = vars[a]
        for _, _, module, name in dt['cases']:
            if module is not None:
                uses.setdefault(module, set()).add(name)
        out = isintent_out(var) and isintent_hide(var)
        if 'vector' in dt:
            rank = dt['vector']
            idx = ', '.join(f'f2py_{a}_i{k}' for k in range(1, rank + 1))
            if 'pointer' in var.get('attrspec', []):
                storage, made = 'pointer', 'associated'
            else:
                storage, made = 'allocatable', 'allocated'
            decls.append(f"type({dt['spec']}), {storage} :: f2py_{a}"
                         f"({','.join(':' * rank)})")
            decls.append(f"type({dt['spec']}), pointer :: f2py_{a}_e")
            decls.append(f'type(c_ptr), pointer :: f2py_{a}_h(:)')
            decls.append(f'integer {idx}, f2py_{a}_k')
            extents = [f'f2py_{a}_n{k} = size(f2py_{a}, {k})'
                       for k in range(1, rank + 1)]
            post += [f'{a} = c_null_ptr',
                     f'f2py_{a}_n1 = -1',
                     f'if ({made}(f2py_{a})) then',
                     *['  ' + line for line in extents],
                     f'  if (size(f2py_{a}) > 0) then',
                     f'    allocate(f2py_{a}_h(size(f2py_{a})))',
                     f'    f2py_{a}_k = 0',
                     *['    ' + line for line in _dt_bound_loops(a, rank, [
                         f'f2py_{a}_k = f2py_{a}_k + 1',
                         f'allocate(f2py_{a}_e)',
                         f'f2py_{a}_e = f2py_{a}({idx})',
                         f'f2py_{a}_h(f2py_{a}_k) = c_loc(f2py_{a}_e)'])],
                     f'    {a} = c_loc(f2py_{a}_h)',
                     '  end if',
                     'end if']
            helpers += _dt_vector_free(dt)
            names[a] = f'f2py_{a}'
        elif isarray(var):
            rank = len(var['dimension'])
            idx = ', '.join(f'f2py_{a}_i{k}' for k in range(1, rank + 1))
            storage = _dt_storage(var)
            decls.append(f"type({dt['spec']}), pointer :: f2py_{a}_e")
            decls.append(f'integer {idx}')
            element = f'f2py_{a}({idx})'
            if storage is None:
                decls.append(f"type({dt['spec']}) f2py_{a}"
                             f"({','.join(var['dimension'])})")
            else:
                # the dummy gets an allocated copy; it is copied back if it
                # keeps its shape, whatever its bounds
                deferred = ','.join(':' * rank)
                store = f'f2py_{a}' if storage == 'allocatable' else (
                    f'f2py_{a}_s')
                decls.append(f"type({dt['spec']}), allocatable, target :: "
                             f"{store}({deferred})")
                pre.append(
                    f"allocate({store}({','.join(var['dimension'])}))")
                if storage == 'pointer':
                    decls.append(f"type({dt['spec']}), pointer :: "
                                 f"f2py_{a}({deferred})")
                    pre.append(f'f2py_{a} => {store}')
                element = f'f2py_{a}(' + ', '.join(
                    f'lbound(f2py_{a}, {k}) + f2py_{a}_i{k} - 1'
                    for k in range(1, rank + 1)) + ')'
            if not out:
                pre += _dt_loops(a, rank, [
                    f'call c_f_pointer({a}({idx}), f2py_{a}_e)',
                    f'f2py_{a}({idx}) = f2py_{a}_e'])
            if out and not dt['bindc']:
                post += _dt_loops(a, rank, [
                    f'allocate(f2py_{a}_e)',
                    f'f2py_{a}_e = f2py_{a}({idx})',
                    f'{a}({idx}) = c_loc(f2py_{a}_e)'])
            elif not (isintent_in(var) and not isintent_inout(var)):
                back = _dt_loops(a, rank, [
                    f'call c_f_pointer({a}({idx}), f2py_{a}_e)',
                    f'f2py_{a}_e = {element}'])
                if storage is not None:
                    made = _DT_MADE[storage]
                    back = ([f'if ({made}(f2py_{a})) then',
                             f'  if (all(shape(f2py_{a}) == shape({a}))) then']
                            + ['    ' + line for line in back]
                            + ['  end if', 'end if'])
                post += back
            names[a] = f'f2py_{a}'
        elif len(dt['cases']) > 1 or (
                _is_polymorphic_arg(var) and _dt_storage(var)):
            cases = []
            for k, (tag, spec, module, _) in enumerate(dt['cases'], 1):
                decl = f'type({spec})' if module is not None else spec
                decls.append(f'{decl}, pointer :: f2py_{a}_{k}')
                cases.append((tag, f'f2py_{a}_{k}'))
            storage = _dt_storage(var)
            if storage is None:
                selects.append((a, cases))
                continue
            # F2018 15.5.2.5: a polymorphic allocatable or pointer dummy
            # needs a polymorphic actual.  Every other dummy, including a
            # class() dummy that is neither allocatable nor a pointer,
            # accepts a type() actual, so the wrapper declares type().
            if not _is_polymorphic_arg(var):
                # A nonpolymorphic dummy must not reach the class()
                # declaration.  Name the routine and the argument so the
                # failure is the same kind of message crackfortran writes.
                detail = (
                    f"wrapper_handles: routine {rout['name']!r} argument "
                    f"{a!r} is not polymorphic, so the wrapper must not "
                    f"declare class()")
                errmess(detail + '\n')
                raise F2PYError(detail)
            decls.append(f"class({dt['spec']}), {storage} :: f2py_{a}")
            if storage == 'pointer':
                pre.append(f'nullify(f2py_{a})')
                make = f'f2py_{a} => {{}}'
            else:
                make = f'allocate(f2py_{a}, source={{}})'
            if len(cases) > 1:
                made = _dt_select([(a, cases)], {},
                                  lambda n: [make.format(n[a])])
            else:
                made = [f'call c_f_pointer({a}, {cases[0][1]})',
                        make.format(cases[0][1])]
            pre += [f'if (c_associated({a})) then',
                    *['  ' + line for line in made], 'end if']
            if storage == 'allocatable' and not (
                    isintent_in(var) and not isintent_inout(var)):
                # copied back if its dynamic type stays the same
                back = []
                for k, (tag, spec, _, _) in enumerate(dt['cases'], 1):
                    back.append([f'call c_f_pointer({a}, f2py_{a}_{k})',
                                 f'select type (f2py_{a}_v => f2py_{a})',
                                 f'type is ({spec})',
                                 f'  f2py_{a}_{k} = f2py_{a}_v',
                                 'end select'])
                if len(cases) > 1:
                    back = ([f'select case (f2py_{a}_tag)']
                            + [line for (tag, _), body in zip(cases, back)
                               for line in [f'case ({tag})',
                                            *['  ' + b for b in body]]]
                            + ['end select'])
                else:
                    back = back[0]
                post += [(f'if (c_associated({a}) .and. allocated(f2py_{a}))'
                          ' then'), *['  ' + line for line in back], 'end if']
            names[a] = f'f2py_{a}'
        else:
            spec = dt['cases'][0][1]
            storage = _dt_storage(var)
            if storage == 'allocatable' and a == result:
                storage = None  # an allocatable result is allocated
            made = _DT_MADE.get(storage)
            if storage is None:
                decls.append(f"type({spec}), pointer :: f2py_{a}")
                if out and not dt['bindc']:
                    pre.append(f'allocate(f2py_{a})')
                    post.append(f'{a} = c_loc(f2py_{a})')
                elif 'optional' in var.get('attrspec', []):
                    pre.append(f'nullify(f2py_{a})')
                    pre.append(f'if (c_associated({a})) '
                               f'call c_f_pointer({a}, f2py_{a})')
                else:
                    pre.append(f'call c_f_pointer({a}, f2py_{a})')
            elif out:
                # an instance owns a copy of what the dummy or result is
                # allocated as or associated with, or there is none
                decls.append(f"type({spec}), {storage} :: f2py_{a}")
                decls.append(f"type({spec}), pointer :: f2py_{a}_e")
                if storage == 'pointer':
                    pre.append(f'nullify(f2py_{a})')
                if dt['bindc']:
                    pre.append(f'call c_f_pointer({a}, f2py_{a}_e)')
                    post += [f'if ({made}(f2py_{a})) then',
                             f'  f2py_{a}_e = f2py_{a}',
                             'else',
                             f'  {a} = c_null_ptr',
                             'end if']
                else:
                    post += [f'{a} = c_null_ptr',
                             f'if ({made}(f2py_{a})) then',
                             f'  allocate(f2py_{a}_e)',
                             f'  f2py_{a}_e = f2py_{a}',
                             f'  {a} = c_loc(f2py_{a}_e)',
                             'end if']
            elif storage == 'pointer':
                decls.append(f"type({spec}), pointer :: f2py_{a}")
                pre.append(f'nullify(f2py_{a})')
                pre.append(f'if (c_associated({a})) '
                           f'call c_f_pointer({a}, f2py_{a})')
            else:
                # the dummy gets an allocated copy of the object
                decls.append(f"type({spec}), allocatable :: f2py_{a}")
                decls.append(f"type({spec}), pointer :: f2py_{a}_e")
                pre += [f'nullify(f2py_{a}_e)',
                        f'if (c_associated({a})) then',
                        f'  call c_f_pointer({a}, f2py_{a}_e)',
                        f'  allocate(f2py_{a}, source=f2py_{a}_e)',
                        'end if']
                if not (isintent_in(var) and not isintent_inout(var)):
                    post.append(f'if (associated(f2py_{a}_e) .and. '
                                f'allocated(f2py_{a})) f2py_{a}_e = f2py_{a}')
            names[a] = f'f2py_{a}'
    use_lines = [f"use {module}, only : {', '.join(sorted(names_))}"
                 for module, names_ in sorted(uses.items())]

    def body(call):
        return _dt_select(selects, names, call)

    return use_lines, decls, pre, post, body, helpers


def _statements(source_file):
    """Yield the statements of a free form source file, without comments
    and with continuation lines joined."""
    with open(source_file) as f:
        pending = ''
        for line in f:
            line = line.split('!')[0].rstrip()
            if line.lstrip().startswith('&'):
                line = line.lstrip()[1:]
            if line.endswith('&'):
                pending += line[:-1]
                continue
            for statement in (pending + line).split(';'):
                if statement.strip():
                    yield statement.strip()
            pending = ''


def _type_definition(source_file, typename):
    """Yield (part, statement) for the statements of a type definition.

    part is 'components' before its CONTAINS statement and 'bindings'
    after it.  crackfortran does not keep these statements, so they are
    read from the source.
    """
    import re
    if not source_file or not os.path.isfile(source_file):
        return
    type_pat = re.compile(
        r'^type\b(?:\s*,\s*[\w()]+)*\s*::\s*' + re.escape(typename) + r'\b',
        re.I)
    end_type_pat = re.compile(r'^end\s*type\b', re.I)
    contains_pat = re.compile(r'^contains\b', re.I)
    part = None
    for statement in _statements(source_file):
        if part is None:
            if type_pat.match(statement):
                part = 'components'
        elif end_type_pat.match(statement):
            return
        elif contains_pat.match(statement):
            part = 'bindings'
        else:
            yield part, statement


def _scan_type_bound_procedures(source_file, typename, attrs=None):
    """Scan Fortran source for type-bound procedure declarations.

    Returns a dict mapping method_name -> impl_name from lines like:
        procedure :: method => impl
        procedure, pass(this) :: method  (impl == method when no =>)
    within the 'contains' section of the named type block, and maps
    a generic binding to the list of its specific bindings.  attrs, when
    given, gets the passed-object dummy of each specific binding, as
    {'pass': name or None for the first dummy, 'nopass': bool}.
    Deferred and private bindings are left out.
    """
    import re
    result = {}
    # F2018 R749: PROCEDURE [(interface)] [[, binding-attr-list] ::]
    # type-bound-proc-decl-list, where each declaration is
    # binding-name [=> procedure-name]
    proc_pat = re.compile(
        r'^procedure\s*(?:\(\s*\w+\s*\))?\s*(?:,(?P<attrs>[^:]*))?'
        r'(?:::)?\s*(?P<decls>\w+\s*(?:=>\s*\w+)?'
        r'(?:\s*,\s*\w+\s*(?:=>\s*\w+)?)*)\s*$', re.I)
    pass_pat = re.compile(r'^pass\s*(?:\(\s*(\w+)\s*\))?$', re.I)
    # F2018 7.5.5 paragraph 3, R751:
    # GENERIC [ , access-spec ] :: generic-spec => binding-name-list
    # Maps a generic name to multiple specific procedures.
    generic_pat = re.compile(
        r'^generic\s*(?:,\s*(\w+))?\s*::\s*(\w+)\s*=>\s*(.+)\s*$',
        re.I)
    # F2018 12.6.4.8: Defined I/O generics
    # GENERIC :: WRITE(FORMATTED) => impl_name
    # GENERIC :: READ(FORMATTED) => impl_name
    dio_pat = re.compile(
        r'^generic\s*(?:,\s*\w+)?\s*::\s*'
        r'(write|read)\s*\(\s*(formatted|unformatted)\s*\)\s*=>\s*(.+)\s*$',
        re.I)

    for part, stripped in _type_definition(source_file, typename):
        if part != 'bindings':
            continue
        m = proc_pat.match(stripped)
        if m:
            binding_attrs = [a.strip().lower()
                             for a in (m.group('attrs') or '').split(',')
                             if a.strip()]
            # Deferred TBPs (F2018 R752) have no implementation
            if 'deferred' in binding_attrs or 'private' in binding_attrs:
                continue
            passed = {'pass': None, 'nopass': 'nopass' in binding_attrs}
            for a in binding_attrs:
                pm = pass_pat.match(a)
                if pm and pm.group(1):
                    passed['pass'] = pm.group(1).lower()
            for decl in m.group('decls').split(','):
                method_name, _, impl_name = decl.partition('=>')
                method_name = method_name.strip().lower()
                result[method_name] = (impl_name.strip().lower()
                                       or method_name)
                if attrs is not None:
                    attrs[method_name] = passed
            continue
        # Defined I/O generics (F2018 12.6.4.8)
        dio_match = dio_pat.match(stripped)
        if dio_match:
            direction = dio_match.group(1).lower()
            formatting = dio_match.group(2).lower()
            impl = dio_match.group(3).strip().lower()
            key = f'__{direction}_{formatting}__'
            result[key] = impl
            continue
        # Generic bindings (F2018 R751): store all specific
        # bindings for runtime dispatch.  The result dict maps
        # generic_name -> impl_name (single binding) or
        # generic_name -> [impl1, impl2, ...] (multiple).
        generic_match = generic_pat.match(stripped)
        if generic_match:
            if (generic_match.group(1) or '').lower() == 'private':
                continue
            generic_name = generic_match.group(2).lower()
            bindings = [
                binding.strip().lower()
                for binding in generic_match.group(3).split(',')
            ]
            if len(bindings) == 1:
                result[generic_name] = bindings[0]
            elif len(bindings) > 1:
                result[generic_name] = bindings

    return result


def _scan_final_subroutines(source_file, typename):
    """Scan Fortran source for FINAL subroutine declarations in a type.

    Returns a set of lowercased final subroutine names from lines like:
        final :: sub1, sub2
    within the 'contains' section of the named type block.

    F2018 7.5.6.1 (R753): FINAL [ :: ] final-subroutine-name-list
    """
    import re
    result = set()
    final_pat = re.compile(
        r'^final\s*(?:::)?\s*(.+)\s*$', re.I)
    for part, stripped in _type_definition(source_file, typename):
        if part != 'bindings':
            continue
        final_match = final_pat.match(stripped)
        if final_match:
            names = final_match.group(1)
            for name in names.split(','):
                name = name.strip().lower()
                if name:
                    result.add(name)

    return result


def _scan_proc_pointer_components(source_file, typename):
    """Scan Fortran source for procedure pointer components in a type.

    Returns a list of dicts, each with:
        'name': component name (lowercased)
        'interface': interface name from PROCEDURE(interface)
        'nopass': True if NOPASS attribute present
        'pass': the passed-object dummy named by PASS(arg), or None for
            the first dummy
        'init_null': True if initialized with => null()

    Scans the component-part (before CONTAINS) for lines like:
        procedure(real_func), pointer, nopass :: func => null()
        procedure(sub_iface), pointer :: action

    F2018 7.5.4.1 (R741): proc-component-def-stmt
    """
    import re
    result = []
    # Match: procedure(interface), pointer [, nopass] [, pass(arg)] :: name [=> null()]
    proc_ptr_pat = re.compile(
        r'^procedure\s*\(\s*(\w+)\s*\)\s*,'      # procedure(interface),
        r'\s*(.*?)'                               # attributes
        r'\s*::\s*(\w+)'                          # :: name
        r'(?:\s*=>\s*null\s*\(\s*\))?'            # optional => null()
        r'\s*$',
        re.I)
    pass_pat = re.compile(r'^pass\s*\(\s*(\w+)\s*\)$', re.I)

    for part, stripped in _type_definition(source_file, typename):
        if part != 'components':
            continue
        m = proc_ptr_pat.match(stripped)
        if m:
            interface = m.group(1).lower()
            attrs_str = m.group(2).lower()
            comp_name = m.group(3).lower()
            attrs = [a.strip() for a in attrs_str.split(',')]
            if 'pointer' not in attrs:
                continue  # must have POINTER attribute
            passed = None
            for a in attrs:
                pm = pass_pat.match(a)
                if pm:
                    passed = pm.group(1).lower()
            result.append({
                'name': comp_name,
                'interface': interface,
                'nopass': 'nopass' in attrs,
                'pass': passed,
                'init_null': re.search(r'=>\s*null\s*\(', stripped,
                                       re.I) is not None,
            })

    return result


def _type_proc_pointers(source_file, tb, type_map):
    """Return the procedure pointer components of a type, inherited ones
    first, as _scan_proc_pointer_components describes them."""
    parent = (_get_extends_parent(tb) or '').lower()
    inherited = (_type_proc_pointers(source_file, type_map[parent], type_map)
                 if parent in type_map else [])
    own = _scan_proc_pointer_components(source_file, tb['name'])
    names = {pp['name'] for pp in own}
    return [pp for pp in inherited if pp['name'] not in names] + own


def _find_interface_block(module_body, interface_name):
    """Find the interface a PROCEDURE(name) declaration refers to.

    Searches the bodies of interface and abstract interface blocks, and
    then the procedures of the module.  Returns the block dict (with
    'args', 'vars', 'block' keys) or None if not found.
    """
    name = interface_name.lower()
    for block in module_body:
        if block.get('block') in ('interface', 'abstract interface'):
            for sub in block.get('body', []):
                if sub.get('name', '').lower() == name:
                    return sub
    for block in module_body:
        if (block.get('block') in ('function', 'subroutine')
                and block.get('name', '').lower() == name):
            return block
    return None


def _proc_pointer_call(pp, module_body):
    """Describe how Python calls procedure pointer component pp.

    Returns (args, result, problem): the dummies Python passes, each as
    (name, var), the variable of the function result or None, and why
    f2py cannot call the component, or None.  Python passes scalars
    that are intent(in) or have no intent, of types with a C type; the
    passed-object dummy (F2018 7.5.4.5) is the instance itself.
    """
    iface = _find_interface_block(module_body, pp['interface'])
    if iface is None:
        return [], None, f'its interface {pp["interface"]} is not found'
    names = list(iface.get('args', []))
    if not pp['nopass'] and names:
        names.remove(pp['pass'] if pp['pass'] in names else names[0])
    variables = iface.get('vars', {})
    args = [(name, variables.get(name, {})) for name in names]
    result = None
    if iface.get('block') == 'function':
        result = variables.get(iface.get('result', iface['name']), {})
    for name, var in args + [('the result', result)] * (result is not None):
        intent = var.get('intent', [])
        if (isarray(var) or _get_member_isoc_type(var) is None
                or _get_member_ctype(var) not in _C_TO_PYOBJ
                or (name != 'the result' and set(intent) - {'in'})):
            return [], None, f'f2py cannot pass {name}'
    return args, result, None


def _gen_proc_pointer_fortran_wrappers(typename, proc_ptrs, module_body):
    """Generate Fortran bind(c) wrappers for procedure pointer components.

    For each procedure pointer component, generates a wrapper like:
        subroutine f2py_call_TYPE_COMP(self_ptr, f2py_ok, arg1, ...) bind(c)
            type(c_ptr), value :: self_ptr
            integer(c_int), intent(out) :: f2py_ok
            type(TYPE), pointer :: self
            call c_f_pointer(self_ptr, self)
            f2py_ok = merge(1, 0, associated(self%comp))
            if (f2py_ok == 1) call self%comp(arg1, ...)
        end subroutine
    where f2py_ok tells whether the component is associated, as calling a
    disassociated procedure pointer is not allowed (F2018 15.5.1).
    """
    lines = []
    for pp in proc_ptrs:
        args, result, problem = _proc_pointer_call(pp, module_body)
        if problem:
            outmess(f'\t\tf2py does not wrap procedure pointer component '
                    f'{pp["name"]} of {typename}, as {problem}\n')
            continue
        comp = pp['name']
        dummies = ['self_ptr', 'f2py_ok'] + [name for name, _ in args]
        decls = ['    type(c_ptr), value :: self_ptr',
                 '    integer(c_int), intent(out) :: f2py_ok']
        decls += [f'    {_get_member_isoc_type(var)}, value :: {name}'
                  for name, var in args]
        actual = ', '.join(_member_actual(var, name) for name, var in args)
        if result is not None:
            dummies.append('f2py_result')
            decls.append(f'    {_get_member_isoc_type(result)}, '
                         'intent(out) :: f2py_result')
            call = (f'f2py_result = '
                    f'{_member_value(result, f"self%{comp}({actual})")}')
        else:
            call = f'call self%{comp}({actual})'
        lines += [(f'  subroutine f2py_call_{typename}_{comp}'
                   f'({", ".join(dummies)}) bind(c)'),
                  '    use iso_c_binding',
                  *decls,
                  f'    type({typename}), pointer :: self',
                  '    call c_f_pointer(self_ptr, self)',
                  '    f2py_ok = 0',
                  f'    if (.not. associated(self%{comp})) return',
                  '    f2py_ok = 1',
                  f'    {call}',
                  f'  end subroutine f2py_call_{typename}_{comp}',
                  '']
    return '\n'.join(lines)


def _gen_proc_pointer_c_methods(typename, proc_ptrs, module_body):
    """Generate C method wrappers and PyMethodDef entries for proc ptrs."""
    funcs = []
    method_entries = []
    capsule_name = f'f2py.{typename}'
    for pp in proc_ptrs:
        args, result, problem = _proc_pointer_call(pp, module_body)
        if problem:
            continue
        comp = pp['name']
        wrapper_name = f'f2py_call_{_fortran_sym(typename)}_{comp}'
        ctypes = [_get_member_ctype(var) for _, var in args]
        extern = ['void *', 'int *'] + ctypes
        decls = [f'    {ctype} {name} = 0;'
                 for (name, _), ctype in zip(args, ctypes)]
        call = ['ptr', '&ok'] + [name for name, _ in args]
        if result is not None:
            rctype = _get_member_ctype(result)
            extern.append(f'{rctype} *')
            decls.append(f'    {rctype} result = 0;')
            call.append('&result')
            ret = 'return ' + _C_TO_PYOBJ[rctype].format(val='result') + ';'
        else:
            ret = 'Py_RETURN_NONE;'
        kwlist = ', '.join([f'"{name}"' for name, _ in args] + ['NULL'])
        fmt = ''.join(_C_TO_PYFORMAT[ctype] for ctype in ctypes)
        parsed = ''.join(f', &{name}' for name, _ in args)
        method_name = f'Py{typename}_method_{comp}'
        decl_str = '\n'.join(decls)
        funcs.append(f"""\
extern void {wrapper_name}({', '.join(extern)});

static PyObject *
{method_name}(PyObject *selfobj, PyObject *args, PyObject *kwds)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        PyErr_SetString(PyExc_RuntimeError,
                        "{typename} object not initialized");
        return NULL;
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    int ok = 0;
{decl_str}
    static char *kwlist[] = {{{kwlist}}};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "{fmt}", kwlist{parsed}))
        return NULL;
    {wrapper_name}({', '.join(call)});
    if (!ok) {{
        PyErr_SetString(PyExc_TypeError,
                        "procedure pointer component {comp} of "
                        "{typename} is not associated");
        return NULL;
    }}
    {ret}
}}
""")
        method_entries.append(
            f'    {{"{comp}", (PyCFunction){method_name}, '
            f'METH_VARARGS | METH_KEYWORDS, '
            f'"Procedure pointer component {comp}"}}')
    return funcs, method_entries


def _gen_defined_io_fortran_wrapper(typename):
    """Generate Fortran bind(c) wrappers that invoke defined I/O.

    Uses the DT edit descriptor (F2018 13.7.6) to call the type's
    write(formatted) routine.  The length of the record is not known
    before it is written, and an internal file of a guessed length can
    fail in ways that do not tell running out of room apart from other
    errors, so the record goes to a scratch file, whose characters are
    counted by nonadvancing input up to the end of the record (F2018
    12.11.4).  The text is returned in an allocation that the second
    wrapper frees.
    """
    sym = _fortran_sym(typename)
    return f"""\
  subroutine f2py_str_{sym}(cptr, text, n, iostat) bind(c)
    use iso_c_binding
    use iso_fortran_env, only: iostat_eor
    type(c_ptr), value :: cptr
    type(c_ptr), intent(out) :: text
    integer(c_int), intent(out) :: n, iostat
    type({typename}), pointer :: obj
    character(kind=c_char, len=:), pointer :: record
    character(kind=c_char) :: c
    integer :: unit
    call c_f_pointer(cptr, obj)
    text = c_null_ptr
    n = 0
    open(newunit=unit, status='scratch', form='formatted', &
         action='readwrite', iostat=iostat)
    if (iostat /= 0) return
    write(unit, '(DT)', iostat=iostat) obj
    if (iostat == 0) then
      rewind(unit)
      do
        read(unit, '(a)', advance='no', iostat=iostat) c
        if (iostat /= 0) exit
        n = n + 1
      end do
      if (iostat == iostat_eor) then
        ! c_loc needs a string that is not of zero length (F2018 18.2.3.6)
        allocate(character(kind=c_char, len=max(n, 1)) :: record)
        iostat = 0
        if (n > 0) then
          rewind(unit)
          read(unit, '(a)', iostat=iostat) record
        end if
        if (iostat == 0) then
          text = c_loc(record)
        else
          deallocate(record)
        end if
      end if
    end if
    close(unit)
  end subroutine f2py_str_{sym}

  subroutine f2py_str_free_{sym}(text, n) bind(c)
    use iso_c_binding
    type(c_ptr), value :: text
    integer(c_int), value :: n
    character(kind=c_char, len=max(n, 1)), pointer :: record
    call c_f_pointer(text, record)
    deallocate(record)
  end subroutine f2py_str_free_{sym}
"""


def _gen_defined_io_read_fortran_wrapper(typename, init_lines=(),
                                         loop_depth=0):
    """Generate Fortran bind(c) wrapper for read(formatted) -> from_string.

    The new object starts as the constructor makes it, see _initial_state,
    and is destroyed again if the READ fails.
    """
    wrapper_name = f'f2py_from_str_{_fortran_sym(typename)}'
    init = ''.join(f'    {line}\n' for line in init_lines)
    loops = ''.join(f', f2py_i{d}' for d in range(1, loop_depth + 1))
    return f"""\
  subroutine {wrapper_name}(buf, buflen, cptr, iostat) bind(c)
    use iso_c_binding
    character(kind=c_char), intent(in) :: buf(*)
    integer(c_int), value :: buflen
    type(c_ptr), intent(out) :: cptr
    integer(c_int), intent(out) :: iostat
    type({typename}), pointer :: obj
    character(len=buflen) :: tmp
    integer :: i{loops}
    allocate(obj)
{init}    do i = 1, buflen
      tmp(i:i) = buf(i)
    end do
    read(tmp, '(DT)', iostat=iostat) obj
    if (iostat == 0) then
      cptr = c_loc(obj)
    else
      deallocate(obj)
      cptr = c_null_ptr
    end if
  end subroutine {wrapper_name}
"""


def _gen_defined_io_tp_str(typename):
    """Generate C tp_str function using Fortran defined I/O."""
    capsule_name = f'f2py.{typename}'
    sym = _fortran_sym(typename)
    return f"""\
extern void f2py_str_{sym}(void *, void **, int *, int *);
extern void f2py_str_free_{sym}(void *, int);

static PyObject *
Py{typename}_tp_str(PyObject *selfobj)
{{
    Py{typename}Object *self = (Py{typename}Object *)selfobj;
    if (self->capsule == NULL) {{
        return PyUnicode_FromString("{typename}(<uninitialized>)");
    }}
    void *ptr = PyCapsule_GetPointer(self->capsule, "{capsule_name}");
    if (ptr == NULL) return NULL;
    void *text = NULL;
    int n = 0, iostat = 0;
    f2py_str_{sym}(ptr, &text, &n, &iostat);
    if (text == NULL) {{
        PyErr_Format(PyExc_ValueError,
                     "write(formatted) of {typename} failed with iostat=%d",
                     iostat);
        return NULL;
    }}
    /* Trim trailing spaces */
    int len = n;
    while (len > 0 && ((const char *)text)[len - 1] == ' ') len--;
    PyObject *result = PyUnicode_FromStringAndSize((const char *)text, len);
    f2py_str_free_{sym}(text, n);
    return result;
}}
"""


def _gen_defined_io_from_string(typename):
    """Generate C class method from_string() using Fortran defined I/O read."""
    capsule_name = f'f2py.{typename}'
    wrapper_name = f'f2py_from_str_{_fortran_sym(typename)}'
    destroy = f'f2py_destroy_{_fortran_sym(typename)}'
    return f"""\
extern void {wrapper_name}(const char *, int, void **, int *);

static PyObject *
Py{typename}_from_string(PyObject *cls, PyObject *args)
{{
    const char *text;
    Py_ssize_t text_len;
    if (!PyArg_ParseTuple(args, "s#", &text, &text_len))
        return NULL;
    if (text_len > INT_MAX) {{
        PyErr_SetString(PyExc_OverflowError, "string too long");
        return NULL;
    }}
    void *ptr = NULL;
    int iostat = 0;
    {wrapper_name}(text, (int)text_len, &ptr, &iostat);
    if (ptr == NULL) {{
        PyErr_Format(PyExc_ValueError,
                     "read(formatted) of {typename} failed with iostat=%d",
                     iostat);
        return NULL;
    }}
    /* tp_new gives the instance its type tag */
    Py{typename}Object *self = (Py{typename}Object *)Py{typename}_tp_new(
        (PyTypeObject *)cls, NULL, NULL);
    if (self == NULL) {{
        {destroy}(ptr);
        return NULL;
    }}
    self->capsule = PyCapsule_New(ptr, "{capsule_name}",
                                  f2py_{typename}_capsule_destructor);
    if (self->capsule == NULL) {{
        {destroy}(ptr);
        Py_DECREF(self);
        return NULL;
    }}
    return (PyObject *)self;
}}
"""


def _resolve_binding_to_routine(binding_name, bound_procs, routine_map):
    """Resolve a binding name to its module routine through the chain."""
    resolved = binding_name
    seen = set()
    while resolved not in routine_map and resolved not in seen:
        seen.add(resolved)
        next_name = bound_procs.get(resolved)
        if isinstance(next_name, list):
            # Multi-binding generic; take first
            next_name = next_name[0]
        if next_name is None:
            break
        resolved = next_name
    return routine_map.get(resolved)


def _python_args(routine):
    """Return the arguments of a wrapped routine in the order Python takes
    them by position: the required ones, then the optional ones."""
    args = [a for a in _routine_args(routine)
            if isintent_nothide(routine['vars'][a])]
    return ([a for a in args if isrequired(routine['vars'][a])]
            + [a for a in args if isoptional(routine['vars'][a])])


def _passed_object(routine, binding):
    """Return (position, name) of the passed-object dummy of a binding in
    the Python arguments of its routine, or (-1, None) for NOPASS.  The
    passed-object dummy is the first dummy unless PASS names another
    (F2018 7.5.4.5)."""
    if binding['nopass'] or not _routine_args(routine):
        return -1, None
    name = binding['pass'] or _routine_args(routine)[0]
    args = _python_args(routine)
    if name not in args:
        return -1, None
    return args.index(name), name


def _dummy_signature(routine, passname):
    """Return C initializers of f2py_dt_dummy entries for the Python
    arguments of a routine, other than its passed-object dummy."""
    kinds = {'integer': 'i', 'real': 'f', 'double precision': 'f',
             'complex': 'c', 'double complex': 'c', 'logical': 'b',
             'character': 's', 'type': 't'}
    entries = []
    for a in _python_args(routine):
        if a == passname:
            continue
        var = routine['vars'][a]
        dt = var.get('f2py_dt')
        pytype = (f"&Py{dt['name']}_Type"
                  if dt is not None and not dt['any'] else 'NULL')
        kind = kinds.get(var.get('typespec', '').lower(), 'f')
        if dt is not None and dt['any']:
            kind = 'o'
        entries.append(f'{{"{a}", \'{kind}\', '
                       f"{len(var.get('dimension', []))}, {pytype}, "
                       f'{int(isrequired(var))}}}')
    entries.append('{NULL, 0, 0, NULL, 0}')
    return entries


def _gen_type_methods(typename, bound_procs, routines, type_map,
                      modulename, attrs=None):
    """Generate PyMethodDef entries and C method functions for type-bound
    procedures.

    Each method calls the wrapped routine of its binding, looked up in
    the Fortran module object, with the instance as the passed-object
    argument.  A generic binding calls the first of its specific bindings
    that the arguments fit.
    """
    funcs = []
    method_entries = []
    attrs = attrs or {}
    no_pass = {'pass': None, 'nopass': False}

    # Build lookup from routine name to routine block, for the routines
    # that are wrapped
    routine_map = {r['name'].lower(): r for r in routines
                   if not r.get('f2py_dt_missing')}

    def call(binding_name):
        routine = _resolve_binding_to_routine(
            binding_name, bound_procs, routine_map)
        if routine is None:
            return None, None
        pos, passname = _passed_object(
            routine, attrs.get(binding_name, no_pass))
        passarg = f'"{passname}"' if passname else 'NULL'
        return routine, (f'f2py_dt_call_bound(#modulename#_module, '
                         f'"{modulename}", "{routine["name"]}", selfobj, '
                         f'{pos}, {passarg}, args, kwds)')

    for method_name, impl_name in bound_procs.items():
        if method_name.startswith('__'):
            continue
        mfunc_name = f'Py{typename}_method_{method_name}'
        lines = ['static PyObject *',
                 f'{mfunc_name}(PyObject *selfobj, PyObject *args, PyObject *kwds)',
                 '{']
        if isinstance(impl_name, list):
            # F2018 15.4.3.4.5: the specific binding whose dummies the
            # actual arguments fit
            for k, binding_name in enumerate(impl_name):
                routine, call_expr = call(binding_name)
                if routine is None:
                    continue
                passname = _passed_object(
                    routine, attrs.get(binding_name, no_pass))[1]
                sig = ',\n        '.join(_dummy_signature(routine, passname))
                lines.append(f'    static const f2py_dt_dummy sig{k}[] = {{\n'
                             f'        {sig}}};')
                lines.append(f'    if (f2py_dt_args_match(args, kwds, sig{k}))')
                lines.append(f'        return {call_expr};')
            lines.append('    PyErr_SetString(PyExc_TypeError, '
                         f'"{method_name}(): no specific procedure matches '
                         f'the arguments");')
            lines.append('    return NULL;')
        else:
            routine, call_expr = call(method_name)
            if routine is None:
                continue
            lines.append(f'    return {call_expr};')
        lines.append('}')
        lines.append('')
        funcs.append('\n'.join(lines))
        method_entries.append(
            f'    {{"{method_name}", (PyCFunction){mfunc_name}, '
            f'METH_VARARGS | METH_KEYWORDS, '
            f'"Type-bound procedure {method_name}"}},')

    return funcs, method_entries
