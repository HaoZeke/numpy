"""

Rules for building C/API module with f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import copy
import hashlib
import re

from ._isocbind import isoc_kindmap
from .auxfuncs import (
    getfortranname,
    isexternal,
    isfunction,
    isfunction_wrap,
    isintent_in,
    isintent_out,
    islogicalfunction,
    ismoduleroutine,
    isscalar,
    issubroutine,
    issubroutine_wrap,
    outmess,
    show,
)


def var2fixfortran(vars, a, fa=None, f90mode=None):
    if fa is None:
        fa = a
    if a not in vars:
        show(vars)
        outmess(f'var2fixfortran: No definition for argument "{a}".\n')
        return ''
    if 'typespec' not in vars[a]:
        show(vars[a])
        outmess(f'var2fixfortran: No typespec for argument "{a}".\n')
        return ''
    vardef = vars[a]['typespec']
    if vardef == 'type' and 'typename' in vars[a]:
        vardef = f"{vardef}({vars[a]['typename']})"
    selector = {}
    lk = ''
    if 'kindselector' in vars[a]:
        selector = vars[a]['kindselector']
        lk = 'kind'
    elif 'charselector' in vars[a]:
        selector = vars[a]['charselector']
        lk = 'len'
    if '*' in selector:
        if f90mode:
            if selector['*'] in ['*', ':', '(*)']:
                vardef = f'{vardef}(len=*)'
            else:
                vardef = f"{vardef}({lk}={selector['*']})"
        elif selector['*'] in ['*', ':']:
            vardef = f"{vardef}*({selector['*']})"
        else:
            vardef = f"{vardef}*{selector['*']}"
    elif 'len' in selector:
        vardef = f"{vardef}(len={selector['len']}"
        if 'kind' in selector:
            vardef = f"{vardef},kind={selector['kind']})"
        else:
            vardef = f'{vardef})'
    elif 'kind' in selector:
        vardef = f"{vardef}(kind={selector['kind']})"

    vardef = f'{vardef} {fa}'
    if 'dimension' in vars[a]:
        vardef = f"{vardef}({','.join(vars[a]['dimension'])})"
    return vardef

def useiso_c_binding(rout):
    useisoc = False
    for value in rout['vars'].values():
        kind_value = value.get('kindselector', {}).get('kind')
        if kind_value in isoc_kindmap:
            return True
    return useisoc


# User-module blocks available while building F90 wrappers for one extension
# (set by rules.buildmodule from the um list; complements crackfortran.usermodules).
_active_user_modules = []
# Module names already emitted in this buildmodule call (63-char uniqueness).
_emitted_cb_module_names = set()


def set_active_user_modules(um):
    """Register python-module ``__user__`` blocks for the current buildmodule call."""
    global _active_user_modules, _emitted_cb_module_names
    _active_user_modules = list(um or [])
    if not um:
        _emitted_cb_module_names = set()


def _iter_routine_blocks(body):
    """Yield function/subroutine blocks nested under *body* (incl. interfaces)."""
    for b in body or []:
        btype = b.get('block')
        if btype in ('function', 'subroutine') and b.get('name'):
            yield b
        elif btype in ('interface', 'abstract interface'):
            yield from _iter_routine_blocks(b.get('body'))


def _user_module_catalog():
    """All known ``__user__`` modules for this build (globals + active um list)."""
    from . import crackfortran
    seen = set()
    out = []
    for um in list(crackfortran.usermodules) + list(_active_user_modules):
        name = um.get('name')
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(um)
    return out


def _remote_name_in_module(use_spec, local_name):
    """Remote procedure name for *local_name* in one module's use-spec.

    Fortran ``use m, only: f => fun`` is stored as
    ``{'only': 1, 'map': {local: remote}}`` (``f => fun``).

    Returns ``None`` when *local_name* is not imported from this module
    (``ONLY`` exclusion, or a rename that hides the original name).
    """
    mapping = (use_spec or {}).get('map') or {}
    want = local_name.lower()
    for local, remote in mapping.items():
        if (local or '').lower() == want:
            return (remote or local).lower()
    # ONLY list: only map keys are in scope under their local names.
    if (use_spec or {}).get('only'):
        return None
    # Bare USE with renames: ``use m, g => f`` hides remote ``f`` under
    # the original name; only local ``g`` resolves to remote ``f``.
    for local, remote in mapping.items():
        rem = (remote or local or '').lower()
        loc = (local or remote or '').lower()
        if rem == want and loc != want:
            return None
    return want


def _callback_routine_blocks(rout):
    """Map lowercased *local* callback dummy name -> cracked routine block.

    Prefer definitions still on ``rout['body']``. Otherwise resolve only
    through modules listed in *this* routine's ``use`` association (never a
    global first-wins table across unrelated hosts).  ``only``/rename maps
    are applied per used module (gh-20157).
    """
    found = {}
    for b in _iter_routine_blocks(rout.get('body')):
        found[b['name'].lower()] = b

    use = rout.get('use') or {}
    if not use:
        return found

    catalog = {um.get('name'): um for um in _user_module_catalog()}
    vars_ = rout.get('vars') or {}
    candidates = list(rout.get('args') or [])
    for e in rout.get('externals') or []:
        if e not in candidates:
            candidates.append(e)

    for a in candidates:
        if a.lower() in found:
            continue
        if a in vars_ and not isexternal(vars_[a]):
            continue
        if a not in vars_ and a not in (rout.get('externals') or []):
            continue
        for mname, spec in use.items():
            um = catalog.get(mname)
            if um is None:
                continue
            routines = {
                b['name'].lower(): b
                for b in _iter_routine_blocks(um.get('body'))
            }
            remote = _remote_name_in_module(spec, a)
            if remote is not None and remote in routines:
                found[a.lower()] = routines[remote]
                break
    return found


def _safe_ident(raw, prefix, taken, max_len=63):
    """Build a Fortran-safe identifier that does not collide with *taken*."""
    base = ''.join(c if (c.isalnum() or c == '_') else '_' for c in raw)
    if not base or base[0].isdigit():
        base = 'r_' + base
    name = f'{prefix}{base}'
    if len(name) > max_len:
        name = name[:max_len]
    if name.lower() not in taken:
        taken.add(name.lower())
        return name
    n = 2
    while True:
        suffix = f'_{n}'
        cand = (name[: max_len - len(suffix)] + suffix)
        if cand.lower() not in taken:
            taken.add(cand.lower())
            return cand
        n += 1


def _cb_iface_module_name(rout, taken=None):
    """Fortran module name holding abstract interfaces for this routine's callbacks."""
    global _emitted_cb_module_names
    if taken is None:
        taken = set()
    taken = set(taken) | set(_emitted_cb_module_names)
    raw = getfortranname(rout)
    # Digest first so 63-char truncation cannot drop uniqueness when two
    # long fortrannames share a long common prefix.
    digest = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]
    name = _safe_ident(f'{digest}_{raw}', 'f2py_cb_ifaces_', taken)
    _emitted_cb_module_names.add(name.lower())
    return name


def _abstract_iface_name(cbname_lower, taken=None):
    """Name of the abstract interface body for callback *cbname_lower*.

    Must differ from the dummy argument name so ``use`` of the module does
    not make the dummy name ambiguous (gfortran: "ambiguous reference").
    """
    if taken is None:
        taken = set()
    return _safe_ident(cbname_lower, 'f2py_ai_', taken)


# Fortran keywords / intrinsics that must never be treated as use-associated
# symbols when harvesting identifiers from expressions or interface text.
_FORTRAN_NON_USE_NAMES = frozenset({
    'kind', 'len', 'selected_int_kind', 'selected_real_kind',
    'selected_char_kind', 'precision', 'digits', 'epsilon', 'huge', 'tiny',
    'range', 'radix', 'minexponent', 'maxexponent', 'spacing', 'rrspacing',
    'nearest', 'scale', 'set_exponent', 'fraction', 'exponent',
    'real', 'integer', 'character', 'logical', 'complex', 'double',
    'precision', 'and', 'or', 'not', 'eq', 'ne', 'lt', 'le', 'gt', 'ge',
    'true', 'false', 'size', 'shape', 'lbound', 'ubound', 'present',
    'associated', 'allocated', 'len_trim', 'trim', 'adjustl', 'adjustr',
    'index', 'scan', 'verify', 'repeat', 'new_line', 'ishft', 'ishftc',
    'iand', 'ior', 'ieor', 'not', 'ibits', 'ibset', 'ibclr', 'btest',
    'transfer', 'reshape', 'pack', 'unpack', 'spread', 'merge', 'max',
    'min', 'abs', 'mod', 'modulo', 'sign', 'dim', 'floor', 'ceiling',
    'nint', 'int', 'real', 'dble', 'cmplx', 'aimag', 'conjg', 'sqrt',
    'exp', 'log', 'log10', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'atan2', 'sinh', 'cosh', 'tanh', 'sum', 'product', 'maxval', 'minval',
    'count', 'any', 'all', 'matmul', 'dot_product', 'transpose',
    'iso_fortran_env', 'iso_c_binding', 'ieee_arithmetic', 'ieee_exceptions',
    'ieee_features',
})

# f2py-generated shape/hidden helpers: f2py_<arg>_d<n>
_F2PY_SHAPE_HELPER = re.compile(r'^f2py_.+_d\d+$', re.I)


def _is_f2py_shape_helper(name):
    return bool(_F2PY_SHAPE_HELPER.match(name or ''))


def _identifiers_from_fortran_expr(expr):
    """Simple identifier tokens in a Fortran expression string."""
    if not expr or not isinstance(expr, str):
        return set()
    return {
        m.group(0).lower()
        for m in re.finditer(r'[A-Za-z_]\w*', expr)
        if m.group(0).lower() not in _FORTRAN_NON_USE_NAMES
    }


def _kind_identifiers_from_vars(vars_, exclude=None):
    """Identifier tokens needed from USE association for *vars_* decls.

    *exclude* drops host dummies / f2py shape helpers that must not be
    imported from a Fortran module.  User parameters named ``f2py_*``
    (e.g. ``f2py_kind``) are *kept*; only f2py shape helpers are dropped.
    """
    exclude = {e.lower() for e in (exclude or [])}
    needed = set()
    for var in (vars_ or {}).values():
        for key in ('kindselector', 'charselector'):
            sel = (var or {}).get(key) or {}
            for sk in ('kind', 'len', '*'):
                val = sel.get(sk)
                if isinstance(val, str):
                    if re.match(r'^[A-Za-z_]\w*$', val):
                        needed.add(val.lower())
                    else:
                        needed |= _identifiers_from_fortran_expr(val)
        for dim in (var or {}).get('dimension') or []:
            needed |= _identifiers_from_fortran_expr(str(dim))
        tname = (var or {}).get('typename')
        if isinstance(tname, str) and re.match(r'^[A-Za-z_]\w*$', tname):
            needed.add(tname.lower())
    cleaned = set()
    for n in needed:
        if n in exclude or n in _FORTRAN_NON_USE_NAMES or _is_f2py_shape_helper(n):
            continue
        cleaned.add(n)
    return cleaned


def _identifiers_needed_from_interface_text(text, exclude=None):
    """Identifiers referenced as kind=/len=/dimension() in interface text."""
    exclude = {e.lower() for e in (exclude or [])}
    needed = set()
    if not text:
        return needed
    for m in re.finditer(r'kind\s*=\s*([A-Za-z_]\w*)', text, re.I):
        needed.add(m.group(1).lower())
    for m in re.finditer(r'len\s*=\s*([A-Za-z_]\w*)', text, re.I):
        needed.add(m.group(1).lower())
    for m in re.finditer(r'dimension\s*\(([^)]*)\)', text, re.I):
        needed |= _identifiers_from_fortran_expr(m.group(1))
    for m in re.finditer(r'type\s*\(\s*([A-Za-z_]\w*)\s*\)', text, re.I):
        needed.add(m.group(1).lower())
    # Also harvest full kind=expr forms (kind=selected_real_kind(...)).
    for m in re.finditer(r'kind\s*=\s*([^,)\n]+)', text, re.I):
        needed |= _identifiers_from_fortran_expr(m.group(1))
    return {
        n for n in needed
        if n not in exclude
        and n not in _FORTRAN_NON_USE_NAMES
        and not _is_f2py_shape_helper(n)
    }


def _use_local_names_from_block(block):
    """Local names introduced by USE association on *block*."""
    names = set()
    for _m, spec in ((block or {}).get('use') or {}).items():
        mapping = (spec or {}).get('map') or {}
        for loc in mapping:
            if loc:
                names.add(loc.lower())
        # bare use: no explicit locals; callers reserve needed identifiers
    return names


def _filtered_use_dict(block, forbidden_names=None):
    """Return a use-dict safe for an abstract-interface body.

    Drops ``__user__`` modules.  Only *local* use-associated names that
    collide with *forbidden_names* are removed.  Unrestricted USE (bare or
    rename-without-ONLY) is tightened to ``only`` of referenced identifiers
    so module entities matching the abstract procedure name do not enter
    the body.  With multiple bare modules, bare USE is left unrestricted
    only when the procedure name is already reserved away from needed
    identifiers (gh-20157).
    """
    forbidden = {n.lower() for n in (forbidden_names or [])}
    needed = _kind_identifiers_from_vars((block or {}).get('vars'))
    use = (block or {}).get('use') or {}
    bare_mods = [
        m for m, spec in use.items()
        if '__user__' not in m
        and not (spec or {}).get('only')
        and not ((spec or {}).get('map') or {})
    ]
    out = {}
    for mname, spec in use.items():
        if '__user__' in mname:
            continue
        spec = copy.deepcopy(spec) if spec else {}
        mapping = dict(spec.get('map') or {})
        is_only = bool(spec.get('only'))
        if mapping and is_only:
            # ONLY list: drop colliding *local* names only.
            mapping = {
                loc: rem for loc, rem in mapping.items()
                if (loc or '').lower() not in forbidden
            }
            if not mapping:
                continue
            spec['map'] = mapping
            spec['only'] = 1
            out[mname] = spec
            continue
        if mapping and not is_only:
            # Rename without ONLY still imports the rest of the module.
            # Tighten to ONLY: keep renames + needed locals.
            new_map = {
                loc: rem for loc, rem in mapping.items()
                if (loc or '').lower() not in forbidden
            }
            for n in needed:
                if n in forbidden:
                    continue
                if n not in {k.lower() for k in new_map}:
                    new_map[n] = n
            if not new_map:
                continue
            out[mname] = {'only': 1, 'map': new_map}
            continue
        if is_only and not mapping:
            continue
        # Bare ``use m``.
        if len(bare_mods) == 1:
            keep = sorted(n for n in needed if n not in forbidden)
            if not keep:
                # No selector deps, but bare use may still be needed for
                # nothing — drop to avoid name collisions.
                continue
            out[mname] = {'only': 1, 'map': {n: n for n in keep}}
        else:
            # Multiple bare modules: cannot attribute selectors per module.
            # Leave unrestricted; abstract naming must avoid needed ids.
            out[mname] = spec if spec else {}
    return out


def _distinct_result_name(abs_name, args, vars_):
    """Pick a RESULT name different from the procedure and its dummies.

    Always stays within Fortran's 63-character identifier limit.
    """
    taken = {a.lower() for a in (args or [])}
    taken.update(k.lower() for k in (vars_ or {}))
    taken.add(abs_name.lower())
    # Prefer ``<abs>_r`` when it fits; otherwise truncate/suffix via
    # _safe_ident (same 63-char bound as abstract procedure names).
    cand = f'{abs_name}_r'
    if len(cand) <= 63 and cand.lower() not in taken:
        taken.add(cand.lower())
        return cand
    return _safe_ident(f'{abs_name}_r', '', taken, max_len=63)


def _rename_callback_block_for_abstract(block, abs_name):
    """Deep-copy *block* as abstract-interface body named *abs_name*.

    When the result is (or was) the function name, give it a *distinct*
    RESULT variable so crack2fortrangen emits the typed declaration.
    ``result(<same-as-function-name>)`` is invalid Fortran and also causes
    vars2fortran to suppress the type.
    """
    b = copy.deepcopy(block)
    old = b.get('name')
    b['name'] = abs_name
    if not old:
        return b
    vars_ = b.setdefault('vars', {})
    args = b.get('args') or []
    # Explicit result equal to the function name, or implicit result.
    if b.get('result') == old or ('result' not in b and old in vars_):
        res = _distinct_result_name(abs_name, args, vars_)
        b['result'] = res
        if old in vars_:
            vars_[res] = vars_.pop(old)
        elif abs_name in vars_:
            vars_[res] = vars_.pop(abs_name)
        return b
    # Separate result variable: leave result/vars alone; only rename the
    # procedure. Do not move vars[old] — that would strip the return type.
    return b


def _build_callback_iface_module(mod_name, cb_blocks, host_rout=None,
                                 abs_map=None):
    """Emit a Fortran module of abstract interfaces for *cb_blocks*.

    *abs_map* maps local callback lower-name -> abstract interface body name.
    Callers ``use`` the module and declare
    ``procedure(<abs>) :: <local>`` for each callback dummy.
    """
    from .crackfortran import crack2fortrangen
    # Interface bodies are separate scoping units.  Put a *filtered* use
    # dict on the block before crack2fortrangen so the emitter cannot
    # reintroduce forbidden names (gh-20157).
    # *host_rout* is retained for call-site stability only.
    lines = [
        f'module {mod_name}',
        '  implicit none',
        '  abstract interface',
    ]
    abs_map = abs_map or {}
    for key, block in sorted(cb_blocks.items(), key=lambda kv: kv[0]):
        abs_name = abs_map.get(key) or _abstract_iface_name(key)
        b = _rename_callback_block_for_abstract(block, abs_name)
        # Forbidden: callback dummies/result/procedure name must not be
        # use-associated inside the body.
        forbidden = {a.lower() for a in (b.get('args') or [])}
        if b.get('result'):
            forbidden.add(b['result'].lower())
        forbidden.add(abs_name.lower())
        b['use'] = _filtered_use_dict(b, forbidden)
        text = crack2fortrangen(b, tab='\n  ', as_interface=True)
        body_lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        if not body_lines:
            continue
        for ln in body_lines:
            lines.append(f'    {ln}')
    # Abstract interfaces close with END INTERFACE (not END ABSTRACT INTERFACE).
    lines.append('  end interface')
    lines.append(f'end module {mod_name}')
    return '\n'.join(lines)


def _is_interface_start(s):
    s = s.strip().lower()
    return s == 'interface' or (
        s.startswith('interface ') and not s.startswith('end'))


def _is_interface_end(s):
    return s.strip().lower().startswith('end interface')


def _is_routine_header(s):
    s = s.strip().lower()
    return (
        s.startswith('function ') or s.startswith('subroutine ')
        or ' function ' in f' {s}' or ' subroutine ' in f' {s}'
    )


def _parse_use_line_locals(s):
    """Return (module_name, is_bare, local_names) for a USE line.

    *local_names* are the names that enter the current scope (left-hand
    side of renames).  *is_bare* is True for unrestricted ``use m``.
    """
    s = s.strip()
    low = s.lower()
    if not low.startswith('use '):
        return None, False, []
    rest = s[4:].strip()
    # strip trailing comments
    rest = rest.split('!')[0].strip().rstrip(',')
    if not rest:
        return None, False, []
    # module name is first token (possibly followed by comma)
    if ',' in rest:
        mname, after = rest.split(',', 1)
        mname = mname.strip()
        after = after.strip()
    else:
        mname, after = rest, ''
    if not after:
        return mname, True, []
    after_l = after.lower()
    # ONLY keyword only — do not strip names like only_thing.
    if re.match(r'^only\s*:', after_l):
        after = re.sub(r'^only\s*:', '', after, count=1, flags=re.I).strip()
    elif re.match(r'^only\s*$', after_l):
        after = ''
    # parse local => remote or local tokens
    locals_ = []
    for part in after.split(','):
        part = part.strip()
        if not part:
            continue
        if '=>' in part:
            loc = part.split('=>', 1)[0].strip()
        else:
            loc = part
        if loc:
            locals_.append(loc)
    return mname, False, locals_


def _use_line_imports_names(s, names):
    """True if a USE line associates any *local* name in *names* into scope.

    Remote names in ``local => remote`` renames do not count.  Bare
    ``use m`` is treated as potentially importing *names* (caller should
    rewrite bare USE rather than keep it next to procedure(f2py_ai_*)).
    """
    s = s.strip().lower()
    if not s.startswith('use ') or not names:
        return False
    mname, is_bare, locals_ = _parse_use_line_locals(s)
    if mname is None:
        return False
    if is_bare:
        # Unrestricted import: any forbidden public name may enter.
        return True
    locals_l = {n.lower() for n in locals_}
    return bool(locals_l & {n.lower() for n in names})


def _sanitize_use_line(line, forbidden_locals, needed_ids=None,
                       force_only=False):
    """Rewrite one USE line so *forbidden_locals* are not use-associated.

    Returns None to drop the line, or a USE statement string to keep.
    Bare ``use m`` becomes ``use m, only: <needed>`` when *needed_ids*
    is provided.  Rename-without-ONLY is tightened to ONLY of renames
    plus needed identifiers (*force_only* / non-only detection).
    """
    if '__user__' in line.lower():
        return None
    mname, is_bare, locals_ = _parse_use_line_locals(line)
    if mname is None:
        return None
    forbidden = {n.lower() for n in (forbidden_locals or [])}
    needed = {n.lower() for n in (needed_ids or [])}
    low = line.strip().lower()
    # ONLY is a keyword after the first comma, not a substring of a name
    # (``use m, only_thing => x`` must not count as ONLY).
    after = low.split(',', 1)[1].strip() if ',' in low else ''
    has_only = (
        after == 'only'
        or after.startswith('only:')
        or after.startswith('only ')
    )
    if is_bare:
        keep = sorted(n for n in needed if n not in forbidden)
        if not keep:
            return None
        return f'use {mname}, only: {", ".join(keep)}'
    # Parse original parts for renames
    parts = []
    raw = line.split(',', 1)[-1] if ',' in line else ''
    raw = raw.strip()
    # Strip ONLY keyword only (not names that merely start with "only").
    if re.match(r'^only\s*:', raw, flags=re.I):
        raw = re.sub(r'^only\s*:', '', raw, count=1, flags=re.I).strip()
    elif re.match(r'^only\s*$', raw, flags=re.I):
        raw = ''
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        loc = part.split('=>', 1)[0].strip()
        if loc.lower() in forbidden:
            continue
        parts.append(part)
    if not has_only or force_only:
        # Rename without ONLY: still imports other public names. Tighten.
        seen_loc = {
            p.split('=>', 1)[0].strip().lower() for p in parts
        }
        for n in sorted(needed):
            if n in forbidden or n in seen_loc:
                continue
            parts.append(n)
            seen_loc.add(n)
        if not parts:
            return None
        return f'use {mname}, only: {", ".join(parts)}'
    if not parts:
        return None
    if len(parts) == len(locals_):
        return line.strip()
    return f'use {mname}, only: {", ".join(parts)}'


def _host_scope_use_lines(saved_interface, abs_map=None, host_vars=None,
                          exclude_ids=None):
    """Collect USE lines at host scope only (not inside nested interfaces).

    Single bare ``use m`` is rewritten to ``use m, only: <needed ids>``.
    Multiple bare modules are left unrestricted (cannot attribute names
    per module); abstract naming reserves needed identifiers separately.
    """
    abs_names = {v.lower() for v in (abs_map or {}).values()}
    exclude = set(exclude_ids or [])
    needed = _kind_identifiers_from_vars(host_vars, exclude=exclude)
    needed |= _identifiers_needed_from_interface_text(
        saved_interface, exclude=exclude | abs_names)
    # Count bare host-scope USE modules first.
    bare_count = 0
    depth = 0
    saw_header = False
    for line in (saved_interface or '').split('\n'):
        s = line.strip().lower()
        if not saw_header:
            if _is_routine_header(s):
                saw_header = True
            continue
        if _is_interface_start(s):
            depth += 1
            continue
        if _is_interface_end(s):
            depth = max(0, depth - 1)
            continue
        if depth != 0 or not s.startswith('use '):
            continue
        if '__user__' in s:
            continue
        _m, is_bare, _locs = _parse_use_line_locals(line)
        if is_bare:
            bare_count += 1

    out = []
    saw_header = False
    depth = 0
    for line in (saved_interface or '').split('\n'):
        s = line.strip().lower()
        if not saw_header:
            if _is_routine_header(s):
                saw_header = True
            continue
        if _is_interface_start(s):
            depth += 1
            continue
        if _is_interface_end(s):
            depth = max(0, depth - 1)
            continue
        if depth != 0 or not s.startswith('use '):
            continue
        if '__user__' in s:
            continue
        _m, is_bare, _locs = _parse_use_line_locals(line)
        if is_bare and bare_count > 1:
            # Leave unrestricted; procedure names are chosen not to clash
            # with needed kind identifiers.
            out.append(line if line.startswith(' ') else f'          {line.strip()}')
            continue
        fixed = _sanitize_use_line(line, abs_names, needed_ids=needed)
        if fixed:
            out.append(
                '          ' + fixed if not fixed.startswith(' ') else fixed)
    return out


def _resolve_saved_interface_kinds(saved_interface, vars_):
    """Replace symbolic ``kind=name`` with resolved kinds from *vars_*.

    ``saved_interface`` is snapshotted in crackfortran *before* postcrack
    evaluates host-local parameters (e.g. ``integer, parameter :: dp =
    kind(1.0d0)``).  After postcrack, ``vars_`` holds numeric kinds for
    arguments, but the frozen interface text still has ``kind=dp`` and is
    uncompilable in a nested interface (no host-local parameter in scope).
    """
    if not saved_interface or not vars_:
        return saved_interface
    lines = saved_interface.split('\n')
    for name, var in vars_.items():
        ks = (var or {}).get('kindselector') or {}
        kind = ks.get('kind')
        if kind is None:
            continue
        kind_s = str(kind).strip()
        # Only rewrite symbolic kind=name → numeric kind.  Leave expression
        # kinds (selected_real_kind(...)) and unresolved names alone.
        if not re.match(r'^\d+$', kind_s):
            continue
        name_l = name.lower()
        fixed = []
        for line in lines:
            if '::' not in line:
                fixed.append(line)
                continue
            # Match whole declaration names after :: (not prefixes: x vs xx).
            rhs = line.split('::', 1)[1]
            decl_names = []
            for part in rhs.split(','):
                tok = part.split('=')[0].strip()
                tok = tok.split('(')[0].strip()
                if tok:
                    decl_names.append(tok.lower())
            if name_l not in decl_names:
                fixed.append(line)
                continue
            # Replace only a bare identifier kind=name, not kind=expr(...).
            if re.search(r'kind\s*=\s*[A-Za-z_]\w*(?!\s*\()', line, re.I):
                line = re.sub(
                    r'kind\s*=\s*[A-Za-z_]\w*(?!\s*\()',
                    f'kind={kind_s}',
                    line,
                    flags=re.I,
                )
            fixed.append(line)
        lines = fixed
    return '\n'.join(lines)


def _ensure_callback_result_typespec(block, host_var=None):
    """Ensure a function callback block has a typed result (for ``implicit none``).

    When crackfortran cannot type the result, the abstract body would be
    untyped under ``implicit none`` (rejected by strict compilers).  Fall
    back to the host dummy's typespec when available.
    """
    if (block or {}).get('block') != 'function':
        return block
    b = block
    vars_ = b.setdefault('vars', {})
    res = b.get('result') or b.get('name')
    if not res:
        return b
    rv = vars_.get(res) or {}
    if rv.get('typespec'):
        return b
    # Prefer host external declaration (often has the result type).
    host = host_var or {}
    if host.get('typespec'):
        typed = dict(host)
        # Do not copy EXTERNAL / intent attrs onto the result.
        if 'attrspec' in typed:
            typed = {
                k: v for k, v in typed.items()
                if k not in ('attrspec', 'intent', 'check', 'depend')
            }
        vars_[res] = typed
        return b
    # Last resort: integer (matches f2py's common default external typing).
    vars_[res] = {'typespec': 'integer'}
    return b


def _rewrite_saved_interface_use_module(saved_interface, cb_orig_names,
                                        mod_name, abs_map, host_vars=None,
                                        exclude_ids=None):
    """Adapt *saved_interface* for module-based callback interfaces.

    * Drop nested interface blocks that define a callback dummy.
    * Drop bare ``external`` for those dummies and ``__user__`` USE.
    * At host scope only: emit ``use <mod_name>``, then host USE, then
      ``procedure(f2py_ai_*)`` before other host specification statements.
    * Nested (non-callback) interface bodies keep their own USE in place —
      interface bodies are separate scoping units; hoisting USE to the host
      breaks kinds inside surviving nested interfaces (gh-20157).
    """
    cb_set = {n.lower() for n in cb_orig_names}
    abs_names = {v.lower() for v in (abs_map or {}).values()}
    exclude = set(exclude_ids or [])
    needed = _kind_identifiers_from_vars(host_vars, exclude=exclude)
    needed |= _identifiers_needed_from_interface_text(
        saved_interface, exclude=exclude | abs_names)
    orig_by_lower = {}
    for n in cb_orig_names:
        orig_by_lower.setdefault(n.lower(), n)

    lines = saved_interface.split('\n')
    drop = [False] * len(lines)
    i = 0
    while i < len(lines):
        s = lines[i].strip().lower()
        if _is_interface_start(s):
            start = i
            depth = 1
            j = i + 1
            chunk = [lines[i]]
            while j < len(lines) and depth:
                sj = lines[j].strip().lower()
                if _is_interface_start(sj):
                    depth += 1
                elif _is_interface_end(sj):
                    depth -= 1
                chunk.append(lines[j])
                j += 1
            block_l = '\n'.join(chunk).lower()
            if any(
                f'function {cb}(' in block_l
                or f'subroutine {cb}(' in block_l
                or f'function {cb} ' in block_l
                or f'subroutine {cb} ' in block_l
                for cb in cb_set
            ):
                for k in range(start, j):
                    drop[k] = True
            i = j
            continue
        i += 1

    # Pre-count bare host-scope USE modules (multi-module → leave bare).
    bare_host_count = 0
    depth_c = 0
    saw_hdr = False
    for idx, ln in enumerate(lines):
        if drop[idx]:
            continue
        sl = ln.strip().lower()
        if not saw_hdr:
            if _is_routine_header(sl):
                saw_hdr = True
            continue
        if _is_interface_start(sl):
            depth_c += 1
            continue
        if _is_interface_end(sl):
            depth_c = max(0, depth_c - 1)
            continue
        if depth_c != 0 or not sl.startswith('use '):
            continue
        if '__user__' in sl:
            continue
        _m, is_bare, _locs = _parse_use_line_locals(ln)
        if is_bare:
            bare_host_count += 1

    header = []
    host_use = []
    body_out = []
    saw_header = False
    i = 0
    while i < len(lines):
        if drop[i]:
            i += 1
            continue
        line = lines[i]
        s = line.strip().lower()

        if not saw_header:
            header.append(line)
            if _is_routine_header(s):
                saw_header = True
            i += 1
            continue

        # Nested interface: keep block intact; leave USE inside the body.
        if _is_interface_start(s):
            depth = 1
            body_out.append(line)
            i += 1
            while i < len(lines) and depth:
                if drop[i]:
                    i += 1
                    continue
                nl = lines[i]
                ns = nl.strip().lower()
                if _is_interface_start(ns):
                    depth += 1
                elif _is_interface_end(ns):
                    depth -= 1
                if ns.startswith('use '):
                    # Nested interfaces are separate scoping units: do *not*
                    # filter against the host callback's abstract names.
                    if '__user__' in ns:
                        i += 1
                        continue
                    body_out.append(nl)
                    i += 1
                    continue
                body_out.append(nl)
                i += 1
            continue

        if s.startswith('external'):
            rest = s[len('external'):].lstrip(' :')
            names = [n.strip() for n in rest.split(',') if n.strip()]
            if names and all(n in cb_set for n in names):
                i += 1
                continue
            body_out.append(line)
            i += 1
            continue

        if s.startswith('use '):
            _m, is_bare, _locs = _parse_use_line_locals(line)
            if is_bare and bare_host_count > 1:
                host_use.append(
                    line if line.startswith(' ') else f'          {line.strip()}')
                i += 1
                continue
            fixed = _sanitize_use_line(line, abs_names, needed_ids=needed)
            if fixed is not None:
                host_use.append('          ' + fixed)
            i += 1
            continue

        body_out.append(line)
        i += 1

    out = list(header)
    out.append(f'          use {mod_name}')
    out.extend(host_use)
    for low in sorted(orig_by_lower):
        out.append(
            f'          procedure({abs_map[low]}) :: {orig_by_lower[low]}'
        )
    out.extend(body_out)
    return '\n'.join(out)


def _prepare_callback_module(rout, args, vars, need_interface):
    """Return (module_src, mod_name, cb_orig_names, abs_map).

    *abs_map* maps local dummy lower-name -> abstract interface body name.
    """
    empty = ('', None, [], {})
    if not need_interface:
        return empty
    all_blocks = _callback_routine_blocks(rout)
    needed = {}
    orig_names = []
    for a in args:
        if not isexternal(vars[a]):
            continue
        block = all_blocks.get(a.lower())
        if block is not None:
            needed[a.lower()] = _ensure_callback_result_typespec(
                copy.deepcopy(block), host_var=vars.get(a))
            orig_names.append(a)
    if not needed:
        return empty
    taken = {a.lower() for a in args}
    taken.update(k.lower() for k in (vars or {}))
    taken.update(e.lower() for e in (rout.get('externals') or []))
    # Reserve identifiers from each callback block so abstract names cannot
    # collide with dummies/results, USE locals, or kind parameters that
    # the body still needs (gh-20157).
    for block in needed.values():
        for a in block.get('args') or []:
            taken.add(a.lower())
        if block.get('result'):
            taken.add(block['result'].lower())
        for k in block.get('vars') or {}:
            taken.add(k.lower())
        if block.get('name'):
            taken.add(block['name'].lower())
        taken |= _use_local_names_from_block(block)
        taken |= _kind_identifiers_from_vars(block.get('vars'))
    # Host-level USE locals / kind ids also reserved (outer wrapper scope).
    taken |= _use_local_names_from_block(rout)
    taken |= _kind_identifiers_from_vars(vars)
    abs_map = {}
    for a in orig_names:
        block = needed[a.lower()]
        use = block.get('use') or {}
        has_bare = any(
            '__user__' not in m
            and not (spec or {}).get('only')
            and not ((spec or {}).get('map') or {})
            for m, spec in use.items()
        )
        raw = a.lower()
        if has_bare:
            # Disambiguate from unrestricted module exports.
            dig = hashlib.sha1(
                f'{getfortranname(rout)}:{a}'.encode('utf-8')
            ).hexdigest()[:6]
            raw = f'{a.lower()}_{dig}'
        abs_map[a.lower()] = _abstract_iface_name(raw, taken)
    mod_name = _cb_iface_module_name(rout, taken)
    return (
        _build_callback_iface_module(
            mod_name, needed, host_rout=rout, abs_map=abs_map),
        mod_name,
        orig_names,
        abs_map,
    )


def _declare_external_args(args, vars, cb_orig_names, abs_map, add):
    """Declare procedure(...) for module-backed callbacks, else EXTERNAL."""
    cb_lower = {n.lower() for n in cb_orig_names}
    dumped = []
    for a in args:
        if not isexternal(vars[a]):
            continue
        if a.lower() in cb_lower:
            add(f'procedure({abs_map[a.lower()]}) :: {a}')
            dumped.append(a)
            continue
        add(f'external {a}')
        dumped.append(a)
    return dumped


def createfuncwrapper(rout, signature=0):
    assert isfunction(rout)

    extra_args = []
    vars = rout['vars']
    for a in rout['args']:
        v = rout['vars'][a]
        for i, d in enumerate(v.get('dimension', [])):
            if d == ':':
                dn = f'f2py_{a}_d{i}'
                dv = {'typespec': 'integer', 'intent': ['hide']}
                dv['='] = f'shape({a}, {i})'
                extra_args.append(dn)
                vars[dn] = dv
                v['dimension'][i] = dn
    rout['args'].extend(extra_args)
    need_interface = bool(extra_args)

    ret = ['']

    def add(line, ret=ret):
        ret[0] = f'{ret[0]}\n      {line}'
    name = rout['name']
    fortranname = getfortranname(rout)
    f90mode = ismoduleroutine(rout)
    newname = f'{name}f2pywrap'

    if newname not in vars:
        vars[newname] = vars[name]
        args = [newname] + rout['args'][1:]
    else:
        args = [newname] + rout['args']

    l_tmpl = var2fixfortran(vars, name, '@@@NAME@@@', f90mode)
    if l_tmpl[:13] == 'character*(*)':
        if f90mode:
            l_tmpl = 'character(len=10)' + l_tmpl[13:]
        else:
            l_tmpl = 'character*10' + l_tmpl[13:]
        charselect = vars[name]['charselector']
        if charselect.get('*', '') == '(*)':
            charselect['*'] = '10'

    l1 = l_tmpl.replace('@@@NAME@@@', newname)
    rl = None

    useisoc = useiso_c_binding(rout)
    sargs = ', '.join(args)
    if f90mode:
        # gh-23598 fix warning
        # Essentially, this gets called again with modules where the name of the
        # function is added to the arguments, which is not required, and removed
        sargs = sargs.replace(f"{name}, ", '')
        args = [arg for arg in args if arg != name]
        rout['args'] = args
        add(f"subroutine f2pywrap_{rout['modulename']}_{name} ({sargs})")
        if not signature:
            add(f"use {rout['modulename']}, only : {fortranname}")
        if useisoc:
            add('use iso_c_binding')
    else:
        add(f'subroutine f2pywrap{name} ({sargs})')
        if useisoc:
            add('use iso_c_binding')
        if not need_interface:
            add(f'external {fortranname}')
            rl = l_tmpl.replace('@@@NAME@@@', '') + ' ' + fortranname

    args = args[1:]
    # Callback interface module (gh-20157): one definition, USE'd in the
    # wrapper and in the nested host interface.
    # f90mode (module CONTAINS): no nested host interface, skip module path.
    module_src, cb_mod_name, cb_via_module, abs_map = _prepare_callback_module(
        rout, args, vars, need_interface and not f90mode)

    exclude_ids = {a.lower() for a in args}
    exclude_ids.update(e.lower() for e in (rout.get('externals') or []))
    if need_interface:
        saved0 = _resolve_saved_interface_kinds(
            rout.get('saved_interface') or '', vars)
        for line in _host_scope_use_lines(
                saved0, abs_map, host_vars=vars, exclude_ids=exclude_ids):
            add(line)
        if cb_mod_name:
            add(f'use {cb_mod_name}')

    dumped_args = _declare_external_args(
        args, vars, cb_via_module, abs_map, add)
    for a in args:
        if a in dumped_args:
            continue
        if isscalar(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        if isintent_in(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        add(var2fixfortran(vars, a, f90mode=f90mode))

    add(l1)
    if rl is not None:
        add(rl)

    if need_interface:
        if f90mode:
            # Module CONTAINS path: use modulename only; no dual EXTERNAL+
            # interface conflict to fix (gh-20157 applies to free routines).
            pass
        else:
            saved = _resolve_saved_interface_kinds(
                rout.get('saved_interface') or '', vars)
            if cb_mod_name:
                saved = _rewrite_saved_interface_use_module(
                    saved, cb_via_module, cb_mod_name, abs_map,
                    host_vars=vars, exclude_ids=exclude_ids)
            add('interface')
            add(saved.lstrip())
            add('end interface')

    sargs = ', '.join([a for a in args if a not in extra_args])

    if not signature:
        if islogicalfunction(rout):
            add(f'{newname} = .not.(.not.{fortranname}({sargs}))')
        else:
            add(f'{newname} = {fortranname}({sargs})')
    if f90mode:
        add(f"end subroutine f2pywrap_{rout['modulename']}_{name}")
    else:
        add('end')
    return module_src, ret[0]


def createsubrwrapper(rout, signature=0):
    assert issubroutine(rout)

    extra_args = []
    vars = rout['vars']
    for a in rout['args']:
        v = rout['vars'][a]
        for i, d in enumerate(v.get('dimension', [])):
            if d == ':':
                dn = f'f2py_{a}_d{i}'
                dv = {'typespec': 'integer', 'intent': ['hide']}
                dv['='] = f'shape({a}, {i})'
                extra_args.append(dn)
                vars[dn] = dv
                v['dimension'][i] = dn
    rout['args'].extend(extra_args)
    need_interface = bool(extra_args)

    ret = ['']

    def add(line, ret=ret):
        ret[0] = f'{ret[0]}\n      {line}'
    name = rout['name']
    fortranname = getfortranname(rout)
    f90mode = ismoduleroutine(rout)

    args = rout['args']

    useisoc = useiso_c_binding(rout)
    sargs = ', '.join(args)
    if f90mode:
        add(f"subroutine f2pywrap_{rout['modulename']}_{name} ({sargs})")
        if useisoc:
            add('use iso_c_binding')
        if not signature:
            add(f"use {rout['modulename']}, only : {fortranname}")
    else:
        add(f'subroutine f2pywrap{name} ({sargs})')
        if useisoc:
            add('use iso_c_binding')
        if not need_interface:
            add(f'external {fortranname}')

    module_src, cb_mod_name, cb_via_module, abs_map = _prepare_callback_module(
        rout, args, vars, need_interface and not f90mode)

    exclude_ids = {a.lower() for a in args}
    exclude_ids.update(e.lower() for e in (rout.get('externals') or []))
    if need_interface:
        saved0 = _resolve_saved_interface_kinds(
            rout.get('saved_interface') or '', vars)
        for line in _host_scope_use_lines(
                saved0, abs_map, host_vars=vars, exclude_ids=exclude_ids):
            add(line)
        if cb_mod_name:
            add(f'use {cb_mod_name}')

    dumped_args = _declare_external_args(
        args, vars, cb_via_module, abs_map, add)
    for a in args:
        if a in dumped_args:
            continue
        if isscalar(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        add(var2fixfortran(vars, a, f90mode=f90mode))

    if need_interface:
        if f90mode:
            # Module CONTAINS path: no dual EXTERNAL+interface conflict.
            pass
        else:
            saved = _resolve_saved_interface_kinds(
                rout.get('saved_interface') or '', vars)
            if cb_mod_name:
                saved = _rewrite_saved_interface_use_module(
                    saved, cb_via_module, cb_mod_name, abs_map,
                    host_vars=vars, exclude_ids=exclude_ids)
            add('interface')
            for line in saved.split('\n'):
                if line.lstrip().startswith('use ') and '__user__' in line:
                    continue
                add(line)
            add('end interface')

    sargs = ', '.join([a for a in args if a not in extra_args])

    if not signature:
        add(f'call {fortranname}({sargs})')
    if f90mode:
        add(f"end subroutine f2pywrap_{rout['modulename']}_{name}")
    else:
        add('end')
    return module_src, ret[0]




def assubr(rout):
    if isfunction_wrap(rout):
        fortranname = getfortranname(rout)
        name = rout['name']
        outmess('\t\tCreating wrapper for Fortran function '
                f'"{name}"("{fortranname}")...\n')
        rout = copy.copy(rout)
        fname = name
        rname = fname
        if 'result' in rout:
            rname = rout['result']
            rout['vars'][fname] = rout['vars'][rname]
        fvar = rout['vars'][fname]
        if not isintent_out(fvar):
            if 'intent' not in fvar:
                fvar['intent'] = []
            fvar['intent'].append('out')
            flag = 1
            for i in fvar['intent']:
                if i.startswith('out='):
                    flag = 0
                    break
            if flag:
                fvar['intent'].append(f'out={rname}')
        rout['args'][:] = [fname] + rout['args']
        return rout, createfuncwrapper(rout)
    if issubroutine_wrap(rout):
        fortranname = getfortranname(rout)
        name = rout['name']
        outmess('\t\tCreating wrapper for Fortran subroutine '
                f'"{name}"("{fortranname}")...\n')
        rout = copy.copy(rout)
        return rout, createsubrwrapper(rout)
    return rout, ('', '')
