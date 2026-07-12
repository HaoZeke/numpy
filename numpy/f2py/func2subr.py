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


def _kind_identifiers_from_vars(vars_):
    """Identifier tokens used as kind/len selectors in *vars_*."""
    needed = set()
    for var in (vars_ or {}).values():
        for key in ('kindselector', 'charselector'):
            sel = (var or {}).get(key) or {}
            for sk in ('kind', 'len', '*'):
                val = sel.get(sk)
                if isinstance(val, str) and re.match(r'^[A-Za-z_]\w*$', val):
                    needed.add(val.lower())
    return needed


def _filtered_use_dict(block, forbidden_names=None):
    """Return a use-dict safe for an abstract-interface body.

    Drops ``__user__`` modules.  Only *local* use-associated names that
    collide with *forbidden_names* are removed (remote names in
    ``local => remote`` renames do not enter the local scope).  Bare
    ``use m`` is rewritten to ``use m, only: <needed>`` so unrestricted
    imports cannot pull in the abstract procedure name (gh-20157).
    """
    forbidden = {n.lower() for n in (forbidden_names or [])}
    needed = _kind_identifiers_from_vars((block or {}).get('vars'))
    out = {}
    for mname, spec in ((block or {}).get('use') or {}).items():
        if '__user__' in mname:
            continue
        spec = copy.deepcopy(spec) if spec else {}
        mapping = dict(spec.get('map') or {})
        is_only = bool(spec.get('only'))
        if mapping:
            # Local names only: ``use m, only: dp => f2py_ai_cb`` keeps
            # local ``dp`` even though the remote symbol matches forbidden.
            mapping = {
                loc: rem for loc, rem in mapping.items()
                if (loc or '').lower() not in forbidden
            }
            if not mapping and is_only:
                continue
            if mapping:
                spec['map'] = mapping
                if is_only:
                    spec['only'] = 1
                out[mname] = spec
            continue
        if is_only:
            # only-list with empty map: nothing usable
            continue
        # Bare ``use m``: restrict to kind identifiers actually referenced.
        keep = sorted(n for n in needed if n not in forbidden)
        if not keep:
            continue
        out[mname] = {
            'only': 1,
            'map': {n: n for n in keep},
        }
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
    if after_l.startswith('only'):
        after = after[4:].lstrip()
        if after.startswith(':'):
            after = after[1:].strip()
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


def _sanitize_use_line(line, forbidden_locals, needed_ids=None):
    """Rewrite one USE line so *forbidden_locals* are not use-associated.

    Returns None to drop the line, or a USE statement string to keep.
    Bare ``use m`` becomes ``use m, only: <needed>`` when *needed_ids*
    is provided.
    """
    if '__user__' in line.lower():
        return None
    mname, is_bare, locals_ = _parse_use_line_locals(line)
    if mname is None:
        return None
    forbidden = {n.lower() for n in (forbidden_locals or [])}
    needed = {n.lower() for n in (needed_ids or [])}
    if is_bare:
        keep = sorted(n for n in needed if n not in forbidden)
        if not keep:
            return None
        return f'use {mname}, only: {", ".join(keep)}'
    # only-list / renames: filter by local name only
    parts = []
    raw = line.split(',', 1)[-1] if ',' in line else ''
    raw = raw.strip()
    if raw.lower().startswith('only'):
        raw = raw[4:].lstrip()
        if raw.startswith(':'):
            raw = raw[1:].strip()
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        loc = part.split('=>', 1)[0].strip()
        if loc.lower() in forbidden:
            continue
        parts.append(part)
    if not parts:
        return None
    if len(parts) == len(locals_):
        return line.strip()
    return f'use {mname}, only: {", ".join(parts)}'


def _host_scope_use_lines(saved_interface, abs_map=None, host_vars=None):
    """Collect USE lines at host scope only (not inside nested interfaces).

    Bare ``use m`` is rewritten to ``use m, only: <kind ids>`` so a module
    parameter that collides with an abstract-interface name cannot make
    ``procedure(f2py_ai_*)`` unclassifiable.
    """
    abs_names = {v.lower() for v in (abs_map or {}).values()}
    needed = _kind_identifiers_from_vars(host_vars)
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
        fixed = _sanitize_use_line(line, abs_names, needed_ids=needed)
        if fixed:
            # preserve indentation style used by wrappers
            out.append('          ' + fixed if not fixed.startswith(' ') else fixed)
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
        # Only rewrite when the var has a resolved (non-identifier) kind.
        if re.match(r'^[A-Za-z_]\w*$', kind_s):
            continue
        name_l = name.lower()
        fixed = []
        for line in lines:
            compact = line.replace(' ', '').lower()
            declares = (
                f'::{name_l}' in compact
                or compact.endswith(f'::{name_l}')
                or re.search(
                    rf'::\s*{re.escape(name)}\b', line, flags=re.I)
            )
            if declares and re.search(r'kind\s*=\s*[A-Za-z_]\w*', line, re.I):
                line = re.sub(
                    r'kind\s*=\s*[A-Za-z_]\w*',
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
                                        mod_name, abs_map, host_vars=None):
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
    needed = _kind_identifiers_from_vars(host_vars)
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

        # Nested interface: keep block intact; leave USE inside the body
        # (sanitize so local collisions with abstract names are dropped).
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
                    fixed = _sanitize_use_line(
                        nl, abs_names, needed_ids=needed)
                    if fixed is None:
                        i += 1
                        continue
                    # keep original indentation
                    indent = nl[:len(nl) - len(nl.lstrip())]
                    body_out.append(indent + fixed)
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
    # Also reserve identifiers from each callback block so abstract names
    # cannot collide with callback dummies/results (gh-20157).
    for block in needed.values():
        for a in block.get('args') or []:
            taken.add(a.lower())
        if block.get('result'):
            taken.add(block['result'].lower())
        for k in block.get('vars') or {}:
            taken.add(k.lower())
        if block.get('name'):
            taken.add(block['name'].lower())
    abs_map = {
        a.lower(): _abstract_iface_name(a.lower(), taken)
        for a in orig_names
    }
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

    if need_interface:
        saved0 = _resolve_saved_interface_kinds(
            rout.get('saved_interface') or '', vars)
        for line in _host_scope_use_lines(
                saved0, abs_map, host_vars=vars):
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
                    host_vars=vars)
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

    if need_interface:
        saved0 = _resolve_saved_interface_kinds(
            rout.get('saved_interface') or '', vars)
        for line in _host_scope_use_lines(
                saved0, abs_map, host_vars=vars):
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
                    host_vars=vars)
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
