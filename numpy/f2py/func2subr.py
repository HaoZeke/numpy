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


def _abstract_iface_name(cbname_lower, rout, taken=None):
    """Stable, unique abstract-interface body name for *cbname_lower*.

    Always includes a short digest of (fortranname, callback) so the
    generated name cannot collide with user module parameters or USE
    association (e.g. a parameter also named ``f2py_ai_cb``).  That removes
    the need to rewrite or filter host/callback USE statements.
    """
    if taken is None:
        taken = set()
    digest = hashlib.sha1(
        f'{getfortranname(rout)}:{cbname_lower}'.encode('utf-8')
    ).hexdigest()[:8]
    return _safe_ident(digest, 'f2py_ai_', taken)


def _distinct_result_name(abs_name, args, vars_):
    """RESULT name different from the procedure, within 63 characters."""
    taken = {a.lower() for a in (args or [])}
    taken.update(k.lower() for k in (vars_ or {}))
    taken.add(abs_name.lower())
    cand = f'{abs_name}_r'
    if len(cand) <= 63 and cand.lower() not in taken:
        taken.add(cand.lower())
        return cand
    return _safe_ident(f'{abs_name}_r', '', taken, max_len=63)


def _rename_callback_block_for_abstract(block, abs_name):
    """Deep-copy *block* as abstract-interface body named *abs_name*.

    When the result is (or was) the function name, give it a *distinct*
    RESULT variable so crack2fortrangen emits the typed declaration.
    """
    b = copy.deepcopy(block)
    old = b.get('name')
    b['name'] = abs_name
    if not old:
        return b
    vars_ = b.setdefault('vars', {})
    args = b.get('args') or []
    if b.get('result') == old or ('result' not in b and old in vars_):
        res = _distinct_result_name(abs_name, args, vars_)
        b['result'] = res
        if old in vars_:
            vars_[res] = vars_.pop(old)
        elif abs_name in vars_:
            vars_[res] = vars_.pop(abs_name)
        return b
    return b


def _drop_user_use(block):
    """Drop f2py ``__user__`` modules from a cracked block's use dict."""
    use = (block or {}).get('use') or {}
    cleaned = {m: s for m, s in use.items() if '__user__' not in m}
    block = block
    block['use'] = cleaned
    return block


def _build_callback_iface_module(mod_name, cb_blocks, host_rout=None,
                                 abs_map=None):
    """Emit a Fortran module of abstract interfaces for *cb_blocks*.

    USE association on each callback is left intact (except ``__user__``).
    Collision avoidance is handled by digest-unique abstract names, not by
    filtering USE lists.
    """
    from .crackfortran import crack2fortrangen
    lines = [
        f'module {mod_name}',
        '  implicit none',
        '  abstract interface',
    ]
    abs_map = abs_map or {}
    for key, block in sorted(cb_blocks.items(), key=lambda kv: kv[0]):
        abs_name = abs_map.get(key)
        if not abs_name:
            # host_rout should always be provided; fall back to key digest
            abs_name = _abstract_iface_name(
                key, host_rout if host_rout is not None else {'name': key}, set())

        b = _rename_callback_block_for_abstract(block, abs_name)
        _drop_user_use(b)
        text = crack2fortrangen(b, tab='\n  ', as_interface=True)
        for ln in text.split('\n'):
            s = ln.strip()
            if s:
                lines.append(f'    {s}')
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


def _host_scope_use_lines(saved_interface):
    """Host-scope USE lines (not nested), dropping only ``__user__`` modules."""
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
        out.append(line if line.startswith(' ') else f'          {line.strip()}')
    return out


def _resolve_saved_interface_kinds(saved_interface, vars_):
    """Replace symbolic ``kind=name`` with numeric kinds from *vars_*.

    ``saved_interface`` is snapshotted before postcrack evaluates host-local
    parameters; after postcrack, argument kinds are numeric but the frozen
    text may still say ``kind=dp``.  Only rewrite bare identifier kinds to
    digit kinds; leave expression kinds alone.
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
        if not re.match(r'^\d+$', kind_s):
            continue
        name_l = name.lower()
        fixed = []
        for line in lines:
            if '::' not in line:
                fixed.append(line)
                continue
            rhs = line.split('::', 1)[1]
            decl_names = []
            for part in rhs.split(','):
                tok = part.split('=')[0].strip().split('(')[0].strip()
                if tok:
                    decl_names.append(tok.lower())
            if name_l not in decl_names:
                fixed.append(line)
                continue
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
    """Ensure a function callback block has a typed result under implicit none."""
    if (block or {}).get('block') != 'function':
        return block
    vars_ = block.setdefault('vars', {})
    res = block.get('result') or block.get('name')
    if not res:
        return block
    if (vars_.get(res) or {}).get('typespec'):
        return block
    host = host_var or {}
    if host.get('typespec'):
        typed = {
            k: v for k, v in host.items()
            if k not in ('attrspec', 'intent', 'check', 'depend')
        }
        vars_[res] = typed
        return block
    vars_[res] = {'typespec': 'integer'}
    return block


def _rewrite_saved_interface_use_module(saved_interface, cb_orig_names,
                                        mod_name, abs_map):
    """Adapt *saved_interface* for module-based callback interfaces.

    Minimal transform (no USE filtering — abstract names are digest-unique):
    * drop nested interfaces that define a callback dummy
    * drop bare ``external`` for those dummies and ``__user__`` USE
    * host-scope USE is hoisted before ``procedure(...)`` (Fortran order)
    * nested-interface USE stays inside those nested bodies
    """
    cb_set = {n.lower() for n in cb_orig_names}
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
    depth = 0
    for idx, line in enumerate(lines):
        if drop[idx]:
            continue
        s = line.strip().lower()
        if not saw_header:
            header.append(line)
            if _is_routine_header(s):
                saw_header = True
            continue
        if _is_interface_start(s):
            depth += 1
            body_out.append(line)
            continue
        if _is_interface_end(s):
            depth = max(0, depth - 1)
            body_out.append(line)
            continue
        if s.startswith('external'):
            rest = s[len('external'):].lstrip(' :')
            names = [n.strip() for n in rest.split(',') if n.strip()]
            if names and all(n in cb_set for n in names):
                continue
            body_out.append(line)
            continue
        if s.startswith('use '):
            if '__user__' in s:
                continue
            if depth == 0:
                host_use.append(line)
            else:
                body_out.append(line)
            continue
        body_out.append(line)

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
    """Return (module_src, mod_name, cb_orig_names, abs_map)."""
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
        a.lower(): _abstract_iface_name(a.lower(), rout, taken)
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
        for line in _host_scope_use_lines(saved0):
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
                    saved, cb_via_module, cb_mod_name, abs_map)
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
        for line in _host_scope_use_lines(saved0):
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
                    saved, cb_via_module, cb_mod_name, abs_map)
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
