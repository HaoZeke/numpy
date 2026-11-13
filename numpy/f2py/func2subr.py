"""

Rules for building C/API module with f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import copy

from ._isocbind import isoc_kindmap
from ._isofenv import isof_kindmap
from .auxfuncs import (
    getcallprotoargument,
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

def useiso_fortran_env(rout):
    useisof = False
    for value in rout['vars'].values():
        kind_value = value.get('kindselector', {}).get('kind')
        if kind_value in isof_kindmap:
            return True
    return useisof

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
    useisof = useiso_fortran_env(rout)
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
        if useisof:
            add('use iso_fortran_env')
    else:
        add(f'subroutine f2pywrap{name} ({sargs})')
        if useisoc:
            add('use iso_c_binding')
        if useisof:
            add('use iso_fortran_env')
        if not need_interface:
            add(f'external {fortranname}')
            rl = l_tmpl.replace('@@@NAME@@@', '') + ' ' + fortranname

    if need_interface:
        for line in rout['saved_interface'].split('\n'):
            if line.lstrip().startswith('use ') and '__user__' not in line:
                add(line)

    args = args[1:]
    dumped_args = []
    for a in args:
        if isexternal(vars[a]):
            add(f'external {a}')
            dumped_args.append(a)
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
            # f90 module already defines needed interface
            pass
        else:
            add('interface')
            add(rout['saved_interface'].lstrip())
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
    return ret[0]


_DEFERRED = {}


def deferred_state(rout):
    args = rout.get('args') if isinstance(rout, dict) else None
    if args is None:
        return {}
    return _DEFERRED.get(id(args), {})


def _deferred_alloc_out(var):
    """Rank-1 allocatable intent(out) with deferred shape."""
    if not isinstance(var, dict):
        return False
    if 'allocatable' not in (var.get('attrspec') or []):
        return False
    intent = [str(item).lower() for item in (var.get('intent') or [])]
    if 'in' in intent or 'inout' in intent or 'out' not in intent:
        return False
    dims = [str(dim).strip() for dim in (var.get('dimension') or [])]
    return dims == [':']


def _deferred_local_decl(item):
    var = item['var']
    name = item['local']
    if item['char']:
        return f'character(len=:), allocatable, target :: {name}(:)'
    typespec = var.get('typespec') or 'integer'
    kind = (var.get('kindselector') or {}).get('kind')
    if kind:
        typespec = f'{typespec}(kind={kind})'
    return f'{typespec}, allocatable, target :: {name}(:)'


def _prepare_deferred(rout):
    """Drop allocatable deferred-shape outputs from the C argument list.

    The Fortran wrapper allocates them and copies the result back through
    a callback. An explicit-shape stand-in is not a legal actual argument
    for an allocatable dummy. The bookkeeping lives on the shared args
    list so the interface pass sees the same signature.
    """
    state = _DEFERRED.get(id(rout['args']))
    if state is not None:
        return state
    kept = []
    allocs = []
    orig = list(rout['args'])
    for name in orig:
        var = rout['vars'].get(name) or {}
        if _deferred_alloc_out(var) and len(allocs) < 4:
            allocs.append({
                'name': name,
                'local': f'f2py_alloc_{name}',
                'char': (var.get('typespec') or '').lower() == 'character',
                'var': var,
                'slot': len(allocs),
                'copy': f'f2py_copy_{len(allocs)}',
            })
        else:
            kept.append(name)
    rout['args'][:] = kept
    extra_args = []
    vars = rout['vars']
    for name in list(rout['args']):
        var = vars[name]
        for i, dim in enumerate(var.get('dimension', [])):
            if dim == ':':
                hidden = f'f2py_{name}_d{i}'
                vars[hidden] = {
                    'typespec': 'integer',
                    'intent': ['hide'],
                    '=': f'shape({name}, {i})',
                }
                var['dimension'][i] = hidden
                extra_args.append(hidden)
    rout['args'].extend(extra_args)
    state = {'allocs': allocs, 'orig': orig, 'extra': extra_args}
    _DEFERRED[id(rout['args'])] = state
    if allocs:
        enh = rout.setdefault('f2pyenhancements', {})
        enh.pop('callprotoargument', None)
        rest = getcallprotoargument(rout) or 'void'
        pointers = []
        for item in allocs:
            if item['char']:
                pointers.append('void (*)(const int *, const int *, const void *)')
            else:
                pointers.append('void (*)(const int *, const void *)')
        parts = pointers + ([] if rest == 'void' else [rest])
        enh['callprotoargument'] = ', '.join(parts)
    return state


def createsubrwrapper(rout, signature=0):
    assert issubroutine(rout)

    state = _prepare_deferred(rout)
    allocs = state['allocs']
    extra_args = state['extra']
    need_interface = bool(extra_args) or bool(allocs)

    ret = ['']

    def add(line, ret=ret):
        ret[0] = f'{ret[0]}\n      {line}'
    name = rout['name']
    fortranname = getfortranname(rout)
    f90mode = ismoduleroutine(rout)

    args = rout['args']
    vars = rout['vars']
    copy_names = [item['copy'] for item in allocs]

    useisoc = useiso_c_binding(rout) or bool(allocs)
    useisof = useiso_fortran_env(rout)
    for item in allocs:
        kind = (item['var'].get('kindselector') or {}).get('kind')
        if kind in isof_kindmap:
            useisof = True
    sargs = ', '.join(copy_names + list(args))
    if f90mode:
        add(f"subroutine f2pywrap_{rout['modulename']}_{name} ({sargs})")
        if useisoc:
            add('use iso_c_binding')
        if useisof:
            add('use iso_fortran_env')
        if not signature:
            add(f"use {rout['modulename']}, only : {fortranname}")
    else:
        add(f'subroutine f2pywrap{name} ({sargs})')
        if useisoc:
            add('use iso_c_binding')
        if useisof:
            add('use iso_fortran_env')
        if not need_interface:
            add(f'external {fortranname}')

    if allocs:
        add('abstract interface')
        add('subroutine f2py_deferred_num(n, src) bind(c)')
        add('use iso_c_binding')
        add('integer(c_int), intent(in) :: n')
        add('type(c_ptr), value :: src')
        add('end subroutine f2py_deferred_num')
        add('subroutine f2py_deferred_str(n, slen, src) bind(c)')
        add('use iso_c_binding')
        add('integer(c_int), intent(in) :: n, slen')
        add('type(c_ptr), value :: src')
        add('end subroutine f2py_deferred_str')
        add('end interface')
        for item in allocs:
            iface = 'f2py_deferred_str' if item['char'] else 'f2py_deferred_num'
            add(f"procedure({iface}) :: {item['copy']}")

    if need_interface and not allocs:
        for line in rout['saved_interface'].split('\n'):
            if line.lstrip().startswith('use ') and '__user__' not in line:
                add(line)

    dumped_args = []
    for a in args:
        if isexternal(vars[a]):
            add(f'external {a}')
            dumped_args.append(a)
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

    if not signature:
        for item in allocs:
            add(_deferred_local_decl(item))

    if need_interface and not allocs:
        if f90mode:
            # f90 module already defines needed interface
            pass
        else:
            add('interface')
            for line in rout['saved_interface'].split('\n'):
                if line.lstrip().startswith('use ') and '__user__' in line:
                    continue
                add(line)
            add('end interface')

    local_name = {item['name']: item['local'] for item in allocs}
    call_args = ', '.join(local_name.get(name, name) for name in state['orig'])

    if not signature:
        add(f'call {fortranname}({call_args})')
        for item in allocs:
            local = item['local']
            copy = item['copy']
            add(f'if (allocated({local})) then')
            add(f'if (size({local}) > 0) then')
            if item['char']:
                add(f'call {copy}(size({local}), len({local}), c_loc({local}))')
            else:
                add(f'call {copy}(size({local}), c_loc({local}))')
            add('else')
            if item['char']:
                add(f'call {copy}(0, 0, c_null_ptr)')
            else:
                add(f'call {copy}(0, c_null_ptr)')
            add('endif')
            add('else')
            if item['char']:
                add(f'call {copy}(0, 0, c_null_ptr)')
            else:
                add(f'call {copy}(0, c_null_ptr)')
            add('endif')
    if f90mode:
        add(f"end subroutine f2pywrap_{rout['modulename']}_{name}")
    else:
        add('end')
    return ret[0]


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
    return rout, ''
