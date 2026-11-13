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
    useisof = useiso_fortran_env(rout)
    sargs = ', '.join(args)
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

    if need_interface:
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

    if need_interface:
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

    sargs = ', '.join([a for a in args if a not in extra_args])

    if not signature:
        add(f'call {fortranname}({sargs})')
    if f90mode:
        add(f"end subroutine f2pywrap_{rout['modulename']}_{name}")
    else:
        add('end')
    return ret[0]


def _allocatable_result(rout):
    """Rank-1 allocatable function result, or None."""
    if not isfunction(rout):
        return None
    rname = rout.get('result', rout['name'])
    var = rout['vars'].get(rname, {})
    dims = var.get('dimension') or []
    if 'allocatable' not in var.get('attrspec', []):
        return None
    if len(dims) != 1:
        return None
    return rname, var


def _alloc_typenum(var):
    from .capi_maps import getctype
    ctype = getctype(var)
    return {
        'char': 'NPY_BYTE',
        'signed_char': 'NPY_BYTE',
        'short': 'NPY_SHORT',
        'int': 'NPY_INT',
        'long': 'NPY_LONG',
        'long_long': 'NPY_LONGLONG',
        'float': 'NPY_FLOAT',
        'double': 'NPY_DOUBLE',
        'long_double': 'NPY_LONGDOUBLE',
        'complex_float': 'NPY_CFLOAT',
        'complex_double': 'NPY_CDOUBLE',
        'complex_long_double': 'NPY_CLONGDOUBLE',
    }.get(ctype, 'NPY_INT')


def _alloc_result_wrapper(rout, rname, var):
    """Fortran wrapper that lets the function allocate, then copies out."""
    name = rout['name']
    fortranname = getfortranname(rout)
    args = [a for a in rout.get('args', []) if a not in (rname, name)]
    lines = [f'subroutine f2pywrap{name}(copy, {", ".join(args)})'.replace(
        '(copy, )', '(copy)')]
    lines.append('external copy')
    for arg in args:
        lines.append(var2fixfortran(rout['vars'], arg, f90mode=True))
    ts = var.get('typespec', 'integer')
    kind = (var.get('kindselector') or {}).get('kind')
    if kind:
        ts = f'{ts}(kind={kind})'
    lines.append(f'{ts}, allocatable :: tmp(:)')
    lines.append('interface')
    arglist = ', '.join(args)
    lines.append(f'function {fortranname}({arglist}) result({rname})')
    for arg in args:
        lines.append(var2fixfortran(rout['vars'], arg, f90mode=True))
    lines.append(f'{ts}, allocatable, dimension(:) :: {rname}')
    lines.append(f'end function {fortranname}')
    lines.append('end interface')
    call_args = f'({arglist})' if arglist else '()'
    lines.append(f'tmp = {fortranname}{call_args}')
    lines.append('if (allocated(tmp)) then')
    lines.append('if (size(tmp) > 0) then')
    lines.append('call copy(size(tmp), tmp)')
    lines.append('else')
    lines.append('call copy(0, 0)')
    lines.append('endif')
    lines.append('else')
    lines.append('call copy(0, 0)')
    lines.append('endif')
    lines.append('end')
    return '\n'.join(f'      {line}' for line in lines) + '\n'


def _as_alloc_subroutine(rout, var):
    """Present an allocatable function to the C wrapper as a subroutine.

    The subroutine is the generated f2pywrap* routine. It calls back into
    C with the allocated array instead of requiring a pre-sized buffer.
    """
    from .auxfuncs import getcallprotoargument
    rname = rout.get('result', rout['name'])
    rout['args'] = [a for a in rout.get('args', []) if a not in (rname, rout['name'])]
    rout['block'] = 'subroutine'
    rout.pop('result', None)
    enhancements = dict(rout.get('f2pyenhancements') or {})
    enhancements['fortranname'] = f"f2pywrap{rout['name']}"
    rout['f2pyenhancements'] = enhancements
    proto = getcallprotoargument(rout)
    callback = 'void (*)(const int *, const char *)'
    if proto and proto != 'void':
        enhancements['callprotoargument'] = f'{callback}, {proto}'
    else:
        enhancements['callprotoargument'] = callback
    rout['_alloc_result_typenum'] = _alloc_typenum(var)
    return rout


def assubr(rout):
    alloc = _allocatable_result(rout)
    if alloc and not ismoduleroutine(rout):
        rname, var = alloc
        original = rout
        rout = copy.deepcopy(rout)
        wrap = _alloc_result_wrapper(original, rname, var)
        return _as_alloc_subroutine(rout, var), wrap
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
