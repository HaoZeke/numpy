"""

Rules for building C/API module with f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import copy
import re

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

# C types that are passed as a single scalar and can be named in an
# extern prototype without a generated typedef.
_CHARLEN_CTYPES = {
    'char': 'char',
    'signed_char': 'signed char',
    'short': 'short',
    'int': 'int',
    'long': 'long',
    'long_long': 'long long',
    'unsigned_char': 'unsigned char',
    'unsigned_short': 'unsigned short',
    'unsigned': 'unsigned',
    'unsigned_long': 'unsigned long',
    'unsigned_long_long': 'unsigned long long',
    'float': 'float',
    'double': 'double',
    'long_double': 'long double',
}


def _result_len_expr(var):
    sel = var.get('charselector') or {}
    if 'len' in sel:
        return str(sel['len']).strip()
    if '*' in sel:
        return str(sel['*']).strip()
    return ''


def _len_expr_is_call(expr):
    """True when the length calls a Fortran function other than len or size.

    ``len(x)`` and ``size(x)`` are rewritten to C. ``nlen(i)`` is not.
    """
    names = re.findall(r'([A-Za-z_]\w*)\s*\(', expr or '')
    allowed = {'len', 'size', 'kind'}
    return any(name.lower() not in allowed for name in names)


def _charlen_helper(rout, vars, result):
    """Build a Fortran integer function that returns len(result).

    ``character(len=nlen(i))`` is a specification expression. Emitting
    ``nlen(i)`` in the C wrapper calls a function that does not exist
    in C. The helper evaluates ``len`` of the Fortran result instead.
    """
    if rout.get('_charlen_helper') == '':
        return None
    if rout.get('_charlen_helper'):
        return rout['_charlen_helper']
    var = vars.get(result) or {}
    expr = _result_len_expr(var)
    if not _len_expr_is_call(expr):
        rout['_charlen_helper'] = ''
        return None
    from .capi_maps import getctype
    args = []
    ctypes = []
    for arg in list(rout.get('args') or []):
        if arg in (result, rout.get('name'), rout.get('result')):
            continue
        avar = vars.get(arg) or {}
        if not isscalar(avar):
            rout['_charlen_helper'] = ''
            return None
        intent = avar.get('intent') or []
        if 'optional' in intent or 'hide' in intent:
            rout['_charlen_helper'] = ''
            return None
        if 'value' in (avar.get('attrspec') or []):
            rout['_charlen_helper'] = ''
            return None
        ctype = _CHARLEN_CTYPES.get(getctype(avar))
        if ctype is None:
            rout['_charlen_helper'] = ''
            return None
        args.append(arg)
        ctypes.append(ctype)
    mod = rout.get('modulename') or 'proc'
    helper = f"f2pycharlen_{mod}_{rout['name']}".lower()
    macro = 'F_FUNC_US' if '_' in helper else 'F_FUNC'
    cargs = ', '.join(f'{ctype} *{arg}' for ctype, arg in zip(ctypes, args))
    ccall = ', '.join(f'&{arg}' for arg in args)
    extern = (f'extern int {macro}({helper},{helper.upper()})({cargs});')
    sel = dict(var.get('charselector') or {})
    sel.pop('len', None)
    sel['*'] = '*'
    sel['f2py_len'] = f'{macro}({helper},{helper.upper()})({ccall})'
    var['charselector'] = sel
    info = {
        'name': helper,
        'args': args,
        'extern': extern,
        'fortran': None,
    }
    rout['_charlen_helper'] = info
    rout['f2py_charlen_extern'] = extern
    return info


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

    # A length such as nlen(i) is not a C expression. character(len=*)
    # takes the hidden length f2py already passes, and a helper asks
    # Fortran for that length before the buffer is allocated.
    charlen = _charlen_helper(rout, vars, name)
    if charlen and charlen.get('fortran') is None and not signature:
        call_args = ', '.join(charlen['args'])
        decls = '\n'.join(
            f'      {var2fixfortran(vars, arg, f90mode=f90mode)}'
            for arg in charlen['args'])
        uses = []
        if f90mode:
            uses.append(f"use {rout['modulename']}, only : {fortranname}")
        if useiso_c_binding(rout):
            uses.append('use iso_c_binding')
        if useiso_fortran_env(rout):
            uses.append('use iso_fortran_env')
        use_block = '\n'.join(f'      {line}' for line in uses)
        body = [f"      integer function {charlen['name']}({call_args})"]
        if use_block:
            body.append(use_block)
        if decls:
            body.append(decls)
        if not f90mode:
            body.append(f'      external {fortranname}')
        body.append(f"      {charlen['name']} = len({fortranname}({call_args}))")
        body.append('      end')
        charlen['fortran'] = '\n'.join(body)

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
    if charlen and charlen.get('fortran') and not signature:
        ret[0] = f"{ret[0]}\n\n{charlen['fortran']}\n"
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
