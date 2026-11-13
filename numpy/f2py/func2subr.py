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

def _allocatable_result(rout):
    """Name of an allocatable array result, or None."""
    name = rout.get('name')
    var = (rout.get('vars') or {}).get(name) or {}
    if 'allocatable' not in (var.get('attrspec') or []):
        return None
    if not var.get('dimension'):
        return None
    return name


def createfuncwrapper(rout, signature=0):
    assert isfunction(rout)

    extra_args = []
    vars = rout['vars']
    # An allocatable result has no size until the function returns.
    # The wrapper mallocs that storage and hands the pointer back.
    alloc_name = None if 'f2py_alloc_ptr' in vars else _allocatable_result(rout)
    alloc_rank = 0
    if alloc_name:
        result = vars[alloc_name]
        alloc_rank = len(result['dimension'])
        result['dimension'] = ['1'] * alloc_rank
        result['attrspec'] = [a for a in result.get('attrspec') or []
                               if a != 'allocatable']
        vars['f2py_alloc_ptr'] = {
            'typespec': 'integer',
            'kindselector': {'kind': '8'},
            'intent': ['hide'],
        }
        for i in range(alloc_rank):
            vars[f'f2py_alloc_d{i}'] = {
                'typespec': 'integer',
                'intent': ['hide'],
            }
        rout['args'].extend(
            ['f2py_alloc_ptr'] + [f'f2py_alloc_d{i}' for i in range(alloc_rank)])
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
        if useisoc or alloc_rank:
            add('use iso_c_binding')
        if useisof:
            add('use iso_fortran_env')
    else:
        add(f'subroutine f2pywrap{name} ({sargs})')
        if useisoc or alloc_rank:
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

    sargs = ', '.join([a for a in args if a not in extra_args
                        and a != newname and not str(a).startswith('f2py_alloc_')])

    if alloc_rank and not signature:
        kind = (vars[name].get('kindselector') or {}).get('kind')
        typed = vars[name]['typespec']
        if kind:
            typed = f'{typed}(kind={kind})'
        ranks = ','.join(':' for _ in range(alloc_rank))
        add('block')
        add(f'{typed}, allocatable :: f2py_tmp({ranks})')
        add(f'{typed}, pointer :: f2py_dst({ranks})')
        add('type(c_ptr) :: f2py_cptr')
        add('integer(c_size_t) :: f2py_bytes')
        add('interface')
        add('function f2py_malloc(n) bind(c, name="malloc")')
        add('import :: c_ptr, c_size_t')
        add('integer(c_size_t), value :: n')
        add('type(c_ptr) :: f2py_malloc')
        add('end function f2py_malloc')
        add('end interface')
        add(f'f2py_tmp = {fortranname}({sargs})')
        for i in range(alloc_rank):
            add(f'f2py_alloc_d{i} = size(f2py_tmp, {i + 1})')
        add('f2py_bytes = int(size(f2py_tmp), c_size_t) * int(storage_size(f2py_tmp) / 8, c_size_t)')
        add('if (f2py_bytes > 0) then')
        add('f2py_cptr = f2py_malloc(f2py_bytes)')
        add('call c_f_pointer(f2py_cptr, f2py_dst, shape(f2py_tmp))')
        add('f2py_dst = f2py_tmp')
        add('f2py_alloc_ptr = transfer(f2py_cptr, f2py_alloc_ptr)')
        add('else')
        add('f2py_alloc_ptr = 0')
        add('end if')
        add('end block')
        dims = ' '.join(f'dims[{i}] = f2py_alloc_d{i};' for i in range(alloc_rank))
        rout.setdefault('f2pyenhancements', {})
        rout['f2pyenhancements']['callstatement'] = (
            '(*f2py_func)(#callfortran#);\n'
            '{\n'
            f'    npy_intp dims[{alloc_rank}];\n'
            '    npy_intp f2py_i, f2py_nb;\n'
            '    int f2py_type;\n'
            '    char *f2py_dst, *f2py_src;\n'
            f'    {dims}\n'
            f'    f2py_type = PyArray_TYPE(capi_{newname}_as_array);\n'
            f'    f2py_nb = PyArray_ITEMSIZE(capi_{newname}_as_array);\n'
            f'    for (f2py_i = 0; f2py_i < {alloc_rank}; ++f2py_i) f2py_nb *= dims[f2py_i];\n'
            f'    Py_DECREF(capi_{newname}_as_array);\n'
            f'    capi_{newname}_as_array = (PyArrayObject *)PyArray_New(&PyArray_Type, {alloc_rank}, dims, f2py_type, NULL, NULL, 0, NPY_ARRAY_F_CONTIGUOUS, NULL);\n'
            '    if (f2py_alloc_ptr && f2py_nb > 0) {\n'
            f'        f2py_dst = (char *)PyArray_DATA(capi_{newname}_as_array);\n'
            '        f2py_src = (char *)f2py_alloc_ptr;\n'
            '        for (f2py_i = 0; f2py_i < f2py_nb; ++f2py_i) f2py_dst[f2py_i] = f2py_src[f2py_i];\n'
            '    }\n'
            '    free((void *)f2py_alloc_ptr);\n'
            '}'
        )
    elif not signature:
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
