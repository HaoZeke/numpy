"""
Build F90 module support for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
__version__ = "$Revision: 1.27 $"[10:-1]

f2py_version = 'See `f2py -v`'

import numpy as np

from . import capi_maps, func2subr

# The environment provided by auxfuncs.py is needed for some calls to eval.
# As the needed functions cannot be determined by static inspection of the
# code, it is safest to use import * pending a major refactoring of f2py.
from .auxfuncs import *
from .crackfortran import undo_rmbadname, undo_rmbadname1

options = {}


def findf90modules(m):
    if ismodule(m):
        return [m]
    if not hasbody(m):
        return []
    ret = []
    for b in m['body']:
        if ismodule(b):
            ret.append(b)
        else:
            ret = ret + findf90modules(b)
    return ret


fgetdims1 = f"""\
      external f2pysetdata
      logical ns
      integer r,i
      integer({np.intp().itemsize}) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then"""

fgetdims2 = """\
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))"""

fgetdims2_sa = """\
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
         !s(r) must be equal to len(d(1))
      end if
      flag = 2
      call f2pysetdata(d,allocated(d))"""


def _saved_parameter_decl(copy_name, var):
    """Fortran declaration of a SAVE copy of a module PARAMETER.

    Flang materializes PARAMETER values as stack temporaries and the init
    wrapper stores that address (gh-22511). A saved local outlives the call.
    """
    typespec = var.get('typespec') or 'integer'
    if typespec == 'type' and var.get('typename'):
        typespec = f"type({var['typename']})"
    selector = ''
    kindselector = var.get('kindselector') or {}
    charselector = var.get('charselector') or {}
    if '*' in kindselector:
        selector = f"*{kindselector['*']}"
    elif 'kind' in kindselector:
        selector = f"(kind={kindselector['kind']})"
    elif '*' in charselector:
        selector = f"*{charselector['*']}"
    elif 'len' in charselector:
        if 'kind' in charselector:
            selector = (
                f"(len={charselector['len']},kind={charselector['kind']})")
        else:
            selector = f"(len={charselector['len']})"
    dims = ''
    if var.get('dimension'):
        dims = f", dimension({','.join(var['dimension'])})"
    return f"{typespec}{selector}{dims}, save :: {copy_name}"


def _parameter_imports(m, pymod):
    """Public parameters brought in by USE.

    Each item is ``(local_name, used_module, remote_name, var)``.
    A bare ``use other`` imports every public parameter; ``only`` and
    ``local => remote`` follow the Fortran statement (gh-3562).
    """
    use = m.get('use') or {}
    if not use:
        return []
    available = {mod['name']: mod for mod in findf90modules(pymod)}
    found = []
    seen = set(m.get('vars') or {})
    for uname, spec in use.items():
        src = available.get(uname)
        if not src:
            continue
        spec = spec or {}
        mapping = spec.get('map') or {}
        if spec.get('only'):
            pairs = list(mapping.items())
        else:
            pairs = [(name, name) for name in (src.get('vars') or {})]
        for local, remote in pairs:
            if not local or local in seen:
                continue
            var = (src.get('vars') or {}).get(remote)
            if not var:
                continue
            if 'parameter' not in (var.get('attrspec') or []):
                continue
            if isprivate(var):
                continue
            found.append((local, uname, remote, var))
            seen.add(local)
    return found


def buildhooks(pymod):
    from . import rules
    ret = {'f90modhooks': [], 'initf90modhooks': [], 'body': [],
           'need': ['F_FUNC', 'arrayobject.h'],
           'separatorsfor': {'includes0': '\n', 'includes': '\n'},
           'docs': ['"Fortran 90/95 modules:\\n"'],
           'latexdoc': []}
    fhooks = ['']

    def fadd(line, s=fhooks):
        s[0] = f'{s[0]}\n      {line}'
    doc = ['']

    def dadd(line, s=doc):
        s[0] = f'{s[0]}\n{line}'

    usenames = getuseblocks(pymod)
    for m in findf90modules(pymod):
        sargs, fargs, efargs, modobjs, notvars, onlyvars = [], [], [], [], [
            m['name']], []
        param_copies = set()
        sargsp = []
        ifargs = []
        mfargs = []
        if hasbody(m):
            for b in m['body']:
                notvars.append(b['name'])
        for n in m['vars'].keys():
            var = m['vars'][n]

            if (n not in notvars and isvariable(var)) and (not l_or(isintent_hide, isprivate)(var)):
                onlyvars.append(n)
                mfargs.append(n)
        imported = _parameter_imports(m, pymod)
        imported_vars = {}
        for local, uname, remote, var in imported:
            imported_vars[local] = var
            onlyvars.append(local)
        outmess(f"\t\tConstructing F90 module support for \"{m['name']}\"...\n")
        # Must agree with the loops below, which wrap only variables and routines.
        if not onlyvars and not any(isroutine(b) for b in m.get('body', [])):
            outmess(f"\t\t\tSkipping {m['name']} since there are no public vars/func in this module...\n")
            continue

        # gh-25186
        if m['name'] in usenames and containscommon(m):
            outmess(f"\t\t\tSkipping {m['name']} since it is in 'use' and contains a common block...\n")
            continue
        # skip modules with derived types
        if m['name'] in usenames and containsderivedtypes(m):
            outmess(f"\t\t\tSkipping {m['name']} since it is in 'use' and contains a derived type...\n")
            continue
        if onlyvars:
            outmess(f"\t\t  Variables: {' '.join(onlyvars)}\n")
        chooks = ['']

        def cadd(line, s=chooks):
            s[0] = f'{s[0]}\n{line}'
        ihooks = ['']

        def iadd(line, s=ihooks):
            s[0] = f'{s[0]}\n{line}'

        vrd = capi_maps.modsign2map(m)
        cadd(f"static FortranDataDef f2py_{m['name']}_def[] = {{")
        dadd(f"\\subsection{{Fortran 90/95 module \\texttt{{{m['name']}}}}}\n")
        if hasnote(m):
            note = m['note']
            if isinstance(note, list):
                note = '\n'.join(note)
            dadd(note)
        if onlyvars:
            dadd('\\begin{description}')
        for n in onlyvars:
            var = m['vars'][n] if n in m['vars'] else imported_vars[n]
            modobjs.append(n)
            ct = capi_maps.getctype(var)
            at = capi_maps.c2capi_map[ct]
            dm = capi_maps.getarrdims(n, var)
            dms = dm['dims'].replace('*', '-1').strip()
            dms = dms.replace(':', '-1').strip()
            if not dms:
                dms = '-1'
            use_fgetdims2 = fgetdims2
            rank = dm['rank']
            elsize = capi_maps.get_elsize(var)
            cadd(f'\t{{"{undo_rmbadname1(n)}",{rank},{{{{{dms}}}}},{at}, '
                 f'{elsize}}},')
            docsign = capi_maps.getarrdocsign(n, var)
            dadd(f'\\item[]{{{{}}\\verb@{docsign}@{{}}}}')
            if hasnote(var):
                note = var['note']
                if isinstance(note, list):
                    note = '\n'.join(note)
                dadd(f'--- {note}')
            if isallocatable(var):
                fargs.append(f"f2py_{m['name']}_getdims_{n}")
                efargs.append(fargs[-1])
                sargs.append(
                    f'void (*{n})(int*,npy_intp*,void(*)(char*,npy_intp*),int*)')
                sargsp.append('void (*)(int*,npy_intp*,void(*)(char*,npy_intp*),int*)')
                iadd(f"\tf2py_{m['name']}_def[i_f2py++].func = {n};")
                fadd(f'subroutine {fargs[-1]}(r,s,f2pysetdata,flag)')
                fadd(f"use {m['name']}, only: d => {undo_rmbadname1(n)}\n")
                fadd('integer flag\n')
                fhooks[0] = fhooks[0] + fgetdims1
                dms = range(1, int(dm['rank']) + 1)
                alloc_args = ','.join(f's({i})' for i in dms)
                fadd(f' allocate(d({alloc_args}))\n')
                fhooks[0] = fhooks[0] + use_fgetdims2
                fadd(f'end subroutine {fargs[-1]}')
            else:
                fargs.append(n)
                mfargs.append(n)
                sargs.append(f'char *{n}')
                sargsp.append('char*')
                iadd(f"\tf2py_{m['name']}_def[i_f2py++].data = {n};")
                if 'parameter' in var.get('attrspec', []):
                    param_copies.add(n)
        if onlyvars:
            dadd('\\end{description}')
        has_routines = False
        if hasbody(m):
            m_name = m['name']
            for b in m['body']:
                b_name = b['name']
                if not isroutine(b):
                    outmess("f90mod_rules.buildhooks:"
                            f" skipping {b['block']} {b_name}\n")
                    continue
                has_routines = True
                modobjs.append(f"{b_name}()")
                b['modulename'] = m_name
                api, wrap = rules.buildapi(b)
                if isfunction(b):
                    fhooks[0] = fhooks[0] + wrap
                    fargs.append(f"f2pywrap_{m_name}_{b_name}")
                    ifargs.append(func2subr.createfuncwrapper(b, signature=1))
                elif wrap:
                    fhooks[0] = fhooks[0] + wrap
                    fargs.append(f"f2pywrap_{m_name}_{b_name}")
                    ifargs.append(
                        func2subr.createsubrwrapper(b, signature=1))
                else:
                    fargs.append(b_name)
                    mfargs.append(fargs[-1])
                api['externroutines'] = []
                ar = applyrules(api, vrd)
                ar['docs'] = []
                ar['docshort'] = []
                ret = dictappend(ret, ar)
                cadd(f'\t{{"{b_name}",-1,{{{{-1}}}},0,0,NULL,(void *)'
                      f'f2py_rout_#modulename#_{m_name}_{b_name},'
                      f'doc_f2py_rout_#modulename#_{m_name}_{b_name}}},')
                sargs.append(f"char *{b_name}")
                sargsp.append('char *')
                iadd(f"\tf2py_{m_name}_def[i_f2py++].data = {b_name};")
        cadd('\t{NULL}\n};\n')
        iadd('}')
        m_name = m['name']
        sargs_str = ','.join(sargs)
        ihooks[0] = (f'static void f2py_setup_{m_name}({sargs_str}) '
                     f'{{\n\tint i_f2py=0;{ihooks[0]}')
        if '_' in m_name:
            F_FUNC = 'F_FUNC_US'
        else:
            F_FUNC = 'F_FUNC'
        sargsp_str = ','.join(sargsp)
        iadd(f'extern void {F_FUNC}(f2pyinit{m_name},'
             f'F2PYINIT{m_name.upper()})(void (*)({sargsp_str}));')
        iadd(f'static void f2py_init_{m_name}(void) {{')
        iadd(f'\t{F_FUNC}(f2pyinit{m_name},'
             f'F2PYINIT{m_name.upper()})(f2py_setup_{m_name});')
        iadd('}\n')
        ret['f90modhooks'] = ret['f90modhooks'] + chooks + ihooks
        if has_routines:
            data_kind = "F90 module"
        else:
            data_kind = "module variable container"
        ret['initf90modhooks'] = [
            '\t{',
            ('\t\tPyObject *tmp = '
            f'PyFortranObject_New(f2py_{m_name}_def,f2py_init_{m_name});'),
            '\t\tif (tmp == NULL) return -1;',
            ('\t\tif (PyFortranObject_SetDataKind(tmp, '
             f'"{data_kind}", "{m_name}") < 0) {{'),
            '\t\t\tPy_DECREF(tmp);',
            '\t\t\treturn -1;',
            '\t\t}',
            ('\t\tif (PyFortranObject_TagModuleRoutines(tmp, m, '
             f'"{m_name}") < 0) {{'),
            '\t\t\tPy_DECREF(tmp);',
            '\t\t\treturn -1;',
            '\t\t}',
            f'\t\tif (PyModule_AddObjectRef(m, "{m_name}", tmp) < 0) {{',
            '\t\t\tPy_DECREF(tmp);',
            '\t\t\treturn -1;',
            '\t\t}',
            '\t\tPy_DECREF(tmp);',
            '\t}',
        ] + ret["initf90modhooks"]
        fadd('')
        fadd(f"subroutine f2pyinit{m_name}(f2pysetupfunc)")
        if mfargs:
            for a in undo_rmbadname(mfargs):
                fadd(f"use {m_name}, only : {a}")
        for local, uname, remote, _var in imported:
            local_f = undo_rmbadname1(local)
            remote_f = undo_rmbadname1(remote)
            if local_f == remote_f:
                fadd(f"use {uname}, only : {remote_f}")
            else:
                fadd(f"use {uname}, only : {local_f} => {remote_f}")
        if ifargs:
            fadd(' '.join(['interface'] + ifargs))
            fadd('end interface')
        fadd('external f2pysetupfunc')
        if efargs:
            for a in undo_rmbadname(efargs):
                fadd(f'external {a}')
        call_args = []
        copies = []
        for a in fargs:
            fa = undo_rmbadname1(a)
            if a in param_copies:
                cname = f'f2py_copy_{fa}'
                # Imported parameters are not in this module's vars.
                var = m['vars'][a] if a in m['vars'] else imported_vars[a]
                fadd(_saved_parameter_decl(cname, var))
                copies.append((cname, fa))
                call_args.append(cname)
            else:
                call_args.append(fa)
        for cname, fa in copies:
            fadd(f'{cname} = {fa}')
        fadd(f"call f2pysetupfunc({','.join(call_args)})")
        fadd(f"end subroutine f2pyinit{m_name}\n")

        dadd('\n'.join(ret['latexdoc']).replace(
            r'\subsection{', r'\subsubsection{'))

        ret['latexdoc'] = []
        modobjs_str = ','.join(undo_rmbadname(modobjs))
        ret['docs'].append(f"\"\t{m_name} --- {modobjs_str}\"")

    ret['routine_defs'] = ''
    ret['doc'] = []
    ret['docshort'] = []
    ret['latexdoc'] = doc[0]
    if len(ret['docs']) <= 1:
        ret['docs'] = ''
    return ret, fhooks[0]
