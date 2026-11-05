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

from . import capi_maps, derived_type_rules, func2subr

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


def _parameter_storage_module_names(block):
    """Reserve parsed and USE-associated program unit names."""
    names = set()

    def collect(node):
        if node.get('name'):
            names.add(node['name'].lower())
        names.update(name.lower() for name in node.get('use', {}))
        for child in node.get('body', []):
            collect(child)

    collect(block)
    return names


def _module_parameter_storage(module, names, occupied_modules):
    """Give module constants persistent storage with their native types."""
    parameters = [n for n in names if isparameter(module['vars'][n])]
    if not parameters:
        return '', {}, '', ''

    module_index = 0
    storage_module = 'f2py_parameter_storage_0'
    while storage_module.lower() in occupied_modules:
        module_index += 1
        storage_module = f'f2py_parameter_storage_{module_index}'
    occupied_modules.add(storage_module.lower())

    occupied = {n.lower() for n in module['vars']}
    occupied.update(b['name'].lower() for b in module.get('body', []))
    occupied.add('f2pysetupfunc')

    def local_name(stem):
        name = stem
        suffix = 0
        while name.lower() in occupied:
            suffix += 1
            name = f'{stem}_{suffix}'
        occupied.add(name.lower())
        return name

    init_name = local_name('f2py_initialize_parameters')
    aliases = {n: local_name(f'f2py_parameter_{i}')
               for i, n in enumerate(parameters)}
    lines = [f'module {storage_module}']
    declarations = []
    assignments = []
    for i, n in enumerate(parameters):
        var = module['vars'][n]
        original = f'f2py_original_{i}'
        value = f'f2py_value_{i}'
        lines.extend([f'use {module["name"]}, only: &',
                      f'& {original} => {undo_rmbadname1(n)}'])
        typespec = var['typespec'].lower()
        if typespec in {'double precision', 'double complex', 'byte'}:
            typespec = {'double precision': 'real',
                        'double complex': 'complex',
                        'byte': 'integer'}[typespec]
        if typespec not in {'integer', 'real', 'complex', 'logical', 'character'}:
            raise ValueError(f'Unsupported module PARAMETER type: {typespec}')
        declaration = f'{typespec}(kind=kind({original})'
        if typespec == 'character':
            declaration += f',len=max(1,len({original}))'
        declaration += '), target, save :: &'
        if isarray(var):
            declarations.extend([declaration,
                                 f'& {value}(max(1,size({original})))'])
            assignments.extend([
                f'if (size({original}) > 0) then',
                f'  {value}(1:size({original})) = &',
                f'  & reshape({original}, [size({original})])',
                'end if',
            ])
        else:
            declarations.extend([declaration, f'& {value}'])
            assignments.append(f'{value} = {original}')
    lines.extend(['implicit none', 'private'])
    lines.append('public :: f2py_initialize')
    lines.extend(f'public :: f2py_value_{i}' for i in range(len(parameters)))
    lines.extend(declarations)
    lines.extend(['contains', 'subroutine f2py_initialize()'])
    lines.extend(assignments)
    lines.extend(['end subroutine f2py_initialize',
                  f'end module {storage_module}', ''])
    imports = [f'use {storage_module}, only: &',
               f'& {init_name} => f2py_initialize']
    for i, n in enumerate(parameters):
        imports.extend([f'use {storage_module}, only: &',
                        f'& {aliases[n]} => f2py_value_{i}'])
    return '\n'.join(lines), aliases, '\n'.join(imports), init_name


def buildhooks(pymod):
    from . import rules
    ret = {'f90modhooks': [], 'f90modtypes': [], 'initf90modhooks': [],
           'body': [],
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
    dt_types = derived_type_rules.describe_types(pymod)
    occupied_modules = _parameter_storage_module_names(pymod)
    for m in findf90modules(pymod):
        sargs, fargs, efargs, modobjs, notvars, onlyvars = [], [], [], [], [
            m['name']], []
        sargsp = []
        ifargs = []
        mfargs = []
        if hasbody(m):
            for b in m['body']:
                notvars.append(b['name'])
        for n in m['vars'].keys():
            var = m['vars'][n]

            if (n not in notvars and isvariable(var)) and (not l_or(isintent_hide, isprivate)(var)):
                if capi_maps.getctype(var) == 'void':
                    outmess(f"\t\t\tSkipping {n} since f2py has no "
                            "C type for the module variable...\n")
                    continue
                onlyvars.append(n)
                mfargs.append(n)
        outmess(f"\t\tConstructing F90 module support for \"{m['name']}\"...\n")
        from ._dt_imports import _has_imported_type_exports
        if (len(onlyvars) == 0 and len(notvars) == 1 and m['name'] in notvars
                and not _has_imported_type_exports(m)):
            outmess(f"\t\t\tSkipping {m['name']} since there are no public vars/func in this module...\n")
            continue

        # gh-25186
        if m['name'] in usenames and containscommon(m):
            outmess(f"\t\t\tSkipping {m['name']} since it is in 'use' and contains a common block...\n")
            continue
        # skip modules with unwrappable derived types
        if m['name'] in usenames and containsderivedtypes(m):
            type_map = derived_type_rules._module_types(m)[0]
            has_unwrappable = any(
                tb['name'].lower() not in type_map
                for tb in derived_type_rules._find_derived_types(m))
            if has_unwrappable:
                outmess(f"\t\t\tSkipping {m['name']} since it is in 'use'"
                        f" and contains an unwrappable derived type...\n")
                continue
        if onlyvars:
            outmess(f"\t\t  Variables: {' '.join(onlyvars)}\n")
        storage, parameter_aliases, storage_use, storage_init = (
            _module_parameter_storage(m, onlyvars, occupied_modules))
        if storage:
            fhooks[0] += '\n' + storage
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
            var = m['vars'][n]
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
                fargs.append(parameter_aliases.get(n, n))
                sargs.append(f'char *{n}')
                sargsp.append('char*')
                iadd(f"\tf2py_{m['name']}_def[i_f2py++].data = {n};")
        if onlyvars:
            dadd('\\end{description}')
        dt_routines = []
        if hasbody(m):
            m_name = m['name']
            finals = derived_type_rules.final_subroutines(m)
            params = {n: v['='] for n, v in m['vars'].items()
                      if isparameter(v) and '=' in v}
            for b in m['body']:
                b_name = b['name']
                if not isroutine(b):
                    if b['block'] != 'type' or (b_name.lower() not in dt_types
                            and (m['name'].lower(), b_name.lower()) not in dt_types):
                        outmess("f90mod_rules.buildhooks:"
                                f" skipping {b['block']} {b_name}\n")
                    continue
                if isprivate(m['vars'].get(b_name, {})):
                    outmess("f90mod_rules.buildhooks: not exporting private"
                            f" procedure {b_name}\n")
                    continue
                if b_name.lower() in finals:
                    outmess("f90mod_rules.buildhooks:"
                            f" skipping {b_name}, which finalizes a"
                            " derived type\n")
                    continue
                b['modulename'] = m_name
                # derived_type_rules passes derived type arguments as
                # handles of their Fortran objects
                if derived_type_rules._has_derived_type_args(b):
                    missing = derived_type_rules.prepare_routine(
                        b, dt_types, params, uses=m.get('use', {}))
                    if missing:
                        outmess("f90mod_rules.buildhooks:"
                                f" skipping {b_name}, as f2py cannot pass"
                                f" {'; '.join(missing)}\n")
                        b['f2py_dt_missing'] = missing
                        continue
                    dt_routines.append(b_name)
                modobjs.append(f"{b_name}()")
                api, wrap = rules.buildapi(b)
                if isfunction(b) and wrap:
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
        # routines with derived type arguments are attributes of the
        # module too, as the derived types are
        aliases = [
            f'\t\tif (PyDict_GetItemString(d, "{n}") == NULL'
            f' && PyDict_SetItemString(d, "{n}",'
            f' PyDict_GetItemString(((PyFortranObject *)tmp)->dict, "{n}"))'
            ' < 0) {Py_DECREF(tmp); return -1;}'
            for n in dt_routines]
        ret['initf90modhooks'] = [
            '\t{',
            ('\t\tPyObject *tmp = '
            f'PyFortranObject_New(f2py_{m_name}_def,f2py_init_{m_name});'),
            f'\t\tPyDict_SetItemString(d, "{m_name}", tmp);',
            *aliases,
            '\t\tPy_XDECREF(tmp);',
            '\t}',
        ] + ret["initf90modhooks"]
        fadd('')
        fadd(f"subroutine f2pyinit{m_name}(f2pysetupfunc)")
        if mfargs:
            for a in undo_rmbadname(mfargs):
                fadd(f"use {m_name}, only : {a}")
        if storage_use:
            fadd(storage_use)
        if ifargs:
            fadd(' '.join(['interface'] + ifargs))
            fadd('end interface')
        fadd('external f2pysetupfunc')
        if efargs:
            for a in undo_rmbadname(efargs):
                fadd(f'external {a}')
        if storage_init:
            fadd(f'call {storage_init}()')
        fadd(f"call f2pysetupfunc({','.join(undo_rmbadname(fargs))})")
        fadd(f"end subroutine f2pyinit{m_name}\n")

        dadd('\n'.join(ret['latexdoc']).replace(
            r'\subsection{', r'\subsubsection{'))

        ret['latexdoc'] = []
        modobjs_str = ','.join(undo_rmbadname(modobjs))
        ret['docs'].append(f"\"\t{m_name} --- {modobjs_str}\"")

    # Merge derived type hooks (opaque pointer types, routine wrappers)
    dt_hooks = derived_type_rules.buildhooks(pymod)
    ret['f90modtypes'].extend(dt_hooks.get('f90modhooks', []))
    ret['initf90modhooks'].extend(dt_hooks.get('initf90modhooks', []))
    ret['need'].extend(dt_hooks.get('need', []))

    # Generate Fortran wrappers for opaque pointer types
    for m in findf90modules(pymod):
        type_blocks = [b for b in m.get('body', [])
                       if b.get('block') == 'type']
        for tb in type_blocks:
            tb['parent_block'] = m
        routines = [b for b in m.get('body', []) if isroutine(b)]
        if type_blocks:
            fwrap = derived_type_rules.generate_fortran_wrappers(
                m['name'], type_blocks, routines=routines,
                module_block=m)
            if fwrap and fwrap.strip():
                fhooks[0] = fhooks[0] + '\n' + fwrap

    ret['routine_defs'] = ''
    ret['doc'] = []
    ret['docshort'] = []
    ret['latexdoc'] = doc[0]
    if len(ret['docs']) <= 1:
        ret['docs'] = ''
    return ret, fhooks[0]
