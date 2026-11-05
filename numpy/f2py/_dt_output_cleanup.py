"""Release unadopted PDT routine outputs when native status reports failure."""

from .auxfuncs import isarray, isintent_hide, isintent_out


def failed_pdt_outputs(rout, mappings):
    """Destroy only exclusive hidden outputs before Python object adoption."""
    outputs = [(name, var, mappings[name])
               for name, var in rout['vars'].items()
               if name in mappings and var.get('f2py_dt', {}).get('pdt_handle')
               and isintent_hide(var) and isintent_out(var)]
    if not outputs:
        return ''
    prefix = 'f2py_pdt_discard'
    while any(value.get('varname', name).startswith(prefix)
              for name, value in mappings.items()):
        prefix += 'x'
    lines = ['  {', '    extern void f2py_pdt_destroy(void *);',
             f'    PyObject *{prefix}_error = PyErr_GetRaisedException();']

    def destroy(pointer, indentation):
        return [indentation + f'if ({pointer} != NULL) {{',
                indentation + f'    f2py_pdt_destroy({pointer});',
                indentation + '    if (PyErr_Occurred()) '
                'PyErr_WriteUnraisable(Py_None);',
                indentation + '}']

    for name, var, mapping in outputs:
        actual = mapping.get('varname', name)
        dt = var['f2py_dt']
        lines += [f'    if ({actual} != NULL) {{']
        if 'vector' in dt:
            lines += [f'        void **{prefix}_handles = (void **){actual};',
                      f'        npy_intp {prefix}_count = 1;',
                      f'        {actual} = NULL;']
            for axis in range(1, dt['vector'] + 1):
                extent = f'f2py_{name}_n{axis}'
                extent = mappings[extent].get('varname', extent)
                lines.append(f'        {prefix}_count *= (npy_intp){extent};')
            lines += [f'        for (npy_intp {prefix}_index = 0;',
                      (f'             {prefix}_index < {prefix}_count; '
                       f'++{prefix}_index) {{'),
                      (f'            void *{prefix}_owned = '
                       f'{prefix}_handles[{prefix}_index];'),
                      f'            {prefix}_handles[{prefix}_index] = NULL;',
                      *destroy(prefix + '_owned', '            '), '        }',
                      (f'        {mapping["dtvectorfree"]}({prefix}_handles, '
                       f'(int){prefix}_count);'),
                      '        if (PyErr_Occurred()) PyErr_WriteUnraisable(Py_None);']
        elif isarray(var):
            lines += [(f'        npy_intp {prefix}_count = '
                       f'PyArray_SIZE(capi_{actual}_as_array);'),
                      f'        for (npy_intp {prefix}_index = 0;',
                      (f'             {prefix}_index < {prefix}_count; '
                       f'++{prefix}_index) {{'),
                      f'            void *{prefix}_owned = {actual}[{prefix}_index];',
                      f'            {actual}[{prefix}_index] = NULL;',
                      *destroy(prefix + '_owned', '            '), '        }']
        else:
            lines += [f'        void *{prefix}_owned = {actual};',
                      f'        {actual} = NULL;',
                      *destroy(prefix + '_owned', '        ')]
        lines += ['    }']
    lines += [f'    PyErr_SetRaisedException({prefix}_error);', '  }']
    return '\n'.join(lines) + '\n'
