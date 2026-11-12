"""
ISO_C_BINDING maps for f2py2e.
Only required declarations/macros/functions will be used.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
# These map to keys in c2py_map
iso_c_binding_map = {
    'integer': {
        'c_int': 'int',
        'c_short': 'short',  # 'short' <=> 'int' for now
        'c_long': 'long',  # 'long' <=> 'int' for now
        'c_long_long': 'long_long',
        'c_signed_char': 'signed_char',
        'c_size_t': 'size_t',
        'c_int8_t': 'signed_char',  # int8_t <=> 'signed_char' for now
        'c_int16_t': 'short',  # int16_t <=> 'short' for now
        'c_int32_t': 'int',  # int32_t <=> 'int' for now
        'c_int64_t': 'long_long',
        'c_int_least8_t': 'signed_char',  # int_least8_t <=> 'signed_char' for now
        'c_int_least16_t': 'short',  # int_least16_t <=> 'short' for now
        'c_int_least32_t': 'int',  # int_least32_t <=> 'int' for now
        'c_int_least64_t': 'long_long',
        'c_int_fast8_t': 'int_fast8_t',
        'c_int_fast16_t': 'int_fast16_t',
        'c_int_fast32_t': 'int_fast32_t',
        'c_int_fast64_t': 'int_fast64_t',
        'c_intmax_t': 'intmax_t',
        'c_intptr_t': 'intptr_t',
        'c_ptrdiff_t': 'ptrdiff_t',
    },
    'real': {
        'c_float': 'float',
        'c_double': 'double',
        'c_long_double': 'long_double'
    },
    'complex': {
        'c_float_complex': 'complex_float',
        'c_double_complex': 'complex_double',
        'c_long_double_complex': 'complex_long_double',
        # C_FLOAT_COMPLEX has the value of C_FLOAT, etc. (F2018 18.2.2)
        'c_float': 'complex_float',
        'c_double': 'complex_double',
        'c_long_double': 'complex_long_double',
    },
    'logical': {
        'c_bool': 'bool'
    },
    'character': {
        'c_char': 'char'
    }
}

# C types of F2018 Table 18.2 whose width the C implementation chooses
# (C11 7.19, 7.20.1), so the wrapper names them instead of assuming a
# width.  Values are the signedness and the printf conversion (C11 7.8.1,
# 7.21.6.1); conversions are spliced into C string literals, so the
# <inttypes.h> macros close and reopen the literal.
isoc_int_types = {
    'size_t': (False, 'zu'),
    'ptrdiff_t': (True, 'td'),
    'intmax_t': (True, 'jd'),
    'intptr_t': (True, '" PRIdPTR "'),
    'int_fast8_t': (True, '" PRIdFAST8 "'),
    'int_fast16_t': (True, '" PRIdFAST16 "'),
    'int_fast32_t': (True, '" PRIdFAST32 "'),
    'int_fast64_t': (True, '" PRIdFAST64 "'),
}

isoc_c2pycode_map = {ctype: ctype for ctype in isoc_int_types}
isoc_c2pycode_map['bool'] = '?'
iso_c2py_map = dict.fromkeys(isoc_int_types, 'int')
iso_c2py_map['bool'] = 'int'

isoc_kindmap = {}
for fortran_type, c_type_dict in iso_c_binding_map.items():
    for c_type in c_type_dict.keys():
        isoc_kindmap[c_type] = fortran_type
