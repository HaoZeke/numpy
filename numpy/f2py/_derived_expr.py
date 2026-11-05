"""Fortran expression values for derived type declarations.

Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

import ast
import collections.abc
import functools
import itertools
import math
import re
from fractions import Fraction

import numpy as np

_integer_kinds = (1, 2, 4, 8, 16)


def _real_formats_for_long_double(nmant, nexp, maxexp):
    # Kinds 4 and 8 are binary32 and binary64. gfortran has both on every
    # target (KIND Type Parameters, The GNU Fortran Compiler).
    formats = {4: (8, 24), 8: (11, 53)}
    # Kind 10 is 80-bit extended precision. gfortran's kind is the mode
    # precision rounded up to bytes, and XFmode is kind 10
    # (gcc/fortran/trans-types.cc, GCC 13.3). C long double has that
    # shape when NumPy reports 63 fraction bits, 15 exponent bits and
    # max exponent 16384. capi_maps.py maps real(10) from the 63 bits.
    # Any other long double has no real(10).
    if (nmant, nexp, maxexp) == (63, 15, 16384):
        formats[10] = (15, 64)
    # Kind 16 is IEEE binary128, (15, 113). gfortran's real(16) is the
    # IBM extended double, two IEEE doubles, when long double has
    # binary64's max exponent and a significand wider than 53 bits
    # (trans-types.cc, the pnan < p branch; numpy/_core/getlimits.py).
    # That kind is left out. It is also left out when long double is
    # itself binary128: powerpc64le -mabi=ieeelongdouble keeps kind 16
    # for the IBM format and uses kind 17 for IEEE binary128
    # (trans-types.cc). A binary128 long double does not identify
    # kind 16, so those targets have no kind 16 here either.
    if (maxexp == 1024 and nmant > 52) or (nmant, maxexp) == (112, 16384):
        return formats
    formats[16] = (15, 113)
    return formats


_ldinfo = np.finfo(np.longdouble)


_real_formats = _real_formats_for_long_double(
    _ldinfo.nmant, _ldinfo.nexp, _ldinfo.maxexp)


_character_kinds = {'ascii': 1, 'iso_10646': 4}


_logical_kinds = (1, 2, 4, 8, 16)


_default_kinds = {'integer': 4, 'real': 4, 'complex': 4, 'logical': 4,
                  'character': 1, 'double precision': 8}


@functools.cache
def _model(typespec, kind):
    """Return the radix, digits and least and greatest exponent of the
    model numbers of a kind, with no exponents for an integer."""
    if typespec == 'integer' and kind in _integer_kinds:
        # every bit but the sign is a digit
        return 2, 8 * kind - 1, None, None
    if typespec in ('real', 'complex') and kind in _real_formats:
        bits, digits = _real_formats[kind]
        # IEEE 754 writes significands from 1 to 2 and the model from 1/2
        # to 1, so the model exponents are one more than IEEE's
        emax = 2 ** (bits - 1)
        return 2, digits, 3 - emax, emax
    raise TypeError(f'f2py knows no {typespec} of kind {kind!r}')


def _huge_number(typespec, kind):
    radix, digits, _, emax = _model(typespec, kind)
    if emax is None:
        return radix ** digits - 1
    return (1 - Fraction(radix) ** -digits) * radix ** emax


def _decimal_exponent(x):
    # the d with 10**d <= x < 10**(d + 1) of a number x > 0, exactly, so
    # INT(LOG10(x)) where x is at least 1
    x = Fraction(x)
    d = math.floor(math.log10(x.numerator) - math.log10(x.denominator))
    while Fraction(10) ** d > x:
        d -= 1
    while Fraction(10) ** (d + 1) <= x:
        d += 1
    return d


@functools.cache
def _precision(kind):
    radix, digits, _, _ = _model('real', kind)
    # INT((p - 1) * LOG10(b)), as the radix is no power of 10
    return _decimal_exponent(Fraction(radix) ** (digits - 1))


@functools.cache
def _range(typespec, kind):
    radix, _, emin, _ = _model(typespec, kind)
    huge = _decimal_exponent(_huge_number(typespec, kind))
    if emin is None:
        return huge
    # INT(MIN(LOG10(huge), -LOG10(tiny))), where tiny is b**(emin - 1)
    return min(huge, _decimal_exponent(Fraction(radix) ** (1 - emin)))


def _default_integer(value):
    return _Integer(value, _default_kinds['integer'])


def _selected_int_kind_func(r):
    # 2**q - 1 holds all of -10**r < n < 10**r exactly when its decimal
    # range is at least r
    kinds = [k for k in _integer_kinds if _range('integer', k) >= r]
    if not kinds:
        return _default_integer(-1)
    # of those the one of least range, and then the least kind
    return _default_integer(min(kinds,
                                key=lambda k: (_range('integer', k), k)))


def _selected_real_kind_func(p=None, r=None, radix=None):
    if p is None and r is None and radix is None:
        raise TypeError('selected_real_kind takes at least one argument')
    p = 0 if p is None else p
    r = 0 if r is None else r
    kinds = [k for k in _real_formats
             if radix is None or _model('real', k)[0] == radix]
    with_p = {k for k in kinds if _precision(k) >= p}
    with_r = {k for k in kinds if _range('real', k) >= r}
    if with_p & with_r:
        # the one of least precision, and then the least kind
        return _default_integer(min(with_p & with_r,
                                    key=lambda k: (_precision(k), k)))
    if not kinds:
        return _default_integer(-5)
    if with_p and with_r:
        return _default_integer(-4)
    return _default_integer(-1 if with_r else -2 if with_p else -3)


def _selected_char_kind_func(name):
    # the name is read without regard to case or trailing blanks
    name = name.rstrip(' ').lower()
    if name == 'default':
        return _default_integer(_default_kinds['character'])
    return _default_integer(_character_kinds.get(name, -1))


class _Integer(int):
    """An integer of a kind, which is None for a kind f2py does not know."""

    def __new__(cls, value, kind=None):
        self = super().__new__(cls, value)
        self.kind = kind
        return self


class _Real:
    """A real of a kind, held exactly as that kind holds it."""

    __slots__ = ('value', 'kind')

    def __init__(self, value, kind=None):
        self.value = _round(Fraction(value), kind)
        self.kind = kind

    def __eq__(self, other):
        if isinstance(other, _Real):
            return self.value == other.value
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return self.value == other
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __float__(self):
        try:
            return float(self.value)
        except OverflowError:
            return math.copysign(math.inf, self.value)

    def __str__(self):
        return _real_text(self.value, self.kind)

    __repr__ = __str__


class _Logical:
    """A logical of a kind."""

    __slots__ = ('value', 'kind')

    def __init__(self, value, kind=None):
        if kind is not None and kind not in _logical_kinds:
            raise TypeError(f'f2py knows no logical of kind {kind!r}')
        self.value = bool(value)
        self.kind = kind

    def __bool__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, (_Logical, bool)):
            return self.value == bool(other)
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return '.true.' if self.value else '.false.'

    __repr__ = __str__


class _Character(str):
    """A character value of a kind."""

    def __new__(cls, value, kind=None):
        if kind is not None and kind not in _character_kinds.values():
            raise TypeError(f'f2py knows no character of kind {kind!r}')
        self = super().__new__(cls, value)
        self.kind = kind
        return self


class _Complex:
    """A complex of a kind, its parts held exactly as that kind holds them."""

    __slots__ = ('real', 'imag', 'kind')

    def __init__(self, real, imag, kind=None):
        self.real = _round(Fraction(real), kind)
        self.imag = _round(Fraction(imag), kind)
        self.kind = kind

    def __eq__(self, other):
        if isinstance(other, _Complex):
            return (self.real, self.imag) == (other.real, other.imag)
        if (isinstance(other, (int, float, complex))
                and not isinstance(other, bool)):
            other = complex(other)
            return self.real == other.real and self.imag == other.imag
        return NotImplemented

    def __hash__(self):
        if not self.imag:
            return hash(self.real)
        return hash(complex(self))

    def __complex__(self):
        return complex(float(_Real(self.real)), float(_Real(self.imag)))

    def __str__(self):
        return (f'({_real_text(self.real, self.kind)}, '
                f'{_real_text(self.imag, self.kind)})')

    __repr__ = __str__


class _Unknown:
    """A name of known type and kind but unknown value, such as a variable,
    which only inquiries take."""

    __slots__ = ('typespec', 'kind')

    def __init__(self, typespec, kind):
        self.typespec = typespec
        self.kind = kind

    def __bool__(self):
        raise TypeError('the value is not known')


_value_types = {_Integer: 'integer', _Real: 'real', _Complex: 'complex',
                _Logical: 'logical', _Character: 'character'}


_numeric_types = ('integer', 'real', 'complex')


def _type_kind(x):
    """Return the type and kind of a value, or of the elements of an
    array."""
    if isinstance(x, (dict, list)):
        elements = list(x.values()) if isinstance(x, dict) else x
        if not elements:
            raise TypeError('an empty array has no type f2py knows')
        x = elements[0]
    for cls, typespec in _value_types.items():
        if isinstance(x, cls):
            return typespec, x.kind
    raise TypeError(f'{x!r} is not a Fortran value')


def _typed(value):
    """Give a value from Python, such as a parameter that a caller passes,
    its Fortran type, of a kind f2py does not know."""
    if isinstance(value, bool):
        return _Logical(value)
    if isinstance(value, int) and not isinstance(value, _Integer):
        return _Integer(value)
    if isinstance(value, float):
        return _Real(value)
    if isinstance(value, dict) and not isinstance(value, _ParameterArray):
        return _ParameterArray({k: _typed(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_typed(v) for v in value]
    return value


def _integer(value, kind):
    """Return an integer of a kind, which must hold it."""
    if kind is not None:
        digits = _model('integer', kind)[1]
        # two's complement holds -huge - 1 as well
        if not -2 ** digits <= value < 2 ** digits:
            raise OverflowError(f'{value} overflows integer({kind})')
    return _Integer(value, kind)


def _exact(x):
    # the exact value of an integer or real
    if isinstance(x, _Real):
        return x.value
    if isinstance(x, _Integer):
        return Fraction(int(x))
    raise TypeError(f'{x!r} is not an integer or real')


def _parts(x):
    # the exact real and imaginary parts of a number
    if isinstance(x, _Complex):
        return x.real, x.imag
    return _exact(x), Fraction(0)


def _exponent(a):
    # the exponent e of a > 0 in the model, 2**(e - 1) <= a < 2**e
    e = a.numerator.bit_length() - a.denominator.bit_length()
    return e + 1 if a >= Fraction(2) ** e else e


def _round(x, kind):
    """Round x to the nearest real of a kind, from halfway to even, as
    IEEE 754 rounds."""
    if kind is None or not x:
        return x
    radix, digits, emin, _ = _model('real', kind)
    a = abs(x)
    # below the least exponent the numbers are spaced as at it
    spacing = Fraction(radix) ** (max(_exponent(a), emin) - digits)
    n, rest = divmod(a, spacing)
    if 2 * rest > spacing or 2 * rest == spacing and n % 2:
        n += 1
    if n * spacing > _huge_number('real', kind):
        raise OverflowError(f'{x} overflows real({kind})')
    return n * spacing if x > 0 else -n * spacing


def _real_text(x, kind):
    """Write a real as Python writes a float, with the fewest digits that
    read back as the same real of its kind."""
    if kind is None:
        try:
            return repr(float(x))
        except OverflowError:
            return repr(math.copysign(math.inf, x))
    if not x:
        return '0.0'
    radix, digits, emin, _ = _model('real', kind)
    a = abs(x)
    e = _exponent(a)
    spacing = Fraction(radix) ** (max(e, emin) - digits)
    # the reals that read as a lie up to halfway to its neighbours, the one
    # below being nearer at a power of the radix, and halfway reads as
    # whichever is even
    below = spacing
    if a == Fraction(radix) ** (e - 1) and e > emin:
        below = spacing / radix
    low, high = a - below / 2, a + spacing / 2
    closed = (a / spacing) % 2 == 0
    point = _decimal_exponent(a) + 1
    for n in itertools.count(1):
        scale = Fraction(10) ** (point - n)
        least = (math.ceil(low / scale) if closed
                 else math.floor(low / scale) + 1)
        most = (math.floor(high / scale) if closed
                else math.ceil(high / scale) - 1)
        if least <= most:
            d = str(min(max(round(a / scale), least), most))
            break
    # a is 0.<significand> * 10**decimal_point
    significand = d.rstrip('0')
    decimal_point = point - n + len(d)
    text = _float_text(significand, decimal_point)
    return '-' + text if x < 0 else text


def _float_text(digits, point):
    # as repr writes a float, positionally from 1e-4 to below 1e16
    if -4 < point <= 16:
        if point <= 0:
            return '0.' + '0' * -point + digits
        if point >= len(digits):
            return digits + '0' * (point - len(digits)) + '.0'
        return digits[:point] + '.' + digits[point:]
    mantissa = digits[0] + ('.' + digits[1:] if len(digits) > 1 else '')
    return f'{mantissa}e{point - 1:+03d}'


def _exponent_bounds(kind):
    # the exponents of the powers of 2 outside of which a real of the kind,
    # or of any kind if kind is None, overflows or rounds to zero
    kinds = _real_formats if kind is None else [kind]
    models = [_model('real', k) for k in kinds]
    return (min(emin - digits - 1 for _, digits, emin, _ in models),
            max(emax + 1 for _, _, _, emax in models))


def _real_power(x, n, kind):
    """Raise an exact real x to an integer power n, for a real of a
    kind."""
    if not x:
        if n < 0:
            raise ZeroDivisionError('0 to a negative power')
        return _Real(Fraction(not n), kind)
    if abs(x) != 1:
        # a power far outside the kind's exponents overflows or rounds to
        # zero, and need not be computed
        low, high = _exponent_bounds(kind)
        log2 = n * (math.log2(abs(x.numerator)) - math.log2(x.denominator))
        if log2 > high:
            raise OverflowError(f'{x}**{n} overflows real({kind})')
        if log2 < low:
            return _Real(0, kind)
    return _Real(x ** n, kind)


def _promote(ta, ka, tb, kb):
    """Return the type and kind of the result of a numeric operation: the
    kind of the operand of greater range of integers and of greater
    precision of reals (Fortran 2018 10.1.5.2.1)."""
    typespec = max(ta, tb, key=_numeric_types.index)
    kinds = [k for t, k in ((ta, ka), (tb, kb))
             if t != 'integer' or typespec == 'integer']
    if None in kinds:
        return typespec, None
    if typespec == 'integer':
        return typespec, max(kinds, key=lambda k: (_range('integer', k), k))
    return typespec, max(kinds, key=lambda k: (_precision(k), k))


def _convert(x, typespec, kind):
    """Convert a value to a type and kind, as assignment does, an integer
    or real taking the real part of a complex."""
    t = _type_kind(x)[0]
    if typespec in _numeric_types and t in _numeric_types:
        real, imag = _parts(x)
        if typespec == 'integer':
            return _integer(math.trunc(real), kind)
        if typespec == 'real':
            return _Real(real, kind)
        return _Complex(real, imag, kind)
    if typespec == t == 'logical':
        return _Logical(x.value, kind)
    if typespec == t == 'character':
        return _Character(x, kind)
    raise TypeError(f'{t} does not convert to {typespec}')


_arithmetic = {'Add': lambda x, y: x + y, 'Sub': lambda x, y: x - y,
               'Mult': lambda x, y: x * y}


def _binary(op, a, b):
    """Apply a binary operator to two values, as Fortran does."""
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        # Python would repeat or join lists, where Fortran works elementwise
        raise TypeError('array arithmetic is not supported')
    (ta, ka), (tb, kb) = _type_kind(a), _type_kind(b)
    if op == 'MatMult':
        # the // of Fortran joins characters of one kind
        if not ta == tb == 'character' or ka != kb:
            raise TypeError('// joins characters of one kind')
        return _Character(str(a) + str(b), ka)
    if ta not in _numeric_types or tb not in _numeric_types:
        raise TypeError(f'{op} is not defined for {ta} and {tb}')
    typespec, kind = _promote(ta, ka, tb, kb)
    if op == 'Pow' and tb == 'integer' and ta != 'integer':
        # an integer power is of the type and kind of its base
        if ta == 'complex':
            return _complex_power(*_parts(a), int(b), ka)
        return _real_power(_exact(a), int(b), ka)
    if typespec == 'complex':
        return _complex_binary(op, *_parts(_convert(a, typespec, kind)),
                               *_parts(_convert(b, typespec, kind)), kind)
    x = _exact(_convert(a, typespec, kind))
    y = _exact(_convert(b, typespec, kind))
    if op == 'Pow':
        if typespec == 'integer':
            return _integer_power(int(x), int(y), kind)
        # a real power is evaluated only when it is an integer, and Fortran
        # forbids a negative base
        if y.denominator != 1 or x < 0:
            raise TypeError(f'{x}**{y} is not supported')
        return _real_power(x, int(y), kind)
    if op == 'Div':
        if not y:
            raise ZeroDivisionError('division by zero')
        if typespec == 'integer':
            # the quotient of integers is truncated
            return _integer(math.trunc(x / y), kind)
        return _Real(x / y, kind)
    if op not in _arithmetic:
        raise TypeError(f'{op} is not Fortran')
    if typespec == 'integer':
        return _integer(int(_arithmetic[op](x, y)), kind)
    return _Real(_arithmetic[op](x, y), kind)


def _complex_binary(op, p, q, r, s, kind):
    # the operation on p + q i and r + s i, each part rounded once
    if op == 'Add':
        return _Complex(p + r, q + s, kind)
    if op == 'Sub':
        return _Complex(p - r, q - s, kind)
    if op == 'Mult':
        return _Complex(p * r - q * s, p * s + q * r, kind)
    if op == 'Div':
        d = r * r + s * s
        if not d:
            raise ZeroDivisionError('division by zero')
        return _Complex((p * r + q * s) / d, (q * r - p * s) / d, kind)
    raise TypeError(f'{op} of a complex is not supported')


def _complex_power(p, q, n, kind):
    """Raise p + q i to an integer power n, for a complex of a kind."""
    if not (p or q):
        if n < 0:
            raise ZeroDivisionError('0 to a negative power')
        return _Complex(Fraction(not n), 0, kind)
    m = p * p + q * q
    if m != 1:
        # a power far outside the kind's exponents overflows or rounds to
        # zero, and need not be computed
        low, high = _exponent_bounds(kind)
        log2 = n * (math.log2(m.numerator) - math.log2(m.denominator)) / 2
        if log2 > high:
            raise OverflowError(f'({p}, {q})**{n} overflows complex({kind})')
        if log2 < low:
            return _Complex(0, 0, kind)
    if n < 0:
        p, q, n = p / m, -q / m, -n
    real, imag = Fraction(1), Fraction(0)
    while n:
        if n % 2:
            real, imag = real * p - imag * q, real * q + imag * p
        p, q, n = p * p - q * q, 2 * p * q, n // 2
    return _Complex(real, imag, kind)


def _integer_power(x, n, kind):
    if n < 0:
        # 1 / x**-n truncated is 0 unless x is 1 or -1
        if not x:
            raise ZeroDivisionError('0 to a negative power')
        return _integer(x ** (n % 2) if abs(x) == 1 else 0, kind)
    # a power of at least 2 beyond the bits of the widest integer overflows
    # it, and need not be computed
    bits = _model('integer', max(_integer_kinds) if kind is None else kind)[1]
    if abs(x) > 1 and n > bits:
        raise OverflowError(f'{x}**{n} overflows integer({kind})')
    return _integer(x ** n, kind)


def _unary(op, a):
    """Apply a unary operator to a value, as Fortran does."""
    if isinstance(a, (list, dict)):
        raise TypeError('array arithmetic is not supported')
    typespec, kind = _type_kind(a)
    if op == 'Not':
        if typespec != 'logical':
            raise TypeError('.not. takes a logical')
        return _Logical(not a, kind)
    if typespec not in _numeric_types:
        raise TypeError(f'{op} is not defined for {typespec}')
    if op == 'UAdd':
        return a
    if typespec == 'integer':
        return _integer(-int(a), kind)
    if typespec == 'complex':
        return _Complex(-a.real, -a.imag, kind)
    return _Real(-a.value, kind)


_relations = {'Eq': lambda x, y: x == y, 'NotEq': lambda x, y: x != y,
              'Lt': lambda x, y: x < y, 'LtE': lambda x, y: x <= y,
              'Gt': lambda x, y: x > y, 'GtE': lambda x, y: x >= y}


def _relational(op, a, b):
    """Compare two values, as Fortran does."""
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        raise TypeError('array comparison is not supported')
    (ta, ka), (tb, kb) = _type_kind(a), _type_kind(b)
    if ta == tb == 'character' and ka == kb:
        # the shorter is compared as if it were padded with blanks
        n = max(len(a), len(b))
        x, y = a.ljust(n), b.ljust(n)
    elif ta in _numeric_types and tb in _numeric_types:
        typespec, kind = _promote(ta, ka, tb, kb)
        x = _parts(_convert(a, typespec, kind))
        y = _parts(_convert(b, typespec, kind))
        if typespec != 'complex':
            x, y = x[0], y[0]
        elif op not in ('Eq', 'NotEq'):
            raise TypeError('complex values are not ordered')
    else:
        raise TypeError(f'{ta} and {tb} do not compare')
    return _Logical(_relations[op](x, y), _default_kinds['logical'])


def _logical(op, *values):
    """Apply .and. or .or. to logical values."""
    if any(isinstance(v, (list, dict)) or _type_kind(v)[0] != 'logical'
           for v in values):
        raise TypeError(f'{op} takes logicals')
    kinds = {v.kind for v in values}
    value = all(values) if op == 'And' else any(values)
    kind = kinds.pop() if len(kinds) == 1 else _default_kinds['logical']
    return _Logical(value, kind)


def _complex_literal(real, imag):
    """Return the complex literal (real, imag), of the kind of its real part
    of greater precision, or of the default real if both are integers."""
    (tr, kr), (ti, ki) = _type_kind(real), _type_kind(imag)
    if not {tr, ti} <= {'integer', 'real'}:
        raise TypeError('the parts of a complex literal are numbers')
    kinds = [k for t, k in ((tr, kr), (ti, ki)) if t == 'real']
    if not kinds:
        kind = _default_kinds['real']
    elif None in kinds:
        kind = None
    else:
        kind = max(kinds, key=lambda k: (_precision(k), k))
    return _Complex(_exact(real), _exact(imag), kind)


_operations = {'_binary': _binary, '_unary': _unary,
               '_relational': _relational, '_logical': _logical,
               '_complex_literal': _complex_literal}


def _operation(name, op, *args):
    return ast.Call(ast.Name(name, ast.Load()), [ast.Constant(op), *args], [])


class _FortranOperations(ast.NodeTransformer):
    """Make each operation of an expression a call of one of _operations,
    which follow Fortran's rules for the types and kinds of operands.
    Python that is not also Fortran is refused."""

    _nodes = (ast.Expression, ast.Name, ast.Load, ast.Call, ast.keyword,
              ast.List, ast.Tuple, ast.BinOp, ast.UnaryOp, ast.BoolOp,
              ast.Compare, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)

    def generic_visit(self, node):
        if not isinstance(node, self._nodes):
            raise TypeError(f'{type(node).__name__} is not Fortran')
        return super().generic_visit(node)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise TypeError('only named functions are called')
        return self.generic_visit(node)

    def visit_Tuple(self, node):
        # (re, im) is a complex literal
        self.generic_visit(node)
        if len(node.elts) != 2:
            raise TypeError('a complex literal has two parts')
        return ast.Call(ast.Name('_complex_literal', ast.Load()), node.elts,
                        [])

    def visit_BinOp(self, node):
        self.generic_visit(node)
        return _operation('_binary', type(node.op).__name__, node.left,
                          node.right)

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        return _operation('_unary', type(node.op).__name__, node.operand)

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        return _operation('_logical', type(node.op).__name__, *node.values)

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) > 1:
            raise TypeError('Fortran relations do not chain')
        return _operation('_relational', type(node.ops[0]).__name__,
                          node.left, node.comparators[0])


_dotted_names = 'eq|ne|lt|le|gt|ge|not|and|or|eqv|neqv|true|false'


_fortran_token_re = re.compile(
    r"(?:\b(?P<char_kind>\w+)_)?(?P<char>'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")"
    r'|(?P<concat>//)'
    r'|\.(?P<logical>true|false)\.(?:_(?P<logical_kind>\w+))?'
    r'|(?P<operator>\.(?:eq|ne|lt|le|gt|ge|not|and|or)\.|/=)'
    rf'|(?<!\w)(?P<number>\d+(?:\.(?!(?:{_dotted_names})\.)\d*)?|\.\d+)'
    r'(?:(?P<letter>[deq])(?P<exponent>[-+]?\d+))?(?:_(?P<kind>\w+))?',
    re.I)


_python_operators = {'.eq.': '==', '.ne.': '!=', '.lt.': '<', '.le.': '<=',
                     '.gt.': '>', '.ge.': '>=', '.not.': 'not',
                     '.and.': 'and', '.or.': 'or', '/=': '!='}


def _literal_kind(kind, typespec, names):
    """Return the kind a literal names, or the default kind."""
    if kind is None:
        return _default_kinds[typespec]
    if kind.isdigit():
        return int(kind)
    try:
        value = names[kind]
    except KeyError:
        raise NameError(f'name {kind!r} is not defined') from None
    if _type_kind(value)[0] != 'integer':
        raise TypeError(f'kind {kind} is not an integer')
    return int(value)


def _number(m, names, c_literals):
    """Return the value of a number literal."""
    digits, letter, kind = m['number'], m['letter'], m['kind']
    if letter is None and '.' not in digits:
        value = int(digits)
        if kind is not None:
            return _integer(value, _literal_kind(kind, 'integer', names))
        try:
            return _integer(value, _default_kinds['integer'])
        except OverflowError:
            # gfortran refuses such a literal, and flang reads it as
            # integer(8), so its kind is not known
            return _Integer(value)
    letter = (letter or 'e').lower()
    if letter == 'e':
        # C reads a floating constant without a suffix as a double
        real_kind = (_literal_kind(kind, 'real', names) if kind is not None
                     else 8 if c_literals else _default_kinds['real'])
    elif kind is None:
        # q is gfortran's real(16); d is real(8)
        real_kind = 8 if letter == 'd' else 16
    else:
        raise TypeError(f'a {letter} exponent takes no kind')
    mantissa, scale = Fraction(digits), int(m['exponent'] or 0)
    if mantissa:
        # a power of ten far outside the kind's exponents overflows or rounds
        # to zero, and need not be computed
        low, high = _exponent_bounds(real_kind)
        log2 = (scale + math.log10(mantissa)) * math.log2(10)
        if log2 > high:
            raise OverflowError(f'{m[0]} overflows real({real_kind})')
        if log2 < low:
            mantissa = 0
    return _Real(mantissa * Fraction(10) ** scale, real_kind)


def _translate(expr, names, c_literals=False):
    """Translate a Fortran expression to Python, each literal in it becoming
    a name for its value, which the returned dict gives."""
    literals = {}

    def literal(value):
        name = f'_literal{len(literals)}'
        literals[name] = value
        return name

    def token(m):
        if m['char'] is not None:
            s = m['char']
            return literal(_Character(
                s[1:-1].replace(2 * s[0], s[0]),
                _literal_kind(m['char_kind'], 'character', names)))
        if m['concat']:
            return ' @ '
        if m['logical']:
            return literal(_Logical(
                m['logical'].lower() == 'true',
                _literal_kind(m['logical_kind'], 'logical', names)))
        if m['operator']:
            return f" {_python_operators[m['operator'].lower()]} "
        return literal(_number(m, names, c_literals))

    source = _fortran_token_re.sub(token, expr)
    # (/ and /) delimit array constructors as [ and ] do
    return source.replace('(/', '[').replace('/)', ']').strip(), literals


class _ParameterArray(dict):
    # pa(2) in an expression is an element of the parameter array pa
    def __call__(self, *index):
        return self[index if len(index) > 1 else index[0]]


class _Names(collections.abc.Mapping):
    """The values of the names in an expression, in their Fortran types."""

    def __init__(self, *maps):
        self.maps = maps

    def __getitem__(self, name):
        for m in self.maps:
            # text is a value that f2py could not evaluate, and a later map
            # may know at least its type
            if name in m and (not isinstance(m[name], str)
                              or isinstance(m[name], _Character)):
                return _typed(m[name])
        raise KeyError(name)

    def __iter__(self):
        return iter({name for m in self.maps for name in m})

    def __len__(self):
        return len({name for m in self.maps for name in m})


def _fortran_eval(expr, g_params, params, c_literals=False):
    """Evaluate a Fortran expression, each value of it of a type and kind.

    With c_literals a real literal without a kind is a double, as in the C
    that f2py writes the initial values of arguments in.
    """
    names = _Names(params, g_params)
    source, literals = _translate(expr, names, c_literals)
    tree = _FortranOperations().visit(ast.parse(source, mode='eval'))
    code = compile(ast.fix_missing_locations(tree), '<fortran>', 'eval')
    # no builtins, as those of Python are not Fortran's
    return eval(code, {'__builtins__': {}, **_operations, **_intrinsics,
                       **literals}, names)


def _elements(array):
    # a parameter array is a dict of its elements, keyed by their indices
    return list(array.values()) if isinstance(array, dict) else list(array)


def _kind_arg(kind, typespec):
    # the kind a kind= argument gives, or the default kind
    if kind is None:
        return _default_kinds[typespec]
    if _type_kind(kind)[0] != 'integer':
        raise TypeError('a kind is an integer')
    return int(kind)


def _real_arg(a):
    if isinstance(a, (list, dict)) or _type_kind(a)[0] != 'real':
        raise TypeError(f'{a!r} is not a real')
    return a.value


def _character_arg(c):
    if isinstance(c, (list, dict)) or _type_kind(c)[0] != 'character':
        raise TypeError(f'{c!r} is not a character')
    return c


def _common_type(args):
    """Return the type and kind that integer or real arguments promote
    to, as gfortran and flang promote the arguments of max and mod."""
    if any(isinstance(a, (list, dict)) for a in args):
        raise TypeError('array arguments are not supported')
    types = [_type_kind(a) for a in args]
    if {t for t, _ in types} not in ({'integer'}, {'real'}):
        raise TypeError('the arguments are integers or reals')
    return functools.reduce(lambda x, y: _promote(*x, *y), types)


def _extremum(pick, args):
    typespec, kind = _common_type(args)
    values = [_exact(_convert(a, typespec, kind)) for a in args]
    return _convert(_Real(pick(values)), typespec, kind)


def _nint(a, kind=None):
    v = _real_arg(a)
    # halves round away from zero
    n = math.floor(abs(v) + Fraction(1, 2))
    return _integer(n if v >= 0 else -n, _kind_arg(kind, 'integer'))


def _remainder(a, p, quotient):
    typespec, kind = _common_type([a, p])
    x = _exact(_convert(a, typespec, kind))
    y = _exact(_convert(p, typespec, kind))
    if not y:
        raise ZeroDivisionError('division by zero')
    return _convert(_Real(x - quotient(x / y) * y), typespec, kind)


def _sign(a, b):
    _common_type([a, b])
    x = abs(_exact(a))
    # the result is of the type and kind of a
    return _convert(_Real(x if _exact(b) >= 0 else -x), *_type_kind(a))


def _reduce(op, array):
    elements = _elements(array)
    if not elements:
        raise TypeError('an empty array is not supported')
    return functools.reduce(functools.partial(_binary, op), elements)


def _integer_arg(i):
    if isinstance(i, (list, dict)) or _type_kind(i)[0] != 'integer':
        raise TypeError(f'{i!r} is not an integer')
    return int(i)


def _char(i, kind=None):
    kind = _kind_arg(kind, 'character')
    # a character of kind k holds a code of 8 * k bits
    if not 0 <= _integer_arg(i) < 2 ** (8 * kind):
        raise TypeError(f'{i} is not a code of a character of kind {kind}')
    return _Character(chr(i), kind)


def _ichar(c, kind=None):
    if len(_character_arg(c)) != 1:
        raise TypeError(f'{c!r} is not of length 1')
    return _integer(ord(c), _kind_arg(kind, 'integer'))


def _repeat(string, ncopies):
    if _integer_arg(ncopies) < 0:
        raise TypeError('repeat takes no negative ncopies')
    return _Character(_character_arg(string) * int(ncopies), string.kind)


def _len(string, kind=None):
    # of a character array, the length of each element
    if isinstance(string, (list, dict)):
        string = _elements(string)[0]
    return _integer(len(_character_arg(string)), _kind_arg(kind, 'integer'))


def _size(array, dim=None, kind=None):
    if not isinstance(array, (list, dict)):
        raise TypeError(f'{array!r} is not an array')
    n = len(array) if dim is None else _extent(array, _integer_arg(dim))
    return _integer(n, _kind_arg(kind, 'integer'))


def _inquired(x):
    # the type and kind of a value, an array or a name of unknown value
    if isinstance(x, _Unknown):
        return x.typespec, x.kind
    return _type_kind(x)


def _complex_arg(z):
    if isinstance(z, (list, dict)) or _type_kind(z)[0] != 'complex':
        raise TypeError(f'{z!r} is not a complex')
    return z


def _cmplx(x, y=None, kind=None):
    # without kind=, of the default real kind, whatever the kinds of x and y
    kind = _kind_arg(kind, 'real')
    if _type_kind(x)[0] == 'complex':
        if y is not None:
            raise TypeError('cmplx of a complex takes no y')
        return _convert(x, 'complex', kind)
    return _Complex(_exact(x), 0 if y is None else _exact(y), kind)


def _real(a, kind=None):
    # of the kind of a complex a without kind=
    if kind is None and _type_kind(a)[0] == 'complex':
        return _convert(a, 'real', a.kind)
    return _convert(a, 'real', _kind_arg(kind, 'real'))


def _kind_func(x):
    kind = _inquired(x)[1]
    if kind is None:
        raise TypeError(f'f2py does not know the kind of {x!r}')
    return _default_integer(kind)


def _inquiry(typespecs, result):
    """Return an inquiry function of the model numbers of a kind, which
    takes a value, an array or a name of one of typespecs."""
    def inquiry(x):
        typespec, kind = _inquired(x)
        if typespec not in typespecs or kind is None:
            raise TypeError(f'the {typespec} of kind {kind} has no such model')
        return result(typespec, kind, *_model(typespec, kind))
    return inquiry


_intrinsics = {
    'int': lambda a, kind=None: _convert(a, 'integer',
                                         _kind_arg(kind, 'integer')),
    'nint': _nint,
    'ceiling': lambda a, kind=None: _integer(math.ceil(_real_arg(a)),
                                             _kind_arg(kind, 'integer')),
    'floor': lambda a, kind=None: _integer(math.floor(_real_arg(a)),
                                           _kind_arg(kind, 'integer')),
    'real': _real,
    'float': lambda a: _convert(a, 'real', _default_kinds['real']),
    'dble': lambda a: _convert(a, 'real', _default_kinds['double precision']),
    'cmplx': _cmplx,
    # an extension of gfortran, of double precision
    'dcmplx': lambda x, y=None: _cmplx(
        x, y, _default_integer(_default_kinds['double precision'])),
    'aimag': lambda z: _Real(_complex_arg(z).imag, z.kind),
    'conjg': lambda z: _Complex(_complex_arg(z).real, -z.imag, z.kind),
    'abs': lambda a: _convert(_Real(abs(_exact(a))), *_type_kind(a)),
    'max': lambda *a: _extremum(max, a),
    'min': lambda *a: _extremum(min, a),
    'mod': lambda a, p: _remainder(a, p, math.trunc),
    'modulo': lambda a, p: _remainder(a, p, math.floor),
    'sign': _sign,
    'merge': lambda tsource, fsource, mask:
        tsource if _logical('And', mask) else fsource,
    'achar': _char,
    'char': _char,
    'iachar': _ichar,
    'ichar': _ichar,
    'repeat': _repeat,
    'trim': lambda string: _Character(_character_arg(string).rstrip(' '),
                                      string.kind),
    'len_trim': lambda string, kind=None: _integer(
        len(_character_arg(string).rstrip(' ')), _kind_arg(kind, 'integer')),
    'len': _len,
    'size': _size,
    'sum': lambda array: _reduce('Add', array),
    'product': lambda array: _reduce('Mult', array),
    'maxval': lambda array: _extremum(max, _elements(array)),
    'minval': lambda array: _extremum(min, _elements(array)),
    'any': lambda mask: _logical('Or', *_elements(mask)),
    'all': lambda mask: _logical('And', *_elements(mask)),
    'kind': _kind_func,
    'selected_int_kind': _selected_int_kind_func,
    'selected_real_kind': _selected_real_kind_func,
    'selected_char_kind': _selected_char_kind_func,
    # the model numbers of 16.4 with radix b, digits q or p and exponents
    # emin to emax, as 16.9 defines these inquiries of them
    'bit_size': _inquiry(('integer',), lambda t, k, b, q, emin, emax:
                         _integer(q + 1, k)),
    'digits': _inquiry(('integer', 'real'), lambda t, k, b, p, emin, emax:
                       _default_integer(p)),
    'epsilon': _inquiry(('real',), lambda t, k, b, p, emin, emax:
                        _Real(Fraction(b) ** (1 - p), k)),
    'huge': _inquiry(('integer', 'real'), lambda t, k, b, p, emin, emax:
                     (_integer if t == 'integer' else _Real)(
                         _huge_number(t, k), k)),
    'maxexponent': _inquiry(('real',), lambda t, k, b, p, emin, emax:
                            _default_integer(emax)),
    'minexponent': _inquiry(('real',), lambda t, k, b, p, emin, emax:
                            _default_integer(emin)),
    'precision': _inquiry(('real', 'complex'), lambda t, k, b, p, emin, emax:
                          _default_integer(_precision(k))),
    'radix': _inquiry(('integer', 'real'), lambda t, k, b, p, emin, emax:
                      _default_integer(b)),
    'range': _inquiry(('integer', 'real', 'complex'),
                      lambda t, k, b, p, emin, emax:
                      _default_integer(_range(t, k))),
    'tiny': _inquiry(('real',), lambda t, k, b, p, emin, emax:
                     _Real(Fraction(b) ** (emin - 1), k)),
}


def _extent(array, dim):
    # a parameter array is keyed by its subscripts, a tuple for rank > 1
    return len({k[dim - 1] if isinstance(k, tuple) else k for k in array})
