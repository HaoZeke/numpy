"""Infer types of numeric callback constants without evaluating their values."""
import ast
import re

from ._derived_expr import _default_kinds, _precision, _range

_number = re.compile(
    r'(?<![\w.])(?P<value>(?:\d+\.\d*|\.\d+|\d+))'
    r'(?P<exponent>[ed][+-]?\d+)?(?:_(?P<kind>\w+))?', re.I)
_numeric_order = ('integer', 'real', 'complex')


def _common(operands):
    typespec = max((item[0] for item in operands), key=_numeric_order.index)
    selected = [item for item in operands
                if item[0] != 'integer' or typespec == 'integer']
    kinds = {item[1] for item in selected}
    if len(kinds) == 1:
        kind = selected[0][1]
    elif all(kind.isdecimal() for kind in kinds):
        if typespec == 'integer':
            def rank(kind):
                return _range('integer', int(kind)), int(kind)
        else:
            def rank(kind):
                return _precision(int(kind)), int(kind)
        kind = max(kinds, key=rank)
    else:
        raise ValueError('mixed symbolic kinds need a declared expression type')
    explicit = any(item[2] for item in selected)
    return typespec, kind, explicit


def literal_expression_type(expression):
    """Return numeric metadata for a closed literal expression, or None.

    Only arithmetic, unary signs, grouping and complex literals enter this
    path. Names, calls and other forms retain the caller's existing analysis.
    No arithmetic values are computed or arrays materialized.
    """
    literals = {}
    prefix = '_f2py_literal_'
    while prefix in expression:
        prefix += '_'

    def token(match):
        value = match['value']
        exponent = (match['exponent'] or '').lower()
        kind = match['kind']
        typespec = 'real' if '.' in value or exponent else 'integer'
        if exponent.startswith('d'):
            if kind is not None:
                raise ValueError('a D exponent has no explicit kind suffix')
            kind = str(_default_kinds['double precision'])
            explicit = True
        else:
            explicit = kind is not None
            kind = (kind.lower() if kind is not None
                    else str(_default_kinds[typespec]))
        name = f'{prefix}{len(literals)}'
        literals[name] = (typespec, kind, explicit)
        return name

    def infer(node):
        if isinstance(node, ast.Name) and node.id in literals:
            return literals[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return infer(node.operand)
        if (isinstance(node, ast.BinOp)
                and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div))):
            return _common([infer(node.left), infer(node.right)])
        if isinstance(node, ast.Tuple) and len(node.elts) == 2:
            parts = [infer(item) for item in node.elts]
            if any(item[0] == 'complex' for item in parts):
                raise ValueError('complex literal parts are integer or real')
            reals = [item for item in parts if item[0] == 'real']
            if not reals:
                return 'complex', str(_default_kinds['complex']), False
            _, kind, explicit = _common(reals)
            return 'complex', kind, explicit
        raise ValueError('not a closed numeric literal expression')

    try:
        translated = _number.sub(token, expression)
        if not literals:
            return None
        result_type, kind, explicit = infer(ast.parse(translated, mode='eval').body)
    except (ValueError, SyntaxError, TypeError, OverflowError, KeyError):
        return None
    result = {'typespec': result_type}
    if explicit:
        result['kindselector'] = {'kind': kind}
    return result
