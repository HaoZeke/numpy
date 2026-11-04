"""Evaluate storage metadata without constructing unrelated parameter values."""

import ast
import math
from collections import ChainMap

from . import crackfortran as cf


class _StorageParameters(dict):
    def __init__(self, variables, inherited):
        local = {name.lower() for name, var in variables.items()
                 if 'parameter' in var.get('attrspec', [])}
        super().__init__((name, value) for name, value in inherited.items()
                         if name.lower() not in local)
        self.variables = variables
        self.aliases = {name.lower(): name for name in variables}
        self.root = self
        self.active = set()

    def __copy__(self):
        result = type(self).__new__(type(self))
        dict.__init__(result, self)
        result.__dict__.update(self.__dict__)
        return result

    def __contains__(self, name):
        return (dict.__contains__(self, name) or name in self.variables
                or name in self.aliases)

    def __missing__(self, name):
        root = self.root
        if dict.__contains__(root, name):
            return dict.__getitem__(root, name)
        canonical = root.aliases.get(name.lower(), name)
        if canonical not in self.variables:
            raise KeyError(name)
        if canonical in root.active:
            raise KeyError(f'cyclic parameter dependency: {canonical}')
        var = self.variables[canonical]
        root.active.add(canonical)
        try:
            if 'parameter' in var.get('attrspec', []):
                values = cf._get_typed_parameters({canonical: var}, root)
                value = dict.__getitem__(values, canonical)
            else:
                typespec, kind = cf._declared_type(canonical, var, root)
                if typespec is None:
                    raise KeyError(name)
                value = cf._Unknown(typespec, kind)
            dict.__setitem__(root, canonical, value)
            dict.__setitem__(root, canonical.lower(), value)
            return value
        finally:
            root.active.remove(canonical)

    def _binding(self, name, names):
        pending = list(names.maps)
        while pending:
            mapping = pending.pop(0)
            if isinstance(mapping, (ChainMap, cf._Names)):
                pending[0:0] = mapping.maps
                continue
            if name not in mapping:
                continue
            if isinstance(mapping, _StorageParameters):
                return 'storage'
            value = mapping[name]
            if isinstance(value, str) and not isinstance(value, cf._Character):
                continue
            return 'local'
        return None

    def _reference(self, node):
        element = isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        name = node.func.id if element else node.id if isinstance(node, ast.Name) else None
        if name is None:
            return None
        name = self.aliases.get(name.lower(), name)
        if name not in self.variables:
            return None
        return name, self.variables[name], element

    def _evaluate_node(self, node, literals, names):
        namespace = cf._Names(*names.maps, literals)
        return cf._fortran_eval(ast.unparse(node), {}, namespace)

    def _length(self, name, var):
        selector = var.get('charselector') or {}
        expression = selector.get('f2py_len', selector.get('len', selector.get('*', '1')))
        if expression in ('*', '(*)'):
            return int(cf._len(self[name]))
        result = cf._fortran_eval(str(expression), {}, self)
        if not isinstance(result, int):
            raise TypeError('a character length is an integer')
        return max(0, int(result))

    def _shape(self, var):
        if 'dimension' in var:
            if any(value in ('*', ':', '..') for value in var['dimension']):
                return None
            values = [cf._fortran_eval(str(value), {}, self)
                      for value in var['dimension']]
            if not all(isinstance(value, int) for value in values):
                raise TypeError('an array extent is an integer')
            return [max(0, int(value)) for value in values]
        dimensions = next((item.removeprefix('dimension').strip()
                           for item in var.get('attrspec', [])
                           if item.startswith('dimension')), None)
        if dimensions is None:
            raise TypeError('SIZE needs an array')
        result = []
        for bound in cf.markoutercomma(dimensions[1:-1]).split('@,@'):
            parts = bound.split(':')
            if not parts[-1] or parts[-1] == '*':
                return None
            upper = cf._fortran_eval(parts[-1], {}, self)
            lower = cf._fortran_eval(parts[0], {}, self) if len(parts) == 2 else 1
            if not isinstance(lower, int) or not isinstance(upper, int):
                raise TypeError('an array bound is an integer')
            result.append(max(0, int(upper) - int(lower) + 1))
        return result

    def _f2py_storage_inquiries(self, tree, literals, names):
        owner = self
        model = {'bit_size', 'digits', 'epsilon', 'huge', 'maxexponent',
                 'minexponent', 'precision', 'radix', 'range', 'tiny', 'kind'}

        class Inquiries(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if not isinstance(node.func, ast.Name):
                    return node
                intrinsic = node.func.id
                if (intrinsic not in model | {'len', 'size'}
                        or owner._binding(intrinsic, names) is not None):
                    return node
                parameters = (('string', 'kind') if intrinsic == 'len' else
                              ('array', 'dim', 'kind') if intrinsic == 'size' else ('x',))
                if len(node.args) > len(parameters):
                    return node
                arguments = dict(zip(parameters, node.args))
                for keyword in node.keywords:
                    if keyword.arg not in parameters or keyword.arg in arguments:
                        return node
                    arguments[keyword.arg] = keyword.value
                if parameters[0] not in arguments:
                    return node
                reference = owner._reference(arguments[parameters[0]])
                if reference is None:
                    return node
                name, var, element = reference
                argument = arguments[parameters[0]]
                spelling = argument.func.id if element else argument.id
                if owner._binding(spelling, names) == 'local':
                    return node
                if intrinsic in model:
                    typespec, kind = cf._declared_type(name, var, owner)
                    if typespec is None or kind is None:
                        return node
                    value = cf._intrinsics[intrinsic](cf._Unknown(typespec, kind))
                else:
                    requested_kind = (owner._evaluate_node(arguments['kind'], literals, names)
                                      if 'kind' in arguments else None)
                    if intrinsic == 'len':
                        if var.get('typespec') != 'character':
                            return node
                        number = owner._length(name, var)
                    else:
                        if element:
                            return node
                        shape = owner._shape(var)
                        if shape is None:
                            return node
                        if 'dim' in arguments:
                            dim = owner._evaluate_node(arguments['dim'], literals, names)
                            if not isinstance(dim, int) or not 1 <= dim <= len(shape):
                                raise ValueError('SIZE dimension is outside the array rank')
                            number = shape[int(dim) - 1]
                        else:
                            number = math.prod(shape)
                    value = cf._integer(number, cf._kind_arg(requested_kind, 'integer'))
                placeholder = '_storage_inquiry' + str(len(literals))
                while placeholder in literals or placeholder in owner:
                    placeholder += '_'
                literals[placeholder] = value
                return ast.copy_location(ast.Name(placeholder, ast.Load()), node)

        return Inquiries().visit(tree)


def storage_parameters(variables, inherited):
    return _StorageParameters(variables, inherited)
