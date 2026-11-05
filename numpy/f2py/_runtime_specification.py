"""Translate integer declaration operations without algebraic reassociation."""

import ast
import re


class RuntimeSpecificationError(ValueError):
    """A declaration has no scalar integer C representation."""


def _call(name, *arguments):
    return ast.Call(ast.Name(name, ast.Load()), list(arguments), [])


class IntegerExpression(ast.NodeTransformer):
    def __init__(self, scope, source, split_outer, dimensions):
        self.scope = scope
        self.source = source
        self.split_outer = split_outer
        self.dimensions = dimensions

    def declaration(self, name):
        binding = self.scope.resolve(name)
        if binding is None or binding.intrinsic_module is not None:
            raise RuntimeSpecificationError(name)
        return binding, binding.scope.declaration(binding.name)

    def visit_Name(self, node):
        binding, variable = self.declaration(node.id)
        if variable.get("typespec") not in ("integer", "logical"):
            raise RuntimeSpecificationError(node.id)
        if self.dimensions(variable) is not None:
            raise RuntimeSpecificationError("An array value needs subscripts")
        if "parameter" in variable.get("attrspec", ()):
            value = self.scope.evaluate(node.id)
            if value is None:
                raise RuntimeSpecificationError(node.id)
            return ast.Constant(value)
        # USE/host parameters are folded; runtime actuals retain the local name.
        return node

    def visit_Constant(self, node):
        if type(node.value) not in (int, bool):
            raise RuntimeSpecificationError(str(node.value))
        return ast.Constant(int(node.value))

    def visit_BinOp(self, node):
        operations = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
            ast.Pow: "pow",
        }
        operation = operations.get(type(node.op))
        if operation is None:
            raise RuntimeSpecificationError(type(node.op).__name__)
        return _call(
            "f2py_spec_" + operation, self.visit(node.left), self.visit(node.right)
        )

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return _call("f2py_spec_sub", ast.Constant(0), operand)
        if isinstance(node.op, ast.Not):
            return _call("f2py_spec_not", operand)
        raise RuntimeSpecificationError(type(node.op).__name__)

    def visit_Compare(self, node):
        original = ast.get_source_segment(self.source, node) or ast.unparse(node)
        value = self.scope.evaluate("merge(1,0,(" + original + "))")
        if value is not None:
            return ast.Constant(value)
        if len(node.ops) != 1:
            raise RuntimeSpecificationError("Chained comparison")
        names = {
            ast.Eq: "eq",
            ast.NotEq: "ne",
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
        }
        operation = names.get(type(node.ops[0]))
        if operation is None:
            raise RuntimeSpecificationError("Comparison operator")
        return _call(
            "f2py_spec_" + operation,
            self.visit(node.left),
            self.visit(node.comparators[0]),
        )

    def visit_BoolOp(self, node):
        name = "and" if isinstance(node.op, ast.And) else "or"
        values = [self.visit(value) for value in node.values]
        result = values[0]
        for value in values[1:]:
            result = _call("f2py_spec_" + name, result, value)
        return result

    def arguments(self, node, names, required):
        if len(node.args) > len(names):
            raise RuntimeSpecificationError("Too many intrinsic arguments")
        result = dict(zip(names, node.args))
        for keyword in node.keywords:
            if keyword.arg not in names or keyword.arg in result:
                raise RuntimeSpecificationError("Invalid intrinsic keyword")
            result[keyword.arg] = keyword.value
        if any(name not in result for name in names[:required]):
            raise RuntimeSpecificationError("Missing intrinsic argument")
        if "kind" in result:
            value = self.scope.evaluate(ast.unparse(result["kind"]))
            if value is None or value <= 0:
                raise RuntimeSpecificationError("An intrinsic KIND must be constant")
        return result

    def array(self, node):
        if not isinstance(node, ast.Name):
            raise RuntimeSpecificationError("An inquiry needs a whole array")
        binding, variable = self.declaration(node.id)
        dimensions = self.dimensions(variable)
        if dimensions is None:
            raise RuntimeSpecificationError("An inquiry needs an array")
        return binding, dimensions

    def bound(self, name, array, dimension):
        binding, dimensions = self.array(array)
        if dimension is None:
            raise RuntimeSpecificationError("A bound vector is not a scalar")
        values = []
        for axis, declaration in enumerate(dimensions, 1):
            bounds = self.split_outer(declaration, ":")
            lower = bounds[0] if len(bounds) == 2 and bounds[0] else "1"
            if len(bounds) > 2:
                raise RuntimeSpecificationError("Invalid array bounds")
            if binding.scope is not self.scope:
                value = binding.scope.evaluate(lower)
                if value is None:
                    raise RuntimeSpecificationError("A host bound must be constant")
                low = ast.Constant(value)
            else:
                low = self.visit(ast.parse(lower, mode="eval").body)
            extent = _call("f2py_spec_size", array, ast.Constant(axis))
            values.append(_call("f2py_spec_" + name, low, extent))
        # Every variadic argument has exactly npy_intp type, including literals.
        return _call(
            "f2py_spec_select",
            self.visit(dimension),
            ast.Constant(len(values)),
            *[_call("f2py_spec_cast", value) for value in values],
        )

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise RuntimeSpecificationError("A declaration call needs a name")
        name = node.func.id.lower()
        binding = self.scope.resolve(name)
        external = (
            binding is not None
            and binding.intrinsic_module is None
            and "external"
            in binding.scope.declaration(binding.name).get("attrspec", ())
        )
        # Closed integer calls, including processor inquiries, use the compiler.
        integer_calls = {
            "int",
            "nint",
            "ceiling",
            "floor",
            "kind",
            "digits",
            "radix",
            "minexponent",
            "maxexponent",
            "precision",
            "range",
            "selected_int_kind",
            "selected_real_kind",
            "bit_size",
            "len",
            "len_trim",
            "size",
            "lbound",
            "ubound",
            "mod",
            "modulo",
            "abs",
            "sign",
            "dim",
            "min",
            "max",
            "merge",
            "index",
            "scan",
            "verify",
            "count",
            "sum",
            "product",
            "maxval",
            "minval",
        }
        if name in integer_calls and not external:
            original = ast.get_source_segment(self.source, node) or ast.unparse(node)
            value = self.scope.evaluate(original)
            if value is not None:
                return ast.Constant(value)
        if binding is not None and binding.intrinsic_module is None:
            variable = binding.scope.declaration(binding.name)
            dimensions = self.dimensions(variable)
            if dimensions is not None:
                if variable.get("typespec") != "integer" or node.keywords:
                    raise RuntimeSpecificationError("An INTEGER array needs subscripts")
                if len(node.args) != len(dimensions):
                    raise RuntimeSpecificationError("Array subscript rank differs")
                indexes = []
                for argument, declaration in zip(node.args, dimensions):
                    bounds = self.split_outer(declaration, ":")
                    low = bounds[0] if len(bounds) == 2 and bounds[0] else "1"
                    lower = self.visit(ast.parse(low, mode="eval").body)
                    indexes.append(
                        _call(
                            "f2py_spec_cast",
                            _call("f2py_spec_sub", self.visit(argument), lower),
                        )
                    )
                return _call("f2py_spec_item", ast.Name(name, ast.Load()), *indexes)
            if "external" in variable.get("attrspec", ()):
                raise RuntimeSpecificationError(
                    "A specification function needs a wrapper"
                )
        if name in ("size", "lbound", "ubound"):
            arguments = self.arguments(node, ("array", "dim", "kind"), 1)
            self.array(arguments["array"])
            dimension = arguments.get("dim")
            if name == "size":
                return _call(
                    "f2py_spec_size",
                    arguments["array"],
                    ast.Constant(0) if dimension is None else self.visit(dimension),
                )
            return self.bound(name, arguments["array"], dimension)
        if name in ("len", "len_trim"):
            arguments = self.arguments(node, ("string", "kind"), 1)
            string = arguments["string"]
            if not isinstance(string, ast.Name):
                raise RuntimeSpecificationError("A runtime string needs a dummy name")
            _, variable = self.declaration(string.id)
            if variable.get("typespec") != "character":
                raise RuntimeSpecificationError("LEN needs CHARACTER")
            length = _call("f2py_slen", string)
            return (
                length if name == "len" else _call("f2py_spec_len_trim", string, length)
            )
        if name in ("min", "max") and not node.keywords and len(node.args) >= 2:
            values = [self.visit(argument) for argument in node.args]
            result = values[0]
            for value in values[1:]:
                result = _call("f2py_spec_" + name, result, value)
            return result
        signatures = {
            "mod": ("a", "p"),
            "modulo": ("a", "p"),
            "abs": ("a",),
            "sign": ("a", "b"),
            "dim": ("x", "y"),
            "merge": ("tsource", "fsource", "mask"),
        }
        if name in signatures:
            names = signatures[name]
            arguments = self.arguments(node, names, len(names))
            return _call(
                "f2py_spec_" + name, *[self.visit(arguments[key]) for key in names]
            )
        if name == "int":
            arguments = self.arguments(node, ("a", "kind"), 1)
            return self.visit(arguments["a"])
        raise RuntimeSpecificationError("Unsupported runtime intrinsic: " + name)

    def generic_visit(self, node):
        if not isinstance(node, ast.Expression):
            raise RuntimeSpecificationError(type(node).__name__)
        return super().generic_visit(node)


def translate(scope, source, split_outer, dimensions):
    """Return an opaque C integer expression or the original signature syntax."""
    original = source
    # Literal suffixes select Fortran types; arithmetic is checked at npy_intp width.
    source = re.sub(r"\b(\d+)_(?:\d+|[a-z]\w*)\b", r"\1", source, flags=re.I)
    replacements = {
        "true": "True",
        "false": "False",
        "eq": "==",
        "ne": "!=",
        "lt": "<",
        "le": "<=",
        "gt": ">",
        "ge": ">=",
        "and": "and",
        "or": "or",
        "not": "not",
    }
    source = re.sub(
        r"\.(true|false|eq|ne|lt|le|gt|ge|and|or|not)\.",
        lambda match: " " + replacements[match[1].lower()] + " ",
        source,
        flags=re.I,
    ).strip()
    try:
        tree = ast.parse(source, mode="eval")
        tree = IntegerExpression(scope, source, split_outer, dimensions).visit(tree)
    except (RuntimeSpecificationError, SyntaxError, TypeError, ValueError):
        return original
    return ast.unparse(ast.fix_missing_locations(tree))
