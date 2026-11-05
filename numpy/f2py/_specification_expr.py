"""Evaluate declaration expressions without changing the public parameter API."""

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass

from . import _compiler_context, _compiler_inquiry, _derived_expr as _values
from ._isocbind import isoc_kindmap

_INTRINSIC_NAMES = {
    "iso_c_binding": frozenset(isoc_kindmap),
    "iso_fortran_env": frozenset(
        (
            "int8",
            "int16",
            "int32",
            "int64",
            "real32",
            "real64",
            "real128",
            "character_kinds",
            "integer_kinds",
            "logical_kinds",
            "real_kinds",
            "numeric_storage_size",
            "character_storage_size",
            "file_storage_size",
            "input_unit",
            "output_unit",
            "error_unit",
            "iostat_end",
            "iostat_eor",
            "iostat_inquire_internal_unit",
            "atomic_int_kind",
            "atomic_logical_kind",
            "stat_stopped_image",
            "stat_locked",
            "stat_locked_other_image",
            "stat_unlocked",
            "stat_failed_image",
            "stat_unassigned",
        )
    ),
}


class UnresolvedSpecification(ValueError):
    """An expression depends on a value supplied by a routine caller."""


_TOKEN = re.compile(
    r"(?P<string>'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")"
    r"|(?P<operator>\.(?:true|false|eq|ne|lt|le|gt|ge|and|or|not|eqv|neqv)\.)"
    r"|(?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[edq][+-]?\d+)?(?:_(?P<kind>[a-z]\w*|\d+))?)"
    r"|(?P<name>[a-z]\w*)",
    re.I,
)

_PROCESSOR_INQUIRIES = frozenset(
    (
        "kind",
        "tiny",
        "huge",
        "epsilon",
        "digits",
        "radix",
        "minexponent",
        "maxexponent",
        "precision",
        "range",
        "selected_real_kind",
        "selected_int_kind",
        "spacing",
        "rrspacing",
        "nearest",
    )
)


def requires_compiler(scope, source, seen=()):
    """Processor inquiries and intrinsic module kinds use the selected compiler."""
    for token in _TOKEN.finditer(source):
        if token.group("string") is not None or token.group("operator") is not None:
            continue
        name = token.group("name") or token.group("kind")
        if name is None or name.isdecimal():
            continue
        binding = scope.resolve(name)
        suffix = source[token.end() :].lstrip()
        if binding is not None and binding.intrinsic_module is not None:
            return True
        if name.lower() in _PROCESSOR_INQUIRIES and suffix.startswith("("):
            if (
                binding is None
                or ConstantProgram.dimensions(binding.scope.declaration(binding.name))
                is None
            ):
                return True
        if binding is None:
            continue
        key = (id(binding.scope), binding.name)
        if key in seen:
            continue
        variable = binding.scope.declaration(binding.name)
        expressions = [variable.get("=", "")]
        expressions.extend(variable.get("kindselector", {}).values())
        if any(
            requires_compiler(binding.scope, expression, (*seen, key))
            for expression in expressions
            if isinstance(expression, str)
        ):
            return True
    return False


def split_outer(source, separator=","):
    """Split declaration separators while preserving strings and parentheses."""
    result, start, depth, index = [], 0, 0, 0
    while index < len(source):
        character = source[index]
        if character in "'\"":
            quote = character
            index += 1
            while index < len(source):
                if source[index] == quote:
                    index += 1
                    if index < len(source) and source[index] == quote:
                        index += 1
                        continue
                    break
                index += 1
            continue
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == separator and depth == 0:
            result.append(source[start:index].strip())
            start = index + 1
        index += 1
    result.append(source[start:].strip())
    return result


@dataclass(frozen=True)
class Binding:
    scope: object
    name: str
    intrinsic_module: str | None = None


class Scope:
    def __init__(self, block, parent, registry, implicit):
        self.variables = copy.deepcopy(block.get("vars", {}))
        self.use = copy.deepcopy(block.get("use", {}))
        self.parent = parent
        self.registry = registry
        self.implicit = copy.deepcopy(implicit)

    def resolve(self, name, seen=()):
        name = name.lower()
        identity = (id(self), name)
        if identity in seen:
            raise UnresolvedSpecification("Cyclic USE association: " + name)
        if name in self.variables:
            variable = self.variables[name]
            access_only = (
                bool(variable.get("attrspec"))
                and set(variable) <= {"attrspec"}
                and set(variable.get("attrspec", ())) <= {"public", "private"}
            )
            if not access_only:
                return Binding(self, name)
        seen = (*seen, identity)
        for module, mapping in self.use.items():
            names = mapping.get("map", {}) if mapping else {}
            only = mapping.get("only", 0) if mapping else 0
            if name in names:
                remote = names[name].lower()
            elif only or name in names.values():
                continue
            else:
                remote = name
            module = module.lower()
            if module in _INTRINSIC_NAMES and remote in _INTRINSIC_NAMES[module]:
                return Binding(self, remote, module)
            owner = self.registry.modules.get(module)
            if owner is None:
                continue
            found = owner.resolve(remote, seen)
            if found is not None:
                variable = found.scope.variables.get(found.name, {})
                attributes = variable.get("attrspec", ())
                default = owner.variables.get("", {}).get("attrspec", ())
                if (
                    "private" in attributes
                    or "private" in default
                    and "public" not in attributes
                ):
                    continue
                return found
        if self.parent is not None:
            return self.parent.resolve(name, seen)
        return None

    def declaration(self, name):
        variable = copy.deepcopy(self.variables[name])
        if "typespec" not in variable:
            for key, value in self.implicit.get(name[0], {}).items():
                variable.setdefault(key, value)
        return variable

    def evaluate(self, source):
        source = source.strip()
        if re.fullmatch(r"[+-]?\d+", source):
            return int(source)
        compiler = _compiler_context.current()
        builder = ConstantProgram(self.registry)
        if not compiler:
            if requires_compiler(self, source):
                return None
            try:
                value = _values._fortran_eval(source, {}, ParameterValues(self))
            except (
                UnresolvedSpecification,
                NameError,
                KeyError,
                TypeError,
                ValueError,
                SyntaxError,
                OverflowError,
                ZeroDivisionError,
            ):
                return None
            if isinstance(value, int) and not isinstance(value, bool):
                return int(value)
            return None
        try:
            expression = builder.expression(self, source)
        except UnresolvedSpecification:
            return None
        return _compiler_inquiry.integer_specification(
            compiler, builder.program(expression)
        )

    def dimension(self, source):
        result = []
        for bound in split_outer(source, ":"):
            if bound in ("", "*"):
                result.append(bound)
                continue
            value = self.evaluate(bound)
            result.append(runtime_integer(self, bound) if value is None else str(value))
        return ":".join(result)

    def fortran_expression(self, source):
        """Retain native operators and reconstruct lexical constant declarations."""
        builder = getattr(self, "native_program", None)
        if builder is None:
            builder = self.native_program = NativeProgram(self.registry)
        return builder.expression(self, source)

    def fortran_extent(self, source):
        bounds = split_outer(source, ":")
        if len(bounds) == 1:
            bounds.insert(0, "1")
        if len(bounds) != 2 or any(bound in ("", "*") for bound in bounds):
            return None
        low, high = [self.fortran_expression(bound) for bound in bounds]
        return f"max(0,({high})-({low})+1)"


class SpecificationScopes:
    def __init__(self, blocks, implicit_rules):
        self.scopes, self.modules = {}, {}

        def collect(block, parent):
            implicit, _ = implicit_rules(block)
            if parent is not None and "implicit" not in block:
                implicit = parent.implicit
            scope = Scope(block, parent, self, implicit)
            self.scopes[id(block)] = scope
            if block.get("block") == "module":
                self.modules[block["name"].lower()] = scope
            for child in block.get("body", ()):
                collect(child, scope)

        for block in blocks:
            collect(block, None)

    def get(self, block):
        return self.scopes.get(id(block))


class ConstantProgram:
    def __init__(self, registry):
        self.names, self.active, self.uses, self.declarations = {}, set(), [], []
        self.prefix = "f2py_spec_"
        occupied = [
            name for scope in registry.scopes.values() for name in scope.variables
        ]
        while any(name.startswith(self.prefix) for name in occupied):
            self.prefix += "x"

    def expression(self, scope, source):
        source = self.inquiries(scope, source)

        def replace(match):
            if match.group("string") is not None or match.group("operator") is not None:
                return match[0]
            if match.group("number") is not None:
                kind = match.group("kind")
                if kind is None or kind.isdecimal():
                    return match[0]
                binding = scope.resolve(kind)
                if binding is None:
                    raise UnresolvedSpecification("Unknown literal kind: " + kind)
                return match[0][: -len(kind)] + self.bind(binding)
            name = match.group("name").lower()
            suffix = source[match.end() :].lstrip()
            if suffix.startswith("=") and not suffix.startswith("=="):
                return match[0]
            binding = scope.resolve(name)
            if suffix.startswith("(") and (
                binding is None
                or binding.intrinsic_module is not None
                or self.dimensions(binding.scope.declaration(binding.name)) is None
            ):
                return match[0]
            if binding is not None:
                return self.bind(binding)
            if name in self.names.values() or name.startswith(self.prefix + "shape_"):
                return match[0]
            if suffix.startswith(("(", "=")):
                return match[0]
            raise UnresolvedSpecification("Runtime or unknown value: " + name)

        return _TOKEN.sub(replace, source)

    def bind(self, binding):
        key = (id(binding.scope), binding.name, binding.intrinsic_module)
        if key in self.active:
            raise UnresolvedSpecification(
                "Cyclic parameter definition: " + binding.name
            )
        if key in self.names:
            return self.names[key]
        name = self.prefix + str(len(self.names))
        self.names[key] = name
        if binding.intrinsic_module is not None:
            self.uses.append(
                f"use, intrinsic :: {binding.intrinsic_module}, "
                f"only: {name} => {binding.name}"
            )
            return name
        variable = binding.scope.declaration(binding.name)
        if "parameter" not in variable.get("attrspec", ()) or "=" not in variable:
            raise UnresolvedSpecification("Runtime value: " + binding.name)
        self.active.add(key)
        try:
            value = self.expression(binding.scope, variable["="])
            type_declaration = self.type_declaration(binding.scope, variable)
            dimension = variable.get("dimension")
            if dimension is None:
                for attribute in variable.get("attrspec", ()):
                    if attribute.startswith("dimension"):
                        dimension = split_outer(
                            attribute[len("dimension") :].strip()[1:-1]
                        )
                        break
            dimensions = (
                ""
                if dimension is None
                else ", dimension("
                + ",".join(self.expression(binding.scope, part) for part in dimension)
                + ")"
            )
            self.declarations.append(
                f"{type_declaration}{dimensions}, parameter :: {name} = {value}"
            )
        finally:
            self.active.remove(key)
        return name

    def type_declaration(self, scope, variable):
        typespec = variable.get("typespec")
        if typespec not in (
            "integer",
            "real",
            "double precision",
            "complex",
            "double complex",
            "logical",
            "character",
            "byte",
        ):
            raise UnresolvedSpecification(
                "Unsupported specification type: " + str(typespec)
            )
        selector = variable.get(
            "charselector" if typespec == "character" else "kindselector", {}
        )
        if "*" in selector:
            return typespec + "*" + self.expression(scope, selector["*"])
        parts = [
            key + "=" + self.expression(scope, value)
            for key, value in selector.items()
            if key in ("kind", "len")
        ]
        return typespec + ("(" + ",".join(parts) + ")" if parts else "")

    def inquiries(self, scope, source):
        output, cursor = [], 0
        for match in _TOKEN.finditer(source):
            if match.start() < cursor or match.group("name") is None:
                continue
            name = match[0].lower()
            if name not in (
                "len",
                "size",
                "shape",
                "lbound",
                "ubound",
                "kind",
                "tiny",
                "huge",
                "epsilon",
                "digits",
                "radix",
                "minexponent",
                "maxexponent",
                "precision",
                "range",
                "bit_size",
            ):
                continue
            callee = scope.resolve(name)
            if callee is not None and callee.intrinsic_module is None:
                declaration = callee.scope.declaration(callee.name)
                if self.dimensions(
                    declaration
                ) is not None or "external" in declaration.get("attrspec", ()):
                    continue
            opening = match.end()
            while opening < len(source) and source[opening].isspace():
                opening += 1
            if opening == len(source) or source[opening] != "(":
                continue
            closing = self.closing(source, opening)
            arguments = split_outer(source[opening + 1 : closing])
            if not arguments or re.fullmatch(r"[a-z]\w*", arguments[0], re.I) is None:
                continue
            binding = scope.resolve(arguments[0])
            if binding is None or binding.intrinsic_module is not None:
                continue
            variable = binding.scope.declaration(binding.name)
            replacement = self.inquiry(
                scope, name, binding.scope, variable, arguments[1:]
            )
            if replacement is None:
                continue
            output.extend((source[cursor : match.start()], replacement))
            cursor = closing + 1
        return "".join(output) + source[cursor:]

    @staticmethod
    def closing(source, opening):
        depth, index = 1, opening + 1
        while index < len(source):
            if source[index] in "'\"":
                quote = source[index]
                index += 1
                while index < len(source):
                    if source[index] == quote:
                        index += 1
                        if index < len(source) and source[index] == quote:
                            index += 1
                            continue
                        break
                    index += 1
                continue
            if source[index] == "(":
                depth += 1
            elif source[index] == ")":
                depth -= 1
                if depth == 0:
                    return index
            index += 1
        raise UnresolvedSpecification("Unbalanced specification inquiry")

    @staticmethod
    def dimensions(variable):
        if "dimension" in variable:
            return list(variable["dimension"])
        for attribute in variable.get("attrspec", ()):
            if attribute.startswith("dimension"):
                return split_outer(attribute[len("dimension") :].strip()[1:-1])
        return None

    def inquiry(self, caller, name, owner, variable, arguments):
        positional, keywords = [], {}
        for argument in arguments:
            assignment = split_outer(argument, "=")
            if len(assignment) == 2:
                keywords[assignment[0].lower()] = self.expression(caller, assignment[1])
            else:
                positional.append(self.expression(caller, argument))
        kind = keywords.get("kind")
        if name == "len":
            if variable.get("typespec") != "character":
                return None
            length = variable.get("charselector", {}).get(
                "len", variable.get("charselector", {}).get("*", "1")
            )
            if length in ("*", ":", "(*)", "(:)"):
                return None
            result = self.expression(owner, length)
            kind = kind or (positional[0] if positional else None)
        elif name in ("size", "shape", "lbound", "ubound"):
            dimensions = self.dimensions(variable)
            if dimensions is None:
                return None
            bounds = []
            for dimension in dimensions:
                parts = split_outer(dimension, ":")
                if len(parts) == 1:
                    parts = ["1", parts[0]]
                if len(parts) != 2 or any(part in ("", "*") for part in parts):
                    raise UnresolvedSpecification("Runtime array shape")
                bounds.append(tuple(self.expression(owner, part) for part in parts))
            values = [
                (
                    f"max(0,({high})-({low})+1)"
                    if name in ("size", "shape")
                    else f"merge(1,({low}),({high})<({low}))"
                    if name == "lbound"
                    else f"merge(0,({high}),({high})<({low}))"
                )
                for low, high in bounds
            ]
            dimension = keywords.get(
                "dim", positional[0] if positional and name != "shape" else None
            )
            if name == "shape" and positional and kind is None:
                kind = positional[0]
            if len(positional) > 1 and kind is None:
                kind = positional[1]
            if kind is not None:
                values = ["int(" + value + ",kind=" + kind + ")" for value in values]
            if name == "size" and dimension is None:
                result = "*".join("(" + value + ")" for value in values)
            else:
                auxiliary = self.prefix + "shape_" + str(len(self.declarations))
                selector = "" if kind is None else "(kind=" + kind + ")"
                self.declarations.append(
                    "integer"
                    + selector
                    + ", parameter :: "
                    + auxiliary
                    + "("
                    + str(len(values))
                    + ") = (/"
                    + ",".join(values)
                    + "/)"
                )
                result = (
                    auxiliary
                    if dimension is None
                    else auxiliary + "(" + dimension + ")"
                )
        else:
            typespec = variable.get("typespec")
            selector = variable.get("kindselector", {})
            selected_kind = selector.get("kind", selector.get("*"))
            selected_kind = (
                None if selected_kind is None else self.expression(owner, selected_kind)
            )
            literals = {
                "integer": "int(0",
                "real": "real(0",
                "complex": "cmplx(0",
                "logical": "logical(.false.",
            }
            if typespec == "double precision":
                literal = "0.0d0"
            elif typespec in literals:
                literal = (
                    literals[typespec]
                    + (",kind=" + selected_kind if selected_kind else "")
                    + ")"
                )
            else:
                return None
            return name + "(" + literal + ")"
        return (
            "(" + result + ")"
            if kind is None
            else "int(" + result + ",kind=" + kind + ")"
        )

    def program(self, expression):
        lines = [
            "program f2py_specification",
            *self.uses,
            "implicit none",
            *self.declarations,
            "integer, parameter :: f2py_integer_check = bit_size(" + expression + ")",
            "integer(kind=kind("
            + expression
            + ")), parameter :: f2py_value = "
            + expression,
            "write(*, '(I0)') f2py_value",
            "end program f2py_specification",
        ]
        output = []
        for line in lines:
            while len(line) > 110:
                split = 110
                if line[split - 1 : split + 1] in ("''", '""'):
                    split -= 1
                output.append(line[:split] + "&")
                line = "&" + line[split:]
            output.append(line)
        return "\n".join(output) + "\n"


class NativeProgram(ConstantProgram):
    """Declaration prolog for wrappers whose shape also needs a C expression."""

    def inquiries(self, scope, source):
        return source

    def bind(self, binding):
        if binding.intrinsic_module is None:
            variable = binding.scope.declaration(binding.name)
            if "parameter" not in variable.get("attrspec", ()):
                return binding.name
        return super().bind(binding)


class ParameterValues(Mapping):
    """Lazy typed values for generation without a selected compiler context."""

    def __init__(self, scope, cache=None, active=None):
        self.scope = scope
        self.cache = {} if cache is None else cache
        self.active = set() if active is None else active

    def __getitem__(self, name):
        binding = self.scope.resolve(name)
        if binding is None or binding.intrinsic_module is not None:
            raise KeyError(name)
        key = (id(binding.scope), binding.name)
        if key in self.cache:
            return self.cache[key]
        if key in self.active:
            raise UnresolvedSpecification("Cyclic parameter definition: " + name)
        variable = binding.scope.declaration(binding.name)
        if "parameter" not in variable.get("attrspec", ()) or "=" not in variable:
            raise KeyError(name)
        if requires_compiler(binding.scope, variable["="]):
            raise UnresolvedSpecification("A processor inquiry requires a compiler")
        self.active.add(key)
        try:
            values = ParameterValues(binding.scope, self.cache, self.active)
            value = _values._fortran_eval(variable["="], {}, values)
            typespec = variable.get("typespec")
            kind = variable.get("kindselector", {}).get(
                "kind", variable.get("kindselector", {}).get("*")
            )
            kind = (
                None if kind is None else int(_values._fortran_eval(kind, {}, values))
            )

            def convert(element):
                if typespec == "integer":
                    return _values._intrinsics["int"](element, kind=kind)
                if typespec in ("real", "double precision"):
                    return _values._intrinsics["real"](
                        element, kind=8 if typespec == "double precision" else kind
                    )
                return element

            if isinstance(value, list):
                dimensions = ConstantProgram.dimensions(variable)
                if dimensions is None or len(dimensions) != 1:
                    raise UnresolvedSpecification(
                        "A selected compiler is required for this parameter array"
                    )
                bounds = split_outer(dimensions[0], ":")
                lower = (
                    1
                    if len(bounds) == 1
                    else int(_values._fortran_eval(bounds[0], {}, values))
                )
                value = _values._ParameterArray(
                    {
                        index + lower: convert(element)
                        for index, element in enumerate(value)
                    }
                )
            else:
                value = convert(value)
            self.cache[key] = value
            return value
        finally:
            self.active.remove(key)

    def __iter__(self):
        return iter(self.scope.variables)

    def __len__(self):
        return len(self.scope.variables)


def runtime_integer(scope, source):
    from ._runtime_specification import translate

    return translate(scope, source, split_outer, ConstantProgram.dimensions)
