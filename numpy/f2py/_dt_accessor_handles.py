"""Expand generated component operations over owned and module-bound objects."""

import re

from ._dt_handles import _handle_description, _handle_operation_parts
from ._dt_helpers import _can_wrap_opaque, _enumerate_specializations


def _procedure_parts(lines):
    """Separate declarations of one generated interoperable procedure."""
    declarations = []
    continued = False
    for index, line in enumerate(lines[1:-1], 1):
        text = line.strip().lower()
        if continued or not text or text.startswith('!'):
            declarations.append(line)
            continued = line.rstrip().endswith('&')
            continue
        if re.match(r'^(?:type\s*\(|class\s*\(|integer\b|real\b|complex\b|'
                    r'logical\b|character\b|implicit\b|external\b|intrinsic\b|'
                    r'procedure\b|import\b|parameter\b|dimension\b|save\b|'
                    r'allocatable\b|pointer\b|target\b)', text):
            declarations.append(line)
            continued = line.rstrip().endswith('&')
            continue
        return declarations, lines[index:-1]
    return declarations, []


def _box_accessors(source, type_map):
    """Dispatch only generated accessors with an explicit PDT unwrap site."""
    descriptions = {}
    for block in type_map.values():
        if not _can_wrap_opaque(block, type_map):
            continue
        for kinds, _ in _enumerate_specializations(block) or [(None, '')]:
            description = _handle_description(block, kind_override=kinds)
            if description is not None:
                descriptions[description['prefix']] = description
    starts = re.compile(r'(?im)^[ \t]*(?:function|subroutine)[ \t]+'
                        r'(?P<name>f2py_(?:get|set|clear|resize)_\w+)\b[^\n]*')
    replacements = []
    for match in starts.finditer(source):
        name = match['name']
        end = re.search(r'(?im)^[ \t]*end[ \t]+(?:function|subroutine)[ \t]+'
                        + re.escape(name) + r'[ \t]*(?:\n|$)', source[match.end():])
        if end is None:
            raise ValueError('unterminated generated component procedure')
        stop = match.end() + end.end()
        lines = source[match.start():stop].splitlines()
        # An explicit expansion from a specialized generator owns its body.
        if any('=> f2py_pdt_header(' in line.lower() for line in lines):
            continue
        declarations, statements = _procedure_parts(lines)
        pointers = {}
        body = []
        for line in statements:
            cast = re.fullmatch(r'\s*(\w+)\s*=>\s*(\w+)_unwrap\((.*)\)\s*', line, re.I)
            if cast is None:
                body.append(line)
                continue
            prefix = cast[2].lower()
            description = descriptions.get(prefix)
            if description is None:
                raise ValueError('PDT accessor has no matching native specialization')
            local = cast[1]
            if local in pointers:
                raise ValueError('generated accessor reconstructs one local twice')
            pointers[local] = dict(handle=cast[3], description=description)
        if not pointers:
            continue
        writes = {'obj'} if name.lower().startswith(
            ('f2py_set_', 'f2py_clear_', 'f2py_resize_')) else set()
        uses, declarations, body = _handle_operation_parts(
            [], declarations, body, pointers, write_targets=writes)
        rendered = '\n'.join([lines[0], *uses, *declarations, *body, lines[-1], ''])
        replacements.append((match.start(), stop, rendered))
    for start, stop, replacement in reversed(replacements):
        source = source[:start] + replacement + source[stop:]
    return source
