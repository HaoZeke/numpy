import pytest

from numpy.f2py import crackfortran


def _rewrite(text, names):
    saved = crackfortran.quiet
    crackfortran.quiet = 1
    try:
        vars_dict = {name: {'typespec': 'character'} for name in names}
        item = ('callstatement', text)
        parents = [('', {'vars': vars_dict}), ('f2pyenhancements', {})]
        _key, value = crackfortran.character_backward_compatibility_hook(
            item, parents, None)
        return value
    finally:
        crackfortran.quiet = saved


# src, character names, base emission, fixed emission
_CASES = [
    ('c = 81', ('c',), '&c = 81', 'c = 81'),
    ('c = d', ('c', 'd'), '&c = &d', 'c = &d'),
    ('foo(c); c = 81', ('c',), 'foo(&c); &c = 81', 'foo(&c); c = 81'),
    ('*c = 81', ('c',), '*&c = 81', '*&c = 81'),
    ('* c = 81', ('c',), '* &c = 81', '* &c = 81'),
    ('*(c) = 81', ('c',), '*(&c) = 81', '*(&c) = 81'),
    ('foo(c)', ('c',), 'foo(&c)', 'foo(&c)'),
    ('(*f2py_func)(x, y, z)', ('x', 'y'),
     '(*f2py_func)(&x, &y, z)', '(*f2py_func)(&x, &y, z)'),
    ('(*f2py_func)(&x, &y, z)', ('x', 'y'),
     '(*f2py_func)(&x, &y, z)', '(*f2py_func)(&x, &y, z)'),
    ('c == d', ('c', 'd'), '&c == &d', '&c == &d'),
    ('c <= d', ('c', 'd'), '&c <= &d', '&c <= &d'),
    ('char *p = c', ('c',), 'char *p = &c', 'char *p = &c'),
    ('char *p = (char *)c', ('c',),
     'char *p = (char *)&c', 'char *p = (char *)&c'),
    ('string p = (string)c', ('c',),
     'string p = (string)&c', 'string p = (string)&c'),
    ('unsigned char t = *c', ('c',),
     'unsigned char t = *&c', 'unsigned char t = *&c'),
    ('size_t n = strlen(c)', ('c',),
     'size_t n = strlen(&c)', 'size_t n = strlen(&c)'),
    ('{ char *p = c; *p = 81; }', ('c',),
     '{ char *p = &c; *p = 81; }', '{ char *p = &c; *p = 81; }'),
    ('{ int n = *c; }', ('c',), '{ int n = *&c; }', '{ int n = *&c; }'),
    ('for (char *p = (char *)c; p; ++p)', ('c',),
     'for (char *p = (char *)&c; p; ++p)',
     'for (char *p = (char *)&c; p; ++p)'),
    ('(void (*)(int))c; foo(d)', ('c', 'd'),
     '(void (*)(int))&c; foo(&d)', '(void (*)(int))&c; foo(&d)'),
    ('flag = seen(c, "pre" "suf")', ('c',),
     'flag = seen(&c, "pre" "suf")', 'flag = seen(&c, "pre" "suf")'),
    ('foo(c, L"hi")', ('c',), 'foo(&c, L"hi")', 'foo(&c, L"hi")'),
    ('n = foo(c, (int (*)[2])d)', ('c', 'd'),
     'n = foo(&c, (int (*)[2])&d)', 'n = foo(&c, (int (*)[2])&d)'),
    ('flag = __builtin_expect(d&&e, 0)', ('d', 'e'),
     'flag = __builtin_expect(&d&&e, 0)',
     'flag = __builtin_expect(&d&&e, 0)'),
    ('flag = __builtin_expect(d && e, 0)', ('d', 'e'),
     'flag = __builtin_expect(&d && &e, 0)',
     'flag = __builtin_expect(&d && &e, 0)'),
    ('foo(c); c = (int (*)(char))d;', ('c', 'd'),
     'foo(&c); &c = (int (*)(char))&d;',
     'foo(&c); c = (int (*)(char))&d;'),
    ('c += 1', ('c',), '&c += 1', '&c += 1'),
    ("flag = (*c == 'a')", ('c',),
     "flag = (*&c == 'a')", "flag = (*&c == 'a')"),
    ('c = (char *)(d + e)', ('c', 'd', 'e'),
     '&c = (char *)(&d + &e)', 'c = (char *)(&d + &e)'),
    ('n = 1.0f', ('f',), 'n = 1.0f', 'n = 1.0f'),
    ('call(c, 1e-3, d)', ('c', 'd', 'e'),
     'call(&c, 1e-3, &d)', 'call(&c, 1e-3, &d)'),
    ('* /*x*/ c = 81', ('c',), '* /*x*/ &c = 81', '* /*x*/ &c = 81'),
    ('c = (int)d', ('c', 'd'), '&c = (int)&d', 'c = (int)&d'),
    ('flag = (c && d)', ('c', 'd'),
     'flag = (&c && &d)', 'flag = (&c && &d)'),
    ('foo((char *)d + e)', ('d', 'e'),
     'foo((char *)&d + &e)', 'foo((char *)&d + &e)'),
    ('foo(d << 1)', ('d',), 'foo(&d << 1)', 'foo(&d << 1)'),
    ('c = d << 1', ('c', 'd'), '&c = &d << 1', 'c = &d << 1'),
    ('flag = !(c)', ('c',), 'flag = !(&c)', 'flag = !(&c)'),
    ('char buf[2] = {c, 0}', ('c',),
     'char buf[2] = {&c, 0}', 'char buf[2] = {&c, 0}'),
    ('{ int n = c; foo(c); }', ('c',),
     '{ int n = &c; foo(&c); }', '{ int n = &c; foo(&c); }'),
    ('*(char *)c = 81', ('c',),
     '*(char *)&c = 81', '*(char *)&c = 81'),
    ('* // comment\nc = 81', ('c',),
     '* // comment\n&c = 81', '* // comment\n&c = 81'),
    ('c =\\\n= c', ('c',),
     '&c =\\\n= &c', '&c =\\\n= &c'),
    ('c //z\n/*x*/ "a" \'b\' \\\n= 81; /*c*/ = 81', ('c',),
     '&c //z\n/*x*/ "a" \'b\' \\\n= 81; /*&c*/ = 81',
     'c //z\n/*x*/ "a" \'b\' \\\n= 81; /*&c*/ = 81'),
]


@pytest.mark.parametrize(('src', 'names', 'base', 'fixed'), _CASES)
def test_callstatement_assignment(src, names, base, fixed):
    got = _rewrite(src, names)
    assert got == fixed, (base, got)
