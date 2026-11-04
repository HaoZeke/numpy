"""Array constructors keep their element delimiters inside entity declarations."""

import pytest

from numpy.f2py.crackfortran import markoutercomma


@pytest.mark.parametrize('declaration, expected', [
    ('a=[1, 2, 3], b=4', 'a=[1, 2, 3]@,@ b=4'),
    ('a=reshape([1,2,3,4], [2,2]), b=5',
     'a=reshape([1,2,3,4], [2,2])@,@ b=5'),
    ('a=[integer :: 1, 2], b=3', 'a=[integer :: 1, 2]@,@ b=3'),
    ('a=[(i, i=1,3)], b=4', 'a=[(i, i=1,3)]@,@ b=4'),
    ("names=['a,b', '[x]'], flag=0", "names=['a,b', '[x]']@,@ flag=0"),
    ('a[1, *], b', 'a[1, *]@,@ b'),
])
def test_entity_commas_preserve_bracketed_values(declaration, expected):
    assert markoutercomma(declaration) == expected


@pytest.mark.parametrize('declaration, expected', [
    ('a=[1, 2] b=[3, 4]', 'a=[1, 2]@ @b=[3, 4]'),
    ("names=['x y', '[ z ]'] flag=0", "names=['x y', '[ z ]']@ @flag=0"),
])
def test_entity_spaces_preserve_bracketed_values(declaration, expected):
    assert markoutercomma(declaration, comma=' ') == expected
