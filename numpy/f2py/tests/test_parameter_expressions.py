import re

import pytest

import numpy as np
from numpy.f2py import crackfortran, f90mod_rules
from numpy.testing import assert_array_equal

from . import util

SOURCE = util.getpath("tests", "src", "modules", "gh22511_integer_extents.f90")


@pytest.mark.parametrize("expression, expected", [
    ("4 / 2", 2),
    ("5 / 2 * 2", 4),
    ("-5 / 2 * 2", -4),
    ("5 / -2 * 2", -4),
    ("7 / 2 / 2", 1),
    ("5 / 2.0 * 2", 5),
    ("2 ** (-2)", 0),
    ("(-1) ** (-3)", -1),
    ("5_8 / 2_8 * 2", 4),
    ("2.9", 2),
    ("9.9d0", 9),
    ("kind(0.0d0)", 8),
    ("len(\x27kind(0.0)\x27)", 9),
    ("((2_8**62) + 1_8) / 3_8", 1537228672809129301),
])
def test_integer_parameter_expression_types(expression, expected):
    variable = dict(typespec="integer", attrspec=["parameter"], kindselector=dict(kind="8"))
    variable["="] = expression
    result = crackfortran.get_parameters(dict(n=variable))
    assert type(result["n"]) is int
    assert result["n"] == expected


def test_integer_parameter_length_normalization():
    module, = crackfortran.crackfortran([str(SOURCE)])
    variables = module["vars"]
    assert variables["n_exact"]["="] == "2"
    assert variables["n_round"]["="] == "4"
    assert variables["n_negative"]["="] == "-4"
    assert variables["word"]["charselector"]["len"] == "2"
    assert variables["rounded"]["charselector"]["len"] == "4"
    assert variables["direct_word"]["charselector"]["len"] == "2"
    assert variables["direct_vector"]["dimension"] == ["4"]


@pytest.mark.parametrize("expression", ["2.0", "2.5"])
def test_real_extent_is_not_an_integer_declaration(expression):
    variable = dict(typespec="character", attrspec=["parameter"], charselector=dict(len=expression))
    variable["="] = "'ab'"
    module = dict(block="module", name="invalid_length", vars=dict(word=variable), body=[])
    with pytest.raises(ValueError, match="Cannot determine module PARAMETER storage extent"):
        f90mod_rules.buildhooks(module)


def character_module(dimensions, length):
    variable = dict(typespec="character", attrspec=["parameter"],
                    dimension=[str(value) for value in dimensions], charselector=dict(len=str(length)))
    variable["="] = "'a'"
    return dict(block="module", name="large_constants", vars=dict(words=variable), body=[])


def test_character_storage_count_is_a_wide_literal():
    module = character_module([50000], 50000)
    if np.iinfo(np.intp).max < 2500000000:
        with pytest.raises(ValueError, match="storage size exceeds"):
            f90mod_rules.buildhooks(module)
    else:
        hooks, _ = f90mod_rules.buildhooks(module)
        generated = "\n".join(hooks["f90modhooks"])
        match = re.search(r"static char [a-zA-Z0-9_]+\[\((\d+)\) >", generated)
        assert match is not None
        assert int(match.group(1)) == 2500000000
        assert "(50000)*(50000)" not in generated
        assert "sizeof(" in generated


@pytest.mark.parametrize("dimensions, length", [
    ([int(np.iinfo(np.intp).max)], 2),
    ([int(np.iinfo(np.intp).max), 2], 1),
])
def test_character_storage_size_overflow_is_rejected(dimensions, length):
    module = character_module(dimensions, length)
    with pytest.raises(ValueError, match="storage size exceeds"):
        f90mod_rules.buildhooks(module)


def test_character_length_must_fit_descriptor():
    module = character_module([0], int(np.iinfo(np.intc).max) + 1)
    with pytest.raises(ValueError, match="extent exceeds"):
        f90mod_rules.buildhooks(module)


def test_zero_character_storage_has_a_zero_copy_bound():
    module = character_module([0, 50000], 50000)
    hooks, _ = f90mod_rules.buildhooks(module)
    generated = "\n".join(hooks["f90modhooks"])
    assert "[(0) > 0 ? (0) : 1]" in generated
    assert "< ((0) > 0 ? sizeof(" in generated


@pytest.mark.slow
class TestIntegerParameterExtents(util.F2PyTest):
    sources = [SOURCE]

    def test_integer_parameters_preserve_lengths_and_values(self):
        module = self.module.integer_extents
        assert int(module.n_exact) == 2
        assert int(module.n_round) == 4
        assert int(module.n_negative) == -4
        assert int(module.n_mixed) == 5
        assert int(module.n_from_real) == 2
        assert int(module.n_from_double) == 9
        assert int(module.n_text) == 9
        assert int(module.n_power) == 0
        assert int(module.n_large) == 1537228672809129301
        assert module.word == b"ab"
        assert module.direct_word == b"ab"
        assert module.rounded == b"abcd"
        assert module.mixed == b"abcde"
        assert module.converted == b"xy"
        assert module.double_word == b"abcdefghi"
        assert module.text_word == b"abcdefghi"
        assert_array_equal(module.vector, [1, 2, 3, 4])
        assert_array_equal(module.direct_vector, [1, 2, 3, 4])
