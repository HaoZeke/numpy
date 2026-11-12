import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestCommonBlock(util.F2PyTest):
    sources = [util.getpath("tests", "src", "common", "block.f")]

    def test_common_block(self):
        self.module.initcb()
        assert self.module.block.long_bn == np.array(1.0, dtype=np.float64)
        assert self.module.block.string_bn == np.array("2", dtype="|S1")
        assert self.module.block.ok == np.array(3, dtype=np.int32)


@pytest.mark.slow
class TestCommonWithUse(util.F2PyTest):
    sources = [util.getpath("tests", "src", "common", "gh19161.f90")]

    def test_common_gh19161(self):
        assert self.module.data.x == 0


@pytest.mark.slow
class TestCommonCharacterAssignment(util.F2PyTest):
    # gh-9370: assigning to a CHARACTER*(3) member must use the declared
    # length; GUARD follows BLOCK_STRING in the common block storage
    sources = [util.getpath("tests", "src", "common", "gh9370.f")]

    def test_bytes_from_intent_out(self):
        self.module.initblock()
        self.module.block.block_string = self.module.foo()
        assert self.module.block.block_string.tobytes() == b"Foo"
        assert self.module.block.guard.tobytes() == b"xyz"

    @pytest.mark.parametrize("value, expected", [
        (b"Fo", b"Fo\x00"),
        (b"FooBar", b"Foo"),
        ("bar", b"bar"),
        (np.array(b"Fo"), b"Fo\x00"),
        (np.array(b"FooBar"), b"Foo"),
    ], ids=["short", "long", "str", "short-array", "long-array"])
    def test_declared_length(self, value, expected):
        self.module.initblock()
        self.module.block.block_string = value
        assert self.module.block.block_string.dtype == np.dtype("S3")
        assert self.module.block.block_string.tobytes() == expected
        assert self.module.block.guard.tobytes() == b"xyz"
