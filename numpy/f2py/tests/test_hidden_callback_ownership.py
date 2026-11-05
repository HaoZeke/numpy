import gc
import sys

import pytest

from . import util


class TestHiddenCallbackOwnership(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh26681.f90")]
    options = ["--lower"]

    def test_fortran_callback_exception_releases_attribute_reference(self):
        callback = self.module.utils.my_abort
        self.module.mypy_abort = callback
        expected = sys.getrefcount(callback)
        try:
            for _ in range(5):
                with pytest.raises(ValueError, match="helpme"):
                    self.module.utils.do_something("helpme")
            gc.collect()
            assert sys.getrefcount(callback) == expected
        finally:
            del self.module.mypy_abort

    def test_python_callback_exception_releases_extra_arguments(self):
        payload = object()
        extra_args = (payload,)

        def callback(message, value):
            assert value is payload
            raise ValueError("callback rejected")

        self.module.mypy_abort = callback
        self.module.mypy_abort_extra_args = extra_args
        callback_references = sys.getrefcount(callback)
        tuple_references = sys.getrefcount(extra_args)
        payload_references = sys.getrefcount(payload)
        try:
            for _ in range(5):
                with pytest.raises(ValueError, match="callback rejected"):
                    self.module.utils.do_something("helpme")
            gc.collect()
            assert sys.getrefcount(callback) == callback_references
            assert sys.getrefcount(extra_args) == tuple_references
            assert sys.getrefcount(payload) == payload_references
        finally:
            del self.module.mypy_abort_extra_args
            del self.module.mypy_abort

    def test_extra_argument_conversion_failure_releases_callback(self):
        class InvalidArguments:
            def __iter__(self):
                raise ValueError("cannot iterate")

        def callback(message):
            raise AssertionError("callback must not run")

        self.module.mypy_abort = callback
        self.module.mypy_abort_extra_args = InvalidArguments()
        expected = sys.getrefcount(callback)
        try:
            for _ in range(5):
                with pytest.raises(self.module.error, match="Failed to convert"):
                    self.module.utils.do_something("helpme")
            gc.collect()
            assert sys.getrefcount(callback) == expected
        finally:
            del self.module.mypy_abort_extra_args
            del self.module.mypy_abort

    def test_invalid_callback_releases_attribute_reference(self):
        callback = object()
        self.module.mypy_abort = callback
        expected = sys.getrefcount(callback)
        try:
            for _ in range(5):
                with pytest.raises(self.module.error, match="processing argument list"):
                    self.module.utils.do_something("helpme")
            gc.collect()
            assert sys.getrefcount(callback) == expected
        finally:
            del self.module.mypy_abort
