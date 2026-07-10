import sys
from io import StringIO

from numpy.f2py import capi_maps


def test_real16_binary128_warns():
    capi_maps.reset_binary128_warning()
    var = {'typespec': 'real', 'kindselector': {'kind': '16'}}
    stderr = StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr
    try:
        assert capi_maps.getctype(var) == 'long_double'
    finally:
        sys.stderr = old_stderr
    assert capi_maps.binary128_module_doc_notice().startswith('WARNING:')
    assert 'binary128' in stderr.getvalue()


def test_complex_long_double_capi_map():
    assert capi_maps.c2capi_map["complex_long_double"] == "NPY_CLONGDOUBLE"


def test_complex_long_double_is_distinct():
    assert capi_maps.c2pycode_map["complex_long_double"] != capi_maps.c2pycode_map["complex_double"]
    assert capi_maps.c2capi_map["complex_long_double"] != capi_maps.c2capi_map["complex_double"]
