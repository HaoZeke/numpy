import re

import pytest

import numpy as np
from numpy.f2py import capi_maps, cfuncs


def test_complex_long_double_capi_map():
    assert capi_maps.c2capi_map["complex_long_double"] == "NPY_CLONGDOUBLE"


def test_complex_long_double_is_distinct():
    assert capi_maps.c2pycode_map["complex_long_double"] != capi_maps.c2pycode_map["complex_double"]
    assert capi_maps.c2capi_map["complex_long_double"] != capi_maps.c2capi_map["complex_double"]


def test_long_long_format():
    # gh-21409: %ld reads a C long, which is 32 bits on Windows
    assert capi_maps.cformat_map["long_long"] == "%lld"


@pytest.mark.parametrize("ctype, dtype", [("long_long", np.longlong),
                                          ("npy_intp", np.intp)])
def test_try_pyarr_typecode(ctype, dtype):
    # gh-21409: arrays with this type code are written to as ctype
    code = re.search(rf"TRYPYARRAYTEMPLATE\({ctype},'(.)'\)",
                     cfuncs.cfuncs[f"try_pyarr_from_{ctype}"]).group(1)
    assert np.dtype(code) == dtype
