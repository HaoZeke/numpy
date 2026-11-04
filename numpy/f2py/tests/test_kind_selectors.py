"""Kind selectors go through crackfortran's parameter evaluator."""
import textwrap

from numpy.f2py import capi_maps, crackfortran

# Kind numbers the parameter evaluator returns for the compilers f2py maps.
# Default real is 4, double precision and real64 are 8, and the quad
# parameter in these tests is 16.
_KIND_SINGLE = 4
_KIND_DOUBLE = 8
_KIND_QUAD = 16


def _modvars(tmp_path, src, name):
    fpath = tmp_path / name
    fpath.write_text(src)
    return crackfortran.crackfortran([str(fpath)])[0]['vars']


def test_kind_suffix_uses_the_parameter():
    # kind(1.0_qp) is the value of qp. A letter in the name is not a
    # guess, so dp = 4 is kind 4 and dp = 8 is kind 8.
    par = {'typespec': 'integer', 'attrspec': ['parameter']}
    base = {
        'qp': {**par, '=': str(_KIND_QUAD)},
        'dp4': {**par, '=': str(_KIND_SINGLE)},
        'dp8': {**par, '=': str(_KIND_DOUBLE)},
        'kq': {**par, '=': 'kind(1.0_qp)'},
        'k4': {**par, '=': 'kind(1.0_dp4)'},
        'k8': {**par, '=': 'kind(1.0_dp8)'},
        'kd': {**par, '=': 'kind(1.0d0)'},
        'ks': {**par, '=': 'kind(1.0)'},
        'ki': {**par, '=': 'kind(0)'},
    }
    got = crackfortran.get_parameters(base)
    assert got['kq'] == _KIND_QUAD
    assert got['k4'] == _KIND_SINGLE
    assert got['k8'] == _KIND_DOUBLE
    assert got['kd'] == _KIND_DOUBLE
    assert got['ks'] == _KIND_SINGLE
    assert got['ki'] == _KIND_SINGLE


def test_module_kind_selectors_resolve(tmp_path):
    src = textwrap.dedent(f"""\
        module kindsel
          integer, parameter :: qp = {_KIND_QUAD}
          integer, parameter :: dp = {_KIND_SINGLE}
          real(kind=kind(1.0_qp)) :: wide
          real(kind=kind(1.0_dp)) :: narrow
          real(kind=kind(1.0d0)) :: dbl
        end module kindsel
        """)
    vs = _modvars(tmp_path, src, 'kindsel.f90')
    assert vs['wide']['kindselector']['kind'] == str(_KIND_QUAD)
    assert vs['narrow']['kindselector']['kind'] == str(_KIND_SINGLE)
    assert vs['dbl']['kindselector']['kind'] == str(_KIND_DOUBLE)
    assert capi_maps.getctype(vs['wide']) == 'long_double'
    assert capi_maps.getctype(vs['narrow']) == 'float'
    assert capi_maps.getctype(vs['dbl']) == 'double'


def test_unresolved_kind_is_not_guessed(capsys):
    # No parameter is in scope, so the text is not turned into a kind.
    var = {'typespec': 'real', 'kindselector': {'kind': 'kind(1.0_dp)'}}
    assert capi_maps.getctype(var) == 'float'
    err = capsys.readouterr().err
    assert 'kind(1.0_dp)' in err
    assert 'double' not in err


def test_iso_fortran_env_names_stay_in_f2cmap(capsys):
    # The names are already in f2cmap. getctype has no second table.
    real = {'typespec': 'real', 'kindselector': {'kind': 'real64'}}
    integer = {'typespec': 'integer', 'kindselector': {'kind': 'int8'}}
    assert capi_maps.getctype(real) == 'double'
    assert capi_maps.getctype(integer) == 'signed_char'
    assert capsys.readouterr().err == ''
