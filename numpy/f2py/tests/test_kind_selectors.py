"""Kind selectors go through crackfortran's parameter evaluator."""
import textwrap

from numpy.f2py import capi_maps, crackfortran


def _modvars(tmp_path, src, name):
    fpath = tmp_path / name
    fpath.write_text(src)
    return crackfortran.crackfortran([str(fpath)])[0]['vars']


def test_kind_suffix_uses_the_parameter():
    # kind(1.0_qp) is the value of qp. A letter in the name is not a
    # guess, so dp = 4 is kind 4 and dp = 8 is kind 8.
    par = {'typespec': 'integer', 'attrspec': ['parameter']}
    base = {
        'qp': {**par, '=': '16'},
        'dp4': {**par, '=': '4'},
        'dp8': {**par, '=': '8'},
        'kq': {**par, '=': 'kind(1.0_qp)'},
        'k4': {**par, '=': 'kind(1.0_dp4)'},
        'k8': {**par, '=': 'kind(1.0_dp8)'},
        'kd': {**par, '=': 'kind(1.0d0)'},
        'ks': {**par, '=': 'kind(1.0)'},
        'ki': {**par, '=': 'kind(0)'},
    }
    got = crackfortran.get_parameters(base)
    assert got['kq'] == 16
    assert got['k4'] == 4
    assert got['k8'] == 8
    assert got['kd'] == 8
    assert got['ks'] == 4
    assert got['ki'] == 4


def test_module_kind_selectors_resolve(tmp_path):
    src = textwrap.dedent("""\
        module kindsel
          integer, parameter :: qp = 16
          integer, parameter :: dp = 4
          real(kind=kind(1.0_qp)) :: wide
          real(kind=kind(1.0_dp)) :: narrow
          real(kind=kind(1.0d0)) :: dbl
        end module kindsel
        """)
    vs = _modvars(tmp_path, src, 'kindsel.f90')
    assert vs['wide']['kindselector']['kind'] == '16'
    assert vs['narrow']['kindselector']['kind'] == '4'
    assert vs['dbl']['kindselector']['kind'] == '8'
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
