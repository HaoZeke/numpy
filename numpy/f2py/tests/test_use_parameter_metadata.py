from numpy.f2py import crackfortran


def test_use_reads_typed_cache_without_variable_registry(monkeypatch):
    cached = crackfortran._Integer(7, 8)
    monkeypatch.setattr(crackfortran, 'f90modulevars', {})
    monkeypatch.setattr(crackfortran, 'f90moduleparams', {'provider': {'n': cached}})
    consumer = dict(name='consumer', block='module', use={'provider': {}})
    parameters = crackfortran.get_useparameters(consumer)
    assert parameters['n'] is cached
    assert parameters['n'].kind == 8


def test_later_module_inquires_used_integer_kind(tmp_path):
    source = tmp_path / 'used_parameter_kind.f90'
    source.write_text(
        'module source_values\n'
        'integer(8), parameter :: n = 7_8 / 2_8 * 2_8\n'
        'end module source_values\n'
        'module result_values\n'
        'use source_values\n'
        'integer, parameter :: width = kind(n)\n'
        'integer, parameter :: bits = bit_size(n)\n'
        'real :: data(width)\n'
        'end module result_values\n', encoding='ascii')
    blocks = {block['name']: block for block in crackfortran.crackfortran([str(source)])}
    assert blocks['result_values']['vars']['data']['dimension'] == ['8']
    source_cache = crackfortran.f90moduleparams['source_values']['n']
    assert source_cache == 6
    assert source_cache.kind == 8
    target_cache = crackfortran.f90moduleparams['result_values']
    assert target_cache['width'] == 8
    assert target_cache['bits'] == 64
    assert 'n' not in target_cache
