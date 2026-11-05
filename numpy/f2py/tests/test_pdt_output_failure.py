"""Unadopted native outputs are released after later copy-back rejection."""

import gc

import pytest

from numpy.f2py import crackfortran, derived_type_rules, rules
from numpy.f2py._dt_routines import prepare_routine, wrapper_handles

from . import util


@pytest.mark.parametrize('name', ['scalar_then_invalidate', 'vector_then_invalidate',
                                  'fixed_then_invalidate'])
def test_failed_output_cleanup_preserves_the_native_error_before_adoption(name):
    path = util.getpath('tests', 'src', 'derived_types', 'pdt_output_failure.f90')
    modules = crackfortran.crackfortran([path])
    module = modules[0]
    pymod = {'block': 'python module', 'name': 'output_failure', 'body': modules}
    descriptions = derived_type_rules.describe_types(pymod)
    routine = next(item for item in module['body'] if item['name'] == name)
    routine['modulename'] = module['name']
    assert prepare_routine(routine, descriptions) == []
    _, declarations, pre, post, body, _ = wrapper_handles(routine, routine['args'])
    native = '\n'.join([*declarations, *pre,
                        *body(lambda names: ['call native(' + ', '.join(
                            names.get(a, a) for a in routine['f2py_dt_args']) + ')']),
                        *post])
    adopted = native.index('_adopt(')
    assert native.index('f2py_pdt_validate(input(', adopted) > adopted
    generated, _ = rules.buildapi(routine)
    source = generated['body']
    status = source.index(routine['f2py_pdt_status'] + ' != 0')
    save = source.index('PyErr_GetRaisedException()', status)
    release = source.index('f2py_pdt_destroy(', save)
    restore = source.index('PyErr_SetRaisedException(', release)
    failure = source.index('f2py_success = 0;', restore)
    assert status < save < release < restore < failure
    assert 'PyExc_ValueError' in source[status:save]
    assert 'PyErr_WriteUnraisable(Py_None)' in source[release:restore]
    assert 'output = NULL;' in source[save:restore] or 'output[' in source[save:restore]
    assert '_dt_owner' not in source[save:restore]
    if name == 'vector_then_invalidate':
        assert '_free_output(' in source[release:restore]


@pytest.mark.slow
class TestPDTOutputFailure(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types', 'pdt_output_failure.f90')]

    @pytest.mark.parametrize('name, expected', [
        ('scalar_then_invalidate', 2),
        ('vector_then_invalidate', 4),
        ('fixed_then_invalidate', 4),
    ])
    def test_later_copyback_failure_releases_every_unadopted_output(
            self, name, expected):
        module = self.module.pdt_output_failure
        for _ in range(3):
            module.reset_sources()
            incoming = module.selected
            with pytest.raises(
                    ValueError,
                    match='invalid derived type argument or native allocation'):
                getattr(module, name)(incoming)
            assert module.selected is None
            assert module.finalized_outputs == expected
            for value in incoming:
                with pytest.raises(ReferenceError):
                    _ = value.tag

    def test_successful_scalar_adoption_keeps_its_native_owner(self):
        module = self.module.pdt_output_failure
        module.reset_sources()
        output = module.scalar_success()
        assert output.tag == 777
        assert module.finalized_outputs == 1
        del output
        gc.collect()
        assert module.finalized_outputs == 2

    def test_successful_vector_adoption_keeps_each_native_owner(self):
        module = self.module.pdt_output_failure
        module.reset_sources()
        output = module.vector_success()
        assert len(output) == 2
        assert [value.tag for value in output] == [777, 777]
        assert module.finalized_outputs == 2
        del output
        gc.collect()
        assert module.finalized_outputs == 4
