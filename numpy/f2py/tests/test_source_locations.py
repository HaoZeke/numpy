"""Source ranges linking generated C wrappers to their Fortran definitions."""

import pytest

from numpy.f2py import capi_maps, crackfortran, f2py2e


def _parse(files):
    crackfortran.reset_global_f2py_vars()
    try:
        return crackfortran.crackfortran([str(path) for path in files])
    finally:
        crackfortran.reset_global_f2py_vars()


@pytest.mark.parametrize('suffix, source, first, last', [
    ('.f90', '! heading\nsubroutine &\n sample(x)\ninteger, intent(out) :: x\nx=7\nend subroutine sample\n', 2, 6),
    ('.f', '      subroutine sample(\n     &x)\n      integer x\n      x=7\n      end\n', 1, 5),
])
def test_definition_range_includes_continuation(tmp_path, suffix, source, first, last):
    path = tmp_path / ('source' + suffix)
    path.write_text(source, encoding='ascii')
    block, = _parse([path])
    assert block['_source_location'] == {
        'filename': str(path), 'start_line': first,
        'end_filename': str(path), 'end_line': last,
    }
    assert crackfortran.current_source_location is None


def test_include_definition_has_its_own_file_range(tmp_path):
    included = tmp_path / 'piece.inc'
    included.write_text('! -*- f90 -*-\nsubroutine helper(x)\ninteger x\nx=3\nend subroutine helper\n', encoding='ascii')
    main = tmp_path / 'main.f90'
    main.write_text("include 'piece.inc'\nsubroutine outer(x)\ninteger x\nx=4\nend subroutine outer\n", encoding='ascii')
    blocks = {block['name']: block for block in _parse([main])}
    assert blocks['helper']['_source_location'] == {
        'filename': str(included), 'start_line': 2,
        'end_filename': str(included), 'end_line': 5,
    }
    assert blocks['outer']['_source_location'] == {
        'filename': str(main), 'start_line': 2,
        'end_filename': str(main), 'end_line': 5,
    }


def test_definition_at_file_boundary_retains_its_filename(tmp_path):
    paths = [tmp_path / 'first.f90', tmp_path / 'second.f90']
    for number, path in enumerate(paths):
        path.write_text(f'subroutine routine{number}(x)\ninteger x\nx=1\nend subroutine routine{number}', encoding='ascii')
    blocks = _parse(paths)
    assert len(blocks) == 2
    for block, path in zip(blocks, paths):
        assert block['_source_location'] == {
            'filename': str(path), 'start_line': 1,
            'end_filename': str(path), 'end_line': 4,
        }


@pytest.mark.parametrize('function', [False, True])
def test_generated_c_names_original_definition(tmp_path, function):
    source = tmp_path / 'source with spaces.f90'
    if function:
        source.write_text('! heading\nreal function sample(x)\nreal x\nsample=x+1\nend function sample\n', encoding='ascii')
    else:
        source.write_text('! heading\nsubroutine sample(x)\ninteger x\nx=7\nend subroutine sample\n', encoding='ascii')
    build = tmp_path / 'generated'
    f2py2e.run_main(['-m', 'source_mapping', '--build-dir', str(build), str(source)])
    generated = (build / 'source_mappingmodule.c').read_text()
    assert f'/* Fortran source: {ascii(str(source))}:2-5 */' in generated
    assert generated.index('/* Fortran source:') < generated.index('static char doc_f2py_rout_source_mapping_sample')


def test_signature_file_definition_range(tmp_path):
    source = tmp_path / 'source_mapping.pyf'
    source.write_text('python module source_mapping\ninterface\nsubroutine sample(x)\ninteger x\nend subroutine sample\nend interface\nend python module source_mapping\n', encoding='ascii')
    build = tmp_path / 'generated'
    f2py2e.run_main(['--build-dir', str(build), str(source)])
    generated = (build / 'source_mappingmodule.c').read_text()
    assert f'/* Fortran source: {ascii(str(source))}:3-5 */' in generated


def test_source_comment_cannot_end_the_comment_or_expand_a_template():
    filename = 'part*/bad#name#.f90\nnext'
    location = {'filename': filename, 'start_line': 2, 'end_filename': filename, 'end_line': 5}
    comment = capi_maps._fortran_source_comment({'_source_location': location})
    assert comment.startswith('/* Fortran source: ')
    assert comment.endswith(':2-5 */')
    assert comment.count('*/') == 1
    assert '* /' in comment and r'\x23name\x23' in comment
    assert '\n' not in comment
    assert r'\n' in comment


def test_manual_routine_without_source_metadata_has_no_invented_location():
    assert capi_maps._fortran_source_comment({'name': 'manual'}) == ''
