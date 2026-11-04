"""Type-bound procedures of a type with extends() are scanned."""
import textwrap

from numpy.f2py._dt_routines import _scan_type_bound_procedures


def test_extends_type_bindings(tmp_path):
    src = tmp_path / 'extends_tbp.f90'
    src.write_text(textwrap.dedent("""\
        module shapes
          type, abstract :: shape
          contains
            procedure :: area => shape_area
          end type shape
          type, extends(shape) :: disk
            real :: radius
          contains
            procedure :: area => disk_area
            procedure :: grow => disk_grow
          end type disk
        end module shapes
        """))
    assert _scan_type_bound_procedures(str(src), 'disk') == {
        'area': 'disk_area',
        'grow': 'disk_grow',
    }
    assert _scan_type_bound_procedures(str(src), 'shape') == {
        'area': 'shape_area',
    }
