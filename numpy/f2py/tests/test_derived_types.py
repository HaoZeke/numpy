"""Tests for derived type wrapping support in f2py."""

import textwrap

import numpy as np
import pytest

from numpy.f2py import crackfortran, derived_type_rules

from . import util


class TestDerivedTypeCodeGeneration:
    """Test C code generation for derived types (no compilation needed)."""

    def test_bindc_struct_generation(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "simple_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        # Should have generated code for both types
        assert len(hooks['f90modhooks']) == 2
        assert len(hooks['initf90modhooks']) > 0

        # Check the bind(c) type generates correct C struct
        bindc_code = hooks['f90modhooks'][0]
        assert 'f2py_cartesian_t' in bindc_code
        assert 'double x;' in bindc_code
        assert 'double y;' in bindc_code
        assert 'double z;' in bindc_code

        # Check Python type is generated
        assert 'PycartesianObject' in bindc_code
        assert 'Pycartesian_Type' in bindc_code
        assert 'Pycartesian_tp_new' in bindc_code
        assert 'Pycartesian_tp_init' in bindc_code
        assert 'Pycartesian_tp_dealloc' in bindc_code
        assert 'Pycartesian_get_x' in bindc_code
        assert 'Pycartesian_set_x' in bindc_code

    def test_opaque_struct_generation(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "simple_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        # Check non-bind(c) Point type uses opaque pointer path
        point_code = hooks['f90modhooks'][1]
        # Opaque path should have extern declarations for Fortran wrappers
        assert 'extern void *f2py_create_point' in point_code
        assert 'f2py_destroy_point' in point_code
        assert 'f2py_get_point_x' in point_code
        assert 'f2py_set_point_x' in point_code
        # Should still have Python type
        assert 'PypointObject' in point_code
        assert 'Pypoint_Type' in point_code

    def test_init_registration_code(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "simple_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        init_code = '\n'.join(hooks['initf90modhooks'])
        assert 'PyType_Ready(&Pycartesian_Type)' in init_code
        assert 'PyModule_AddObject(m, "cartesian"' in init_code
        assert 'PyType_Ready(&Pypoint_Type)' in init_code
        assert 'PyModule_AddObject(m, "point"' in init_code

    def test_repr_generation(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "simple_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        bindc_code = hooks['f90modhooks'][0]
        assert 'tp_repr' in bindc_code
        assert 'snprintf' in bindc_code

    def test_capsule_destructor(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "simple_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        bindc_code = hooks['f90modhooks'][0]
        assert 'f2py_cartesian_capsule_destructor' in bindc_code
        assert 'PyMem_Free' in bindc_code
        assert 'f2py.cartesian' in bindc_code  # capsule name

    def test_empty_module_no_hooks(self):
        """Module without derived types should generate empty hooks."""
        hooks = derived_type_rules.buildhooks({'name': 'empty',
                                               'block': 'module',
                                               'body': [],
                                               'vars': {}})
        assert hooks['f90modhooks'] == []
        assert hooks['initf90modhooks'] == []


@pytest.mark.slow
class TestSimpleBindCType(util.F2PyTest):
    """Test compilation and runtime behavior of bind(c) derived types."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "simple_bindc.f90")]

    def test_type_exists(self):
        """The cartesian type should be accessible as a class."""
        # Types are registered at module level (not on the fortran submodule)
        assert hasattr(self.module, 'cartesian')

    def test_construct_default(self):
        """Construct with default (zero) values."""
        p = self.module.cartesian()
        assert p.x == 0.0
        assert p.y == 0.0
        assert p.z == 0.0

    def test_construct_with_args(self):
        """Construct with positional arguments."""
        p = self.module.cartesian(1.0, 2.0, 3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_construct_with_kwargs(self):
        """Construct with keyword arguments."""
        p = self.module.cartesian(x=1.0, z=3.0)
        assert p.x == 1.0
        assert p.y == 0.0
        assert p.z == 3.0

    def test_set_properties(self):
        """Set member values after construction."""
        p = self.module.cartesian()
        p.x = 10.0
        p.y = 20.0
        p.z = 30.0
        assert p.x == 10.0
        assert p.y == 20.0
        assert p.z == 30.0

    def test_repr(self):
        """String representation includes member values."""
        p = self.module.cartesian(1.0, 2.0, 3.0)
        r = repr(p)
        assert 'cartesian' in r
        assert '1' in r

    def test_multiple_instances(self):
        """Multiple instances should be independent."""
        p1 = self.module.cartesian(1.0, 0.0, 0.0)
        p2 = self.module.cartesian(0.0, 2.0, 0.0)
        assert p1.x == 1.0
        assert p2.x == 0.0
        assert p1.y == 0.0
        assert p2.y == 2.0

    def test_opaque_point_exists(self):
        """Non-bind(c) Point type should also be wrapped."""
        assert hasattr(self.module, 'point')

    def test_opaque_point_construct(self):
        """Non-bind(c) Point uses opaque pointer path."""
        p = self.module.point(3.0, 4.0)
        assert abs(p.x - 3.0) < 1e-6
        assert abs(p.y - 4.0) < 1e-6

    def test_opaque_point_set(self):
        """Set members on opaque type."""
        p = self.module.point()
        p.x = 7.0
        p.y = 8.0
        assert abs(p.x - 7.0) < 1e-6
        assert abs(p.y - 8.0) < 1e-6

    def test_opaque_point_repr(self):
        p = self.module.point(1.5, 2.5)
        r = repr(p)
        assert 'point' in r

    def test_opaque_point_independence(self):
        """Multiple opaque instances should be independent."""
        p1 = self.module.point(1.0, 0.0)
        p2 = self.module.point(0.0, 2.0)
        assert abs(p1.x - 1.0) < 1e-6
        assert abs(p2.x - 0.0) < 1e-6
        p1.x = 99.0
        assert abs(p2.x - 0.0) < 1e-6


@pytest.mark.slow
class TestMixedScalarTypes(util.F2PyTest):
    """Test opaque types with mixed scalar members (real, integer, double)."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "mixed_types.f90")]

    def test_particle_exists(self):
        assert hasattr(self.module, 'particle')

    def test_particle_construct(self):
        p = self.module.particle(1.0, 2.0, 3.0, 9.8, 42)
        assert abs(p.x - 1.0) < 1e-6
        assert abs(p.y - 2.0) < 1e-6
        assert abs(p.z - 3.0) < 1e-6
        assert abs(p.mass - 9.8) < 1e-6
        assert p.id == 42

    def test_particle_set_integer(self):
        p = self.module.particle()
        p.id = 100
        assert p.id == 100

    def test_particle_set_float(self):
        p = self.module.particle()
        p.mass = 1.67e-27
        assert abs(p.mass - 1.67e-27) < 1e-33

    def test_gridindex(self):
        g = self.module.gridindex(10, 20, 30)
        assert g.i == 10
        assert g.j == 20
        assert g.k == 30

    def test_gridindex_repr(self):
        g = self.module.gridindex(1, 2, 3)
        r = repr(g)
        assert 'gridindex' in r

    def test_vector3d_double(self):
        v = self.module.vector3d(1.1, 2.2, 3.3)
        assert abs(v.vx - 1.1) < 1e-12
        assert abs(v.vy - 2.2) < 1e-12
        assert abs(v.vz - 3.3) < 1e-12

    def test_multiple_types_coexist(self):
        """All three types should work simultaneously."""
        p = self.module.particle(1.0, 2.0, 3.0, 4.0, 5)
        g = self.module.gridindex(10, 20, 30)
        v = self.module.vector3d(0.1, 0.2, 0.3)
        assert p.id == 5
        assert g.k == 30
        assert abs(v.vz - 0.3) < 1e-12


class TestArrayMemberCodeGen:
    """Test code generation for array member support (no compilation)."""

    def test_bindc_array_struct(self):
        """bind(c) type with array should generate C array fields."""
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_members_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        # Vec3 has float v[3]
        vec3_code = hooks['f90modhooks'][1]
        assert 'float v[3]' in vec3_code

    def test_bindc_array_getter_uses_numpy(self):
        """Array getter should use PyArray_SimpleNewFromData."""
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_members_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        vec3_code = hooks['f90modhooks'][1]
        assert 'PyArray_SimpleNewFromData' in vec3_code

    def test_bindc_array_setter_uses_memcpy(self):
        """Array setter should copy via memcpy."""
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_members_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        vec3_code = hooks['f90modhooks'][1]
        assert 'memcpy' in vec3_code

    def test_bindc_array_repr(self):
        """Repr should show <array(N)> for array members."""
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_members_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        vec3_code = hooks['f90modhooks'][1]
        assert '<array(3)>' in vec3_code

    def test_opaque_array_fortran_wrappers(self):
        """Opaque type array wrappers should generate Fortran getter/setter."""
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_members_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        from numpy.f2py.f90mod_rules import findf90modules
        for m in findf90modules(module):
            type_blocks = derived_type_rules._find_derived_types(m)
            src = derived_type_rules.generate_fortran_wrappers(
                m['name'], type_blocks)
        assert src is not None
        # Array getter returns c_ptr via c_loc
        assert 'c_loc(obj%wavelengths)' in src
        # Array setter takes array arg and copies
        assert 'f2py_set_spectrum_wavelengths' in src

    def test_opaque_array_extern_decls(self):
        """Opaque C code should have correct extern decls for arrays."""
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_members_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        spectrum_code = hooks['f90modhooks'][0]
        # Array getter returns void*
        assert 'extern void *f2py_get_spectrum_wavelengths' in spectrum_code
        # Array setter takes typed pointer
        assert 'extern void f2py_set_spectrum_wavelengths' in spectrum_code

    def test_opaque_array_repr(self):
        """Opaque repr should show <array(N)> for array members."""
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_members_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        spectrum_code = hooks['f90modhooks'][0]
        assert '<array(5)>' in spectrum_code


@pytest.mark.slow
class TestArrayMemberBindC(util.F2PyTest):
    """Test bind(c) types with fixed-size array members."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "array_members_bindc.f90")]

    def test_vec3_exists(self):
        assert hasattr(self.module, 'vec3')

    def test_vec3_default_array(self):
        v = self.module.vec3()
        arr = v.v
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, [0.0, 0.0, 0.0])

    def test_vec3_set_array(self):
        v = self.module.vec3()
        v.v = [1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(v.v, [1.0, 2.0, 3.0])

    def test_vec3_array_is_view(self):
        """bind(c) array getter returns a view (writes go through)."""
        v = self.module.vec3()
        arr = v.v
        arr[0] = 99.0
        np.testing.assert_almost_equal(v.v[0], 99.0)

    def test_vec3_wrong_size_rejected(self):
        v = self.module.vec3()
        with pytest.raises(ValueError):
            v.v = [1.0, 2.0]  # wrong size

    def test_matrix2x2_exists(self):
        assert hasattr(self.module, 'matrix2x2')

    def test_matrix2x2_array_and_scalar(self):
        m = self.module.matrix2x2(scale=2.5)
        assert abs(m.scale - 2.5) < 1e-12
        arr = m.data
        assert arr.shape == (4,)
        np.testing.assert_array_equal(arr, [0.0, 0.0, 0.0, 0.0])
        m.data = [1.0, 0.0, 0.0, 1.0]
        np.testing.assert_array_almost_equal(m.data, [1.0, 0.0, 0.0, 1.0])

    def test_matrix2x2_repr(self):
        m = self.module.matrix2x2(scale=1.0)
        r = repr(m)
        assert 'matrix2x2' in r
        assert '<array(4)>' in r


@pytest.mark.slow
class TestArrayMemberOpaque(util.F2PyTest):
    """Test opaque types with fixed-size array members."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "array_members_opaque.f90")]

    def test_spectrum_exists(self):
        assert hasattr(self.module, 'spectrum')

    def test_spectrum_default_arrays(self):
        s = self.module.spectrum()
        wl = s.wavelengths
        assert isinstance(wl, np.ndarray)
        assert wl.shape == (5,)

    def test_spectrum_set_array(self):
        s = self.module.spectrum()
        s.wavelengths = [400.0, 500.0, 600.0, 700.0, 800.0]
        np.testing.assert_array_almost_equal(
            s.wavelengths, [400.0, 500.0, 600.0, 700.0, 800.0])

    def test_spectrum_scalar_member(self):
        s = self.module.spectrum(npeaks=3)
        assert s.npeaks == 3

    def test_spectrum_array_is_copy(self):
        """Opaque array getter returns a copy (unlike bind(c) view)."""
        s = self.module.spectrum()
        s.wavelengths = [1.0, 2.0, 3.0, 4.0, 5.0]
        arr = s.wavelengths
        arr[0] = 999.0
        # Should NOT affect the stored data (copy semantics)
        np.testing.assert_almost_equal(s.wavelengths[0], 1.0)

    def test_spectrum_repr(self):
        s = self.module.spectrum(npeaks=2)
        r = repr(s)
        assert 'spectrum' in r
        assert '<array(5)>' in r

    def test_triangle_exists(self):
        assert hasattr(self.module, 'triangle')

    def test_triangle_set_get(self):
        t = self.module.triangle()
        verts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        t.vertices = verts
        np.testing.assert_array_almost_equal(t.vertices, verts)


class TestTypeAsArgCodeGen:
    """Test code generation for routines with derived type arguments."""

    def test_has_derived_type_args(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_as_arg.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        from numpy.f2py.f90mod_rules import findf90modules
        for m in findf90modules(mod[0]):
            for b in m.get('body', []):
                if b.get('name') == 'translate_particle':
                    assert derived_type_rules._has_derived_type_args(b)
                if b.get('name') == 'particle_distance':
                    assert derived_type_rules._has_derived_type_args(b)

    def test_routine_wrapper_generated(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_as_arg.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        # Should have: type wrapper + routine wrappers + method table
        all_code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_routine_translate_particle' in all_code
        assert 'f2py_routine_particle_distance' in all_code
        assert 'f2py_routine_make_particle' in all_code
        assert 'f2py_routine_create_particle' in all_code
        assert 'f2py_type_arg_mod_derived_methods' in all_code

    def test_function_returning_type_generated(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_as_arg.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        all_code = '\n'.join(hooks['f90modhooks'])
        # Function returning type should have void* extern
        assert 'extern void *' in all_code
        assert 'f2py_wrap_create_particle' in all_code
        # Should wrap result in capsule
        assert 'f2py_routine_create_particle' in all_code
        assert 'PyCapsule_New' in all_code

    def test_fortran_wrappers_for_routines(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_as_arg.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        from numpy.f2py.f90mod_rules import findf90modules
        for m in findf90modules(mod[0]):
            type_blocks = derived_type_rules._find_derived_types(m)
            routines = [b for b in m.get('body', [])
                        if b.get('block') in ('subroutine', 'function')]
            src = derived_type_rules.generate_fortran_wrappers(
                m['name'], type_blocks, routines=routines)
        assert src is not None
        assert 'f2py_wrap_translate_particle' in src
        assert 'f2py_wrap_particle_distance' in src
        assert 'f2py_wrap_create_particle' in src
        assert 'c_f_pointer' in src
        # Function returning type should allocate temp and use c_loc
        assert 'f2py_temp_result' in src
        assert 'c_loc(f2py_temp_result)' in src


@pytest.mark.slow
class TestTypeAsArg(util.F2PyTest):
    """Test routines that take derived type arguments."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "type_as_arg.f90")]

    def test_particle_type_exists(self):
        assert hasattr(self.module, 'particle')

    def test_translate_exists(self):
        assert hasattr(self.module, 'translate_particle')

    def test_translate_particle(self):
        p = self.module.particle(1.0, 2.0, 3.0, 10.0)
        self.module.translate_particle(p, 0.5, 0.5, 0.5)
        assert abs(p.x - 1.5) < 1e-6
        assert abs(p.y - 2.5) < 1e-6
        assert abs(p.z - 3.5) < 1e-6

    def test_particle_distance(self):
        p1 = self.module.particle(0.0, 0.0, 0.0, 1.0)
        p2 = self.module.particle(3.0, 4.0, 0.0, 1.0)
        dist = self.module.particle_distance(p1, p2)
        assert abs(dist - 5.0) < 1e-6

    def test_make_particle(self):
        p = self.module.make_particle(1.0, 2.0, 3.0, 9.8)
        assert abs(p.x - 1.0) < 1e-6
        assert abs(p.mass - 9.8) < 1e-6

    def test_create_particle_exists(self):
        assert hasattr(self.module, 'create_particle')

    def test_create_particle_function(self):
        p = self.module.create_particle(1.0, 2.0, 3.0, 9.8)
        assert abs(p.x - 1.0) < 1e-6
        assert abs(p.y - 2.0) < 1e-6
        assert abs(p.z - 3.0) < 1e-6
        assert abs(p.mass - 9.8) < 1e-6

    def test_create_particle_returns_correct_type(self):
        p = self.module.create_particle(0.0, 0.0, 0.0, 1.0)
        p_type = type(self.module.particle())
        assert isinstance(p, p_type)

    def test_create_particle_independent_instances(self):
        p1 = self.module.create_particle(1.0, 2.0, 3.0, 10.0)
        p2 = self.module.create_particle(4.0, 5.0, 6.0, 20.0)
        assert abs(p1.x - 1.0) < 1e-6
        assert abs(p2.x - 4.0) < 1e-6

    def test_create_particle_mutable(self):
        p = self.module.create_particle(1.0, 2.0, 3.0, 10.0)
        p.x = 99.0
        assert abs(p.x - 99.0) < 1e-6

    def test_create_particle_usable_as_arg(self):
        p = self.module.create_particle(1.0, 2.0, 3.0, 10.0)
        self.module.translate_particle(p, 0.5, 0.5, 0.5)
        assert abs(p.x - 1.5) < 1e-6


class TestTypeBoundProcCodeGen:
    """Test code generation for type-bound procedures."""

    def test_scan_bound_procedures(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_bound_proc.f90")
        procs = derived_type_rules._scan_type_bound_procedures(
            str(fpath), 'Counter')
        assert 'increment' in procs
        assert procs['increment'] == 'counter_increment'
        assert 'get_count' in procs
        assert procs['get_count'] == 'counter_get_count'
        assert 'reset' in procs
        assert procs['reset'] == 'counter_reset'

    def test_hooks_include_methods(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_bound_proc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        all_code = '\n'.join(hooks['f90modhooks'])
        assert 'Pycounter_method_increment' in all_code
        assert 'Pycounter_method_get_count' in all_code
        assert 'Pycounter_methods' in all_code
        assert '.tp_methods' in all_code


@pytest.mark.slow
class TestTypeBoundProc(util.F2PyTest):
    """Test type-bound procedures (methods on derived types)."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "type_bound_proc.f90")]

    def test_counter_exists(self):
        assert hasattr(self.module, 'counter')

    def test_counter_has_methods(self):
        c = self.module.counter()
        assert hasattr(c, 'increment')
        assert hasattr(c, 'get_count')
        assert hasattr(c, 'reset')

    def test_counter_increment(self):
        c = self.module.counter()
        c.increment(5)
        assert c.count == 5
        c.increment(3)
        assert c.count == 8

    def test_counter_get_count(self):
        c = self.module.counter(count=42)
        val = c.get_count()
        assert val == 42

    def test_counter_reset(self):
        c = self.module.counter(count=100)
        c.reset()
        assert c.count == 0


class TestNestedTypeCodeGen:
    """Test code generation for nested derived types (no compiler)."""

    def test_bindc_nested_type_detected(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "nested_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_vec2_t' in code
        assert 'f2py_particle_t' in code

    def test_bindc_nested_struct_layout(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "nested_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        # The Particle struct should contain Vec2 members
        assert 'f2py_vec2_t pos;' in code
        assert 'f2py_vec2_t vel;' in code

    def test_opaque_nested_type_detected(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "nested_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        assert 'Pyvec2Object' in code
        assert 'PyparticleObject' in code


@pytest.mark.slow
class TestNestedBindC(util.F2PyTest):
    """Test nested bind(c) derived types with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "nested_bindc.f90")]

    def test_vec2_exists(self):
        assert hasattr(self.module, 'vec2')

    def test_particle_exists(self):
        assert hasattr(self.module, 'particle')

    def test_particle_has_nested_members(self):
        p = self.module.particle(mass=1.0)
        pos = p.pos
        assert hasattr(pos, 'x')
        assert hasattr(pos, 'y')

    def test_particle_nested_getter(self):
        p = self.module.particle(mass=2.5)
        pos = p.pos
        assert pos.x == 0.0
        assert pos.y == 0.0

    def test_particle_nested_setter(self):
        p = self.module.particle(mass=1.0)
        v = self.module.vec2(x=3.0, y=4.0)
        p.pos = v
        pos = p.pos
        assert pos.x == 3.0
        assert pos.y == 4.0

    def test_particle_nested_independence(self):
        """Nested getter returns a copy, not a reference."""
        p = self.module.particle(mass=1.0)
        v = self.module.vec2(x=1.0, y=2.0)
        p.pos = v
        v.x = 99.0
        pos = p.pos
        assert pos.x == 1.0  # unchanged

    def test_particle_mass(self):
        p = self.module.particle(mass=42.0)
        assert p.mass == 42.0

    def test_particle_both_nested(self):
        p = self.module.particle(mass=1.0)
        pos = self.module.vec2(x=1.0, y=2.0)
        vel = self.module.vec2(x=3.0, y=4.0)
        p.pos = pos
        p.vel = vel
        assert p.pos.x == 1.0
        assert p.vel.y == 4.0


class TestArrayOfTypesCodeGen:
    """Test code generation for arrays of derived types (no compiler)."""

    def test_bindc_array_of_types_detected(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_of_types_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_atom_t' in code
        assert 'f2py_molecule_t' in code

    def test_bindc_array_struct_layout(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_of_types_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_atom_t atoms[3];' in code


@pytest.mark.slow
class TestArrayOfTypesBindC(util.F2PyTest):
    """Test arrays of bind(c) derived types with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "array_of_types_bindc.f90")]

    def test_atom_exists(self):
        assert hasattr(self.module, 'atom')

    def test_molecule_exists(self):
        assert hasattr(self.module, 'molecule')

    def test_molecule_atoms_getter(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        assert isinstance(atoms, list)
        assert len(atoms) == 3

    def test_molecule_atoms_default_zero(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        for a in atoms:
            assert a.x == 0.0
            assert a.y == 0.0
            assert a.z == 0.0
            assert a.atomic_num == 0

    def test_molecule_atoms_setter(self):
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        o = self.module.atom(x=1.0, y=0.0, z=0.0, atomic_num=8)
        h2 = self.module.atom(x=2.0, y=0.0, z=0.0, atomic_num=1)
        m.atoms = [h, o, h2]
        atoms = m.atoms
        assert atoms[0].atomic_num == 1
        assert atoms[1].atomic_num == 8
        assert atoms[1].x == 1.0
        assert atoms[2].x == 2.0

    def test_molecule_atoms_independence(self):
        """Elements in returned list are copies."""
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=1.0, y=2.0, z=3.0, atomic_num=1)
        m.atoms = [h, h, h]
        h.x = 99.0  # modify original
        atoms = m.atoms
        assert atoms[0].x == 1.0  # unchanged

    def test_molecule_natoms(self):
        m = self.module.molecule(natoms=3)
        assert m.natoms == 3


class TestInheritanceCodeGen:
    """Test code generation for types with extends(parent)."""

    def test_parent_generated_before_child(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        # Should generate wrappers for Shape, Circle, LabeledCircle
        all_code = '\n'.join(hooks['f90modhooks'])
        shape_pos = all_code.index('PyshapeObject')
        circle_pos = all_code.index('PycircleObject')
        labeled_pos = all_code.index('PylabeledcircleObject')
        # Parent before child in generated code
        assert shape_pos < circle_pos < labeled_pos

    def test_child_has_tp_base(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        all_code = '\n'.join(hooks['f90modhooks'])
        # Circle should have tp_base pointing to Shape
        assert '.tp_base = &Pyshape_Type,' in all_code
        # LabeledCircle should have tp_base pointing to Circle
        assert '.tp_base = &Pycircle_Type,' in all_code
        # Shape should NOT have tp_base
        shape_code = hooks['f90modhooks'][0]
        assert 'tp_base' not in shape_code

    def test_child_getset_only_own_members(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        # Circle's getset should only have 'radius', not 'area'/'color'
        circle_code = hooks['f90modhooks'][1]
        getset_start = circle_code.index('Pycircle_getset')
        getset_block = circle_code[getset_start:]
        sentinel = getset_block.index('{NULL}')
        getset_block = getset_block[:sentinel]
        assert '"radius"' in getset_block
        assert '"area"' not in getset_block
        assert '"color"' not in getset_block

    def test_child_extern_has_all_members(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        circle_code = hooks['f90modhooks'][1]
        # Child extern decls should cover inherited members too
        assert 'f2py_get_circle_area' in circle_code
        assert 'f2py_get_circle_color' in circle_code
        assert 'f2py_get_circle_radius' in circle_code

    def test_fortran_wrappers_generated(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        type_blocks = [b for b in module['body']
                       if b.get('block') == 'type']
        source = derived_type_rules.generate_fortran_wrappers(
            'inheritance_mod', type_blocks)
        assert source is not None
        # All three types should have create/destroy wrappers
        assert 'f2py_create_shape' in source
        assert 'f2py_create_circle' in source
        assert 'f2py_create_labeledcircle' in source
        # Child wrappers should access inherited members
        assert 'f2py_get_circle_area' in source
        assert 'f2py_get_labeledcircle_radius' in source


class TestInheritanceOpaque(util.F2PyTest):
    """Test type inheritance with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "inheritance_opaque.f90")]

    def test_shape_exists(self):
        assert hasattr(self.module, 'shape')

    def test_circle_exists(self):
        assert hasattr(self.module, 'circle')

    def test_labeledcircle_exists(self):
        assert hasattr(self.module, 'labeledcircle')

    def test_shape_members(self):
        s = self.module.shape(area=10.0, color=5)
        assert abs(s.area - 10.0) < 1e-6
        assert s.color == 5

    def test_circle_own_member(self):
        c = self.module.circle(area=0.0, color=1, radius=2.5)
        assert abs(c.radius - 2.5) < 1e-6

    def test_circle_inherited_members(self):
        c = self.module.circle(area=12.0, color=3, radius=2.5)
        assert abs(c.area - 12.0) < 1e-6
        assert c.color == 3

    def test_circle_set_inherited_member(self):
        c = self.module.circle(area=0.0, color=0, radius=1.0)
        c.area = 3.14
        assert abs(c.area - 3.14) < 1e-2

    def test_circle_isinstance_shape(self):
        """Circle should be an instance of Shape (Python inheritance)."""
        c = self.module.circle(area=0.0, color=0, radius=1.0)
        s_type = type(self.module.shape())
        assert isinstance(c, s_type)

    def test_labeledcircle_all_members(self):
        lc = self.module.labeledcircle(
            area=5.0, color=2, radius=1.0, label_id=42)
        assert abs(lc.area - 5.0) < 1e-6
        assert lc.color == 2
        assert abs(lc.radius - 1.0) < 1e-6
        assert lc.label_id == 42

    def test_labeledcircle_isinstance_chain(self):
        """LabeledCircle should be instance of Circle and Shape."""
        lc = self.module.labeledcircle(
            area=0.0, color=0, radius=0.0, label_id=0)
        c_type = type(self.module.circle())
        s_type = type(self.module.shape())
        assert isinstance(lc, c_type)
        assert isinstance(lc, s_type)

    def test_labeledcircle_set_grandparent_member(self):
        lc = self.module.labeledcircle(
            area=0.0, color=0, radius=0.0, label_id=0)
        lc.area = 99.0
        assert abs(lc.area - 99.0) < 1e-6

    def test_repr_shows_all_members(self):
        c = self.module.circle(area=1.0, color=2, radius=3.0)
        r = repr(c)
        assert 'circle' in r.lower()
        assert 'area' in r
        assert 'radius' in r


class TestInheritanceCodeGen:
    """Test code generation for types with extends(parent)."""

    def test_parent_generated_before_child(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        # Should generate wrappers for Shape, Circle, LabeledCircle
        all_code = '\n'.join(hooks['f90modhooks'])
        shape_pos = all_code.index('PyshapeObject')
        circle_pos = all_code.index('PycircleObject')
        labeled_pos = all_code.index('PylabeledcircleObject')
        # Parent before child in generated code
        assert shape_pos < circle_pos < labeled_pos

    def test_child_has_tp_base(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        all_code = '\n'.join(hooks['f90modhooks'])
        # Circle should have tp_base pointing to Shape
        assert '.tp_base = &Pyshape_Type,' in all_code
        # LabeledCircle should have tp_base pointing to Circle
        assert '.tp_base = &Pycircle_Type,' in all_code
        # Shape should NOT have tp_base
        shape_code = hooks['f90modhooks'][0]
        assert 'tp_base' not in shape_code

    def test_child_getset_only_own_members(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        # Circle's getset should only have 'radius', not 'area'/'color'
        circle_code = hooks['f90modhooks'][1]
        getset_start = circle_code.index('Pycircle_getset')
        getset_block = circle_code[getset_start:]
        sentinel = getset_block.index('{NULL}')
        getset_block = getset_block[:sentinel]
        assert '"radius"' in getset_block
        assert '"area"' not in getset_block
        assert '"color"' not in getset_block

    def test_child_extern_has_all_members(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        circle_code = hooks['f90modhooks'][1]
        # Child extern decls should cover inherited members too
        assert 'f2py_get_circle_area' in circle_code
        assert 'f2py_get_circle_color' in circle_code
        assert 'f2py_get_circle_radius' in circle_code

    def test_fortran_wrappers_generated(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        type_blocks = [b for b in module['body']
                       if b.get('block') == 'type']
        source = derived_type_rules.generate_fortran_wrappers(
            'inheritance_mod', type_blocks)
        assert source is not None
        # All three types should have create/destroy wrappers
        assert 'f2py_create_shape' in source
        assert 'f2py_create_circle' in source
        assert 'f2py_create_labeledcircle' in source
        # Child wrappers should access inherited members
        assert 'f2py_get_circle_area' in source
        assert 'f2py_get_labeledcircle_radius' in source


class TestInheritanceOpaque(util.F2PyTest):
    """Test type inheritance with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "inheritance_opaque.f90")]

    def test_shape_exists(self):
        assert hasattr(self.module, 'shape')

    def test_circle_exists(self):
        assert hasattr(self.module, 'circle')

    def test_labeledcircle_exists(self):
        assert hasattr(self.module, 'labeledcircle')

    def test_shape_members(self):
        s = self.module.shape(area=10.0, color=5)
        assert abs(s.area - 10.0) < 1e-6
        assert s.color == 5

    def test_circle_own_member(self):
        c = self.module.circle(area=0.0, color=1, radius=2.5)
        assert abs(c.radius - 2.5) < 1e-6

    def test_circle_inherited_members(self):
        c = self.module.circle(area=12.0, color=3, radius=2.5)
        assert abs(c.area - 12.0) < 1e-6
        assert c.color == 3

    def test_circle_set_inherited_member(self):
        c = self.module.circle(area=0.0, color=0, radius=1.0)
        c.area = 3.14
        assert abs(c.area - 3.14) < 1e-2

    def test_circle_isinstance_shape(self):
        """Circle should be an instance of Shape (Python inheritance)."""
        c = self.module.circle(area=0.0, color=0, radius=1.0)
        s_type = type(self.module.shape())
        assert isinstance(c, s_type)

    def test_labeledcircle_all_members(self):
        lc = self.module.labeledcircle(
            area=5.0, color=2, radius=1.0, label_id=42)
        assert abs(lc.area - 5.0) < 1e-6
        assert lc.color == 2
        assert abs(lc.radius - 1.0) < 1e-6
        assert lc.label_id == 42

    def test_labeledcircle_isinstance_chain(self):
        """LabeledCircle should be instance of Circle and Shape."""
        lc = self.module.labeledcircle(
            area=0.0, color=0, radius=0.0, label_id=0)
        c_type = type(self.module.circle())
        s_type = type(self.module.shape())
        assert isinstance(lc, c_type)
        assert isinstance(lc, s_type)

    def test_labeledcircle_set_grandparent_member(self):
        lc = self.module.labeledcircle(
            area=0.0, color=0, radius=0.0, label_id=0)
        lc.area = 99.0
        assert abs(lc.area - 99.0) < 1e-6

    def test_repr_shows_all_members(self):
        c = self.module.circle(area=1.0, color=2, radius=3.0)
        r = repr(c)
        assert 'circle' in r.lower()
        assert 'area' in r
        assert 'radius' in r


class TestInheritanceCodeGen:
    """Test code generation for types with extends(parent)."""

    def test_parent_generated_before_child(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        # Should generate wrappers for Shape, Circle, LabeledCircle
        all_code = '\n'.join(hooks['f90modhooks'])
        shape_pos = all_code.index('PyshapeObject')
        circle_pos = all_code.index('PycircleObject')
        labeled_pos = all_code.index('PylabeledcircleObject')
        # Parent before child in generated code
        assert shape_pos < circle_pos < labeled_pos

    def test_child_has_tp_base(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        all_code = '\n'.join(hooks['f90modhooks'])
        # Circle should have tp_base pointing to Shape
        assert '.tp_base = &Pyshape_Type,' in all_code
        # LabeledCircle should have tp_base pointing to Circle
        assert '.tp_base = &Pycircle_Type,' in all_code
        # Shape should NOT have tp_base
        shape_code = hooks['f90modhooks'][0]
        assert 'tp_base' not in shape_code

    def test_child_getset_only_own_members(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        # Circle's getset should only have 'radius', not 'area'/'color'
        circle_code = hooks['f90modhooks'][1]
        getset_start = circle_code.index('Pycircle_getset')
        getset_block = circle_code[getset_start:]
        sentinel = getset_block.index('{NULL}')
        getset_block = getset_block[:sentinel]
        assert '"radius"' in getset_block
        assert '"area"' not in getset_block
        assert '"color"' not in getset_block

    def test_child_extern_has_all_members(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        circle_code = hooks['f90modhooks'][1]
        # Child extern decls should cover inherited members too
        assert 'f2py_get_circle_area' in circle_code
        assert 'f2py_get_circle_color' in circle_code
        assert 'f2py_get_circle_radius' in circle_code

    def test_fortran_wrappers_generated(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        type_blocks = [b for b in module['body']
                       if b.get('block') == 'type']
        source = derived_type_rules.generate_fortran_wrappers(
            'inheritance_mod', type_blocks)
        assert source is not None
        # All three types should have create/destroy wrappers
        assert 'f2py_create_shape' in source
        assert 'f2py_create_circle' in source
        assert 'f2py_create_labeledcircle' in source
        # Child wrappers should access inherited members
        assert 'f2py_get_circle_area' in source
        assert 'f2py_get_labeledcircle_radius' in source


class TestInheritanceOpaque(util.F2PyTest):
    """Test type inheritance with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "inheritance_opaque.f90")]

    def test_shape_exists(self):
        assert hasattr(self.module, 'shape')

    def test_circle_exists(self):
        assert hasattr(self.module, 'circle')

    def test_labeledcircle_exists(self):
        assert hasattr(self.module, 'labeledcircle')

    def test_shape_members(self):
        s = self.module.shape(area=10.0, color=5)
        assert abs(s.area - 10.0) < 1e-6
        assert s.color == 5

    def test_circle_own_member(self):
        c = self.module.circle(area=0.0, color=1, radius=2.5)
        assert abs(c.radius - 2.5) < 1e-6

    def test_circle_inherited_members(self):
        c = self.module.circle(area=12.0, color=3, radius=2.5)
        assert abs(c.area - 12.0) < 1e-6
        assert c.color == 3

    def test_circle_set_inherited_member(self):
        c = self.module.circle(area=0.0, color=0, radius=1.0)
        c.area = 3.14
        assert abs(c.area - 3.14) < 1e-2

    def test_circle_isinstance_shape(self):
        """Circle should be an instance of Shape (Python inheritance)."""
        c = self.module.circle(area=0.0, color=0, radius=1.0)
        s_type = type(self.module.shape())
        assert isinstance(c, s_type)

    def test_labeledcircle_all_members(self):
        lc = self.module.labeledcircle(
            area=5.0, color=2, radius=1.0, label_id=42)
        assert abs(lc.area - 5.0) < 1e-6
        assert lc.color == 2
        assert abs(lc.radius - 1.0) < 1e-6
        assert lc.label_id == 42

    def test_labeledcircle_isinstance_chain(self):
        """LabeledCircle should be instance of Circle and Shape."""
        lc = self.module.labeledcircle(
            area=0.0, color=0, radius=0.0, label_id=0)
        c_type = type(self.module.circle())
        s_type = type(self.module.shape())
        assert isinstance(lc, c_type)
        assert isinstance(lc, s_type)

    def test_labeledcircle_set_grandparent_member(self):
        lc = self.module.labeledcircle(
            area=0.0, color=0, radius=0.0, label_id=0)
        lc.area = 99.0
        assert abs(lc.area - 99.0) < 1e-6

    def test_repr_shows_all_members(self):
        c = self.module.circle(area=1.0, color=2, radius=3.0)
        r = repr(c)
        assert 'circle' in r.lower()
        assert 'area' in r
        assert 'radius' in r


@pytest.mark.slow
class TestArrayOfTypesOpaque(util.F2PyTest):
    """Test arrays of opaque derived types with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "array_of_types_opaque.f90")]

    def test_atom_exists(self):
        assert hasattr(self.module, 'atom')

    def test_molecule_exists(self):
        assert hasattr(self.module, 'molecule')

    def test_molecule_atoms_getter(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        assert isinstance(atoms, list)
        assert len(atoms) == 3

    def test_molecule_atoms_default_zero(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        for a in atoms:
            assert a.x == 0.0
            assert a.y == 0.0
            assert a.z == 0.0
            assert a.atomic_num == 0

    def test_molecule_atoms_setter(self):
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        o = self.module.atom(x=1.0, y=0.0, z=0.0, atomic_num=8)
        h2 = self.module.atom(x=2.0, y=0.0, z=0.0, atomic_num=1)
        m.atoms = [h, o, h2]
        atoms = m.atoms
        assert atoms[0].atomic_num == 1
        assert atoms[1].atomic_num == 8
        assert atoms[1].x == 1.0
        assert atoms[2].x == 2.0

    def test_molecule_atoms_independence(self):
        """Elements in returned list are copies."""
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=1.0, y=2.0, z=3.0, atomic_num=1)
        m.atoms = [h, h, h]
        h.x = 99.0  # modify original
        atoms = m.atoms
        assert atoms[0].x == 1.0  # unchanged

    def test_molecule_natoms(self):
        m = self.module.molecule(natoms=3)
        assert m.natoms == 3


class TestInheritanceCodeGen:
    """Test code generation for types with extends(parent)."""

    def test_parent_generated_before_child(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        # Should generate wrappers for Particle, ChargedParticle, Electron
        all_code = '\n'.join(hooks['f90modhooks'])
        particle_pos = all_code.index('PyparticleObject')
        charged_pos = all_code.index('PychargedparticleObject')
        electron_pos = all_code.index('PyelectronObject')
        # Parent before child in generated code
        assert particle_pos < charged_pos < electron_pos

    def test_child_has_tp_base(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        all_code = '\n'.join(hooks['f90modhooks'])
        # ChargedParticle should have tp_base pointing to Particle
        assert '.tp_base = &Pyparticle_Type,' in all_code
        # Electron should have tp_base pointing to ChargedParticle
        assert '.tp_base = &Pychargedparticle_Type,' in all_code
        # Particle should NOT have tp_base
        particle_code = hooks['f90modhooks'][0]
        assert 'tp_base' not in particle_code

    def test_child_getset_has_all_members(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        # ChargedParticle's getset should have all members
        # (inherited + own) since capsule names differ per type
        charged_code = hooks['f90modhooks'][1]
        getset_start = charged_code.index('Pychargedparticle_getset')
        getset_block = charged_code[getset_start:]
        sentinel = getset_block.index('{NULL}')
        getset_block = getset_block[:sentinel]
        assert '"spin"' in getset_block
        assert '"mass"' in getset_block
        assert '"charge"' in getset_block

    def test_child_extern_has_all_members(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        charged_code = hooks['f90modhooks'][1]
        # Child extern decls should cover inherited members too
        assert 'f2py_get_chargedparticle_mass' in charged_code
        assert 'f2py_get_chargedparticle_charge' in charged_code
        assert 'f2py_get_chargedparticle_spin' in charged_code

    def test_fortran_wrappers_generated(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "inheritance_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        type_blocks = [b for b in module['body']
                       if b.get('block') == 'type']
        source = derived_type_rules.generate_fortran_wrappers(
            'inheritance_mod', type_blocks)
        assert source is not None
        # All three types should have create/destroy wrappers
        assert 'f2py_create_particle' in source
        assert 'f2py_create_chargedparticle' in source
        assert 'f2py_create_electron' in source
        # Child wrappers should access inherited members
        assert 'f2py_get_chargedparticle_mass' in source
        assert 'f2py_get_electron_spin' in source


@pytest.mark.slow
class TestInheritanceOpaque(util.F2PyTest):
    """Test type inheritance with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "inheritance_opaque.f90")]

    def test_particle_exists(self):
        assert hasattr(self.module, 'particle')

    def test_chargedparticle_exists(self):
        assert hasattr(self.module, 'chargedparticle')

    def test_electron_exists(self):
        assert hasattr(self.module, 'electron')

    def test_particle_members(self):
        p = self.module.particle(mass=1.5, charge=0)
        assert abs(p.mass - 1.5) < 1e-6
        assert p.charge == 0

    def test_charged_own_member(self):
        cp = self.module.chargedparticle(mass=0.0, charge=1, spin=0.5)
        assert abs(cp.spin - 0.5) < 1e-6

    def test_charged_inherited_members(self):
        cp = self.module.chargedparticle(mass=938.0, charge=1, spin=0.5)
        assert abs(cp.mass - 938.0) < 1e-1
        assert cp.charge == 1

    def test_charged_set_inherited_member(self):
        cp = self.module.chargedparticle(mass=0.0, charge=0, spin=0.0)
        cp.mass = 105.7
        assert abs(cp.mass - 105.7) < 1e-1

    def test_charged_isinstance_particle(self):
        """ChargedParticle should be an instance of Particle."""
        cp = self.module.chargedparticle(mass=0.0, charge=0, spin=0.0)
        p_type = type(self.module.particle())
        assert isinstance(cp, p_type)

    def test_electron_all_members(self):
        e = self.module.electron(
            mass=0.511, charge=-1, spin=0.5, shell_num=2)
        assert abs(e.mass - 0.511) < 1e-3
        assert e.charge == -1
        assert abs(e.spin - 0.5) < 1e-6
        assert e.shell_num == 2

    def test_electron_isinstance_chain(self):
        """Electron should be instance of ChargedParticle and Particle."""
        e = self.module.electron(
            mass=0.0, charge=0, spin=0.0, shell_num=0)
        cp_type = type(self.module.chargedparticle())
        p_type = type(self.module.particle())
        assert isinstance(e, cp_type)
        assert isinstance(e, p_type)

    def test_electron_set_grandparent_member(self):
        e = self.module.electron(
            mass=0.0, charge=0, spin=0.0, shell_num=0)
        e.mass = 99.0
        assert abs(e.mass - 99.0) < 1e-6

    def test_repr_shows_all_members(self):
        cp = self.module.chargedparticle(mass=1.0, charge=2, spin=3.0)
        r = repr(cp)
        assert 'chargedparticle' in r.lower()
        assert 'mass' in r
        assert 'spin' in r


class TestOperatorCodeGen:
    """Test code generation for operator overloading."""

    def test_operator_detected(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "operator_overload.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        all_code = '\n'.join(hooks['f90modhooks'])
        # Should generate nb_add slot
        assert 'nb_add' in all_code
        # Should generate nb_subtract slot
        assert 'nb_subtract' in all_code
        # Should generate richcompare
        assert 'tp_richcompare' in all_code

    def test_number_methods_struct(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "operator_overload.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        all_code = '\n'.join(hooks['f90modhooks'])
        assert 'PyNumberMethods' in all_code
        assert 'tp_as_number' in all_code

    def test_fortran_operator_wrappers(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "operator_overload.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        type_blocks = [b for b in module['body']
                       if b.get('block') == 'type']
        source = derived_type_rules.generate_fortran_wrappers(
            'operator_mod', type_blocks, module_block=module)
        assert source is not None
        assert 'f2py_op_vec2_vec2_add' in source
        assert 'f2py_op_vec2_vec2_sub' in source
        assert 'f2py_op_vec2_vec2_eq' in source


@pytest.mark.slow
class TestOperatorOverload(util.F2PyTest):
    """Test operator overloading with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "operator_overload.f90")]

    def test_vec2_exists(self):
        assert hasattr(self.module, 'vec2')

    def test_vec2_add(self):
        v1 = self.module.vec2(x=1.0, y=2.0)
        v2 = self.module.vec2(x=3.0, y=4.0)
        v3 = v1 + v2
        assert abs(v3.x - 4.0) < 1e-6
        assert abs(v3.y - 6.0) < 1e-6

    def test_vec2_sub(self):
        v1 = self.module.vec2(x=5.0, y=8.0)
        v2 = self.module.vec2(x=2.0, y=3.0)
        v3 = v1 - v2
        assert abs(v3.x - 3.0) < 1e-6
        assert abs(v3.y - 5.0) < 1e-6

    def test_vec2_eq_true(self):
        v1 = self.module.vec2(x=1.0, y=2.0)
        v2 = self.module.vec2(x=1.0, y=2.0)
        assert v1 == v2

    def test_vec2_eq_false(self):
        v1 = self.module.vec2(x=1.0, y=2.0)
        v2 = self.module.vec2(x=3.0, y=4.0)
        assert not (v1 == v2)

    def test_vec2_add_returns_new_instance(self):
        v1 = self.module.vec2(x=1.0, y=2.0)
        v2 = self.module.vec2(x=3.0, y=4.0)
        v3 = v1 + v2
        # v1 and v2 should be unchanged
        assert abs(v1.x - 1.0) < 1e-6
        assert abs(v2.x - 3.0) < 1e-6
        # v3 is a new object
        assert v3 is not v1 and v3 is not v2

    def test_vec2_add_type_mismatch(self):
        v1 = self.module.vec2(x=1.0, y=2.0)
        result = v1.__add__(42)
        assert result is NotImplemented

    def test_vec2_chained_ops(self):
        v1 = self.module.vec2(x=1.0, y=2.0)
        v2 = self.module.vec2(x=3.0, y=4.0)
        v3 = self.module.vec2(x=5.0, y=6.0)
        v4 = v1 + v2 + v3
        assert abs(v4.x - 9.0) < 1e-6
        assert abs(v4.y - 12.0) < 1e-6


class TestTypeArrayArgCodeGen:
    """Test code generation for routines with array-of-type arguments."""

    def test_array_arg_detected(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_array_arg.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        # Should generate the Atom type
        assert 'PyatomObject' in code

    def test_routine_wrapper_generated(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_array_arg.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        # Should generate wrappers for routines with array-of-type args
        assert 'f2py_routine_sum_positions' in code
        assert 'f2py_routine_count_element' in code

    def test_list_extraction_code(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_array_arg.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        # Should check for list type and extract capsule pointers
        assert 'PyList_Check' in code
        assert 'atoms_ptrs' in code

    def test_fortran_wrapper_generated(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_array_arg.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        type_blocks = [b for b in module['body']
                       if b.get('block') == 'type']
        routines = [b for b in module['body']
                    if b.get('block') in ('subroutine', 'function')]
        fortran = derived_type_rules.generate_fortran_wrappers(
            module['name'], type_blocks, routines=routines,
            module_block=module)
        # Fortran wrapper should accept c_ptr array
        assert fortran is not None
        assert 'atoms_ptrs' in fortran
        assert 'atoms_n' in fortran
        assert 'c_f_pointer' in fortran


@pytest.mark.slow
class TestTypeArrayArgOpaque(util.F2PyTest):
    """Test routines with array-of-type arguments (compilation)."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "type_array_arg.f90")]

    def test_atom_exists(self):
        assert hasattr(self.module, 'atom')

    def test_sum_positions_exists(self):
        assert hasattr(self.module, 'sum_positions')

    def test_count_element_exists(self):
        assert hasattr(self.module, 'count_element')

    def test_sum_positions_basic(self):
        a1 = self.module.atom(x=1.0, y=2.0, z=3.0, atomic_num=1)
        a2 = self.module.atom(x=4.0, y=5.0, z=6.0, atomic_num=8)
        a3 = self.module.atom(x=7.0, y=8.0, z=9.0, atomic_num=1)
        tx, ty, tz = self.module.sum_positions([a1, a2, a3])
        assert abs(tx - 12.0) < 1e-6
        assert abs(ty - 15.0) < 1e-6
        assert abs(tz - 18.0) < 1e-6

    def test_count_element_basic(self):
        a1 = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        a2 = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=8)
        a3 = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        cnt = self.module.count_element([a1, a2, a3], z=1)
        assert cnt == 2

    def test_count_element_none_found(self):
        a1 = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        a2 = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        cnt = self.module.count_element([a1, a2], z=8)
        assert cnt == 0

    def test_sum_positions_empty_list(self):
        tx, ty, tz = self.module.sum_positions([])
        assert abs(tx) < 1e-6
        assert abs(ty) < 1e-6
        assert abs(tz) < 1e-6

    def test_sum_positions_single_element(self):
        a = self.module.atom(x=3.14, y=2.72, z=1.41, atomic_num=6)
        tx, ty, tz = self.module.sum_positions([a])
        assert abs(tx - 3.14) < 1e-5
        assert abs(ty - 2.72) < 1e-5
        assert abs(tz - 1.41) < 1e-5

    def test_wrong_type_in_list(self):
        with pytest.raises(TypeError):
            self.module.sum_positions([42, "not an atom"])

    def test_not_a_list(self):
        with pytest.raises(TypeError):
            self.module.sum_positions("not a list")


class TestAssumedShapeCodeGen:
    """Test code generation for assumed-shape array-of-type arguments."""

    def test_no_invalid_c_for_colon_dim(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_assumed_shape_arg.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        all_code = '\n'.join(hooks['f90modhooks'])
        # Should generate wrapper for sum_vectors
        assert 'f2py_routine_sum_vectors' in all_code
        # Should NOT have `: =` (invalid C from assumed-shape dim)
        assert ': =' not in all_code

    def test_fortran_wrapper_for_assumed_shape(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "type_assumed_shape_arg.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        from numpy.f2py.f90mod_rules import findf90modules
        for m in findf90modules(mod[0]):
            type_blocks = derived_type_rules._find_derived_types(m)
            routines = [b for b in m.get('body', [])
                        if b.get('block') in ('subroutine', 'function')]
            src = derived_type_rules.generate_fortran_wrappers(
                m['name'], type_blocks, routines=routines)
        assert src is not None
        assert 'f2py_wrap_sum_vectors' in src
        assert 'vecs_ptrs' in src
        assert 'vecs_n' in src


@pytest.mark.slow
class TestAssumedShapeOpaque(util.F2PyTest):
    """Test assumed-shape array-of-type arguments with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "type_assumed_shape_arg.f90")]

    def test_vec3_exists(self):
        assert hasattr(self.module, 'vec3')

    def test_sum_vectors_exists(self):
        assert hasattr(self.module, 'sum_vectors')

    def test_sum_vectors_basic(self):
        v1 = self.module.vec3(x=1.0, y=2.0, z=3.0)
        v2 = self.module.vec3(x=4.0, y=5.0, z=6.0)
        tx, ty, tz = self.module.sum_vectors([v1, v2])
        assert abs(tx - 5.0) < 1e-6
        assert abs(ty - 7.0) < 1e-6
        assert abs(tz - 9.0) < 1e-6

    def test_sum_vectors_empty(self):
        tx, ty, tz = self.module.sum_vectors([])
        assert abs(tx) < 1e-6
        assert abs(ty) < 1e-6
        assert abs(tz) < 1e-6

    def test_sum_vectors_type_error(self):
        with pytest.raises(TypeError):
            self.module.sum_vectors([42, "not a vec3"])


class TestCharMemberCodeGen:
    """Test code generation for character string members."""

    def test_char_member_extern_decls(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "char_member.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        all_code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_get_person_name' in all_code
        assert 'f2py_set_person_name' in all_code
        assert 'char *' in all_code

    def test_char_member_getter_code(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "char_member.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        all_code = '\n'.join(hooks['f90modhooks'])
        assert 'memset' in all_code
        assert 'PyUnicode_FromStringAndSize' in all_code

    def test_char_member_fortran_wrappers(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "char_member.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        from numpy.f2py.f90mod_rules import findf90modules
        for m in findf90modules(mod[0]):
            type_blocks = derived_type_rules._find_derived_types(m)
            src = derived_type_rules.generate_fortran_wrappers(
                m['name'], type_blocks)
        assert src is not None
        assert 'f2py_get_person_name' in src
        assert 'f2py_set_person_name' in src
        assert 'character(c_char)' in src


@pytest.mark.slow
class TestCharMember(util.F2PyTest):
    """Test character string members with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "char_member.f90")]

    def test_person_exists(self):
        assert hasattr(self.module, 'person')

    def test_set_get_name(self):
        p = self.module.person(age=30)
        p.name = "Alice"
        assert p.name == "Alice"

    def test_name_round_trip(self):
        p = self.module.person(age=25)
        p.name = "Bob"
        assert p.name == "Bob"
        p.name = "Charlie"
        assert p.name == "Charlie"

    def test_short_string_no_trailing_spaces(self):
        p = self.module.person(age=20)
        p.name = "Hi"
        # Should not have trailing spaces
        assert p.name == "Hi"
        assert len(p.name) == 2

    def test_long_string_truncated(self):
        p = self.module.person(age=20)
        long_name = "A" * 100
        p.name = long_name
        # Should truncate to character(len=32)
        assert len(p.name) == 32

    def test_empty_string(self):
        p = self.module.person(age=20)
        p.name = ""
        assert p.name == ""

    def test_exact_length_string(self):
        p = self.module.person(age=20)
        exact = "A" * 32
        p.name = exact
        assert p.name == exact
        assert len(p.name) == 32

    def test_name_with_scalar_member(self):
        p = self.module.person(age=42)
        p.name = "Test"
        assert p.age == 42
        assert p.name == "Test"

    def test_name_type_error(self):
        p = self.module.person(age=20)
        with pytest.raises(TypeError):
            p.name = 42


class TestAllocMemberCodeGen:
    """Test code generation for allocatable array members."""

    def test_alloc_extern_decls(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "alloc_member.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        all_code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_get_datavec_values_allocated' in all_code
        assert 'f2py_get_datavec_values_size' in all_code
        assert 'f2py_get_datavec_values_data' in all_code
        assert 'f2py_set_datavec_values' in all_code

    def test_alloc_getter_has_none_check(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "alloc_member.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        all_code = '\n'.join(hooks['f90modhooks'])
        assert 'Py_RETURN_NONE' in all_code
        assert '_allocated' in all_code

    def test_alloc_fortran_wrappers(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "alloc_member.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        from numpy.f2py.f90mod_rules import findf90modules
        for m in findf90modules(mod[0]):
            type_blocks = derived_type_rules._find_derived_types(m)
            src = derived_type_rules.generate_fortran_wrappers(
                m['name'], type_blocks)
        assert src is not None
        assert 'f2py_get_datavec_values_allocated' in src
        assert 'f2py_set_datavec_values' in src
        assert 'c_loc' in src


@pytest.mark.slow
class TestAllocMember(util.F2PyTest):
    """Test allocatable array members with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "alloc_member.f90")]

    def test_datavec_exists(self):
        assert hasattr(self.module, 'datavec')

    def test_values_starts_none(self):
        d = self.module.datavec(tag=1)
        assert d.values is None

    def test_set_get_values(self):
        d = self.module.datavec(tag=1)
        d.values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = d.values
        assert result is not None
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_set_none_deallocates(self):
        d = self.module.datavec(tag=1)
        d.values = np.array([1.0, 2.0], dtype=np.float32)
        assert d.values is not None
        d.values = None
        assert d.values is None

    def test_resize(self):
        d = self.module.datavec(tag=1)
        d.values = np.array([1.0, 2.0], dtype=np.float32)
        assert len(d.values) == 2
        d.values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        assert len(d.values) == 4
        np.testing.assert_array_almost_equal(
            d.values, [10.0, 20.0, 30.0, 40.0])

    def test_scalar_member_coexists(self):
        d = self.module.datavec(tag=42)
        d.values = np.array([1.0], dtype=np.float32)
        assert d.tag == 42
        assert len(d.values) == 1


class TestIntentOutTypeArrayCodeGen:
    """Test code generation for intent(out) allocatable arrays of types."""

    def test_wrapper_generated(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "intent_out_type_array.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        all_code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_routine_make_sequence' in all_code

    def test_output_array_wrapper_code(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "intent_out_type_array.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        all_code = '\n'.join(hooks['f90modhooks'])
        # Should have void** extern for the output array
        assert 'void **' in all_code
        # Should build a Python list from results
        assert 'PyList_New' in all_code
        assert 'PyList_SET_ITEM' in all_code

    def test_fortran_wrapper_has_heap_alloc(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "intent_out_type_array.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        from numpy.f2py.f90mod_rules import findf90modules
        for m in findf90modules(mod[0]):
            type_blocks = derived_type_rules._find_derived_types(m)
            routines = [b for b in m.get('body', [])
                        if b.get('block') in ('subroutine', 'function')]
            src = derived_type_rules.generate_fortran_wrappers(
                m['name'], type_blocks, routines=routines)
        assert src is not None
        assert 'f2py_wrap_make_sequence' in src
        assert 'results_ptrs' in src
        assert 'results_n' in src
        # Heap allocation for each element
        assert 'results_heap_tmp' in src
        assert 'c_loc' in src

    def test_out_args_not_in_kwlist(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "intent_out_type_array.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        all_code = '\n'.join(hooks['f90modhooks'])
        # nresults is intent(out), should not be in kwlist
        assert '"nresults"' not in all_code
        # results is intent(out) array, should not be in kwlist
        assert '"results"' not in all_code
        # start and count should be in kwlist
        assert '"start"' in all_code
        assert '"count"' in all_code


@pytest.mark.slow
class TestIntentOutTypeArray(util.F2PyTest):
    """Test intent(out) allocatable arrays of types with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "intent_out_type_array.f90")]

    def test_indexedval_exists(self):
        assert hasattr(self.module, 'indexedval')

    def test_make_sequence_exists(self):
        assert hasattr(self.module, 'make_sequence')

    def test_make_sequence_basic(self):
        result = self.module.make_sequence(start=1.0, count=3)
        # Returns (results_list, nresults) tuple
        results, nresults = result
        assert nresults == 3
        assert isinstance(results, list)
        assert len(results) == 3
        assert abs(results[0].value - 1.0) < 1e-6
        assert results[0].idx == 1
        assert abs(results[1].value - 2.0) < 1e-6
        assert results[1].idx == 2
        assert abs(results[2].value - 3.0) < 1e-6
        assert results[2].idx == 3

    def test_make_sequence_empty(self):
        result = self.module.make_sequence(start=0.0, count=0)
        results, nresults = result
        assert nresults == 0
        assert isinstance(results, list)
        assert len(results) == 0

    def test_make_sequence_single(self):
        result = self.module.make_sequence(start=5.0, count=1)
        results, nresults = result
        assert nresults == 1
        assert len(results) == 1
        assert abs(results[0].value - 5.0) < 1e-6
        assert results[0].idx == 1

    def test_result_members_accessible(self):
        result = self.module.make_sequence(start=10.0, count=2)
        results, nresults = result
        r = results[0]
        assert hasattr(r, 'value')
        assert hasattr(r, 'idx')
        assert abs(r.value - 10.0) < 1e-6
        assert r.idx == 1


class TestPolymorphicClassNormalization:
    """Test that class(T) is normalized to type(T) by crackfortran."""

    def test_class_parsed_as_type(self):
        import tempfile
        src = (
            "module poly_mod\n"
            "  implicit none\n"
            "  type :: Base\n"
            "    integer :: id\n"
            "  end type\n"
            "contains\n"
            "  subroutine process(obj)\n"
            "    class(Base), intent(in) :: obj\n"
            "  end subroutine\n"
            "end module\n"
        )
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.f90', delete=False) as f:
            f.write(src)
            f.flush()
            mod = crackfortran.crackfortran([f.name])
        for b in mod:
            for sub in b.get('body', []):
                if sub.get('name') == 'process':
                    var = sub['vars'].get('obj', {})
                    # class(Base) should be normalized to type(base)
                    assert var.get('typespec') == 'type'
                    assert var.get('typename') == 'base'
                    return
        raise AssertionError("process subroutine not found in parsed output")


class TestArrayOfTypesCodeGen:
    """Test code generation for arrays of derived types (no compiler)."""

    def test_bindc_array_of_types_detected(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_of_types_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_atom_t' in code
        assert 'f2py_molecule_t' in code

    def test_bindc_array_struct_layout(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_of_types_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_atom_t atoms[3];' in code


@pytest.mark.slow
class TestArrayOfTypesBindC(util.F2PyTest):
    """Test arrays of bind(c) derived types with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "array_of_types_bindc.f90")]

    def test_atom_exists(self):
        assert hasattr(self.module, 'atom')

    def test_molecule_exists(self):
        assert hasattr(self.module, 'molecule')

    def test_molecule_atoms_getter(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        assert isinstance(atoms, list)
        assert len(atoms) == 3

    def test_molecule_atoms_default_zero(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        for a in atoms:
            assert a.x == 0.0
            assert a.y == 0.0
            assert a.z == 0.0
            assert a.atomic_num == 0

    def test_molecule_atoms_setter(self):
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        o = self.module.atom(x=1.0, y=0.0, z=0.0, atomic_num=8)
        h2 = self.module.atom(x=2.0, y=0.0, z=0.0, atomic_num=1)
        m.atoms = [h, o, h2]
        atoms = m.atoms
        assert atoms[0].atomic_num == 1
        assert atoms[1].atomic_num == 8
        assert atoms[1].x == 1.0
        assert atoms[2].x == 2.0

    def test_molecule_atoms_independence(self):
        """Elements in returned list are copies."""
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=1.0, y=2.0, z=3.0, atomic_num=1)
        m.atoms = [h, h, h]
        h.x = 99.0  # modify original
        atoms = m.atoms
        assert atoms[0].x == 1.0  # unchanged

    def test_molecule_natoms(self):
        m = self.module.molecule(natoms=3)
        assert m.natoms == 3


@pytest.mark.slow
class TestArrayOfTypesOpaque(util.F2PyTest):
    """Test arrays of opaque derived types with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "array_of_types_opaque.f90")]

    def test_atom_exists(self):
        assert hasattr(self.module, 'atom')

    def test_molecule_exists(self):
        assert hasattr(self.module, 'molecule')

    def test_molecule_atoms_getter(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        assert isinstance(atoms, list)
        assert len(atoms) == 3

    def test_molecule_atoms_default_zero(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        for a in atoms:
            assert a.x == 0.0
            assert a.y == 0.0
            assert a.z == 0.0
            assert a.atomic_num == 0

    def test_molecule_atoms_setter(self):
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        o = self.module.atom(x=1.0, y=0.0, z=0.0, atomic_num=8)
        h2 = self.module.atom(x=2.0, y=0.0, z=0.0, atomic_num=1)
        m.atoms = [h, o, h2]
        atoms = m.atoms
        assert atoms[0].atomic_num == 1
        assert atoms[1].atomic_num == 8
        assert atoms[1].x == 1.0
        assert atoms[2].x == 2.0

    def test_molecule_atoms_independence(self):
        """Elements in returned list are copies."""
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=1.0, y=2.0, z=3.0, atomic_num=1)
        m.atoms = [h, h, h]
        h.x = 99.0  # modify original
        atoms = m.atoms
        assert atoms[0].x == 1.0  # unchanged

    def test_molecule_natoms(self):
        m = self.module.molecule(natoms=3)
        assert m.natoms == 3


class TestArrayOfTypesCodeGen:
    """Test code generation for arrays of derived types (no compiler)."""

    def test_bindc_array_of_types_detected(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_of_types_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_atom_t' in code
        assert 'f2py_molecule_t' in code

    def test_bindc_array_struct_layout(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_of_types_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_atom_t atoms[3];' in code


@pytest.mark.slow
class TestArrayOfTypesBindC(util.F2PyTest):
    """Test arrays of bind(c) derived types with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "array_of_types_bindc.f90")]

    def test_atom_exists(self):
        assert hasattr(self.module, 'atom')

    def test_molecule_exists(self):
        assert hasattr(self.module, 'molecule')

    def test_molecule_atoms_getter(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        assert isinstance(atoms, list)
        assert len(atoms) == 3

    def test_molecule_atoms_default_zero(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        for a in atoms:
            assert a.x == 0.0
            assert a.y == 0.0
            assert a.z == 0.0
            assert a.atomic_num == 0

    def test_molecule_atoms_setter(self):
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        o = self.module.atom(x=1.0, y=0.0, z=0.0, atomic_num=8)
        h2 = self.module.atom(x=2.0, y=0.0, z=0.0, atomic_num=1)
        m.atoms = [h, o, h2]
        atoms = m.atoms
        assert atoms[0].atomic_num == 1
        assert atoms[1].atomic_num == 8
        assert atoms[1].x == 1.0
        assert atoms[2].x == 2.0

    def test_molecule_atoms_independence(self):
        """Elements in returned list are copies."""
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=1.0, y=2.0, z=3.0, atomic_num=1)
        m.atoms = [h, h, h]
        h.x = 99.0  # modify original
        atoms = m.atoms
        assert atoms[0].x == 1.0  # unchanged

    def test_molecule_natoms(self):
        m = self.module.molecule(natoms=3)
        assert m.natoms == 3


@pytest.mark.slow
class TestArrayOfTypesOpaque(util.F2PyTest):
    """Test arrays of opaque derived types with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "array_of_types_opaque.f90")]

    def test_atom_exists(self):
        assert hasattr(self.module, 'atom')

    def test_molecule_exists(self):
        assert hasattr(self.module, 'molecule')

    def test_molecule_atoms_getter(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        assert isinstance(atoms, list)
        assert len(atoms) == 3

    def test_molecule_atoms_default_zero(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        for a in atoms:
            assert a.x == 0.0
            assert a.y == 0.0
            assert a.z == 0.0
            assert a.atomic_num == 0

    def test_molecule_atoms_setter(self):
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        o = self.module.atom(x=1.0, y=0.0, z=0.0, atomic_num=8)
        h2 = self.module.atom(x=2.0, y=0.0, z=0.0, atomic_num=1)
        m.atoms = [h, o, h2]
        atoms = m.atoms
        assert atoms[0].atomic_num == 1
        assert atoms[1].atomic_num == 8
        assert atoms[1].x == 1.0
        assert atoms[2].x == 2.0

    def test_molecule_atoms_independence(self):
        """Elements in returned list are copies."""
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=1.0, y=2.0, z=3.0, atomic_num=1)
        m.atoms = [h, h, h]
        h.x = 99.0  # modify original
        atoms = m.atoms
        assert atoms[0].x == 1.0  # unchanged

    def test_molecule_natoms(self):
        m = self.module.molecule(natoms=3)
        assert m.natoms == 3


@pytest.mark.slow
class TestNestedOpaque(util.F2PyTest):
    """Test nested opaque (non-bind(c)) derived types with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "nested_opaque.f90")]

    def test_vec2_exists(self):
        assert hasattr(self.module, 'vec2')

    def test_particle_exists(self):
        assert hasattr(self.module, 'particle')

    def test_particle_has_nested_members(self):
        p = self.module.particle(mass=1.0)
        pos = p.pos
        assert hasattr(pos, 'x')
        assert hasattr(pos, 'y')

    def test_particle_nested_getter(self):
        p = self.module.particle(mass=2.5)
        pos = p.pos
        assert pos.x == 0.0
        assert pos.y == 0.0

    def test_particle_nested_setter(self):
        p = self.module.particle(mass=1.0)
        v = self.module.vec2(x=3.0, y=4.0)
        p.pos = v
        pos = p.pos
        assert pos.x == 3.0
        assert pos.y == 4.0

    def test_particle_nested_independence(self):
        """Nested getter returns a copy, not a reference."""
        p = self.module.particle(mass=1.0)
        v = self.module.vec2(x=1.0, y=2.0)
        p.pos = v
        v.x = 99.0
        pos = p.pos
        assert pos.x == 1.0  # unchanged

    def test_particle_mass(self):
        p = self.module.particle(mass=42.0)
        assert p.mass == 42.0

    def test_particle_both_nested(self):
        p = self.module.particle(mass=1.0)
        pos = self.module.vec2(x=1.0, y=2.0)
        vel = self.module.vec2(x=3.0, y=4.0)
        p.pos = pos
        p.vel = vel
        assert p.pos.x == 1.0
        assert p.vel.y == 4.0


class TestArrayOfTypesCodeGen:
    """Test code generation for arrays of derived types (no compiler)."""

    def test_bindc_array_of_types_detected(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_of_types_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_atom_t' in code
        assert 'f2py_molecule_t' in code

    def test_bindc_array_struct_layout(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "array_of_types_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        hooks = derived_type_rules.buildhooks(mod[0])
        code = '\n'.join(hooks['f90modhooks'])
        assert 'f2py_atom_t atoms[3];' in code


@pytest.mark.slow
class TestArrayOfTypesBindC(util.F2PyTest):
    """Test arrays of bind(c) derived types with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "array_of_types_bindc.f90")]

    def test_atom_exists(self):
        assert hasattr(self.module, 'atom')

    def test_molecule_exists(self):
        assert hasattr(self.module, 'molecule')

    def test_molecule_atoms_getter(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        assert isinstance(atoms, list)
        assert len(atoms) == 3

    def test_molecule_atoms_default_zero(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        for a in atoms:
            assert a.x == 0.0
            assert a.y == 0.0
            assert a.z == 0.0
            assert a.atomic_num == 0

    def test_molecule_atoms_setter(self):
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        o = self.module.atom(x=1.0, y=0.0, z=0.0, atomic_num=8)
        h2 = self.module.atom(x=2.0, y=0.0, z=0.0, atomic_num=1)
        m.atoms = [h, o, h2]
        atoms = m.atoms
        assert atoms[0].atomic_num == 1
        assert atoms[1].atomic_num == 8
        assert atoms[1].x == 1.0
        assert atoms[2].x == 2.0

    def test_molecule_atoms_independence(self):
        """Elements in returned list are copies."""
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=1.0, y=2.0, z=3.0, atomic_num=1)
        m.atoms = [h, h, h]
        h.x = 99.0  # modify original
        atoms = m.atoms
        assert atoms[0].x == 1.0  # unchanged

    def test_molecule_natoms(self):
        m = self.module.molecule(natoms=3)
        assert m.natoms == 3


@pytest.mark.slow
class TestArrayOfTypesOpaque(util.F2PyTest):
    """Test arrays of opaque derived types with compilation."""
    sources = [util.getpath("tests", "src", "derived_types",
                            "array_of_types_opaque.f90")]

    def test_atom_exists(self):
        assert hasattr(self.module, 'atom')

    def test_molecule_exists(self):
        assert hasattr(self.module, 'molecule')

    def test_molecule_atoms_getter(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        assert isinstance(atoms, list)
        assert len(atoms) == 3

    def test_molecule_atoms_default_zero(self):
        m = self.module.molecule(natoms=3)
        atoms = m.atoms
        for a in atoms:
            assert a.x == 0.0
            assert a.y == 0.0
            assert a.z == 0.0
            assert a.atomic_num == 0

    def test_molecule_atoms_setter(self):
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=0.0, y=0.0, z=0.0, atomic_num=1)
        o = self.module.atom(x=1.0, y=0.0, z=0.0, atomic_num=8)
        h2 = self.module.atom(x=2.0, y=0.0, z=0.0, atomic_num=1)
        m.atoms = [h, o, h2]
        atoms = m.atoms
        assert atoms[0].atomic_num == 1
        assert atoms[1].atomic_num == 8
        assert atoms[1].x == 1.0
        assert atoms[2].x == 2.0

    def test_molecule_atoms_independence(self):
        """Elements in returned list are copies."""
        m = self.module.molecule(natoms=3)
        h = self.module.atom(x=1.0, y=2.0, z=3.0, atomic_num=1)
        m.atoms = [h, h, h]
        h.x = 99.0  # modify original
        atoms = m.atoms
        assert atoms[0].x == 1.0  # unchanged

    def test_molecule_natoms(self):
        m = self.module.molecule(natoms=3)
        assert m.natoms == 3


class TestComplexMemberCodeGen:
    """Test C code generation for types with complex-valued members."""

    def test_bindc_complex_struct(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "complex_member_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        all_code = '\n'.join(hooks['f90modhooks'])
        # complex.h in needs (placed in #includes# section)
        assert 'complex.h' in hooks.get('need', [])
        # C struct has complex fields
        assert 'double _Complex reactance' in all_code
        # Getter uses creal/cimag
        assert 'PyComplex_FromDoubles' in all_code
        # Setter uses _Complex_I
        assert '_Complex_I' in all_code

    def test_opaque_complex_extern_decls(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "complex_member_opaque.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        all_code = '\n'.join(hooks['f90modhooks'])
        assert 'double _Complex' in all_code
        assert 'f2py_get_wavefunction_amplitude' in all_code
        assert 'f2py_set_wavefunction_amplitude' in all_code

    def test_complex_repr(self):
        fpath = util.getpath("tests", "src", "derived_types",
                             "complex_member_bindc.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        module = mod[0]
        hooks = derived_type_rules.buildhooks(module)

        all_code = '\n'.join(hooks['f90modhooks'])
        # repr should include (%g+%gj) format for complex
        assert '%g+%gj' in all_code


@pytest.mark.slow
class TestComplexMemberBindC(util.F2PyTest):
    sources = [util.getpath("tests", "src", "derived_types",
                            "complex_member_bindc.f90")]

    def test_impedance_exists(self):
        assert hasattr(self.module, 'impedance')

    def test_impedance_create(self):
        z = self.module.impedance(resistance=50.0, reactance=10+20j)
        assert z.resistance == 50.0
        assert z.reactance == (10+20j)

    def test_impedance_set_complex(self):
        z = self.module.impedance()
        z.reactance = 3+4j
        assert z.reactance == (3+4j)

    def test_impedance_real_only(self):
        z = self.module.impedance(reactance=5.0)
        assert z.reactance == (5+0j)

    def test_impedance_repr(self):
        z = self.module.impedance(resistance=100.0, reactance=1+2j)
        r = repr(z)
        assert 'impedance' in r
        assert '100' in r

    def test_signal_sample_exists(self):
        assert hasattr(self.module, 'signal_sample')

    def test_signal_sample_float_complex(self):
        s = self.module.signal_sample(value=1.5+2.5j, timestamp=0.1)
        assert abs(s.value - (1.5+2.5j)) < 1e-6
        assert abs(s.timestamp - 0.1) < 1e-6


@pytest.mark.slow
class TestComplexMemberOpaque(util.F2PyTest):
    sources = [util.getpath("tests", "src", "derived_types",
                            "complex_member_opaque.f90")]

    def test_wavefunction_exists(self):
        assert hasattr(self.module, 'wavefunction')

    def test_wavefunction_create(self):
        wf = self.module.wavefunction(amplitude=1+0j, phase=0.0)
        assert wf.amplitude == (1+0j)
        assert wf.phase == 0.0

    def test_wavefunction_set_complex(self):
        wf = self.module.wavefunction()
        wf.amplitude = 0.5+0.5j
        assert wf.amplitude == (0.5+0.5j)

    def test_wavefunction_roundtrip(self):
        wf = self.module.wavefunction(amplitude=3-4j, phase=1.57)
        assert wf.amplitude == (3-4j)
        wf.amplitude = -1+2j
        assert wf.amplitude == (-1+2j)
        assert wf.phase == 1.57
