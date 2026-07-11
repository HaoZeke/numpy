project('f2py_examples', 'c',
  version : '0.1',
  license: 'BSD-3',
  meson_version: '>=1.1.0',
  default_options : ['warning_level=2'],
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

# NumPy >= 2.0 ships pkg-config metadata, so the include directories
# resolve without probing the interpreter. This also avoids the
# absolute-path rejection meson raises when a virtual environment
# lives inside the source tree (gh-26973).
np_dep = dependency('numpy')

incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_f2py = include_directories(incdir_f2py)

py.extension_module('fib2',
  [
    'fib1.f',
    'fib2module.c',  # note: this assumes f2py was manually run before!
    incdir_f2py / 'fortranobject.c',
  ],
  include_directories: inc_f2py,
  dependencies : [py_dep, np_dep],
  install : true
)
