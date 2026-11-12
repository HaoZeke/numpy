project('f2py_examples', 'c',
  version : '0.1',
  license: 'BSD-3',
  meson_version: '>=1.3.0',
  default_options : ['warning_level=2'],
)

add_languages('fortran')

fs = import('fs')
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

# Meson rejects absolute paths into the project directory, which is where
# NumPy lives when the virtual environment does, so pass relative ones
inc_np = include_directories(
  fs.relative_to(incdir_numpy, meson.current_source_dir()),
  fs.relative_to(incdir_f2py, meson.current_source_dir()),
)

py.extension_module('fib2',
  [
    'fib1.f',
    'fib2module.c',  # note: this assumes f2py was manually run before!
  ],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  link_language: 'fortran',
  install : true
)
