====================
Using Python as glue
====================

Python is often used as a "glue" language: compiled code written in C, C++, or
Fortran does the heavy computation, and Python code combines the pieces,
handles input and output, and provides a convenient interface. NumPy arrays
make this practical, because the data of an array is a block of memory
described by a data type, a shape, and strides, which compiled code can work
with directly (see :ref:`memory-layout`).

This page describes the main ways to call compiled code from Python and pass
NumPy arrays to it. It is an overview, not a complete guide to any one tool;
each section links to the documentation of the tool it describes. To help you
compare the approaches, the F2PY, Cython, and ctypes sections implement the
same two functions:

* ``zadd`` adds two one-dimensional arrays of complex numbers, element by
  element.
* ``dfilter2d`` applies a 3x3 averaging filter to the interior of a
  two-dimensional array of ``float64`` values.


Calling compiled code from Python
=================================

Every operation that the Python interpreter executes has some overhead, so a
Python loop over the elements of an array is much slower than the same loop in
a compiled language. NumPy's array operations avoid this overhead for many
tasks. You may still need compiled code when an algorithm cannot be written
efficiently with array operations, or when you want to use an existing library
written in C, C++, or Fortran.

There are two basic approaches:

* Write an *extension module*: a shared library that uses the Python C API, and
  that you import like any other Python module. You can write the module by
  hand, or generate it with a tool such as F2PY or Cython.
* Load an ordinary shared library at runtime and call its functions directly,
  with the :mod:`ctypes` module of the Python standard library or with a
  package such as CFFI.

.. warning::

    Compiled code can crash the Python interpreter, for example by reading or
    writing outside the memory of an array. None of the approaches on this page
    protect against errors in the compiled code itself. You need to know how
    the compiled code expects its data to be laid out, and how NumPy lays out
    the data of the arrays that you pass to it.


Hand-written extension modules
==============================

:ref:`writing-an-extension` explains how to write an extension module by hand
with the Python and NumPy C APIs. The Python documentation describes the
general process in `Extending and Embedding the Python Interpreter
<https://docs.python.org/3/extending/index.html>`_.

A hand-written wrapper function typically:

1. Converts its Python arguments to C types, usually with
   :c:func:`PyArg_ParseTuple`. For arrays, you can use converter functions
   such as :c:func:`PyArray_Converter` with the ``"O&"`` format.
2. Obtains arrays with the data type, number of dimensions, and memory layout
   that the compiled code expects, for example with
   :c:func:`PyArray_FROM_OTF`.
3. Calls the compiled code with pointers to the array data.
4. Converts the result to a Python object.

A hand-written wrapper gives you complete control over the interface, with no
overhead beyond the calls that you make. The cost is that you have to write,
debug, and maintain the C code yourself, including reference counting and
error handling. The tools described in the rest of this page generate this
code for you.

To call Fortran code from a hand-written C wrapper, you also need to know how
your Fortran compiler names procedures and passes arguments. Unless the
procedure has the ``BIND(C)`` attribute, this depends on the compiler: many
compilers append an underscore to the lowercase name of the procedure, pass all
arguments by reference (as pointers), and pass the lengths of character
arguments as extra hidden arguments. Fortran 2003 and later standardize
interoperability with C: the ``BIND(C)`` attribute gives a procedure a C name
and calling convention, the kind parameters of the ``ISO_C_BINDING`` intrinsic
module, such as ``C_DOUBLE``, give its arguments C-compatible types, and the
``VALUE`` attribute passes an argument by value. F2PY, described in the next
section, handles these details for you.


F2PY
====

.. index::
   single: f2py

F2PY (:mod:`numpy.f2py`) is included with NumPy and generates extension modules
for Fortran code. It reads Fortran source files, or signature files (``.pyf``)
that describe the interface to the Fortran code, writes the C code for the
extension module and, when called with ``-c``, compiles and links everything
with Meson and your Fortran compiler. The generated wrapper functions convert
NumPy arrays and other array-like objects to the data type, shape, and memory
layout (column-major, or "Fortran", order) that the Fortran code expects, and
copy the data only when the input does not already match.

Save the complex addition routine as ``add.f``:

.. literalinclude:: ../f2py/code/add-improved.f
   :language: fortran

Fortran compilers treat the lines that start with ``CF2PY`` as comments, but
F2PY reads them as directives: ``INTENT(OUT)`` makes ``C`` the return value of
the Python function, ``INTENT(HIDE)`` removes ``N`` from its arguments, and the
declarations ``A(N)``, ``B(N)``, and ``C(N)`` tell F2PY that ``N`` is the length
of the arrays. Save the averaging filter as ``filter.f``:

.. literalinclude:: ../f2py/code/filter.f
   :language: fortran

Here, ``N`` and ``M`` are hidden and taken from the shape of ``A``, and ``B``
is returned. Build the two extension modules with:

.. code-block:: bash

    python -m numpy.f2py -c -m add add.f
    python -m numpy.f2py -c -m filter filter.f

Then use them from Python:

.. code-block:: python

    import numpy as np

    import add
    import filter

    add.zadd([1, 2, 3], [4j, 5j, 6j])  # array([1.+4.j, 2.+5.j, 3.+6.j])

    img = np.random.default_rng().random((200, 300))
    out = filter.dfilter2d(img)

The wrappers check their arguments. For example, ``add.zadd`` raises a
``ValueError`` if the two arrays have different lengths, instead of reading
past the end of the shorter one. Because ``img`` is in C order, F2PY passes a
Fortran-ordered copy of it to ``DFILTER2D``. To avoid the copy for large
arrays, create them in Fortran order, for example with
:func:`numpy.asfortranarray`. The returned array ``out`` is in Fortran order.

Fortran supports complex numbers and multidimensional arrays directly, so
numerical code in Fortran is often shorter than the equivalent C code; compare
``filter.f`` with the C version in the ctypes example below. Building an F2PY
extension module requires a Fortran compiler, such as gfortran or LLVM Flang,
in addition to a C compiler.

For more information, see the :ref:`F2PY documentation <f2py>`, in
particular :ref:`f2py-examples`, which explains these examples step by step,
and :ref:`f2py-bldsys`, which explains how to build F2PY extension modules as
part of a Python package.


Cython
======

.. index::
   single: cython

`Cython <https://cython.org>`_ is a compiler for a superset of the Python
language that adds optional static type declarations and lets you call C and
C++ functions directly. It translates a Cython module (a ``.pyx`` file) to C or
C++ code, which is then compiled into an extension module. Cython is a good fit
when the extension module contains your own algorithms, not only calls to an
existing library, because Cython code can loop over typed, multidimensional
arrays at the speed of C. See `Cython for NumPy users
<https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html>`_
in the Cython documentation for a tutorial.


Building a Cython extension module
----------------------------------

To build and install a Cython extension module as part of a Python package,
use a build system that supports Cython. The following example uses the
`meson-python <https://mesonbuild.com/meson-python/>`_ build backend and the
`Meson <https://mesonbuild.com>`_ build system, which NumPy itself uses.

First, create a file named ``my_extension.pyx``:

.. code-block:: cython

   cimport numpy as np

   def say_hello():
       print("Hello!")

In the same directory, create an ``__init__.py`` file with the following
content:

.. code-block:: python

   from .my_extension import say_hello

Next, create two files that configure the build. The first is
``meson.build``:

.. code-block:: meson

  project(
    'module_with_extension',
    'c', 'cython',
    version: '0.0.1',
    license: 'MIT',
  )

  py = import('python').find_installation(pure: false)

  numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION']

  np_dep = declare_dependency(dependencies: dependency('numpy'),
                              compile_args: numpy_nodepr_api)

  py.extension_module(
    'my_extension',
    'my_extension.pyx',
    dependencies: [np_dep],
    install: true,
    subdir: 'my_module_with_extension',
  )

  py.install_sources(
    '__init__.py',
    subdir: 'my_module_with_extension',
  )

The second is ``pyproject.toml``:

.. code-block:: toml

   [build-system]
   build-backend = "mesonpy"
   requires = [
       "meson-python",
       "Cython>=3.0.0",
       "numpy",
   ]

   [project]
   name = "my_module_with_extension"
   version = "0.0.1"
   license = "MIT"
   dependencies = ["numpy"]

You can now build and install the package, and call the function defined in
the extension module from Python:

.. code-block:: bash

   $ pip install .
   $ python -c "from my_module_with_extension import say_hello; say_hello()"
   Hello!

The NumPy dependency in ``meson.build`` is needed only if the module uses the
NumPy C API through ``cimport numpy``, as the following examples do. Cython can
also compile ordinary Python code, which usually makes it run somewhat faster;
declaring the C types of variables with ``cdef`` gives larger speed-ups.

See the Meson and meson-python documentation for more details on how to build
more complicated extension modules.


Complex addition in Cython
--------------------------

Here is a Cython module named ``add.pyx`` that implements the complex addition
function:

.. code-block:: cython

    cimport cython
    cimport numpy as np
    import numpy as np

    # Initialize the NumPy C API.
    np.import_array()

    #@cython.boundscheck(False)
    def zadd(in1, in2):
        cdef double complex[:] a = np.asarray(in1, dtype=np.complex128).ravel()
        cdef double complex[:] b = np.asarray(in2, dtype=np.complex128).ravel()

        out = np.empty(a.shape[0], np.complex128)
        cdef double complex[:] c = out

        for i in range(c.shape[0]):
            c[i] = a[i] + b[i]

        return out

The ``cimport`` statement loads the declarations of the NumPy C API from the
``.pxd`` files that are installed with NumPy. NumPy is imported twice:
``cimport`` makes the NumPy C API available, and the regular ``import`` imports
NumPy at runtime so that you can use its Python API.

The module also uses Cython's *typed memoryviews*, which are similar to NumPy
arrays at the C level: they are shaped and strided arrays that know their own
extent, unlike a C array addressed through a bare pointer. The syntax ``double
complex[:]`` denotes a one-dimensional array of complex numbers in double
precision, with arbitrary strides. A contiguous array of ints would be
``int[::1]``, and a matrix of floats would be ``float[:, :]``. See `Typed
Memoryviews <https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html>`_
in the Cython documentation for details.

The commented-out ``cython.boundscheck`` decorator turns bounds checking of
memoryview accesses on or off for a single function. Turning it off makes the
code faster, at the expense of safety: here, an array ``in2`` that is shorter
than ``in1`` would then be read past its end instead of raising an
``IndexError``.

Apart from the memoryview declarations, the function reads like Python code.
Cython infers the C type of the loop variable ``i`` from the ``range`` call.
Cython also has a special syntax for NumPy arrays, but typed memoryviews are
preferred.


Image filter in Cython
----------------------

The averaging filter is similarly concise in Cython:

.. code-block:: cython

    cimport numpy as np
    import numpy as np

    np.import_array()

    def filter(img):
        cdef double[:, :] a = np.asarray(img, dtype=np.double)
        out = np.zeros(img.shape, dtype=np.double)
        cdef double[:, ::1] b = out

        cdef np.npy_intp i, j

        for i in range(1, a.shape[0] - 1):
            for j in range(1, a.shape[1] - 1):
                b[i, j] = (a[i, j]
                           + .5 * (  a[i-1, j] + a[i+1, j]
                                   + a[i, j-1] + a[i, j+1])
                           + .25 * (  a[i-1, j-1] + a[i-1, j+1]
                                    + a[i+1, j-1] + a[i+1, j+1]))

        return out

The loops run at the speed of C, and the index computations are done only as
needed. If the code above is compiled as a module named ``image``, you can
filter a two-dimensional array ``img`` with:

.. code-block:: python

    import image
    out = image.filter(img)

Two things are worth noting about this code. First, a memoryview cannot be
returned to Python as an array, so the function creates a NumPy array ``out``
and computes the result through a view ``b`` of it. Second, ``b`` is typed
``double[:, ::1]``, a two-dimensional array with contiguous rows (C order).
Specifying the memory layout explicitly can speed up some algorithms, because
they can skip stride computations.


Advantages and disadvantages
----------------------------

Cython is used to write extension modules by many scientific Python libraries,
including SciPy, pandas, scikit-learn, and scikit-image, and by the XML
processing library lxml. The language and compiler are actively maintained.

Keep in mind that:

1. When you write your own algorithms, and sometimes when you wrap existing C
   libraries, you need some familiarity with C. In particular, when you manage
   memory with ``malloc`` and related functions, it is easy to introduce memory
   leaks. However, compiling a Python module renamed to ``.pyx`` can already
   speed it up, and adding a few type declarations can give dramatic speed-ups
   for some code.

2. It is easy to lose a clean separation between Python and C, which makes it
   harder to reuse your C code in projects that do not use Python.

3. The C code that Cython generates is hard to read and modify, and its
   compilation typically produces harmless warnings.

Cython-generated extension modules are distributed like any other extension
module. Cython is a capable tool both for wrapping C code and for writing
fast extension modules, and is especially useful if you prefer not to write C
or Fortran code yourself.


ctypes
======

.. index::
   single: ctypes

:mod:`ctypes` is a module of the Python standard library that loads a shared
library and calls the functions in it directly from Python, without an
extension module. This gives you access to any library with a C interface,
without compiling anything specific to Python. The drawback is that ctypes does
very little checking: if the types of the arguments, or the sizes of the arrays,
do not match what the function expects, the result is wrong answers or a crash,
as in C. This is especially true when you pass the memory address of the
data of an array, because the function has no way to know where the array
ends.

Robust use of ctypes therefore involves a layer of Python code that checks the
data types and sizes of the arrays before it calls the function. These checks,
and the conversion of the arguments to C types that ctypes itself performs,
make each call slower than a call to a function in an extension module.
However, the overhead is negligible if the function does a significant amount
of work.

To use ctypes, you:

1. Build a shared library, or use an existing one.
2. Load the shared library.
3. Convert the Python objects to arguments that ctypes understands.
4. Call the function from the library with these arguments.


Building a shared library
-------------------------

How you build a shared library depends on your platform and compiler. Keep in
mind that:

- A shared library must be compiled and linked in a special way, for example
  with the ``-shared`` and ``-fPIC`` options of GCC and Clang.

- On Windows, the functions that you want to call must be exported from the
  DLL. You can export them with the ``__declspec(dllexport)`` storage-class
  specifier in the C definition of each function, or with a ``.def`` file that
  lists them. For example, a ``mylib.def`` file might contain::

      LIBRARY mylib.dll
      EXPORTS
      cool_function1
      cool_function2

Build systems such as Meson can build shared libraries on all platforms (see
`shared_library() <https://mesonbuild.com/Reference-manual_functions.html#shared_library>`_
in the Meson documentation), and meson-python can install a shared library
inside your Python package (see `Using shared libraries
<https://mesonbuild.com/meson-python/how-to-guides/shared-libraries.html>`_
in the meson-python documentation).


Loading the shared library
--------------------------

You can load a shared library from its full path with the ``cdll`` object of
ctypes:

.. code-block:: python

    lib = ctypes.cdll[<full_path_name>]

If you load a library by name instead of by path, as in ``ctypes.cdll.mylib``,
the platform's library search path is used, which may find a different library
with the same name. The :func:`ctypes.util.find_library` function can help you
find the name of a system library, but it is not foolproof.

Because the file name extension of shared libraries varies between platforms
(for example, ``.so`` on Linux, ``.dylib`` on macOS, and ``.dll`` on Windows),
:func:`numpy.ctypeslib.load_library` is often more convenient. It takes the
name of the library, including any prefix such as ``lib`` but without the
extension, and the directory that contains the library. It returns a ctypes
library object, or raises an ``OSError`` if the library cannot be found or
loaded. ``load_library`` always loads the library with the cdecl calling
convention. On Windows, see ``ctypes.windll`` and ``ctypes.oledll`` in the
ctypes documentation for ways to load libraries that use other calling
conventions.


Converting arguments
--------------------

ctypes automatically converts Python integers, bytes, and strings to the
equivalent C types, and ``None`` to a ``NULL`` pointer. Other objects must be
converted to ctypes types. ctypes offers two ways to pass other objects to a
function:

1. Give the object that you want to pass an ``_as_parameter_`` attribute.
   ctypes passes the value of this attribute to the function instead of the
   object. The value must be an integer, bytes, a string, a ctypes instance, or
   another object with an ``_as_parameter_`` attribute.

2. Set the ``argtypes`` attribute of the function to a list with one type per
   argument. Each type has a class method named ``from_param`` that checks an
   argument and converts it to an object that ctypes understands, as listed
   for the first method.

NumPy supports both methods, and the second one is safer, because it can check
the arrays that you pass. The
:attr:`~numpy.ndarray.ctypes` attribute of an array returns an object whose
``_as_parameter_`` attribute is a ``c_void_p`` pointer to the data of the
array, so you can pass ``arr.ctypes`` directly to a function that expects a
pointer to the data. You are then responsible for passing an array with the
correct data type, shape, and flags; passing an inappropriate array can crash
the interpreter.

For the second method, the class factory :func:`numpy.ctypeslib.ndpointer`
creates a class that you can use as an entry of the ``argtypes`` attribute.
Its ``from_param`` method checks the properties of each array that you pass:
any combination of the data type, the number of dimensions, the shape, and the
flags that you specify in the call to ``ndpointer``. If the array does not
match, ``from_param`` raises a ``TypeError``; otherwise, it returns the
``ctypes`` attribute of the array, which ctypes passes on as a pointer to the
array data.

The ``ctypes`` attribute of an array provides more information that you may
want to pass to a function. Its ``data`` attribute is the address of the data
as a Python integer, and its ``shape`` and ``strides`` attributes are ctypes
arrays of :class:`numpy.ctypeslib.c_intp` integers (or ``None`` for a 0-d
array). The ``data_as``, ``shape_as``, and ``strides_as`` methods return the
same information as ctypes types of your choice. Unlike ``data``, the object
that ``data_as`` returns keeps a reference to the array, so the memory stays
valid while you use the pointer.


Calling the function
--------------------

The functions of a shared library are available both as attributes and as
items of the library object; use items for function names that are not valid
Python identifiers. For example, if ``./mylib.so`` contains a function named
``cool_function1``, you can access it with either:

.. code-block:: python

    lib = numpy.ctypeslib.load_library('mylib', '.')
    func1 = lib.cool_function1  # or equivalently
    func1 = lib['cool_function1']

By default, ctypes assumes that a function returns an ``int``. To change this,
set the ``restype`` attribute of the function. Use ``None`` for a function that
does not return anything (``void``):

.. code-block:: python

    func1.restype = None

To have ctypes check the arguments of each call, set the ``argtypes``
attribute, using :func:`numpy.ctypeslib.ndpointer` for array arguments. Using
``ndpointer`` makes it significantly safer to pass arrays to a C function. You
may still want to wrap the function in a Python function that is easier to use,
for example by hiding arguments that can be computed from others, or by
creating output arrays. :func:`numpy.require` is useful in such a wrapper to
get an array with the required data type and flags from any input.


Complete example
----------------

This example implements the complex addition function and the averaging filter
with ctypes. The C code is in a file named ``code.c``. It contains a ``zadd``
function, together with similar functions ``cadd``, ``dadd``, and ``sadd`` that
handle the ``complex64``, ``float64``, and ``float32`` data types:

.. code-block:: c

    #include <stddef.h>

    typedef struct {double real; double imag;} cdouble;
    typedef struct {float real; float imag;} cfloat;

    /* Add arrays of contiguous data */
    void zadd(cdouble *a, cdouble *b, cdouble *c, ptrdiff_t n)
    {
        while (n--) {
            c->real = a->real + b->real;
            c->imag = a->imag + b->imag;
            a++; b++; c++;
        }
    }

    void cadd(cfloat *a, cfloat *b, cfloat *c, ptrdiff_t n)
    {
        while (n--) {
            c->real = a->real + b->real;
            c->imag = a->imag + b->imag;
            a++; b++; c++;
        }
    }

    void dadd(double *a, double *b, double *c, ptrdiff_t n)
    {
        while (n--) {
            *c++ = *a++ + *b++;
        }
    }

    void sadd(float *a, float *b, float *c, ptrdiff_t n)
    {
        while (n--) {
            *c++ = *a++ + *b++;
        }
    }

The sizes, strides, and shapes are of type ``ptrdiff_t``, a signed integer
type that has the same size as :class:`numpy.ctypeslib.c_intp` on the
platforms that NumPy supports. The file also contains the ``dfilter2d``
function:

.. code-block:: c

    /*
     * Assumes b is contiguous and a has strides that are multiples of
     * sizeof(double)
     */
    void
    dfilter2d(double *a, double *b, ptrdiff_t *astrides, ptrdiff_t *dims)
    {
        ptrdiff_t i, j, M, N, S0, S1;
        ptrdiff_t r, c, rm1, rp1, cp1, cm1;

        M = dims[0]; N = dims[1];
        S0 = astrides[0] / (ptrdiff_t)sizeof(double);
        S1 = astrides[1] / (ptrdiff_t)sizeof(double);
        for (i = 1; i < M - 1; i++) {
            r = i*S0;
            rp1 = r + S0;
            rm1 = r - S0;
            for (j = 1; j < N - 1; j++) {
                c = j*S1;
                cp1 = c + S1;
                cm1 = c - S1;
                b[i*N + j] = a[r + c] +
                    (a[rp1 + c] + a[rm1 + c] +
                     a[r + cp1] + a[r + cm1])*0.5 +
                    (a[rp1 + cp1] + a[rp1 + cm1] +
                     a[rm1 + cp1] + a[rm1 + cm1])*0.25;
            }
        }
    }

Unlike the Fortran version, to which F2PY passes a Fortran-ordered copy of the
input when necessary, this function works directly on arrays with arbitrary
strides, so non-contiguous input is not copied. The price is more complicated
indexing code than in ``filter.f``.

Compile the code into a shared library. On Linux, for example, use::

    gcc -shared -fPIC -o code.so code.c

This creates a shared library named ``code.so`` in the current directory. On
Windows, either add ``__declspec(dllexport)`` in front of ``void`` in each
function definition, or write a ``code.def`` file that lists the names of the
functions to export.

Next, write a Python interface to the shared library. Create a file named
``interface.py`` that starts with the following lines:

.. code-block:: python

    import ctypes
    import os

    import numpy as np

    __all__ = ['add', 'filter2d']

    _path = os.path.dirname(__file__)
    lib = np.ctypeslib.load_library('code', _path)
    _typedict = {'zadd': np.complex128, 'sadd': np.float32,
                 'cadd': np.complex64, 'dadd': np.float64}
    for name, _type in _typedict.items():
        func = getattr(lib, name)
        func.restype = None
        func.argtypes = [
            np.ctypeslib.ndpointer(_type, flags='aligned, contiguous'),
            np.ctypeslib.ndpointer(_type, flags='aligned, contiguous'),
            np.ctypeslib.ndpointer(_type,
                                   flags='aligned, contiguous, writeable'),
            np.ctypeslib.c_intp,
        ]

This code loads the shared library ``code`` from the directory that contains
``interface.py``. It sets the return type of the addition functions to
``void``, and makes ctypes check that the first three arguments are aligned,
contiguous arrays of the right data type, the third one writeable, and that
the fourth argument is an integer.

The setup of the filtering function is similar. Its first two arguments must
be two-dimensional ``float64`` arrays, and the last two are pointers to
``c_intp`` integers, for the strides and the shape of the input array:

.. code-block:: python

    lib.dfilter2d.restype = None
    lib.dfilter2d.argtypes = [
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='aligned'),
        np.ctypeslib.ndpointer(np.float64, ndim=2,
                               flags='aligned, contiguous, writeable'),
        ctypes.POINTER(np.ctypeslib.c_intp),
        ctypes.POINTER(np.ctypeslib.c_intp),
    ]

Next, define a function that selects the addition function to call, based on
the data type of the input:

.. code-block:: python

    def select(dtype):
        if dtype.char in '?bBhHf':
            return lib.sadd, np.float32
        elif dtype.char == 'F':
            return lib.cadd, np.complex64
        elif dtype.char in 'DG':
            return lib.zadd, np.complex128
        else:
            return lib.dadd, np.float64

Finally, write the two functions that the interface exports:

.. code-block:: python

    def add(a, b):
        requires = ['CONTIGUOUS', 'ALIGNED']
        a = np.asanyarray(a)
        func, dtype = select(a.dtype)
        a = np.require(a, dtype, requires)
        b = np.require(b, dtype, requires)
        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        c = np.empty_like(a)
        func(a, b, c, a.size)
        return c

and:

.. code-block:: python

    def filter2d(a):
        a = np.require(a, np.float64, ['ALIGNED'])
        b = np.zeros(a.shape, dtype=np.float64)
        lib.dfilter2d(a, b, a.ctypes.strides, a.ctypes.shape)
        return b

The shape check in ``add`` is necessary because ``ndpointer`` does not know
that the arrays must have the same size: without it, a shorter ``b`` would be
read past its end. ``filter2d`` creates the output array with
:func:`numpy.zeros` because ``dfilter2d`` requires it to be C-contiguous,
whatever the memory layout of the input.


Advantages and disadvantages
----------------------------

Using ctypes is a powerful way to connect Python with arbitrary C code. Its
advantages include:

* clean separation of C code from Python code
* no new syntax to learn besides Python and C
* reuse of existing C code, including shared libraries written for other
  purposes, with a Python wrapper
* integration with NumPy through the ``ctypes`` attribute of arrays
* argument checking with the ``ndpointer`` class factory

Its disadvantages include:

- You need a shared library; ctypes cannot use static libraries.

- You have to find the shared library at runtime, and write the code that
  checks and converts the arguments yourself.

- There is very little support for C++ code and its calling conventions. You
  will probably need a C wrapper around C++ code to use it with ctypes, or you
  can use a C++ binding tool such as pybind11 or nanobind instead.

ctypes is most useful for calling existing shared libraries, and for
experimenting without a Python-specific build step. For new compiled code in a
Python package, F2PY and Cython are generally more convenient, because they
generate the code that checks and converts the arguments for you.


Other tools
===========

.. index::
   single: swig
   single: SIP
   single: Boost.Python

The following tools are also widely used to call compiled code from Python.
Most of them are not specific to NumPy; see their documentation for how they
handle NumPy arrays. Tools that are not NumPy-specific can usually access the
data of an array through the buffer protocol or DLPack (see
:ref:`basics.interoperability`).

`pybind11 <https://pybind11.readthedocs.io>`_ and `nanobind <https://nanobind.readthedocs.io>`_
    Libraries for writing extension modules in C++. pybind11 represents NumPy
    arrays with its ``py::array_t`` class template, and nanobind supports NumPy
    arrays, and arrays from other libraries, with its ``nb::ndarray`` class
    template.

`Boost.Python <https://www.boost.org/doc/libs/release/libs/python/doc/html/index.html>`_
    A C++ library, part of the Boost collection, for exposing C++ classes and
    functions to Python. Its `NumPy extension
    <https://www.boost.org/doc/libs/release/libs/python/doc/html/numpy/index.html>`_,
    ``boost::python::numpy``, supports NumPy arrays.

`SWIG <https://www.swig.org>`_
    The Simplified Wrapper and Interface Generator generates wrappers for C and
    C++ libraries, for Python and many other languages, from an interface file
    that can include the header files of the library. NumPy provides SWIG
    typemaps for arrays in the file ``numpy.i``, in the ``tools/swig`` directory
    of the NumPy source repository; see :doc:`../reference/swig`.

`SIP <https://python-sip.readthedocs.io>`_
    A tool for creating Python bindings for C and C++ libraries, developed by
    Riverbank Computing to create the Python bindings for Qt. Its interface
    files look like C++ header files with additional directives.

`CFFI <https://cffi.readthedocs.io>`_
    A package for calling C code from Python, either by loading a shared library
    at runtime, as ctypes does, or by compiling an extension module from C
    declarations.

`PyO3 <https://pyo3.rs>`_
    Rust bindings for Python. The `rust-numpy <https://github.com/PyO3/rust-numpy>`_
    crate adds support for NumPy arrays.
