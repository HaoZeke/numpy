``f2py`` derived type support
-----------------------------

``f2py`` can now wrap Fortran derived types as Python classes.
It wraps module routines that take or return derived types.
Those routines used to stop ``f2py`` with ``KeyError: 'void'``.
An instance reaches Fortran as a handle of its Fortran object.

* ``bind(c)`` types map to C structs, with direct field access.
  Array members are NumPy views of the Fortran storage.
* **Opaque types** need no ``bind(c)``.
  ``f2py`` writes Fortran ``bind(c)`` accessors for them.

Supported features:

* Scalar and array arguments of any rank, including assumed-shape arrays.
* ``intent(out)`` arguments and the value a function gives back are new instances.
* Allocatable and pointer arguments.
  ``None`` means the dummy is unallocated or disassociated.
* ``class(T)`` arguments, which keep the dynamic type of the instance.
* ``class(*)`` arguments, which take Python scalars and instances.
* Fixed, allocatable, and pointer array members, of rank 1 and above.
* Nested types and arrays of derived types.
* Inheritance with ``extends``, including abstract base types.
* ``isinstance()`` against those base types.
* Type-bound procedures and generic bindings, with runtime dispatch.
* Finalizers.
* ``f2py`` exposes procedure pointer components, and Python can call them.
* Operator overloading for arithmetic, comparison, and assignment.
* ``write(formatted)`` becomes ``str()`` through the DT edit descriptor.
  ``read(formatted)`` becomes ``from_string()`` on the class.
* Character members, fixed-length and deferred-length allocatable.
* Complex members, using ``npy_cfloat`` and ``npy_cdouble``.
* Default initialization and ``PRIVATE`` components.
* ``SEQUENCE`` types.
* Kinds and lengths from module parameters and ``iso_fortran_env``.
* ``kind``-parameterized types, with one specialization per kind value.
* LEN-parameterized types build on gfortran, and LLVM flang cannot yet compile an object of such a type.
* Array members accept array API input, including objects that provide dlpack.

A member ``f2py`` cannot expose is left out of the Python class.
The type, and routines that take it, are still wrapped.
One example is the pointer to the next node of a linked list.
Types with a coarray component are not wrapped.
Fortran 2018 C825 says such an object cannot be a pointer.
