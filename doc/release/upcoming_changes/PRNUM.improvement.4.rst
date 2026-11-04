F2PY extensions use Python's Limited API
----------------------------------------
F2PY compiles generated wrappers and its runtime against the Python 3.12
Limited API, matching NumPy's minimum Python version. The Meson backend
selects the corresponding stable ABI and rejects disabling the Limited API.
The NumPy C ABI remains a separate compatibility requirement.

The Fortran wrapper type is an immutable, final heap type associated with its
owning module. Instance storage uses ``PyObject_GetTypeData`` and the garbage
collector visits both the instance dictionary and its type. Code embedding
the runtime passes the owning module to ``PyFortranObject_New`` and
``PyFortranObject_NewAsAttr`` and allocates ``F2PyRuntimeState`` as module
state, with its traverse, clear and free hooks. Direct access to the static
``PyFortran_Type`` and the ``PyFortranObject`` object layout is replaced by
the runtime's constructor and check functions.

Free-threaded extensions target Python 3.15's stable ABI for free-threaded
builds. Python 3.13 and 3.14 have no free-threaded stable ABI and cannot build
these extensions. User-supplied C sources must also use the selected Limited
API; arbitrary CPython internals in those sources are rejected at compilation.
Diagnostics that print a Python object are written to ``sys.stderr``.
