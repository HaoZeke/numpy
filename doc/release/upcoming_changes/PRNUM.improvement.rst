f2py routine objects pickle by reference
-----------------------------------------
``f2py`` routine objects now pickle by import path. ``pickle.loads``
returns the same object, including an F90 module routine such as
``m.t``. ``copy.copy`` and ``copy.deepcopy`` return that same object.
F90 module objects, module variable containers, and ``COMMON`` blocks
still raise ``TypeError``: they wrap Fortran memory. Copy the array
data out of those objects when a value must be pickled.

The extension must be regenerated and rebuilt with this F2PY support.
Installing a NumPy update does not replace the wrapper code embedded
in an existing extension binary. The defining extension must be
importable in the process which restores the routine.
