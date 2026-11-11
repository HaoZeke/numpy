f2py routine objects pickle by reference
-----------------------------------------
``f2py`` routine objects now pickle by import path. ``pickle.loads``
returns the same object, including an F90 module routine such as
``m.t``. ``copy.copy`` and ``copy.deepcopy`` return that same object.
F90 module objects, module variable containers, and ``COMMON`` blocks
still raise ``TypeError``: they wrap Fortran memory. Copy the array
data out of those objects when a value must be pickled.
