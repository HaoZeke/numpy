``f2py`` wrappers keep dtype reference counts intact on conversion errors
-----------------------------------------------------------------------
An ``f2py``-generated wrapper that failed to convert an array argument, for
example because a string was passed for a ``real`` array, released a reference
to a built-in NumPy dtype that it did not own. After a few such calls NumPy
reported a reference count error, which aborts builds with assertions enabled.
