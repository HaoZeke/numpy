``f2py`` routines can share an ``intent(callback)`` external
------------------------------------------------------------
Several routines of one module can now declare the same
``intent(callback)`` external, in a signature file or with ``!f2py``
directives. ``f2py`` used to emit one C definition of the external per
routine, so the module failed to compile. The routines now share a
single definition, and ``f2py`` raises an error when their declarations
of the external have different signatures.
