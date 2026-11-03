``f2py`` no longer handles ``integer(8)`` values as C ``long``
--------------------------------------------------------------
On Windows, where C ``long`` is 32 bits, ``check`` failure messages for
``integer(8)`` arguments showed only the low 32 bits of the value, and an
``intent(inout)`` ``integer(8)`` scalar passed as a 0-d `numpy.uint` array was
written back as 8 bytes into its 4-byte element.
