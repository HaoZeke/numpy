``f2py`` symbolic expressions accept integer kinds of character literals
------------------------------------------------------------------------
Converting ``numpy.f2py.symbolic.as_string(obj, kind)`` to a string no
longer raises ``TypeError`` when ``kind`` is an integer other than 1.
A character literal such as ``4_"abc"`` now parses with the integer kind
``4``, as integer and real literals already did, so parsing and printing
round-trip.
