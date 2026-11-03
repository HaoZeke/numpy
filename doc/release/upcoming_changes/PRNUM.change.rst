``f2py`` keeps C operator precedence in dimension specifications
-----------------------------------------------------------------
C logical operators (``&&``, ``||`` and ``!``) and casts such as
``(int)v`` in the ``dimension`` specifications of signature files are now
parsed with their C precedence and no longer emit ``ExprWarning``.
Previously they could be reordered with arithmetic and comparisons, so
that, for example, ``*range=='I' && iu-il+1==n`` was compiled as
``*range == 1 + 'I' && iu - il == n`` and produced a wrongly sized array,
and ``(int)(x+y)`` was compiled as the invalid C ``int(x + y)``.
