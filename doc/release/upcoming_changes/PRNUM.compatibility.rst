``f2py`` rejects the options of the removed ``distutils`` backend
-----------------------------------------------------------------
``f2py`` now exits with an error that names the replacement when it is passed
``--backend distutils`` or one of the build options that only the removed
``distutils`` backend understood: ``--fcompiler``, ``--compiler``,
``--f77exec``, ``--f90exec``, ``--help-fcompiler``, ``--opt``, ``--arch``,
``--noopt``, ``--noarch``, ``--link-<resource>`` and ``--help-link``.
``f2py -c`` used to ignore them, so that for example ``--f90exec`` did not
change the Fortran compiler used for the build. Select the compilers with the
``FC`` and ``CC`` environment variables, pass compiler flags with
``--f77flags`` or ``--f90flags``, and link dependencies with ``--dep``.
