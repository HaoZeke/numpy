``f2py`` exits with status 1 on rejected CLI input
--------------------------------------------------
``numpy.f2py`` now exits with status 1 when given an unknown option.
``numpy.f2py`` now exits with status 1 if the signature file already exists.
That exit applies when ``--overwrite-signature`` was not given.
