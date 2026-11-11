* ``f2py`` leaves a Fortran ``OPTIONAL`` dummy absent when the Python caller
  omits it or passes ``None``, so ``present()`` is false.
  The new ``absent`` attribute selects this call.
  A signature-file ``optional`` attribute keeps the previous default value
  unless the file also names ``absent``.
  An unsupported kind from Fortran source drops ``absent`` and keeps the
  previous call, after a build message.
  A signature file that writes ``absent`` on such a kind is an error.
  An explicit init expression also drops ``absent`` and keeps that default.
