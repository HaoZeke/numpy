``f2py`` no longer aborts when a callback raises under Python TLS
-----------------------------------------------------------------
An exception raised in a callback no longer aborts Python when the
module is built with ``-DF2PY_USE_PYTHON_TLS``, or when ``f2py`` finds
no thread-local storage support in the C compiler.
