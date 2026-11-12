``f2py`` keeps the dtype reference when array creation fails
------------------------------------------------------------
When an ``f2py`` wrapped routine failed to convert an array argument, for example a ragged list, or to create an ``intent(hide)`` array, the dtype of the argument lost a reference.
Repeated failures led to NumPy's "Reference count error detected" for builtin dtypes, and to a use after free for ``character`` arguments.
