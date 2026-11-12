f2py keeps spaces in argv tokens
--------------------------------
f2py no longer splits one argv element that holds a flag and its value.
Pass ``--include-paths`` and the path as separate arguments.
