f2py raises errors for NumPy scalars and unallocated arrays
-----------------------------------------------------------
A character argument used to crash on a NumPy scalar.
It now raises ``TypeError``.
An unallocated array used to crash a pointer read.
It now raises ``ValueError`` and names that array.
