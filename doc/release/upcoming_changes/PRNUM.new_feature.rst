``f2py --dep`` accepts keyword arguments for meson's ``dependency()``
---------------------------------------------------------------------
``--dep`` accepts meson ``dependency()`` keywords in brackets.
An example is ``--dep "mpi[language: 'fortran']"``.
Plain ``--dep mpi`` resolves the C MPI library.
Fortran code calling MPI then fails to import.
One undefined symbol is ``mpi_comm_rank_``.
