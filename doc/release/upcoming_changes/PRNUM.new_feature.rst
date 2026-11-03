``f2py --dep`` accepts keyword arguments for meson's ``dependency()``
---------------------------------------------------------------------
Keyword arguments for meson's ``dependency()`` can now be appended in
brackets to a ``--dep`` value, e.g. ``--dep "mpi[language: 'fortran']"``.
Fortran code calling MPI needs this, since a plain ``--dep mpi`` resolves
to the C MPI library without the Fortran bindings, and the module then
fails to import with an undefined symbol such as ``mpi_comm_rank_``.
