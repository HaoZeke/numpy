.. _f2py-win-conda:

=========================
F2PY and Conda on Windows
=========================

This guide assumes ``scoop`` is available for installing tools without
administrative access.

.. code-block:: powershell

  Invoke-Expression (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')

Set up a ``conda`` environment:

.. code-block:: powershell

	scoop install miniconda3
	# For conda activate / deactivate in powershell
	conda install -n root -c pscondaenvs pscondaenvs
	Powershell -c Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
	conda init powershell
	# Open a new shell for the rest

``conda`` pulls packages from ``msys2``, but the workflow differs enough to warrant separate discussion.

.. warning::

	As of 30-01-2022, the `MSYS2 binaries`_ shipped with ``conda`` are **outdated** and this approach is **not preferred**.



.. _MSYS2 binaries: https://github.com/conda-forge/conda-forge.github.io/issues/1044
