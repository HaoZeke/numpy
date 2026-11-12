.. _f2py-windows:

=================
F2PY and Windows
=================

.. warning::

   F2PY support for Windows is not always at par with Linux support

Broadly speaking, there are two issues working with F2PY on Windows:

- the Fortran compiler, the C compiler and the Python interpreter must agree on
  the C runtime library and the binary interface, and,
- a built extension module must find the runtime libraries of its Fortran
  compiler when it is imported.

The focus of this section is to establish a guideline for developing and
extending Fortran modules for Python natively, via F2PY on Windows.

``f2py -c`` builds with ``meson``, which needs a C compiler and a Fortran
compiler that work together. The supported toolchains are:

- Mingw-w64 C/C++/Fortran compilers (``gcc``, ``gfortran``)
- Intel compilers (``ifx`` with MSVC or ``icx``)
- Clang-cl + LLVM Flang
- MSVC + LLVM Flang

MSVC cannot be combined with Mingw-w64 ``gfortran``. The MSVC linker cannot use
the MinGW Fortran runtime, and the build stops with ``LNK1181: cannot open input
file 'gfortran.lib'``.

Overview
========

From a user perspective, the most UNIX compatible Windows
development environment is through emulation, either via the Windows Subsystem
for Linux, or facilitated by Docker. In a similar vein, traditional
virtualization methods like VirtualBox are also reasonable methods to develop
UNIX tools on Windows.

Native Windows builds no longer need a commercial compiler: Intel oneAPI,
MSYS2, conda-forge and LLVM all provide free Fortran compilers. Briefly, then,
for an end user:

Intel oneAPI (free)
   The LLVM based Intel compilers (``ifx``, ``icx``) compile natively and link
   with MSVC. The classic ``ifort`` compiler was discontinued with the oneAPI
   2025 release. See :ref:`f2py-win-intel`.

MSYS2 (FOSS)
   In conjunction with the ``mingw-w64`` project, provides ``gcc`` and
   ``gfortran`` together with its own Python and NumPy packages. See
   :ref:`f2py-win-msys2`.

Conda (FOSS)
   The `conda-forge`_ channel provides LLVM Flang, its default Fortran compiler
   on Windows, and the Mingw-w64 ``gcc`` and ``gfortran`` compilers. See
   :ref:`f2py-win-conda`.

LLVM Flang (FOSS)
   `LLVM Flang`_ compiles natively and links with MSVC or ``clang-cl``. The LLVM
   installer for Windows includes ``flang.exe`` from LLVM 22 onwards. The older
   "classic" `legacy version of Flang`_ is unmaintained.

Windows Subsystem for Linux
   ``f2py`` works as on Linux, and builds Linux extension modules. See
   :ref:`f2py-win-wsl`.

Cygwin (FOSS)
   Can also be used for ``gfortran``. However, the POSIX API compatibility layer
   provided by Cygwin is meant to compile UNIX software on Windows. Modules built
   there need the Cygwin runtime and load only into Cygwin's own Python.

PGI Compilers (historical)
   Superseded by the Nvidia HPC SDK, which has no `native Windows support`_.
   See :ref:`f2py-win-pgi`.

LFortran (FOSS)
   An LLVM based compiler under active development. Not all of the Fortran that
   F2PY supports can be compiled yet.

.. _f2py-win-compilers:

Selecting the compilers
=======================

``f2py -c`` writes a ``meson.build`` file and runs the ``meson`` and ``ninja``
commands, so both must be on ``PATH``. Installing them with ``pip`` into an
activated virtual environment, as in the baseline below, puts them there.

``meson`` reads the ``CC`` and ``FC`` environment variables to choose the C and
Fortran compilers. Without them it searches ``PATH``, preferring the Intel
Fortran compilers over ``gfortran`` and LLVM Flang on Windows, and MSVC is only
found automatically when no other C compiler is on ``PATH``. Set both variables
when more than one toolchain is installed, for example in PowerShell:

.. code-block:: powershell

   $env:CC = 'gcc'
   $env:FC = 'gfortran'
   python -m numpy.f2py -c fib1.f -m fib1

.. _f2py-win-dlls:

Runtime libraries at import time
================================

A module built by ``f2py`` depends on the runtime libraries of its Fortran
compiler, for example ``libgfortran-5.dll`` for Mingw-w64 ``gfortran`` or
``libifcoremd.dll`` for ``ifx``. Since Python 3.8, `Windows resolves these
dependencies`_ only from the system directories, the directory of the
interpreter, the directory containing the extension module, and directories
registered with :func:`os.add_dll_directory`. ``PATH`` is not searched. A module
whose runtime libraries are only reachable through ``PATH`` builds without
errors and then fails to import:

.. code-block:: text

   ImportError: DLL load failed while importing fib1: The specified module could not be found.

Any one of the following fixes this:

- register the directory that contains the runtime libraries before the import,
  for example for MSYS2:

  .. code-block:: python

     import os
     os.add_dll_directory(r"C:\msys64\ucrt64\bin")
     import fib1

- copy the runtime libraries next to the built module, or,
- link the runtime libraries statically, where the toolchain supports it (see
  :ref:`f2py-win-msys2`).

MSYS2 installs its Python next to the MinGW runtime libraries, and the Python
from conda-forge registers the environment's ``Library\bin`` directory, so
modules built and imported inside those environments need none of this.

.. _f2py-win-baseline:

Baseline
========

For this document we will assume the following basic tools:

- The IDE being considered is the community supported `Microsoft Visual Studio Code`_
- The terminal being used is the `Windows Terminal`_
- The shell environment is assumed to be `PowerShell 7.x`_
- A 64-bit CPython 3.12 or later from python.org. For Python 3.14 and later the
  recommended installer is the `Python install manager`_, which provides the
  ``py`` and ``python`` commands; ``py install`` adds further runtimes.
- The Microsoft Visual C++ (MSVC) toolset, for the toolchains that need it

Work inside a virtual environment, so that ``meson`` and ``ninja`` end up on
``PATH``:

.. code-block:: powershell

   python -m venv .venv
   # Once per user, if PowerShell refuses to run the activation script:
   #   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .venv\Scripts\Activate.ps1
   python -m pip install numpy meson ninja

Other Python distributions work too: conda (:ref:`f2py-win-conda`), the Python
packaged by MSYS2 (:ref:`f2py-win-msys2`), and the `Intel Distribution for
Python`_, which installs through conda or ``pip``. The Microsoft Store packages
need extra care, see :ref:`f2py-win-store`.

With this baseline configuration, we will further consider a configuration
matrix as follows:

.. _table-f2py-winsup-mat:

.. table:: Support matrix

  +----------------------+--------------------+------------------------------+
  | **Fortran Compiler** | **C/C++ Compiler** | **Source**                   |
  +======================+====================+==============================+
  | Intel ifx            | MSVC / icx         | Intel oneAPI installer       |
  +----------------------+--------------------+------------------------------+
  | Mingw-w64 gfortran   | Mingw-w64 gcc      | MSYS2 (UCRT64), conda-forge  |
  +----------------------+--------------------+------------------------------+
  | LLVM Flang           | MSVC / clang-cl    | conda-forge, LLVM installer  |
  +----------------------+--------------------+------------------------------+
  | GFortran             | GCC                | WSL (Linux modules)          |
  +----------------------+--------------------+------------------------------+

For an understanding of the key issues motivating the need for such a matrix
`Pauli Virtanen's in-depth post on wheels with Fortran for Windows`_ is an
excellent resource. An entertaining explanation of an application binary
interface (ABI) can be found in this post by `JeanHeyd Meneide`_.

PowerShell and MSVC
====================

MSVC is installed either via Visual Studio or the lighter (preferred)
`Build Tools for Visual Studio`_ with the ``Desktop development with C++``
workload.

.. note::

  This can take a significant amount of time as it includes a download of around
  2GB and requires a restart.

The installer does not put MSVC on ``PATH``. When ``meson`` finds no other C
compiler on ``PATH``, it activates the newest Visual Studio installation for
64-bit builds by itself (see also :ref:`building-from-source`). To load the
environment explicitly, for example to pair MSVC with LLVM Flang while other
compilers are installed, start a 64-bit Developer PowerShell. ``vswhere.exe``
finds the installation:

.. code-block:: powershell

   $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
   & "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
   Get-Command cl

The ``Developer PowerShell`` and ``Developer Command Prompt`` shortcuts target
32-bit x86 unless told otherwise, which does not match a 64-bit Python. Pass
``-Arch amd64 -HostArch amd64`` as above (Visual Studio 2022 version 17.1 or
later), or use the ``x64 Native Tools Command Prompt`` from ``cmd.exe``.

.. _f2py-win-store:

Microsoft Store Python
======================

The Microsoft Store has offered each Python version as a separate app, which
the CPython documentation describes as intended mainly for interactive use.
For Python 3.14 and later the Store provides the `Python install manager`_
instead. The per-version Store apps documented for Python 3.13 redirect
writes to shared locations such as ``TEMP`` and ``%APPDATA%`` into a private
copy (`Store package known issues`_). ``f2py -c`` builds in a temporary
directory, so pass ``--build-dir`` to keep the build files where other
programs can find them.

Scripts installed by ``pip``, including ``meson.exe`` and ``ninja.exe``, land
in a per-user ``Scripts`` directory that is not on ``PATH``, so ``f2py -c``
cannot start ``meson``. Either use a virtual environment as in the baseline,
or add that directory to ``PATH``:

  .. code-block:: powershell

     $env:Path += ";" + (python -c "import sysconfig; print(sysconfig.get_path('scripts', 'nt_user'))")

``Get-Command python`` shows which interpreter runs. A result under
``AppData\Local\Microsoft\WindowsApps`` is an app execution alias, and the
Windows settings page ``Manage app execution aliases`` controls which
installation it starts.

.. _f2py-win-wsl:

Windows Subsystem for Linux
===========================

Inside a WSL distribution ``f2py`` is used as on Linux: install ``gfortran`` and
the Python development headers with the distribution's package manager, then
``numpy``, ``meson`` and ``ninja``. The resulting extension module is a Linux
shared library. It can be imported by a Python running inside WSL, but not by a
Windows Python. For speed, keep the sources on the Linux file system, for
example under ``~``, rather than under ``/mnt/c``, as `Microsoft recommends`_.

.. toctree::
   :maxdepth: 2

   intel
   msys2
   conda
   pgi


.. _Microsoft Visual Studio Code: https://code.visualstudio.com/Download
.. _Build Tools for Visual Studio: https://visualstudio.microsoft.com/downloads/
.. _Windows Terminal: https://www.microsoft.com/en-us/p/windows-terminal/9n0dx20hk701?activetab=pivot:overviewtab
.. _PowerShell 7.x: https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows
.. _Python install manager: https://docs.python.org/3/using/windows.html#python-install-manager
.. _Store package known issues: https://docs.python.org/3.13/using/windows.html#known-issues
.. _Windows resolves these dependencies: https://docs.python.org/3/whatsnew/3.8.html#bpo-36085-whatsnew
.. _Intel Distribution for Python: https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html
.. _Microsoft recommends: https://learn.microsoft.com/en-us/windows/wsl/filesystems#file-storage-and-performance-across-file-systems
.. _Pauli Virtanen's in-depth post on wheels with Fortran for Windows: https://pav.iki.fi/blog/2017-10-08/pywingfortran.html#building-python-wheels-with-fortran-for-windows
.. _JeanHeyd Meneide: https://thephd.dev/binary-banshees-digital-demons-abi-c-c++-help-me-god-please
.. _legacy version of Flang: https://github.com/flang-compiler/flang
.. _native Windows support: https://developer.nvidia.com/nvidia-hpc-sdk-downloads#collapseFour
.. _conda-forge: https://conda-forge.org/docs/maintainer/infrastructure/#compilers-supplied-by-conda-forge
.. _LLVM Flang: https://flang.llvm.org/docs/
