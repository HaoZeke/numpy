.. _f2py-boilerplating:

====================================
Boilerplate reduction and templating
====================================

Using FYPP for binding generic interfaces
=========================================

``f2py`` does not support binding interface blocks directly. A common
workaround was to use ``tempita`` with ``.pyf.src`` files, as done in the
bindings `that are part of scipy`_. ``tempita`` support has been removed and
is no longer recommended.

.. note::
    The reason interfaces cannot be supported within ``f2py`` itself is because
    they don't correspond to exported symbols in compiled libraries.

    .. code:: sh

       ❯ nm gen.o
        0000000000000078 T __add_mod_MOD_add_complex
        0000000000000000 T __add_mod_MOD_add_complex_dp
        0000000000000150 T __add_mod_MOD_add_integer
        0000000000000124 T __add_mod_MOD_add_real
        00000000000000ee T __add_mod_MOD_add_real_dp

This section covers techniques that combine ``f2py`` with `fypp`_ to emulate
generic interfaces and reduce repetition when binding multiple similar
functions.


Basic example: Addition module
------------------------------

Let us build on the example (from the user guide, :ref:`f2py-examples`) of a
subroutine which takes in two arrays and returns its sum.

.. literalinclude:: ./../code/add.f
    :language: fortran


Recast this into modern Fortran:

.. literalinclude:: ./../code/advanced/boilerplating/src/adder_base.f90
    :language: fortran

Rather than adding intents by hand for each variant, we can template the
construction of similar functions with FYPP:

.. literalinclude:: ./../code/advanced/boilerplating/src/gen_adder.f90.fypp

Pre-process to generate the full Fortran code:

.. code:: sh

       ❯ fypp gen_adder.f90.fypp > adder.f90

This output can then be wrapped by ``f2py``.

Now consider maintaining the bindings in a separate file. The following
``.pyf`` can be generated for a single subroutine via
``f2py -m adder adder_base.f90 -h adder.pyf``:

.. literalinclude:: ./../code/advanced/boilerplating/src/base_adder.pyf
    :language: fortran

With the docstring:

.. literalinclude:: ./../code/advanced/boilerplating/res/base_docstring.dat
    :language: reST

This is already reasonable. However, ``n`` should not be passed by the
caller, so we make some adjustments:

.. literalinclude:: ./../code/advanced/boilerplating/src/improved_base_adder.pyf
    :language: fortran

This produces the docstring:

.. literalinclude:: ./../code/advanced/boilerplating/res/improved_docstring.dat
    :language: reST

Finally, template over this in the same manner to produce bindings that use
``f2py`` directives with minimal repetition:

.. literalinclude:: ./../code/advanced/boilerplating/src/adder.pyf.fypp

The full build sequence:

.. code:: sh

   fypp gen_adder.f90.fypp > adder.f90
   fypp adder.pyf.fypp > adder.pyf
   f2py -m adder -c adder.pyf adder.f90 --backend meson

.. _`fypp`: https://fypp.readthedocs.io/en/stable/fypp.html
.. _`that are part of scipy`: https://github.com/scipy/scipy/blob/c93da6f46dbed8b3cc0ccd2495b5678f7b740a03/scipy/linalg/clapack.pyf.src
