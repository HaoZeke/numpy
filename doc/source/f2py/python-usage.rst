==================================
Using F2PY bindings in Python
==================================

This page describes common usage patterns for F2PY-generated wrappers in
Python, organized by argument type. For additional examples, see
:ref:`f2py-examples`.

Fortran type objects
====================

F2PY exposes all wrappers for Fortran/C routines, common blocks, and Fortran 90
module data as ``fortran`` type objects in Python. Routine wrappers are callable;
data wrappers have attributes referring to the data members.

Every ``fortran`` type object has a ``_cpointer`` attribute containing a
:c:type:`PyCapsule` that holds the C pointer to the underlying Fortran/C function
or variable. These capsules can be passed as callback arguments to other F2PY
generated functions, bypassing the Python C/API layer. This is useful when the
callback's implementation is itself written in C or Fortran and wrapped with
F2PY (or any tool that produces ``PyCapsule`` objects).

Consider a Fortran 77 file ``ftype.f``:

.. literalinclude:: ./code/ftype.f
  :language: fortran

and a wrapper built with ``f2py -c ftype.f -m ftype``.

In Python, the types of ``foo`` and ``data`` and access to individual wrapped
objects look like this:

.. literalinclude:: ./code/results/ftype_session.dat
  :language: python


Scalar arguments
=================

A scalar argument to an F2PY generated wrapper function can be an ordinary
Python scalar (integer, float, complex number) or any sequence object (list,
tuple, array, string) of scalars. For sequences, the first element is passed
to the Fortran routine as the scalar argument.

.. note::

   * When type-casting is required and there is possible loss of information via
     narrowing e.g. when type-casting float to integer or complex to float, F2PY
     *does not* raise an exception.

     * For complex to real type-casting only the real part of a complex number
       is used.

   * ``intent(inout)`` scalar arguments are assumed to be array objects in
     order to have *in situ* changes be effective. It is recommended to use
     arrays with proper type but also other types work. :ref:`Read more about
     the intent attribute <f2py-attributes>`.

Consider the following Fortran 77 code:

.. literalinclude:: ./code/scalar.f
  :language: fortran

and wrap it using ``f2py -c -m scalar scalar.f``.

In Python:

.. literalinclude:: ./code/results/scalar_session.dat
  :language: python


String arguments
=================

F2PY generated wrapper functions accept almost any Python object as a string
argument because ``str`` is applied to non-string objects. NumPy arrays used as
string arguments must have type code ``'S1'`` or ``'b'`` (the legacy ``'c'``
or ``'1'`` typecodes). See :ref:`arrays.scalars` for details.

A string argument can have any length. Strings longer than expected are
truncated silently. Strings shorter than expected are padded with ``\0``.

.. TODO: review this section once https://github.com/numpy/numpy/pull/19388 is merged.

Because Python strings are immutable, an ``intent(inout)`` argument expects an
array version of a string in order to have *in situ* changes be effective.

Consider the following Fortran 77 code:

.. literalinclude:: ./code/string.f
  :language: fortran

and wrap it using ``f2py -c -m mystring string.f``.

Python session:

.. literalinclude:: ./code/results/string_session.dat
  :language: python


Array arguments
================

Array arguments for F2PY generated wrapper functions accept arbitrary sequences
convertible to NumPy arrays, with two exceptions:

* ``intent(inout)`` array arguments must always be
  :term:`proper-contiguous <contiguous>` and have a compatible ``dtype``,
  otherwise an exception is raised.
* ``intent(inplace)`` array arguments must be arrays. If these have
  incompatible order or size, a converted copy is passed in, which is
  copied back into the original array on exit (see the ``intent(inplace)``
  :ref:`attribute <f2py-attributes>` for more information).

If a NumPy array is :term:`proper-contiguous <contiguous>` and has a matching
type, it is passed directly to the wrapped Fortran/C function. Otherwise, F2PY
makes an element-wise copy with the correct contiguity and type.

F2PY handles storage order automatically and copies arrays only when
necessary. For very large multidimensional arrays close to available physical
memory, pass proper-contiguous arrays with matching types to avoid copies.

To transform input arrays to column major storage order before passing
them to Fortran routines, use the function `numpy.asfortranarray`.

Consider the following Fortran 77 code:

.. literalinclude:: ./code/array.f
  :language: fortran

and wrap it using ``f2py -c -m arr array.f -DF2PY_REPORT_ON_ARRAY_COPY=1``.

In Python:

.. literalinclude:: ./code/results/array_session.dat
  :language: python

.. _Call-back arguments:

Call-back arguments
====================

F2PY supports calling Python functions from Fortran or C codes.

Consider the following Fortran 77 code:

.. literalinclude:: ./code/callback.f
  :language: fortran

and wrap it using ``f2py -c -m callback callback.f``.

In Python:

.. literalinclude:: ./code/results/callback_session.dat
  :language: python

In the above example F2PY was able to guess accurately the signature
of the call-back function. However, sometimes F2PY cannot establish the
appropriate signature; in these cases the signature of the call-back
function must be explicitly defined in the signature file.

To facilitate this, signature files may contain special modules (the names of
these modules contain the special ``__user__`` sub-string) that define the
various signatures for call-back functions.  Callback arguments in routine
signatures have the ``external`` attribute (see also the ``intent(callback)``
:ref:`attribute <f2py-attributes>`). To relate a callback argument with its
signature in a ``__user__`` module block, a ``use`` statement can be utilized as
illustrated below. The same signature for a callback argument can be referred to
in different routine signatures.

We use the same Fortran 77 code as in the previous example but now
we will pretend that F2PY was not able to guess the signatures of
call-back arguments correctly. First, we create an initial signature
file ``callback2.pyf`` using F2PY::

    f2py -m callback2 -h callback2.pyf callback.f

Then modify it as follows

.. include:: ./code/callback2.pyf
  :literal:

Finally, we build the extension module using
``f2py -c callback2.pyf callback.f``.

An example Python session for this snippet would be identical to the previous
example except that the argument names would differ.

Sometimes a Fortran package may require that users provide routines that the
package will use. F2PY can construct an interface to such routines so that
Python functions can be called from Fortran.

Consider the following Fortran 77 subroutine that takes an array as its input
and applies a function ``func`` to its elements.

.. literalinclude:: ./code/calculate.f
  :language: fortran

The Fortran code expects that the function ``func`` has been defined externally.
In order to use a Python function for ``func``, it must have an attribute
``intent(callback)`` and it must be specified before the ``external`` statement.

Finally, build an extension module using ``f2py -c -m foo calculate.f``

In Python:

.. literalinclude:: ./code/results/calculate_session.dat
  :language: python

The function is included as an argument to the python function call to the
Fortran subroutine even though it was *not* in the Fortran subroutine argument
list. The "external" keyword refers to the C function generated by f2py, not the
Python function itself. The python function is essentially being supplied to the
C function.

The callback function may also be explicitly set in the module. Then it is not
necessary to pass the function in the argument list to the Fortran function.
This may be desired if the Fortran function calling the Python callback function
is itself called by another Fortran function.

Consider the following Fortran 77 subroutine:

.. literalinclude:: ./code/extcallback.f
  :language: fortran

and wrap it using ``f2py -c -m pfromf extcallback.f``.

In Python:

.. literalinclude:: ./code/results/extcallback_session.dat
  :language: python

.. note::

   When using modified Fortran code via ``callstatement`` or other directives,
   the wrapped Python function must be called as a callback, otherwise only the
   bare Fortran routine will be used. For more details, see
   https://github.com/numpy/numpy/issues/26681#issuecomment-2466460943

Resolving arguments to call-back functions
------------------------------------------

F2PY generated interfaces are flexible with respect to callback arguments. For
each callback argument, F2PY introduces an additional optional argument
``<name>_extra_args`` that passes extra arguments to user-provided callbacks.

If a F2PY generated wrapper function expects the following call-back argument::

  def fun(a_1,...,a_n):
     ...
     return x_1,...,x_k

but the following Python function

::

  def gun(b_1,...,b_m):
     ...
     return y_1,...,y_l

is provided by a user, and in addition,

::

  fun_extra_args = (e_1,...,e_p)

is used, then the following rules are applied when a Fortran or C function
evaluates the call-back argument ``gun``:

* If ``p == 0`` then ``gun(a_1, ..., a_q)`` is called, here
  ``q = min(m, n)``.
* If ``n + p <= m`` then ``gun(a_1, ..., a_n, e_1, ..., e_p)`` is called.
* If ``p <= m < n + p`` then ``gun(a_1, ..., a_q, e_1, ..., e_p)`` is called,
  and here ``q=m-p``.
* If ``p > m`` then ``gun(e_1, ..., e_m)`` is called.
* If ``n + p`` is less than the number of required arguments to ``gun`` then an
  exception is raised.

If the function ``gun`` may return any number of objects as a tuple; then the
following rules are applied:

* If ``k < l``, then ``y_{k + 1}, ..., y_l`` are ignored.
* If ``k > l``, then only ``x_1, ..., x_l`` are set.


Common blocks
==============

F2PY generates wrappers for ``common`` blocks defined in a routine signature
block. Common blocks are visible to all Fortran code linked to the current
extension module, but not to other extension modules (a consequence of how
Python imports shared libraries). In Python, these wrappers are ``fortran``
type objects with dynamic attributes for each data member. Accessing an
attribute returns a NumPy array (Fortran-contiguous for multidimensional data)
that links directly to the common block's memory. Data members can be changed
by assignment or by in-place modification of the array.

Consider the following Fortran 77 code:

.. literalinclude:: ./code/common.f
  :language: fortran

and wrap it using ``f2py -c -m common common.f``.

In Python:

.. literalinclude:: ./code/results/common_session.dat
  :language: python


Fortran 90 module data
=======================

F2PY handles Fortran 90 module data the same way as Fortran 77 common blocks.

Consider the following Fortran 90 code:

.. literalinclude:: ./code/moddata.f90
  :language: fortran

and wrap it using ``f2py -c -m moddata moddata.f90``.

In Python:

.. literalinclude:: ./code/results/moddata_session.dat
  :language: python


Allocatable arrays
===================

F2PY has basic support for Fortran 90 module allocatable arrays.

Consider the following Fortran 90 code:

.. literalinclude:: ./code/allocarr.f90
  :language: fortran

and wrap it using ``f2py -c -m allocarr allocarr.f90``.

In Python:

.. literalinclude:: ./code/results/allocarr_session.dat
  :language: python
