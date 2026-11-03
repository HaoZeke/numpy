``f2py`` exposes module data accessed with ``use``
--------------------------------------------------
A Fortran module now exposes the variables and parameters it accesses with
``use`` from another module wrapped in the same extension, so ``test2.a``
exists when ``test2`` contains ``use test``. ``only`` lists, renames and
``private`` statements are respected, and the order of the source files does
not matter. Signature files written with ``-h`` list these entities under the
using module as well.
