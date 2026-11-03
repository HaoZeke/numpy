``f2py -c`` links prebuilt libraries after the wrapper objects
--------------------------------------------------------------
Static (``.a``) and shared (``.so``, ``.dylib``) libraries passed to
``f2py -c`` on the command line are now linked after the generated wrapper
objects, from where they are. Previously the meson backend copied them into
its temporary build directory and placed them before the wrapper objects, so
their symbols were missing when the module was imported. Object files given
with a directory are now found as well.
