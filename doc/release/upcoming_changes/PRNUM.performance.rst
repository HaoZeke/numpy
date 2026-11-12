Faster ``f2py -c`` builds with the meson backend
------------------------------------------------
``f2py -c`` now spends less time configuring its meson build. The generated
``meson.build`` no longer starts two Python interpreters to look up the NumPy
and ``f2py`` include directories, and the C and Fortran compilers are only
detected and sanity checked for the host machine instead of for both the host
and the build machine. For a small extension module this cuts the time of
``meson setup`` by about a third and of the whole build by roughly 20%.
