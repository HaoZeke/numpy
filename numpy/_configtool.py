import argparse
import sys
from pathlib import Path

from .lib._utils_impl import get_include
from .version import __version__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print the version and exit.",
    )
    parser.add_argument(
        "--cflags",
        action="store_true",
        help="Compile flag needed when using the NumPy headers.",
    )
    parser.add_argument(
        "--pkgconfigdir",
        action="store_true",
        help=("Print the pkgconfig directory in which `numpy.pc` is stored "
              "(useful for setting $PKG_CONFIG_PATH)."),
    )
    parser.add_argument(
        "--f2pycflags",
        action="store_true",
        help=("Compile flags needed when building f2py-generated sources "
              "(the numpy and numpy.f2py include directories)."),
    )
    parser.add_argument(
        "--fortranobject",
        action="store_true",
        help=("Print the path of fortranobject.c, which must be compiled "
              "into every f2py-generated extension."),
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.cflags:
        print("-I" + get_include())
    if args.pkgconfigdir:
        _path = Path(get_include()) / '..' / 'lib' / 'pkgconfig'
        print(_path.resolve())
    if args.f2pycflags:
        from .f2py import get_include as f2py_get_include
        print("-I" + get_include() + " -I" + f2py_get_include())
    if args.fortranobject:
        from .f2py import get_include as f2py_get_include
        print(Path(f2py_get_include()) / 'fortranobject.c')


if __name__ == "__main__":
    main()
