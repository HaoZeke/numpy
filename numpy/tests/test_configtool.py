import importlib.metadata
import os
import pathlib
import shutil

import pytest

import numpy as np
import numpy._core.include
import numpy._core.lib.pkgconfig
from numpy.testing import HAS_SUBPROCESSES, IS_EDITABLE, IS_INSTALLED, NUMPY_ROOT
from numpy.testing._private.utils import run_subprocess

INCLUDE_DIR = NUMPY_ROOT / '_core' / 'include'
PKG_CONFIG_DIR = NUMPY_ROOT / '_core' / 'lib' / 'pkgconfig'
F2PY_INCLUDE_DIR = pathlib.Path(np.f2py.get_include())


@pytest.mark.skipif(not IS_INSTALLED,
                    reason="`numpy-config` not expected to be installed")
@pytest.mark.skipif(not HAS_SUBPROCESSES,
                    reason="platform cannot start subprocesses")
class TestNumpyConfig:
    def check_numpyconfig(self, arg):
        res = run_subprocess(['numpy-config', arg])
        return res.stdout.strip()

    def test_configtool_version(self):
        stdout = self.check_numpyconfig('--version')
        assert stdout == np.__version__

    def test_configtool_cflags(self):
        stdout = self.check_numpyconfig('--cflags')
        assert f'-I{os.fspath(INCLUDE_DIR)}' in stdout

    def test_configtool_pkgconfigdir(self):
        stdout = self.check_numpyconfig('--pkgconfigdir')
        assert pathlib.Path(stdout) == PKG_CONFIG_DIR.resolve()

    def test_configtool_f2py_includedir(self):
        stdout = self.check_numpyconfig('--f2py-includedir')
        assert pathlib.Path(stdout).resolve() == F2PY_INCLUDE_DIR.resolve()


@pytest.mark.skipif(not IS_INSTALLED,
                    reason="numpy must be installed to check its entrypoints")
def test_pkg_config_entrypoint():
    (entrypoint,) = importlib.metadata.entry_points(group='pkg_config', name='numpy')
    assert entrypoint.value == numpy._core.lib.pkgconfig.__name__


@pytest.mark.skipif(not IS_INSTALLED,
                    reason="numpy.pc is only available when numpy is installed")
@pytest.mark.skipif(IS_EDITABLE, reason="editable installs don't have a numpy.pc")
def test_pkg_config_config_exists():
    assert PKG_CONFIG_DIR.joinpath('numpy.pc').is_file()


@pytest.mark.skipif(not IS_INSTALLED,
                    reason="numpy.pc is only available when numpy is installed")
@pytest.mark.skipif(IS_EDITABLE, reason="editable installs don't have a numpy.pc")
@pytest.mark.skipif(not HAS_SUBPROCESSES,
                    reason="platform cannot start subprocesses")
@pytest.mark.skipif(shutil.which('pkg-config') is None,
                    reason="pkg-config is not available")
def test_pkg_config_f2py_includedir():
    env = {**os.environ, 'PKG_CONFIG_PATH': os.fspath(PKG_CONFIG_DIR)}
    res = run_subprocess(
        ['pkg-config', '--variable=f2py_includedir', 'numpy'], env=env)
    f2py_includedir = pathlib.Path(res.stdout.strip())
    assert f2py_includedir.resolve() == F2PY_INCLUDE_DIR.resolve()
